from __future__ import annotations

import builtins
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from paperlens_core.budget import BudgetManager
from paperlens_core.config import BudgetConfig, ProviderConfig, CoreConfig
from paperlens_core.control import ControlState
from paperlens_core.db import ArtifactDb
from paperlens_core.engine import PaperLensEngine
from paperlens_core.events import EventWriter
from paperlens_core.agents.llm import (
    JsonLlmClient,
    json_retry_completion_minimum,
    parse_json_text_for_schema,
)
from paperlens_core.library import (
    LIBRARY_RECORD_FILENAME,
    LIBRARY_RECORD_SCHEMA_VERSION,
    build_library_ask_prompt,
    doctor_library,
    expand_search_query_terms,
    normalize_library_answer,
    read_library_records,
    search_library,
    validate_memory_record,
    write_paperlens_library,
)
from paperlens_core.main import configure_utf8_stdio
from paperlens_core.memory_v3 import (
    MEMORY_V3_SCHEMA_VERSION,
    read_paper_memory_v3,
    validate_paper_memory_v3,
    write_paper_memory_v3_file,
)
from paperlens_core.memory_store import MEMORY_PATCH_SCHEMA_VERSION, PaperMemoryStore
from paperlens_core.protocol import RunRequest
from paperlens_core.workflow.agent import (
    REPORT_PLAN_SCHEMA,
    REPORT_SECTION_SCHEMA,
    PaperLensWorkflow,
    build_memory_critic_prompt,
    build_memory_repair_prompt,
    build_report_plan_prompt,
    build_report_section_audit_prompt,
    build_report_section_prompt,
    clean_model_markdown,
    compose_agentic_paper_report,
    downgrade_exhausted_targeted_reread,
    fallback_memory_audit,
    final_report_audit_acceptable,
    memory_repair_round_budget,
    memory_audit_acceptable,
    memory_audit_needs_targeted_reread,
    normalize_memory_audit,
    readable_model_body,
    render_freeform_paper_report,
    render_paperlens_report,
    sanitize_reader_hostile_text,
    select_rolling_read_pages,
    select_targeted_reread_pages,
    user_visible_review_items,
    validate_paperlens_output,
    visual_crop_bbox_for_page,
    write_final_report_bundle,
)
from paperlens_core.workflow.stages import resolve_workflow_stages
from paperlens_core.runtime import PaperLensRuntime
from paperlens_core.quality import evaluate_capsule_quality
from paperlens_core.qa import (
    ASK_SYSTEM_PROMPT,
    answer_question,
    build_ask_prompt,
    classify_question,
    normalize_answer,
)
from paperlens_core.schemas import (
    ClassificationDecision,
    EvidenceRef,
    PaperCard,
    PaperRecord,
    ReviewItem,
    SkimCard,
)


def test_workflow_stage_order_is_lean():
    from paperlens_core.workflow.stages import WORKFLOW_STAGE_ORDER

    assert WORKFLOW_STAGE_ORDER == [
        "stage_00_ingest",
        "stage_01_parse",
        "stage_02_parse_verify",
        "stage_03_skim",
        "stage_07_normal_read",
        "stage_08_evidence_verify",
        "stage_15_export",
        "stage_17_manifest",
    ]


def test_output_validation_rejects_fallback_reports(tmp_path):
    (tmp_path / "papers").mkdir()
    (tmp_path / "PaperLens.md").write_text(
        "# PaperLens\n\n- [A] [Bad](./papers/bad.md) - Model final-report generation failed\n",
        encoding="utf-8",
    )
    (tmp_path / "papers" / "bad.md").write_text(
        "# Bad\n\nreport_generation_failed: upstream error\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as exc:
        validate_paperlens_output(tmp_path)

    assert "Fallback report marker" in str(exc.value)


def test_output_validation_rejects_escaped_newline_markers(tmp_path):
    (tmp_path / "papers").mkdir()
    (tmp_path / "PaperLens.md").write_text(
        "# PaperLens\n\n- [A] [Bad](./papers/bad.md)\n", encoding="utf-8"
    )
    (tmp_path / "papers" / "bad.md").write_text("# Bad\n\nfirst\\n\\nsecond\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as exc:
        validate_paperlens_output(tmp_path)

    assert "Escaped newline marker" in str(exc.value)


def test_output_validation_rejects_full_page_visual_embeds(tmp_path):
    (tmp_path / "papers").mkdir()
    (tmp_path / "PaperLens.md").write_text(
        "# PaperLens\n\n- [A] [Bad](./papers/bad.md)\n", encoding="utf-8"
    )
    (tmp_path / "papers" / "bad.md").write_text(
        '# Bad\n\n<img src="../.paperlens/pages/p_test/page_0001.png">\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as exc:
        validate_paperlens_output(tmp_path)

    assert "Full-page render embedded" in str(exc.value)


def test_output_validation_rejects_reader_hostile_report_phrases(tmp_path):
    (tmp_path / "papers").mkdir()
    (tmp_path / "PaperLens.md").write_text(
        "# PaperLens\n\n- [A] [Bad](./papers/bad.md)\n", encoding="utf-8"
    )
    (tmp_path / "papers" / "bad.md").write_text(
        "# Bad\n\n由于提供的证据不完整，这里不展开。\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError) as exc:
        validate_paperlens_output(tmp_path)

    assert "Reader-hostile implementation wording" in str(exc.value)


def test_output_validation_counts_only_canonical_paper_reports(tmp_path):
    (tmp_path / "papers").mkdir()
    (tmp_path / ".paperlens" / "library" / "index").mkdir(parents=True)
    memory_v3_dir = tmp_path / ".paperlens" / "data" / "memory" / "v3"
    memory_v3_dir.mkdir(parents=True)
    (tmp_path / "PaperLens.md").write_text(
        "# PaperLens\n\n- [A] [Good](./papers/p_test.md)\n",
        encoding="utf-8",
    )
    (tmp_path / "papers" / "p_test.md").write_text("# Good\n\nA useful report.\n", encoding="utf-8")
    (tmp_path / ".paperlens" / "library" / LIBRARY_RECORD_FILENAME).write_text(
        "{}\n", encoding="utf-8"
    )
    (tmp_path / ".paperlens" / "library" / "index" / "search_index.json").write_text(
        "[]", encoding="utf-8"
    )
    (memory_v3_dir / "p_test.paper_memory.v3.json").write_text("{}", encoding="utf-8")

    result = validate_paperlens_output(tmp_path)

    assert result["status"] == "PASS"
    assert result["paper_reports"] == 1
    assert result["paper_report_files"] == 1
    assert result["paper_memory_v3"] == 1


def test_render_freeform_report_normalizes_model_newlines():
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    decision = ClassificationDecision(
        paper_id="p_test",
        class_label="A",
        confidence=0.9,
        false_negative_risk=0.1,
    )
    markdown = render_freeform_paper_report(
        paper=paper,
        decision=decision,
        model_report={
            "grade": "A",
            "read_recommendation": "重点关注",
            "one_line_reason": "核心理由\\n不该断在行内",
            "explanation_markdown": "第一段。\\n\\n第二段。",
            "uncertainty_note": "",
        },
        report_audit={"verdict": "PASS", "safe_usage_note": ""},
    )

    assert "\\n" not in markdown
    assert "第一段。\n\n第二段。" in markdown
    assert "> 核心理由 不该断在行内" in markdown


def test_freeform_report_sanitizes_hostile_phrases_and_redundant_anchor():
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    decision = ClassificationDecision(
        paper_id="p_test",
        class_label="A",
        confidence=0.9,
        false_negative_risk=0.1,
    )
    markdown = render_freeform_paper_report(
        paper=paper,
        decision=decision,
        model_report={
            "grade": "A",
            "read_recommendation": "重点关注",
            "one_line_reason": "供给的片段显示它提出了一个抽象。",
            "core_takeaway": "把 KV cache 当成分页内存，而不是连续数组。",
            "explanation_markdown": (
                "这篇论文的核心抽象是把 KV cache 当成分页内存，而不是连续数组。"
                "为什么这很重要？因为连续分配会制造碎片。"
                "\n\n你给到的片段没有覆盖全部实验，提供的页面和提供的证据也没有完整展示。"
            ),
            "uncertainty_note": "",
        },
        report_audit={"verdict": "PASS", "safe_usage_note": ""},
    )

    assert "供给的片段" not in markdown
    assert "你给到" not in markdown
    assert "提供的页面" not in markdown
    assert "提供的证据" not in markdown
    assert "当前自动阅读证据" in markdown
    assert "这篇论文的核心抽象是把 KV cache 当成分页内存" not in markdown
    assert "为什么这很重要？因为连续分配会制造碎片。" in markdown


def test_paperlens_report_hides_zero_cost_from_resume_without_calls():
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    decision = ClassificationDecision(
        paper_id="p_test",
        class_label="A",
        confidence=0.9,
        false_negative_risk=0.1,
    )

    markdown = render_paperlens_report(
        rows=[
            {
                "paper": paper,
                "decision": decision,
                "report_name": "p_test.md",
                "model_report": {"one_line_reason": "核心理由"},
            }
        ],
        review_items=[],
        budget={"estimated_usd": 0.0, "calls": 0},
        topic=None,
        idea=None,
        formal_run=True,
    )

    assert "模型成本估算" not in markdown


def test_output_language_is_a_single_rendering_preference():
    assert CoreConfig(output_language="zh").output_language == "zh"
    assert CoreConfig(output_language="en").output_language == "en"


def test_read_mode_defaults_to_standard_only():
    assert CoreConfig().read_mode == "standard"
    assert CoreConfig(read_mode="standard").read_mode == "standard"
    with pytest.raises(ValueError):
        CoreConfig(read_mode="deep")
    with pytest.raises(ValueError):
        CoreConfig(read_mode="unknown")


def test_mimo_model_names_are_normalized_for_requests():
    provider = ProviderConfig(model=" MiMo-V2.5-Pro ")

    assert provider.model == "mimo-v2.5-pro"
    assert provider.request_model() == "mimo-v2.5-pro"
    assert ProviderConfig(model="gpt-5.5").model == "gpt-5.5"
    assert ProviderConfig(reasoning_model="MIMO-V2.5").reasoning_model == "mimo-v2.5"


def test_report_markdown_repair_splits_inline_headings_without_rewriting_claims():
    cleaned = clean_model_markdown(
        "问题背景。## 核心抽象\nPagedAttention彻底消除了内存碎片，实现了近乎零浪费。 "
        "## 机制如何改变系统行为 这种分页机制带来了几个改变： 1. **消除内存碎片**：更灵活。"
    )
    rendered = sanitize_reader_hostile_text(cleaned)

    assert "问题背景。\n\n## 核心抽象" in cleaned
    assert "## 核心抽象\nPagedAttention" in cleaned
    assert "## 机制如何改变系统行为\n\n这种分页机制" in cleaned
    assert "改变：\n\n1. **消除内存碎片**" in cleaned
    assert rendered == cleaned


def test_core_engine_records_pipeline_exceptions(monkeypatch, tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    def fail_pipeline(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("paperlens_core.engine.run_pipeline", fail_pipeline)

    result = PaperLensEngine().run_job(
        RunRequest(
            input_dir=input_dir,
            output_dir=output_dir,
            config_overrides={"offline_debug": True, "provider": {"kind": "none"}},
        )
    )

    assert result.status == "error"
    assert result.data["reason"] == "boom"
    assert "boom" in (output_dir / ".paperlens" / "data" / "events.jsonl").read_text(
        encoding="utf-8"
    )


def test_report_plan_prompt_uses_memory_and_local_tool_context():
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    decision = ClassificationDecision(
        paper_id="p_test",
        class_label="A",
        confidence=0.9,
        false_negative_risk=0.1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "PAGE ONE front matter",
                "captions": [],
                "figures": [],
                "tables": [],
            },
            {
                "page_no": 2,
                "text": "PAGE TWO should not be included",
                "captions": [],
                "figures": [],
                "tables": [],
            },
            {
                "page_no": 5,
                "text": "PAGE FIVE key evaluation evidence",
                "captions": [],
                "figures": [],
                "tables": [],
            },
        ]
    }
    memory = {
        "schema_version": "paper_memory.v3",
        "paper_id": "p_test",
        "problem_frame": {"problem": "key problem", "why_it_matters": "why it matters"},
        "core_abstractions": [{"id": "A001", "text": "core abstraction"}],
        "evidence": [{"id": "E5", "page": 5, "interpretation": "evaluation evidence"}],
        "claims": [{"id": "C1", "text": "claim", "evidence_refs": ["E5"]}],
    }

    prompt = build_report_plan_prompt(
        paper=paper,
        skim=None,
        decision=decision,
        card=None,
        paper_memory=memory,
        layout=layout,
        topic=None,
        idea=None,
        output_language="zh",
        read_mode="standard",
    )

    assert "ReportPlan" in prompt
    assert "PAGE FIVE key evaluation evidence" in prompt
    assert '"page_no": 5' in prompt
    assert "whole report" in prompt


def test_report_section_prompt_uses_expanded_memory_and_plan():
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    memory = {
        "paper_memory_v3": {
            "schema_version": "paper_memory.v3",
            "paper_id": "p_test",
            "problem_frame": {
                "problem": "A detailed problem statement.",
                "why_it_matters": "The system consequence matters.",
                "scope": "system",
            },
            "core_abstractions": [
                {
                    "id": "A001",
                    "text": "A reusable abstraction.",
                    "misunderstanding_guard": "Do not over-extend the analogy.",
                    "evidence_refs": ["E001"],
                }
            ],
            "mechanism": {
                "overview": "The mechanism overview needs several moving parts.",
                "steps": [
                    {"id": "M001", "text": "First mechanism step that should reach report agent."},
                    {"id": "M002", "text": "Second mechanism step that should reach report agent."},
                ],
            },
            "evaluation": {
                "summary": "The evaluation summary.",
                "items": [{"id": "V001", "text": "Benchmark and metric detail."}],
            },
            "conceptual_bridge": {"needed": False, "reader_gap": "", "bridge_text": "", "terms": []},
            "concepts": [{"term": "KV cache", "explanation": "Background concept for the report."}],
            "claims": [
                {
                    "id": "C001",
                    "text": "A supported mechanism claim.",
                    "type": "mechanism",
                    "provenance": "explicit",
                    "confidence": "high",
                    "critic_status": "checked",
                    "evidence_refs": ["E001"],
                }
            ],
            "evidence": [
                {
                    "id": "E001",
                    "source_type": "text_span",
                    "page": 5,
                    "section": "Design",
                    "excerpt_or_caption": "Source text for the claim.",
                    "interpretation": "Why this source matters.",
                    "reliability": "direct",
                }
            ],
            "figures_tables": [
                {"id": "F001", "source_type": "figure", "page": 5, "caption": "Important figure."}
            ],
            "limitations": ["A limitation that should not be lost."],
            "open_questions": [],
            "audit_trail": {},
        }
    }

    plan = {
        "paper_id": "p_test",
        "grade": "A",
        "read_recommendation": "重点关注",
        "one_line_reason": "reason",
        "core_takeaway": "takeaway",
        "sections": [
            {
                "section_id": "mechanism",
                "section_kind": "mechanism",
                "title": "机制如何工作",
                "purpose": "explain mechanism",
                "focus_queries": ["mechanism"],
                "claim_ids": ["C001"],
                "evidence_refs": ["E001"],
                "target_pages": [5],
                "detail_questions": ["What data structures make it work?"],
            }
        ],
    }

    prompt = build_report_section_prompt(
        paper=paper,
        paper_memory=memory["paper_memory_v3"],
        layout={"pages": []},
        plan=plan,
        section_plan=plan["sections"][0],
        previous_summaries=[],
        output_language="zh",
        read_mode="standard",
    )

    assert REPORT_PLAN_SCHEMA["properties"]["sections"]["maxItems"] == 7
    assert "section_kind" in REPORT_PLAN_SCHEMA["properties"]["sections"]["items"]["properties"]
    assert "detail_questions" in REPORT_PLAN_SCHEMA["properties"]["sections"]["items"]["properties"]
    assert REPORT_SECTION_SCHEMA["properties"]["paragraphs"]["maxItems"] == 12
    assert "write only this planned section" in prompt
    assert "Mechanism section contract" in prompt
    assert "state or bottleneck" in prompt
    assert "data structures" in prompt
    assert "request/object lifecycle" in prompt
    assert "First mechanism step that should reach report agent" in prompt
    assert "Benchmark and metric detail" in prompt
    assert "KV cache" in prompt
    assert "Source text for the claim" in prompt


def test_report_section_prompt_can_target_english_output():
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    prompt = build_report_section_prompt(
        paper=paper,
        paper_memory={},
        layout={"pages": []},
        plan={"grade": "A", "sections": []},
        section_plan={
            "section_id": "idea",
            "section_kind": "orientation",
            "title": "Core idea",
            "purpose": "explain",
            "focus_queries": [],
            "claim_ids": [],
            "evidence_refs": [],
            "target_pages": [],
        },
        previous_summaries=[],
        output_language="en",
        read_mode="standard",
    )

    assert "output_language: en" in prompt
    assert "Task: write only this planned section" in prompt


def test_report_section_audit_prompt_checks_only_one_section():
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    section_plan = {
        "section_id": "evidence",
        "title": "证据",
        "purpose": "check evidence",
        "focus_queries": ["evidence"],
        "claim_ids": ["C1"],
        "evidence_refs": ["E5"],
        "target_pages": [5],
    }
    prompt = build_report_section_audit_prompt(
        paper=paper,
        paper_memory={"evidence": [{"id": "E5", "page": 5}]},
        layout={"pages": [{"page_no": 5, "text": "PAGE FIVE raw source text"}]},
        plan={"grade": "A", "sections": [section_plan]},
        section_plan=section_plan,
        section={"section_id": "evidence", "markdown": "report body"},
        output_language="zh",
        read_mode="standard",
    )

    assert "generated_section" in prompt
    assert "Audit one generated report section" in prompt
    assert "PAGE FIVE raw source text" in prompt


def test_freeform_report_hides_audit_todo_and_never_embeds_full_page_visuals():
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    decision = ClassificationDecision(
        paper_id="p_test",
        class_label="A",
        confidence=0.9,
        false_negative_risk=0.1,
    )
    card = PaperCard(
        paper_id="p_test",
        evidence_refs=[EvidenceRef(paper_id="p_test", page_no=3, figure_id="fig_1")],
    )
    markdown = render_freeform_paper_report(
        paper=paper,
        decision=decision,
        card=card,
        layout={
            "pages": [
                {
                    "page_no": 3,
                    "captions": [{"text": "Figure 1 shows the system overview."}],
                    "figures": [{"figure_id": "fig_1"}],
                }
            ]
        },
        model_report={
            "grade": "A",
            "read_recommendation": "重点关注",
            "one_line_reason": "核心理由",
            "core_takeaway": "先把这篇论文理解成一个内存抽象变化。",
            "explanation_markdown": "这是一份自然语言胶囊。",
            "uncertainty_note": "本报告计划基于已验证的PaperMemoryV3内容；具体数值仍需按需核对。",
            "key_visual_pages": [{"page_no": 3, "reason": "这张图帮你先看到系统抽象。"}],
        },
        report_audit={
            "verdict": "PASS_WITH_WEAKNESSES",
            "safe_usage_note": "English internal note",
            "correction_notes": ["Fix this internal issue"],
        },
    )

    assert "复核备注" not in markdown
    assert "需要修正或确认" not in markdown
    assert "Fix this internal issue" not in markdown
    assert "**先抓住这个抽象：** 先把这篇论文理解成一个内存抽象变化。" in markdown
    assert "## 关键图表" not in markdown
    assert "../.paperlens/pages/p_test/page_0003.png" not in markdown
    assert "这张图帮你先看到系统抽象。" not in markdown
    assert "Figure 1 shows the system overview." not in markdown
    assert "本报告" not in markdown
    assert "本报告计划" not in markdown
    assert "PaperMemoryV3" not in markdown
    assert "具体数值仍需按需核对。" in markdown
    assert "可信边界：" in markdown
    assert markdown.index("这是一份自然语言胶囊。") < markdown.index("可信边界：")


def test_visual_crop_bbox_uses_table_and_nearby_caption():
    crop = visual_crop_bbox_for_page(
        {
            "page_no": 2,
            "page_width": 600,
            "page_height": 800,
            "tables": [{"bbox": [80, 100, 280, 190]}],
            "captions": [{"bbox": [70, 210, 300, 245], "text": "Figure 2. Memory waste."}],
        }
    )

    assert crop is not None
    assert crop[0] <= 70
    assert crop[1] <= 100
    assert crop[2] >= 300
    assert crop[3] >= 245


def test_visual_crop_bbox_rejects_full_page_bbox():
    crop = visual_crop_bbox_for_page(
        {
            "page_no": 1,
            "page_width": 600,
            "page_height": 800,
            "figures": [{"bbox": [0, 0, 600, 800]}],
            "captions": [],
        }
    )

    assert crop is None


def test_readable_model_body_splits_very_long_paragraphs():
    long_body = "第一句解释背景。" * 45 + "第二句解释机制。" * 45 + "第三句解释证据。" * 45

    readable = readable_model_body(long_body)

    assert "\n\n" in readable
    assert all(len(part) <= 700 for part in readable.split("\n\n"))


def test_internal_targeted_reread_items_are_not_user_visible():
    items = [
        ReviewItem(
            item_id="targeted:p_test",
            paper_id="p_test",
            item_type="NEED_TARGETED_REREAD",
            reason="missing intermediate PaperCard detail",
        ),
        ReviewItem(
            item_id="evidence:p_test",
            paper_id="p_test",
            item_type="evidence",
            reason="real unsupported claim",
        ),
    ]

    visible = user_visible_review_items(items)

    assert [item.item_id for item in visible] == ["evidence:p_test"]


def test_final_report_audit_acceptance_gate():
    assert final_report_audit_acceptable({"verdict": "PASS"})
    assert final_report_audit_acceptable({"verdict": "PASS_WITH_WEAKNESSES"})
    assert not final_report_audit_acceptable({"verdict": "NEED_HUMAN_REVIEW"})
    assert not final_report_audit_acceptable(None)


def test_memory_patch_set_updates_capsule_fields(tmp_path):
    data_dir = tmp_path / "out" / ".paperlens" / "data"
    store = PaperMemoryStore(data_dir)
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    skim = SkimCard(paper_id="p_test", problem="solves cache pressure")
    decision = ClassificationDecision(
        paper_id="p_test",
        class_label="A",
        confidence=0.8,
        false_negative_risk=0.2,
    )
    store.initialize(
        paper=paper,
        skim=skim,
        decision=decision,
        card=None,
        layout=None,
        source="test_seed",
        prefer_existing=False,
    )
    memory = store.apply_patch_set(
        "p_test",
        {
            "paper_id": "p_test",
            "operations": [
                {"op": "add_read_pages", "payload": {"pages": [1, 2]}},
                {
                    "op": "set_problem_frame",
                    "payload": {
                        "problem": "The paper turns cache eviction into a queueing problem.",
                        "why_it_matters": "Cache eviction policy affects hit rate and cost.",
                    },
                },
                {
                    "op": "set_core_abstraction",
                    "payload": {
                        "text": "FIFO queues can approximate expensive eviction decisions."
                    },
                },
                {
                    "op": "set_mechanism_overview",
                    "payload": {"text": "Use simple queues instead of precise recency tracking."},
                },
                {
                    "op": "set_evaluation_summary",
                    "payload": {"text": "Evaluated against common cache traces."},
                },
                {
                    "op": "upsert_concept",
                    "payload": {
                        "term": "cache eviction",
                        "explanation": "Choosing what item to remove.",
                    },
                },
                {
                    "op": "set_conceptual_bridge",
                    "payload": {
                        "needed": True,
                        "reader_gap": "The reader may not know why eviction matters.",
                        "bridge_text": "A cache has finite space, so every miss can force a removal choice.",
                    },
                },
                {
                    "op": "upsert_conceptual_bridge_term",
                    "payload": {
                        "term": "cache eviction",
                        "explanation": "Choosing which cached item to remove.",
                        "paper_role": "It is the decision problem the paper simplifies.",
                        "provenance": "background",
                    },
                },
                {
                    "op": "upsert_evidence",
                    "payload": {
                        "id": "E010",
                        "page": 1,
                        "interpretation": "FIFO can be competitive.",
                        "excerpt_or_caption": "FIFO",
                        "reliability": "direct",
                    },
                },
                {
                    "op": "upsert_claim",
                    "payload": {
                        "id": "C010",
                        "text": "FIFO can be competitive.",
                        "type": "evaluation",
                        "confidence": "high",
                        "provenance": "explicit",
                        "evidence_refs": ["E010"],
                    },
                },
                {"op": "add_limitation", "payload": {"text": "Trace coverage may not generalize."}},
                {"op": "add_open_question", "payload": {"text": "When does FIFO fail?"}},
            ],
        },
        source="test_patch",
    )

    assert memory["problem_frame"]["problem"].startswith("The paper turns")
    assert memory["core_abstractions"][0]["text"].startswith("FIFO queues")
    assert memory["concepts"][0]["term"] == "cache eviction"
    assert memory["conceptual_bridge"]["needed"] is True
    assert memory["conceptual_bridge"]["terms"][0]["provenance"] == "background"
    assert any(claim["confidence"] == "high" for claim in memory["claims"])
    assert "When does FIFO fail?" in memory["open_questions"]


def test_paper_memory_v3_builds_claim_evidence_ir(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    paper = PaperRecord(
        paper_id="p_vllm",
        file_path="vllm.pdf",
        file_hash="hash",
        canonical_title="PagedAttention",
        page_count=12,
    )
    skim = SkimCard(
        paper_id="p_vllm",
        problem="KV cache fragmentation limits LLM serving.",
        method_type="paged KV cache",
        system_scope="LLM serving",
        evidence_refs=[EvidenceRef(paper_id="p_vllm", page_no=2, verification_status="PASS")],
    )
    decision = ClassificationDecision(
        paper_id="p_vllm",
        class_label="A",
        confidence=0.9,
        false_negative_risk=0.1,
    )
    card = PaperCard(
        paper_id="p_vllm",
        contribution_claims=["PagedAttention maps logical KV blocks to physical GPU blocks."],
        mechanisms=["A block table decouples logical KV order from physical allocation."],
        evaluation=["The evaluation measures serving throughput and memory waste."],
        evidence_refs=[
            EvidenceRef(paper_id="p_vllm", page_no=5, figure_id="fig_1", verification_status="PASS")
        ],
    )
    store = PaperMemoryStore(data_dir)
    store.initialize(
        paper=paper,
        skim=skim,
        decision=decision,
        card=card,
        layout={"pages": [{"page_no": 5, "captions": [{"text": "Figure 1. System overview."}]}]},
        source="test_seed",
        prefer_existing=False,
    )
    memory = store.apply_patch_set(
        "p_vllm",
        {
            "paper_id": "p_vllm",
            "operations": [
                {"op": "add_read_pages", "payload": {"pages": [1, 2, 5]}},
                {
                    "op": "set_problem_frame",
                    "payload": {
                        "problem": "KV cache fragmentation is a serving bottleneck.",
                        "why_it_matters": "KV cache pressure limits LLM serving concurrency.",
                    },
                },
                {
                    "op": "set_core_abstraction",
                    "payload": {"text": "Treat KV cache as paged memory."},
                },
                {
                    "op": "set_mechanism_overview",
                    "payload": {"text": "Split KV cache into fixed-size blocks."},
                },
                {
                    "op": "set_evaluation_summary",
                    "payload": {"text": "Compare memory waste and throughput."},
                },
                {
                    "op": "upsert_concept",
                    "payload": {
                        "term": "KV cache",
                        "explanation": "Stored keys and values reused during decoding.",
                    },
                },
                {
                    "op": "set_conceptual_bridge",
                    "payload": {
                        "needed": True,
                        "reader_gap": "KV cache, batch, and decode are needed before PagedAttention is intuitive.",
                        "bridge_text": "During decode, each request keeps growing KV state; batching is limited by how much of that state fits in GPU memory.",
                    },
                },
                {
                    "op": "upsert_conceptual_bridge_term",
                    "payload": {
                        "term": "KV cache",
                        "explanation": "Stored key/value tensors from earlier tokens.",
                        "paper_role": "This is the state PagedAttention manages.",
                        "provenance": "background",
                    },
                },
                {
                    "op": "upsert_conceptual_bridge_term",
                    "payload": {
                        "term": "decode",
                        "explanation": "Autoregressive generation after the prompt.",
                        "paper_role": "Decode makes the KV cache grow token by token.",
                        "provenance": "background",
                    },
                },
                {
                    "op": "upsert_evidence",
                    "payload": {
                        "id": "E010",
                        "page": 2,
                        "interpretation": "Paging reduces fragmentation.",
                        "excerpt_or_caption": "memory waste",
                        "reliability": "direct",
                    },
                },
                {
                    "op": "upsert_claim",
                    "payload": {
                        "id": "C010",
                        "text": "Paging reduces KV cache fragmentation.",
                        "type": "mechanism",
                        "confidence": "high",
                        "provenance": "explicit",
                        "evidence_refs": ["E010"],
                    },
                },
                {
                    "op": "add_limitation",
                    "payload": {"text": "Benefit depends on KV cache pressure."},
                },
                {"op": "set_memory_audit", "payload": {"status": "PASS"}},
            ],
        },
        source="test_patch",
    )
    written = write_paper_memory_v3_file(data_dir, memory)
    loaded = read_paper_memory_v3(output_dir, "p_vllm")

    assert memory["schema_version"] == MEMORY_V3_SCHEMA_VERSION
    assert validate_paper_memory_v3(memory) == []
    assert memory["conceptual_bridge"]["terms"][0]["term"] == "KV cache"
    assert written.name.endswith("paper_memory.v3.json")
    assert loaded["claims"][0]["id"] == "C001"
    assert loaded["claims"][0]["evidence_refs"]


def test_paper_memory_store_materializes_patchable_v3_state(tmp_path):
    data_dir = tmp_path / "out" / ".paperlens" / "data"
    store = PaperMemoryStore(data_dir)
    paper = PaperRecord(
        paper_id="p_store",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Patchable Memory",
        page_count=4,
    )
    skim = SkimCard(
        paper_id="p_store",
        problem="KV cache wastes memory.",
        evidence_refs=[EvidenceRef(paper_id="p_store", page_no=1, verification_status="PASS")],
    )
    decision = ClassificationDecision(
        paper_id="p_store",
        class_label="A",
        confidence=0.8,
        false_negative_risk=0.1,
    )
    memory = store.initialize(
        paper=paper,
        skim=skim,
        decision=decision,
        card=None,
        layout={"pages": [{"page_no": 1, "captions": []}]},
        source="test_seed",
        prefer_existing=False,
    )
    seeded = store.apply_patch_set(
        "p_store",
        {
            "paper_id": "p_store",
            "operations": [
                {"op": "add_read_pages", "payload": {"pages": [1]}},
                {
                    "op": "set_problem_frame",
                    "payload": {"problem": "KV cache wastes serving memory."},
                },
                {
                    "op": "upsert_evidence",
                    "payload": {
                        "id": "E010",
                        "page": 1,
                        "interpretation": "KV cache memory waste.",
                    },
                },
                {
                    "op": "upsert_claim",
                    "payload": {
                        "id": "C010",
                        "text": "The paper studies KV cache memory waste.",
                        "confidence": "medium",
                        "evidence_refs": ["E010"],
                    },
                },
            ],
        },
        source="test_seed_patch",
    )
    repaired = store.apply_patch(
        "p_store",
        {
            "operation": "upsert_claim",
            "source": "memory_repair",
            "payload": {
                "text": "Paged allocation reduces avoidable KV cache waste.",
                "type": "mechanism",
                "confidence": "medium",
                "provenance": "inferred",
            },
        },
    )
    log_text = store.patch_log_path("p_store").read_text(encoding="utf-8")

    assert memory["schema_version"] == MEMORY_V3_SCHEMA_VERSION
    assert seeded["reading_context"]["pages_read"] == [1]
    assert "Paged allocation reduces" in json.dumps(repaired["claims"], ensure_ascii=False)
    assert MEMORY_PATCH_SCHEMA_VERSION in log_text
    assert repaired["claims"]


def test_ask_prompt_includes_memory_without_report_body(tmp_path):
    prompt = build_ask_prompt(
        report_path=tmp_path / "papers" / "p_test.md",
        paper_id="p_test",
        question="这个 idea 是什么？",
        paper_memory_v3={
            "schema_version": "paper_memory.v3",
            "problem_frame": {"problem": "核心主张"},
            "conceptual_bridge": {
                "needed": True,
                "bridge_text": "背景桥接",
                "terms": [{"term": "decode", "explanation": "逐 token 生成"}],
            },
            "claims": [{"id": "C001", "text": "关键 claim", "evidence_refs": ["E001"]}],
            "evidence": [{"id": "E001", "page": 1, "interpretation": "关键证据"}],
        },
        pages=[{"page_no": 1, "text": "page text", "captions": [], "visual_notes": []}],
        question_type="mechanism",
    )

    assert ("leg" + "acy_memory_") not in prompt
    assert "paper_memory_v3_ir:" in prompt
    assert "核心主张" in prompt
    assert "C001" in prompt
    assert "背景桥接" in prompt
    assert "question_type: mechanism" in prompt
    assert "agent_context_pack:" in prompt
    assert "paper_card:" not in prompt
    assert "current_paper_report:" not in prompt
    assert "短报告" not in prompt


def test_qa_question_classifier_marks_challenge_questions():
    assert classify_question("这个结论我不信，回原文核对") == "evidence_check"
    assert classify_question("PagedAttention 是不是 OS paging？解释一下") == "clarification"
    assert (
        classify_question("PagedAttention 和 OS paging 到底像在哪里，不像在哪里？") == "comparison"
    )


def test_ask_prompt_requires_background_claim_separation():
    assert "background knowledge" in ASK_SYSTEM_PROMPT
    assert "not a paper claim" in ASK_SYSTEM_PROMPT
    assert "source_attribution" in ASK_SYSTEM_PROMPT
    assert "supplied excerpts" not in ASK_SYSTEM_PROMPT


def test_paper_qa_normalization_preserves_source_attribution():
    answer = normalize_answer(
        {
            "answer_markdown": "论文主张是 KV cache 分页；背景上这类似 OS paging。",
            "cited_pages": [2],
            "confidence": "medium",
            "source_attribution": {
                "paper_claims": ["KV cache is paged."],
                "paperlens_inferences": ["This is a memory-management abstraction."],
                "background_context": ["OS paging is background context."],
                "evidence_limits": ["No exact throughput number cited."],
            },
        }
    )

    assert answer["source_attribution"]["paper_claims"] == ["KV cache is paged."]
    assert answer["source_attribution"]["background_context"] == [
        "OS paging is background context."
    ]


def test_paper_qa_normalization_removes_reader_hostile_wording():
    answer = normalize_answer(
        {
            "answer_markdown": "The supplied excerpts do not cover all ablations. 你给到的片段没有完整评测表，提供的材料和提供的证据也不完整。",
            "cited_pages": [2],
            "confidence": "medium",
            "source_attribution": {
                "paper_claims": [],
                "paperlens_inferences": [],
                "background_context": [],
                "evidence_limits": [
                    "The user provided no exact throughput number in the supplied excerpts."
                ],
            },
        }
    )

    payload = json.dumps(answer, ensure_ascii=False)
    assert "supplied excerpts" not in payload
    assert "The user provided" not in payload
    assert "你给到" not in payload
    assert "提供的材料" not in payload
    assert "提供的证据" not in payload
    assert "automatic reading evidence" in payload
    assert "自动读取到的证据" in payload


def test_paper_qa_low_confidence_adds_evidence_limit_for_minimal_shape():
    answer = normalize_answer(
        {
            "answer_markdown": "只能保守回答。",
            "confidence": "low",
        }
    )

    assert answer["source_attribution"]["evidence_limits"]


def test_paperlens_library_writes_single_memory_asset_and_searches(tmp_path):
    output_dir = tmp_path / "out"
    paper = PaperRecord(
        paper_id="p_vllm",
        file_path="vllm.pdf",
        file_hash="hash",
        canonical_title="Efficient Memory Management for LLM Serving with PagedAttention",
        page_count=16,
    )
    skim = SkimCard(
        paper_id="p_vllm",
        problem="KV cache fragmentation limits LLM serving throughput.",
        method_type="paged KV cache management",
        system_scope="LLM serving",
    )
    decision = ClassificationDecision(
        paper_id="p_vllm",
        class_label="A",
        confidence=0.9,
        false_negative_risk=0.1,
    )
    card = PaperCard(
        paper_id="p_vllm",
        contribution_claims=["PagedAttention maps logical KV blocks to physical GPU blocks."],
        mechanisms=["Block table indirection enables non-contiguous KV cache allocation."],
        evaluation=["The paper evaluates serving throughput and memory waste."],
    )

    write_paperlens_library(
        output_dir=output_dir,
        rows=[
            {
                "paper": paper,
                "skim": skim,
                "decision": decision,
                "card": card,
                "paper_memory_v3": {
                    "schema_version": "paper_memory.v3",
                    "paper_id": "p_vllm",
                    "problem_frame": {"problem": "KV cache wastes GPU memory during LLM serving."},
                    "core_abstractions": [
                        {
                            "id": "A001",
                            "text": "Use OS-style paging for LLM KV cache.",
                            "evidence_refs": ["E001"],
                        }
                    ],
                    "mechanism": {
                        "steps": [
                            {
                                "id": "M001",
                                "text": "KV cache is split into fixed-size blocks and mapped through a block table.",
                            }
                        ]
                    },
                    "evaluation": {
                        "items": [
                            {
                                "id": "V001",
                                "text": "Experiments compare serving throughput and memory waste.",
                            }
                        ]
                    },
                    "concepts": [
                        {
                            "term": "KV cache",
                            "explanation": "Transformer state kept for generation.",
                        }
                    ],
                    "conceptual_bridge": {"needed": False, "terms": []},
                    "claims": [
                        {
                            "id": "C001",
                            "text": "Paging reduces KV cache fragmentation.",
                            "type": "mechanism",
                            "provenance": "explicit",
                            "confidence": "high",
                            "evidence_refs": ["E001"],
                            "critic_status": "checked",
                        }
                    ],
                    "evidence": [
                        {
                            "id": "E001",
                            "source_type": "text_span",
                            "page": 2,
                            "interpretation": "The paper describes paging KV cache blocks.",
                            "reliability": "direct",
                        }
                    ],
                    "limitations": ["Benefit depends on KV cache pressure."],
                    "open_questions": ["When does block-table overhead matter?"],
                },
                "report_name": "p_vllm_pagedattention.md",
                "report_title": "Efficient Memory Management for LLM Serving with PagedAttention",
                "model_report": {
                    "one_line_reason": "它把 KV cache 管理变成分页式状态管理。",
                    "core_takeaway": "KV cache 可以像分页内存一样被管理。",
                    "read_recommendation": "重点关注",
                },
                "report_audit": {"verdict": "PASS_WITH_WEAKNESSES"},
            }
        ],
        topic=None,
        idea=None,
    )

    records = read_library_records(output_dir)
    result = search_library(output_dir=output_dir, query="KV cache paging", limit=3)
    chinese_result = search_library(
        output_dir=output_dir, query="哪些论文讲内存管理和缓存？", limit=3
    )

    assert (output_dir / ".paperlens" / "library" / LIBRARY_RECORD_FILENAME).exists()
    assert (output_dir / ".paperlens" / "library" / "index" / "search_index.json").exists()
    assert records[0]["schema_version"] == LIBRARY_RECORD_SCHEMA_VERSION
    assert records[0]["outputs"]["briefing_md"] == "papers/p_vllm_pagedattention.md"
    assert records[0]["memory"]["mechanism_steps"]
    assert records[0]["memory"]["reader_takeaways"]
    assert records[0]["quality"]["claim_count"] >= 1
    assert validate_memory_record(records[0]) == []
    assert doctor_library(output_dir)["status"] == "PASS"
    assert result["matches"][0]["paper"]["paper_id"] == "p_vllm"
    assert chinese_result["matches"][0]["paper"]["paper_id"] == "p_vllm"


def test_library_search_expands_common_chinese_system_terms():
    terms = expand_search_query_terms("这些论文里哪些和内存管理、系统调度、缓存淘汰有关？")

    assert "memory" in terms
    assert "management" in terms
    assert "scheduling" in terms
    assert "cache" in terms
    assert "eviction" in terms


def test_core_cli_configures_utf8_stdio_for_non_ascii_output(monkeypatch):
    class EncodingCheckingStream:
        def __init__(self) -> None:
            self.encoding = "gbk"
            self.errors = "strict"

        def reconfigure(self, *, encoding: str, errors: str) -> None:
            self.encoding = encoding
            self.errors = errors

        def write(self, text: str) -> int:
            text.encode(self.encoding, errors=self.errors)
            return len(text)

        def flush(self) -> None:
            return None

    fake_stdout = EncodingCheckingStream()
    fake_stderr = EncodingCheckingStream()
    monkeypatch.setattr(sys, "stdout", fake_stdout)
    monkeypatch.setattr(sys, "stderr", fake_stderr)

    configure_utf8_stdio()
    print("FIFO Queues are All You Need ∗")

    assert fake_stdout.encoding == "utf-8"
    assert fake_stderr.encoding == "utf-8"


def test_openai_compatible_base_url_is_not_modified():
    client = JsonLlmClient(
        ProviderConfig(
            kind="openai-compatible",
            base_url="https://gateway.example/",
            model="fake-model",
            api_key="fake-key",
        )
    )

    assert client._base_url("https://api.openai.com/v1") == "https://gateway.example"


def test_mimo_compatible_payload_disables_thinking(monkeypatch):
    monkeypatch.delenv("PAPERLENS_MIMO_THINKING", raising=False)
    monkeypatch.delenv("PAPERLENS_MIMO_THINKING_SCHEMAS", raising=False)
    captured: list[dict[str, object]] = []

    def fake_post_json(self, endpoint, payload, headers):  # noqa: ANN001
        captured.append(payload)
        return (
            {
                "choices": [
                    {
                        "message": {
                            "content": '{"ok": true}',
                        }
                    }
                ],
                "usage": {},
            },
            {},
        )

    monkeypatch.setattr(JsonLlmClient, "_post_json", fake_post_json)
    client = JsonLlmClient(
        ProviderConfig(
            kind="openai-compatible",
            base_url="https://gateway.example/v1",
            model="MiMo-V2.5",
            api_key="fake-key",
        )
    )

    result = client.invoke_json(
        system_prompt="Return JSON.",
        user_prompt="Return ok.",
        schema_name="mimo_test",
        schema={
            "type": "object",
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
        max_tokens=123,
    )

    assert result.data == {"ok": True}
    assert captured[0]["model"] == "mimo-v2.5"
    assert captured[0]["thinking"] == {"type": "disabled"}
    assert captured[0]["max_completion_tokens"] == 123
    assert "max_tokens" not in captured[0]


def test_mimo_thinking_can_be_enabled_for_selected_schema(monkeypatch):
    monkeypatch.setenv("PAPERLENS_MIMO_THINKING", "enabled")
    monkeypatch.setenv("PAPERLENS_MIMO_THINKING_SCHEMAS", "mimo_test")
    captured: list[dict[str, object]] = []

    def fake_post_json(self, endpoint, payload, headers):  # noqa: ANN001
        captured.append(payload)
        return (
            {
                "choices": [{"message": {"content": '{"ok": true}'}}],
                "usage": {},
            },
            {},
        )

    monkeypatch.setattr(JsonLlmClient, "_post_json", fake_post_json)
    client = JsonLlmClient(
        ProviderConfig(
            kind="openai-compatible",
            base_url="https://gateway.example/v1",
            model="mimo-v2.5",
            api_key="fake-key",
        )
    )

    result = client.invoke_json(
        system_prompt="Return JSON.",
        user_prompt="Return ok.",
        schema_name="mimo_test",
        schema={
            "type": "object",
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
        max_tokens=123,
    )

    assert result.data == {"ok": True}
    assert captured[0]["thinking"] == {"type": "enabled"}
    assert captured[0]["max_completion_tokens"] == 123


def test_chat_json_retry_expands_budget_after_truncation(monkeypatch):
    captured: list[dict[str, object]] = []
    responses = [
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": '{"ok":'},
                }
            ],
            "usage": {},
        },
        {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"ok": true}'},
                }
            ],
            "usage": {},
        },
    ]

    def fake_post_json(self, endpoint, payload, headers):  # noqa: ANN001
        captured.append(payload)
        return responses.pop(0), {}

    monkeypatch.setattr(JsonLlmClient, "_post_json", fake_post_json)
    client = JsonLlmClient(
        ProviderConfig(
            kind="openai-compatible",
            base_url="https://gateway.example/v1",
            model="mimo-v2.5",
            api_key="fake-key",
        )
    )

    result = client.invoke_json(
        system_prompt="Return JSON.",
        user_prompt="Return ok.",
        schema_name="mimo_test",
        schema={
            "type": "object",
            "required": ["ok"],
            "properties": {"ok": {"type": "boolean"}},
        },
        max_tokens=123,
    )

    assert result.data == {"ok": True}
    assert len(captured) == 2
    assert captured[0]["max_completion_tokens"] == 123
    assert captured[1]["response_format"] == {"type": "json_object"}
    assert captured[1]["max_completion_tokens"] >= 1600


def test_large_memory_schema_retry_has_higher_completion_floor():
    assert (
        json_retry_completion_minimum(
            "paperlens_memory_repair", max_tokens=1000, has_images=False
        )
        >= 4000
    )


def test_json_schema_parser_fills_nullable_required_defaults():
    data = parse_json_text_for_schema(
        '{"items": ["a"]}',
        "nullable_test",
        {
            "type": "object",
            "required": ["items", "optional_note"],
            "properties": {
                "items": {"type": "array", "items": {"type": "string"}},
                "optional_note": {"type": ["string", "null"]},
            },
        },
    )

    assert data == {"items": ["a"], "optional_note": None}


def test_library_ask_prompt_keeps_source_grounding():
    matches = [
        {
            "score": 9.0,
            "paper": {
                "paper_id": "p_vllm",
                "title": "PagedAttention",
                "grade": "A",
                "tags": ["kv", "cache"],
                "outputs": {"briefing_md": "papers/p_vllm.md"},
                "memory": {"core_idea": "KV cache paging"},
                "provenance": {"evidence_refs": [{"page_no": 2}]},
            },
        }
    ]

    prompt = build_library_ask_prompt(question="哪些论文讲 KV cache？", matches=matches)

    assert "retrieved_library_records" in prompt
    assert "specific papers" in prompt
    assert "cross-paper synthesis" in prompt
    assert "source_attribution" in prompt


def test_library_rejects_old_library_record_schema_in_dev(tmp_path):
    output_dir = tmp_path / "out"
    memory_dir = output_dir / ".paperlens" / "library"
    memory_dir.mkdir(parents=True)
    old_record = {
        "schema_version": "paper_memory.v1",
        "paper_id": "p_old",
        "title": "Old Paper",
        "grade": "B",
        "memory": {"brief": "old brief"},
        "provenance": {},
        "outputs": {"briefing_md": "papers/p_old.md"},
    }
    (memory_dir / LIBRARY_RECORD_FILENAME).write_text(
        json.dumps(old_record, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    assert read_library_records(output_dir) == []
    doctor = doctor_library(output_dir)
    assert doctor["status"] == "WARN"
    assert doctor["unsupported_versions"] == ["paper_memory.v1"]


def test_library_answer_normalization_preserves_source_attribution():
    answer = normalize_library_answer(
        {
            "answer_markdown": "PagedAttention 和 Memtis 都围绕内存管理，但一个服务 LLM，一个做 tiering。",
            "related_papers": [
                {
                    "paper_id": "p_vllm",
                    "title": "PagedAttention",
                    "report_path": "papers/p_vllm.md",
                    "why_related": "KV cache memory management",
                }
            ],
            "confidence": "medium",
            "source_attribution": {
                "paper_claims": ["PagedAttention manages KV cache memory."],
                "cross_paper_synthesis": ["Both are memory-management papers."],
                "background_context": ["Memory tiering is a systems background concept."],
                "evidence_limits": ["No benchmark numbers compared across papers."],
            },
        }
    )

    assert answer["source_attribution"]["paper_claims"] == [
        "PagedAttention manages KV cache memory."
    ]
    assert answer["source_attribution"]["cross_paper_synthesis"] == [
        "Both are memory-management papers."
    ]


def test_memory_audit_status_controls_targeted_reread():
    weak = normalize_memory_audit(
        {
            "status": "NEED_TARGETED_REREAD",
            "unsupported_claims": [],
            "missing_items": ["missing evaluation results"],
            "reread_requests": [
                {"reason": "find experiments", "page_no": None, "keyword": "evaluation"}
            ],
            "repair_instructions": ["add concrete results"],
            "safe_to_generate_capsule": False,
            "confidence": "medium",
        }
    )
    ok = normalize_memory_audit(
        {
            "status": "PASS_WITH_WEAKNESSES",
            "unsupported_claims": [],
            "missing_items": ["minor detail"],
            "reread_requests": [],
            "repair_instructions": [],
            "safe_to_generate_capsule": True,
            "confidence": "high",
        }
    )

    assert memory_audit_needs_targeted_reread(weak)
    assert not memory_audit_acceptable(weak)
    assert not memory_audit_needs_targeted_reread(ok)
    assert memory_audit_acceptable(ok)


def test_fallback_memory_audit_is_visible_but_usable():
    audit = fallback_memory_audit(reason="provider timeout", phase="memory_critic")

    assert audit["status"] == "PASS_WITH_WEAKNESSES"
    assert audit["safe_to_generate_capsule"] is True
    assert memory_audit_acceptable(audit)
    assert any("memory_critic" in item for item in audit["missing_items"])


def test_exhausted_targeted_reread_becomes_visible_weakness():
    audit = normalize_memory_audit(
        {
            "status": "NEED_TARGETED_REREAD",
            "unsupported_claims": [],
            "missing_items": ["missing evaluation details"],
            "reread_requests": [
                {"reason": "find evaluation", "page_no": None, "keyword": "evaluation"}
            ],
            "repair_instructions": ["add evidence model"],
            "safe_to_generate_capsule": False,
            "confidence": "medium",
        }
    )

    downgraded = downgrade_exhausted_targeted_reread(audit)

    assert downgraded["status"] == "PASS_WITH_WEAKNESSES"
    assert downgraded["safe_to_generate_capsule"] is True
    assert memory_audit_acceptable(downgraded)
    assert any("exhausted" in item for item in downgraded["repair_instructions"])


def test_targeted_reread_selects_requested_pages_and_keywords():
    pages = [
        SimpleNamespace(page_no=1, text="abstract and introduction", captions=[], visual_notes=[]),
        SimpleNamespace(
            page_no=4, text="system design and core mechanism", captions=[], visual_notes=[]
        ),
        SimpleNamespace(
            page_no=8,
            text="evaluation results throughput latency experiment",
            captions=[],
            visual_notes=[],
        ),
    ]
    audit = normalize_memory_audit(
        {
            "status": "NEED_TARGETED_REREAD",
            "unsupported_claims": [],
            "missing_items": ["need evaluation numbers"],
            "reread_requests": [
                {"reason": "inspect method page", "page_no": 4, "keyword": None},
                {"reason": "find evaluation", "page_no": None, "keyword": "throughput experiment"},
            ],
            "repair_instructions": [],
            "safe_to_generate_capsule": False,
            "confidence": "medium",
        }
    )

    selected = select_targeted_reread_pages(pages, audit, already_read={1})

    assert [page.page_no for page in selected][:2] == [4, 8]


def test_paperlens_runtime_searches_and_reads_pages():
    pages = [
        SimpleNamespace(page_no=1, text="abstract", captions=[]),
        SimpleNamespace(
            page_no=5,
            text="KV cache fragmentation appears in evaluation results.",
            captions=[{"text": "Figure 2: memory waste"}],
            figures=[{"id": "fig2"}],
            tables=[],
            visual_notes=[],
        ),
    ]
    runtime = PaperLensRuntime(artifacts=pages)

    search = runtime.search_text("KV cache evaluation")
    pack = runtime.read_pages([5])
    audit_context = runtime.audit_context(
        memory={
            "paper_id": "p_test",
            "core_abstraction": "KV cache fragmentation",
            "pages_read": [1],
            "claims": [
                {
                    "id": "C001",
                    "text": "KV cache fragmentation limits serving density.",
                    "type": "motivation",
                    "provenance": "explicit",
                    "confidence": "medium",
                    "critic_status": "unchecked",
                    "evidence_refs": [],
                }
            ],
        },
        read_artifacts=[pages[0]],
    )

    assert search.results[0]["page_no"] == 5
    assert "KV cache" in pack.results[0]["text"]
    assert 5 in audit_context["candidate_unread_pages"]
    context_pack = audit_context["agent_context_pack"]
    assert context_pack["schema_version"] == "paperlens.context_pack.v1"
    assert context_pack["always_context"]["paper_id"] == "p_test"
    assert context_pack["budget"]["whole_paper_in_context"] is False
    assert context_pack["tool_trace"]


def test_targeted_reread_skips_already_read_requested_pages(monkeypatch):
    monkeypatch.delenv("PAPERLENS_TARGETED_REREAD_MAX_PAGES", raising=False)
    pages = [
        SimpleNamespace(
            page_no=page_no, text=f"page {page_no} evaluation design", captions=[], visual_notes=[]
        )
        for page_no in range(1, 9)
    ]
    audit = normalize_memory_audit(
        {
            "status": "NEED_TARGETED_REREAD",
            "unsupported_claims": [],
            "missing_items": ["need method and evaluation"],
            "reread_requests": [
                {"reason": f"inspect page {page_no}", "page_no": page_no, "keyword": None}
                for page_no in range(2, 8)
            ],
            "repair_instructions": [],
            "safe_to_generate_capsule": False,
            "confidence": "medium",
        }
    )

    selected = select_targeted_reread_pages(pages, audit, already_read={2, 3})

    selected_pages = [page.page_no for page in selected]
    assert selected_pages[:4] == [4, 5, 6, 7]
    assert 2 not in selected_pages
    assert 3 not in selected_pages
    assert len(selected) <= 6


def test_rolling_page_limit_env_can_make_formal_smoke_tiny(monkeypatch):
    monkeypatch.setenv("PAPERLENS_ROLLING_MAX_PAGES", "1")
    pages = [
        SimpleNamespace(page_no=page_no, text=f"page {page_no} abstract evaluation", captions=[])
        for page_no in range(1, 5)
    ]
    skim = SkimCard(paper_id="p_test", problem="problem")
    decision = ClassificationDecision(
        paper_id="p_test", class_label="A", confidence=0.9, false_negative_risk=0.1
    )

    selected = select_rolling_read_pages(pages, skim, decision)

    assert [page.page_no for page in selected] == [1]


def test_standard_read_mode_reads_all_usable_pages_by_default(monkeypatch):
    monkeypatch.delenv("PAPERLENS_ROLLING_MAX_PAGES", raising=False)
    pages = [
        SimpleNamespace(page_no=page_no, text=f"page {page_no} design evaluation", captions=[])
        for page_no in range(1, 20)
    ]
    skim = SkimCard(paper_id="p_test", problem="problem")
    decision = ClassificationDecision(
        paper_id="p_test", class_label="B", confidence=0.9, false_negative_risk=0.1
    )

    selected = select_rolling_read_pages(pages, skim, decision, read_mode="standard")

    assert [page.page_no for page in selected] == list(range(1, 20))


def test_stage07_partial_chunk_failure_preserves_successful_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLENS_ALLOW_LLM_FALLBACK", "1")
    monkeypatch.setenv("PAPERLENS_ROLLING_CHUNK_PAGES", "1")
    output_dir = tmp_path / "out"
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    pipeline = PaperLensWorkflow(
        input_dir=input_dir,
        output_dir=output_dir,
        config=CoreConfig(
            read_mode="standard",
            provider=ProviderConfig(
                kind="openai-compatible",
                base_url="https://example.invalid",
                api_key="test-key",
                model="fake-model",
            )
        ),
        events=EventWriter(
            "run_test",
            output_dir / ".paperlens" / "data" / "events.jsonl",
            output_dir / ".paperlens" / "data" / "errors.jsonl",
        ),
        control=ControlState(),
    )
    pipeline.prepare_output()
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    skim = SkimCard(paper_id="p_test", problem="problem")
    decision = ClassificationDecision(
        paper_id="p_test", class_label="A", confidence=0.9, false_negative_risk=0.1
    )
    fallback = PaperCard(paper_id="p_test", contribution_claims=["fallback claim"])
    pages = [
        SimpleNamespace(page_no=page_no, text=f"page {page_no} design evaluation", captions=[])
        for page_no in range(1, 4)
    ]

    def fake_read_chunk(**kwargs):
        page_no = kwargs["artifacts"][0].page_no
        if page_no == 2:
            raise RuntimeError("provider timeout")
        return {
            "paper_id": paper.paper_id,
            "operations": [
                {"op": "add_read_pages", "payload": {"pages": [page_no]}},
                {"op": "set_core_abstraction", "payload": {"text": "rolling thesis"}},
                {
                    "op": "upsert_evidence",
                    "payload": {
                        "id": f"E{page_no:03d}",
                        "page": page_no,
                        "interpretation": f"claim from page {page_no}",
                    },
                },
                {
                    "op": "upsert_claim",
                    "payload": {
                        "id": f"C{page_no:03d}",
                        "text": f"claim from page {page_no}",
                        "confidence": "medium",
                        "evidence_refs": [f"E{page_no:03d}"],
                    },
                },
            ],
        }

    monkeypatch.setattr(pipeline, "read_rolling_memory_chunk", fake_read_chunk)
    monkeypatch.setattr(
        pipeline, "audit_and_repair_paper_memory", lambda **kwargs: kwargs["memory"]
    )

    result = pipeline.run_rolling_paper_read(
        client=SimpleNamespace(),
        stage="stage_07_normal_read",
        paper=paper,
        skim=skim,
        decision=decision,
        artifacts=pages,
        fallback=fallback,
    )

    memory = read_paper_memory_v3(output_dir, "p_test")
    failures = memory["audit_trail"]["partial_read_failures"]
    questions = memory["open_questions"]
    pages_read = memory["reading_context"]["pages_read"]
    claims_text = json.dumps(memory["claims"], ensure_ascii=False)
    assert result.paper_id == "p_test"
    assert "claim from page 3" in result.contribution_claims
    assert pages_read == [1, 3]
    assert failures[0]["pages"] == [2]
    assert any("p.2" in item for item in questions)
    assert "claim from page 3" in claims_text
    pipeline.db.close()


def test_memory_prompts_include_audit_and_targeted_pages():
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    skim = SkimCard(paper_id="p_test", problem="solves cache pressure")
    decision = ClassificationDecision(
        paper_id="p_test",
        class_label="A",
        confidence=0.8,
        false_negative_risk=0.2,
    )
    memory = {
        "schema_version": "paper_memory.v3",
        "paper_id": "p_test",
        "problem_frame": {"problem": "Core thesis"},
        "claims": [{"id": "C001", "text": "Core thesis", "evidence_refs": []}],
        "evidence": [],
        "audit_trail": {},
    }
    pages = [
        SimpleNamespace(
            page_no=2,
            text="evaluation results",
            captions=[],
            visual_notes=[],
            figures=[],
            tables=[],
            low_confidence_flags=[],
        )
    ]
    audit = normalize_memory_audit(
        {
            "status": "NEED_TARGETED_REREAD",
            "unsupported_claims": [],
            "missing_items": ["missing evaluation"],
            "reread_requests": [
                {"reason": "find evaluation", "page_no": 2, "keyword": "evaluation"}
            ],
            "repair_instructions": ["add evidence model"],
            "safe_to_generate_capsule": False,
            "confidence": "medium",
        }
    )

    critic_prompt = build_memory_critic_prompt(
        paper=paper,
        skim=skim,
        decision=decision,
        memory=memory,
        artifacts=pages,
    )
    repair_prompt = build_memory_repair_prompt(
        paper=paper,
        skim=skim,
        decision=decision,
        memory=memory,
        audit=audit,
        artifacts=pages,
    )

    assert "paper_memory_to_audit" in critic_prompt
    assert "agent_context_pack" in critic_prompt
    assert "memory_audit" in repair_prompt
    assert "targeted_reread_pages" in repair_prompt
    assert "MemoryPatchSet" in repair_prompt
    assert "memory_patch_protocol" in repair_prompt


def test_read_mode_controls_memory_repair_round_budget(monkeypatch):
    monkeypatch.delenv("PAPERLENS_MEMORY_REPAIR_ROUNDS", raising=False)
    assert memory_repair_round_budget("standard") == 2
    monkeypatch.setenv("PAPERLENS_MEMORY_REPAIR_ROUNDS", "1")
    assert memory_repair_round_budget("standard") == 1


def test_zero_repair_budget_disables_targeted_reread_without_repair_call(tmp_path, monkeypatch):
    monkeypatch.setenv("PAPERLENS_MEMORY_REPAIR_ROUNDS", "0")
    output_dir = tmp_path / "out"
    pipeline = PaperLensWorkflow(
        input_dir=tmp_path,
        output_dir=output_dir,
        config=CoreConfig(
            read_mode="standard",
            provider=ProviderConfig(
                kind="openai-compatible",
                base_url="https://example.invalid",
                api_key="test-key",
                model="fake-model",
            ),
        ),
        events=EventWriter(
            "run_test",
            output_dir / ".paperlens" / "data" / "events.jsonl",
            output_dir / ".paperlens" / "data" / "errors.jsonl",
        ),
        control=ControlState(),
    )
    pipeline.prepare_output()
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    skim = SkimCard(paper_id="p_test", problem="problem")
    decision = ClassificationDecision(
        paper_id="p_test", class_label="A", confidence=0.9, false_negative_risk=0.1
    )
    pipeline.memory_store.initialize(
        paper=paper,
        skim=skim,
        decision=decision,
        card=None,
        layout=None,
        source="test_seed",
        prefer_existing=False,
    )
    memory = pipeline.memory_store.apply_patch_set(
        "p_test",
        {
            "paper_id": "p_test",
            "operations": [
                {"op": "add_read_pages", "payload": {"pages": [1]}},
                {"op": "set_problem_frame", "payload": {"problem": "usable memory"}},
            ],
        },
        source="test_patch",
    )
    pages = [
        SimpleNamespace(page_no=1, text="page 1", captions=[], figures=[], tables=[]),
        SimpleNamespace(
            page_no=2, text="evaluation page", captions=[], figures=[], tables=[]
        ),
    ]
    audit = normalize_memory_audit(
        {
            "status": "NEED_TARGETED_REREAD",
            "missing_items": ["missing evaluation detail"],
            "reread_requests": [{"reason": "check evaluation", "page_no": 2}],
            "repair_instructions": ["repair evaluation"],
            "safe_to_generate_capsule": False,
            "confidence": "medium",
        }
    )
    monkeypatch.setattr(pipeline, "audit_paper_memory", lambda **_kwargs: audit)

    def fail_if_called(**_kwargs):
        raise AssertionError("standard mode should not auto repair")

    monkeypatch.setattr(pipeline, "repair_paper_memory_with_targeted_reread", fail_if_called)

    result = pipeline.audit_and_repair_paper_memory(
        client=SimpleNamespace(),
        stage="stage_07_normal_read",
        paper=paper,
        skim=skim,
        decision=decision,
        memory=memory,
        all_artifacts=pages,
        read_artifacts=[pages[0]],
    )

    assert result["audit_trail"]["memory_audit"]["status"] == "PASS_WITH_WEAKNESSES"
    assert pipeline.db.list_review_items()[0].reason.startswith("Targeted reread was disabled")
    pipeline.db.close()


def test_memory_repair_failure_keeps_existing_memory_under_strict_mode(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("PAPERLENS_ALLOW_LLM_FALLBACK", raising=False)
    output_dir = tmp_path / "out"
    pipeline = PaperLensWorkflow(
        input_dir=tmp_path,
        output_dir=output_dir,
        config=CoreConfig(
            read_mode="standard",
            provider=ProviderConfig(
                kind="openai-compatible",
                base_url="https://example.invalid",
                api_key="test-key",
                model="fake-model",
            )
        ),
        events=EventWriter(
            "run_test",
            output_dir / ".paperlens" / "data" / "events.jsonl",
            output_dir / ".paperlens" / "data" / "errors.jsonl",
        ),
        control=ControlState(),
    )
    pipeline.prepare_output()
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    skim = SkimCard(paper_id="p_test", problem="problem")
    decision = ClassificationDecision(
        paper_id="p_test", class_label="A", confidence=0.9, false_negative_risk=0.1
    )
    pipeline.memory_store.initialize(
        paper=paper,
        skim=skim,
        decision=decision,
        card=None,
        layout=None,
        source="test_seed",
        prefer_existing=False,
    )
    memory = pipeline.memory_store.apply_patch_set(
        "p_test",
        {
            "paper_id": "p_test",
            "operations": [
                {"op": "add_read_pages", "payload": {"pages": [1]}},
                {
                    "op": "set_problem_frame",
                    "payload": {"problem": "usable audited memory"},
                },
                {
                    "op": "upsert_evidence",
                    "payload": {"id": "E001", "page": 1, "interpretation": "main claim"},
                },
                {
                    "op": "upsert_claim",
                    "payload": {
                        "id": "C001",
                        "text": "main claim",
                        "confidence": "medium",
                        "evidence_refs": ["E001"],
                    },
                },
            ],
        },
        source="test_patch",
    )
    pages = [
        SimpleNamespace(page_no=1, text="page 1", captions=[]),
        SimpleNamespace(page_no=2, text="evaluation page", captions=[]),
    ]
    audit = normalize_memory_audit(
        {
            "status": "NEED_TARGETED_REREAD",
            "missing_items": ["missing evaluation detail"],
            "reread_requests": [{"reason": "check evaluation", "page_no": 2}],
            "repair_instructions": ["repair evaluation"],
            "safe_to_generate_capsule": False,
            "confidence": "medium",
        }
    )

    monkeypatch.setattr(pipeline, "audit_paper_memory", lambda **_kwargs: audit)

    def fail_repair(**_kwargs):
        raise RuntimeError("Model response was truncated before JSON completed")

    monkeypatch.setattr(pipeline, "repair_paper_memory_with_targeted_reread", fail_repair)

    result = pipeline.audit_and_repair_paper_memory(
        client=SimpleNamespace(),
        stage="stage_07_normal_read",
        paper=paper,
        skim=skim,
        decision=decision,
        memory=memory,
        all_artifacts=pages,
        read_artifacts=[pages[0]],
    )

    assert result["problem_frame"]["problem"] == "usable audited memory"
    audit_result = result["audit_trail"]["memory_audit"]
    assert audit_result["status"] == "PASS_WITH_WEAKNESSES"
    assert "memory_repair did not complete" in audit_result["missing_items"]
    assert pipeline.db.list_review_items()[0].item_type == "MEMORY_REPAIR_FAILED"
    pipeline.db.close()


def test_default_budget_rates_estimate_nonzero_cost():
    snapshot = BudgetManager(BudgetConfig(max_usd=0)).record_usage(
        {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
    )

    assert snapshot.estimated_usd == pytest.approx(5.25)


def test_visual_verification_default_is_parse_issue_only():
    config = CoreConfig()

    assert config.visual_verification_mode == "parse_issues"
    assert config.visual_verification_max_pages == 6


def test_agentic_report_composer_uses_step_cache(tmp_path):
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    decision = ClassificationDecision(
        paper_id="p_test",
        class_label="A",
        confidence=0.9,
        false_negative_risk=0.1,
    )
    calls: list[str] = []

    class FakeClient:
        config = SimpleNamespace(kind="openai-compatible", model="fake-model")

        def invoke_json(self, *, schema_name, **_kwargs):
            calls.append(schema_name)
            if schema_name == "paperlens_report_plan":
                data = {
                    "paper_id": "p_test",
                    "grade": "A",
                    "read_recommendation": "重点关注",
                    "one_line_reason": "这篇论文提出了清晰的系统抽象。",
                    "core_takeaway": "把复杂机制压缩成可执行的系统抽象。",
                    "sections": [
                        {
                            "section_id": "idea",
                            "section_kind": "orientation",
                            "title": "核心抽象",
                            "purpose": "explain idea",
                            "focus_queries": ["core"],
                            "claim_ids": [],
                            "evidence_refs": [],
                            "target_pages": [],
                        }
                    ],
                    "key_visual_pages": [],
                    "uncertainty_note": "",
                }
            elif schema_name == "paperlens_report_section":
                data = {
                    "section_id": "idea",
                    "title": "核心抽象",
                    "paragraphs": ["核心思想是把复杂机制压缩成可执行的系统抽象。"],
                    "used_claim_ids": [],
                    "used_evidence_refs": [],
                    "uncertainty_note": "",
                }
            elif schema_name == "paperlens_report_section_audit":
                data = {
                    "verdict": "PASS",
                    "unsupported_items": [],
                    "missing_items": [],
                    "repair_instructions": [],
                    "safe_usage_note": "",
                }
            else:
                raise AssertionError(schema_name)
            return SimpleNamespace(
                data=data,
                usage={"prompt_tokens": 10, "completion_tokens": 20},
                endpoint="fake",
                request_id=schema_name,
            )

    agent_runs: list[dict[str, object]] = []
    usages: list[dict[str, int]] = []
    kwargs = {
        "client": FakeClient(),
        "data_dir": tmp_path / "debug",
        "stage": "stage_15_export",
        "paper": paper,
        "skim": None,
        "decision": decision,
        "card": None,
        "paper_memory": {"schema_version": "paper_memory.v3", "paper_id": "p_test"},
        "layout": {"pages": []},
        "topic": None,
        "idea": None,
        "output_language": "zh",
        "record_usage": lambda _stage, usage: usages.append(usage),
        "record_agent_run": agent_runs.append,
        "cache_dir": tmp_path / "cache",
    }

    first, first_audit = compose_agentic_paper_report(**kwargs)
    second, second_audit = compose_agentic_paper_report(**kwargs)

    assert calls == [
        "paperlens_report_plan",
        "paperlens_report_section",
        "paperlens_report_section_audit",
    ]
    assert first["explanation_markdown"] == second["explanation_markdown"]
    assert first_audit == second_audit
    assert usages == [
        {"prompt_tokens": 10, "completion_tokens": 20},
        {"prompt_tokens": 10, "completion_tokens": 20},
        {"prompt_tokens": 10, "completion_tokens": 20},
    ]
    assert agent_runs[-1]["status"] == "CACHE_HIT"


def test_report_composer_repairs_only_the_failed_section(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    for relative in [
        "papers",
        ".paperlens/data",
    ]:
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
    )
    decision = ClassificationDecision(
        paper_id="p_test",
        class_label="A",
        confidence=0.9,
        false_negative_risk=0.1,
    )

    class FakeClient:
        config = SimpleNamespace(kind="openai-compatible", model="fake-model")

        def invoke_json(self, *, schema_name, **_kwargs):
            if schema_name == "paperlens_report_plan":
                data = {
                    "paper_id": "p_test",
                    "grade": "A",
                    "read_recommendation": "重点关注",
                    "one_line_reason": "核心理由",
                    "core_takeaway": "核心抽象",
                    "sections": [
                        {
                            "section_id": "idea",
                            "section_kind": "orientation",
                            "title": "核心抽象",
                            "purpose": "explain idea",
                            "focus_queries": [],
                            "claim_ids": [],
                            "evidence_refs": [],
                            "target_pages": [],
                        }
                    ],
                    "key_visual_pages": [],
                    "uncertainty_note": "",
                }
            elif schema_name == "paperlens_report_section":
                section_calls = sum(
                    1
                    for run in agent_runs
                    if "report_section_idea" in str(run.get("agent_run_id"))
                )
                data = {
                    "section_id": "idea",
                    "title": "核心抽象",
                    "paragraphs": [
                        "修复后的分段正文。" if section_calls else "含有过度结论的分段正文。"
                    ],
                    "used_claim_ids": [],
                    "used_evidence_refs": [],
                    "uncertainty_note": "",
                }
            elif schema_name == "paperlens_report_section_audit":
                audit_calls = sum(
                    1
                    for run in agent_runs
                    if "report_section_audit_idea" in str(run.get("agent_run_id"))
                )
                data = (
                    {
                        "verdict": "PASS_WITH_WEAKNESSES",
                        "unsupported_items": [],
                        "missing_items": [],
                        "repair_instructions": [],
                        "safe_usage_note": "Usable with evidence boundary.",
                    }
                    if audit_calls
                    else {
                        "verdict": "REPAIR",
                        "unsupported_items": ["overclaim"],
                        "missing_items": [],
                        "repair_instructions": ["remove overclaim"],
                        "safe_usage_note": "Needs section repair.",
                    }
                )
            else:
                raise AssertionError(schema_name)
            return SimpleNamespace(
                data=data,
                usage={"prompt_tokens": 10, "completion_tokens": 20},
                endpoint="fake",
                request_id=schema_name,
            )

    agent_runs: list[dict[str, object]] = []
    written = write_final_report_bundle(
        output_dir=output_dir,
        data_dir=data_dir,
        evidence_dir=output_dir / ".paperlens",
        client=FakeClient(),
        record_usage=lambda _stage, _usage: None,
        record_agent_run=agent_runs.append,
        stage="stage_15_export",
        papers=[paper],
        skim_cards=[],
        decisions=[decision],
        paper_cards=[],
        review_items=[],
        budget={},
        config={"offline_debug": False},
        topic=None,
        idea=None,
        cache_dir=output_dir / ".paperlens" / "cache",
    )

    assert any(path.name == "PaperLens.md" for path in written)
    report = (output_dir / "papers" / "p_test_test_paper.md").read_text(encoding="utf-8")
    assert "修复后的分段正文" in report
    assert not any("final_paper_report_repair" in str(run) for run in agent_runs)


def test_export_reuses_existing_paper_report_on_resume(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    for relative in [
        "papers",
        ".paperlens/data",
    ]:
        (output_dir / relative).mkdir(parents=True, exist_ok=True)
    existing_report = output_dir / "papers" / "p_test_test_paper.md"
    existing_report.write_text("# Test Paper\n\n已经完成的报告。", encoding="utf-8")
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
    )
    decision = ClassificationDecision(
        paper_id="p_test",
        class_label="A",
        confidence=0.9,
        false_negative_risk=0.1,
    )

    class FakeClient:
        config = SimpleNamespace(kind="openai-compatible", model="fake-model")

        def invoke_json(self, **_kwargs):
            raise AssertionError("existing reports must not be regenerated during resume")

    agent_runs: list[dict[str, object]] = []
    written = write_final_report_bundle(
        output_dir=output_dir,
        data_dir=data_dir,
        evidence_dir=output_dir / ".paperlens",
        client=FakeClient(),
        record_usage=lambda _stage, _usage: None,
        record_agent_run=agent_runs.append,
        stage="stage_15_export",
        papers=[paper],
        skim_cards=[],
        decisions=[decision],
        paper_cards=[],
        review_items=[],
        budget={},
        config={"offline_debug": False},
        topic=None,
        idea=None,
        cache_dir=output_dir / ".paperlens" / "cache",
    )

    assert existing_report in written
    assert (output_dir / "PaperLens.md").exists()
    assert existing_report.read_text(encoding="utf-8") == "# Test Paper\n\n已经完成的报告。"
    assert agent_runs == []


def test_paper_question_answer_uses_cache(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    (output_dir / "papers").mkdir(parents=True)
    (output_dir / "papers" / "p_test_Test.md").write_text(
        "# Test Paper\n\n报告正文", encoding="utf-8"
    )
    calls = {"count": 0}

    class FakeClient:
        def __init__(self, provider):
            self.config = provider

        def invoke_json(self, **_kwargs):
            calls["count"] += 1
            return SimpleNamespace(
                data={
                    "answer_markdown": "这篇论文的核心是一个可复用的系统抽象。",
                    "cited_pages": [1],
                    "confidence": "high",
                },
                usage={"prompt_tokens": 11, "completion_tokens": 22},
                endpoint="fake",
                request_id="qa1",
            )

    monkeypatch.setattr("paperlens_core.qa.JsonLlmClient", FakeClient)
    config = CoreConfig(
        provider=ProviderConfig(
            kind="openai-compatible",
            base_url="https://example.invalid",
            api_key="test-key",
            model="fake-model",
        )
    )

    first = answer_question(
        output_dir=output_dir, config=config, paper_id="p_test", question="核心 idea 是什么？"
    )
    second = answer_question(
        output_dir=output_dir, config=config, paper_id="p_test", question="核心 idea 是什么？"
    )

    assert calls["count"] == 1
    assert first["cache_hit"] is False
    assert second["cache_hit"] is True
    assert second["usage"] == {}


def test_resume_stage_resolution_supports_late_export():
    assert resolve_workflow_stages(from_stage="stage_15_export") == [
        "stage_15_export",
        "stage_17_manifest",
    ]
    with pytest.raises(ValueError):
        resolve_workflow_stages(only_stage="stage_15_export_reports")
    with pytest.raises(ValueError):
        resolve_workflow_stages(from_stage="stage_15_export", only_stage="stage_07_normal_read")


def test_db_can_reload_completed_pipeline_state(tmp_path):
    db = ArtifactDb(tmp_path / "state.sqlite")
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    skim = SkimCard(paper_id="p_test", problem="problem")
    decision = ClassificationDecision(
        paper_id="p_test",
        class_label="A",
        confidence=0.9,
        false_negative_risk=0.1,
    )
    db.upsert_paper(paper)
    db.upsert_skim(skim)
    db.upsert_classification(decision)

    assert db.list_papers()[0].paper_id == "p_test"
    assert db.list_skim_cards()[0].problem == "problem"
    assert db.list_classifications()[0].class_label == "A"
    db.close()


def test_stage07_persists_each_paper_card_for_resume(tmp_path):
    output_dir = tmp_path / "out"
    pipeline = PaperLensWorkflow(
        input_dir=tmp_path / "in",
        output_dir=output_dir,
        config=CoreConfig(offline_debug=True),
        events=EventWriter(
            "run_test",
            output_dir / ".paperlens" / "data" / "events.jsonl",
            output_dir / ".paperlens" / "data" / "errors.jsonl",
        ),
        control=ControlState(),
    )
    pipeline.prepare_output()
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    card = PaperCard(paper_id="p_test", contribution_claims=["claim"])

    pipeline.persist_paper_card("stage_07_normal_read", paper, card)

    assert pipeline.db.list_paper_cards()[0].paper_id == "p_test"
    assert not (output_dir / ".paperlens" / "data" / "cards").exists()
    pipeline.db.close()


def test_pipeline_failure_marks_run_json_failed(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    pipeline = PaperLensWorkflow(
        input_dir=input_dir,
        output_dir=output_dir,
        config=CoreConfig(offline_debug=True),
        events=EventWriter(
            "run_test",
            output_dir / ".paperlens" / "data" / "events.jsonl",
            output_dir / ".paperlens" / "data" / "errors.jsonl",
        ),
        control=ControlState(),
    )

    def fail_stage():
        raise RuntimeError("planned failure")

    monkeypatch.setattr(pipeline, "stage_00_ingest", fail_stage)

    with pytest.raises(RuntimeError, match="planned failure"):
        pipeline.run(only_stage="stage_00_ingest")

    run_json = json.loads(
        (output_dir / ".paperlens" / "data" / "run.json").read_text(encoding="utf-8")
    )
    assert run_json["status"] == "failed"
    assert "planned failure" in run_json["error"]


def test_only_stage_intermediate_run_marks_partial_completed(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    pipeline = PaperLensWorkflow(
        input_dir=input_dir,
        output_dir=output_dir,
        config=CoreConfig(offline_debug=True),
        events=EventWriter(
            "run_test",
            output_dir / ".paperlens" / "data" / "events.jsonl",
            output_dir / ".paperlens" / "data" / "errors.jsonl",
        ),
        control=ControlState(),
    )

    monkeypatch.setattr(pipeline, "stage_00_ingest", lambda: None)

    manifest = pipeline.run(only_stage="stage_00_ingest")

    run_json = json.loads(
        (output_dir / ".paperlens" / "data" / "run.json").read_text(encoding="utf-8")
    )
    assert manifest["partial_run"] is True
    assert run_json["status"] == "partial_completed"
    assert not (output_dir / ".paperlens" / "data" / "reports" / "output_validation.json").exists()


def test_stage03_persists_skim_classification_for_resume(tmp_path):
    output_dir = tmp_path / "out"
    pipeline = PaperLensWorkflow(
        input_dir=tmp_path / "in",
        output_dir=output_dir,
        config=CoreConfig(offline_debug=True),
        events=EventWriter(
            "run_test",
            output_dir / ".paperlens" / "data" / "events.jsonl",
            output_dir / ".paperlens" / "data" / "errors.jsonl",
        ),
        control=ControlState(),
    )
    pipeline.prepare_output()
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    skim = SkimCard(paper_id="p_test", problem="stable progress")
    decision = ClassificationDecision(
        paper_id="p_test",
        class_label="A",
        confidence=0.9,
        false_negative_risk=0.1,
    )

    pipeline.persist_skim_classification("stage_03_skim", paper, skim, decision)

    assert pipeline.db.list_skim_cards()[0].problem == "stable progress"
    assert pipeline.db.list_classifications()[0].class_label == "A"
    assert not (output_dir / ".paperlens" / "data" / "cards").exists()
    pipeline.db.close()


def test_resume_from_stage03_loads_partial_skim_state(tmp_path):
    output_dir = tmp_path / "out"
    pipeline = PaperLensWorkflow(
        input_dir=tmp_path / "in",
        output_dir=output_dir,
        config=CoreConfig(offline_debug=True),
        events=EventWriter(
            "run_test",
            output_dir / ".paperlens" / "data" / "events.jsonl",
            output_dir / ".paperlens" / "data" / "errors.jsonl",
        ),
        control=ControlState(),
    )
    pipeline.prepare_output()
    paper = PaperRecord(
        paper_id="p_test", file_path="paper.pdf", file_hash="hash", canonical_title="Test Paper"
    )
    pipeline.db.upsert_paper(paper)
    pipeline.db.upsert_skim(SkimCard(paper_id="p_test", problem="existing skim"))
    pipeline.db.upsert_classification(
        ClassificationDecision(
            paper_id="p_test",
            class_label="B",
            confidence=0.8,
            false_negative_risk=0.2,
        )
    )

    pipeline.load_completed_state_for_stage("stage_03_skim")

    assert pipeline.papers[0].paper_id == "p_test"
    assert pipeline.skim_cards[0].problem == "existing skim"
    assert pipeline.classifications[0].class_label == "B"
    pipeline.db.close()


def test_event_writer_survives_closed_stdout(tmp_path, monkeypatch):
    writer = EventWriter("run_test", tmp_path / "events.jsonl", tmp_path / "errors.jsonl")

    def broken_print(*_args, **_kwargs):
        raise OSError("closed pipe")

    monkeypatch.setattr(builtins, "print", broken_print)

    writer.emit("progress", message="still write file")

    assert "still write file" in (tmp_path / "events.jsonl").read_text(encoding="utf-8")


def test_capsule_quality_flags_generic_template_output():
    text = """
## 结论
这篇论文主要介绍了一种方法，具有一定参考价值。
## 核心思想
本文提出了一种方法。
## 证据和实验
未来工作可以进一步研究。
"""

    result = evaluate_capsule_quality(text, expected_terms=["consensus", "timeouts", "quorum"])

    assert result["score"] < 6.0
    assert "generic_or_fallback_language" in result["issues"]
    assert "old_template_shape" in result["issues"]


def test_capsule_quality_flags_reader_hostile_output():
    text = """
# Efficient Memory Management for Large Language Model Serving with PagedAttention

你给到的片段没有完整评测表，所以这里不写死具体倍数。

需要修正或确认：
- Add the evaluation setup.
- Include the compaction-vs-paging motivation.

<img src="../.paperlens/pages/p_test/page_0001.png">
"""

    result = evaluate_capsule_quality(text, expected_terms=["KV cache", "paging"])

    assert result["score"] < 7.0
    assert "implementation_context_leak" in result["issues"]
    assert "visible_repair_or_audit_todo" in result["issues"]
    assert "full_page_visual_embed" in result["issues"]


def test_capsule_quality_flags_wall_of_text_and_bullet_dump():
    wall = " ".join(["PagedAttention uses KV cache paging to improve LLM serving throughput."] * 80)
    bullets = "\n".join(f"- item {index} with evaluation and limitation" for index in range(12))

    wall_result = evaluate_capsule_quality(
        wall,
        expected_terms=["KV cache", "paging", "throughput"],
        min_chars=120,
    )
    bullet_result = evaluate_capsule_quality(
        "Intro paragraph with core idea, mechanism, evidence, value, and limitation.\n\n" + bullets,
        expected_terms=["evaluation", "limitation"],
        min_chars=120,
    )

    assert "reader_hostile_wall_of_text" in wall_result["issues"]
    assert "mostly_bullet_points" in bullet_result["issues"]


def test_capsule_quality_accepts_dense_freeform_explanation():
    text = (
        "QuePaxa 的核心价值不是又写了一个 Paxos 变体，而是把 consensus 里最容易被工程实现误用的 "
        "timeouts 从正确性路径里拿出去：节点不再把一次超时当成 leader 失效的事实，而是围绕 quorum "
        "和 clock uncertainty 组织协议推进。它的机制重点是让正常路径保持接近传统 Paxos 的成本，"
        "但在网络抖动和尾延迟出现时减少错误换主带来的吞吐崩塌。作者用 evaluation 对比不同故障和 "
        "latency workload，说明这个抽象在高抖动环境里更稳。真正有启发的是，它把“超时只是本地猜测”"
        "这个概念提升成系统设计约束；限制是收益依赖部署里的时钟和网络假设，不能自动证明所有共识系统都该这样改。"
    )

    result = evaluate_capsule_quality(
        text,
        expected_terms=["consensus", "timeouts", "quorum", "Paxos", "clock"],
        min_chars=120,
    )

    assert result["score"] >= 8.0
    assert len(result["matched_terms"]) >= 4


def test_quality_benchmark_tracks_ten_sosp_papers():
    benchmark = json.loads(
        Path("tests/quality_benchmark/sosp_systems.json").read_text(encoding="utf-8")
    )

    assert benchmark["min_score"] == 8.0
    assert len(benchmark["papers"]) == 10
    assert {paper["title_contains"] for paper in benchmark["papers"]} >= {"Grove", "QuePaxa", "Sia"}
    assert all(paper.get("qa_checks") for paper in benchmark["papers"])
    assert all(paper.get("forbidden_terms") for paper in benchmark["papers"])


def test_quality_benchmark_qa_check_scores_report_and_memory():
    from scripts.score_quality_benchmark import score_qa_check

    result = score_qa_check(
        {
            "question": "PagedAttention 的核心抽象是什么？",
            "expected_terms": ["KV cache", "paging", "block table"],
            "min_hits": 3,
        },
        text="PagedAttention applies paging to the KV cache.",
        record={"memory": {"mechanism_steps": ["A block table maps logical blocks."]}},
    )

    assert result["status"] == "PASS"
