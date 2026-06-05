from __future__ import annotations

import json
from typing import Any

from paperlens_core.memory_v3 import list_payload
from paperlens_core.report.composer_context import (
    compact_decision_for_report,
    compact_paper_card_for_report,
    compact_skim_for_report,
)
from paperlens_core.report.memory_context import (
    compact_paper_memory_for_report,
    report_focus_pages,
    report_focus_queries,
)
from paperlens_core.report.text import compact_string_list
from paperlens_core.runtime import PaperLensRuntime, context_pack_prompt
from paperlens_core.schemas import ClassificationDecision, PaperCard, PaperRecord, SkimCard

REPORT_PLAN_SYSTEM_PROMPT = """
You are the PaperLens ReportPlanner skill.
Plan a clear knowledge capsule from PaperMemoryV3.
Choose the reading order that best explains this paper. Use tools if you need more grounding.
The report is a derived view; do not invent facts outside memory/evidence.
Plan for a Theseus-grade capsule: complete, sectioned, and detailed enough to teach the paper.
For normal A/B/C papers, cover orientation/background, core mechanism, implementation or
algorithm path, evaluation evidence, value/tradeoffs, and limitations/boundaries. Do not merge
evaluation, value, and limitations into one thin section when the memory has enough material.
Return final_json matching the ReportPlan schema.
""".strip()


REPORT_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "paper_id",
        "grade",
        "read_recommendation",
        "one_line_reason",
        "core_takeaway",
        "sections",
    ],
    "properties": {
        "paper_id": {"type": "string"},
        "grade": {"type": "string", "enum": ["A", "B", "C", "HOLD"]},
        "read_recommendation": {
            "type": "string",
            "enum": ["重点关注", "标准读", "低优先级", "需确认"],
        },
        "one_line_reason": {"type": "string"},
        "core_takeaway": {"type": "string"},
        "sections": {
            "type": "array",
            "minItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "section_id",
                    "title",
                    "purpose",
                    "focus_queries",
                    "claim_ids",
                    "evidence_refs",
                    "target_pages",
                ],
                "properties": {
                    "section_id": {"type": "string"},
                    "section_kind": {
                        "type": "string",
                        "enum": [
                            "orientation",
                            "background",
                            "mechanism",
                            "evaluation",
                            "value",
                            "limits",
                            "other",
                        ],
                    },
                    "title": {"type": "string"},
                    "purpose": {"type": "string"},
                    "focus_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "claim_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "target_pages": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "detail_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "avoid": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "key_visual_pages": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["page_no", "reason"],
                "properties": {
                    "page_no": {"type": "integer"},
                    "reason": {"type": "string"},
                },
            },
        },
        "uncertainty_note": {"type": "string"},
    },
}


REPORT_SECTION_SYSTEM_PROMPT = """
You are the PaperLens ReportComposer skill.
Write the requested report section as connected prose.
Use PaperMemory and tools for grounding. Explain mechanisms and background when useful.
Keep paper claims, interpretation, background knowledge, and evidence limits distinguishable.
Write at Theseus-grade depth: a section should explain why the idea exists, how the mechanism
works, what evidence supports it, and what boundary limits it when those are relevant to the
planned section. Prefer 2-4 compact but substantive paragraphs over a one-paragraph summary.
Return final_json matching the ReportSection schema.
""".strip()


REPORT_SECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "section_id",
        "title",
        "paragraphs",
        "used_claim_ids",
        "used_evidence_refs",
        "uncertainty_note",
    ],
    "properties": {
        "section_id": {"type": "string"},
        "title": {"type": "string"},
        "paragraphs": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
        },
        "markdown": {"type": "string"},
        "used_claim_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "used_evidence_refs": {
            "type": "array",
            "items": {"type": "string"},
        },
        "uncertainty_note": {"type": "string"},
    },
}


REPORT_SECTION_AUDITOR_SYSTEM_PROMPT = """
You are the PaperLens SectionAuditor hook.
Audit one generated section against PaperMemory and paper evidence.
Use tools when a claim needs checking. Prefer explicit evidence boundaries over brittle certainty.
Mark REPAIR when a section is factually unsupported, overclaims, or is too shallow to satisfy
its planned purpose. Missing reader-critical mechanism, evaluation, or limitation context is a
real quality defect, not just a style preference.
Return final_json matching the ReportSectionAudit schema.
""".strip()


REPORT_SECTION_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "unsupported_items",
        "missing_items",
        "repair_instructions",
        "safe_usage_note",
    ],
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "PASS_WITH_WEAKNESSES", "REPAIR"]},
        "unsupported_items": {"type": "array", "items": {"type": "string"}},
        "missing_items": {"type": "array", "items": {"type": "string"}},
        "repair_instructions": {"type": "array", "items": {"type": "string"}},
        "safe_usage_note": {"type": "string"},
    },
}


def build_report_plan_prompt(
    *,
    paper: PaperRecord,
    skim: SkimCard | None,
    decision: ClassificationDecision | None,
    card: PaperCard | None,
    paper_memory: dict[str, Any],
    layout: dict[str, Any],
    topic: str | None,
    idea: str | None,
    output_language: str,
    read_mode: str,
) -> str:
    pages = list_payload(layout.get("pages"))
    runtime = PaperLensRuntime(artifacts=pages)
    memory_for_report = compact_paper_memory_for_report(paper_memory)
    focus_queries = report_focus_queries(paper_memory, paper=paper, skim=skim, card=card)
    context = runtime.build_context_pack(
        stage="report_plan",
        objective=(
            "Plan a streamed PaperLens report. Use PaperMemory as durable state and local "
            "paper-tool observations only to choose section focus and grounding."
        ),
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        classification=decision.class_label if decision else None,
        memory=paper_memory,
        focus_queries=focus_queries,
        focus_pages=report_focus_pages(paper_memory, skim=skim, card=card),
        output_contract={
            "type": "ReportPlan",
            "rule": "Plan section-level generation. Do not write the report in this step.",
        },
        search_limit=4,
        page_text_limit=900,
    ).as_dict()
    return "\n\n".join(
        [
            f"paper_id: {paper.paper_id}",
            f"title: {paper.canonical_title or 'unknown'}",
            f"output_language: {output_language}",
            f"read_mode: {read_mode}",
            "user_topic: " + (topic or "not provided"),
            "user_idea: " + (idea or "not provided"),
            "skim_card:",
            json.dumps(compact_skim_for_report(skim), ensure_ascii=False),
            "classification:",
            json.dumps(compact_decision_for_report(decision), ensure_ascii=False),
            "paper_card:",
            json.dumps(compact_paper_card_for_report(card), ensure_ascii=False),
            "paper_memory:",
            json.dumps(memory_for_report, ensure_ascii=False),
            "agent_context_pack:",
            context_pack_prompt(context),
            (
                "Task: create a report plan for a Standard PaperLens capsule. The plan should let "
                "later section calls explain the paper clearly without any single call writing the "
                "whole report."
            ),
            (
                "Completeness contract: plan 5-7 focused sections for A/B/C papers when evidence "
                "exists. Keep mechanism, implementation path, evaluation, value/tradeoffs, and "
                "limitations as separate sections unless the memory is truly too sparse. Each "
                "section needs concrete focus queries, claim ids, evidence refs, or target pages "
                "when available."
            ),
        ]
    )


def build_report_section_prompt(
    *,
    paper: PaperRecord,
    paper_memory: dict[str, Any],
    layout: dict[str, Any],
    plan: dict[str, Any],
    section_plan: dict[str, Any],
    previous_summaries: list[dict[str, str]],
    output_language: str,
    read_mode: str,
    section_audit: dict[str, Any] | None = None,
) -> str:
    pages = list_payload(layout.get("pages"))
    runtime = PaperLensRuntime(artifacts=pages)
    context = runtime.build_context_pack(
        stage="report_section",
        objective=(
            "Write one report section from PaperMemory and focused paper-tool observations. "
            "Do not write other sections."
        ),
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        classification=str(plan.get("grade") or ""),
        memory=paper_memory,
        focus_queries=list_payload(section_plan.get("focus_queries")),
        focus_pages=list_payload(section_plan.get("target_pages")),
        output_contract={
            "type": "ReportSection",
            "rule": "Return exactly one section body with used claim/evidence ids.",
        },
        search_limit=4,
        page_text_limit=2200 if report_section_is_mechanism(section_plan) else 1400,
    ).as_dict()
    detail_contract = report_section_detail_contract(section_plan)
    parts = [
        f"paper_id: {paper.paper_id}",
        f"title: {paper.canonical_title or 'unknown'}",
        f"output_language: {output_language}",
        f"read_mode: {read_mode}",
        "paper_memory:",
        json.dumps(compact_paper_memory_for_report(paper_memory), ensure_ascii=False),
        "report_plan:",
        json.dumps(compact_report_plan(plan), ensure_ascii=False),
        "section_to_write:",
        json.dumps(section_plan, ensure_ascii=False),
        "section_detail_contract:",
        detail_contract,
        "previous_section_summaries:",
        json.dumps(previous_summaries[-4:], ensure_ascii=False),
        "agent_context_pack:",
        context_pack_prompt(context),
    ]
    if section_audit:
        parts.extend(
            [
                "previous_section_audit:",
                json.dumps(section_audit, ensure_ascii=False),
                (
                    "Task: rewrite only this section to address the audit. Remove unsupported "
                    "claims; add missing context only when memory/evidence supports it."
                ),
            ]
        )
    else:
        parts.append(
            "Task: write only this planned section. Do not include the heading. Produce "
            "2-4 substantive paragraphs when the memory contains enough material; do not return "
            "a thin abstract-style summary."
        )
    return "\n\n".join(parts)


def build_report_section_audit_prompt(
    *,
    paper: PaperRecord,
    paper_memory: dict[str, Any],
    layout: dict[str, Any],
    plan: dict[str, Any],
    section_plan: dict[str, Any],
    section: dict[str, Any],
    output_language: str,
    read_mode: str,
) -> str:
    pages = list_payload(layout.get("pages"))
    runtime = PaperLensRuntime(artifacts=pages)
    context = runtime.build_context_pack(
        stage="report_section_audit",
        objective=(
            "Audit one generated report section against durable PaperMemory and focused local "
            "paper observations."
        ),
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        classification=str(plan.get("grade") or ""),
        memory=paper_memory,
        focus_queries=list_payload(section_plan.get("focus_queries")),
        focus_pages=list_payload(section_plan.get("target_pages")),
        output_contract={
            "type": "ReportSectionAudit",
            "rule": "Return REPAIR only for issues that matter for factuality or reader usefulness.",
        },
        search_limit=4,
        page_text_limit=1200,
    ).as_dict()
    return "\n\n".join(
        [
            f"paper_id: {paper.paper_id}",
            f"title: {paper.canonical_title or 'unknown'}",
            f"output_language: {output_language}",
            f"read_mode: {read_mode}",
            "paper_memory:",
            json.dumps(compact_paper_memory_for_report(paper_memory), ensure_ascii=False),
            "report_plan:",
            json.dumps(compact_report_plan(plan), ensure_ascii=False),
            "section_plan:",
            json.dumps(section_plan, ensure_ascii=False),
            "section_detail_contract:",
            report_section_detail_contract(section_plan),
            "generated_section:",
            json.dumps(section, ensure_ascii=False),
            "agent_context_pack:",
            context_pack_prompt(context),
            (
                "Task: audit this section. Focus on unsupported facts, overclaims, missing "
                "reader-critical context, and whether used_claim_ids/used_evidence_refs match the prose."
            ),
            (
                "Depth contract: if the section does not answer its section_detail_contract or is "
                "too shallow to teach the planned topic, return REPAIR with concrete repair instructions."
            ),
        ]
    )


def infer_report_section_kind(*, section_id: str, title: str, purpose: Any) -> str:
    text = " ".join([section_id, title, str(purpose or "")]).lower()
    if any(
        token in text
        for token in [
            "mechanism",
            "algorithm",
            "design",
            "architecture",
            "system",
            "机制",
            "算法",
            "架构",
            "系统",
            "如何工作",
        ]
    ):
        return "mechanism"
    if any(
        token in text for token in ["evaluation", "result", "benchmark", "实验", "评估", "性能"]
    ):
        return "evaluation"
    if any(token in text for token in ["limit", "scope", "boundary", "局限", "边界", "适用"]):
        return "limits"
    if any(
        token in text for token in ["background", "problem", "motivation", "背景", "问题", "动机"]
    ):
        return "background"
    if any(token in text for token in ["value", "transfer", "价值", "启发"]):
        return "value"
    return "other"


def report_section_is_mechanism(section_plan: dict[str, Any]) -> bool:
    if str(section_plan.get("section_kind") or "") == "mechanism":
        return True
    inferred = infer_report_section_kind(
        section_id=str(section_plan.get("section_id") or ""),
        title=str(section_plan.get("title") or ""),
        purpose=section_plan.get("purpose"),
    )
    return inferred == "mechanism"


def report_section_detail_contract(section_plan: dict[str, Any]) -> str:
    questions = compact_string_list(section_plan.get("detail_questions"), limit=8, max_chars=180)
    if report_section_is_mechanism(section_plan):
        defaults = [
            "What state or bottleneck exists before the paper's abstraction is introduced?",
            "What is the new abstraction, and what exactly does it re-map or decouple?",
            "Which data structures, tables, schedulers, allocators, or runtime components make it work?",
            "Walk through one request/object lifecycle step by step.",
            "Explain why each step changes memory, compute, latency, or coordination behavior.",
            "Name the main tradeoffs, overheads, and cases where the mechanism becomes less useful.",
        ]
        merged = _merge_string_lists(questions, defaults, limit=10)
        return (
            "Mechanism section contract: answer these questions in connected prose, using only "
            "PaperMemory/evidence and local observations. Do not turn this into a bullet checklist: "
            + json.dumps(merged, ensure_ascii=False)
        )
    section_kind = str(section_plan.get("section_kind") or "")
    if section_kind in {"orientation", "background"}:
        defaults = [
            "Name the paper's concrete problem or bottleneck, not just the broad area.",
            "Explain why prior approaches are insufficient in this paper's framing.",
            "State the paper's core abstraction and the main misunderstanding to avoid.",
            "Keep background concepts separate from claims the paper actually proves.",
        ]
        merged = _merge_string_lists(questions, defaults, limit=8)
        return (
            "Orientation section contract: answer these questions in connected prose: "
            + json.dumps(merged, ensure_ascii=False)
        )
    if section_kind == "evaluation":
        defaults = [
            "Describe datasets/workloads, metrics, baselines, and hardware or setting when available.",
            "Report the headline numbers or qualitative findings that directly support the claim.",
            "Explain which result supports which mechanism or value claim.",
            "State what the evaluation does not prove.",
        ]
        merged = _merge_string_lists(questions, defaults, limit=8)
        return (
            "Evaluation section contract: answer these questions in connected prose: "
            + json.dumps(merged, ensure_ascii=False)
        )
    if section_kind == "value":
        defaults = [
            "Explain the transferable lesson or product/research value.",
            "Name the scenario where the idea is strongest.",
            "Name the tradeoff the paper chooses and what it gives up.",
            "Avoid generic praise; tie value to evidence and mechanism.",
        ]
        merged = _merge_string_lists(questions, defaults, limit=8)
        return "Value section contract: answer these questions in connected prose: " + json.dumps(
            merged, ensure_ascii=False
        )
    if section_kind == "limits":
        defaults = [
            "List the paper's actual evaluation scope and assumptions.",
            "Identify exact numbers, baselines, or implementation details that need source checking.",
            "Explain deployment, reproducibility, scaling, or generalization risks when relevant.",
            "End with open questions that would affect whether a reader should trust or use the result.",
        ]
        merged = _merge_string_lists(questions, defaults, limit=8)
        return "Limits section contract: answer these questions in connected prose: " + json.dumps(
            merged, ensure_ascii=False
        )
    if questions:
        return "Section-specific questions to answer: " + json.dumps(questions, ensure_ascii=False)
    return "No additional section-specific detail contract."


def compact_report_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "grade": plan.get("grade"),
        "one_line_reason": plan.get("one_line_reason"),
        "core_takeaway": plan.get("core_takeaway"),
        "sections": [
            {
                "section_id": section.get("section_id"),
                "section_kind": section.get("section_kind"),
                "title": section.get("title"),
                "purpose": section.get("purpose"),
                "claim_ids": section.get("claim_ids"),
                "evidence_refs": section.get("evidence_refs"),
                "detail_questions": section.get("detail_questions"),
            }
            for section in list_payload(plan.get("sections"))
            if isinstance(section, dict)
        ],
    }


def _merge_string_lists(left: list[str], right: list[str], *, limit: int) -> list[str]:
    merged = []
    for item in left + right:
        cleaned = item.strip()
        if cleaned and cleaned not in merged:
            merged.append(cleaned)
        if len(merged) >= limit:
            break
    return merged
