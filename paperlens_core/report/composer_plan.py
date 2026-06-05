from __future__ import annotations

import re
from typing import Any

from paperlens_core.memory_v3 import dict_value, list_payload, safe_int
from paperlens_core.report.composer_output import normalize_key_visual_pages
from paperlens_core.report.composer_prompts import infer_report_section_kind
from paperlens_core.report.memory_context import core_memory_view_dict
from paperlens_core.report.text import (
    clean_model_inline_text,
    clean_model_markdown,
    compact_string_list,
    recommendation_for_grade,
)
from paperlens_core.schemas import ClassificationDecision, PaperRecord


def normalize_report_plan(
    data: dict[str, Any],
    *,
    paper: PaperRecord,
    decision: ClassificationDecision | None,
    paper_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    grade = str(data.get("grade") or (decision.class_label if decision else "HOLD")).upper()
    if grade not in {"A", "B", "C", "HOLD"}:
        grade = "HOLD"
    recommendation = str(data.get("read_recommendation") or "").strip()
    if recommendation not in {"重点关注", "标准读", "低优先级", "需确认"}:
        recommendation = recommendation_for_grade(grade)
    sections = [
        normalize_report_section_plan(item)
        for item in list_payload(data.get("sections"))
        if isinstance(item, dict)
    ]
    if not sections:
        sections = default_report_sections()
    sections = ensure_report_plan_coverage(
        sections, grade=grade, paper_memory=dict_value(paper_memory)
    )
    return {
        "paper_id": str(data.get("paper_id") or paper.paper_id),
        "grade": grade,
        "read_recommendation": recommendation,
        "one_line_reason": clean_model_inline_text(data.get("one_line_reason")),
        "core_takeaway": clean_model_markdown(data.get("core_takeaway")),
        "sections": sections[:7],
        "key_visual_pages": normalize_key_visual_pages(data.get("key_visual_pages")),
        "uncertainty_note": clean_model_markdown(data.get("uncertainty_note")),
    }


def ensure_report_plan_coverage(
    sections: list[dict[str, Any]], *, grade: str, paper_memory: dict[str, Any]
) -> list[dict[str, Any]]:
    """Supplement thin model plans with a complete PaperLens capsule profile."""
    normalized = [normalize_report_section_plan(section) for section in sections]
    if len(normalized) >= 7:
        return order_report_sections(normalized)[:7]
    desired = desired_report_section_templates(grade=grade, paper_memory=paper_memory)
    for template in desired:
        if len(normalized) >= 7:
            break
        if report_plan_template_is_covered(template, normalized):
            continue
        normalized.append(build_supplemental_report_section(template, paper_memory))
    minimum = minimum_report_sections_for_grade(grade)
    if len(normalized) < minimum:
        for template in default_report_section_templates():
            if len(normalized) >= minimum or len(normalized) >= 7:
                break
            if report_plan_template_is_covered(template, normalized):
                continue
            normalized.append(build_supplemental_report_section(template, paper_memory))
    return order_report_sections(normalized)[:7]


def order_report_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = list(enumerate(sections))
    indexed.sort(key=lambda item: (report_section_order_index(item[1]), item[0]))
    return [section for _index, section in indexed]


def report_section_order_index(section: dict[str, Any]) -> int:
    kind = str(section.get("section_kind") or "")
    text = " ".join(
        [
            str(section.get("section_id") or ""),
            str(section.get("title") or ""),
            str(section.get("purpose") or ""),
        ]
    ).lower()
    if kind in {"orientation", "background"}:
        return 0
    if kind == "mechanism" and any(
        token in text
        for token in [
            "implementation",
            "runtime",
            "training",
            "inference",
            "实现",
            "细节",
            "路径",
            "训练",
            "推理",
        ]
    ):
        return 2
    if kind == "mechanism":
        return 1
    if kind == "evaluation":
        return 3
    if kind == "value":
        return 4
    if kind == "limits":
        return 5
    return 6


def minimum_report_sections_for_grade(grade: str) -> int:
    if grade in {"A", "B", "C"}:
        return 5
    return 4


def desired_report_section_templates(
    *, grade: str, paper_memory: dict[str, Any]
) -> list[dict[str, Any]]:
    templates = default_report_section_templates()
    if grade in {"A", "B", "C"} and should_add_implementation_section(paper_memory):
        return templates
    return [template for template in templates if template["section_id"] != "implementation"]


def default_report_section_templates() -> list[dict[str, Any]]:
    return [
        {
            "section_id": "orientation",
            "section_kind": "orientation",
            "title": "论文概览与核心问题",
            "purpose": "Explain the paper's problem, why it matters, and the core abstraction.",
            "coverage_group": "orientation",
            "detail_questions": [
                "What concrete problem or bottleneck is the paper attacking?",
                "Why do prior approaches fail or leave a gap?",
                "What is the paper's core abstraction or thesis?",
                "What should readers not over-claim from this paper?",
            ],
        },
        {
            "section_id": "mechanism",
            "section_kind": "mechanism",
            "title": "核心机制与设计结构",
            "purpose": "Explain the main mechanism in reader order.",
            "coverage_group": "mechanism",
            "detail_questions": [
                "What state, representation, or model view exists before the mechanism?",
                "What new abstraction or component changes that state?",
                "How do the main components connect?",
                "Why should this mechanism improve the target metric or behavior?",
            ],
        },
        {
            "section_id": "implementation",
            "section_kind": "mechanism",
            "title": "关键实现路径与细节",
            "purpose": "Walk through the implementation, algorithm, runtime path, or training/inference path.",
            "coverage_group": "implementation",
            "detail_questions": [
                "What data structures, equations, modules, or runtime components make it concrete?",
                "Walk through one request, object, sample, or inference lifecycle step by step.",
                "Which parameters, losses, schedules, or invariants matter?",
                "Where are the main overheads or fragile assumptions introduced?",
            ],
        },
        {
            "section_id": "evaluation",
            "section_kind": "evaluation",
            "title": "实验评估与证据",
            "purpose": "Explain datasets/workloads, metrics, baselines, headline results, and evidence limits.",
            "coverage_group": "evaluation",
            "detail_questions": [
                "What datasets, workloads, metrics, and baselines are used?",
                "What are the headline quantitative results?",
                "Which ablations or qualitative results support the mechanism?",
                "What exact result boundaries should readers keep in mind?",
            ],
        },
        {
            "section_id": "value",
            "section_kind": "value",
            "title": "价值、适用场景与权衡",
            "purpose": "Explain why the result is useful, where it transfers, and what tradeoff it chooses.",
            "coverage_group": "value",
            "detail_questions": [
                "What practical or conceptual lesson transfers beyond this paper?",
                "Which users, systems, tasks, or research directions benefit most?",
                "What tradeoff does the paper choose instead of optimizing everything?",
                "When would this idea be less useful?",
            ],
        },
        {
            "section_id": "limits",
            "section_kind": "limits",
            "title": "局限性与可信边界",
            "purpose": "State scope, assumptions, missing evidence, and open questions without burying them.",
            "coverage_group": "limits",
            "detail_questions": [
                "What evaluation scope, assumptions, or missing details limit the conclusion?",
                "Which claims require going back to the source before citation?",
                "What deployment, reproducibility, scaling, or generalization risks remain?",
                "What follow-up questions should a reader ask?",
            ],
        },
    ]


def should_add_implementation_section(paper_memory: dict[str, Any]) -> bool:
    core = core_memory_view_dict(paper_memory)
    if core:
        if list_payload(core.get("implementation_nodes")):
            return True
        if any(
            node.get("kind") in {"implementation", "mechanism", "evaluation", "result"}
            for node in list_payload(core.get("fact_nodes"))
        ):
            return True
    memory = paper_memory_v3_dict(paper_memory)
    mechanism = dict_value(memory.get("mechanism"))
    implementation = dict_value(memory.get("implementation_details"))
    if len(list_payload(mechanism.get("steps"))) >= 2:
        return True
    if list_payload(implementation.get("components")):
        return True
    if list_payload(memory.get("figures_tables")):
        return True
    return bool(dict_value(memory.get("evaluation")).get("summary"))


def report_plan_template_is_covered(
    template: dict[str, Any], sections: list[dict[str, Any]]
) -> bool:
    group = str(template.get("coverage_group") or template.get("section_kind") or "")
    if group == "orientation":
        return any(
            section.get("section_kind") in {"orientation", "background"} for section in sections
        )
    if group == "implementation":
        mechanism_sections = [
            section for section in sections if section.get("section_kind") == "mechanism"
        ]
        if len(mechanism_sections) >= 2:
            return True
        return any(
            any(
                token
                in " ".join(
                    [
                        str(section.get("section_id") or ""),
                        str(section.get("title") or ""),
                        str(section.get("purpose") or ""),
                    ]
                ).lower()
                for token in [
                    "implementation",
                    "runtime",
                    "algorithm",
                    "training",
                    "inference",
                    "实现",
                    "细节",
                    "运行",
                    "训练",
                    "推理",
                    "路径",
                ]
            )
            for section in sections
        )
    return any(section.get("section_kind") == group for section in sections)


def build_supplemental_report_section(
    template: dict[str, Any], paper_memory: dict[str, Any]
) -> dict[str, Any]:
    group = str(template.get("coverage_group") or template.get("section_kind") or "other")
    seed = report_section_seed_context(group, paper_memory)
    return normalize_report_section_plan(
        {
            "section_id": template["section_id"],
            "section_kind": template["section_kind"],
            "title": template["title"],
            "purpose": template["purpose"],
            "focus_queries": seed["focus_queries"],
            "claim_ids": seed["claim_ids"],
            "evidence_refs": seed["evidence_refs"],
            "target_pages": seed["target_pages"],
            "detail_questions": template["detail_questions"],
            "avoid": [],
        }
    )


def report_section_seed_context(group: str, paper_memory: dict[str, Any]) -> dict[str, Any]:
    core = core_memory_view_dict(paper_memory)
    if core:
        return core_report_section_seed_context(group, core)
    memory = paper_memory_v3_dict(paper_memory)
    claims = [claim for claim in list_payload(memory.get("claims")) if isinstance(claim, dict)]
    evidence = [item for item in list_payload(memory.get("evidence")) if isinstance(item, dict)]
    claim_ids: list[str] = []
    evidence_refs: list[str] = []
    focus_queries: list[str] = []
    target_pages: list[int] = []

    def add_claim(claim: dict[str, Any]) -> None:
        claim_id = _string_or_none(claim.get("id"))
        if claim_id and claim_id not in claim_ids:
            claim_ids.append(claim_id)
        text = _string_or_none(claim.get("text"))
        if text:
            focus_queries.append(text)
        for ref in _normalized_string_list(claim.get("evidence_refs")):
            if ref not in evidence_refs:
                evidence_refs.append(ref)

    def claim_matches(claim: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(claim.get("type") or ""),
                str(claim.get("text") or ""),
                " ".join(_normalized_string_list(claim.get("risk_tags"))),
            ]
        ).lower()
        if group in {"mechanism", "implementation"}:
            return any(
                token in text
                for token in [
                    "mechanism",
                    "design",
                    "algorithm",
                    "architecture",
                    "implementation",
                    "implication",
                    "机制",
                    "设计",
                    "算法",
                    "架构",
                    "实现",
                ]
            )
        if group == "evaluation":
            return any(
                token in text
                for token in [
                    "evaluation",
                    "comparison",
                    "benchmark",
                    "metric",
                    "performance",
                    "实验",
                    "评估",
                    "基线",
                    "性能",
                ]
            )
        if group == "limits":
            return any(
                token in text
                for token in [
                    "limitation",
                    "scope",
                    "risk",
                    "assumption",
                    "boundary",
                    "局限",
                    "边界",
                    "假设",
                ]
            )
        if group == "value":
            return any(
                token in text
                for token in [
                    "value",
                    "tradeoff",
                    "application",
                    "deployment",
                    "efficiency",
                    "robustness",
                    "价值",
                    "权衡",
                    "适用",
                    "部署",
                ]
            )
        return True

    for claim in claims:
        if claim_matches(claim):
            add_claim(claim)
        if len(claim_ids) >= 6:
            break
    if not claim_ids and group in {"orientation", "value"}:
        for claim in claims[:4]:
            add_claim(claim)

    evidence_by_id = {str(item.get("id")): item for item in evidence if item.get("id")}
    for ref in evidence_refs:
        item = evidence_by_id.get(ref)
        if item:
            page = safe_int(item.get("page"))
            if page and page not in target_pages:
                target_pages.append(page)
    for item in evidence:
        text = " ".join(
            [
                str(item.get("source_type") or ""),
                str(item.get("section") or ""),
                str(item.get("interpretation") or ""),
                str(item.get("excerpt_or_caption") or ""),
            ]
        ).lower()
        if group == "evaluation" and not any(
            token in text for token in ["table", "result", "metric", "baseline", "实验", "评估"]
        ):
            continue
        if group in {"mechanism", "implementation"} and not any(
            token in text
            for token in ["figure", "design", "architecture", "equation", "module", "机制", "架构"]
        ):
            continue
        evidence_id = _string_or_none(item.get("id"))
        if evidence_id and evidence_id not in evidence_refs:
            evidence_refs.append(evidence_id)
        page = safe_int(item.get("page"))
        if page and page not in target_pages:
            target_pages.append(page)
        if len(evidence_refs) >= 8 and len(target_pages) >= 4:
            break

    frame = dict_value(memory.get("problem_frame"))
    mechanism = dict_value(memory.get("mechanism"))
    evaluation = dict_value(memory.get("evaluation"))
    if group == "orientation":
        focus_queries.extend([frame.get("problem"), frame.get("why_it_matters")])
        for item in list_payload(memory.get("core_abstractions"))[:2]:
            if isinstance(item, dict):
                focus_queries.append(str(item.get("text") or ""))
    elif group in {"mechanism", "implementation"}:
        focus_queries.append(str(mechanism.get("overview") or ""))
        for step in list_payload(mechanism.get("steps"))[:5]:
            if isinstance(step, dict):
                focus_queries.append(str(step.get("text") or ""))
    elif group == "evaluation":
        focus_queries.append(str(evaluation.get("summary") or ""))
        for item in list_payload(evaluation.get("items"))[:4]:
            if isinstance(item, dict):
                focus_queries.append(str(item.get("text") or ""))
    elif group == "limits":
        focus_queries.extend(_normalized_string_list(memory.get("limitations"))[:5])
        focus_queries.extend(_normalized_string_list(memory.get("open_questions"))[:4])

    return {
        "focus_queries": compact_string_list(focus_queries, limit=5, max_chars=180),
        "claim_ids": claim_ids[:8],
        "evidence_refs": evidence_refs[:10],
        "target_pages": target_pages[:6],
    }


def core_report_section_seed_context(group: str, core: dict[str, Any]) -> dict[str, Any]:
    fact_nodes = [node for node in list_payload(core.get("fact_nodes")) if isinstance(node, dict)]
    selected = [node for node in fact_nodes if core_fact_matches_group(node, group)]
    if not selected and group in {"orientation", "value"}:
        selected = fact_nodes[:4]
    focus_queries = [node.get("label") for node in selected[:6]]
    claim_ids = [
        _string_or_none(node.get("node_id")) or "" for node in selected if node.get("node_id")
    ]
    evidence_refs: list[str] = []
    target_pages: list[int] = []
    for node in selected:
        for evidence_id in _normalized_string_list(node.get("evidence_ids")):
            if evidence_id not in evidence_refs:
                evidence_refs.append(evidence_id)
        for source_id in _normalized_string_list(node.get("source_ids")):
            if source_id not in evidence_refs:
                evidence_refs.append(source_id)
        for page in list_payload(node.get("pages")):
            page_no = safe_int(page)
            if page_no and page_no not in target_pages:
                target_pages.append(page_no)
    return {
        "focus_queries": compact_string_list(focus_queries, limit=5, max_chars=180),
        "claim_ids": claim_ids[:8],
        "evidence_refs": evidence_refs[:10],
        "target_pages": target_pages[:6],
    }


def core_fact_matches_group(node: dict[str, Any], group: str) -> bool:
    kind = _string_or_none(node.get("kind")) or ""
    label = (_string_or_none(node.get("label")) or "").lower()
    if group == "orientation":
        return kind in {"problem", "claim", "concept"}
    if group in {"mechanism", "implementation"}:
        return kind in {"mechanism", "implementation"} or any(
            token in label
            for token in ["mechanism", "implementation", "algorithm", "module", "机制", "实现"]
        )
    if group == "evaluation":
        return kind in {"evaluation", "result"}
    if group == "limits":
        return kind == "limitation"
    if group == "value":
        return kind in {"claim", "result", "concept"}
    return True


def paper_memory_v3_dict(memory: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    if memory.get("schema_version") == "paper_memory.v3":
        return memory
    nested = dict_value(memory.get("paper_memory_v3"))
    return nested if nested.get("schema_version") == "paper_memory.v3" else {}


def normalize_report_section_plan(data: dict[str, Any]) -> dict[str, Any]:
    section_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(data.get("section_id") or "")).strip("_")
    title = clean_model_inline_text(data.get("title"))
    if not section_id:
        section_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", title.lower()).strip("_") or "section"
    section_kind = str(data.get("section_kind") or "").strip()
    if section_kind not in {
        "orientation",
        "background",
        "mechanism",
        "evaluation",
        "value",
        "limits",
        "other",
    }:
        section_kind = infer_report_section_kind(
            section_id=section_id, title=title, purpose=data.get("purpose")
        )
    return {
        "section_id": section_id[:40],
        "section_kind": section_kind,
        "title": title or "正文",
        "purpose": clean_model_inline_text(data.get("purpose")),
        "focus_queries": compact_string_list(data.get("focus_queries"), limit=5, max_chars=180),
        "claim_ids": compact_string_list(data.get("claim_ids"), limit=8, max_chars=40),
        "evidence_refs": compact_string_list(data.get("evidence_refs"), limit=10, max_chars=40),
        "target_pages": [
            page
            for page in (safe_int(value) for value in list_payload(data.get("target_pages")))
            if page
        ],
        "detail_questions": compact_string_list(
            data.get("detail_questions"), limit=8, max_chars=180
        ),
        "avoid": compact_string_list(data.get("avoid"), limit=5, max_chars=160),
    }


def default_report_sections() -> list[dict[str, Any]]:
    return [
        build_supplemental_report_section(template, {})
        for template in default_report_section_templates()
    ]


def _normalized_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str):
            cleaned = re.sub(r"\s+", " ", item).strip()
            if cleaned:
                result.append(cleaned)
    return result[:20]


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned or None
