from __future__ import annotations

import re
from typing import Any

from paperlens_core.memory_v3 import list_payload
from paperlens_core.report.text import (
    clean_model_inline_text,
    clean_model_markdown,
    compact_reason,
    compact_string_list,
    readable_model_body,
    recommendation_for_grade,
    sanitize_reader_hostile_text,
    user_facing_uncertainty_note,
)
from paperlens_core.schemas import ClassificationDecision, PaperRecord


def normalize_report_section(
    data: dict[str, Any], *, section_plan: dict[str, Any]
) -> dict[str, Any]:
    paragraphs = compact_string_list(data.get("paragraphs"), limit=12, max_chars=1800)
    markdown_source = "\n\n".join(paragraphs) if paragraphs else data.get("markdown")
    return {
        "section_id": str(data.get("section_id") or section_plan.get("section_id") or "section"),
        "title": clean_model_inline_text(data.get("title"))
        or str(section_plan.get("title") or "正文"),
        "paragraphs": paragraphs,
        "markdown": sanitize_reader_hostile_text(readable_model_body(markdown_source)),
        "used_claim_ids": compact_string_list(data.get("used_claim_ids"), limit=12, max_chars=40),
        "used_evidence_refs": compact_string_list(
            data.get("used_evidence_refs"), limit=16, max_chars=40
        ),
        "uncertainty_note": clean_model_markdown(data.get("uncertainty_note")),
    }


def normalize_report_section_audit(data: dict[str, Any]) -> dict[str, Any]:
    verdict = str(data.get("verdict") or "REPAIR").upper()
    if verdict not in {"PASS", "PASS_WITH_WEAKNESSES", "REPAIR"}:
        verdict = "REPAIR"
    unsupported_items = compact_string_list(data.get("unsupported_items"), limit=6, max_chars=240)
    if unsupported_items:
        verdict = "REPAIR"
    return {
        "verdict": verdict,
        "unsupported_items": unsupported_items,
        "missing_items": compact_string_list(data.get("missing_items"), limit=6, max_chars=240),
        "repair_instructions": compact_string_list(
            data.get("repair_instructions"), limit=6, max_chars=240
        ),
        "safe_usage_note": clean_model_markdown(data.get("safe_usage_note")),
    }


def enforce_section_depth_audit(
    audit: dict[str, Any], *, section: dict[str, Any], section_plan: dict[str, Any]
) -> dict[str, Any]:
    issue = section_depth_issue(section, section_plan)
    if not issue:
        return audit
    result = dict(audit)
    result["missing_items"] = compact_string_list(
        list_payload(result.get("missing_items")) + [issue], limit=6, max_chars=240
    )
    result["repair_instructions"] = compact_string_list(
        list_payload(result.get("repair_instructions"))
        + [
            "Rewrite this section with concrete mechanism/evidence/boundary detail from PaperMemory and focused pages."
        ],
        limit=6,
        max_chars=240,
    )
    result["verdict"] = "REPAIR"
    if not result.get("safe_usage_note"):
        result["safe_usage_note"] = "Section is too thin for its planned purpose."
    return result


def section_depth_issue(section: dict[str, Any], section_plan: dict[str, Any]) -> str:
    markdown = readable_model_body(section.get("markdown"))
    normalized = re.sub(r"\s+", "", markdown)
    char_count = len(normalized)
    kind = str(section_plan.get("section_kind") or "other")
    thresholds = {
        "orientation": 420,
        "background": 420,
        "mechanism": 650,
        "evaluation": 560,
        "value": 430,
        "limits": 380,
        "other": 360,
    }
    minimum = thresholds.get(kind, thresholds["other"])
    section_id = section_plan.get("section_id") or section.get("section_id")
    if char_count < minimum:
        return (
            f"Section '{section_id}' is too thin for {kind} coverage "
            f"({char_count} chars; expected at least {minimum})."
        )
    paragraphs = [
        paragraph
        for paragraph in re.split(r"\n\s*\n", markdown)
        if clean_model_inline_text(paragraph)
    ]
    if kind in {"mechanism", "evaluation"} and len(paragraphs) < 2:
        return (
            f"Section '{section_id}' needs at least two substantive paragraphs for {kind} coverage."
        )
    return ""


def report_section_is_more_substantive(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    candidate_text = re.sub(r"\s+", "", readable_model_body(candidate.get("markdown")))
    current_text = re.sub(r"\s+", "", readable_model_body(current.get("markdown")))
    return len(candidate_text) >= max(len(current_text) + 120, int(len(current_text) * 1.2))


def assemble_agentic_report(
    *,
    paper: PaperRecord,
    decision: ClassificationDecision | None,
    plan: dict[str, Any],
    sections: list[dict[str, Any]],
    section_audits: list[dict[str, Any]],
    output_language: str,
) -> dict[str, Any]:
    grade = str(plan.get("grade") or (decision.class_label if decision else "HOLD")).upper()
    if grade not in {"A", "B", "C", "HOLD"}:
        grade = "HOLD"
    body_parts = []
    for section in sections:
        title = clean_model_inline_text(section.get("title"))
        markdown = readable_model_body(section.get("markdown"))
        if not markdown:
            continue
        if title:
            body_parts.append(f"## {title}\n\n{markdown}")
        else:
            body_parts.append(markdown)
    if not body_parts:
        body_parts.append(
            "模型没有生成可用的分段讲解。"
            if output_language == "zh"
            else "No usable section draft was generated."
        )
    uncertainty_parts = [user_facing_uncertainty_note(plan.get("uncertainty_note"))]
    if any(audit.get("verdict") != "PASS" for audit in section_audits):
        uncertainty_parts.append(
            "部分段落存在证据边界；具体数值、基线、硬件配置和外推结论建议按需追问。"
            if output_language == "zh"
            else "Some sections have evidence boundaries; ask follow-up questions before relying on exact numbers, baselines, hardware setup, or broad extrapolations."
        )
    return {
        "grade": grade,
        "review_status": section_review_status(section_audits, output_language=output_language),
        "read_recommendation": plan.get("read_recommendation") or recommendation_for_grade(grade),
        "one_line_reason": clean_model_inline_text(plan.get("one_line_reason"))
        or compact_reason(
            str(plan.get("core_takeaway") or paper.canonical_title or paper.paper_id)
        ),
        "core_takeaway": clean_model_markdown(plan.get("core_takeaway")),
        "explanation_markdown": "\n\n".join(body_parts),
        "uncertainty_note": "; ".join(item for item in uncertainty_parts if item),
        "key_visual_pages": normalize_key_visual_pages(plan.get("key_visual_pages")),
        "report_plan": plan,
        "section_audits": section_audits,
    }


def aggregate_section_audits(section_audits: list[dict[str, Any]]) -> dict[str, Any]:
    if not section_audits:
        return {
            "verdict": "NEED_HUMAN_REVIEW",
            "unsupported_items": [],
            "missing_items": ["No report sections were generated"],
            "correction_notes": ["ReportComposer produced no section artifacts"],
            "safe_usage_note": "No usable section-level report was generated.",
        }
    if any(audit.get("verdict") in {"REPAIR", "PASS_WITH_WEAKNESSES"} for audit in section_audits):
        verdict = "PASS_WITH_WEAKNESSES"
    else:
        verdict = "PASS"
    return {
        "verdict": verdict,
        "unsupported_items": compact_string_list(
            [
                item
                for audit in section_audits
                for item in list_payload(audit.get("unsupported_items"))
            ],
            limit=5,
            max_chars=240,
        ),
        "missing_items": compact_string_list(
            [item for audit in section_audits for item in list_payload(audit.get("missing_items"))],
            limit=5,
            max_chars=240,
        ),
        "correction_notes": compact_string_list(
            [
                item
                for audit in section_audits
                for item in list_payload(audit.get("repair_instructions"))
            ],
            limit=5,
            max_chars=240,
        ),
        "safe_usage_note": "; ".join(
            item
            for item in compact_string_list(
                [audit.get("safe_usage_note") for audit in section_audits],
                limit=3,
                max_chars=220,
            )
            if item
        ),
    }


def section_review_status(section_audits: list[dict[str, Any]], *, output_language: str) -> str:
    if not section_audits:
        return "需要人工确认" if output_language == "zh" else "needs human review"
    if any(audit.get("verdict") == "REPAIR" for audit in section_audits):
        return (
            "已分段复核（有未修复边界）"
            if output_language == "zh"
            else "section-audited with unresolved boundaries"
        )
    if any(audit.get("verdict") == "PASS_WITH_WEAKNESSES" for audit in section_audits):
        return (
            "已分段复核（有证据边界）"
            if output_language == "zh"
            else "section-audited with evidence boundaries"
        )
    return "已分段复核" if output_language == "zh" else "section-audited"


def normalize_key_visual_pages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    pages = []
    for item in value:
        if not isinstance(item, dict):
            continue
        page_no = item.get("page_no")
        if not isinstance(page_no, int) or page_no <= 0:
            continue
        pages.append(
            {
                "page_no": page_no,
                "reason": compact_reason(
                    clean_model_inline_text(item.get("reason")), max_chars=180
                ),
            }
        )
        if len(pages) >= 3:
            break
    return pages
