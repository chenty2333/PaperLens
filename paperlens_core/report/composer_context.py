from __future__ import annotations

from typing import Any

from paperlens_core.report.text import (
    compact_reason,
    compact_string_list,
    recommendation_for_grade,
)
from paperlens_core.schemas import ClassificationDecision, PaperCard, PaperRecord, SkimCard


def compact_skim_for_report(skim: SkimCard | None) -> dict[str, Any]:
    if skim is None:
        return {}
    return {
        "problem": compact_reason(skim.problem, max_chars=240),
        "method_type": skim.method_type,
        "system_scope": skim.system_scope,
        "evaluation_type": skim.evaluation_type,
        "danger_signals": compact_string_list(skim.danger_signals, limit=3, max_chars=120),
        "evidence_pages": [ref.page_no for ref in skim.evidence_refs[:4]],
    }


def compact_decision_for_report(decision: ClassificationDecision | None) -> dict[str, Any]:
    if decision is None:
        return {}
    return {
        "class_label": decision.class_label,
        "confidence": decision.confidence,
        "false_negative_risk": decision.false_negative_risk,
        "reason_codes": compact_string_list(decision.reason_codes, limit=3, max_chars=90),
        "audit_status": decision.audit_status,
        "validation_status": decision.validation_status,
        "validation_notes": compact_string_list(decision.validation_notes, limit=2, max_chars=110),
    }


def compact_paper_card_for_report(card: PaperCard | None) -> dict[str, Any]:
    if card is None:
        return {}
    return {
        "contribution_claims": compact_string_list(
            card.contribution_claims, limit=3, max_chars=170
        ),
        "mechanisms": compact_string_list(card.mechanisms, limit=3, max_chars=170),
        "evaluation": compact_string_list(card.evaluation, limit=2, max_chars=160),
        "limitations": compact_string_list(card.limitations, limit=2, max_chars=160),
        "assumptions": compact_string_list(card.assumptions, limit=2, max_chars=120),
        "verification_status": card.verification_status,
        "evidence_pages": [ref.page_no for ref in card.evidence_refs[:4]],
    }


def fallback_model_paper_report(
    *,
    paper: PaperRecord,
    skim: SkimCard | None,
    decision: ClassificationDecision | None,
    card: PaperCard | None,
    paper_memory: dict[str, Any] | None,
    layout: dict[str, Any],
    topic: str | None,
    idea: str | None,
    reason: str,
    output_language: str = "zh",
) -> dict[str, Any]:
    _ = (paper_memory, layout, topic, idea)
    grade = decision.class_label if decision else "HOLD"
    limitation_note = f"Model final-report generation failed: {reason}"
    body_parts = []
    if output_language == "en":
        if skim and skim.problem:
            body_parts.append(f"This paper is roughly about: {skim.problem}")
        if card and card.contribution_claims:
            body_parts.append(
                "Current reliable signals: " + "; ".join(card.contribution_claims[:3]) + "."
            )
        if card and card.mechanisms:
            body_parts.append(
                "Possible mechanisms include: " + "; ".join(card.mechanisms[:3]) + "."
            )
        body_parts.append(
            "Final explanation generation failed, so this is a conservative fallback rather than a reliable capsule."
        )
        review_status = "needs human review"
    elif skim and skim.problem:
        body_parts.append(f"这篇论文大致在处理：{skim.problem}")
        if card and card.contribution_claims:
            body_parts.append(
                "当前能确定的主要线索是：" + "；".join(card.contribution_claims[:3]) + "。"
            )
        if card and card.mechanisms:
            body_parts.append("可能的关键做法包括：" + "；".join(card.mechanisms[:3]) + "。")
        body_parts.append(
            "但最终讲解生成失败，所以这份报告只能作为保守兜底结果，不能当成可靠论文总结。"
        )
        review_status = "需要人工确认"
    else:
        body_parts.append(
            "但最终讲解生成失败，所以这份报告只能作为保守兜底结果，不能当成可靠论文总结。"
        )
        review_status = "需要人工确认"
    return {
        "grade": grade,
        "review_status": review_status,
        "read_recommendation": recommendation_for_grade(grade),
        "one_line_reason": limitation_note,
        "explanation_markdown": "\n\n".join(body_parts),
        "uncertainty_note": limitation_note,
    }
