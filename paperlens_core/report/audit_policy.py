from __future__ import annotations

import re
from typing import Any

from paperlens_core.memory_v3 import dict_value, list_payload, normalized_string_list


def final_report_audit_acceptable(report_audit: dict[str, Any] | None) -> bool:
    if not report_audit:
        return False
    return report_audit.get("verdict") in {"PASS", "PASS_WITH_WEAKNESSES"}


def combine_report_and_memory_audits(
    report_audit: dict[str, Any] | None, paper_memory: dict[str, Any] | None
) -> dict[str, Any] | None:
    if report_audit is None:
        return None
    memory = _paper_memory_v3_dict(dict_value(paper_memory))
    memory_audit = dict_value(dict_value(memory.get("audit_trail")).get("memory_audit"))
    if not memory_audit:
        return report_audit
    result = dict(report_audit)
    verdict = str(result.get("verdict") or "NEED_HUMAN_REVIEW")
    memory_status = str(memory_audit.get("status") or "").upper()
    memory_safe = bool(memory_audit.get("safe_to_generate_capsule"))
    memory_confidence = str(memory_audit.get("confidence") or "low")
    if memory_status == "NEED_HUMAN_REVIEW" or not memory_safe:
        verdict = "NEED_HUMAN_REVIEW"
    elif memory_status == "PASS_WITH_WEAKNESSES" or memory_confidence == "low":
        if verdict == "PASS":
            verdict = "PASS_WITH_WEAKNESSES"
    result["verdict"] = verdict
    result["unsupported_items"] = _compact_string_list(
        list_payload(result.get("unsupported_items"))
        + normalized_string_list(memory_audit.get("unsupported_claims")),
        limit=5,
        max_chars=240,
    )
    result["missing_items"] = _compact_string_list(
        list_payload(result.get("missing_items"))
        + normalized_string_list(memory_audit.get("missing_items")),
        limit=5,
        max_chars=240,
    )
    result["correction_notes"] = _compact_string_list(
        list_payload(result.get("correction_notes"))
        + normalized_string_list(memory_audit.get("repair_instructions")),
        limit=5,
        max_chars=240,
    )
    notes = _compact_string_list(
        [result.get("safe_usage_note"), memory_audit_safe_usage_note(memory_audit)],
        limit=2,
        max_chars=260,
    )
    result["safe_usage_note"] = "; ".join(notes)
    return result


def memory_audit_safe_usage_note(memory_audit: dict[str, Any]) -> str:
    status = str(memory_audit.get("status") or "").upper()
    confidence = str(memory_audit.get("confidence") or "low")
    if status == "PASS":
        return ""
    if status == "NEED_HUMAN_REVIEW" or not memory_audit.get("safe_to_generate_capsule"):
        return "PaperMemory audit requires human review before treating this capsule as reviewed."
    if confidence == "low":
        return "PaperMemory audit confidence is low; keep evidence boundaries visible."
    return "PaperMemory audit passed with evidence boundaries."


def _paper_memory_v3_dict(memory: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    if memory.get("schema_version") == "paper_memory.v3":
        return memory
    nested = dict_value(memory.get("paper_memory_v3"))
    return nested if nested.get("schema_version") == "paper_memory.v3" else {}


def _compact_string_list(value: Any, *, limit: int, max_chars: int) -> list[str]:
    return [
        _compact_text(item, max_chars=max_chars)
        for item in normalized_string_list(value)[:limit]
    ]


def _compact_text(text: str, *, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    for mark in "。！？.!?；;，,":
        index = cleaned.rfind(mark, 0, max_chars)
        if index >= 40:
            return cleaned[: index + 1]
    return cleaned[:max_chars].rstrip() + "..."
