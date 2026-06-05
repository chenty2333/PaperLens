from __future__ import annotations

import re
from typing import Any

from paperlens_core.memory_store import PaperMemoryStore, normalize_memory_patch_set
from paperlens_core.memory_v3 import dict_value, normalized_string_list


def normalize_memory_audit(data: dict[str, Any]) -> dict[str, Any]:
    status = str(data.get("status") or "NEED_HUMAN_REVIEW").upper()
    if status not in {"PASS", "PASS_WITH_WEAKNESSES", "NEED_HUMAN_REVIEW"}:
        status = "NEED_HUMAN_REVIEW"
    safe_to_generate = (
        bool(data.get("safe_to_generate_capsule")) if "safe_to_generate_capsule" in data else False
    )
    if status == "NEED_HUMAN_REVIEW":
        safe_to_generate = False
    return {
        "status": status,
        "unsupported_claims": normalized_string_list(data.get("unsupported_claims"))[:6],
        "missing_items": normalized_string_list(data.get("missing_items"))[:8],
        "repair_instructions": normalized_string_list(
            data.get("repair_instructions") or data.get("correction_notes")
        )[:8],
        "safe_to_generate_capsule": safe_to_generate,
        "confidence": str(data.get("confidence") or "low")
        if str(data.get("confidence") or "low") in {"high", "medium", "low"}
        else "low",
    }


def fallback_memory_audit(*, reason: str, phase: str) -> dict[str, Any]:
    return normalize_memory_audit(
        {
            "status": "PASS_WITH_WEAKNESSES",
            "unsupported_claims": [],
            "missing_items": [f"{phase} did not complete"],
            "repair_instructions": [
                f"{phase} failed during this run; treat the memory as usable but weak until rerun succeeds.",
                _compact_text(reason, max_chars=220),
            ],
            "safe_to_generate_capsule": True,
            "confidence": "low",
        }
    )


def memory_audit_acceptable(audit: dict[str, Any]) -> bool:
    return audit.get("status") in {"PASS", "PASS_WITH_WEAKNESSES"} and bool(
        audit.get("safe_to_generate_capsule")
    )


def memory_without_audit(memory: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in memory.items() if key != "memory_audit"}
    audit_trail = result.get("audit_trail")
    if isinstance(audit_trail, dict):
        result["audit_trail"] = {
            key: value
            for key, value in audit_trail.items()
            if key not in {"memory_audit", "report_audit"}
        }
    return result


def apply_memory_audit_patch(
    store: PaperMemoryStore,
    paper_id: str,
    audit: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    return store.apply_patch_set(
        paper_id,
        {
            "paper_id": paper_id,
            "operations": [{"op": "set_memory_audit", "payload": audit}],
        },
        source=source,
    )


def ensure_memory_audit_operation(
    patch_set: dict[str, Any],
    *,
    paper_id: str,
    phase: str,
) -> dict[str, Any]:
    normalized = normalize_memory_patch_set(patch_set, paper_id=paper_id)
    operations = normalized.setdefault("operations", [])
    for operation in operations:
        if operation.get("op") == "set_memory_audit":
            operation["payload"] = normalize_memory_audit(dict_value(operation.get("payload")))
            return normalized
    operations.append(
        {
            "op": "set_memory_audit",
            "payload": fallback_memory_audit(
                reason="Verifier did not include an explicit audit operation.",
                phase=phase,
            ),
        }
    )
    return normalized


def _compact_text(text: str, *, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    for mark in "。！？.!?；;，,":
        index = cleaned.rfind(mark, 0, max_chars)
        if index >= 40:
            return cleaned[: index + 1]
    return cleaned[:max_chars].rstrip() + "..."
