from __future__ import annotations

import json
import os
import re
from typing import Any

from paperlens_core.memory_store import normalize_memory_patch_set
from paperlens_core.memory_v3 import (
    dict_value,
    list_payload,
    normalized_string_list,
    safe_int,
    string_or_none,
)


def paper_memory_has_recoverable_content(memory: dict[str, Any]) -> bool:
    if memory.get("schema_version") == "paper_memory.v3":
        if list_payload(memory.get("claims")):
            return True
        if list_payload(memory.get("core_abstractions")):
            return True
        if dict_value(memory.get("problem_frame")).get("problem"):
            return True
        return bool(memory_v3_pages_read(memory))
    if string_or_none(memory.get("core_thesis")):
        return True
    if memory.get("claims") and isinstance(memory.get("claims"), list):
        return True
    if memory.get("mechanism") and isinstance(memory.get("mechanism"), dict):
        return True
    return bool(memory.get("pages_read"))


def ensure_read_pages_operation(
    patch_set: dict[str, Any], *, paper_id: str, pages: list[int]
) -> dict[str, Any]:
    normalized = normalize_memory_patch_set(patch_set, paper_id=paper_id)
    normalized["paper_id"] = paper_id
    existing = [
        operation
        for operation in normalized.get("operations", [])
        if operation.get("op") == "add_read_pages"
    ]
    if not existing:
        normalized.setdefault("operations", []).insert(
            0,
            {
                "op": "add_read_pages",
                "payload": {"pages": [page for page in pages if isinstance(page, int)]},
            },
        )
    return normalized


def memory_v3_pages_read(memory: dict[str, Any]) -> set[int]:
    context = dict_value(memory.get("reading_context"))
    return {
        page
        for page in [safe_int(value) for value in context.get("pages_read", [])]
        if page is not None and page > 0
    }


def select_high_risk_memory_claims(
    memory: dict[str, Any], *, limit: int = 10
) -> list[dict[str, Any]]:
    claims = [claim for claim in list_payload(memory.get("claims")) if isinstance(claim, dict)]
    evidence = {
        str(item.get("id")): item
        for item in list_payload(memory.get("evidence"))
        if isinstance(item, dict) and item.get("id")
    }

    def score(claim: dict[str, Any]) -> tuple[int, str]:
        refs = normalized_string_list(claim.get("evidence_refs"))
        risk_tags = normalized_string_list(claim.get("risk_tags"))
        value = 0
        if not refs:
            value += 8
        if claim.get("confidence") in {"low", "medium"}:
            value += 3
        if claim.get("critic_status") in {"unchecked", "disputed"}:
            value += 4
        if claim.get("provenance") in {"inferred", "background"}:
            value += 2
        if any(
            tag in {"needs_evidence", "number_sensitive", "analogy_overreach"} for tag in risk_tags
        ):
            value += 3
        if claim.get("type") in {"evaluation", "comparison", "limitation"}:
            value += 2
        if refs and not any(ref in evidence for ref in refs):
            value += 5
        return (-value, str(claim.get("id") or claim.get("text") or ""))

    selected = sorted(claims, key=score)[:limit]
    return [
        {
            "id": claim.get("id"),
            "text": claim.get("text"),
            "type": claim.get("type"),
            "provenance": claim.get("provenance"),
            "confidence": claim.get("confidence"),
            "critic_status": claim.get("critic_status"),
            "risk_tags": normalized_string_list(claim.get("risk_tags"))[:6],
            "evidence_refs": normalized_string_list(claim.get("evidence_refs"))[:8],
        }
        for claim in selected
    ]


def select_central_verification_pages(
    *,
    memory: dict[str, Any],
    all_artifacts: list[Any],
    read_artifacts: list[Any],
) -> list[Any]:
    max_pages = _bounded_env_int(
        "PAPERLENS_MEMORY_VERIFY_MAX_PAGES", default=8, minimum=3, maximum=14
    )
    by_no = {getattr(artifact, "page_no", None): artifact for artifact in all_artifacts}
    selected: list[Any] = []
    selected_pages: set[int] = set()

    def add(page_no: Any) -> None:
        page = safe_int(page_no)
        if page is None or page not in by_no or page in selected_pages:
            return
        selected.append(by_no[page])
        selected_pages.add(page)

    evidence_page_by_id = {}
    for item in list_payload(memory.get("evidence")):
        if not isinstance(item, dict):
            continue
        page = safe_int(item.get("page"))
        if item.get("id") and page:
            evidence_page_by_id[str(item.get("id"))] = page
        add(page)
        if len(selected) >= max_pages // 2:
            break

    for claim in select_high_risk_memory_claims(memory, limit=8):
        for ref in normalized_string_list(claim.get("evidence_refs")):
            add(safe_int(ref) or evidence_page_by_id.get(ref))
        if len(selected) >= max_pages:
            break

    for artifact in read_artifacts:
        add(getattr(artifact, "page_no", None))
        if len(selected) >= max_pages:
            return selected[:max_pages]

    keyword_pages = []
    for artifact in all_artifacts:
        page_no = getattr(artifact, "page_no", None)
        text = _normalize_for_search(
            " ".join(
                [
                    str(getattr(artifact, "text", "") or "")[:1800],
                    json.dumps(getattr(artifact, "captions", [])[:4], ensure_ascii=False),
                ]
            )
        )
        score = sum(
            1
            for word in [
                "abstract",
                "introduction",
                "overview",
                "design",
                "implementation",
                "evaluation",
                "experiment",
                "result",
                "ablation",
                "limitation",
            ]
            if word in text
        )
        if score and isinstance(page_no, int):
            keyword_pages.append((score, page_no))
    for _score, page_no in sorted(keyword_pages, key=lambda item: (-item[0], item[1])):
        add(page_no)
        if len(selected) >= max_pages:
            break
    if not selected:
        for artifact in all_artifacts[:max_pages]:
            add(getattr(artifact, "page_no", None))
    return selected[:max_pages]


def _bounded_env_int(name: str, *, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def _normalize_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
