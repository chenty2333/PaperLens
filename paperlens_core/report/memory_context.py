from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from paperlens_core.core_manifest import inspect_core_v2_artifact_set
from paperlens_core.memory_v3 import (
    dict_value,
    list_payload,
    memory_v3_prompt_view,
    normalized_string_list,
    safe_int,
    string_or_none,
)
from paperlens_core.runtime import read_typed_artifact
from paperlens_core.schemas import PaperCard, PaperRecord, SkimCard

REPORT_MEMORY_CONTEXT_SCHEMA_VERSION = "paperlens.report_memory_context.v1"
CORE_MEMORY_VIEW_SCHEMA_VERSION = "paper_memory.view.v1"


def build_report_memory_context(
    *,
    data_dir: Path,
    paper_id: str,
    paper_memory_v3: dict[str, Any],
) -> dict[str, Any]:
    manifest = inspect_core_v2_artifact_set(data_dir, paper_id)
    if manifest.get("consumable") is True:
        root = data_dir / "core" / "v2" / paper_id
        try:
            envelope = read_typed_artifact(
                root / "paper_memory_view.v1.json",
                expected_type="paper_memory_view",
            )
        except (FileNotFoundError, ValueError):
            envelope = None
        if envelope is not None and isinstance(envelope.data, dict):
            return {
                "schema_version": REPORT_MEMORY_CONTEXT_SCHEMA_VERSION,
                "source_of_truth": "core_v2_paper_memory_view",
                "fallback_policy": (
                    "Use core_memory_view for paper-specific facts. legacy_memory_v3 is "
                    "supplemental context only."
                ),
                "quality": {
                    "artifact_set_status": manifest.get("status"),
                    "publish_status": manifest.get("publish_status"),
                    "consumable": manifest.get("consumable"),
                    "issues": manifest.get("issues", []),
                },
                "core_memory_view": envelope.data,
                "legacy_memory_v3": memory_v3_prompt_view(paper_memory_v3),
            }
    return memory_v3_prompt_view(paper_memory_v3)


def core_memory_view_dict(memory: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    if memory.get("schema_version") == CORE_MEMORY_VIEW_SCHEMA_VERSION:
        return memory
    nested = dict_value(memory.get("core_memory_view"))
    return nested if nested.get("schema_version") == CORE_MEMORY_VIEW_SCHEMA_VERSION else {}


def report_focus_queries(
    memory: dict[str, Any], *, paper: PaperRecord, skim: SkimCard | None, card: PaperCard | None
) -> list[str]:
    queries = [paper.canonical_title or paper.paper_id]
    core = core_memory_view_dict(memory)
    if core:
        queries.extend(
            string_or_none(node.get("label")) or ""
            for node in list_payload(core.get("fact_nodes"))[:8]
            if isinstance(node, dict)
        )
    if skim:
        queries.extend([skim.problem, skim.method_type, skim.evaluation_type])
    if card:
        queries.extend(card.contribution_claims[:2])
        queries.extend(card.mechanisms[:2])
        queries.extend(card.evaluation[:2])
    frame = dict_value(memory.get("problem_frame"))
    queries.extend([frame.get("problem"), frame.get("why_it_matters")])
    for item in list_payload(memory.get("core_abstractions"))[:3]:
        if isinstance(item, dict):
            queries.append(str(item.get("text") or ""))
    mechanism = dict_value(memory.get("mechanism"))
    queries.append(str(mechanism.get("overview") or ""))
    evaluation = dict_value(memory.get("evaluation"))
    queries.append(str(evaluation.get("summary") or ""))
    return compact_string_list(queries, limit=8, max_chars=180)


def report_focus_pages(
    memory: dict[str, Any], *, skim: SkimCard | None, card: PaperCard | None
) -> list[int]:
    pages = []
    core = core_memory_view_dict(memory)
    if core:
        for page in core_memory_pages(core):
            if page not in pages:
                pages.append(page)
    for item in list_payload(memory.get("evidence"))[:12]:
        if isinstance(item, dict):
            page = safe_int(item.get("page"))
            if page and page not in pages:
                pages.append(page)
    if skim:
        for ref in skim.evidence_refs[:4]:
            page = safe_int(getattr(ref, "page_no", None))
            if page and page not in pages:
                pages.append(page)
    if card:
        for ref in card.evidence_refs[:6]:
            page = safe_int(getattr(ref, "page_no", None))
            if page and page not in pages:
                pages.append(page)
    return pages[:10]


def select_focused_report_pages(
    *,
    pages: list[Any],
    paper_memory: dict[str, Any] | None,
    skim: SkimCard | None,
    card: PaperCard | None,
    max_pages: int,
) -> list[dict[str, Any]]:
    dict_pages = [page for page in pages if isinstance(page, dict)]
    by_no = {
        int(page.get("page_no")): page
        for page in dict_pages
        if isinstance(page.get("page_no"), int)
    }
    selected: list[int] = []

    def add(page_no: Any) -> None:
        if not isinstance(page_no, int):
            return
        if page_no not in by_no or page_no in selected:
            return
        selected.append(page_no)

    memory = paper_memory if isinstance(paper_memory, dict) else {}
    core = core_memory_view_dict(memory)
    if core:
        for page_no in core_memory_pages(core):
            add(page_no)
    v3 = (
        dict_value(memory.get("paper_memory_v3"))
        if "paper_memory_v3" in memory
        else dict_value(memory)
    )
    evidence_id_to_page = {
        item.get("id"): item.get("page")
        for item in list_payload(v3.get("evidence"))
        if isinstance(item, dict) and item.get("id")
    }
    for item in list_payload(v3.get("evidence"))[:10]:
        if isinstance(item, dict):
            add(item.get("page"))
    for claim in list_payload(v3.get("claims"))[:8]:
        if not isinstance(claim, dict):
            continue
        for ref in (
            claim.get("evidence_refs") if isinstance(claim.get("evidence_refs"), list) else []
        ):
            if isinstance(ref, int):
                add(ref)
            elif isinstance(ref, str):
                add(evidence_id_to_page.get(ref))
            elif isinstance(ref, dict):
                add(ref.get("page") or ref.get("page_no"))

    for ref in (skim.evidence_refs if skim else [])[:4]:
        add(getattr(ref, "page_no", None))
    for ref in (card.evidence_refs if card else [])[:6]:
        add(getattr(ref, "page_no", None))

    if not selected:
        for page in dict_pages[: min(2, max_pages)]:
            add(page.get("page_no"))
    return [by_no[page_no] for page_no in selected[:max_pages]]


def compact_paper_memory_for_report(memory: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    core = core_memory_view_dict(memory)
    if core:
        result: dict[str, Any] = {
            "source_of_truth": "core_v2_paper_memory_view",
            "core_memory_view": compact_core_memory_view_for_report(core),
        }
        quality = dict_value(memory.get("quality"))
        if quality:
            result["quality"] = quality
        fallback = dict_value(memory.get("legacy_memory_v3"))
        if fallback.get("schema_version") == "paper_memory.v3":
            result["legacy_memory_v3"] = compact_memory_v3_for_report(fallback)
            result["legacy_memory_policy"] = memory.get("fallback_policy")
        return result
    v3 = (
        dict_value(memory.get("paper_memory_v3"))
        if "paper_memory_v3" in memory
        else dict_value(memory)
    )
    if v3.get("schema_version") != "paper_memory.v3":
        return {}
    audit_trail = dict_value(v3.get("audit_trail"))
    return {
        "paper_memory_v3": compact_memory_v3_for_report(v3),
        "memory_audit": compact_memory_audit_for_report(
            dict_value(audit_trail.get("memory_audit"))
        ),
        "report_audit": {
            key: value
            for key, value in dict_value(audit_trail.get("report_audit")).items()
            if key in {"verdict", "safe_usage_note"}
        },
    }


def compact_core_memory_view_for_report(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": memory.get("schema_version"),
        "paper_id": memory.get("paper_id"),
        "metadata": dict_value(memory.get("metadata")),
        "report_readiness": memory.get("report_readiness"),
        "fact_nodes": [
            {
                "node_id": item.get("node_id"),
                "kind": item.get("kind"),
                "label": _compact_text(string_or_none(item.get("label")) or "", max_chars=520),
                "confidence": item.get("confidence"),
                "provenance": item.get("provenance"),
                "evidence_ids": normalized_string_list(item.get("evidence_ids"))[:8],
                "source_ids": normalized_string_list(item.get("source_ids"))[:8],
                "pages": [
                    page
                    for page in (safe_int(value) for value in list_payload(item.get("pages")))
                    if page
                ][:8],
                "extracted_numbers": list_payload(item.get("extracted_numbers"))[:8],
            }
            for item in list_payload(memory.get("fact_nodes"))[:16]
            if isinstance(item, dict)
        ],
        "evaluation_matrix": [
            {
                "node_id": item.get("node_id"),
                "kind": item.get("kind"),
                "label": _compact_text(string_or_none(item.get("label")) or "", max_chars=420),
                "source_ids": normalized_string_list(item.get("source_ids"))[:8],
                "pages": [
                    page
                    for page in (safe_int(value) for value in list_payload(item.get("pages")))
                    if page
                ][:8],
                "extracted_numbers": list_payload(item.get("extracted_numbers"))[:8],
            }
            for item in list_payload(memory.get("evaluation_matrix"))[:10]
            if isinstance(item, dict)
        ],
        "evidence_sources": compact_core_evidence_sources(memory.get("evidence_sources")),
        "relationship_edges": list_payload(memory.get("relationship_edges"))[:12],
        "unresolved_audit_findings": normalized_string_list(
            memory.get("unresolved_audit_findings")
        )[:12],
    }


def compact_core_evidence_sources(value: Any) -> list[dict[str, Any]]:
    raw_sources = value.values() if isinstance(value, dict) else []
    result = []
    for item in raw_sources:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "source_id": item.get("source_id"),
                "kind": item.get("kind"),
                "page_no": item.get("page_no"),
                "section_id": item.get("section_id"),
                "excerpt": _compact_text(
                    string_or_none(item.get("excerpt")) or "", max_chars=320
                ),
            }
        )
        if len(result) >= 16:
            break
    return result


def core_memory_pages(core: dict[str, Any]) -> list[int]:
    pages: list[int] = []
    for node in [
        *list_payload(core.get("fact_nodes")),
        *list_payload(core.get("evaluation_matrix")),
    ]:
        for value in list_payload(node.get("pages")):
            page = safe_int(value)
            if page and page not in pages:
                pages.append(page)
    sources = dict_value(core.get("evidence_sources"))
    for source in sources.values():
        if isinstance(source, dict):
            page = safe_int(source.get("page_no"))
            if page and page not in pages:
                pages.append(page)
    return pages[:16]


def compact_memory_audit_for_report(audit: dict[str, Any]) -> dict[str, Any]:
    if not audit:
        return {}
    return {
        "status": audit.get("status"),
        "missing_items": compact_string_list(audit.get("missing_items"), limit=2, max_chars=120),
        "unsupported_claims": compact_string_list(
            audit.get("unsupported_claims"), limit=2, max_chars=120
        ),
        "repair_instructions": compact_string_list(
            audit.get("repair_instructions"), limit=2, max_chars=140
        ),
        "safe_to_generate_capsule": audit.get("safe_to_generate_capsule"),
        "confidence": audit.get("confidence"),
    }


def compact_memory_v3_for_report(memory: dict[str, Any]) -> dict[str, Any]:
    mechanism = dict_value(memory.get("mechanism"))
    evaluation = dict_value(memory.get("evaluation"))
    implementation = dict_value(memory.get("implementation_details"))
    reading_context = dict_value(memory.get("reading_context"))
    return {
        "reading_context": {
            "grade": reading_context.get("grade"),
            "default_view": reading_context.get("default_view"),
            "read_depth": reading_context.get("read_depth"),
            "pages_read": reading_context.get("pages_read"),
        },
        "problem_frame": {
            "problem": _compact_text(
                string_or_none(dict_value(memory.get("problem_frame")).get("problem")) or "",
                max_chars=420,
            ),
            "why_it_matters": _compact_text(
                string_or_none(dict_value(memory.get("problem_frame")).get("why_it_matters"))
                or "",
                max_chars=420,
            ),
            "scope": dict_value(memory.get("problem_frame")).get("scope"),
        },
        "core_abstractions": [
            {
                "id": item.get("id"),
                "text": _compact_text(string_or_none(item.get("text")) or "", max_chars=420),
                "kind": item.get("kind"),
                "misunderstanding_guard": _compact_text(
                    string_or_none(item.get("misunderstanding_guard")) or "",
                    max_chars=260,
                ),
                "evidence_refs": normalized_string_list(item.get("evidence_refs"))[:6],
            }
            for item in list_payload(memory.get("core_abstractions"))[:3]
            if isinstance(item, dict)
        ],
        "mechanism_overview": _compact_text(
            string_or_none(mechanism.get("overview")) or "", max_chars=650
        ),
        "mechanism_steps": [
            {
                "id": item.get("id"),
                "text": _compact_text(string_or_none(item.get("text")) or "", max_chars=360),
            }
            for item in list_payload(mechanism.get("steps"))[:10]
            if isinstance(item, dict)
        ],
        "implementation_components": [
            {
                "name": item.get("name") or item.get("component") or item.get("id"),
                "role": _compact_text(
                    string_or_none(item.get("role") or item.get("text") or item.get("description"))
                    or "",
                    max_chars=280,
                ),
            }
            for item in list_payload(implementation.get("components"))[:8]
            if isinstance(item, dict)
        ],
        "evaluation_summary": _compact_text(
            string_or_none(evaluation.get("summary")) or "", max_chars=520
        ),
        "evaluation_items": [
            {
                "id": item.get("id"),
                "text": _compact_text(string_or_none(item.get("text")) or "", max_chars=320),
            }
            for item in list_payload(evaluation.get("items"))[:8]
            if isinstance(item, dict)
        ],
        "claims_tested": normalized_string_list(evaluation.get("claims_tested"))[:10],
        "conceptual_bridge": compact_conceptual_bridge(memory.get("conceptual_bridge")),
        "concepts": [
            {
                "term": _compact_text(string_or_none(item.get("term")) or "", max_chars=100),
                "explanation": _compact_text(
                    string_or_none(item.get("explanation")) or "", max_chars=260
                ),
            }
            for item in list_payload(memory.get("concepts"))[:10]
            if isinstance(item, dict)
        ],
        "claims": [
            {
                "id": item.get("id"),
                "text": _compact_text(string_or_none(item.get("text")) or "", max_chars=360),
                "type": item.get("type"),
                "provenance": item.get("provenance"),
                "confidence": item.get("confidence"),
                "critic_status": item.get("critic_status"),
                "risk_tags": normalized_string_list(item.get("risk_tags"))[:6],
                "evidence_refs": normalized_string_list(item.get("evidence_refs"))[:8],
                "depends_on": normalized_string_list(item.get("depends_on"))[:6],
            }
            for item in list_payload(memory.get("claims"))[:16]
            if isinstance(item, dict)
        ],
        "evidence": [
            {
                "id": item.get("id"),
                "source_type": item.get("source_type"),
                "page": item.get("page"),
                "section": item.get("section"),
                "excerpt_or_caption": _compact_text(
                    string_or_none(item.get("excerpt_or_caption")) or "", max_chars=360
                ),
                "interpretation": _compact_text(
                    string_or_none(item.get("interpretation")) or "", max_chars=260
                ),
                "reliability": item.get("reliability"),
            }
            for item in list_payload(memory.get("evidence"))[:18]
            if isinstance(item, dict)
        ],
        "figures_tables": [
            {
                "id": item.get("id"),
                "source_type": item.get("source_type"),
                "page": item.get("page"),
                "caption": _compact_text(
                    string_or_none(item.get("caption")) or "", max_chars=320
                ),
            }
            for item in list_payload(memory.get("figures_tables"))[:12]
            if isinstance(item, dict)
        ],
        "limitations": compact_string_list(memory.get("limitations"), limit=8, max_chars=300),
        "relations": [
            {
                "type": item.get("type") or item.get("relation"),
                "target": _compact_text(
                    string_or_none(item.get("target") or item.get("paper") or item.get("concept"))
                    or "",
                    max_chars=180,
                ),
                "description": _compact_text(
                    string_or_none(item.get("description") or item.get("text")) or "",
                    max_chars=260,
                ),
            }
            for item in list_payload(memory.get("relations"))[:8]
            if isinstance(item, dict)
        ],
        "open_questions": compact_string_list(memory.get("open_questions"), limit=6, max_chars=260),
    }


def compact_conceptual_bridge(value: Any) -> dict[str, Any]:
    bridge = dict_value(value)
    terms = []
    for item in list_payload(bridge.get("terms"))[:5]:
        if not isinstance(item, dict):
            continue
        term = _compact_text(string_or_none(item.get("term")) or "", max_chars=80)
        explanation = _compact_text(string_or_none(item.get("explanation")) or "", max_chars=150)
        if not term or not explanation:
            continue
        terms.append(
            {
                "term": term,
                "explanation": explanation,
                "paper_role": _compact_text(
                    string_or_none(item.get("paper_role")) or "", max_chars=130
                ),
                "provenance": item.get("provenance"),
            }
        )
    return {
        "needed": bool(bridge.get("needed") or terms or string_or_none(bridge.get("bridge_text"))),
        "reader_gap": _compact_text(string_or_none(bridge.get("reader_gap")) or "", max_chars=160),
        "bridge_text": _compact_text(
            string_or_none(bridge.get("bridge_text")) or "", max_chars=320
        ),
        "terms": terms,
    }


def compact_string_list(value: Any, *, limit: int, max_chars: int) -> list[str]:
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
