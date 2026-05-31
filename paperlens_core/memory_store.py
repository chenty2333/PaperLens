from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paperlens_core.memory_v3 import (
    CLAIM_CONFIDENCE,
    CLAIM_PROVENANCE,
    CLAIM_STATUS,
    CLAIM_TYPES,
    MEMORY_V3_DIR,
    MEMORY_V3_SCHEMA_VERSION,
    dict_value,
    extract_figures_tables,
    infer_misunderstanding_guard,
    list_payload,
    read_paper_memory_v3,
    safe_int,
    validate_paper_memory_v3,
    write_paper_memory_v3_bundle,
)
from paperlens_core.schemas import ClassificationDecision, PaperCard, PaperRecord, SkimCard


MEMORY_PATCH_SCHEMA_VERSION = "paper_memory.patch.v1"


MEMORY_PATCH_SET_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["paper_id", "operations"],
    "properties": {
        "paper_id": {"type": "string"},
        "rationale": {"type": "string"},
        "operations": {
            "type": "array",
            "maxItems": 32,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["op", "payload"],
                "properties": {
                    "op": {
                        "type": "string",
                        "enum": [
                            "add_read_pages",
                            "set_problem_frame",
                            "set_core_abstraction",
                            "set_mechanism_overview",
                            "upsert_mechanism_step",
                            "set_evaluation_summary",
                            "upsert_evaluation_item",
                            "upsert_concept",
                            "set_conceptual_bridge",
                            "upsert_conceptual_bridge_term",
                            "upsert_evidence",
                            "upsert_claim",
                            "link_claim_evidence",
                            "mark_claim_disputed",
                            "add_limitation",
                            "add_open_question",
                            "set_memory_audit",
                            "set_report_audit",
                            "add_partial_read_failure",
                            "add_user_override",
                        ],
                    },
                    "payload": {"type": "object"},
                },
            },
        },
    },
}


class PaperMemoryStore:
    """Versioned PaperMemoryV3 store with a durable patch log.

    The store is the runtime boundary between model outputs and user-facing
    products. Model calls produce MemoryPatchSet objects; PaperMemoryStore is
    the only component allowed to mutate the materialized PaperMemoryV3 state.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.root = data_dir / MEMORY_V3_DIR

    def read(self, paper_id: str) -> dict[str, Any]:
        memory = read_paper_memory_v3(self.data_dir, paper_id)
        return memory if memory.get("schema_version") == MEMORY_V3_SCHEMA_VERSION else {}

    def initialize(
        self,
        *,
        paper: PaperRecord,
        skim: SkimCard | None,
        decision: ClassificationDecision | None,
        card: PaperCard | None,
        layout: dict[str, Any] | None,
        source: str,
        prefer_existing: bool = True,
    ) -> dict[str, Any]:
        existing = self.read(paper.paper_id) if prefer_existing else {}
        if existing:
            return existing
        memory = initial_memory_v3(paper=paper, skim=skim, decision=decision, card=card, layout=layout)
        self.write(memory)
        self.append_patch(
            paper.paper_id,
            operation="initialize_memory",
            source=source,
            payload=memory_fingerprint(memory),
        )
        return memory

    def write(self, memory: dict[str, Any]) -> list[Path]:
        return write_paper_memory_v3_bundle(self.data_dir, memory)

    def apply_patch_set(
        self,
        paper_id: str,
        patch_set: dict[str, Any],
        *,
        source: str,
    ) -> dict[str, Any]:
        memory = self.read(paper_id)
        if not memory:
            raise FileNotFoundError(f"No PaperMemoryV3 found for paper_id={paper_id}")
        normalized = normalize_memory_patch_set(patch_set, paper_id=paper_id)
        next_memory = apply_memory_patch_set(memory, normalized)
        next_memory.setdefault("audit_trail", {})["validation_issues"] = validate_paper_memory_v3(
            next_memory
        )
        self.write(next_memory)
        self.append_patch(
            paper_id,
            operation="apply_patch_set",
            source=source,
            payload=normalized,
            memory_hash_after=hash_json(next_memory),
        )
        return next_memory

    def apply_patch(self, paper_id: str, patch: dict[str, Any], *, source: str = "manual") -> dict[str, Any]:
        op = str(patch.get("op") or patch.get("operation") or "")
        payload = dict_value(patch.get("payload"))
        return self.apply_patch_set(
            paper_id,
            {"paper_id": paper_id, "operations": [{"op": op, "payload": payload}]},
            source=source,
        )

    def append_patch(
        self,
        paper_id: str,
        *,
        operation: str,
        source: str,
        payload: dict[str, Any] | None = None,
        memory_hash_after: str | None = None,
    ) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        row = {
            "schema_version": MEMORY_PATCH_SCHEMA_VERSION,
            "paper_id": paper_id,
            "created_at": now_iso(),
            "operation": operation,
            "source": source,
            "payload": payload or {},
            "memory_hash_after": memory_hash_after or hash_json(self.read(paper_id)),
        }
        path = self.patch_log_path(paper_id)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")
        return path

    def patch_log_path(self, paper_id: str) -> Path:
        return self.root / f"{paper_id}.memory_patches.jsonl"


def apply_memory_patch(memory: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    operation = str(patch.get("op") or patch.get("operation") or "")
    payload = dict_value(patch.get("payload"))
    result = json.loads(json.dumps(memory, ensure_ascii=False, default=str))
    if operation == "add_read_pages":
        add_read_pages(result, payload)
    elif operation == "set_problem_frame":
        set_problem_frame(result, payload)
    elif operation == "set_core_abstraction":
        set_core_abstraction(result, payload)
    elif operation == "set_mechanism_overview":
        result.setdefault("mechanism", {})["overview"] = none_if_empty(payload.get("overview") or payload.get("text")) or ""
    elif operation == "upsert_mechanism_step":
        upsert_numbered_text(
            result.setdefault("mechanism", {}).setdefault("steps", []),
            payload,
            prefix="M",
            limit=32,
        )
    elif operation == "set_evaluation_summary":
        result.setdefault("evaluation", {})["summary"] = none_if_empty(payload.get("summary") or payload.get("text")) or ""
    elif operation == "upsert_evaluation_item":
        upsert_numbered_text(
            result.setdefault("evaluation", {}).setdefault("items", []),
            payload,
            prefix="V",
            limit=32,
        )
    elif operation == "set_memory_audit":
        result.setdefault("audit_trail", {})["memory_audit"] = payload
    elif operation == "set_report_audit":
        result.setdefault("audit_trail", {})["report_audit"] = payload
    elif operation == "add_open_question":
        add_string(result.setdefault("open_questions", []), payload.get("text"), limit=16)
    elif operation == "upsert_concept" or operation == "add_concept":
        upsert_by_text(
            result.setdefault("concepts", []),
            {
                "term": str(payload.get("term") or "").strip(),
                "explanation": str(payload.get("explanation") or "").strip(),
            },
            key="term",
            limit=16,
        )
    elif operation == "set_conceptual_bridge":
        set_conceptual_bridge(result, payload)
    elif operation == "upsert_conceptual_bridge_term":
        upsert_bridge_term(result, payload)
    elif operation == "upsert_evidence":
        evidence = result.setdefault("evidence", [])
        item = normalize_patch_evidence(payload, evidence)
        upsert_by_text(evidence, item, key="id", limit=64)
    elif operation == "upsert_claim":
        claims = result.setdefault("claims", [])
        item = normalize_patch_claim(payload, claims)
        upsert_by_text(claims, item, key="id", limit=64)
    elif operation == "link_claim_evidence":
        link_claim_evidence(result, payload)
    elif operation == "mark_claim_disputed":
        mark_claim_disputed(result.setdefault("claims", []), payload)
    elif operation == "add_limitation":
        add_string(result.setdefault("limitations", []), payload.get("text"), limit=16)
    elif operation == "add_partial_read_failure":
        result.setdefault("audit_trail", {}).setdefault("partial_read_failures", []).append(payload)
    elif operation == "add_user_override":
        result.setdefault("user_overrides", []).append(payload)
    return result


def apply_memory_patch_set(memory: dict[str, Any], patch_set: dict[str, Any]) -> dict[str, Any]:
    result = memory
    operations = patch_set.get("operations") if isinstance(patch_set.get("operations"), list) else []
    for operation in operations:
        if isinstance(operation, dict):
            result = apply_memory_patch(result, operation)
    refresh_derived_evaluation_claims(result)
    return result


def normalize_memory_patch_set(patch_set: dict[str, Any], *, paper_id: str) -> dict[str, Any]:
    operations = []
    raw_operations = patch_set.get("operations") if isinstance(patch_set.get("operations"), list) else []
    for item in raw_operations:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op") or item.get("operation") or "").strip()
        payload = dict_value(item.get("payload"))
        if op not in MEMORY_PATCH_OPERATIONS:
            continue
        operations.append({"op": op, "payload": payload})
        if len(operations) >= 32:
            break
    return {
        "schema_version": MEMORY_PATCH_SCHEMA_VERSION,
        "paper_id": str(patch_set.get("paper_id") or paper_id),
        "rationale": str(patch_set.get("rationale") or "").strip(),
        "operations": operations,
    }


MEMORY_PATCH_OPERATIONS = set(MEMORY_PATCH_SET_SCHEMA["properties"]["operations"]["items"]["properties"]["op"]["enum"])


def initial_memory_v3(
    *,
    paper: PaperRecord,
    skim: SkimCard | None,
    decision: ClassificationDecision | None,
    card: PaperCard | None,
    layout: dict[str, Any] | None,
) -> dict[str, Any]:
    grade = decision.class_label if decision else "HOLD"
    evidence = initial_evidence_items(skim=skim, card=card)
    claims = initial_claim_items(paper=paper, skim=skim, card=card, evidence=evidence)
    mechanism_steps = [
        text
        for text in [
            skim.method_type if skim else None,
            skim.system_scope if skim else None,
            *(card.mechanisms if card else []),
        ]
        if text and text != "unknown"
    ][:6]
    evaluation_items = [
        text
        for text in [skim.evaluation_type if skim else None, *(card.evaluation if card else [])]
        if text and text != "unknown"
    ][:6]
    title = paper.canonical_title or paper.paper_id
    problem = (skim.problem if skim else None) or title
    return {
        "schema_version": MEMORY_V3_SCHEMA_VERSION,
        "paper_id": paper.paper_id,
        "metadata": {
            "title": title,
            "authors": list(paper.authors or []),
            "year": paper.year,
            "venue": paper.venue,
            "doi": paper.doi,
            "arxiv_id": paper.arxiv_id,
            "pdf_sha256": paper.file_hash,
            "original_path": paper.file_path,
            "pages": paper.page_count,
        },
        "reading_context": {
            "grade": grade,
            "grade_controls": "reading_investment_not_truthfulness",
            "default_view": default_report_view(grade),
            "read_depth": default_read_depth(grade),
            "pages_read": [],
            "parse_quality": paper.parse_quality,
        },
        "problem_frame": {
            "problem": problem,
            "why_it_matters": problem,
            "scope": (skim.system_scope if skim else None) or "",
        },
        "core_abstractions": [
            {
                "id": "A001",
                "text": problem,
                "kind": "seed",
                "evidence_refs": first_claim_evidence_refs(claims),
                "misunderstanding_guard": infer_misunderstanding_guard(problem),
            }
        ],
        "mechanism": {
            "overview": join_sentences(mechanism_steps),
            "steps": [
                {"id": f"M{index:03d}", "text": text}
                for index, text in enumerate(mechanism_steps, start=1)
            ],
        },
        "implementation_details": {
            "components": [],
            "assumptions": list(card.assumptions[:6] if card else []),
            "reproduction_hooks": {
                "repo_url_candidates": [],
                "artifact_claims": [],
                "dependencies": [],
                "hardware_assumptions": [],
                "benchmark_commands": [],
                "datasets": [],
                "reproducibility_risks": [],
            },
        },
        "evaluation": {
            "summary": join_sentences(evaluation_items),
            "items": [
                {"id": f"V{index:03d}", "text": text}
                for index, text in enumerate(evaluation_items, start=1)
            ],
            "claims_tested": [claim["id"] for claim in claims if claim.get("type") == "evaluation"],
        },
        "concepts": [],
        "conceptual_bridge": {"needed": False, "reader_gap": "", "bridge_text": "", "terms": []},
        "claims": claims,
        "evidence": evidence,
        "figures_tables": extract_figures_tables(layout),
        "limitations": list(card.limitations[:6] if card else []),
        "relations": [],
        "open_questions": [],
        "audit_trail": {
            "created_at": now_iso(),
            "memory_audit": {},
            "report_audit": {},
            "partial_read_failures": [],
            "validation_issues": [],
        },
        "user_overrides": [],
    }


def initial_evidence_items(*, skim: SkimCard | None, card: PaperCard | None) -> list[dict[str, Any]]:
    items = []
    for ref in [*(skim.evidence_refs if skim else []), *(card.evidence_refs if card else [])]:
        if not any(item.get("page") == ref.page_no for item in items):
            items.append(
                {
                    "id": f"E{len(items) + 1:03d}",
                    "source_type": "figure" if ref.figure_id else "table" if ref.table_id else "text_span",
                    "page": ref.page_no,
                    "section": ref.section,
                    "excerpt_or_caption": None,
                    "visual_region": ref.bbox,
                    "interpretation": "Seed evidence from skim/classification.",
                    "reliability": "indirect",
                }
            )
        if len(items) >= 12:
            break
    return items


def initial_claim_items(
    *,
    paper: PaperRecord,
    skim: SkimCard | None,
    card: PaperCard | None,
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    claims = []

    def add(text: str | None, claim_type: str) -> None:
        text = str(text or "").strip()
        if not text or any(normalize_compare(item.get("text")) == normalize_compare(text) for item in claims):
            return
        claims.append(
            {
                "id": f"C{len(claims) + 1:03d}",
                "text": text,
                "type": claim_type if claim_type in CLAIM_TYPES else "motivation",
                "provenance": "inferred" if not evidence else "explicit",
                "confidence": "medium",
                "evidence_refs": [item["id"] for item in evidence[:4] if item.get("id")],
                "depends_on": [],
                "risk_tags": [] if evidence else ["needs_evidence"],
                "critic_status": "checked" if evidence else "unchecked",
            }
        )

    add(skim.problem if skim else None, "motivation")
    if card:
        for value in card.contribution_claims[:6]:
            add(value, "implication")
    if not claims:
        add(paper.canonical_title or paper.paper_id, "motivation")
    return claims


def normalize_patch_claim(payload: dict[str, Any], existing: list[Any]) -> dict[str, Any]:
    claim_id = str(payload.get("id") or "").strip() or next_id(existing, "C")
    claim_type = str(payload.get("type") or "implication").strip()
    if claim_type not in CLAIM_TYPES:
        claim_type = "implication"
    provenance = str(payload.get("provenance") or "inferred").strip()
    if provenance not in CLAIM_PROVENANCE:
        provenance = "inferred"
    confidence = str(payload.get("confidence") or "medium").strip()
    if confidence not in CLAIM_CONFIDENCE:
        confidence = "medium"
    status = str(payload.get("critic_status") or "unchecked").strip()
    if status not in CLAIM_STATUS:
        status = "unchecked"
    return {
        "id": claim_id,
        "text": str(payload.get("text") or payload.get("claim") or "").strip(),
        "type": claim_type,
        "provenance": provenance,
        "confidence": confidence,
        "evidence_refs": string_list(payload.get("evidence_refs"))[:8],
        "depends_on": string_list(payload.get("depends_on"))[:8],
        "risk_tags": string_list(payload.get("risk_tags"))[:8],
        "critic_status": status,
    }


def normalize_patch_evidence(payload: dict[str, Any], existing: list[Any]) -> dict[str, Any]:
    evidence_id = str(payload.get("id") or "").strip() or next_id(existing, "E")
    page = safe_int(payload.get("page") or payload.get("page_no"))
    reliability = str(payload.get("reliability") or "indirect").strip()
    if reliability not in {"direct", "indirect"}:
        reliability = "indirect"
    return {
        "id": evidence_id,
        "source_type": str(payload.get("source_type") or "text_span").strip(),
        "page": page,
        "section": none_if_empty(payload.get("section")),
        "excerpt_or_caption": none_if_empty(payload.get("excerpt_or_caption") or payload.get("quote")),
        "visual_region": payload.get("visual_region"),
        "interpretation": none_if_empty(payload.get("interpretation") or payload.get("claim")),
        "reliability": reliability,
    }


def add_read_pages(memory: dict[str, Any], payload: dict[str, Any]) -> None:
    context = memory.setdefault("reading_context", {})
    pages = context.setdefault("pages_read", [])
    page_values = payload.get("pages") if isinstance(payload.get("pages"), list) else []
    for value in page_values:
        page = safe_int(value)
        if page and page not in pages:
            pages.append(page)
    pages.sort()


def set_problem_frame(memory: dict[str, Any], payload: dict[str, Any]) -> None:
    frame = memory.setdefault("problem_frame", {})
    for key in ["problem", "why_it_matters", "scope"]:
        value = none_if_empty(payload.get(key))
        if value:
            frame[key] = value


def set_core_abstraction(memory: dict[str, Any], payload: dict[str, Any]) -> None:
    text = none_if_empty(payload.get("text") or payload.get("abstraction"))
    if not text:
        return
    abstractions = memory.setdefault("core_abstractions", [])
    item = {
        "id": str(payload.get("id") or "A001"),
        "text": text,
        "kind": str(payload.get("kind") or "primary"),
        "evidence_refs": string_list(payload.get("evidence_refs"))[:8],
        "misunderstanding_guard": none_if_empty(payload.get("misunderstanding_guard"))
        or infer_misunderstanding_guard(text),
    }
    upsert_by_text(abstractions, item, key="id", limit=8)


def set_conceptual_bridge(memory: dict[str, Any], payload: dict[str, Any]) -> None:
    bridge = memory.setdefault("conceptual_bridge", {})
    if "needed" in payload:
        bridge["needed"] = bool(payload.get("needed"))
    for key in ["reader_gap", "bridge_text"]:
        value = none_if_empty(payload.get(key))
        if value:
            bridge[key] = value
    bridge.setdefault("terms", [])


def upsert_bridge_term(memory: dict[str, Any], payload: dict[str, Any]) -> None:
    bridge = memory.setdefault("conceptual_bridge", {})
    bridge["needed"] = True
    terms = bridge.setdefault("terms", [])
    provenance = str(payload.get("provenance") or "background").strip()
    if provenance not in {"explicit", "inferred", "background"}:
        provenance = "background"
    upsert_by_text(
        terms,
        {
            "term": str(payload.get("term") or "").strip(),
            "explanation": str(payload.get("explanation") or "").strip(),
            "paper_role": str(payload.get("paper_role") or payload.get("role") or "").strip(),
            "provenance": provenance,
        },
        key="term",
        limit=12,
    )


def upsert_numbered_text(items: list[Any], payload: dict[str, Any], *, prefix: str, limit: int) -> None:
    text = none_if_empty(payload.get("text") or payload.get("step") or payload.get("summary"))
    if not text:
        return
    item = {"id": str(payload.get("id") or next_id(items, prefix)), "text": text}
    upsert_by_text(items, item, key="id", limit=limit)


def link_claim_evidence(memory: dict[str, Any], payload: dict[str, Any]) -> None:
    claim_id = str(payload.get("claim_id") or payload.get("id") or "").strip()
    evidence_refs = string_list(payload.get("evidence_refs") or payload.get("evidence_ids"))[:8]
    if not claim_id or not evidence_refs:
        return
    claims = memory.get("claims") if isinstance(memory.get("claims"), list) else []
    for claim in claims:
        if isinstance(claim, dict) and claim.get("id") == claim_id:
            refs = claim.setdefault("evidence_refs", [])
            for ref in evidence_refs:
                if ref not in refs:
                    refs.append(ref)
            if refs and claim.get("critic_status") == "unchecked":
                claim["critic_status"] = "checked"
            return


def refresh_derived_evaluation_claims(memory: dict[str, Any]) -> None:
    evaluation = dict_value(memory.get("evaluation"))
    claim_ids = [
        str(claim.get("id"))
        for claim in (memory.get("claims") if isinstance(memory.get("claims"), list) else [])
        if isinstance(claim, dict) and claim.get("type") == "evaluation" and claim.get("id")
    ]
    evaluation["claims_tested"] = claim_ids
    memory["evaluation"] = evaluation


def mark_claim_disputed(claims: list[Any], payload: dict[str, Any]) -> None:
    target_id = str(payload.get("id") or "").strip()
    target_text = normalize_compare(payload.get("text") or payload.get("claim"))
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        if target_id and claim.get("id") == target_id:
            claim["critic_status"] = "disputed"
            claim.setdefault("risk_tags", []).append("user_challenged")
            return
        if target_text and normalize_compare(claim.get("text")) == target_text:
            claim["critic_status"] = "disputed"
            claim.setdefault("risk_tags", []).append("user_challenged")
            return


def upsert_by_text(items: list[Any], item: dict[str, Any], *, key: str, limit: int) -> None:
    value = str(item.get(key) or "").strip()
    if not value:
        return
    for index, existing in enumerate(items):
        if isinstance(existing, dict) and str(existing.get(key) or "").strip() == value:
            items[index] = {**existing, **{k: v for k, v in item.items() if v not in (None, "", [])}}
            return
    if len(items) < limit:
        items.append(item)


def add_string(items: list[Any], value: Any, *, limit: int) -> None:
    text = str(value or "").strip()
    if text and text not in items and len(items) < limit:
        items.append(text)


def next_id(items: list[Any], prefix: str) -> str:
    max_seen = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        value = str(item.get("id") or "")
        if value.startswith(prefix) and value[len(prefix) :].isdigit():
            max_seen = max(max_seen, int(value[len(prefix) :]))
    return f"{prefix}{max_seen + 1:03d}"


def memory_fingerprint(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": memory.get("schema_version"),
        "claim_count": len(list_payload(memory.get("claims"))),
        "evidence_count": len(list_payload(memory.get("evidence"))),
        "validation_issues": validate_paper_memory_v3(memory)[:12],
        "memory_hash": hash_json(memory),
    }


def hash_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def none_if_empty(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def normalize_compare(value: Any) -> str:
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def first_claim_evidence_refs(claims: list[dict[str, Any]]) -> list[str]:
    for claim in claims:
        refs = string_list(claim.get("evidence_refs"))
        if refs:
            return refs[:4]
    return []


def join_sentences(items: list[str]) -> str:
    return " ".join(item.rstrip(".。") + "。" for item in items if item).strip()


def default_read_depth(grade: str) -> str:
    return {
        "A": "standard_high_value",
        "B": "standard",
        "C": "standard_low_priority",
        "HOLD": "standard_needs_confirmation",
    }.get(grade, "standard")


def default_report_view(grade: str) -> str:
    return {
        "A": "standard_high_value",
        "B": "standard",
        "C": "standard_low_priority",
        "HOLD": "standard_with_confirmation_needed",
    }.get(grade, "standard")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
