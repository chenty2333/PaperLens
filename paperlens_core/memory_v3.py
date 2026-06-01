from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MEMORY_V3_SCHEMA_VERSION = "paper_memory.v3"
MEMORY_V3_DIR = "memory/v3"

CLAIM_PROVENANCE = {"explicit", "inferred", "background", "external"}
CLAIM_CONFIDENCE = {"high", "medium", "low"}
CLAIM_STATUS = {"unchecked", "checked", "repaired", "disputed"}
CLAIM_TYPES = {"motivation", "mechanism", "evaluation", "limitation", "comparison", "implication"}


def validate_paper_memory_v3(memory: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    if memory.get("schema_version") != MEMORY_V3_SCHEMA_VERSION:
        issues.append("unsupported_schema_version")
    paper_id = string_or_empty(memory.get("paper_id"))
    if not paper_id:
        issues.append("missing_paper_id")
    claims = list_payload(memory.get("claims"))
    evidence = list_payload(memory.get("evidence"))
    if not isinstance(memory.get("conceptual_bridge"), dict):
        issues.append("missing_conceptual_bridge")
    bridge = dict_value(memory.get("conceptual_bridge"))
    evidence_ids = {string_or_empty(item.get("id")) for item in evidence if string_or_empty(item.get("id"))}
    if not claims:
        issues.append("empty_claims")
    if not evidence:
        issues.append("empty_evidence")
    if bridge:
        if bridge.get("needed") and not string_or_empty(bridge.get("bridge_text")):
            issues.append("conceptual_bridge_missing_text")
        for item in list_payload(bridge.get("terms")):
            term = string_or_empty(item.get("term")) if isinstance(item, dict) else ""
            if not term:
                issues.append("conceptual_bridge_term_missing_term")
                continue
            if item.get("provenance") not in {"explicit", "inferred", "background"}:
                issues.append(f"conceptual_bridge_bad_provenance:{term}")
    seen_claims: set[str] = set()
    for claim in claims:
        claim_id = string_or_empty(claim.get("id"))
        if not claim_id:
            issues.append("claim_missing_id")
        elif claim_id in seen_claims:
            issues.append(f"duplicate_claim_id:{claim_id}")
        seen_claims.add(claim_id)
        if not string_or_empty(claim.get("text")):
            issues.append(f"claim_missing_text:{claim_id or 'unknown'}")
        if claim.get("type") not in CLAIM_TYPES:
            issues.append(f"claim_bad_type:{claim_id or 'unknown'}")
        if claim.get("provenance") not in CLAIM_PROVENANCE:
            issues.append(f"claim_bad_provenance:{claim_id or 'unknown'}")
        if claim.get("confidence") not in CLAIM_CONFIDENCE:
            issues.append(f"claim_bad_confidence:{claim_id or 'unknown'}")
        if claim.get("critic_status") not in CLAIM_STATUS:
            issues.append(f"claim_bad_status:{claim_id or 'unknown'}")
        for ref in normalized_string_list(claim.get("evidence_refs")):
            if ref not in evidence_ids:
                issues.append(f"claim_unknown_evidence:{claim_id or 'unknown'}:{ref}")
    seen_evidence: set[str] = set()
    for item in evidence:
        evidence_id = string_or_empty(item.get("id"))
        if not evidence_id:
            issues.append("evidence_missing_id")
        elif evidence_id in seen_evidence:
            issues.append(f"duplicate_evidence_id:{evidence_id}")
        seen_evidence.add(evidence_id)
        if not string_or_empty(item.get("source_type")):
            issues.append(f"evidence_missing_source_type:{evidence_id or 'unknown'}")
    return issues


def write_paper_memory_v3_file(data_dir: Path, memory: dict[str, Any]) -> Path:
    issues = validate_paper_memory_v3(memory)
    memory.setdefault("audit_trail", {})["validation_issues"] = issues
    paper_id = string_or_empty(memory.get("paper_id")) or "unknown"
    root = data_dir / MEMORY_V3_DIR
    root.mkdir(parents=True, exist_ok=True)
    return write_json(root / f"{paper_id}.paper_memory.v3.json", memory)


def read_paper_memory_v3(base_dir: Path, paper_id: str) -> dict[str, Any]:
    data_dir = resolve_data_dir(base_dir)
    path = data_dir / MEMORY_V3_DIR / f"{paper_id}.paper_memory.v3.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def memory_v3_prompt_view(memory: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": memory.get("schema_version"),
        "paper_id": memory.get("paper_id"),
        "reading_context": memory.get("reading_context"),
        "problem_frame": memory.get("problem_frame"),
        "core_abstractions": memory.get("core_abstractions"),
        "mechanism": memory.get("mechanism"),
        "evaluation": memory.get("evaluation"),
        "conceptual_bridge": memory.get("conceptual_bridge"),
        "concepts": memory.get("concepts"),
        "claims": list_payload(memory.get("claims"))[:12],
        "evidence": list_payload(memory.get("evidence"))[:16],
        "limitations": memory.get("limitations"),
        "open_questions": memory.get("open_questions"),
        "audit_trail": memory.get("audit_trail"),
    }


def extract_figures_tables(layout: dict[str, Any] | None) -> list[dict[str, Any]]:
    pages = list_payload(dict_value(layout).get("pages"))
    items: list[dict[str, Any]] = []
    for page in pages:
        page_no = safe_int(page.get("page_no"))
        for key, source_type in [("figures", "figure"), ("tables", "table")]:
            for raw in list_payload(page.get(key))[:4]:
                bbox = raw.get("bbox")
                if bbox and is_full_page_bbox(page, bbox):
                    bbox = None
                items.append(
                    {
                        "id": f"F{len(items) + 1:03d}" if source_type == "figure" else f"T{len(items) + 1:03d}",
                        "source_type": source_type,
                        "page": page_no,
                        "caption": caption_for_page({"pages": [page]}, page_no),
                        "visual_region": bbox,
                    }
                )
        if len(items) >= 16:
            break
    return items


def infer_misunderstanding_guard(text: str) -> str | None:
    normalized = text.lower()
    if "paging" in normalized or "page" in normalized or "分页" in normalized:
        return "Paging analogies should be treated as an abstraction transfer, not as full OS virtual memory equivalence."
    if "cache" in normalized:
        return "Cache analogies should not be extended beyond the paper's stated mechanism and workload."
    return None


def caption_for_page(layout: dict[str, Any] | None, page_no: int | None) -> str | None:
    for page in list_payload(dict_value(layout).get("pages")):
        if safe_int(page.get("page_no")) != page_no:
            continue
        captions = []
        for caption in list_payload(page.get("captions"))[:3]:
            text = string_or_empty(caption.get("text"))
            if text:
                captions.append(text)
        return " ".join(captions)[:400] or None
    return None


def is_full_page_bbox(page: dict[str, Any], bbox: Any) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    width = safe_float(page.get("page_width"))
    height = safe_float(page.get("page_height"))
    if not width or not height:
        return False
    try:
        x0, y0, x1, y1 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return False
    return (x1 - x0) >= width * 0.9 and (y1 - y0) >= height * 0.9


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


def resolve_data_dir(base_dir: Path) -> Path:
    if (base_dir / ".paperlens" / "data").exists():
        return base_dir / ".paperlens" / "data"
    if base_dir.name == "data" and base_dir.parent.name == ".paperlens":
        return base_dir
    return base_dir / ".paperlens" / "data"


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def normalize_concepts(value: Any) -> list[dict[str, str]]:
    result = []
    for item in list_payload(value):
        term = string_or_empty(item.get("term"))
        explanation = string_or_empty(item.get("explanation"))
        if term and explanation and not any(existing["term"].lower() == term.lower() for existing in result):
            result.append({"term": term, "explanation": explanation})
        if len(result) >= 12:
            break
    return result


def normalize_conceptual_bridge(value: Any, concepts: list[dict[str, str]]) -> dict[str, Any]:
    _ = concepts
    bridge = dict_value(value)
    terms = normalize_conceptual_bridge_terms(bridge.get("terms"))
    bridge_text = string_or_empty(bridge.get("bridge_text"))
    reader_gap = string_or_empty(bridge.get("reader_gap"))
    return {
        "needed": bool(bridge.get("needed") or bridge_text or terms),
        "reader_gap": reader_gap,
        "bridge_text": bridge_text,
        "terms": terms[:6],
    }


def normalize_conceptual_bridge_terms(value: Any) -> list[dict[str, str]]:
    result = []
    for item in list_payload(value):
        if not isinstance(item, dict):
            continue
        term = string_or_empty(item.get("term"))
        explanation = string_or_empty(item.get("explanation"))
        if not term or not explanation:
            continue
        provenance = string_or_empty(item.get("provenance")).lower() or "background"
        if provenance not in {"explicit", "inferred", "background"}:
            provenance = "background"
        if any(existing["term"].lower() == term.lower() for existing in result):
            continue
        result.append(
            {
                "term": term,
                "explanation": explanation,
                "paper_role": string_or_empty(item.get("paper_role") or item.get("role")) or explanation,
                "provenance": provenance,
            }
        )
        if len(result) >= 6:
            break
    return result


def first_string(*values: Any) -> str:
    for value in values:
        text = string_or_empty(value)
        if text:
            return text
    return ""


def join_sentences(items: list[str]) -> str:
    return " ".join(item.rstrip(".。") + "。" for item in items if item).strip()


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_payload(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def normalized_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = string_or_empty(item)
        if text and text not in result:
            result.append(text)
    return result


def string_or_empty(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def string_or_none(value: Any) -> str | None:
    text = string_or_empty(value)
    return text or None


def safe_int(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return None
    return result


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def compact_compare_text(value: Any) -> str:
    return re.sub(r"[\W_]+", "", string_or_empty(value).lower())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
