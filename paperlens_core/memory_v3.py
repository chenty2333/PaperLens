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


def write_paper_memory_v3_bundle(data_dir: Path, memory: dict[str, Any]) -> list[Path]:
    issues = validate_paper_memory_v3(memory)
    memory.setdefault("audit_trail", {})["validation_issues"] = issues
    paper_id = string_or_empty(memory.get("paper_id")) or "unknown"
    root = data_dir / MEMORY_V3_DIR
    root.mkdir(parents=True, exist_ok=True)
    paths = [
        write_json(root / f"{paper_id}.paper_memory.v3.json", memory),
        write_json(root / f"{paper_id}.memory_audit.json", dict_value(memory.get("audit_trail")).get("memory_audit") or {}),
        write_json(root / f"{paper_id}.report_audit.json", dict_value(memory.get("audit_trail")).get("report_audit") or {}),
        write_jsonl(root / f"{paper_id}.claim_index.jsonl", memory.get("claims")),
        write_jsonl(root / f"{paper_id}.evidence_index.jsonl", memory.get("evidence")),
        write_text(root / f"{paper_id}.inspector.md", render_memory_v3_inspector(memory)),
    ]
    return paths


def write_memory_v3_indexes(data_dir: Path, memories: list[dict[str, Any]]) -> list[Path]:
    root = data_dir / MEMORY_V3_DIR
    root.mkdir(parents=True, exist_ok=True)
    claims = []
    evidence = []
    for memory in memories:
        paper_id = string_or_empty(memory.get("paper_id"))
        for claim in list_payload(memory.get("claims")):
            claims.append({"paper_id": paper_id, **claim})
        for item in list_payload(memory.get("evidence")):
            evidence.append({"paper_id": paper_id, **item})
    return [
        write_jsonl(root / "claim_index.jsonl", claims),
        write_jsonl(root / "evidence_index.jsonl", evidence),
    ]


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


def inspect_paper_memory_v3(
    *,
    output_dir: Path,
    paper_id: str | None = None,
    section: str = "summary",
    claim_id: str | None = None,
) -> str:
    data_dir = resolve_data_dir(output_dir)
    root = data_dir / MEMORY_V3_DIR
    if paper_id is None:
        candidates = sorted(root.glob("*.paper_memory.v3.json"))
        if not candidates:
            raise FileNotFoundError(f"No PaperMemoryV3 files found under {root}")
        paper_id = candidates[0].name.split(".", 1)[0]
    memory = read_paper_memory_v3(output_dir, paper_id)
    if not memory:
        raise FileNotFoundError(f"No PaperMemoryV3 found for paper_id={paper_id}")
    return render_memory_v3_inspector(memory, section=section, claim_id=claim_id)


def render_memory_v3_inspector(
    memory: dict[str, Any],
    *,
    section: str = "summary",
    claim_id: str | None = None,
) -> str:
    title = dict_value(memory.get("metadata")).get("title") or memory.get("paper_id")
    lines = [f"# PaperMemoryV3 Inspector: {title}", ""]
    lines.append(f"- Paper: `{memory.get('paper_id')}`")
    lines.append(f"- Schema: `{memory.get('schema_version')}`")
    lines.append(f"- Grade: `{dict_value(memory.get('reading_context')).get('grade')}`")
    issues = validate_paper_memory_v3(memory)
    lines.append(f"- Validation: {'PASS' if not issues else 'WARN'}")
    if issues:
        lines.extend(["", "## Validation Issues", ""])
        lines.extend(f"- {issue}" for issue in issues[:40])
    if section == "audit":
        return "\n".join(lines + render_audit_section(memory)).rstrip() + "\n"
    if section == "claims" or claim_id:
        return "\n".join(lines + render_claims_section(memory, claim_id=claim_id)).rstrip() + "\n"
    if section == "evidence":
        return "\n".join(lines + render_evidence_section(memory)).rstrip() + "\n"
    if section == "concepts":
        return "\n".join(lines + render_concepts_section(memory)).rstrip() + "\n"
    return "\n".join(
        lines
        + render_summary_section(memory)
        + render_concepts_section(memory)
        + render_claims_section(memory, limit=8)
        + render_evidence_section(memory, limit=8)
        + render_audit_section(memory)
    ).rstrip() + "\n"


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


def render_summary_section(memory: dict[str, Any]) -> list[str]:
    problem = dict_value(memory.get("problem_frame")).get("problem") or ""
    abstractions = list_payload(memory.get("core_abstractions"))
    mechanism = dict_value(memory.get("mechanism")).get("overview") or ""
    evaluation = dict_value(memory.get("evaluation")).get("summary") or ""
    lines = ["", "## Summary", ""]
    if problem:
        lines.append(f"- Problem: {problem}")
    if abstractions:
        lines.append(f"- Core abstraction: {abstractions[0].get('text')}")
    if mechanism:
        lines.append(f"- Mechanism: {mechanism}")
    if evaluation:
        lines.append(f"- Evaluation: {evaluation}")
    return lines


def render_concepts_section(memory: dict[str, Any]) -> list[str]:
    bridge = dict_value(memory.get("conceptual_bridge"))
    terms = list_payload(bridge.get("terms"))
    concepts = list_payload(memory.get("concepts"))
    lines = ["", "## Conceptual Bridge", ""]
    if not bridge and not concepts:
        lines.append("No conceptual bridge.")
        return lines
    lines.append(f"- Needed: `{bool(bridge.get('needed'))}`")
    if string_or_empty(bridge.get("reader_gap")):
        lines.append(f"- Reader gap: {bridge.get('reader_gap')}")
    if string_or_empty(bridge.get("bridge_text")):
        lines.append(f"- Bridge: {bridge.get('bridge_text')}")
    if terms:
        lines.extend(["", "| Term | Provenance | Role | Explanation |", "|---|---|---|---|"])
        for item in terms:
            if not isinstance(item, dict):
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        escape_table(item.get("term")),
                        escape_table(item.get("provenance")),
                        escape_table(item.get("paper_role")),
                        escape_table(item.get("explanation")),
                    ]
                )
                + " |"
            )
    elif concepts:
        lines.extend(["", "| Term | Explanation |", "|---|---|"])
        for item in concepts[:8]:
            if isinstance(item, dict):
                lines.append(
                    f"| {escape_table(item.get('term'))} | {escape_table(item.get('explanation'))} |"
                )
    return lines


def render_claims_section(
    memory: dict[str, Any],
    *,
    claim_id: str | None = None,
    limit: int | None = None,
) -> list[str]:
    claims = list_payload(memory.get("claims"))
    if claim_id:
        claims = [claim for claim in claims if claim.get("id") == claim_id]
    if limit is not None:
        claims = claims[:limit]
    lines = ["", "## Claims", ""]
    if not claims:
        lines.append("No claims.")
        return lines
    lines.extend(["| ID | Type | Provenance | Confidence | Evidence | Text |", "|---|---|---|---|---|---|"])
    for claim in claims:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_table(claim.get("id")),
                    escape_table(claim.get("type")),
                    escape_table(claim.get("provenance")),
                    escape_table(claim.get("confidence")),
                    escape_table(", ".join(normalized_string_list(claim.get("evidence_refs")))),
                    escape_table(claim.get("text")),
                ]
            )
            + " |"
        )
    return lines


def render_evidence_section(memory: dict[str, Any], *, limit: int | None = None) -> list[str]:
    evidence = list_payload(memory.get("evidence"))
    if limit is not None:
        evidence = evidence[:limit]
    lines = ["", "## Evidence", ""]
    if not evidence:
        lines.append("No evidence.")
        return lines
    lines.extend(["| ID | Source | Page | Reliability | Interpretation |", "|---|---|---|---|---|"])
    for item in evidence:
        lines.append(
            "| "
            + " | ".join(
                [
                    escape_table(item.get("id")),
                    escape_table(item.get("source_type")),
                    escape_table(item.get("page")),
                    escape_table(item.get("reliability")),
                    escape_table(item.get("interpretation") or item.get("excerpt_or_caption")),
                ]
            )
            + " |"
        )
    return lines


def render_audit_section(memory: dict[str, Any]) -> list[str]:
    audit = dict_value(memory.get("audit_trail"))
    memory_audit = dict_value(audit.get("memory_audit"))
    report_audit = dict_value(audit.get("report_audit"))
    lines = ["", "## Audit", ""]
    lines.append(f"- Memory audit: `{memory_audit.get('status', 'missing')}`")
    lines.append(f"- Report audit: `{report_audit.get('verdict', 'missing')}`")
    issues = normalized_string_list(audit.get("validation_issues"))
    if issues:
        lines.append(f"- Validation issues: {', '.join(issues[:8])}")
    return lines


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


def write_jsonl(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list_payload(payload)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows), encoding="utf-8")
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
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


def escape_table(value: Any) -> str:
    return string_or_empty(str(value) if value is not None else "").replace("|", "\\|").replace("\n", " ")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
