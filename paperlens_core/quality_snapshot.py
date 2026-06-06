from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from paperlens_core.audit.suite import FACT_NODE_KINDS
from paperlens_core.events import write_json
from paperlens_core.graph import ClaimGraph
from paperlens_core.runtime import ArtifactEnvelope


CORE_QUALITY_SNAPSHOT_SCHEMA_VERSION = "paperlens.core_quality_snapshot.v1"
CORE_QUALITY_SNAPSHOT_ARTIFACT = "core_quality_snapshot"


def write_core_quality_snapshot(output_dir: Path) -> Path:
    data = build_core_quality_snapshot(output_dir)
    path = paperlens_data_dir(output_dir) / "core_quality_snapshot.v1.json"
    envelope = ArtifactEnvelope(
        artifact_type=CORE_QUALITY_SNAPSHOT_ARTIFACT,
        artifact_version="v1",
        producer="paperlens_core_quality_snapshot",
        data=data,
    )
    write_json(path, json.loads(envelope.model_dump_json()))
    return path


def build_core_quality_snapshot(output_dir: Path) -> dict[str, Any]:
    data_dir = paperlens_data_dir(output_dir)
    core_root = data_dir / "core" / "v2"
    qa_rows = read_jsonl(data_dir / "qa_trace.jsonl")
    event_rows = read_jsonl(data_dir / "events.jsonl")
    papers = []
    if core_root.exists():
        for paper_root in sorted(core_root.iterdir()):
            if paper_root.is_dir():
                papers.append(build_paper_quality_snapshot(paper_root, qa_rows=qa_rows))
    aggregate = aggregate_quality(papers, qa_rows=qa_rows, event_rows=event_rows)
    return {
        "schema_version": CORE_QUALITY_SNAPSHOT_SCHEMA_VERSION,
        "paper_count": len(papers),
        "aggregate": aggregate,
        "papers": papers,
    }


def build_paper_quality_snapshot(
    paper_root: Path,
    *,
    qa_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    paper_id = paper_root.name
    graph = read_claim_graph(paper_root / "claim_graph.v1.json")
    core_metrics = read_envelope_dict(
        paper_root / "quality_metrics.v1.json",
        expected_type="core_quality_metrics",
    )
    audit_findings = read_envelope_list(
        paper_root / "audit_findings.v1.json",
        expected_type="audit_findings",
    )
    report_findings = read_envelope_list(
        paper_root / "report_audit_findings.v1.json",
        expected_type="report_audit_findings",
    )
    report_draft = read_envelope_dict(
        paper_root / "report_draft.v1.json",
        expected_type="graph_report_draft",
    )

    fact_node_count = count_fact_nodes(graph)
    numeric_fact_node_count = count_numeric_fact_nodes(graph)
    number_not_located_count = count_findings(audit_findings, "number_not_located_in_source")
    unsupported_fact_node_count = count_findings(audit_findings, "unsupported_fact_node")
    missing_source_count = count_findings(audit_findings, "missing_dom_source")
    report_paragraph_count = count_report_paragraphs(report_draft)
    unsupported_report_paragraph_count = count_report_unsupported_paragraphs(report_findings)
    qa_metrics = qa_metrics_for_paper(qa_rows, paper_id)

    return {
        "paper_id": paper_id,
        "publish_status": core_metrics.get("publish_status"),
        "fact_node_count": fact_node_count,
        "supported_fact_node_count": int_value(core_metrics.get("supported_fact_node_count")),
        "evidence_coverage": float_or_none(core_metrics.get("evidence_coverage")),
        "numeric_fact_node_count": numeric_fact_node_count,
        "number_not_located_count": number_not_located_count,
        "numeric_locatable_rate": rate(
            numeric_fact_node_count - number_not_located_count,
            numeric_fact_node_count,
            default=1.0,
        ),
        "extracted_number_count": int_value(core_metrics.get("extracted_number_count")),
        "extracted_number_not_located_count": int_value(
            core_metrics.get("extracted_number_not_located_count")
        ),
        "extracted_number_locatable_rate": float_or_none(
            core_metrics.get("extracted_number_locatable_rate")
        ),
        "unsupported_fact_node_count": unsupported_fact_node_count,
        "unsupported_fact_node_rate": rate(
            unsupported_fact_node_count,
            fact_node_count,
            default=0.0,
        ),
        "reading_required_output_count": int_value(
            core_metrics.get("reading_required_output_count")
        ),
        "reading_required_output_covered_count": int_value(
            core_metrics.get("reading_required_output_covered_count")
        ),
        "reading_required_output_coverage": float_or_none(
            core_metrics.get("reading_required_output_coverage")
        ),
        "missing_reading_required_output_count": len(
            list_value(core_metrics.get("missing_reading_required_outputs"))
        ),
        "missing_source_count": missing_source_count,
        "audit_error_count": int_value(core_metrics.get("audit_error_count")),
        "audit_warning_count": int_value(core_metrics.get("audit_warning_count")),
        "report_paragraph_count": report_paragraph_count,
        "unsupported_report_paragraph_count": unsupported_report_paragraph_count,
        "unsupported_report_paragraph_rate": rate(
            unsupported_report_paragraph_count,
            report_paragraph_count,
            default=0.0,
        ),
        "qa": qa_metrics,
    }


def aggregate_quality(
    papers: list[dict[str, Any]],
    *,
    qa_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_qa = len(qa_rows)
    qa_cache_hits = sum(1 for row in qa_rows if row.get("cache_hit") is True)
    qa_graph_hits = sum(1 for row in qa_rows if qa_row_has_graph_hit(row))
    qa_graph_context_selected = sum(1 for row in qa_rows if qa_row_selected_graph_context(row))
    cache_event_count = sum(1 for row in event_rows if row.get("event") == "cache_hit")
    blocked = [paper for paper in papers if paper.get("publish_status") == "BLOCKED"]
    draft_weak = [paper for paper in papers if paper.get("publish_status") == "DRAFT_WEAK"]
    return {
        "paper_count": len(papers),
        "blocked_paper_count": len(blocked),
        "draft_weak_paper_count": len(draft_weak),
        "average_evidence_coverage": average_metric(papers, "evidence_coverage"),
        "average_numeric_locatable_rate": average_metric(papers, "numeric_locatable_rate"),
        "average_extracted_number_locatable_rate": average_metric(
            papers, "extracted_number_locatable_rate"
        ),
        "average_unsupported_fact_node_rate": average_metric(papers, "unsupported_fact_node_rate"),
        "average_reading_required_output_coverage": average_metric(
            papers, "reading_required_output_coverage"
        ),
        "average_unsupported_report_paragraph_rate": average_metric(
            papers, "unsupported_report_paragraph_rate"
        ),
        "qa_total": total_qa,
        "qa_graph_hit_count": qa_graph_hits,
        "qa_graph_hit_rate": rate(qa_graph_hits, total_qa, default=0.0),
        "qa_graph_context_selected_count": qa_graph_context_selected,
        "qa_graph_context_selected_rate": rate(
            qa_graph_context_selected, total_qa, default=0.0
        ),
        "qa_cache_hit_count": qa_cache_hits,
        "qa_cache_hit_rate": rate(qa_cache_hits, total_qa, default=0.0),
        "cache_hit_event_count": cache_event_count,
    }


def qa_metrics_for_paper(rows: list[dict[str, Any]], paper_id: str) -> dict[str, Any]:
    paper_rows = [row for row in rows if row.get("paper_id") == paper_id]
    total = len(paper_rows)
    graph_hits = sum(1 for row in paper_rows if qa_row_has_graph_hit(row))
    graph_context_selected = sum(1 for row in paper_rows if qa_row_selected_graph_context(row))
    source_cited = sum(1 for row in paper_rows if row.get("cited_source_ids"))
    cache_hits = sum(1 for row in paper_rows if row.get("cache_hit") is True)
    return {
        "total": total,
        "graph_hit_count": graph_hits,
        "graph_hit_rate": rate(graph_hits, total, default=0.0),
        "graph_context_selected_count": graph_context_selected,
        "graph_context_selected_rate": rate(graph_context_selected, total, default=0.0),
        "source_cited_count": source_cited,
        "source_cited_rate": rate(source_cited, total, default=0.0),
        "cache_hit_count": cache_hits,
        "cache_hit_rate": rate(cache_hits, total, default=0.0),
    }


def qa_row_has_graph_hit(row: dict[str, Any]) -> bool:
    return bool(row.get("cited_source_ids") and row.get("selected_graph_nodes"))


def qa_row_selected_graph_context(row: dict[str, Any]) -> bool:
    return bool(row.get("selected_graph_nodes"))


def read_claim_graph(path: Path) -> ClaimGraph | None:
    data = read_envelope_dict(path, expected_type="claim_graph")
    if not data:
        return None
    return ClaimGraph.model_validate(data)


def read_envelope_dict(path: Path, *, expected_type: str) -> dict[str, Any]:
    data = read_envelope_data(path, expected_type=expected_type)
    return data if isinstance(data, dict) else {}


def read_envelope_list(path: Path, *, expected_type: str) -> list[dict[str, Any]]:
    data = read_envelope_data(path, expected_type=expected_type)
    return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []


def read_envelope_data(path: Path, *, expected_type: str) -> dict[str, Any] | list[Any]:
    if not path.exists():
        return {}
    envelope = ArtifactEnvelope.model_validate_json(path.read_text(encoding="utf-8"))
    return envelope.require_type(expected_type).data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def count_fact_nodes(graph: ClaimGraph | None) -> int:
    if graph is None:
        return 0
    return sum(1 for node in graph.nodes.values() if node.kind in FACT_NODE_KINDS)


def count_numeric_fact_nodes(graph: ClaimGraph | None) -> int:
    if graph is None:
        return 0
    return sum(
        1
        for node in graph.nodes.values()
        if node.kind in FACT_NODE_KINDS and re.search(r"\d", node.label)
    )


def count_findings(findings: list[dict[str, Any]], code: str) -> int:
    return sum(1 for finding in findings if finding.get("code") == code)


def count_report_paragraphs(report_draft: dict[str, Any]) -> int:
    total = 0
    for section in (
        report_draft.get("sections") if isinstance(report_draft.get("sections"), list) else []
    ):
        if isinstance(section, dict) and isinstance(section.get("paragraphs"), list):
            total += len([item for item in section["paragraphs"] if isinstance(item, dict)])
    return total


def count_report_unsupported_paragraphs(findings: list[dict[str, Any]]) -> int:
    paragraph_ids = set()
    for finding in findings:
        code = str(finding.get("code") or "")
        if not code.startswith("report_paragraph_"):
            continue
        finding_id = str(finding.get("finding_id") or "")
        match = re.match(r"paragraph_[^:]+:([^:]+)", finding_id)
        paragraph_ids.add(match.group(1) if match else finding_id)
    return len(paragraph_ids)


def rate(numerator: int | float, denominator: int | float, *, default: float) -> float:
    if not denominator:
        return default
    return round(float(numerator) / float(denominator), 4)


def average_metric(items: list[dict[str, Any]], key: str) -> float | None:
    values = [float(item[key]) for item in items if isinstance(item.get(key), (int, float))]
    if not values:
        return None
    return round(sum(values) / len(values), 4)


def float_or_none(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def paperlens_data_dir(output_dir: Path) -> Path:
    return output_dir / ".paperlens" / "data"
