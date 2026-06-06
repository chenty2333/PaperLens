from __future__ import annotations

from pydantic import BaseModel

from paperlens_core.audit.findings import AuditFinding
from paperlens_core.audit.suite import (
    FACT_NODE_KINDS,
    contains_number,
    extracted_number_texts,
    publish_status_from_findings,
)
from paperlens_core.dom.paper_dom import PaperDOM
from paperlens_core.graph.claim_graph import ClaimGraph
from paperlens_core.reading.tasks import ReadingPlan


class CoreQualityMetrics(BaseModel):
    schema_version: str = "paperlens_core_quality.v1"
    paper_id: str
    fact_node_count: int
    supported_fact_node_count: int
    evidence_coverage: float
    numeric_fact_node_count: int
    number_not_located_count: int
    numeric_locatable_rate: float
    extracted_number_count: int
    extracted_number_not_located_count: int
    extracted_number_locatable_rate: float
    unsupported_fact_node_count: int
    unsupported_fact_node_rate: float
    reading_required_output_count: int
    reading_required_output_covered_count: int
    reading_required_output_coverage: float
    missing_reading_required_outputs: list[str]
    missing_source_count: int
    audit_error_count: int
    audit_warning_count: int
    publish_status: str


def compute_core_quality_metrics(
    *,
    dom: PaperDOM,
    graph: ClaimGraph,
    findings: list[AuditFinding],
    reading_plan: ReadingPlan | None = None,
) -> CoreQualityMetrics:
    fact_nodes = [node for node in graph.nodes.values() if node.kind in FACT_NODE_KINDS]
    supported = [node for node in fact_nodes if graph.evidence_ids_for(node.node_id)]
    denominator = len(fact_nodes) or 1
    numeric_fact_nodes = [node for node in fact_nodes if contains_number(node.label)]
    unsupported_fact_nodes = [
        finding for finding in findings if finding.code == "unsupported_fact_node"
    ]
    number_not_located = [
        finding for finding in findings if finding.code == "number_not_located_in_source"
    ]
    extracted_numbers = [
        number
        for node in fact_nodes
        for number in extracted_number_texts(node.payload.get("extracted_numbers"))
    ]
    extracted_numbers_not_located = [
        finding
        for finding in findings
        if finding.code == "extracted_number_not_located_in_source"
    ]
    missing_sources = [finding for finding in findings if finding.code == "missing_dom_source"]
    errors = [finding for finding in findings if finding.severity == "ERROR"]
    warnings = [finding for finding in findings if finding.severity == "WARNING"]
    expected_output_keys = reading_required_output_keys(reading_plan)
    covered_output_keys = graph_covered_required_output_keys(graph)
    missing_output_keys = [
        output_key for output_key in expected_output_keys if output_key not in covered_output_keys
    ]
    return CoreQualityMetrics(
        paper_id=dom.paper_id,
        fact_node_count=len(fact_nodes),
        supported_fact_node_count=len(supported),
        evidence_coverage=round(len(supported) / denominator, 4),
        numeric_fact_node_count=len(numeric_fact_nodes),
        number_not_located_count=len(number_not_located),
        numeric_locatable_rate=metric_rate(
            len(numeric_fact_nodes) - len(number_not_located),
            len(numeric_fact_nodes),
            default=1.0,
        ),
        extracted_number_count=len(extracted_numbers),
        extracted_number_not_located_count=len(extracted_numbers_not_located),
        extracted_number_locatable_rate=metric_rate(
            len(extracted_numbers) - len(extracted_numbers_not_located),
            len(extracted_numbers),
            default=1.0,
        ),
        unsupported_fact_node_count=len(unsupported_fact_nodes),
        unsupported_fact_node_rate=metric_rate(
            len(unsupported_fact_nodes),
            len(fact_nodes),
            default=0.0,
        ),
        reading_required_output_count=len(expected_output_keys),
        reading_required_output_covered_count=len(expected_output_keys) - len(missing_output_keys),
        reading_required_output_coverage=metric_rate(
            len(expected_output_keys) - len(missing_output_keys),
            len(expected_output_keys),
            default=1.0,
        ),
        missing_reading_required_outputs=missing_output_keys,
        missing_source_count=len(missing_sources),
        audit_error_count=len(errors),
        audit_warning_count=len(warnings),
        publish_status=publish_status_from_findings(findings).value,
    )


def metric_rate(numerator: int, denominator: int, *, default: float) -> float:
    if not denominator:
        return default
    return round(numerator / denominator, 4)


def reading_required_output_keys(reading_plan: ReadingPlan | None) -> list[str]:
    if reading_plan is None:
        return []
    result = []
    for task in reading_plan.tasks:
        for output in task.required_outputs:
            output_key = f"{task.task_id}:{output}"
            if output_key not in result:
                result.append(output_key)
    return result


def graph_covered_required_output_keys(graph: ClaimGraph) -> set[str]:
    result: set[str] = set()
    for node in graph.nodes.values():
        task_id = str(node.payload.get("task_id") or "").strip()
        if not task_id:
            continue
        covered_outputs = node.payload.get("covered_outputs")
        if not isinstance(covered_outputs, list):
            continue
        for item in covered_outputs:
            output = str(item or "").strip()
            if output:
                result.add(f"{task_id}:{output}")
    return result
