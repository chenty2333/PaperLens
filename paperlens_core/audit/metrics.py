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
    missing_source_count: int
    audit_error_count: int
    audit_warning_count: int
    publish_status: str


def compute_core_quality_metrics(
    *,
    dom: PaperDOM,
    graph: ClaimGraph,
    findings: list[AuditFinding],
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
        missing_source_count=len(missing_sources),
        audit_error_count=len(errors),
        audit_warning_count=len(warnings),
        publish_status=publish_status_from_findings(findings).value,
    )


def metric_rate(numerator: int, denominator: int, *, default: float) -> float:
    if not denominator:
        return default
    return round(numerator / denominator, 4)
