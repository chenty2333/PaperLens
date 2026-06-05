from __future__ import annotations

from pydantic import BaseModel

from paperlens_core.audit.findings import AuditFinding
from paperlens_core.audit.suite import FACT_NODE_KINDS, publish_status_from_findings
from paperlens_core.dom.paper_dom import PaperDOM
from paperlens_core.graph.claim_graph import ClaimGraph


class CoreQualityMetrics(BaseModel):
    schema_version: str = "paperlens_core_quality.v1"
    paper_id: str
    fact_node_count: int
    supported_fact_node_count: int
    evidence_coverage: float
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
    missing_sources = [finding for finding in findings if finding.code == "missing_dom_source"]
    errors = [finding for finding in findings if finding.severity == "ERROR"]
    warnings = [finding for finding in findings if finding.severity == "WARNING"]
    return CoreQualityMetrics(
        paper_id=dom.paper_id,
        fact_node_count=len(fact_nodes),
        supported_fact_node_count=len(supported),
        evidence_coverage=round(len(supported) / denominator, 4),
        missing_source_count=len(missing_sources),
        audit_error_count=len(errors),
        audit_warning_count=len(warnings),
        publish_status=publish_status_from_findings(findings).value,
    )
