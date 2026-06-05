from __future__ import annotations

import re

from paperlens_core.audit.findings import AuditFinding, AuditSeverity, PublishStatus
from paperlens_core.dom.paper_dom import PaperDOM
from paperlens_core.graph.claim_graph import ClaimGraph


FACT_NODE_KINDS = {
    "problem",
    "claim",
    "mechanism",
    "implementation",
    "evaluation",
    "result",
    "limitation",
}


def audit_claim_graph(graph: ClaimGraph, dom: PaperDOM) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    dom_source_ids = dom.source_ids()
    for node in graph.nodes.values():
        if node.kind == "evidence":
            source_id = str(node.payload.get("source_id") or "")
            if source_id not in dom_source_ids:
                findings.append(
                    AuditFinding(
                        finding_id=f"missing_source:{node.node_id}",
                        severity=AuditSeverity.ERROR,
                        code="missing_dom_source",
                        message=f"Evidence node points to missing PaperDOM source_id: {source_id}",
                        node_id=node.node_id,
                        source_ids=[source_id] if source_id else [],
                    )
                )
            continue
        if node.kind in FACT_NODE_KINDS and not graph.evidence_ids_for(node.node_id):
            findings.append(
                AuditFinding(
                    finding_id=f"unsupported:{node.node_id}",
                    severity=AuditSeverity.ERROR,
                    code="unsupported_fact_node",
                    message=f"{node.kind} node has no supporting evidence edge",
                    node_id=node.node_id,
                )
            )
        if node.kind in {"claim", "result", "evaluation"} and contains_number(node.label):
            source_ids = [
                str(graph.nodes[evidence_id].payload.get("source_id") or "")
                for evidence_id in graph.evidence_ids_for(node.node_id)
                if evidence_id in graph.nodes
            ]
            if source_ids and not any(
                source_text_contains_number(dom, source_id) for source_id in source_ids
            ):
                findings.append(
                    AuditFinding(
                        finding_id=f"number_not_located:{node.node_id}",
                        severity=AuditSeverity.WARNING,
                        code="number_not_located_in_source",
                        message="Numeric claim is supported by sources that do not visibly contain a number",
                        node_id=node.node_id,
                        source_ids=source_ids,
                    )
                )
    return findings


def publish_status_from_findings(findings: list[AuditFinding]) -> PublishStatus:
    if any(item.severity == AuditSeverity.ERROR for item in findings):
        return PublishStatus.BLOCKED
    if any(item.severity == AuditSeverity.WARNING for item in findings):
        return PublishStatus.REVIEWED_WITH_LIMITS
    return PublishStatus.REVIEWED


def contains_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


def source_text_contains_number(dom: PaperDOM, source_id: str) -> bool:
    for span in dom.spans:
        if span.source_id == source_id:
            return contains_number(span.text)
    for item in [*dom.figures, *dom.tables, *dom.equations]:
        if item.source_id == source_id:
            return contains_number(
                str(getattr(item, "caption", None) or getattr(item, "latex_or_text", ""))
            )
    return False
