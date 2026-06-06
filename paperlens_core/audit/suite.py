from __future__ import annotations

import re

from paperlens_core.audit.findings import AuditFinding, AuditSeverity, PublishStatus
from paperlens_core.dom.paper_dom import PaperDOM
from paperlens_core.grounding import text_overlaps_any_reference
from paperlens_core.graph.claim_graph import ClaimGraph
from paperlens_core.reading.tasks import ReadingPlan


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
    if graph.paper_id != dom.paper_id:
        findings.append(
            AuditFinding(
                finding_id=f"claim_graph_paper_id_mismatch:{graph.paper_id}:{dom.paper_id}",
                severity=AuditSeverity.ERROR,
                code="claim_graph_paper_id_mismatch",
                message=f"ClaimGraph paper_id does not match PaperDOM: {graph.paper_id} != {dom.paper_id}",
            )
        )
    for index, edge in enumerate(graph.edges, start=1):
        source_node = graph.nodes.get(edge.source_id)
        target_node = graph.nodes.get(edge.target_id)
        if source_node is None:
            findings.append(
                AuditFinding(
                    finding_id=f"dangling_edge_source:{index}:{edge.source_id}",
                    severity=AuditSeverity.ERROR,
                    code="dangling_graph_edge_source",
                    message=f"ClaimGraph edge source_id does not exist: {edge.source_id}",
                    node_id=edge.source_id,
                )
            )
        if target_node is None:
            findings.append(
                AuditFinding(
                    finding_id=f"dangling_edge_target:{index}:{edge.target_id}",
                    severity=AuditSeverity.ERROR,
                    code="dangling_graph_edge_target",
                    message=f"ClaimGraph edge target_id does not exist: {edge.target_id}",
                    node_id=edge.target_id,
                )
            )
        if (
            edge.kind == "supported_by"
            and target_node is not None
            and target_node.kind != "evidence"
        ):
            findings.append(
                AuditFinding(
                    finding_id=f"support_edge_target_not_evidence:{index}:{edge.target_id}",
                    severity=AuditSeverity.ERROR,
                    code="support_edge_target_not_evidence",
                    message=(
                        "ClaimGraph supported_by edge must target an evidence node backed by a "
                        f"PaperDOM source_id, got {target_node.kind}: {edge.target_id}"
                    ),
                    node_id=edge.target_id,
                )
            )
        if (
            edge.kind == "supported_by"
            and source_node is not None
            and source_node.kind == "evidence"
        ):
            findings.append(
                AuditFinding(
                    finding_id=f"support_edge_source_is_evidence:{index}:{edge.source_id}",
                    severity=AuditSeverity.ERROR,
                    code="support_edge_source_is_evidence",
                    message="ClaimGraph supported_by edge source cannot be an evidence node",
                    node_id=edge.source_id,
                )
            )
        if (
            edge.kind != "supported_by"
            and source_node is not None
            and source_node.kind == "evidence"
        ):
            findings.append(
                AuditFinding(
                    finding_id=f"relationship_edge_source_is_evidence:{index}:{edge.source_id}",
                    severity=AuditSeverity.ERROR,
                    code="relationship_edge_source_is_evidence",
                    message="ClaimGraph relationship edge source cannot be an evidence node",
                    node_id=edge.source_id,
                )
            )
        if (
            edge.kind != "supported_by"
            and target_node is not None
            and target_node.kind == "evidence"
        ):
            findings.append(
                AuditFinding(
                    finding_id=f"relationship_edge_target_is_evidence:{index}:{edge.target_id}",
                    severity=AuditSeverity.ERROR,
                    code="relationship_edge_target_is_evidence",
                    message="ClaimGraph relationship edge target cannot be an evidence node",
                    node_id=edge.target_id,
                )
            )
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
        if node.kind in FACT_NODE_KINDS:
            evidence_source_ids = fact_node_source_ids(graph, node.node_id)
            evidence_texts = [
                text
                for source_id in evidence_source_ids
                if (text := source_text_for_id(dom, source_id))
            ]
            if evidence_texts and not text_overlaps_any_reference(node.label, evidence_texts):
                findings.append(
                    AuditFinding(
                        finding_id=f"source_text_mismatch:{node.node_id}",
                        severity=AuditSeverity.ERROR,
                        code="fact_node_text_not_grounded_in_evidence_source",
                        message=(
                            f"{node.kind} node text does not overlap its declared PaperDOM "
                            "evidence source text"
                        ),
                        node_id=node.node_id,
                        source_ids=evidence_source_ids,
                    )
                )
        if node.kind in FACT_NODE_KINDS and node.payload.get("confidence") == "low":
            findings.append(
                AuditFinding(
                    finding_id=f"low_confidence:{node.node_id}",
                    severity=AuditSeverity.WARNING,
                    code="low_confidence_fact_node",
                    message=f"{node.kind} node is source-bound but low confidence",
                    node_id=node.node_id,
                )
            )
        if node.kind in FACT_NODE_KINDS and str(node.payload.get("uncertainty") or "").startswith(
            "Deterministic bootstrap observation"
        ):
            findings.append(
                AuditFinding(
                    finding_id=f"bootstrap:{node.node_id}",
                    severity=AuditSeverity.WARNING,
                    code="bootstrap_observation",
                    message="Fact node comes from deterministic bootstrap and needs task-specific reading",
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
        if node.kind in FACT_NODE_KINDS:
            source_ids = fact_node_source_ids(graph, node.node_id)
            source_texts = [
                text
                for source_id in source_ids
                if (text := source_text_for_id(dom, source_id))
            ]
            for number_text in extracted_number_texts(node.payload.get("extracted_numbers")):
                if source_texts and not any(
                    number_text_is_located(number_text, source_text) for source_text in source_texts
                ):
                    findings.append(
                        AuditFinding(
                            finding_id=(
                                "extracted_number_not_located:"
                                f"{node.node_id}:{normalized_number_text(number_text)}"
                            ),
                            severity=AuditSeverity.WARNING,
                            code="extracted_number_not_located_in_source",
                            message=(
                                "Extracted numeric value is not visibly located in the "
                                "declared PaperDOM evidence source text"
                            ),
                            node_id=node.node_id,
                            source_ids=source_ids,
                        )
                    )
    return findings


def audit_reading_required_outputs(
    graph: ClaimGraph,
    reading_plan: ReadingPlan | None,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    for output_key in missing_reading_required_output_keys(graph, reading_plan):
        task_id, output = output_key.split(":", 1)
        findings.append(
            AuditFinding(
                finding_id=f"missing_reading_required_output:{task_id}:{output}",
                severity=AuditSeverity.WARNING,
                code="missing_reading_required_output",
                message=(
                    f"Reading task {task_id} required output '{output}' was not covered by "
                    "any ClaimGraph fact node."
                ),
            )
        )
    return findings


def missing_reading_required_output_keys(
    graph: ClaimGraph,
    reading_plan: ReadingPlan | None,
) -> list[str]:
    expected_output_keys = reading_required_output_keys(reading_plan)
    covered_output_keys = graph_covered_required_output_keys(graph)
    return [
        output_key for output_key in expected_output_keys if output_key not in covered_output_keys
    ]


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


def publish_status_from_findings(findings: list[AuditFinding]) -> PublishStatus:
    if any(item.severity == AuditSeverity.ERROR for item in findings):
        return PublishStatus.BLOCKED
    if any(item.code == "bootstrap_observation" for item in findings):
        return PublishStatus.DRAFT_WEAK
    if any(item.code == "missing_reading_required_output" for item in findings):
        return PublishStatus.DRAFT_WEAK
    if any(item.severity == AuditSeverity.WARNING for item in findings):
        return PublishStatus.REVIEWED_WITH_LIMITS
    return PublishStatus.REVIEWED


def contains_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


def fact_node_source_ids(graph: ClaimGraph, node_id: str) -> list[str]:
    source_ids = []
    for evidence_id in graph.evidence_ids_for(node_id):
        evidence_node = graph.nodes.get(evidence_id)
        source_id = str((evidence_node.payload if evidence_node else {}).get("source_id") or "")
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)
    return source_ids


def source_text_for_id(dom: PaperDOM, source_id: str) -> str:
    for section in dom.sections:
        if section.source_id == source_id:
            return section.title
    for span in dom.spans:
        if span.source_id == source_id:
            return span.text
    for figure in dom.figures:
        if figure.source_id == source_id:
            return figure.caption or ""
    for table in dom.tables:
        if table.source_id == source_id:
            return table.caption or ""
    for equation in dom.equations:
        if equation.source_id == source_id:
            return equation.latex_or_text
    return ""


def source_text_contains_number(dom: PaperDOM, source_id: str) -> bool:
    return contains_number(source_text_for_id(dom, source_id))


def extracted_number_texts(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def number_text_is_located(number_text: str, source_text: str) -> bool:
    needle = normalized_number_text(number_text)
    haystack = normalized_number_text(source_text)
    return bool(needle and needle in haystack)


def normalized_number_text(text: str) -> str:
    return re.sub(r"[\s,]+", "", str(text or "").casefold())
