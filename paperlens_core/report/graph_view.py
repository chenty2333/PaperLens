from __future__ import annotations

from pydantic import BaseModel, Field

from paperlens_core.audit import AuditFinding, AuditSeverity
from paperlens_core.graph import ClaimGraph


class ReportParagraph(BaseModel):
    paragraph_id: str
    markdown: str
    used_node_ids: list[str] = Field(default_factory=list)
    used_evidence_ids: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    section_id: str
    title: str
    paragraphs: list[ReportParagraph] = Field(default_factory=list)


class GraphReportDraft(BaseModel):
    schema_version: str = "graph_report_draft.v1"
    paper_id: str
    sections: list[ReportSection] = Field(default_factory=list)


SECTION_TITLES = {
    "problem": "Problem Frame",
    "claim": "Claims",
    "mechanism": "Mechanism",
    "implementation": "Implementation",
    "evaluation": "Evaluation",
    "result": "Results",
    "limitation": "Limitations",
    "concept": "Concept Bridge",
}


def build_report_draft_from_graph(
    graph: ClaimGraph, *, max_nodes_per_kind: int = 8
) -> GraphReportDraft:
    sections: list[ReportSection] = []
    for kind, title in SECTION_TITLES.items():
        nodes = [node for node in graph.nodes.values() if node.kind == kind][:max_nodes_per_kind]
        if not nodes:
            continue
        paragraphs = []
        for index, node in enumerate(nodes, start=1):
            evidence_ids = graph.evidence_ids_for(node.node_id)
            paragraphs.append(
                ReportParagraph(
                    paragraph_id=f"{kind}_{index:02d}",
                    markdown=node.label,
                    used_node_ids=[node.node_id],
                    used_evidence_ids=evidence_ids,
                )
            )
        sections.append(
            ReportSection(
                section_id=kind,
                title=title,
                paragraphs=paragraphs,
            )
        )
    return GraphReportDraft(paper_id=graph.paper_id, sections=sections)


def audit_report_draft_against_graph(
    draft: GraphReportDraft,
    graph: ClaimGraph,
) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    node_ids = set(graph.nodes)
    evidence_node_ids = {node.node_id for node in graph.nodes.values() if node.kind == "evidence"}
    for section in draft.sections:
        for paragraph in section.paragraphs:
            if not paragraph.used_node_ids:
                findings.append(
                    AuditFinding(
                        finding_id=f"paragraph_missing_nodes:{paragraph.paragraph_id}",
                        severity=AuditSeverity.ERROR,
                        code="report_paragraph_missing_node_ids",
                        message="Report paragraph does not declare graph node IDs",
                    )
                )
            if not paragraph.used_evidence_ids:
                findings.append(
                    AuditFinding(
                        finding_id=f"paragraph_missing_evidence:{paragraph.paragraph_id}",
                        severity=AuditSeverity.ERROR,
                        code="report_paragraph_missing_evidence_ids",
                        message="Report paragraph does not declare evidence IDs",
                    )
                )
            for node_id in paragraph.used_node_ids:
                if node_id not in node_ids:
                    findings.append(
                        AuditFinding(
                            finding_id=f"paragraph_unknown_node:{paragraph.paragraph_id}:{node_id}",
                            severity=AuditSeverity.ERROR,
                            code="report_paragraph_unknown_node_id",
                            message=f"Report paragraph references unknown graph node: {node_id}",
                            node_id=node_id,
                        )
                    )
            for evidence_id in paragraph.used_evidence_ids:
                if evidence_id not in evidence_node_ids:
                    findings.append(
                        AuditFinding(
                            finding_id=(
                                f"paragraph_unknown_evidence:{paragraph.paragraph_id}:{evidence_id}"
                            ),
                            severity=AuditSeverity.ERROR,
                            code="report_paragraph_unknown_evidence_id",
                            message=f"Report paragraph references unknown evidence node: {evidence_id}",
                            node_id=evidence_id,
                        )
                    )
    return findings
