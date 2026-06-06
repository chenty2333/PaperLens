from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from paperlens_core.audit import AuditFinding, AuditSeverity
from paperlens_core.dom import PaperDOM
from paperlens_core.grounding import text_overlaps_any_reference
from paperlens_core.graph import ClaimGraph


class ReportParagraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paragraph_id: str
    markdown: str
    used_node_ids: list[str] = Field(default_factory=list)
    used_evidence_ids: list[str] = Field(default_factory=list)


class ReportSection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    section_id: str
    title: str
    paragraphs: list[ReportParagraph] = Field(default_factory=list)


class GraphReportDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "graph_report_draft.v2"
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

NUMBER_TEXT_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s?%|[A-Za-z]{1,6})?"
)


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
    if draft.paper_id != graph.paper_id:
        findings.append(
            AuditFinding(
                finding_id=f"report_draft_paper_id_mismatch:{draft.paper_id}:{graph.paper_id}",
                severity=AuditSeverity.ERROR,
                code="report_draft_paper_id_mismatch",
                message=f"Report draft paper_id does not match ClaimGraph: {draft.paper_id} != {graph.paper_id}",
            )
        )
    node_ids = set(graph.nodes)
    evidence_node_ids = {node.node_id for node in graph.nodes.values() if node.kind == "evidence"}
    support_edges = {
        (edge.source_id, edge.target_id) for edge in graph.edges if edge.kind == "supported_by"
    }
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
            known_node_ids = [node_id for node_id in paragraph.used_node_ids if node_id in node_ids]
            known_evidence_ids = [
                evidence_id
                for evidence_id in paragraph.used_evidence_ids
                if evidence_id in evidence_node_ids
            ]
            known_source_ids = graph_report_source_ids(
                graph=graph,
                evidence_ids=known_evidence_ids,
            )
            if known_evidence_ids and not known_source_ids:
                findings.append(
                    AuditFinding(
                        finding_id=f"paragraph_missing_sources:{paragraph.paragraph_id}",
                        severity=AuditSeverity.ERROR,
                        code="report_paragraph_missing_source_ids",
                        message=(
                            "Report paragraph declares evidence IDs, but those evidence nodes do "
                            "not expose PaperDOM source IDs"
                        ),
                    )
                )
            for node_id in known_node_ids:
                if not any(
                    (node_id, evidence_id) in support_edges for evidence_id in known_evidence_ids
                ):
                    findings.append(
                        AuditFinding(
                            finding_id=(
                                "paragraph_node_without_declared_support:"
                                f"{paragraph.paragraph_id}:{node_id}"
                            ),
                            severity=AuditSeverity.ERROR,
                            code="report_paragraph_node_missing_declared_evidence",
                            message=(
                                "Report paragraph declares a graph node but none of its declared "
                                "evidence IDs support that node"
                            ),
                            node_id=node_id,
                        )
                    )
            paragraph_overlaps_any_node = bool(known_node_ids) and text_overlaps_any_reference(
                paragraph.markdown,
                [graph.nodes[node_id].label for node_id in known_node_ids],
            )
            if known_node_ids and not paragraph_overlaps_any_node:
                findings.append(
                    AuditFinding(
                        finding_id=f"paragraph_text_not_grounded:{paragraph.paragraph_id}",
                        severity=AuditSeverity.ERROR,
                        code="report_paragraph_text_not_grounded_in_declared_nodes",
                        message=(
                            "Report paragraph text does not overlap declared ClaimGraph node labels"
                        ),
                        node_id=known_node_ids[0],
                        source_ids=known_source_ids,
                    )
                )
            for node_id in known_node_ids:
                if paragraph_overlaps_any_node and not text_overlaps_any_reference(
                    paragraph.markdown, [graph.nodes[node_id].label]
                ):
                    findings.append(
                        AuditFinding(
                            finding_id=(
                                "paragraph_declared_node_not_used_in_text:"
                                f"{paragraph.paragraph_id}:{node_id}"
                            ),
                            severity=AuditSeverity.ERROR,
                            code="report_paragraph_declared_node_not_used_in_text",
                            message=(
                                "Report paragraph declares a ClaimGraph node whose label is not "
                                "reflected in the paragraph text"
                            ),
                            node_id=node_id,
                        )
                    )
            declared_node_texts = [graph.nodes[node_id].label for node_id in known_node_ids]
            if paragraph_overlaps_any_node:
                for number_text in number_texts(paragraph.markdown):
                    if declared_node_texts and not any(
                        number_text_is_located(number_text, node_text)
                        for node_text in declared_node_texts
                    ):
                        findings.append(
                            AuditFinding(
                                finding_id=(
                                    "paragraph_number_not_grounded:"
                                    f"{paragraph.paragraph_id}:"
                                    f"{normalized_number_text(number_text)}"
                                ),
                                severity=AuditSeverity.ERROR,
                                code="report_paragraph_number_not_grounded_in_declared_nodes",
                                message=(
                                    "Report paragraph includes a numeric value that is not present "
                                    "in its declared ClaimGraph node labels"
                                ),
                                node_id=known_node_ids[0] if known_node_ids else None,
                                source_ids=known_source_ids,
                            )
                        )
            for evidence_id in known_evidence_ids:
                if not any((node_id, evidence_id) in support_edges for node_id in known_node_ids):
                    findings.append(
                        AuditFinding(
                            finding_id=(
                                "paragraph_evidence_without_declared_node:"
                                f"{paragraph.paragraph_id}:{evidence_id}"
                            ),
                            severity=AuditSeverity.ERROR,
                            code="report_paragraph_evidence_not_linked_to_declared_node",
                            message=(
                                "Report paragraph declares an evidence ID that does not support "
                                "any declared graph node"
                            ),
                            node_id=evidence_id,
                            source_ids=evidence_source_ids(graph, evidence_id),
                        )
                    )
    return findings


def evidence_source_ids(graph: ClaimGraph, evidence_id: str) -> list[str]:
    node = graph.nodes.get(evidence_id)
    if node is None:
        return []
    source_id = str(node.payload.get("source_id") or "")
    return [source_id] if source_id else []


def render_graph_report_markdown(
    *,
    title: str,
    draft: GraphReportDraft,
    graph: ClaimGraph,
    dom: PaperDOM,
    quality: dict | None = None,
) -> str:
    quality = quality or {}
    lines = [
        f"# {title or draft.paper_id}",
        "",
        "> Deterministic ClaimGraph report view. Paragraph facts come from graph nodes; evidence",
        "> is declared with ClaimGraph evidence IDs and PaperDOM source IDs.",
        "",
        f"- Paper ID: `{draft.paper_id}`",
        f"- Graph schema: `{graph.schema_version}`",
    ]
    publish_status = quality.get("publish_status")
    if publish_status:
        lines.append(f"- Publish status: `{publish_status}`")
    lines.append("")
    for section in draft.sections:
        lines.extend([f"## {section.title}", ""])
        for paragraph in section.paragraphs:
            node_ids = [node_id for node_id in paragraph.used_node_ids if node_id in graph.nodes]
            evidence_ids = [
                evidence_id
                for evidence_id in paragraph.used_evidence_ids
                if evidence_id in graph.nodes
            ]
            source_ids = graph_report_source_ids(graph=graph, evidence_ids=evidence_ids)
            lines.extend(
                [
                    paragraph.markdown.strip(),
                    "",
                    f"- ClaimGraph nodes: {inline_code_list(node_ids)}",
                    f"- Evidence nodes: {inline_code_list(evidence_ids)}",
                    f"- PaperDOM sources: {inline_code_list(source_ids)}",
                ]
            )
            for source_id in source_ids[:4]:
                source = describe_dom_source(dom, source_id)
                if source:
                    lines.append(f"  - {source}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def graph_report_source_ids(*, graph: ClaimGraph, evidence_ids: list[str]) -> list[str]:
    source_ids = []
    for evidence_id in evidence_ids:
        for source_id in evidence_source_ids(graph, evidence_id):
            if source_id not in source_ids:
                source_ids.append(source_id)
    return source_ids


def inline_code_list(values: list[str]) -> str:
    return ", ".join(f"`{value}`" for value in values) if values else "`none`"


def describe_dom_source(dom: PaperDOM, source_id: str) -> str:
    for span in dom.spans:
        if span.source_id == source_id:
            return (
                f"`{source_id}` ({span.kind}, page {span.page_no}): "
                f"{compact_source_text(span.text)}"
            )
    for figure in dom.figures:
        if figure.source_id == source_id:
            return (
                f"`{source_id}` ({figure.kind}, page {figure.page_no}): "
                f"{compact_source_text(figure.caption or '')}"
            )
    for table in dom.tables:
        if table.source_id == source_id:
            return (
                f"`{source_id}` ({table.kind}, page {table.page_no}): "
                f"{compact_source_text(table.caption or '')}"
            )
    for equation in dom.equations:
        if equation.source_id == source_id:
            return (
                f"`{source_id}` ({equation.kind}, page {equation.page_no}): "
                f"{compact_source_text(equation.latex_or_text)}"
            )
    for section in dom.sections:
        if section.source_id == source_id:
            return (
                f"`{source_id}` ({section.kind}, page {section.page_no}): "
                f"{compact_source_text(section.title)}"
            )
    return ""


def compact_source_text(text: str, *, limit: int = 260) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def number_texts(text: str) -> list[str]:
    result: list[str] = []
    for match in NUMBER_TEXT_PATTERN.finditer(str(text or "")):
        value = match.group(0).strip()
        if value and value not in result:
            result.append(value)
    return result


def number_text_is_located(number_text: str, source_text: str) -> bool:
    needle = normalized_number_text(number_text)
    haystack = normalized_number_text(source_text)
    return bool(needle and needle in haystack)


def normalized_number_text(text: str) -> str:
    return re.sub(r"[\s,]+", "", str(text or "").casefold())
