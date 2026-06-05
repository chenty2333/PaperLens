from __future__ import annotations

import re

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

TOKEN_PATTERN = re.compile(r"[a-z0-9][a-z0-9_+./%-]{2,}|[\u4e00-\u9fff]+", re.IGNORECASE)
CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]+")
REPORT_GROUNDING_STOPWORDS = {
    "about",
    "also",
    "and",
    "are",
    "author",
    "authors",
    "based",
    "been",
    "being",
    "claim",
    "claims",
    "does",
    "for",
    "from",
    "has",
    "have",
    "into",
    "its",
    "new",
    "not",
    "paper",
    "propose",
    "proposed",
    "proposes",
    "result",
    "results",
    "show",
    "shows",
    "study",
    "that",
    "the",
    "their",
    "these",
    "this",
    "those",
    "using",
    "was",
    "were",
    "with",
    "work",
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
            if known_node_ids and not paragraph_text_overlaps_nodes(
                paragraph.markdown,
                [graph.nodes[node_id].label for node_id in known_node_ids],
            ):
                findings.append(
                    AuditFinding(
                        finding_id=f"paragraph_text_not_grounded:{paragraph.paragraph_id}",
                        severity=AuditSeverity.ERROR,
                        code="report_paragraph_text_not_grounded_in_declared_nodes",
                        message=(
                            "Report paragraph text does not overlap declared ClaimGraph node labels"
                        ),
                        node_id=known_node_ids[0],
                        source_ids=[
                            source_id
                            for evidence_id in known_evidence_ids
                            for source_id in evidence_source_ids(graph, evidence_id)
                        ],
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


def paragraph_text_overlaps_nodes(markdown: str, node_labels: list[str]) -> bool:
    paragraph_tokens = meaningful_tokens(markdown)
    if not paragraph_tokens:
        return False
    normalized_markdown = normalize_for_substring(markdown)
    for label in node_labels:
        normalized_label = normalize_for_substring(label)
        if normalized_label and normalized_label in normalized_markdown:
            return True
        label_tokens = meaningful_tokens(label)
        overlap = paragraph_tokens & label_tokens
        if len(overlap) >= 2 or any(is_specific_token(token) for token in overlap):
            return True
    return False


def meaningful_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for match in TOKEN_PATTERN.findall(text.lower()):
        if CJK_PATTERN.fullmatch(match):
            tokens.update(cjk_ngrams(match))
            continue
        token = match.strip("._-/")
        if token and token not in REPORT_GROUNDING_STOPWORDS:
            tokens.add(token)
    return tokens


def cjk_ngrams(text: str) -> set[str]:
    if len(text) < 2:
        return set()
    tokens = set()
    for size in (2, 3, 4):
        if len(text) >= size:
            tokens.update(text[index : index + size] for index in range(len(text) - size + 1))
    if len(text) <= 8:
        tokens.add(text)
    return tokens


def is_specific_token(token: str) -> bool:
    return len(token) >= 8 or any(character.isdigit() for character in token)


def normalize_for_substring(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()
