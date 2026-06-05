from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from paperlens_core.dom import PaperDOM
from paperlens_core.graph.claim_graph import ClaimGraph


class PaperMemoryRelationshipEdge(BaseModel):
    source_id: str
    target_id: str
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)


class PaperMemoryFactNode(BaseModel):
    node_id: str
    kind: str
    label: str
    confidence: str | None = None
    provenance: str | None = None
    uncertainty: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    extracted_numbers: list[dict[str, Any]] = Field(default_factory=list)


class PaperMemoryEvidenceSource(BaseModel):
    source_id: str
    kind: str
    page_no: int | None = None
    section_id: str | None = None
    excerpt: str = ""


class PaperMemoryEvaluationItem(BaseModel):
    node_id: str
    kind: str
    label: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    extracted_numbers: list[dict[str, Any]] = Field(default_factory=list)


class PaperMemoryView(BaseModel):
    schema_version: str = "paper_memory.view.v1"
    paper_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    fact_nodes: list[PaperMemoryFactNode] = Field(default_factory=list)
    evidence_sources: dict[str, PaperMemoryEvidenceSource] = Field(default_factory=dict)
    evaluation_matrix: list[PaperMemoryEvaluationItem] = Field(default_factory=list)
    problem_nodes: list[str] = Field(default_factory=list)
    contribution_nodes: list[str] = Field(default_factory=list)
    mechanism_nodes: list[str] = Field(default_factory=list)
    implementation_nodes: list[str] = Field(default_factory=list)
    evaluation_nodes: list[str] = Field(default_factory=list)
    result_nodes: list[str] = Field(default_factory=list)
    limitation_nodes: list[str] = Field(default_factory=list)
    concept_nodes: list[str] = Field(default_factory=list)
    evidence_index: dict[str, list[str]] = Field(default_factory=dict)
    relationship_edges: list[PaperMemoryRelationshipEdge] = Field(default_factory=list)
    unresolved_audit_findings: list[str] = Field(default_factory=list)
    report_readiness: str = "DRAFT_WEAK"


def materialize_paper_memory(
    graph: ClaimGraph,
    *,
    dom: PaperDOM | None = None,
    metadata: dict[str, Any] | None = None,
    unresolved_audit_findings: list[str] | None = None,
    report_readiness: str = "DRAFT_WEAK",
) -> PaperMemoryView:
    by_kind: dict[str, list[str]] = {}
    evidence_index: dict[str, list[str]] = {}
    relationship_edges: list[PaperMemoryRelationshipEdge] = []
    source_index = paper_dom_source_index(dom) if dom is not None else {}
    evidence_sources: dict[str, PaperMemoryEvidenceSource] = {}
    fact_nodes: list[PaperMemoryFactNode] = []
    evaluation_matrix: list[PaperMemoryEvaluationItem] = []
    for node in graph.nodes.values():
        by_kind.setdefault(node.kind, []).append(node.node_id)
        evidence = graph.evidence_ids_for(node.node_id)
        if evidence:
            evidence_index[node.node_id] = evidence
        if node.kind == "evidence":
            continue
        source_ids = fact_source_ids(graph, evidence)
        pages = fact_pages(source_ids, source_index)
        for source_id in source_ids:
            source = source_index.get(source_id)
            if source is not None:
                evidence_sources[source_id] = source
        fact_node = PaperMemoryFactNode(
            node_id=node.node_id,
            kind=node.kind,
            label=node.label,
            confidence=string_or_none(node.payload.get("confidence")),
            provenance=string_or_none(node.payload.get("provenance")),
            uncertainty=string_or_none(node.payload.get("uncertainty")),
            evidence_ids=evidence,
            source_ids=source_ids,
            pages=pages,
            extracted_numbers=[
                item for item in list_payload(node.payload.get("extracted_numbers"))[:8]
            ],
        )
        fact_nodes.append(fact_node)
        if node.kind in {"evaluation", "result"}:
            evaluation_matrix.append(
                PaperMemoryEvaluationItem(
                    node_id=fact_node.node_id,
                    kind=fact_node.kind,
                    label=fact_node.label,
                    evidence_ids=fact_node.evidence_ids,
                    source_ids=fact_node.source_ids,
                    pages=fact_node.pages,
                    extracted_numbers=fact_node.extracted_numbers,
                )
            )
    for edge in graph.edges:
        if edge.kind == "supported_by":
            continue
        if edge.source_id not in graph.nodes or edge.target_id not in graph.nodes:
            continue
        relationship_edges.append(
            PaperMemoryRelationshipEdge(
                source_id=edge.source_id,
                target_id=edge.target_id,
                kind=edge.kind,
                payload=edge.payload,
            )
        )
    return PaperMemoryView(
        paper_id=graph.paper_id,
        metadata=metadata or {},
        fact_nodes=fact_nodes,
        evidence_sources=evidence_sources,
        evaluation_matrix=evaluation_matrix,
        problem_nodes=by_kind.get("problem", []),
        contribution_nodes=by_kind.get("claim", []),
        mechanism_nodes=by_kind.get("mechanism", []),
        implementation_nodes=by_kind.get("implementation", []),
        evaluation_nodes=by_kind.get("evaluation", []),
        result_nodes=by_kind.get("result", []),
        limitation_nodes=by_kind.get("limitation", []),
        concept_nodes=by_kind.get("concept", []),
        evidence_index=evidence_index,
        relationship_edges=relationship_edges,
        unresolved_audit_findings=unresolved_audit_findings or [],
        report_readiness=report_readiness,
    )


def fact_source_ids(graph: ClaimGraph, evidence_ids: list[str]) -> list[str]:
    source_ids = []
    for evidence_id in evidence_ids:
        evidence_node = graph.nodes.get(evidence_id)
        source_id = str((evidence_node.payload if evidence_node else {}).get("source_id") or "")
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)
    return source_ids


def fact_pages(
    source_ids: list[str],
    source_index: dict[str, PaperMemoryEvidenceSource],
) -> list[int]:
    pages = []
    for source_id in source_ids:
        source = source_index.get(source_id)
        page_no = source.page_no if source else None
        if isinstance(page_no, int) and page_no not in pages:
            pages.append(page_no)
    return pages


def paper_dom_source_index(dom: PaperDOM) -> dict[str, PaperMemoryEvidenceSource]:
    result: dict[str, PaperMemoryEvidenceSource] = {}
    for span in dom.spans:
        result[span.source_id] = PaperMemoryEvidenceSource(
            source_id=span.source_id,
            kind=span.kind,
            page_no=span.page_no,
            section_id=span.section_id,
            excerpt=compact_excerpt(span.text),
        )
    for figure in dom.figures:
        result[figure.source_id] = PaperMemoryEvidenceSource(
            source_id=figure.source_id,
            kind=figure.kind,
            page_no=figure.page_no,
            excerpt=compact_excerpt(figure.caption or ""),
        )
    for table in dom.tables:
        result[table.source_id] = PaperMemoryEvidenceSource(
            source_id=table.source_id,
            kind=table.kind,
            page_no=table.page_no,
            excerpt=compact_excerpt(table.caption or ""),
        )
    for equation in dom.equations:
        result[equation.source_id] = PaperMemoryEvidenceSource(
            source_id=equation.source_id,
            kind=equation.kind,
            page_no=equation.page_no,
            section_id=equation.section_id,
            excerpt=compact_excerpt(equation.latex_or_text),
        )
    for section in dom.sections:
        result[section.source_id] = PaperMemoryEvidenceSource(
            source_id=section.source_id,
            kind=section.kind,
            page_no=section.page_no,
            excerpt=compact_excerpt(section.title),
        )
    return result


def compact_excerpt(text: str, *, limit: int = 420) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def list_payload(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def string_or_none(value: Any) -> str | None:
    text = value.strip() if isinstance(value, str) else ""
    return text or None
