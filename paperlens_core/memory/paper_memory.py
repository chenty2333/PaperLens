from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from paperlens_core.audit import AuditFinding, AuditSeverity
from paperlens_core.dom import PaperDOM
from paperlens_core.graph.claim_graph import ClaimGraph


class PaperMemoryRelationshipEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    kind: str
    payload: dict[str, Any] = Field(default_factory=dict)


class PaperMemoryAuditIssue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    severity: str
    code: str
    message: str
    node_id: str | None = None
    source_ids: list[str] = Field(default_factory=list)


class PaperMemoryFactNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    audit_status: str = "REVIEWED"
    audit_issue_ids: list[str] = Field(default_factory=list)


class PaperMemoryEvidenceSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    kind: str
    page_no: int | None = None
    section_id: str | None = None
    excerpt: str = ""


class PaperMemoryEvaluationItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    node_id: str
    kind: str
    label: str
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    extracted_numbers: list[dict[str, Any]] = Field(default_factory=list)
    audit_status: str = "REVIEWED"
    audit_issue_ids: list[str] = Field(default_factory=list)


class PaperMemoryView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "paper_memory.view.v2"
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
    audit_issues: list[PaperMemoryAuditIssue] = Field(default_factory=list)
    audit_issues_by_node: dict[str, list[str]] = Field(default_factory=dict)
    audit_issues_by_source: dict[str, list[str]] = Field(default_factory=dict)
    report_readiness: str = "DRAFT_WEAK"


def materialize_paper_memory(
    graph: ClaimGraph,
    *,
    dom: PaperDOM | None = None,
    metadata: dict[str, Any] | None = None,
    unresolved_audit_findings: list[str] | None = None,
    audit_findings: list[AuditFinding | dict[str, Any]] | None = None,
    report_readiness: str = "DRAFT_WEAK",
) -> PaperMemoryView:
    by_kind: dict[str, list[str]] = {}
    evidence_index: dict[str, list[str]] = {}
    relationship_edges: list[PaperMemoryRelationshipEdge] = []
    source_index = paper_dom_source_index(dom) if dom is not None else {}
    evidence_sources: dict[str, PaperMemoryEvidenceSource] = {}
    fact_nodes: list[PaperMemoryFactNode] = []
    evaluation_matrix: list[PaperMemoryEvaluationItem] = []
    audit_issues = normalize_audit_issues(audit_findings or [])
    audit_issues_by_node = index_audit_issues_by_node(audit_issues)
    audit_issues_by_source = index_audit_issues_by_source(audit_issues)
    for node in graph.nodes.values():
        by_kind.setdefault(node.kind, []).append(node.node_id)
        evidence = graph.evidence_ids_for(node.node_id)
        if evidence:
            evidence_index[node.node_id] = evidence
        if node.kind == "evidence":
            continue
        source_ids = fact_source_ids(graph, evidence)
        pages = fact_pages(source_ids, source_index)
        audit_issue_ids = fact_node_audit_issue_ids(
            node_id=node.node_id,
            evidence_ids=evidence,
            source_ids=source_ids,
            issues_by_node=audit_issues_by_node,
            issues_by_source=audit_issues_by_source,
        )
        audit_status = audit_status_for_issue_ids(audit_issue_ids, audit_issues)
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
            audit_status=audit_status,
            audit_issue_ids=audit_issue_ids,
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
                    audit_status=fact_node.audit_status,
                    audit_issue_ids=fact_node.audit_issue_ids,
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
        unresolved_audit_findings=unresolved_audit_finding_ids(
            explicit_ids=unresolved_audit_findings,
            audit_issues=audit_issues,
        ),
        audit_issues=audit_issues,
        audit_issues_by_node=audit_issues_by_node,
        audit_issues_by_source=audit_issues_by_source,
        report_readiness=report_readiness,
    )


def normalize_audit_issues(
    findings: list[AuditFinding | dict[str, Any]],
) -> list[PaperMemoryAuditIssue]:
    result: list[PaperMemoryAuditIssue] = []
    seen: set[str] = set()
    for finding in findings:
        if isinstance(finding, AuditFinding):
            data = finding.model_dump()
        elif isinstance(finding, dict):
            data = finding
        else:
            continue
        finding_id = str(data.get("finding_id") or "").strip()
        if not finding_id or finding_id in seen:
            continue
        seen.add(finding_id)
        result.append(
            PaperMemoryAuditIssue(
                finding_id=finding_id,
                severity=str(data.get("severity") or AuditSeverity.WARNING.value),
                code=str(data.get("code") or "unknown_audit_finding"),
                message=str(data.get("message") or ""),
                node_id=string_or_none(data.get("node_id")),
                source_ids=dedupe_strings(data.get("source_ids")),
            )
        )
    return result


def index_audit_issues_by_node(
    audit_issues: list[PaperMemoryAuditIssue],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for issue in audit_issues:
        if not issue.node_id:
            continue
        result.setdefault(issue.node_id, []).append(issue.finding_id)
    return result


def index_audit_issues_by_source(
    audit_issues: list[PaperMemoryAuditIssue],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for issue in audit_issues:
        for source_id in issue.source_ids:
            result.setdefault(source_id, []).append(issue.finding_id)
    return result


def fact_node_audit_issue_ids(
    *,
    node_id: str,
    evidence_ids: list[str],
    source_ids: list[str],
    issues_by_node: dict[str, list[str]],
    issues_by_source: dict[str, list[str]],
) -> list[str]:
    result: list[str] = []
    for issue_id in issues_by_node.get(node_id, []):
        append_unique(result, issue_id)
    for evidence_id in evidence_ids:
        for issue_id in issues_by_node.get(evidence_id, []):
            append_unique(result, issue_id)
    for source_id in source_ids:
        for issue_id in issues_by_source.get(source_id, []):
            append_unique(result, issue_id)
    return result


def audit_status_for_issue_ids(
    issue_ids: list[str],
    audit_issues: list[PaperMemoryAuditIssue],
) -> str:
    if not issue_ids:
        return "REVIEWED"
    severities = {
        issue.finding_id: issue.severity
        for issue in audit_issues
        if issue.finding_id in set(issue_ids)
    }
    if any(severity == AuditSeverity.ERROR.value for severity in severities.values()):
        return "BLOCKED"
    if any(severity == AuditSeverity.WARNING.value for severity in severities.values()):
        return "REVIEWED_WITH_LIMITS"
    return "REVIEWED"


def unresolved_audit_finding_ids(
    *,
    explicit_ids: list[str] | None,
    audit_issues: list[PaperMemoryAuditIssue],
) -> list[str]:
    if explicit_ids is not None:
        return dedupe_strings(explicit_ids)
    return [issue.finding_id for issue in audit_issues]


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


def dedupe_strings(value: Any) -> list[str]:
    result: list[str] = []
    if not isinstance(value, list):
        return result
    for item in value:
        text = str(item).strip()
        if text:
            append_unique(result, text)
    return result


def append_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)
