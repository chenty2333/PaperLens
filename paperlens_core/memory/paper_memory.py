from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from paperlens_core.graph.claim_graph import ClaimGraph


class PaperMemoryView(BaseModel):
    schema_version: str = "paper_memory.view.v1"
    paper_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    problem_nodes: list[str] = Field(default_factory=list)
    contribution_nodes: list[str] = Field(default_factory=list)
    mechanism_nodes: list[str] = Field(default_factory=list)
    implementation_nodes: list[str] = Field(default_factory=list)
    evaluation_nodes: list[str] = Field(default_factory=list)
    result_nodes: list[str] = Field(default_factory=list)
    limitation_nodes: list[str] = Field(default_factory=list)
    concept_nodes: list[str] = Field(default_factory=list)
    evidence_index: dict[str, list[str]] = Field(default_factory=dict)
    unresolved_audit_findings: list[str] = Field(default_factory=list)
    report_readiness: str = "DRAFT_WEAK"


def materialize_paper_memory(
    graph: ClaimGraph,
    *,
    metadata: dict[str, Any] | None = None,
    unresolved_audit_findings: list[str] | None = None,
    report_readiness: str = "DRAFT_WEAK",
) -> PaperMemoryView:
    by_kind: dict[str, list[str]] = {}
    evidence_index: dict[str, list[str]] = {}
    for node in graph.nodes.values():
        by_kind.setdefault(node.kind, []).append(node.node_id)
        evidence = graph.evidence_ids_for(node.node_id)
        if evidence:
            evidence_index[node.node_id] = evidence
    return PaperMemoryView(
        paper_id=graph.paper_id,
        metadata=metadata or {},
        problem_nodes=by_kind.get("problem", []),
        contribution_nodes=by_kind.get("claim", []),
        mechanism_nodes=by_kind.get("mechanism", []),
        implementation_nodes=by_kind.get("implementation", []),
        evaluation_nodes=by_kind.get("evaluation", []),
        result_nodes=by_kind.get("result", []),
        limitation_nodes=by_kind.get("limitation", []),
        concept_nodes=by_kind.get("concept", []),
        evidence_index=evidence_index,
        unresolved_audit_findings=unresolved_audit_findings or [],
        report_readiness=report_readiness,
    )
