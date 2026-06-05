from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from paperlens_core.reading.observation import ObservationCard, ObservationType


NodeKind = Literal[
    "problem",
    "claim",
    "mechanism",
    "implementation",
    "evaluation",
    "result",
    "limitation",
    "concept",
    "evidence",
]

EdgeKind = Literal[
    "supported_by",
    "contradicted_by",
    "depends_on",
    "explains",
    "implements",
    "evaluated_by",
    "limited_by",
    "compared_with",
]


class GraphNode(BaseModel):
    node_id: str
    kind: NodeKind
    label: str
    payload: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    source_id: str
    target_id: str
    kind: EdgeKind
    payload: dict[str, Any] = Field(default_factory=dict)


class ClaimGraph(BaseModel):
    schema_version: str = "claim_graph.v1"
    paper_id: str
    nodes: dict[str, GraphNode] = Field(default_factory=dict)
    edges: list[GraphEdge] = Field(default_factory=list)

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.node_id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        if not any(
            existing.source_id == edge.source_id
            and existing.target_id == edge.target_id
            and existing.kind == edge.kind
            for existing in self.edges
        ):
            self.edges.append(edge)

    def evidence_ids_for(self, node_id: str) -> list[str]:
        return [
            edge.target_id
            for edge in self.edges
            if edge.source_id == node_id and edge.kind == "supported_by"
        ]


OBSERVATION_NODE_KIND: dict[ObservationType, NodeKind] = {
    ObservationType.PROBLEM: "problem",
    ObservationType.CLAIM: "claim",
    ObservationType.MECHANISM: "mechanism",
    ObservationType.IMPLEMENTATION: "implementation",
    ObservationType.EVALUATION: "evaluation",
    ObservationType.RESULT: "result",
    ObservationType.LIMITATION: "limitation",
    ObservationType.CONCEPT: "concept",
}


def graph_from_observations(paper_id: str, observations: list[ObservationCard]) -> ClaimGraph:
    graph = ClaimGraph(paper_id=paper_id)
    for card in observations:
        node_kind = OBSERVATION_NODE_KIND[card.observation_type]
        node_id = f"{node_kind}:{card.observation_id}"
        graph.add_node(
            GraphNode(
                node_id=node_id,
                kind=node_kind,
                label=card.statement,
                payload={
                    "observation_id": card.observation_id,
                    "task_id": card.task_id,
                    "confidence": card.confidence,
                    "provenance": card.provenance,
                    "uncertainty": card.uncertainty,
                    "extracted_numbers": card.extracted_numbers,
                },
            )
        )
        for source_id in card.source_ids:
            evidence_id = f"evidence:{source_id}"
            graph.add_node(
                GraphNode(
                    node_id=evidence_id,
                    kind="evidence",
                    label=source_id,
                    payload={"source_id": source_id},
                )
            )
            graph.add_edge(GraphEdge(source_id=node_id, target_id=evidence_id, kind="supported_by"))
    return graph
