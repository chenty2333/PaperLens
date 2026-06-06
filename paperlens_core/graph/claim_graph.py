from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

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
    model_config = ConfigDict(extra="forbid")

    node_id: str
    kind: NodeKind
    label: str
    payload: dict[str, Any] = Field(default_factory=dict)


class GraphEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_id: str
    target_id: str
    kind: EdgeKind
    payload: dict[str, Any] = Field(default_factory=dict)


class ClaimGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "claim_graph.v1"
    paper_id: str
    nodes: dict[str, GraphNode] = Field(default_factory=dict)
    edges: list[GraphEdge] = Field(default_factory=list)

    def add_node(self, node: GraphNode) -> None:
        existing = self.nodes.get(node.node_id)
        if existing is not None:
            if graph_node_identity_payload(existing) != graph_node_identity_payload(node):
                raise ValueError(f"conflicting graph node_id: {node.node_id}")
            return
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
            if edge.source_id == node_id
            and edge.kind == "supported_by"
            and self.nodes.get(edge.target_id) is not None
            and self.nodes[edge.target_id].kind == "evidence"
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
    obs_to_node: dict[str, str] = {}
    for card in observations:
        if card.paper_id != paper_id:
            raise ValueError(f"observation paper_id mismatch: {card.paper_id} != {paper_id}")
        node_kind = OBSERVATION_NODE_KIND[card.observation_type]
        node_id = observation_node_id(card)
        obs_to_node[card.observation_id] = node_id
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
                    "covered_outputs": card.covered_outputs,
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

    for card in observations:
        source_node_id = obs_to_node.get(card.observation_id)
        if source_node_id is None:
            continue
        for rel in card.proposed_relations:
            target_node_id = obs_to_node.get(rel.target_observation_id)
            if target_node_id is None:
                continue
            if source_node_id == target_node_id:
                continue
            graph.add_edge(
                GraphEdge(
                    source_id=source_node_id,
                    target_id=target_node_id,
                    kind=rel.kind,
                    payload={"proposed_by": card.observation_id},
                )
            )

    return graph


def graph_node_identity_payload(node: GraphNode) -> dict[str, Any]:
    return node.model_dump(mode="json")


def observation_node_id(card: ObservationCard) -> str:
    node_kind = OBSERVATION_NODE_KIND[card.observation_type]
    return f"{node_kind}:{card.observation_id}"
