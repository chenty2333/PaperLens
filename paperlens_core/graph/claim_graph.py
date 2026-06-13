from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

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

    schema_version: str = "claim_graph.v2"
    paper_id: str
    nodes: dict[str, GraphNode] = Field(default_factory=dict)
    edges: list[GraphEdge] = Field(default_factory=list)
    _edge_keys: set[tuple[str, str, str]] = PrivateAttr(default_factory=set)
    _evidence_ids_by_node: dict[str, list[str]] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        self._rebuild_edge_indexes()

    def _rebuild_edge_indexes(self) -> None:
        self._edge_keys = set()
        self._evidence_ids_by_node = {}
        for edge in self.edges:
            self._index_edge(edge)

    def _index_edge(self, edge: GraphEdge) -> None:
        self._edge_keys.add(edge_identity_key(edge))
        target_node = self.nodes.get(edge.target_id)
        if edge.kind == "supported_by" and target_node is not None and target_node.kind == "evidence":
            evidence_ids = self._evidence_ids_by_node.setdefault(edge.source_id, [])
            if edge.target_id not in evidence_ids:
                evidence_ids.append(edge.target_id)

    def add_node(self, node: GraphNode) -> None:
        existing = self.nodes.get(node.node_id)
        if existing is not None:
            if graph_node_identity_payload(existing) != graph_node_identity_payload(node):
                raise ValueError(f"conflicting graph node_id: {node.node_id}")
            return
        self.nodes[node.node_id] = node
        if node.kind == "evidence":
            for edge in self.edges:
                if edge.target_id == node.node_id and edge.kind == "supported_by":
                    evidence_ids = self._evidence_ids_by_node.setdefault(edge.source_id, [])
                    if node.node_id not in evidence_ids:
                        evidence_ids.append(node.node_id)

    def add_edge(self, edge: GraphEdge) -> None:
        key = edge_identity_key(edge)
        if key in self._edge_keys:
            return
        self.edges.append(edge)
        self._index_edge(edge)

    def evidence_ids_for(self, node_id: str) -> list[str]:
        return list(self._evidence_ids_by_node.get(node_id, []))


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
                    "source_ids": list(card.source_ids),
                    "confidence": card.confidence,
                    "provenance": card.provenance,
                    "uncertainty": card.uncertainty,
                    "covered_outputs": card.covered_outputs,
                    "evidence_quotes": card.evidence_quotes,
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


def materialize_relation_candidates(
    graph: ClaimGraph,
    candidates: list[Any],
    obs_to_node: dict[str, str],
) -> None:
    for candidate in candidates:
        source_id = getattr(candidate, "source_observation_id", None) or candidate.get("source_observation_id", "")
        target_id = getattr(candidate, "target_observation_id", None) or candidate.get("target_observation_id", "")
        kind = getattr(candidate, "kind", None) or candidate.get("kind", "")
        source_node_id = obs_to_node.get(str(source_id))
        target_node_id = obs_to_node.get(str(target_id))
        if source_node_id is None or target_node_id is None:
            continue
        if source_node_id == target_node_id:
            continue
        graph.add_edge(
            GraphEdge(
                source_id=source_node_id,
                target_id=target_node_id,
                kind=str(kind),
                payload={"from_relation_candidate": True},
            )
        )


def graph_node_identity_payload(node: GraphNode) -> dict[str, Any]:
    return node.model_dump(mode="json")


def edge_identity_key(edge: GraphEdge) -> tuple[str, str, str]:
    return (edge.source_id, edge.target_id, edge.kind)


def observation_node_id(card: ObservationCard) -> str:
    node_kind = OBSERVATION_NODE_KIND[card.observation_type]
    return f"{node_kind}:{card.observation_id}"
