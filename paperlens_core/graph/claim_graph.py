from __future__ import annotations

from typing import Any, Literal, cast, get_args

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

EDGE_KINDS = set(get_args(EdgeKind))
EDGE_KIND_ALIASES = {
    "contradict": "contradicted_by",
    "contradicts": "contradicted_by",
    "contradicted_by": "contradicted_by",
    "depend_on": "depends_on",
    "depends_on": "depends_on",
    "explain": "explains",
    "explains": "explains",
    "implement": "implements",
    "implements": "implements",
    "evaluate": "evaluated_by",
    "evaluates": "evaluated_by",
    "evaluated_by": "evaluated_by",
    "limit": "limited_by",
    "limits": "limited_by",
    "limited_by": "limited_by",
    "compare_with": "compared_with",
    "compares_with": "compared_with",
    "compared_with": "compared_with",
}


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
    observation_node_ids: dict[str, str] = {}
    for card in observations:
        node_kind = OBSERVATION_NODE_KIND[card.observation_type]
        node_id = observation_node_id(card)
        observation_node_ids[card.observation_id] = node_id
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
    add_proposed_observation_links(graph, observations, observation_node_ids)
    return graph


def observation_node_id(card: ObservationCard) -> str:
    node_kind = OBSERVATION_NODE_KIND[card.observation_type]
    return f"{node_kind}:{card.observation_id}"


def add_proposed_observation_links(
    graph: ClaimGraph,
    observations: list[ObservationCard],
    observation_node_ids: dict[str, str],
) -> None:
    for card in observations:
        for link in card.proposed_links:
            source_id = resolve_graph_endpoint(
                str(link.get("source_id") or ""),
                graph=graph,
                observation_node_ids=observation_node_ids,
            )
            target_id = resolve_graph_endpoint(
                str(link.get("target_id") or ""),
                graph=graph,
                observation_node_ids=observation_node_ids,
            )
            edge_kind = normalize_edge_kind(link.get("kind"))
            if not source_id or not target_id or not edge_kind or source_id == target_id:
                continue
            graph.add_edge(
                GraphEdge(
                    source_id=source_id,
                    target_id=target_id,
                    kind=cast(EdgeKind, edge_kind),
                    payload={"proposed_by_observation_id": card.observation_id},
                )
            )


def resolve_graph_endpoint(
    value: str,
    *,
    graph: ClaimGraph,
    observation_node_ids: dict[str, str],
) -> str | None:
    endpoint = value.strip()
    if endpoint in graph.nodes:
        return endpoint
    if endpoint in observation_node_ids:
        return observation_node_ids[endpoint]
    return None


def normalize_edge_kind(value: Any) -> str | None:
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    edge_kind = EDGE_KIND_ALIASES.get(key)
    if edge_kind in EDGE_KINDS:
        return edge_kind
    return None
