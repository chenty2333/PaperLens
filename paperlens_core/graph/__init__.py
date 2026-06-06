from __future__ import annotations

from paperlens_core.graph.claim_graph import (
    ClaimGraph,
    GraphEdge,
    GraphNode,
    graph_from_observations,
)
from paperlens_core.graph.relation_pass import apply_relation_pass


def build_claim_graph(paper_id: str, observation_cards: list) -> ClaimGraph:
    graph = graph_from_observations(paper_id, observation_cards)
    return apply_relation_pass(graph)


__all__ = [
    "ClaimGraph",
    "GraphEdge",
    "GraphNode",
    "apply_relation_pass",
    "build_claim_graph",
    "graph_from_observations",
]
