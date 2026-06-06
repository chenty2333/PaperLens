from __future__ import annotations

from paperlens_core.graph.claim_graph import (
    ClaimGraph,
    GraphEdge,
    GraphNode,
    graph_from_observations,
    materialize_relation_candidates,
    observation_node_id,
)
from paperlens_core.graph.relation_pass import apply_relation_pass


def build_claim_graph(
    paper_id: str,
    observation_cards: list,
    relation_candidates: list | None = None,
) -> ClaimGraph:
    graph = graph_from_observations(paper_id, observation_cards)
    if relation_candidates:
        obs_to_node = {
            card.observation_id: observation_node_id(card)
            for card in observation_cards
            if hasattr(card, "observation_id")
        }
        materialize_relation_candidates(graph, relation_candidates, obs_to_node)
    return apply_relation_pass(graph)


__all__ = [
    "ClaimGraph",
    "GraphEdge",
    "GraphNode",
    "apply_relation_pass",
    "build_claim_graph",
    "graph_from_observations",
    "materialize_relation_candidates",
]
