from __future__ import annotations

import re
from typing import Any

from paperlens_core.graph.claim_graph import ClaimGraph, GraphEdge

RELATION_TERMS = {
    "related",
    "baseline",
    "compared",
    "comparison",
    "prior",
    "previous",
    "vs",
    "versus",
    "improves over",
    "outperforms",
}

CONTRADICT_TERMS = {
    "however",
    "but",
    "although",
    "unlike",
    "in contrast",
    "on the other hand",
    "conversely",
    "differs from",
    "incompatible",
    "not compatible",
    "disagree",
}

KIND_PAIR_RULES: dict[tuple[str, str], str] = {
    ("mechanism", "problem"): "explains",
    ("implementation", "mechanism"): "implements",
    ("evaluation", "claim"): "evaluated_by",
    ("evaluation", "mechanism"): "evaluated_by",
    ("evaluation", "implementation"): "evaluated_by",
    ("result", "claim"): "evaluated_by",
    ("result", "mechanism"): "evaluated_by",
    ("limitation", "claim"): "limited_by",
    ("limitation", "mechanism"): "limited_by",
    ("limitation", "implementation"): "limited_by",
    ("claim", "mechanism"): "depends_on",
    ("claim", "implementation"): "depends_on",
    ("mechanism", "concept"): "depends_on",
    ("implementation", "concept"): "depends_on",
}


def apply_relation_pass(graph: ClaimGraph) -> ClaimGraph:
    for source_node in graph.nodes.values():
        if source_node.kind == "evidence":
            continue
        for target_node in graph.nodes.values():
            if target_node.kind == "evidence":
                continue
            if source_node.node_id == target_node.node_id:
                continue

            edge_kind = _infer_edge_kind(source_node, target_node)
            if edge_kind is None:
                continue
            if _edge_exists(graph, source_node.node_id, target_node.node_id, edge_kind):
                continue
            graph.add_edge(
                GraphEdge(
                    source_id=source_node.node_id,
                    target_id=target_node.node_id,
                    kind=edge_kind,
                )
            )

    return graph


def _infer_edge_kind(source_node: Any, target_node: Any) -> str | None:
    source_label = _normalize(str(source_node.label or ""))
    target_label = _normalize(str(target_node.label or ""))

    if _kind_pair_rule(source_node.kind, target_node.kind):
        if _share_source_ids(source_node, target_node):
            return _kind_pair_rule(source_node.kind, target_node.kind)

    if source_node.kind in {"evaluation", "result"} and target_node.kind in {
        "claim",
        "mechanism",
        "implementation",
    }:
        if _share_source_ids(source_node, target_node):
            return "evaluated_by"
        if _text_references_node(source_label, target_label):
            return "evaluated_by"

    if any(term in source_label for term in RELATION_TERMS):
        if source_node.kind in {"evaluation", "result", "claim"}:
            if target_node.kind in {"claim", "mechanism", "implementation"}:
                if _share_source_ids(source_node, target_node):
                    return "compared_with"

    if any(term in source_label for term in CONTRADICT_TERMS):
        if target_node.kind in {"claim", "mechanism"}:
            if source_node.kind in {"claim", "evaluation", "result"}:
                if _share_source_ids(source_node, target_node):
                    return "contradicted_by"

    if source_node.kind in {"problem", "claim"} and target_node.kind == "mechanism":
        if _share_source_ids(source_node, target_node):
            return "depends_on"

    return None


def _kind_pair_rule(source_kind: str, target_kind: str) -> str | None:
    return KIND_PAIR_RULES.get((source_kind, target_kind))


def _share_source_ids(source_node: Any, target_node: Any) -> bool:
    source_sids = _payload_source_ids(source_node)
    target_sids = _payload_source_ids(target_node)
    if not source_sids or not target_sids:
        return False
    return bool(set(source_sids) & set(target_sids))


def _payload_source_ids(node: Any) -> list[str]:
    if hasattr(node, "payload") and isinstance(node.payload, dict):
        return node.payload.get("source_ids", []) or []
    return []


def _text_references_node(text: str, node_label: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]{4,}", node_label.lower()))
    if not tokens:
        return False
    overlap = sum(1 for token in tokens if token in text.lower())
    return overlap >= 2


def _edge_exists(graph: ClaimGraph, source_id: str, target_id: str, kind: str) -> bool:
    for edge in graph.edges:
        if edge.source_id == source_id and edge.target_id == target_id and edge.kind == kind:
            return True
    return False


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
