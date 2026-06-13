from __future__ import annotations

import re
from typing import Any


def build_cross_paper_relations(records: list[dict[str, Any]]) -> dict[str, Any]:
    if len(records) < 2:
        return {"method_families": [], "dataset_groups": [], "paper_relations": []}

    method_families = _cluster_method_families(records)
    dataset_groups = _cluster_datasets(records)
    paper_relations = _discover_paper_relations(records, method_families, dataset_groups)

    return {
        "method_families": method_families,
        "dataset_groups": dataset_groups,
        "paper_relations": paper_relations,
    }


def _cluster_method_families(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families: list[dict[str, Any]] = []
    by_term: dict[str, list[str]] = {}

    for record in records:
        paper_id = str(record.get("paper_id") or "")
        if not paper_id:
            continue
        graph = record.get("graph_summary") if isinstance(record.get("graph_summary"), dict) else {}
        method_family = graph.get("method_family") if isinstance(graph.get("method_family"), list) else []
        for label in method_family:
            term = _normalize_term(str(label))
            if len(term) < 3:
                continue
            by_term.setdefault(term, []).append(paper_id)

    seen_terms: set[str] = set()
    for term, paper_ids in sorted(by_term.items(), key=lambda item: -len(item[1])):
        base_term = _base_term(term, seen_terms)
        if base_term and base_term in seen_terms:
            continue
        deduped = list(dict.fromkeys(paper_ids))
        if len(deduped) < 2:
            continue
        families.append({"term": term, "paper_ids": deduped, "paper_count": len(deduped)})
        seen_terms.add(term)

    return families


def _cluster_datasets(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    by_dataset: dict[str, list[str]] = {}

    for record in records:
        paper_id = str(record.get("paper_id") or "")
        if not paper_id:
            continue
        graph = record.get("graph_summary") if isinstance(record.get("graph_summary"), dict) else {}
        datasets = graph.get("evaluation_datasets") if isinstance(graph.get("evaluation_datasets"), list) else []
        for dataset in datasets:
            term = _normalize_term(str(dataset))
            if len(term) < 3:
                continue
            by_dataset.setdefault(term, []).append(paper_id)

    for term, paper_ids in sorted(by_dataset.items(), key=lambda item: -len(item[1])):
        deduped = list(dict.fromkeys(paper_ids))
        if len(deduped) < 2:
            continue
        groups.append({"dataset": term, "paper_ids": deduped, "paper_count": len(deduped)})

    return groups


def _discover_paper_relations(
    records: list[dict[str, Any]],
    method_families: list[dict[str, Any]],
    dataset_groups: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    relations: list[dict[str, Any]] = []

    for family in method_families:
        paper_ids = family.get("paper_ids") if isinstance(family.get("paper_ids"), list) else []
        for i, pid_a in enumerate(paper_ids):
            for pid_b in paper_ids[i + 1:]:
                relations.append({
                    "source_paper_id": pid_a,
                    "target_paper_id": pid_b,
                    "kind": "shared_method",
                    "detail": str(family.get("term") or ""),
                })

    for group in dataset_groups:
        paper_ids = group.get("paper_ids") if isinstance(group.get("paper_ids"), list) else []
        for i, pid_a in enumerate(paper_ids):
            for pid_b in paper_ids[i + 1:]:
                relations.append({
                    "source_paper_id": pid_a,
                    "target_paper_id": pid_b,
                    "kind": "shared_dataset",
                    "detail": str(group.get("dataset") or ""),
                })

    concept_index = _build_concept_index(records)
    for concept, paper_ids in concept_index.items():
        for i, pid_a in enumerate(paper_ids):
            for pid_b in paper_ids[i + 1:]:
                relations.append({
                    "source_paper_id": pid_a,
                    "target_paper_id": pid_b,
                    "kind": "shared_concept",
                    "detail": concept,
                })

    return _dedupe_relations(relations)


def _build_concept_index(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for record in records:
        paper_id = str(record.get("paper_id") or "")
        if not paper_id:
            continue
        concepts = record.get("concepts") if isinstance(record.get("concepts"), list) else []
        if not concepts:
            memory = record.get("memory") if isinstance(record.get("memory"), dict) else {}
            concepts = memory.get("concepts") or []
        for item in concepts:
            term = _normalize_term(item.get("term") if isinstance(item, dict) else str(item))
            if len(term) < 3:
                continue
            index.setdefault(term, []).append(paper_id)
    return {term: list(dict.fromkeys(pids)) for term, pids in index.items() if len(pids) >= 2}


def _dedupe_relations(relations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    result: list[dict[str, Any]] = []
    for rel in relations:
        key = (
            str(rel.get("source_paper_id") or ""),
            str(rel.get("target_paper_id") or ""),
            str(rel.get("kind") or ""),
            str(rel.get("detail") or ""),
        )
        if key in seen:
            continue
        seen.add(key)
        result.append(rel)
    return result


def _normalize_term(value: str) -> str:
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", str(value).lower())
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]{2,}", text)
    return " ".join(tokens).strip()


def _base_term(term: str, seen: set[str]) -> str | None:
    for seen_term in seen:
        if term in seen_term or seen_term in term:
            return seen_term
    return None
