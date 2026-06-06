from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from paperlens_core.audit import (
    audit_claim_graph,
    audit_reading_required_outputs,
    publish_status_from_findings,
)
from paperlens_core.audit.findings import AuditFinding
from paperlens_core.dom import PaperDOM
from paperlens_core.graph import ClaimGraph, GraphNode
from paperlens_core.reading import ReadingPlan
from paperlens_core.runtime import ArtifactEnvelope
from paperlens_core.core_manifest import inspect_core_v2_artifact_root


GRAPH_LIBRARY_SUMMARY_SCHEMA_VERSION = "paperlens.graph_library_summary.v1"
CONSUMABLE_GRAPH_STATUSES = {"REVIEWED", "REVIEWED_WITH_LIMITS"}

GRAPH_LIBRARY_NODE_KINDS = [
    "problem",
    "claim",
    "mechanism",
    "implementation",
    "evaluation",
    "result",
    "limitation",
    "concept",
]

METHOD_NODE_KINDS = {"mechanism", "implementation", "concept"}
CLAIM_NODE_KINDS = {"claim", "problem"}
EVALUATION_NODE_KINDS = {"evaluation", "result"}
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
METRIC_TERMS = {
    "accuracy",
    "latency",
    "throughput",
    "speedup",
    "memory",
    "cost",
    "runtime",
    "f1",
    "auc",
    "recall",
    "precision",
}


def read_core_v2_graph_summary(output_dir: Path, paper_id: str) -> dict[str, Any]:
    root = output_dir / ".paperlens" / "data" / "core" / "v2" / paper_id
    if not root.exists():
        return {}
    artifact_manifest = inspect_core_v2_artifact_root(root, paper_id)
    quality = read_optional_envelope_data(root / "quality_metrics.v2.json", "core_quality_metrics")
    memory_view = read_optional_envelope_data(
        root / "paper_memory_view.v2.json", "paper_memory_view"
    )
    dom_path = root / "paper_dom.v2.json"
    graph_path = root / "claim_graph.v2.json"
    if not dom_path.exists() or not graph_path.exists():
        return unavailable_claim_graph_summary(
            root=root,
            paper_id=paper_id,
            artifact_manifest=artifact_manifest,
            quality=quality if isinstance(quality, dict) else {},
            memory_view=memory_view if isinstance(memory_view, dict) else {},
        )
    try:
        dom_payload = read_envelope_data(dom_path, expected_type="paper_dom")
        graph_payload = read_envelope_data(graph_path, expected_type="claim_graph")
    except Exception:
        return unavailable_claim_graph_summary(
            root=root,
            paper_id=paper_id,
            artifact_manifest=artifact_manifest,
            quality=quality if isinstance(quality, dict) else {},
            memory_view=memory_view if isinstance(memory_view, dict) else {},
        )
    if not isinstance(dom_payload, dict) or not isinstance(graph_payload, dict):
        return unavailable_claim_graph_summary(
            root=root,
            paper_id=paper_id,
            artifact_manifest=artifact_manifest,
            quality=quality if isinstance(quality, dict) else {},
            memory_view=memory_view if isinstance(memory_view, dict) else {},
        )
    try:
        dom = PaperDOM.model_validate(dom_payload)
        graph = ClaimGraph.model_validate(graph_payload)
    except Exception:
        return unavailable_claim_graph_summary(
            root=root,
            paper_id=paper_id,
            artifact_manifest=artifact_manifest,
            quality=quality if isinstance(quality, dict) else {},
            memory_view=memory_view if isinstance(memory_view, dict) else {},
        )
    reading_plan = read_optional_reading_plan(root / "reading_plan.v2.json")
    return summarize_claim_graph_for_library(
        dom=dom,
        graph=graph,
        reading_plan=reading_plan,
        quality=quality if isinstance(quality, dict) else {},
        memory_view=memory_view if isinstance(memory_view, dict) else {},
        artifact_manifest=artifact_manifest,
        root=root,
    )


def unavailable_claim_graph_summary(
    *,
    root: Path,
    paper_id: str,
    artifact_manifest: dict[str, Any],
    quality: dict[str, Any],
    memory_view: dict[str, Any],
) -> dict[str, Any]:
    metadata = dict_value(memory_view.get("metadata"))
    return {
        "schema_version": GRAPH_LIBRARY_SUMMARY_SCHEMA_VERSION,
        "paper_id": paper_id,
        "metadata": {
            "title": metadata.get("title") or paper_id,
            "year": metadata.get("year"),
            "grade": metadata.get("grade"),
        },
        "source": {
            "paper_dom": relative_core_path(root / "paper_dom.v2.json"),
            "claim_graph": relative_core_path(root / "claim_graph.v2.json"),
            "quality_metrics": relative_core_path(root / "quality_metrics.v2.json"),
            "paper_memory_view": relative_core_path(root / "paper_memory_view.v2.json"),
        },
        "quality": unavailable_graph_quality_summary(
            quality=quality,
            artifact_manifest=artifact_manifest,
            memory_view=memory_view,
        ),
        "node_counts": {kind: 0 for kind in [*GRAPH_LIBRARY_NODE_KINDS, "evidence"]},
        "graph_access": graph_non_consumable_policy(artifact_manifest),
        "problem_nodes": [],
        "claim_nodes": [],
        "method_family": [],
        "mechanism_nodes": [],
        "implementation_nodes": [],
        "evaluation_nodes": [],
        "result_nodes": [],
        "limitation_nodes": [],
        "concept_nodes": [],
        "evaluation_datasets": [],
        "evaluation_metrics": [],
        "evaluation_dataset_mentions": [],
        "evaluation_metric_mentions": [],
        "relations": [],
    }


def summarize_claim_graph_for_library(
    *,
    dom: PaperDOM,
    graph: ClaimGraph,
    reading_plan: ReadingPlan | None,
    quality: dict[str, Any],
    memory_view: dict[str, Any],
    artifact_manifest: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    metadata = dict_value(memory_view.get("metadata"))
    current_audit_findings = [
        *audit_claim_graph(graph, dom),
        *audit_reading_required_outputs(graph, reading_plan),
    ]
    current_publish_status = publish_status_from_findings(current_audit_findings).value
    quality_summary = graph_quality_summary(
        quality=quality,
        memory_view=memory_view,
        artifact_manifest=artifact_manifest,
        current_audit_findings=current_audit_findings,
        current_publish_status=current_publish_status,
    )
    node_counts = {
        kind: len([node for node in graph.nodes.values() if node.kind == kind])
        for kind in [*GRAPH_LIBRARY_NODE_KINDS, "evidence"]
    }
    base_summary: dict[str, Any] = {
        "schema_version": GRAPH_LIBRARY_SUMMARY_SCHEMA_VERSION,
        "paper_id": dom.paper_id,
        "metadata": {
            "title": metadata.get("title") or dom.title,
            "year": metadata.get("year"),
            "grade": metadata.get("grade"),
        },
        "source": {
            "paper_dom": relative_core_path(root / "paper_dom.v2.json"),
            "claim_graph": relative_core_path(root / "claim_graph.v2.json"),
            "quality_metrics": relative_core_path(root / "quality_metrics.v2.json"),
            "paper_memory_view": relative_core_path(root / "paper_memory_view.v2.json"),
        },
        "quality": quality_summary,
        "node_counts": node_counts,
    }
    if not graph_summary_is_consumable(artifact_manifest):
        return {
            **base_summary,
            "graph_access": graph_non_consumable_policy(artifact_manifest),
            "problem_nodes": [],
            "claim_nodes": [],
            "method_family": [],
            "mechanism_nodes": [],
            "implementation_nodes": [],
            "evaluation_nodes": [],
            "result_nodes": [],
            "limitation_nodes": [],
            "concept_nodes": [],
            "evaluation_datasets": [],
            "evaluation_metrics": [],
            "evaluation_dataset_mentions": [],
            "evaluation_metric_mentions": [],
            "relations": [],
        }
    current_access = current_graph_non_consumable_policy(current_publish_status)
    if current_access:
        return {
            **base_summary,
            "graph_access": current_access,
            "problem_nodes": [],
            "claim_nodes": [],
            "method_family": [],
            "mechanism_nodes": [],
            "implementation_nodes": [],
            "evaluation_nodes": [],
            "result_nodes": [],
            "limitation_nodes": [],
            "concept_nodes": [],
            "evaluation_datasets": [],
            "evaluation_metrics": [],
            "evaluation_dataset_mentions": [],
            "evaluation_metric_mentions": [],
            "relations": [],
        }
    source_index = core_v2_source_index(dom)
    nodes_by_kind = {
        kind: [
            summarize_graph_node(node, graph=graph, source_index=source_index)
            for node in graph.nodes.values()
            if node.kind == kind
        ]
        for kind in GRAPH_LIBRARY_NODE_KINDS
    }
    method_nodes = [item for kind in METHOD_NODE_KINDS for item in nodes_by_kind.get(kind, [])]
    claim_nodes = [item for kind in CLAIM_NODE_KINDS for item in nodes_by_kind.get(kind, [])]
    evaluation_nodes = [
        item for kind in EVALUATION_NODE_KINDS for item in nodes_by_kind.get(kind, [])
    ]
    dataset_mentions = extract_term_mentions(
        evaluation_nodes,
        extractor=extract_dataset_terms,
    )
    metric_mentions = extract_term_mentions(
        evaluation_nodes,
        extractor=extract_metric_terms,
    )
    return {
        **base_summary,
        "graph_access": "readable",
        "problem_nodes": nodes_by_kind.get("problem", [])[:6],
        "claim_nodes": nodes_by_kind.get("claim", [])[:10],
        "method_family": compact_labels(method_nodes, limit=8),
        "mechanism_nodes": nodes_by_kind.get("mechanism", [])[:8],
        "implementation_nodes": nodes_by_kind.get("implementation", [])[:6],
        "evaluation_nodes": nodes_by_kind.get("evaluation", [])[:8],
        "result_nodes": nodes_by_kind.get("result", [])[:8],
        "limitation_nodes": nodes_by_kind.get("limitation", [])[:8],
        "concept_nodes": nodes_by_kind.get("concept", [])[:8],
        "evaluation_datasets": [item["term"] for item in dataset_mentions[:12]],
        "evaluation_metrics": [item["term"] for item in metric_mentions[:12]],
        "evaluation_dataset_mentions": dataset_mentions[:12],
        "evaluation_metric_mentions": metric_mentions[:12],
        "relations": (
            graph_relationship_edges(
                memory_view=memory_view,
                graph=graph,
                source_index=source_index,
            )
            or relation_nodes([*claim_nodes, *method_nodes, *evaluation_nodes])
        )[:8],
    }


def graph_quality_summary(
    *,
    quality: dict[str, Any],
    memory_view: dict[str, Any],
    artifact_manifest: dict[str, Any],
    current_audit_findings: list[AuditFinding],
    current_publish_status: str,
) -> dict[str, Any]:
    artifact_publish_status = artifact_manifest.get("artifact_publish_status") or quality.get(
        "publish_status"
    )
    manifest_current_publish_status = (
        artifact_manifest.get("current_audit_publish_status") or current_publish_status
    )
    if artifact_manifest.get("status") == "COMPLETE":
        effective_publish_status = (
            artifact_manifest.get("publish_status")
            or manifest_current_publish_status
            or artifact_publish_status
        )
    else:
        effective_publish_status = artifact_manifest.get("publish_status")
    current_issue_codes = artifact_manifest.get("current_audit_issue_codes")
    if not isinstance(current_issue_codes, list):
        current_issue_codes = sorted({finding.code for finding in current_audit_findings})
    return {
        "artifact_set_status": artifact_manifest.get("status"),
        "artifact_set_consumable": artifact_manifest.get("consumable"),
        "artifact_set_issues": artifact_manifest.get("issues", []),
        "publish_status": effective_publish_status,
        "artifact_publish_status": artifact_publish_status,
        "current_audit_publish_status": manifest_current_publish_status,
        "current_audit_error_count": first_present(
            artifact_manifest.get("current_audit_error_count"),
            current_audit_finding_count(current_audit_findings, severity="ERROR"),
        ),
        "current_audit_warning_count": first_present(
            artifact_manifest.get("current_audit_warning_count"),
            current_audit_finding_count(current_audit_findings, severity="WARNING"),
        ),
        "current_audit_issue_codes": current_issue_codes,
        "memory_report_readiness": memory_view.get("report_readiness"),
        "evidence_coverage": quality.get("evidence_coverage"),
        "reading_required_output_coverage": quality.get("reading_required_output_coverage"),
        "reading_required_output_count": quality.get("reading_required_output_count"),
        "reading_required_output_covered_count": quality.get(
            "reading_required_output_covered_count"
        ),
        "fact_node_count": quality.get("fact_node_count"),
        "supported_fact_node_count": quality.get("supported_fact_node_count"),
        "audit_error_count": quality.get("audit_error_count"),
        "audit_warning_count": quality.get("audit_warning_count"),
    }


def unavailable_graph_quality_summary(
    *,
    quality: dict[str, Any],
    artifact_manifest: dict[str, Any],
    memory_view: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact_set_status": artifact_manifest.get("status"),
        "artifact_set_consumable": artifact_manifest.get("consumable"),
        "artifact_set_issues": artifact_manifest.get("issues", []),
        "publish_status": artifact_manifest.get("publish_status"),
        "artifact_publish_status": artifact_manifest.get("artifact_publish_status")
        or quality.get("publish_status"),
        "current_audit_publish_status": artifact_manifest.get("current_audit_publish_status"),
        "current_audit_error_count": artifact_manifest.get("current_audit_error_count"),
        "current_audit_warning_count": artifact_manifest.get("current_audit_warning_count"),
        "current_audit_issue_codes": artifact_manifest.get("current_audit_issue_codes", []),
        "memory_report_readiness": memory_view.get("report_readiness"),
        "evidence_coverage": quality.get("evidence_coverage"),
        "reading_required_output_coverage": quality.get("reading_required_output_coverage"),
        "reading_required_output_count": quality.get("reading_required_output_count"),
        "reading_required_output_covered_count": quality.get(
            "reading_required_output_covered_count"
        ),
        "fact_node_count": quality.get("fact_node_count"),
        "supported_fact_node_count": quality.get("supported_fact_node_count"),
        "audit_error_count": quality.get("audit_error_count"),
        "audit_warning_count": quality.get("audit_warning_count"),
    }


def graph_summary_is_consumable(artifact_manifest: dict[str, Any]) -> bool:
    return artifact_manifest.get("consumable") is True


def graph_non_consumable_policy(artifact_manifest: dict[str, Any]) -> str:
    issues = set(str(issue) for issue in artifact_manifest.get("issues", []))
    if "missing:core_manifest.v2.json" in issues:
        return "missing_core_v2_manifest"
    if "missing:paper_dom.v2.json" in issues:
        return "missing_core_v2_paper_dom"
    if "missing:claim_graph.v2.json" in issues:
        return "missing_core_v2_claim_graph"
    if "missing:quality_metrics.v2.json" in issues:
        return "missing_core_v2_quality_metrics"
    artifact_publish_status = str(artifact_manifest.get("artifact_publish_status") or "")
    publish_status = str(
        artifact_manifest.get("current_audit_publish_status")
        or artifact_manifest.get("publish_status")
        or ""
    )
    if publish_status == "BLOCKED":
        if artifact_publish_status == "BLOCKED":
            return "blocked_by_core_v2_audit"
        return "blocked_by_current_graph_audit"
    if artifact_manifest.get("current_audit_publish_status"):
        if artifact_publish_status == publish_status:
            return "not_reviewed_by_core_v2_audit"
        return "not_reviewed_by_current_graph_audit"
    return "not_reviewed_by_core_v2_audit"


def current_graph_non_consumable_policy(current_publish_status: str) -> str | None:
    if current_publish_status in CONSUMABLE_GRAPH_STATUSES:
        return None
    if current_publish_status == "BLOCKED":
        return "blocked_by_current_graph_audit"
    return "not_reviewed_by_current_graph_audit"


def current_audit_finding_count(
    findings: list[AuditFinding],
    *,
    severity: str,
) -> int:
    return sum(1 for finding in findings if finding.severity == severity)


def first_present(value: Any, fallback: Any) -> Any:
    return fallback if value is None else value


def summarize_graph_node(
    node: GraphNode,
    *,
    graph: ClaimGraph,
    source_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    evidence_ids = graph.evidence_ids_for(node.node_id)
    source_ids = []
    pages = []
    evidence_samples = []
    for evidence_id in evidence_ids:
        evidence_node = graph.nodes.get(evidence_id)
        source_id = str((evidence_node.payload if evidence_node else {}).get("source_id") or "")
        source = source_index.get(source_id)
        if not source:
            continue
        source_ids.append(source_id)
        page_no = source.get("page_no")
        if isinstance(page_no, int) and page_no not in pages:
            pages.append(page_no)
        evidence_samples.append(
            {
                "source_id": source_id,
                "page_no": page_no,
                "text": source.get("text") or source.get("caption") or source.get("equation") or "",
            }
        )
    return {
        "node_id": node.node_id,
        "kind": node.kind,
        "label": compact_text(node.label, max_chars=420),
        "confidence": node.payload.get("confidence"),
        "provenance": node.payload.get("provenance"),
        "uncertainty": node.payload.get("uncertainty"),
        "evidence_ids": evidence_ids[:8],
        "source_ids": source_ids[:8],
        "pages": pages[:8],
        "evidence_samples": evidence_samples[:3],
    }


def build_graph_summary_search_text(summary: dict[str, Any]) -> str:
    if not summary:
        return ""
    parts = [
        json.dumps(summary.get("metadata", {}), ensure_ascii=False),
        json.dumps(summary.get("method_family", []), ensure_ascii=False),
        json.dumps(summary.get("evaluation_datasets", []), ensure_ascii=False),
        json.dumps(summary.get("evaluation_metrics", []), ensure_ascii=False),
        json.dumps(summary.get("evaluation_dataset_mentions", []), ensure_ascii=False),
        json.dumps(summary.get("evaluation_metric_mentions", []), ensure_ascii=False),
        json.dumps(summary.get("relations", []), ensure_ascii=False),
    ]
    for key in [
        "problem_nodes",
        "claim_nodes",
        "mechanism_nodes",
        "implementation_nodes",
        "evaluation_nodes",
        "result_nodes",
        "limitation_nodes",
        "concept_nodes",
    ]:
        for item in list_payload(summary.get(key)):
            parts.append(str(item.get("label") or ""))
            parts.extend(
                str(source.get("text") or "")
                for source in list_payload(item.get("evidence_samples"))
            )
    return "\n".join(part for part in parts if part)


def first_graph_label(summary: dict[str, Any], *keys: str) -> str:
    for key in keys:
        for node in list_payload(summary.get(key)):
            label = string_or_empty(node.get("label"))
            if label:
                return label
    return ""


def graph_node_labels(summary: dict[str, Any], *keys: str) -> list[str]:
    labels = []
    for key in keys:
        for node in list_payload(summary.get(key)):
            label = compact_text(node.get("label"), max_chars=280)
            if label and label not in labels:
                labels.append(label)
            if len(labels) >= 10:
                return labels
    return labels


def normalize_graph_claims(summary: dict[str, Any]) -> list[dict[str, Any]]:
    claims = []
    for node in [
        *list_payload(summary.get("claim_nodes")),
        *list_payload(summary.get("problem_nodes")),
    ]:
        label = string_or_empty(node.get("label"))
        if not label:
            continue
        claims.append(
            {
                "claim": label,
                "confidence": string_or_empty(node.get("confidence")) or "medium",
                "evidence_pages": [page for page in node.get("pages", []) if isinstance(page, int)][
                    :8
                ],
                "node_id": string_or_empty(node.get("node_id")),
                "source_ids": [
                    source_id
                    for source_id in node.get("source_ids", [])
                    if isinstance(source_id, str)
                ][:8],
            }
        )
        if len(claims) >= 12:
            break
    return claims


def normalize_graph_concepts(summary: dict[str, Any]) -> list[dict[str, str]]:
    concepts = []
    for label in dict.fromkeys(summary.get("method_family", []) or []):
        text = string_or_empty(label)
        if not text:
            continue
        concepts.append({"term": compact_text(text, max_chars=80), "explanation": text})
        if len(concepts) >= 8:
            break
    return concepts


def normalize_graph_evidence(summary: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = []
    for node in graph_summary_nodes(summary):
        label = string_or_empty(node.get("label"))
        for sample in list_payload(node.get("evidence_samples")):
            source_id = string_or_empty(sample.get("source_id"))
            quote = string_or_none(sample.get("text"))
            if not source_id:
                continue
            evidence.append(
                {
                    "page_no": int_or_none(sample.get("page_no")),
                    "claim": label,
                    "quote": quote,
                    "source_id": source_id,
                    "node_id": string_or_empty(node.get("node_id")),
                }
            )
            if len(evidence) >= 12:
                return evidence
    return evidence


def graph_provenance(summary: dict[str, Any]) -> dict[str, Any]:
    if not summary:
        return {}
    node_ids = []
    source_ids = []
    for node in graph_summary_nodes(summary):
        node_id = string_or_empty(node.get("node_id"))
        if node_id and node_id not in node_ids:
            node_ids.append(node_id)
        for source_id in (
            node.get("source_ids", []) if isinstance(node.get("source_ids"), list) else []
        ):
            source_text = string_or_empty(source_id)
            if source_text and source_text not in source_ids:
                source_ids.append(source_text)
    return {
        "paths": dict_value(summary.get("source")),
        "node_ids": node_ids[:32],
        "source_ids": source_ids[:48],
        "publish_status": dict_value(summary.get("quality")).get("publish_status"),
    }


def graph_summary_tags(summary: dict[str, Any]) -> list[str]:
    if not summary:
        return []
    text = " ".join(
        [
            " ".join(string_or_empty(item) for item in summary.get("method_family", []) or []),
            " ".join(
                string_or_empty(item) for item in summary.get("evaluation_datasets", []) or []
            ),
            " ".join(string_or_empty(item) for item in summary.get("evaluation_metrics", []) or []),
        ]
    )
    return [token for token in tokenize_for_search(text) if not token.isdigit()][:10]


def compact_graph_summary_for_index(value: Any) -> dict[str, Any]:
    summary = dict_value(value)
    if not summary:
        return {}
    return {
        "schema_version": summary.get("schema_version"),
        "graph_access": summary.get("graph_access"),
        "source": dict_value(summary.get("source")),
        "quality": dict_value(summary.get("quality")),
        "node_counts": dict_value(summary.get("node_counts")),
        "problem_nodes": compact_graph_nodes(summary.get("problem_nodes"))[:4],
        "claim_nodes": compact_graph_nodes(summary.get("claim_nodes"))[:8],
        "method_family": (summary.get("method_family") or [])[:6]
        if isinstance(summary.get("method_family"), list)
        else [],
        "mechanism_nodes": compact_graph_nodes(summary.get("mechanism_nodes"))[:6],
        "evaluation_datasets": (summary.get("evaluation_datasets") or [])[:8]
        if isinstance(summary.get("evaluation_datasets"), list)
        else [],
        "evaluation_metrics": (summary.get("evaluation_metrics") or [])[:8]
        if isinstance(summary.get("evaluation_metrics"), list)
        else [],
        "evaluation_dataset_mentions": compact_term_mentions(
            summary.get("evaluation_dataset_mentions")
        ),
        "evaluation_metric_mentions": compact_term_mentions(
            summary.get("evaluation_metric_mentions")
        ),
        "relations": summary.get("relations", [])[:8]
        if isinstance(summary.get("relations"), list)
        else [],
    }


def compact_graph_summary_for_agent(value: Any) -> dict[str, Any]:
    summary = dict_value(value)
    if not summary:
        return {}
    return {
        **compact_graph_summary_for_index(summary),
        "problem_nodes": compact_graph_nodes(summary.get("problem_nodes")),
        "claim_nodes": compact_graph_nodes(summary.get("claim_nodes")),
        "mechanism_nodes": compact_graph_nodes(summary.get("mechanism_nodes")),
        "evaluation_nodes": compact_graph_nodes(summary.get("evaluation_nodes")),
        "result_nodes": compact_graph_nodes(summary.get("result_nodes")),
        "limitation_nodes": compact_graph_nodes(summary.get("limitation_nodes")),
        "relations": summary.get("relations", [])[:8]
        if isinstance(summary.get("relations"), list)
        else [],
    }


def compact_graph_nodes(value: Any) -> list[dict[str, Any]]:
    nodes = []
    for node in list_payload(value)[:8]:
        nodes.append(
            {
                "node_id": node.get("node_id"),
                "kind": node.get("kind"),
                "label": compact_text(node.get("label"), max_chars=260),
                "source_ids": node.get("source_ids", [])[:6]
                if isinstance(node.get("source_ids"), list)
                else [],
                "pages": node.get("pages", [])[:6] if isinstance(node.get("pages"), list) else [],
            }
        )
    return nodes


def compact_term_mentions(value: Any) -> list[dict[str, Any]]:
    mentions = []
    for item in list_payload(value)[:8]:
        mentions.append(
            {
                "term": item.get("term"),
                "node_ids": item.get("node_ids", [])[:6]
                if isinstance(item.get("node_ids"), list)
                else [],
                "source_ids": item.get("source_ids", [])[:6]
                if isinstance(item.get("source_ids"), list)
                else [],
                "pages": item.get("pages", [])[:6] if isinstance(item.get("pages"), list) else [],
            }
        )
    return mentions


def graph_summary_nodes(summary: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = []
    for key in [
        "problem_nodes",
        "claim_nodes",
        "mechanism_nodes",
        "implementation_nodes",
        "evaluation_nodes",
        "result_nodes",
        "limitation_nodes",
        "concept_nodes",
    ]:
        nodes.extend(list_payload(summary.get(key)))
    return nodes


def core_v2_source_index(dom: PaperDOM) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for span in dom.spans:
        result[span.source_id] = {
            "source_id": span.source_id,
            "kind": span.kind,
            "page_no": span.page_no,
            "section_id": span.section_id,
            "text": compact_text(span.text, max_chars=1000),
        }
    for figure in dom.figures:
        result[figure.source_id] = {
            "source_id": figure.source_id,
            "kind": figure.kind,
            "page_no": figure.page_no,
            "caption": compact_text(figure.caption or "", max_chars=800),
        }
    for table in dom.tables:
        result[table.source_id] = {
            "source_id": table.source_id,
            "kind": table.kind,
            "page_no": table.page_no,
            "caption": compact_text(table.caption or "", max_chars=800),
        }
    for equation in dom.equations:
        result[equation.source_id] = {
            "source_id": equation.source_id,
            "kind": equation.kind,
            "page_no": equation.page_no,
            "equation": compact_text(equation.latex_or_text, max_chars=800),
        }
    return result


def read_envelope_data(path: Path, *, expected_type: str) -> dict[str, Any] | list[Any]:
    envelope = ArtifactEnvelope.model_validate_json(path.read_text(encoding="utf-8"))
    return envelope.require_type(expected_type).data


def read_optional_envelope_data(path: Path, expected_type: str) -> dict[str, Any] | list[Any]:
    if not path.exists():
        return {}
    try:
        return read_envelope_data(path, expected_type=expected_type)
    except Exception:
        return {}


def read_optional_reading_plan(path: Path) -> ReadingPlan | None:
    data = read_optional_envelope_data(path, "reading_plan")
    if not isinstance(data, dict):
        return None
    try:
        return ReadingPlan.model_validate(data)
    except Exception:
        return None


def compact_labels(nodes: list[dict[str, Any]], *, limit: int) -> list[str]:
    labels = []
    for node in nodes:
        label = compact_text(node.get("label"), max_chars=220)
        if label and label not in labels:
            labels.append(label)
        if len(labels) >= limit:
            break
    return labels


def relation_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for node in nodes:
        label = str(node.get("label") or "")
        lowered = label.lower()
        if not any(term in lowered for term in RELATION_TERMS):
            continue
        result.append(
            {
                "node_id": node.get("node_id"),
                "label": compact_text(label, max_chars=260),
                "source_ids": node.get("source_ids", [])[:6],
                "pages": node.get("pages", [])[:6],
            }
        )
    return result


def graph_relationship_edges(
    *,
    memory_view: dict[str, Any],
    graph: ClaimGraph,
    source_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for edge in list_payload(memory_view.get("relationship_edges")):
        source_id = string_or_empty(edge.get("source_id"))
        target_id = string_or_empty(edge.get("target_id"))
        kind = string_or_empty(edge.get("kind"))
        source_node = graph.nodes.get(source_id)
        target_node = graph.nodes.get(target_id)
        if not source_node or not target_node or not kind:
            continue
        source_ids = []
        pages = []
        for node_id in [source_node.node_id, target_node.node_id]:
            for evidence_id in graph.evidence_ids_for(node_id):
                evidence_node = graph.nodes.get(evidence_id)
                paper_source_id = string_or_empty(
                    (evidence_node.payload if evidence_node else {}).get("source_id")
                )
                if paper_source_id and paper_source_id not in source_ids:
                    source_ids.append(paper_source_id)
                page_no = source_index.get(paper_source_id, {}).get("page_no")
                if isinstance(page_no, int) and page_no not in pages:
                    pages.append(page_no)
        result.append(
            {
                "source_id": source_node.node_id,
                "source_kind": source_node.kind,
                "source_label": compact_text(source_node.label, max_chars=220),
                "target_id": target_node.node_id,
                "target_kind": target_node.kind,
                "target_label": compact_text(target_node.label, max_chars=220),
                "kind": kind,
                "source_ids": source_ids[:8],
                "pages": pages[:8],
            }
        )
    return result


def extract_dataset_terms(text: str) -> list[str]:
    terms = []
    for pattern in [
        r"\bDataset[-_ ][A-Za-z0-9_.-]+\b",
        r"\b[A-Z][A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)+\b",
        r"\bon\s+([A-Z][A-Za-z0-9_.-]{2,})\b",
    ]:
        for match in re.finditer(pattern, text):
            value = match.group(1) if match.groups() else match.group(0)
            add_unique(terms, value.strip())
    return terms


def extract_metric_terms(text: str) -> list[str]:
    lowered = text.lower()
    terms = [term for term in METRIC_TERMS if term in lowered]
    for match in re.finditer(r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?%?(?![A-Za-z0-9_])", text):
        add_unique(terms, match.group(0))
    return terms


def extract_term_mentions(
    nodes: list[dict[str, Any]],
    *,
    extractor: Any,
) -> list[dict[str, Any]]:
    mentions: dict[str, dict[str, Any]] = {}
    for node in nodes:
        node_id = string_or_empty(node.get("node_id"))
        source_ids = [
            source_id for source_id in node.get("source_ids", []) if isinstance(source_id, str)
        ]
        pages = [page for page in node.get("pages", []) if isinstance(page, int)]
        text_parts = [string_or_empty(node.get("label"))]
        for sample in list_payload(node.get("evidence_samples")):
            text_parts.append(
                string_or_empty(sample.get("text"))
                or string_or_empty(sample.get("caption"))
                or string_or_empty(sample.get("equation"))
            )
        for term in extractor(" ".join(part for part in text_parts if part)):
            entry = mentions.setdefault(
                term,
                {
                    "term": term,
                    "node_ids": [],
                    "source_ids": [],
                    "pages": [],
                },
            )
            if node_id:
                add_unique(entry["node_ids"], node_id)
            for source_id in source_ids:
                add_unique(entry["source_ids"], source_id)
            for page in pages:
                if page not in entry["pages"]:
                    entry["pages"].append(page)
    return list(mentions.values())


def relative_core_path(path: Path) -> str:
    parts = path.parts
    if ".paperlens" not in parts:
        return str(path)
    index = parts.index(".paperlens")
    return "/".join(parts[index:])


def compact_text(value: Any, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def add_unique(target: list[str], value: str) -> None:
    if value and value not in target:
        target.append(value)


def tokenize_for_search(text: Any) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text or "").lower()).strip()
    return re.findall(r"[a-z0-9_+.-]{2,}|[\u4e00-\u9fff]{2,}", normalized)


def string_or_none(value: Any) -> str | None:
    text = string_or_empty(value)
    return text or None


def string_or_empty(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def int_or_none(value: Any) -> int | None:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer > 0 else None


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_payload(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []
