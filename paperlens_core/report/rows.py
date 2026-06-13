from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

from paperlens_core.schemas import ClassificationDecision, PaperRecord, SkimCard

CLASS_LABEL_SORT_RANK = {"A": 0, "HOLD": 1, "B": 2, "C": 3}


def _report_slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return cleaned[:48] or "evidence"


def paper_report_filename(paper: PaperRecord) -> str:
    return f"{paper.paper_id}_{_report_slug(paper.canonical_title or paper.paper_id)[:60]}.md"


def markdown_link(label: str, href: str) -> str:
    return f"[{markdown_link_label(label)}]({href})"


def markdown_link_label(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").replace("\\", "\\\\"))
    return text.replace("[", "\\[").replace("]", "\\]").strip()


def report_href(report_name: Any, *, prefix: str = "papers") -> str:
    name = str(report_name or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    safe_name = quote(name or "report.md", safe="")
    return f"{prefix.rstrip('/')}/{safe_name}"


def row_decision(row: dict[str, Any]) -> ClassificationDecision:
    decision = row.get("decision")
    if isinstance(decision, ClassificationDecision):
        return decision
    return ClassificationDecision(
        paper_id=row["paper"].paper_id,
        class_label="HOLD",
        confidence=0.0,
        false_negative_risk=1.0,
        reason_codes=["missing_classification"],
    )


def read_effort_rank(label: str) -> int:
    return {"C": 0, "B": 1, "HOLD": 2, "A": 3}.get(label, 2)


def higher_read_effort_label(left: str, right: str) -> str:
    return left if read_effort_rank(left) >= read_effort_rank(right) else right


def reading_priority_key(row: dict[str, Any]) -> tuple[int, float, str]:
    decision = row_decision(row)
    class_rank = CLASS_LABEL_SORT_RANK.get(decision.class_label, CLASS_LABEL_SORT_RANK["HOLD"])
    return (class_rank, -decision.false_negative_risk, row["paper"].paper_id)


def read_decision(row: dict[str, Any]) -> str:
    decision = row_decision(row)
    if decision.class_label in {"A", "HOLD"}:
        return "Close read"
    if decision.class_label == "B":
        return "Selective read"
    if decision.false_negative_risk >= 0.45:
        return "Background"
    return "Skip"


def novelty_risk(row: dict[str, Any]) -> str:
    decision = row_decision(row)
    skim = row.get("skim")
    signal_count = len(skim.danger_signals) if isinstance(skim, SkimCard) else 0
    if decision.class_label == "A" or decision.false_negative_risk >= 0.7:
        return "HIGH"
    if decision.class_label in {"B", "HOLD"} or signal_count:
        return "MEDIUM"
    return "LOW"


def row_relation(row: dict[str, Any]) -> str:
    skim = row.get("skim")
    decision = row_decision(row)
    if isinstance(skim, SkimCard):
        scope = skim.system_scope or "unknown scope"
        method = skim.method_type or "unknown method"
        signals = (
            ", ".join(skim.danger_signals) if skim.danger_signals else "no explicit danger signals"
        )
        return f"{decision.class_label}-class paper; method={method}; scope={scope}; value signals={signals}."
    return f"{decision.class_label}-class paper with limited extracted context."


def report_link_lines(rows: list[dict[str, Any]]) -> list[str]:
    result = []
    for row in rows:
        paper = row["paper"]
        label = f"{paper.paper_id}: {paper.canonical_title or paper.paper_id}"
        result.append(
            f"- {markdown_link(label, report_href(row.get('report_name')))}"
        )
    return result


def describe_rows(rows: list[dict[str, Any]]) -> list[str]:
    result = []
    for row in rows:
        paper = row["paper"]
        label = f"{paper.paper_id}: {paper.canonical_title or paper.paper_id}"
        result.append(
            f"- {markdown_link(label, report_href(row.get('report_name')))} - {row_relation(row)}"
        )
    return result


def cluster_rows_by_scope(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    clusters: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        skim = row.get("skim")
        if not isinstance(skim, SkimCard):
            cluster = "Unknown"
        else:
            cluster = skim.system_scope or skim.method_type or "Unknown"
        clusters.setdefault(cluster, []).append(row)
    return clusters


def classification_counts(decisions: list[ClassificationDecision]) -> dict[str, int]:
    counts = {"A": 0, "B": 0, "C": 0, "HOLD": 0}
    for decision in decisions:
        counts[decision.class_label] = counts.get(decision.class_label, 0) + 1
    return counts
