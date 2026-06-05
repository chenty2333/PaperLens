from __future__ import annotations

import re
from typing import Any

from paperlens_core.report.rows import (
    classification_counts,
    reading_priority_key,
    row_decision,
)
from paperlens_core.schemas import PaperCard, PaperRecord, ReviewItem, SkimCard


def render_paperlens_report(
    *,
    rows: list[dict[str, Any]],
    review_items: list[ReviewItem],
    budget: dict[str, Any],
    topic: str | None,
    idea: str | None,
    formal_run: bool,
    output_language: str = "zh",
) -> str:
    ranked = sorted(rows, key=reading_priority_key)
    counts = classification_counts([row_decision(row) for row in rows])
    visible_reviews = user_visible_review_items(review_items)
    if output_language == "en":
        lines = [
            "# PaperLens",
            "",
            (
                f"Read {len(rows)} papers with the same standard loop: {counts['A']} high priority, "
                f"{counts['B']} standard, {counts['C']} lower priority, {counts['HOLD']} need confirmation."
            ),
        ]
        if not formal_run:
            lines.append("This is an offline debug result for structure checks only.")
    else:
        lines = [
            "# PaperLens",
            "",
            (
                f"本次按同一标准流程阅读 {len(rows)} 篇：高优先级 {counts['A']} 篇，标准 {counts['B']} 篇，"
                f"低优先级 {counts['C']} 篇，待确认 {counts['HOLD']} 篇。"
            ),
        ]
        if not formal_run:
            lines.append("这是离线调试结果，只能检查解析和输出结构，不能作为论文判断。")
    if topic or idea:
        context = "；".join(item for item in [topic, idea] if item)
        lines.extend(
            ["", ("Reading goal: " if output_language == "en" else "阅读目标：") + context]
        )
    if output_language == "en":
        lines.extend(
            [
                "",
                "## Contents",
                "",
            ]
        )
    else:
        lines.extend(["", "## 目录", ""])

    for row in ranked:
        decision = row_decision(row)
        reason = one_line_row_reason(row)
        graph_link = core_graph_report_link(row, output_language=output_language)
        suffix = f"；{graph_link}" if graph_link and output_language != "en" else ""
        if graph_link and output_language == "en":
            suffix = f"; {graph_link}"
        lines.append(
            f"- [{decision.class_label}] [{display_row_title(row)}](./papers/{row['report_name']}) - {reason}{suffix}"
        )
    if visible_reviews:
        if output_language == "en":
            lines.extend(
                [
                    "",
                    f"Needs confirmation: {len(visible_reviews)} items.",
                ]
            )
        else:
            lines.extend(
                [
                    "",
                    f"需要确认：{len(visible_reviews)} 项。",
                ]
            )
    estimated_usd = float(budget.get("estimated_usd") or 0)
    call_count = int(budget.get("calls") or 0)
    if budget.get("estimated_usd") is not None and (estimated_usd > 0 or call_count > 0):
        label = "Estimated model cost" if output_language == "en" else "模型成本估算"
        lines.append(f"{label}: ${estimated_usd:.4f}.")
    return "\n".join(lines) + "\n"


def core_graph_report_link(row: dict[str, Any], *, output_language: str) -> str:
    name = _string_or_none(row.get("core_graph_report_name"))
    if not name:
        return ""
    label = "Core graph report" if output_language == "en" else "事实图报告"
    return f"[{label}](./papers/{name})"


def one_line_row_reason(row: dict[str, Any]) -> str:
    decision = row_decision(row)
    model_report = row.get("model_report")
    if isinstance(model_report, dict):
        reason = _sanitize_reader_hostile_text(
            _clean_model_inline_text(model_report.get("one_line_reason"))
        )
        if reason:
            return _compact_text(reason, max_chars=110)
    skim = row.get("skim")
    card = row.get("card")
    if isinstance(card, PaperCard) and card.contribution_claims:
        return _normalize_excerpt(card.contribution_claims[0], limit=90)
    if isinstance(skim, SkimCard) and skim.problem:
        return _normalize_excerpt(skim.problem, limit=90)
    if decision.reason_codes:
        return ", ".join(decision.reason_codes[:3])
    return "没有足够理由说明。"


def markdown_title(markdown: str) -> str | None:
    for line in markdown.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or None
    return None


def display_row_title(row: dict[str, Any]) -> str:
    paper = row["paper"]
    title = _string_or_none(row.get("report_title")) or display_paper_title(paper)
    return humanize_display_title(title, paper.paper_id)


def display_paper_title(paper: PaperRecord) -> str:
    return humanize_display_title(paper.canonical_title or paper.paper_id, paper.paper_id)


def humanize_display_title(title: str, fallback: str) -> str:
    if re.match(r"^\d+[_-]", title):
        title = re.sub(r"^\d+[_-]+", "", title).replace("_", " ").strip()
    if title.islower() and len(title.split()) <= 4:
        title = title.title()
    return title or fallback


def user_visible_review_items(review_items: list[ReviewItem]) -> list[ReviewItem]:
    return [item for item in review_items if not is_internal_review_noise(item)]


def is_internal_review_noise(item: ReviewItem) -> bool:
    reason = item.reason.strip()
    if item.item_type == "WEAK_EVIDENCE_BOUNDARY":
        return True
    hidden_reasons = {
        "skim_level_or_keyword_evidence",
        "visual_required_pages",
    }
    return reason in hidden_reasons


def _clean_model_inline_text(value: Any) -> str:
    text = value.strip() if isinstance(value, str) else ""
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    return re.sub(r"\s+", " ", text).strip()


def _sanitize_reader_hostile_text(text: str | None) -> str:
    if not text:
        return ""
    replacements = {
        "你给到的片段": "当前自动阅读证据",
        "你给到的摘录": "当前自动阅读证据",
        "你给到": "当前自动阅读证据",
        "你提供的片段": "当前自动阅读证据",
        "你提供的摘录": "当前自动阅读证据",
        "你提供": "当前自动阅读证据",
        "供给的片段": "当前自动阅读证据",
        "供给片段": "当前自动阅读证据",
        "供给的图示": "自动阅读证据中的图示",
        "提供的页面": "当前自动阅读证据",
        "提供的材料": "当前自动阅读证据",
        "提供的证据": "当前自动阅读证据",
        "the supplied excerpts": "the automatic reading evidence",
        "supplied excerpts": "automatic reading evidence",
        "provided excerpts": "automatic reading evidence",
        "provided excerpt": "automatic reading evidence",
        "the user provided": "the current evidence contains",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def _compact_text(text: str, *, max_chars: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    for mark in "。！？.!?；;，,":
        index = cleaned.rfind(mark, 0, max_chars)
        if index >= 40:
            return cleaned[: index + 1]
    return cleaned[:max_chars].rstrip() + "..."


def _normalize_excerpt(text: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0] + " ..."


def _string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned or None
