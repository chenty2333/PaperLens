from __future__ import annotations

import re
from typing import Any


def recommendation_for_grade(grade: str) -> str:
    return {"A": "重点关注", "B": "标准读", "C": "低优先级", "HOLD": "需确认"}.get(grade, "需确认")


def compact_reason(text: str, *, max_chars: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    for mark in "。！？.!?；;，,":
        index = cleaned.rfind(mark, 0, max_chars)
        if index >= 40:
            return cleaned[: index + 1]
    return cleaned[:max_chars].rstrip() + "..."


def clean_model_markdown(value: Any) -> str:
    text = value.strip() if isinstance(value, str) else ""
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    text = repair_markdown_boundaries(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def repair_markdown_boundaries(text: str) -> str:
    sentence_boundary = r"([。！？!?；;：:])"
    text = re.sub(sentence_boundary + r"\s*(#{1,6}\s+)", r"\1\n\n\2", text)
    text = re.sub(r"([^\n])\s+(#{1,6}\s+)", r"\1\n\n\2", text)
    text = re.sub(
        sentence_boundary + r"\s+((?:\d+\.|[-*])\s+(?:\*\*|[^\s]))",
        r"\1\n\n\2",
        text,
    )
    heading_prefixes = (
        "核心抽象|问题背景|机制|证据|实验|评估|价值|可迁移性|局限|限制|边界|"
        "误解防护|关键图表|来源边界"
    )
    text = re.sub(
        rf"(?m)^(#{{1,6}}\s+(?:{heading_prefixes})[^\s。:：\n]{{0,12}}(?:[:：])?)[ \t]+(?=\S)",
        r"\1\n\n",
        text,
    )
    return text


def readable_model_body(value: Any) -> str:
    text = clean_model_markdown(value)
    if not text:
        return ""
    paragraphs = [
        paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()
    ]
    rewritten = []
    for paragraph in paragraphs:
        if (
            len(paragraph) <= 900
            or "\n" in paragraph
            or paragraph.lstrip().startswith(("-", "*", "1."))
        ):
            rewritten.append(paragraph)
            continue
        sentences = re.findall(
            r".+?(?:[。！？!?；;](?=\s|[\u4e00-\u9fffA-Za-z0-9])|[。！？!?；;]$|$)",
            paragraph,
        )
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        groups: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            if current and current_len + len(sentence) > 520:
                groups.append("".join(current).strip())
                current = []
                current_len = 0
            current.append(sentence)
            current_len += len(sentence)
        if current:
            groups.append("".join(current).strip())
        rewritten.extend(groups or [paragraph])
    return "\n\n".join(rewritten).strip()


def clean_model_inline_text(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_model_markdown(value)).strip()


def compact_compare_text(text: str) -> str:
    return re.sub(r"\s+", "", text).strip().lower()


def sanitize_reader_hostile_text(text: str | None) -> str:
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


def user_facing_uncertainty_note(value: Any) -> str:
    """Remove report-planning/internal memory wording from reader-facing uncertainty."""
    text = clean_model_markdown(value)
    if not text:
        return ""
    internal_markers = (
        "本报告计划",
        "本报告",
        "报告计划",
        "paper memory",
        "papermemory",
        "report plan",
        "memoryv3",
        "将严格遵循",
        "will strictly follow",
        "plan is based",
    )
    chunks = re.split(r"(?:[;；]\s*|\n\s*\n)", text)
    kept: list[str] = []
    for chunk in chunks:
        cleaned = chunk.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(marker in lowered for marker in internal_markers):
            continue
        kept.append(cleaned)
    return "; ".join(kept)
