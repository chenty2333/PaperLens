from __future__ import annotations

import re
from typing import Any


GENERIC_BAD_PHRASES = [
    "这篇论文主要介绍了",
    "本文提出了一种方法",
    "具有一定参考价值",
    "未来工作可以进一步研究",
    "模型没有给出可用讲解",
    "deterministic fallback",
    "model final-report generation failed",
]

OLD_TEMPLATE_HEADINGS = [
    "## 结论",
    "## 核心思想",
    "## 论文怎么做的",
    "## 证据和实验",
    "## 我认为最有价值的点",
    "## 不确定和需要复查",
    "## 分级记录",
]

READER_HOSTILE_PHRASES = [
    "你给到",
    "你提供",
    "供给片段",
    "供给的片段",
    "提供的页面",
    "提供的材料",
    "提供的证据",
    "the supplied excerpts",
    "supplied excerpts",
    "the user provided",
    "provided excerpt",
]

VISIBLE_REPAIR_MARKERS = [
    "需要修正或确认",
    "需要补充",
    "repair TODO",
    "TODO",
    "unsupported_items",
    "missing_items",
]

QUALITY_SIGNAL_GROUPS = {
    "core_idea": ["核心", "idea", "abstraction", "抽象", "主张", "thesis"],
    "mechanism": ["机制", "做法", "设计", "system", "protocol", "algorithm", "scheduler"],
    "evidence": ["实验", "评估", "evaluation", "benchmark", "trace", "workload", "throughput", "latency"],
    "value": ["价值", "启发", "useful", "why it matters", "适合", "值得"],
    "limits": ["限制", "不确定", "假设", "不能证明", "limitation", "tradeoff"],
}


def evaluate_capsule_quality(
    text: str,
    *,
    expected_terms: list[str] | None = None,
    min_chars: int = 500,
    max_chars: int = 3200,
) -> dict[str, Any]:
    normalized = normalize_text(text)
    issues: list[str] = []
    score = 10.0

    if "\\n" in text:
        issues.append("escaped_newline")
        score -= 2.0
    if len(normalized) < min_chars:
        issues.append("too_short")
        score -= 2.0
    if len(normalized) > max_chars:
        issues.append("too_long")
        score -= min(2.0, (len(normalized) - max_chars) / 900)

    if contains_full_page_visual_embed(text):
        issues.append("full_page_visual_embed")
        score -= 3.0

    bad_hits = [phrase for phrase in GENERIC_BAD_PHRASES if phrase.lower() in normalized.lower()]
    if bad_hits:
        issues.append("generic_or_fallback_language")
        score -= min(3.0, len(bad_hits) * 1.0)

    hostile_hits = [phrase for phrase in READER_HOSTILE_PHRASES if phrase.lower() in normalized.lower()]
    if hostile_hits:
        issues.append("implementation_context_leak")
        score -= min(2.0, len(hostile_hits) * 0.7)

    repair_hits = [phrase for phrase in VISIBLE_REPAIR_MARKERS if phrase.lower() in normalized.lower()]
    if repair_hits:
        issues.append("visible_repair_or_audit_todo")
        score -= min(2.0, len(repair_hits) * 0.6)

    heading_hits = [heading for heading in OLD_TEMPLATE_HEADINGS if heading in text]
    if len(heading_hits) >= 3:
        issues.append("old_template_shape")
        score -= 1.5

    paragraph_lengths = [
        len(normalize_text(paragraph))
        for paragraph in re.split(r"\n\s*\n", text)
        if normalize_text(paragraph)
        and not normalize_text(paragraph).startswith(("#", ">", "<figure", "<img", "<figcaption", "</figure"))
    ]
    if any(length > 1400 for length in paragraph_lengths):
        issues.append("reader_hostile_wall_of_text")
        score -= 1.3
    if len(paragraph_lengths) >= 3 and max(paragraph_lengths) > sum(paragraph_lengths) * 0.65:
        issues.append("unbalanced_article_shape")
        score -= 0.8

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    bullet_lines = [line for line in nonempty_lines if line.startswith(("-", "*", "1.", "2.", "3.", "4.", "5."))]
    if len(nonempty_lines) >= 8 and len(bullet_lines) / len(nonempty_lines) > 0.55:
        issues.append("mostly_bullet_points")
        score -= 1.0

    signal_hits = {
        name: any(term.lower() in normalized.lower() for term in terms)
        for name, terms in QUALITY_SIGNAL_GROUPS.items()
    }
    missing_signals = [name for name, hit in signal_hits.items() if not hit]
    if missing_signals:
        issues.append("missing_quality_signals:" + ",".join(missing_signals))
        score -= min(2.5, len(missing_signals) * 0.5)

    expected_terms = expected_terms or []
    matched_terms = [term for term in expected_terms if term.lower() in normalized.lower()]
    if expected_terms:
        coverage = len(matched_terms) / len(expected_terms)
        if coverage < 0.4:
            issues.append("low_expected_term_coverage")
            score -= 2.0
        elif coverage < 0.7:
            issues.append("partial_expected_term_coverage")
            score -= 0.8

    return {
        "score": round(max(0.0, min(10.0, score)), 2),
        "issues": issues,
        "char_count": len(normalized),
        "signal_hits": signal_hits,
        "expected_terms": expected_terms,
        "matched_terms": matched_terms,
    }


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def contains_full_page_visual_embed(text: str) -> bool:
    return bool(
        re.search(
            r"""<img\s+[^>]*src=["'][^"']*\.paperlens/pages/[^"']*/page_\d{4}\.png["']""",
            text,
            flags=re.IGNORECASE,
        )
    )
