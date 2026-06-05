from __future__ import annotations

import os
import re
from typing import Any

from paperlens_core.schemas import ClassificationDecision, SkimCard


def select_rolling_read_pages(
    artifacts: list[Any],
    skim: SkimCard,
    decision: ClassificationDecision,
    *,
    read_mode: str = "standard",
) -> list[Any]:
    _ = (decision, read_mode)
    pages = [page for page in artifacts if _normalize_excerpt(getattr(page, "text", ""), limit=80)]
    if not pages:
        return []
    max_pages_env = os.getenv("PAPERLENS_ROLLING_MAX_PAGES")
    if max_pages_env is None:
        return pages
    default_max_pages = "14"
    try:
        max_pages = int(max_pages_env)
    except ValueError:
        max_pages = int(default_max_pages)
    max_pages = max(1, min(max_pages, 24))
    by_no = {page.page_no: page for page in pages}
    selected: list[int] = []

    def add(page_no: int | None) -> None:
        if page_no and page_no in by_no and page_no not in selected:
            selected.append(page_no)

    for page in pages[:3]:
        add(page.page_no)
    for ref in skim.evidence_refs:
        add(ref.page_no)
    keywords = [
        "abstract",
        "introduction",
        "overview",
        "design",
        "implementation",
        "evaluation",
        "experiment",
        "limitation",
        "conclusion",
    ]
    for page in pages:
        haystack = _normalize_for_search(
            " ".join([page.text[:1400]] + [str(c.get("text") or "") for c in page.captions[:4]])
        )
        if any(keyword in haystack for keyword in keywords):
            add(page.page_no)
        if len(selected) >= max_pages - 1:
            break
    for page in pages[-2:]:
        add(page.page_no)
    return [by_no[page_no] for page_no in selected[:max_pages]]


def _normalize_excerpt(text: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0] + " ..."


def _normalize_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()
