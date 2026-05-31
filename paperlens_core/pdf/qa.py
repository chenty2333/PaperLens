from __future__ import annotations

from collections import Counter
from typing import Any

from paperlens_core.schemas import PageArtifact


def parse_quality(artifacts: list[PageArtifact]) -> tuple[str, dict[str, Any]]:
    if not artifacts:
        return "FAIL", {"reason": "no_pages"}

    total_pages = len(artifacts)
    empty_pages = sum(1 for page in artifacts if len(page.text.strip()) < 50)
    total_chars = sum(len(page.text) for page in artifacts)
    replacement_chars = sum(page.text.count("\ufffd") for page in artifacts)
    visual_required_pages = [page.page_no for page in artifacts if page.visual_required]
    flags = Counter(flag for page in artifacts for flag in page.low_confidence_flags)

    empty_page_ratio = empty_pages / total_pages
    garbled_text_ratio = replacement_chars / max(total_chars, 1)
    avg_chars_per_page = total_chars / total_pages

    if empty_page_ratio > 0.7 or avg_chars_per_page < 80:
        status = "OCR_REQUIRED"
    elif len(visual_required_pages) / total_pages > 0.5 and avg_chars_per_page < 300:
        status = "VLM_PAGE_MODE"
    elif empty_page_ratio > 0.25 or garbled_text_ratio > 0.02:
        status = "PASS_WITH_WEAKNESSES"
    else:
        status = "PASS"

    return status, {
        "text_coverage_ratio": 1.0 - empty_page_ratio,
        "garbled_text_ratio": garbled_text_ratio,
        "empty_page_ratio": empty_page_ratio,
        "avg_chars_per_page": avg_chars_per_page,
        "visual_required_pages": visual_required_pages,
        "low_confidence_flags": dict(flags),
    }
