from __future__ import annotations

import re
from typing import Any, Iterable


def page_no(page: Any) -> int | None:
    if isinstance(page, dict):
        value = page.get("page_no")
    else:
        value = getattr(page, "page_no", None)
    return value if isinstance(value, int) else None


def page_list_field(page: Any, name: str) -> list[Any]:
    value = page.get(name) if isinstance(page, dict) else getattr(page, name, [])
    return value if isinstance(value, list) else []


def page_captions(page: Any) -> list[Any]:
    return page_list_field(page, "captions")


def page_source_ids(page: Any) -> list[str]:
    return dedupe_source_ids(
        [
            *explicit_source_ids(page),
            *page_text_source_ids(page),
            *page_visual_source_ids(page),
        ]
    )


def page_text_source_ids(page: Any) -> list[str]:
    source_ids = []
    for block in page_list_field(page, "blocks"):
        if not isinstance(block, dict):
            continue
        source_id = str(block.get("source_id") or "").strip()
        if source_id:
            source_ids.append(source_id)
    if source_ids:
        return dedupe_source_ids(source_ids)
    return []


def page_visual_source_ids(page: Any) -> list[str]:
    source_ids = []
    for field_name in ("figures", "tables"):
        for item in page_list_field(page, field_name):
            if isinstance(item, dict):
                source_id = str(item.get("source_id") or "").strip()
                if source_id:
                    source_ids.append(source_id)
    return dedupe_source_ids(source_ids)


def explicit_source_ids(page: Any) -> list[str]:
    value = page.get("source_ids") if isinstance(page, dict) else getattr(page, "source_ids", [])
    return [str(item).strip() for item in list_payload(value) if str(item).strip()]


def dedupe_source_ids(values: Iterable[Any]) -> list[str]:
    result = []
    for value in values:
        source_id = str(value or "").strip()
        if source_id and source_id not in result:
            result.append(source_id)
    return result


def list_payload(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def compact_text(text: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."
