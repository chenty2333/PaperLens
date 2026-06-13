from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from paperlens_core.schemas import PageArtifact, PaperRecord


def _with_page(page: PageArtifact, item: dict[str, Any], item_type: str) -> dict[str, Any]:
    payload = dict(item)
    payload["paper_id"] = page.paper_id
    payload["page_no"] = page.page_no
    payload["item_type"] = item_type
    return payload


def build_layout_index(
    paper: PaperRecord,
    artifacts: list[PageArtifact],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    sections: list[dict[str, Any]] = []
    figures: list[dict[str, Any]] = []
    tables: list[dict[str, Any]] = []
    captions: list[dict[str, Any]] = []
    evidence_candidates: list[dict[str, Any]] = []

    for page in artifacts:
        sections.extend(_with_page(page, item, "section") for item in page.section_candidates)
        figures.extend(_with_page(page, item, "figure") for item in page.figures)
        tables.extend(_with_page(page, item, "table") for item in page.tables)
        captions.extend(_with_page(page, item, "caption") for item in page.captions)
        for index, block in enumerate(page.blocks[:24], start=1):
            text = " ".join(str(block.get("text") or "").split())
            source_id = str(block.get("source_id") or "").strip()
            if len(text) < 40 or not source_id:
                continue
            evidence_candidates.append(
                {
                    "paper_id": page.paper_id,
                    "page_no": page.page_no,
                    "source_id": source_id,
                    "bbox": block.get("bbox"),
                    "text_preview": text[:320],
                }
            )

    return {
        "paper": paper.model_dump(),
        "metrics": metrics,
        "page_count": len(artifacts),
        "visual_required_pages": [page.page_no for page in artifacts if page.visual_required],
        "sections": sections,
        "figures": merge_visual_items(figures, "figure"),
        "tables": merge_visual_items(tables, "table"),
        "captions": captions,
        "evidence_candidates": evidence_candidates[:500],
        "query_index": build_query_index(
            sections=sections,
            figures=figures,
            tables=tables,
            captions=captions,
            evidence_candidates=evidence_candidates,
        ),
    }


def build_query_index(
    *,
    sections: list[dict[str, Any]],
    figures: list[dict[str, Any]],
    tables: list[dict[str, Any]],
    captions: list[dict[str, Any]],
    evidence_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    by_page: dict[str, dict[str, int]] = {}
    for collection_name, rows in [
        ("sections", sections),
        ("figures", figures),
        ("tables", tables),
        ("captions", captions),
        ("evidence_candidates", evidence_candidates),
    ]:
        for row in rows:
            key = str(row.get("page_no") or "")
            if not key:
                continue
            by_page.setdefault(
                key,
                {
                    "sections": 0,
                    "figures": 0,
                    "tables": 0,
                    "captions": 0,
                    "evidence_candidates": 0,
                },
            )
            by_page[key][collection_name] += 1
    return {
        "by_page": by_page,
        "lookup_hints": {
            "section": "Use sections[] filtered by page_no and text/title.",
            "figure": "Use figures[] and captions[]; merged figures may span adjacent pages.",
            "table": "Use tables[] and captions[]; merged tables may span adjacent pages.",
            "evidence": "Use evidence_candidates[] for text spans with bbox.",
        },
    }


def merge_visual_items(items: list[dict[str, Any]], item_type: str) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for item in sorted(
        items,
        key=lambda row: (str(row.get("label") or row.get("id") or ""), int(row.get("page_no") or 0)),
    ):
        label = str(item.get("label") or item.get("id") or item.get("caption") or "").strip().lower()
        if label and merged:
            previous = merged[-1]
            previous_label = str(previous.get("label") or previous.get("id") or previous.get("caption") or "").strip().lower()
            previous_pages = previous.get("pages")
            item_page = int(item.get("page_no") or 0)
            previous_page = (
                int(previous_pages[-1] or 0)
                if isinstance(previous_pages, list) and previous_pages
                else 0
            )
            if (
                previous_label == label
                and isinstance(previous_pages, list)
                and item_page <= previous_page + 1
            ):
                previous_pages.append(item.get("page_no"))
                previous.setdefault("merged_items", []).append(item)
                continue
        payload = dict(item)
        payload["item_type"] = item_type
        payload["pages"] = [item.get("page_no")]
        payload["merged_items"] = [item]
        merged.append(payload)
    return merged


def write_layout_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "paper_id",
                "sections",
                "figures",
                "tables",
                "captions",
                "visual_required_pages",
                "evidence_candidates",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
