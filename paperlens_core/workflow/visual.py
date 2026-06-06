from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from paperlens_core.schemas import PaperRecord


VLM_PAGE_READER_SYSTEM_PROMPT = """
You are the PaperLens VLMPageReader.
Read rendered PDF page images directly. Extract only visible page facts useful for later
evidence binding: architecture diagrams, tables, plots, captions, section titles, and text
that was likely missed by text extraction. Do not invent content. Return JSON only.
""".strip()


VLM_PAGE_NOTES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["visual_summary", "page_notes", "risk_notes"],
    "properties": {
        "visual_summary": {"type": "string"},
        "risk_notes": {"type": "array", "items": {"type": "string"}},
        "page_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["page_no", "visible_text", "figures", "tables", "evidence_queries"],
                "properties": {
                    "page_no": {"type": "integer", "minimum": 1},
                    "visible_text": {"type": "array", "items": {"type": "string"}},
                    "figures": {"type": "array", "items": {"type": "string"}},
                    "tables": {"type": "array", "items": {"type": "string"}},
                    "evidence_queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["keyword", "quote"],
                            "properties": {
                                "keyword": {"type": ["string", "null"]},
                                "quote": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
            },
        },
    },
}


def build_vlm_page_prompt(*, paper: PaperRecord, artifacts: list[Any]) -> str:
    pages = []
    for artifact in artifacts:
        pages.append(
            {
                "page_no": artifact.page_no,
                "render_path": artifact.render_path,
                "parse_flags": artifact.low_confidence_flags,
                "text_preview": normalize_excerpt(artifact.text or "", limit=900),
                "captions": artifact.captions[:4],
            }
        )
    return "\n\n".join(
        [
            f"paper_id: {paper.paper_id}",
            f"title: {paper.canonical_title or 'unknown'}",
            f"parse_quality: {paper.parse_quality or 'unknown'}",
            "Images are supplied in the same order as this page metadata.",
            "Extract visible page facts, figure/table readings, and evidence queries.",
            json.dumps(pages, ensure_ascii=False),
        ]
    )


def hash_file_bytes(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "missing"


def normalize_excerpt(text: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0] + " ..."
