from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any, Protocol

from paperlens_core.agents.llm import JsonLlmClient, llm_call_context
from paperlens_core.config import CoreConfig
from paperlens_core.events import EventWriter
from paperlens_core.schemas import PaperRecord
from paperlens_core.workflow.utils import hash_text


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


class VisualWorkflowContext(Protocol):
    config: CoreConfig
    events: EventWriter

    def cache_path(self, stage: str, paper_id: str, key_payload: dict[str, Any]) -> Path: ...

    def read_cache_payload(self, path: Path) -> dict[str, Any] | None: ...

    def write_cache_payload(self, path: Path, payload: dict[str, Any]) -> None: ...

    def write_agent_run(self, payload: dict[str, Any]) -> None: ...

    def record_llm_usage(self, stage: str, usage: dict[str, Any]) -> None: ...


def run_vlm_page_mode(
    workflow: VisualWorkflowContext,
    *,
    client: JsonLlmClient,
    paper: PaperRecord,
    artifacts: list[Any],
    stage: str,
) -> dict[str, Any]:
    visual_artifacts = [artifact for artifact in artifacts if artifact.render_path]
    pages = [artifact.page_no for artifact in visual_artifacts]
    image_paths = [Path(artifact.render_path) for artifact in visual_artifacts]
    if not visual_artifacts:
        return {
            "paper_id": paper.paper_id,
            "agent_run_id": f"vlm_{paper.paper_id}_skipped",
            "page_notes": [],
            "visual_summary": "",
            "risk_notes": ["No rendered page images were available for VLM verification."],
        }
    key_payload = {
        "version": "vlm-page-v1",
        "model": workflow.config.provider.model,
        "visual_detail": workflow.config.visual_detail,
        "paper_hash": paper.file_hash,
        "pages": pages,
        "text_hashes": [hash_text(getattr(artifact, "text", "")) for artifact in artifacts],
        "image_hashes": [hash_file_bytes(path) for path in image_paths],
    }
    cache_path = workflow.cache_path("vlm_page_notes", paper.paper_id, key_payload)
    cached = workflow.read_cache_payload(cache_path)
    if cached and isinstance(cached.get("data"), dict):
        workflow.events.emit(
            "cache_hit",
            stage=stage,
            message=f"VLM page cache hit for {paper.paper_id}",
            data={"paper_id": paper.paper_id, "pages": pages, "cache": str(cache_path)},
        )
        data = cached["data"]
        page_notes = data.get("page_notes") if isinstance(data.get("page_notes"), list) else []
        return {
            "paper_id": paper.paper_id,
            "agent_run_id": str(cached.get("agent_run_id") or f"vlm_{paper.paper_id}_cache"),
            "page_notes": page_notes,
            "visual_summary": data.get("visual_summary"),
            "risk_notes": data.get("risk_notes"),
        }
    agent_run_id = f"vlm_{paper.paper_id}_{uuid.uuid4().hex[:8]}"
    workflow.events.emit(
        "agent_run_started",
        stage=stage,
        message=f"VLM page read {paper.paper_id}",
        data={"paper_id": paper.paper_id, "agent_run_id": agent_run_id},
    )
    with llm_call_context(
        stage=stage,
        paper_id=paper.paper_id,
        operation="vlm_page_read",
        schema_name="paperlens_vlm_page_notes",
        pages=pages,
    ):
        raw = client.invoke_json_with_images(
            system_prompt=VLM_PAGE_READER_SYSTEM_PROMPT,
            user_prompt=build_vlm_page_prompt(paper=paper, artifacts=visual_artifacts),
            image_paths=image_paths,
            schema_name="paperlens_vlm_page_notes",
            schema=VLM_PAGE_NOTES_SCHEMA,
            max_tokens=6000,
            detail=workflow.config.visual_detail,
        )
    workflow.write_agent_run(
        {
            "agent_run_id": agent_run_id,
            "paper_id": paper.paper_id,
            "stage": stage,
            "provider_kind": workflow.config.provider.kind,
            "model": workflow.config.provider.model,
            "endpoint": raw.endpoint,
            "request_id": raw.request_id,
            "usage": raw.usage,
            "status": "PASS",
        }
    )
    workflow.record_llm_usage(stage, raw.usage)
    workflow.write_cache_payload(
        cache_path,
        {
            "key": key_payload,
            "data": raw.data,
            "usage": raw.usage,
            "request_id": raw.request_id,
            "endpoint": raw.endpoint,
            "agent_run_id": agent_run_id,
        },
    )
    page_notes = raw.data.get("page_notes") if isinstance(raw.data.get("page_notes"), list) else []
    return {
        "paper_id": paper.paper_id,
        "agent_run_id": agent_run_id,
        "page_notes": page_notes,
        "visual_summary": raw.data.get("visual_summary"),
        "risk_notes": raw.data.get("risk_notes"),
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
        return f"missing:{hash_text(str(path))[:16]}"


def normalize_excerpt(text: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0] + " ..."
