from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PaperRecord(BaseModel):
    paper_id: str
    file_path: str
    file_hash: str
    canonical_title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    page_count: int = 0
    duplicate_group: str | None = None
    status: str = "NEW"
    parse_quality: str | None = None
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    bibtex_key: str | None = None
    side_statuses: list[str] = Field(default_factory=list)


class PageArtifact(BaseModel):
    paper_id: str
    page_no: int
    text: str
    page_width: float | None = None
    page_height: float | None = None
    render_width: int | None = None
    render_height: int | None = None
    blocks: list[dict[str, Any]] = Field(default_factory=list)
    words: list[dict[str, Any]] = Field(default_factory=list)
    images: list[dict[str, Any]] = Field(default_factory=list)
    tables: list[dict[str, Any]] = Field(default_factory=list)
    figures: list[dict[str, Any]] = Field(default_factory=list)
    captions: list[dict[str, Any]] = Field(default_factory=list)
    visual_notes: list[dict[str, Any]] = Field(default_factory=list)
    render_path: str | None = None
    low_confidence_flags: list[str] = Field(default_factory=list)
    visual_required: bool = False
    section_candidates: list[dict[str, Any]] = Field(default_factory=list)
    crop_paths: list[str] = Field(default_factory=list)
    artifact_version: str = "v1"


class ArtifactVersion(BaseModel):
    artifact_id: str
    paper_id: str | None = None
    artifact_type: str
    path: str
    content_hash: str
    version: int = 1
    depends_on: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=now_iso)


class PaperState(BaseModel):
    paper_id: str
    state: str = "NEW"
    side_statuses: list[str] = Field(default_factory=list)
    current_stage: str | None = None
    retry_count: int = 0
    last_error: str | None = None
    updated_at: str = Field(default_factory=now_iso)


class ReviewItem(BaseModel):
    item_id: str
    paper_id: str | None = None
    item_type: str
    status: Literal["OPEN", "RESOLVED", "SKIPPED"] = "OPEN"
    priority: int = 2
    reason: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)


class SkimCard(BaseModel):
    paper_id: str
    problem: str | None = None
    method_type: str | None = None
    system_scope: str | None = None
    evaluation_type: str | None = None
    danger_signals: list[str] = Field(default_factory=list)
    evidence_source_ids: list[str] = Field(default_factory=list)
    confidence: float = 0.0


ClassLabel = Literal["A", "B", "C", "HOLD"]


class ClassificationDecision(BaseModel):
    paper_id: str
    class_label: ClassLabel
    confidence: float
    false_negative_risk: float
    reason_codes: list[str] = Field(default_factory=list)
    audit_status: str = "PENDING"
    preliminary_label: ClassLabel | None = None
    validator_label: ClassLabel | None = None
    validation_status: str | None = None
    validation_notes: list[str] = Field(default_factory=list)

