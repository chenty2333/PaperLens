from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class ObservationType(StrEnum):
    PROBLEM = "problem"
    CLAIM = "claim"
    MECHANISM = "mechanism"
    IMPLEMENTATION = "implementation"
    EVALUATION = "evaluation"
    RESULT = "result"
    LIMITATION = "limitation"
    CONCEPT = "concept"


class ObservationCard(BaseModel):
    observation_id: str
    paper_id: str
    task_id: str
    observation_type: ObservationType
    statement: str
    source_ids: list[str]
    confidence: Literal["high", "medium", "low"] = "medium"
    provenance: Literal["explicit", "inferred", "background"] = "explicit"
    uncertainty: str | None = None
    extracted_numbers: list[dict[str, Any]] = Field(default_factory=list)
    proposed_links: list[dict[str, str]] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @field_validator("paper_id", "task_id", "statement")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("observation text fields cannot be blank")
        return value

    @field_validator("source_ids")
    @classmethod
    def require_sources_for_paper_facts(cls, value: list[str]) -> list[str]:
        result = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in result:
                result.append(text)
        if not result:
            raise ValueError("observation cards must cite at least one PaperDOM source_id")
        return result


class ObservationLog(BaseModel):
    schema_version: str = "observation_log.v1"
    paper_id: str
    cards: tuple[ObservationCard, ...] = ()

    def append(self, card: ObservationCard) -> "ObservationLog":
        if card.paper_id != self.paper_id:
            raise ValueError(f"observation paper_id mismatch: {card.paper_id} != {self.paper_id}")
        if any(existing.observation_id == card.observation_id for existing in self.cards):
            raise ValueError(f"duplicate observation_id: {card.observation_id}")
        return self.model_copy(update={"cards": (*self.cards, card)})


def make_observation_id(
    *,
    task_id: str,
    observation_type: str,
    statement: str,
    source_ids: list[str],
) -> str:
    payload = {
        "task_id": task_id,
        "observation_type": observation_type,
        "statement": statement.strip(),
        "source_ids": sorted({item for item in source_ids if item}),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"obs_{digest}"
