from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Iterable, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ObservationType(StrEnum):
    PROBLEM = "problem"
    CLAIM = "claim"
    MECHANISM = "mechanism"
    IMPLEMENTATION = "implementation"
    EVALUATION = "evaluation"
    RESULT = "result"
    LIMITATION = "limitation"
    CONCEPT = "concept"


PROPOSED_RELATION_KINDS = {
    "contradicted_by",
    "depends_on",
    "explains",
    "implements",
    "evaluated_by",
    "limited_by",
    "compared_with",
}


class ProposedRelation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target_observation_id: str
    kind: str

    @field_validator("target_observation_id")
    @classmethod
    def nonempty_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("target_observation_id cannot be blank")
        return value

    @field_validator("kind")
    @classmethod
    def valid_kind(cls, value: str) -> str:
        value = value.strip()
        if value not in PROPOSED_RELATION_KINDS:
            raise ValueError(
                f"proposed relation kind must be one of {sorted(PROPOSED_RELATION_KINDS)}: {value}"
            )
        return value


class ObservationCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation_id: str
    paper_id: str
    task_id: str
    observation_type: ObservationType
    statement: str
    source_ids: list[str]
    confidence: Literal["high", "medium", "low"] = "medium"
    provenance: Literal["explicit", "inferred"] = "explicit"
    uncertainty: str | None = None
    covered_outputs: list[str] = Field(default_factory=list)
    extracted_numbers: list[dict[str, Any]] = Field(default_factory=list)
    proposed_relations: list[ProposedRelation] = Field(default_factory=list)
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

    @field_validator("covered_outputs")
    @classmethod
    def clean_covered_outputs(cls, value: list[str]) -> list[str]:
        result = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in result:
                result.append(text)
        return result


class ObservationLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "observation_log.v1"
    paper_id: str
    cards: tuple[ObservationCard, ...] = ()

    def append(self, card: ObservationCard) -> "ObservationLog":
        if card.paper_id != self.paper_id:
            raise ValueError(f"observation paper_id mismatch: {card.paper_id} != {self.paper_id}")
        if any(existing.observation_id == card.observation_id for existing in self.cards):
            raise ValueError(f"duplicate observation_id: {card.observation_id}")
        return self.model_copy(update={"cards": (*self.cards, card)})

    def append_many(
        self,
        cards: Iterable[ObservationCard],
        *,
        on_duplicate: Literal["error", "ignore"] = "error",
    ) -> "ObservationLog":
        if on_duplicate not in {"error", "ignore"}:
            raise ValueError(f"unsupported duplicate policy: {on_duplicate}")
        if on_duplicate == "error":
            log = self
            for card in cards:
                log = log.append(card)
            return log

        accepted = list(self.cards)
        by_id = {card.observation_id: card for card in accepted}
        for card in cards:
            if card.paper_id != self.paper_id:
                raise ValueError(f"observation paper_id mismatch: {card.paper_id} != {self.paper_id}")
            existing = by_id.get(card.observation_id)
            if existing is not None:
                if observation_identity_payload(existing) != observation_identity_payload(card):
                    raise ValueError(f"conflicting duplicate observation_id: {card.observation_id}")
                continue
            accepted.append(card)
            by_id[card.observation_id] = card
        return self.model_copy(update={"cards": tuple(accepted)})

    def merge(
        self,
        other: "ObservationLog",
        *,
        on_duplicate: Literal["error", "ignore"] = "error",
    ) -> "ObservationLog":
        if other.paper_id != self.paper_id:
            raise ValueError(f"observation paper_id mismatch: {other.paper_id} != {self.paper_id}")
        return self.append_many(other.cards, on_duplicate=on_duplicate)


def observation_identity_payload(card: ObservationCard) -> dict[str, Any]:
    payload = card.model_dump()
    payload.pop("created_at", None)
    return payload


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
