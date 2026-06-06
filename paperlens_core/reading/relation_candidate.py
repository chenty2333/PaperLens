from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


RELATION_CANDIDATE_KINDS = {
    "contradicted_by",
    "depends_on",
    "explains",
    "implements",
    "evaluated_by",
    "limited_by",
    "compared_with",
}


class RelationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_observation_id: str
    target_observation_id: str
    kind: str
    confidence: Literal["high", "medium", "low"] = "medium"

    @field_validator("source_observation_id", "target_observation_id")
    @classmethod
    def nonempty_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("observation_id cannot be blank")
        return value

    @field_validator("kind")
    @classmethod
    def valid_kind(cls, value: str) -> str:
        value = value.strip()
        if value not in RELATION_CANDIDATE_KINDS:
            raise ValueError(
                f"relation candidate kind must be one of {sorted(RELATION_CANDIDATE_KINDS)}: {value}"
            )
        return value


class RelationCandidateLog(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "relation_candidate_log.v2"
    paper_id: str
    candidates: tuple[RelationCandidate, ...] = ()

    def append(self, candidate: RelationCandidate) -> "RelationCandidateLog":
        if candidate.source_observation_id == candidate.target_observation_id:
            raise ValueError("relation candidate cannot target itself")
        return self.model_copy(update={"candidates": (*self.candidates, candidate)})


def validate_relation_candidates(
    candidates: list[RelationCandidate],
    observation_ids: set[str],
) -> list[RelationCandidate]:
    valid: list[RelationCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for candidate in candidates:
        if candidate.source_observation_id not in observation_ids:
            continue
        if candidate.target_observation_id not in observation_ids:
            continue
        key = (
            candidate.source_observation_id,
            candidate.target_observation_id,
            candidate.kind,
        )
        if key in seen:
            continue
        seen.add(key)
        valid.append(candidate)
    return valid
