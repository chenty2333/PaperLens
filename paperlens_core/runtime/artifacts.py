from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ArtifactEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    """Standard runtime envelope for every model or deterministic artifact.

    The envelope is intentionally flat: task nodes exchange typed artifacts,
    not JSON strings nested inside other JSON payloads.
    """

    artifact_type: str
    artifact_version: str = "v1"
    data: dict[str, Any] | list[Any]
    producer: str
    source_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=now_iso)

    @field_validator("artifact_type", "artifact_version", "producer")
    @classmethod
    def nonempty_identifier(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("artifact envelope identifiers cannot be blank")
        return value

    @field_validator("source_ids")
    @classmethod
    def clean_source_ids(cls, value: list[str]) -> list[str]:
        result = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in result:
                result.append(text)
        return result

    def require_type(self, artifact_type: str) -> "ArtifactEnvelope":
        if self.artifact_type != artifact_type:
            raise ValueError(f"Expected artifact_type={artifact_type}, got {self.artifact_type}")
        return self

    def as_cache_payload(self) -> dict[str, Any]:
        return self.model_dump()
