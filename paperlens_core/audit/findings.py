from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AuditSeverity(StrEnum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


class PublishStatus(StrEnum):
    BLOCKED = "BLOCKED"
    DRAFT_WEAK = "DRAFT_WEAK"
    REVIEWED_WITH_LIMITS = "REVIEWED_WITH_LIMITS"
    REVIEWED = "REVIEWED"


class AuditFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    severity: AuditSeverity
    code: str
    message: str
    node_id: str | None = None
    source_ids: list[str] = Field(default_factory=list)
