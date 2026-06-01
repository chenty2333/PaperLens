from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


JsonObject = dict[str, Any]


class CoreResult(BaseModel):
    status: Literal["ok", "error"] = "ok"
    data: JsonObject = Field(default_factory=dict)


class RunResult(CoreResult):
    data: JsonObject = Field(default_factory=dict)


class CoreRequest(BaseModel):
    config_path: Path | None = None
    config_overrides: JsonObject = Field(default_factory=dict)


class RunRequest(CoreRequest):
    input_dir: Path
    output_dir: Path
    run_id: str | None = None
    from_stage: str | None = None
    only_stage: str | None = None
    use_stdin_control: bool = False


class PaperQuestionRequest(CoreRequest):
    output_dir: Path
    question: str
    paper_id: str | None = None


class LibraryBuildRequest(BaseModel):
    output_dir: Path


class LibraryRebuildIndexRequest(BaseModel):
    output_dir: Path


class LibraryDoctorRequest(BaseModel):
    output_dir: Path


class LibrarySearchRequest(BaseModel):
    output_dir: Path
    query: str
    limit: int = 8


class LibraryQuestionRequest(CoreRequest):
    output_dir: Path
    question: str
    limit: int = 8

