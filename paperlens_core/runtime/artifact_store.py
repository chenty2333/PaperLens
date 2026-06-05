from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from paperlens_core.runtime.artifacts import ArtifactEnvelope


def make_artifact_envelope(
    *,
    artifact_type: str,
    data: dict[str, Any] | list[Any],
    producer: str,
    artifact_version: str = "v1",
    source_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactEnvelope:
    return ArtifactEnvelope(
        artifact_type=artifact_type,
        artifact_version=artifact_version,
        data=data,
        producer=producer,
        source_ids=source_ids or [],
        metadata=metadata or {},
    )


def write_typed_artifact(
    path: Path,
    *,
    artifact_type: str,
    data: dict[str, Any] | list[Any],
    producer: str,
    artifact_version: str = "v1",
    source_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ArtifactEnvelope:
    envelope = make_artifact_envelope(
        artifact_type=artifact_type,
        artifact_version=artifact_version,
        data=data,
        producer=producer,
        source_ids=source_ids,
        metadata=metadata,
    )
    write_artifact_envelope(path, envelope)
    return envelope


def write_artifact_envelope(path: Path, envelope: ArtifactEnvelope) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(envelope.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_typed_artifact(path: Path, *, expected_type: str) -> ArtifactEnvelope:
    return read_artifact_envelope(path).require_type(expected_type)


def read_artifact_envelope(path: Path) -> ArtifactEnvelope:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact envelope: {path}")
    try:
        return ArtifactEnvelope.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid artifact envelope: {path}") from exc
