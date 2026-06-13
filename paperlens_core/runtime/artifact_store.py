from __future__ import annotations

from pathlib import Path
from typing import Any

from paperlens_core.runtime.artifacts import ArtifactEnvelope
from paperlens_core.storage import atomic_write_json


MAX_ARTIFACT_ENVELOPE_BYTES = 100_000_000


def make_artifact_envelope(
    *,
    artifact_type: str,
    data: dict[str, Any] | list[Any],
    producer: str,
    artifact_version: str = "v2",
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
    artifact_version: str = "v2",
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
    atomic_write_json(path, envelope.model_dump(mode="json"))


def read_typed_artifact(path: Path, *, expected_type: str) -> ArtifactEnvelope:
    return read_artifact_envelope(path).require_type(expected_type)


def read_artifact_envelope(path: Path) -> ArtifactEnvelope:
    if not path.exists():
        raise FileNotFoundError(f"Missing artifact envelope: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ValueError(f"Cannot read artifact envelope: {path}") from exc
    if size > MAX_ARTIFACT_ENVELOPE_BYTES:
        raise ValueError(
            f"Artifact envelope is too large: {size} bytes exceeds "
            f"{MAX_ARTIFACT_ENVELOPE_BYTES}"
        )
    try:
        return ArtifactEnvelope.model_validate_json(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid artifact envelope: {path}") from exc
