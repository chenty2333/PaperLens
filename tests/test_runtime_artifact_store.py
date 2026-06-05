from __future__ import annotations

import pytest

from paperlens_core.runtime import (
    read_artifact_envelope,
    read_typed_artifact,
    write_typed_artifact,
)


def test_typed_artifact_store_round_trips_flat_envelope(tmp_path) -> None:
    path = tmp_path / "core" / "paper_dom.v1.json"

    envelope = write_typed_artifact(
        path,
        artifact_type="paper_dom",
        data={"paper_id": "p_test", "spans": []},
        producer="unit",
        source_ids=["span:p_test:1", "span:p_test:1", ""],
        metadata={"paper_id": "p_test"},
    )
    loaded = read_typed_artifact(path, expected_type="paper_dom")

    assert envelope.artifact_type == "paper_dom"
    assert loaded.data == {"paper_id": "p_test", "spans": []}
    assert loaded.source_ids == ["span:p_test:1"]
    assert loaded.metadata["paper_id"] == "p_test"


def test_typed_artifact_store_rejects_wrong_type(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    write_typed_artifact(
        path,
        artifact_type="observation_log",
        data={"cards": []},
        producer="unit",
    )

    with pytest.raises(ValueError, match="Expected artifact_type=claim_graph"):
        read_typed_artifact(path, expected_type="claim_graph")


def test_artifact_store_rejects_invalid_json(tmp_path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid artifact envelope"):
        read_artifact_envelope(path)
