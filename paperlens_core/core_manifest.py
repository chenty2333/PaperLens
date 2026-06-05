from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from paperlens_core.runtime import read_artifact_envelope, write_typed_artifact


CORE_V2_ENVELOPE_SCHEMA_VERSION = "paperlens_core.v2.bootstrap"
CORE_V2_MANIFEST_SCHEMA_VERSION = "core_v2_manifest.v1"
CORE_V2_MANIFEST_FILENAME = "core_manifest.v1.json"
CORE_V2_CONSUMABLE_STATUSES = {"REVIEWED", "REVIEWED_WITH_LIMITS"}
CORE_V2_REQUIRED_ARTIFACTS = {
    "paper_dom": ("paper_dom.v1.json", "paper_dom"),
    "reading_plan": ("reading_plan.v1.json", "reading_plan"),
    "observation_log": ("observation_log.v1.json", "observation_log"),
    "claim_graph": ("claim_graph.v1.json", "claim_graph"),
    "audit_findings": ("audit_findings.v1.json", "audit_findings"),
    "quality_metrics": ("quality_metrics.v1.json", "core_quality_metrics"),
    "paper_memory_view": ("paper_memory_view.v1.json", "paper_memory_view"),
    "report_draft": ("report_draft.v1.json", "graph_report_draft"),
    "report_audit_findings": ("report_audit_findings.v1.json", "report_audit_findings"),
}


def write_core_v2_manifest(
    root: Path,
    paper_id: str,
    *,
    producer: str = "paperlens_core_v2_manifest",
) -> Path:
    path = root / CORE_V2_MANIFEST_FILENAME
    manifest = build_core_v2_manifest(root, paper_id)
    write_typed_artifact(
        path,
        artifact_type="core_v2_manifest",
        artifact_version="v1",
        data=manifest,
        producer=producer,
        metadata={"paper_id": paper_id, "schema_version": CORE_V2_ENVELOPE_SCHEMA_VERSION},
    )
    return path


def inspect_core_v2_artifact_set(data_dir: Path, paper_id: str) -> dict[str, Any]:
    return inspect_core_v2_artifact_root(data_dir / "core" / "v2" / paper_id, paper_id)


def inspect_core_v2_artifact_root(root: Path, paper_id: str) -> dict[str, Any]:
    manifest = build_core_v2_manifest(root, paper_id)
    manifest_path = root / CORE_V2_MANIFEST_FILENAME
    manifest["manifest_artifact"] = {
        "path": CORE_V2_MANIFEST_FILENAME,
        "exists": manifest_path.exists(),
    }
    if not manifest_path.exists():
        manifest["issues"].append(f"missing:{CORE_V2_MANIFEST_FILENAME}")
    else:
        try:
            envelope = read_artifact_envelope(manifest_path)
        except ValueError as exc:
            manifest["issues"].append(f"invalid_envelope:{CORE_V2_MANIFEST_FILENAME}:{exc}")
        else:
            manifest["manifest_artifact"].update(
                {
                    "artifact_type": envelope.artifact_type,
                    "artifact_version": envelope.artifact_version,
                    "producer": envelope.producer,
                    "sha256": sha256_file(manifest_path),
                }
            )
            if envelope.artifact_type != "core_v2_manifest":
                manifest["issues"].append(
                    f"type_mismatch:{CORE_V2_MANIFEST_FILENAME}:"
                    f"expected=core_v2_manifest:actual={envelope.artifact_type}"
                )
    if manifest["issues"]:
        manifest["status"] = "INCOMPLETE"
        manifest["consumable"] = False
    return manifest


def build_core_v2_manifest(root: Path, paper_id: str) -> dict[str, Any]:
    issues: list[str] = []
    required_artifacts: dict[str, dict[str, Any]] = {}
    publish_status: str | None = None
    for key, (filename, expected_type) in CORE_V2_REQUIRED_ARTIFACTS.items():
        path = root / filename
        entry: dict[str, Any] = {
            "path": filename,
            "expected_artifact_type": expected_type,
            "exists": path.exists(),
        }
        if not path.exists():
            issues.append(f"missing:{filename}")
            required_artifacts[key] = entry
            continue
        entry["sha256"] = sha256_file(path)
        try:
            envelope = read_artifact_envelope(path)
        except ValueError as exc:
            issues.append(f"invalid_envelope:{filename}:{exc}")
            required_artifacts[key] = entry
            continue
        entry.update(
            {
                "artifact_type": envelope.artifact_type,
                "artifact_version": envelope.artifact_version,
                "producer": envelope.producer,
            }
        )
        if envelope.artifact_type != expected_type:
            issues.append(
                f"type_mismatch:{filename}:expected={expected_type}:actual={envelope.artifact_type}"
            )
        metadata_paper_id = str(envelope.metadata.get("paper_id") or "")
        if metadata_paper_id and metadata_paper_id != paper_id:
            issues.append(f"metadata_paper_id_mismatch:{filename}:{metadata_paper_id}")
        if isinstance(envelope.data, dict):
            data_paper_id = str(envelope.data.get("paper_id") or "")
            if data_paper_id and data_paper_id != paper_id:
                issues.append(f"data_paper_id_mismatch:{filename}:{data_paper_id}")
            if key == "quality_metrics":
                publish_status = str(envelope.data.get("publish_status") or "") or None
        required_artifacts[key] = entry
    complete = not issues
    consumable = complete and publish_status in CORE_V2_CONSUMABLE_STATUSES
    return {
        "schema_version": CORE_V2_MANIFEST_SCHEMA_VERSION,
        "paper_id": paper_id,
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "publish_status": publish_status,
        "consumable": consumable,
        "required_artifacts": required_artifacts,
        "issues": issues,
    }


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
