from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from paperlens_core.audit import (
    AuditFinding,
    AuditSeverity,
    audit_claim_graph,
    audit_claim_graph_from_observation_log,
    audit_observation_log,
    audit_reading_required_outputs,
    audit_relation_candidates,
    publish_status_from_findings,
)
from paperlens_core.dom import PaperDOM
from paperlens_core.graph import ClaimGraph
from paperlens_core.reading import ObservationLog, ReadingPlan, RelationCandidateLog
from paperlens_core.runtime import read_artifact_envelope, write_typed_artifact


CORE_V2_ENVELOPE_SCHEMA_VERSION = "paperlens_core.v2.bootstrap"
CORE_V2_MANIFEST_SCHEMA_VERSION = "core_v2_manifest.v2"
CORE_V2_MANIFEST_FILENAME = "core_manifest.v2.json"
CORE_V2_CONSUMABLE_STATUSES = {"REVIEWED", "REVIEWED_WITH_LIMITS", "DRAFT_WEAK"}
CORE_V2_REQUIRED_ARTIFACTS = {
    "paper_dom": ("paper_dom.v2.json", "paper_dom"),
    "reading_plan": ("reading_plan.v2.json", "reading_plan"),
    "observation_log": ("observation_log.v2.json", "observation_log"),
    "claim_graph": ("claim_graph.v2.json", "claim_graph"),
    "audit_findings": ("audit_findings.v2.json", "audit_findings"),
    "quality_metrics": ("quality_metrics.v2.json", "core_quality_metrics"),
    "paper_memory_view": ("paper_memory_view.v2.json", "paper_memory_view"),
    "report_draft": ("report_draft.v2.json", "graph_report_draft"),
    "report_audit_findings": ("report_audit_findings.v2.json", "report_audit_findings"),
}

CORE_V2_OPTIONAL_ARTIFACTS = {
    "relation_candidate_log": ("relation_candidate_log.v2.json", "relation_candidate_log"),
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
        artifact_version="v2",
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
        manifest["publish_status"] = None
        manifest["consumable"] = False
    return manifest


def build_core_v2_manifest(root: Path, paper_id: str) -> dict[str, Any]:
    issues: list[str] = []
    required_artifacts: dict[str, dict[str, Any]] = {}
    optional_artifacts: dict[str, dict[str, Any]] = {}
    artifact_publish_status: str | None = None

    def _check_artifact(key: str, filename: str, expected_type: str) -> dict[str, Any]:
        path = root / filename
        entry: dict[str, Any] = {
            "path": filename,
            "expected_artifact_type": expected_type,
            "exists": path.exists(),
        }
        if not path.exists():
            return entry
        entry["sha256"] = sha256_file(path)
        try:
            envelope = read_artifact_envelope(path)
        except ValueError as exc:
            issues.append(f"invalid_envelope:{filename}:{exc}")
            return entry
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
                nonlocal artifact_publish_status
                artifact_publish_status = str(envelope.data.get("publish_status") or "") or None
        return entry

    for key, (filename, expected_type) in CORE_V2_REQUIRED_ARTIFACTS.items():
        entry = _check_artifact(key, filename, expected_type)
        if not entry["exists"]:
            issues.append(f"missing:{filename}")
        required_artifacts[key] = entry

    for key, (filename, expected_type) in CORE_V2_OPTIONAL_ARTIFACTS.items():
        entry = _check_artifact(key, filename, expected_type)
        optional_artifacts[key] = entry
    current_audit_summary: dict[str, Any] = {}
    if not issues:
        try:
            current_audit_summary = build_current_audit_summary(root)
        except (ValueError, ValidationError) as exc:
            issues.append(f"current_audit_invalid:{exc}")
    complete = not issues
    current_publish_status = str(current_audit_summary.get("publish_status") or "") or None
    publish_status = current_publish_status if complete else None
    consumable = complete and publish_status in CORE_V2_CONSUMABLE_STATUSES
    return {
        "schema_version": CORE_V2_MANIFEST_SCHEMA_VERSION,
        "paper_id": paper_id,
        "status": "COMPLETE" if complete else "INCOMPLETE",
        "publish_status": publish_status,
        "artifact_publish_status": artifact_publish_status,
        "current_audit_publish_status": current_publish_status,
        "current_audit_error_count": current_audit_summary.get("error_count"),
        "current_audit_warning_count": current_audit_summary.get("warning_count"),
        "current_audit_issue_codes": current_audit_summary.get("issue_codes", []),
        "consumable": consumable,
        "required_artifacts": required_artifacts,
        "optional_artifacts": optional_artifacts,
        "issues": issues,
    }


def build_current_audit_summary(root: Path) -> dict[str, Any]:
    findings = current_core_v2_findings(root)
    return {
        "publish_status": publish_status_from_findings(findings).value,
        "error_count": audit_finding_count(findings, severity=AuditSeverity.ERROR),
        "warning_count": audit_finding_count(findings, severity=AuditSeverity.WARNING),
        "issue_codes": sorted({finding.code for finding in findings}),
    }


def current_core_v2_findings(root: Path) -> list[AuditFinding]:
    from paperlens_core.report.graph_view import (
        GraphReportDraft,
        audit_report_draft_against_graph,
    )

    dom = PaperDOM.model_validate(
        read_required_artifact_data(root, "paper_dom.v2.json", expected_type="paper_dom")
    )
    reading_plan = ReadingPlan.model_validate(
        read_required_artifact_data(root, "reading_plan.v2.json", expected_type="reading_plan")
    )
    observation_log = ObservationLog.model_validate(
        read_required_artifact_data(
            root,
            "observation_log.v2.json",
            expected_type="observation_log",
        )
    )
    graph = ClaimGraph.model_validate(
        read_required_artifact_data(root, "claim_graph.v2.json", expected_type="claim_graph")
    )
    report_draft = GraphReportDraft.model_validate(
        read_required_artifact_data(
            root,
            "report_draft.v2.json",
            expected_type="graph_report_draft",
        )
    )
    relation_log = _load_relation_log(root)
    relation_candidates = list(relation_log.candidates) if relation_log else None
    observation_ids = {card.observation_id for card in observation_log.cards}
    return [
        *audit_observation_log(observation_log, dom, reading_plan),
        *audit_claim_graph_from_observation_log(
            graph, observation_log, relation_candidates=relation_candidates
        ),
        *audit_claim_graph(graph, dom),
        *audit_reading_required_outputs(graph, reading_plan),
        *audit_report_draft_against_graph(report_draft, graph),
        *(audit_relation_candidates(relation_log, observation_ids) if relation_log else []),
    ]


def _load_relation_log(root: Path) -> RelationCandidateLog | None:
    path = root / "relation_candidate_log.v2.json"
    if not path.exists():
        return None
    try:
        envelope = read_artifact_envelope(path)
    except (FileNotFoundError, ValueError):
        return None
    if envelope.artifact_type != "relation_candidate_log":
        return None
    if not isinstance(envelope.data, dict):
        return None
    try:
        return RelationCandidateLog.model_validate(envelope.data)
    except Exception:
        return None


def audit_finding_count(findings: list[AuditFinding], *, severity: AuditSeverity) -> int:
    return sum(1 for finding in findings if finding.severity == severity)


def read_required_artifact_data(
    root: Path,
    filename: str,
    *,
    expected_type: str,
) -> dict[str, Any] | list[Any]:
    path = root / filename
    try:
        envelope = read_artifact_envelope(path)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{filename}:{exc}") from exc
    try:
        envelope.require_type(expected_type)
    except ValueError as exc:
        raise ValueError(f"{filename}:{exc}") from exc
    return envelope.data


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
