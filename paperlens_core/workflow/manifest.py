from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from paperlens_core.core_manifest import inspect_core_v2_artifact_root


CORE_MANIFEST_CONSISTENCY_KEYS = [
    "schema_version",
    "paper_id",
    "status",
    "publish_status",
    "artifact_publish_status",
    "current_audit_publish_status",
    "current_audit_error_count",
    "current_audit_warning_count",
    "current_audit_issue_codes",
    "consumable",
    "required_artifacts",
    "issues",
]


def validate_paperlens_output(
    output_dir: Path,
    *,
    expected_report_names: set[str] | None = None,
    expected_paper_ids: set[str] | None = None,
) -> dict[str, Any]:
    main_report = output_dir / "PaperLens.md"
    library_records_path = output_dir / ".paperlens" / "library" / "library_records.jsonl"
    search_index_path = output_dir / ".paperlens" / "library" / "index" / "search_index.json"
    issues = []
    checked_links = 0
    render_markers = ["\\r\\n", "\\n"]
    reader_hostile_markers = [
        "supplied excerpts",
        "the user provided",
        "你给到",
        "供给的片段",
        "供给的图示",
        "提供的页面",
        "提供的材料",
        "提供的证据",
    ]
    if not main_report.exists():
        issues.append("PaperLens.md is missing")
        main_report_link_targets: set[str] = set()
    elif not main_report.read_text(encoding="utf-8").strip():
        issues.append("PaperLens.md is empty")
        main_report_link_targets = set()
    else:
        markdown = main_report.read_text(encoding="utf-8")
        main_report_link_targets = set(local_markdown_link_targets(markdown))
        for marker in render_markers:
            if marker in markdown:
                issues.append(f"Escaped newline marker in PaperLens.md: {marker}")
        for target in main_report_link_targets:
            checked_links += 1
            target_path = resolve_markdown_target(output_dir, target)
            if target_path is None:
                continue
            if not target_path.exists():
                issues.append(f"Missing link target: {target}")
            elif target_path.is_file() and not target_path.read_text(encoding="utf-8").strip():
                issues.append(f"Empty link target: {target}")
    if not library_records_path.exists():
        issues.append(".paperlens/library/library_records.jsonl is missing")
    elif not library_records_path.read_text(encoding="utf-8").strip():
        issues.append(".paperlens/library/library_records.jsonl is empty")
    library_records_by_paper_id = read_library_records_by_paper_id(library_records_path)
    if not search_index_path.exists():
        issues.append(".paperlens/library/index/search_index.json is missing")
    papers_dir = output_dir / "papers"
    all_paper_report_files = sorted(papers_dir.glob("*.md")) if papers_dir.exists() else []
    if expected_report_names is not None:
        existing_names = {report.name for report in all_paper_report_files}
        for missing_name in sorted(expected_report_names - existing_names):
            issues.append(f"Missing expected paper report: papers/{missing_name}")
    paper_reports = (
        [report for report in all_paper_report_files if report.name in expected_report_names]
        if expected_report_names is not None
        else list(all_paper_report_files)
    )
    if not paper_reports:
        issues.append("No per-paper Markdown reports were written")
    empty_reports = [
        report.name for report in paper_reports if not report.read_text(encoding="utf-8").strip()
    ]
    for report in empty_reports:
        issues.append(f"Empty paper report: papers/{report}")
    for report in paper_reports:
        text = report.read_text(encoding="utf-8")
        if ".paperlens/pages/" in text or ".paperlens\\pages\\" in text:
            issues.append(f"Full-page render embedded in papers/{report.name}")
        for marker in reader_hostile_markers:
            if marker in text:
                issues.append(
                    f"Reader-hostile implementation wording in papers/{report.name}: {marker}"
                )
                break
        for marker in render_markers:
            if marker in text:
                issues.append(f"Escaped newline marker in papers/{report.name}: {marker}")
                break
    core_root = output_dir / ".paperlens" / "data" / "core" / "v2"
    core_paper_ids = (
        sorted(expected_paper_ids)
        if expected_paper_ids is not None
        else sorted(path.name for path in core_root.iterdir() if path.is_dir())
        if core_root.exists()
        else []
    )
    for paper_id in core_paper_ids:
        paper_root = core_root / paper_id
        if not paper_root.exists():
            issues.append(f"Core v2 artifact root is missing for {paper_id}")
            continue
        core_manifest_path = paper_root / "core_manifest.v1.json"
        if not core_manifest_path.exists():
            issues.append(f"Core v2 manifest is missing for {paper_id}")
            continue
        try:
            core_manifest = json.loads(core_manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            issues.append(f"Core v2 manifest is invalid JSON for {paper_id}")
            continue
        if core_manifest.get("artifact_type") != "core_v2_manifest":
            issues.append(f"Core v2 manifest has wrong artifact_type for {paper_id}")
            continue
        data = core_manifest.get("data") if isinstance(core_manifest.get("data"), dict) else {}
        if data.get("status") != "COMPLETE":
            issues.append(f"Core v2 manifest is incomplete for {paper_id}")
            continue
        inspected_manifest = inspect_core_v2_artifact_root(paper_root, paper_id)
        if inspected_manifest.get("status") != "COMPLETE":
            issues.append(f"Core v2 artifact set is incomplete for {paper_id}")
            continue
        manifest_mismatches = core_manifest_data_mismatches(
            stored=data,
            inspected=inspected_manifest,
        )
        if manifest_mismatches:
            issues.append(
                f"Core v2 manifest is stale for {paper_id}: "
                + ", ".join(manifest_mismatches[:3])
            )
            continue
        if inspected_manifest.get("consumable") is not True:
            issues.append(
                f"Core v2 manifest is not consumable for {paper_id}: "
                f"publish_status={inspected_manifest.get('publish_status')}"
            )
            continue
        library_record = library_records_by_paper_id.get(paper_id, {})
        if not library_record:
            issues.append(f"Library record is missing for consumable core v2 paper {paper_id}")
            continue
        core_graph_report = (
            library_record.get("outputs", {}).get("core_graph_report_md")
            if isinstance(library_record.get("outputs"), dict)
            else None
        )
        if not isinstance(core_graph_report, str) or not core_graph_report.strip():
            issues.append(f"Core graph report output is missing for {paper_id}")
            continue
        core_graph_path = resolve_markdown_target(output_dir, core_graph_report)
        if core_graph_path is None or not core_graph_path.exists():
            issues.append(f"Core graph report output target is missing for {paper_id}")
        elif core_graph_path.is_file() and not core_graph_path.read_text(encoding="utf-8").strip():
            issues.append(f"Core graph report output target is empty for {paper_id}")
        if not markdown_target_is_linked(main_report_link_targets, core_graph_report):
            issues.append(f"Core graph report is not linked from PaperLens.md for {paper_id}")
    result = {
        "status": "PASS" if not issues else "FAIL",
        "checked_links": checked_links,
        "paper_reports": len(paper_reports),
        "paper_report_files": len(all_paper_report_files),
        "library_records": library_records_path.exists(),
        "issues": issues,
    }
    if issues:
        raise RuntimeError("Output validation failed: " + "; ".join(issues[:5]))
    return result


def core_manifest_data_mismatches(
    *,
    stored: dict[str, Any],
    inspected: dict[str, Any],
) -> list[str]:
    mismatches = []
    for key in CORE_MANIFEST_CONSISTENCY_KEYS:
        stored_value = stored.get(key)
        inspected_value = inspected.get(key)
        if stored_value != inspected_value:
            mismatches.append(f"{key} stored={stored_value!r} current={inspected_value!r}")
    return mismatches


def read_library_records_by_paper_id(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(row, dict):
            continue
        paper_id = str(row.get("paper_id") or "").strip()
        if paper_id:
            records[paper_id] = row
    return records


def markdown_target_is_linked(targets: set[str], expected: str) -> bool:
    normalized_expected = normalize_markdown_target(expected)
    return any(normalize_markdown_target(target) == normalized_expected for target in targets)


def normalize_markdown_target(target: str) -> str:
    target = re.split(r"[?#]", target, maxsplit=1)[0].strip()
    while target.startswith("./"):
        target = target[2:]
    return target.replace("\\", "/")


def summarize_model_calls(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"calls": 0, "by_status": {}, "by_stage": {}, "by_schema": {}, "maxima": {}}
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    by_status: dict[str, int] = {}
    by_stage: dict[str, dict[str, Any]] = {}
    by_schema: dict[str, dict[str, Any]] = {}
    maxima = {
        "payload_bytes": 0,
        "prompt_chars": 0,
        "duration_seconds": 0.0,
        "image_count": 0,
    }
    for row in rows:
        status = str(row.get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        context = row.get("context") if isinstance(row.get("context"), dict) else {}
        stage = str(context.get("stage") or "unknown")
        schema = str(row.get("schema_name") or context.get("schema_name") or "unknown")
        usage = row.get("usage") if isinstance(row.get("usage"), dict) else {}
        add_model_call_summary_row(by_stage, stage, row, usage)
        add_model_call_summary_row(by_schema, schema, row, usage)
        for key in maxima:
            value = row.get(key)
            if isinstance(value, (int, float)):
                maxima[key] = max(maxima[key], value)
    return {
        "calls": len(rows),
        "by_status": by_status,
        "by_stage": by_stage,
        "by_schema": by_schema,
        "maxima": maxima,
    }


def add_model_call_summary_row(
    bucket: dict[str, dict[str, Any]],
    key: str,
    row: dict[str, Any],
    usage: dict[str, Any],
) -> None:
    item = bucket.setdefault(
        key,
        {
            "calls": 0,
            "payload_bytes": 0,
            "prompt_chars": 0,
            "duration_seconds": 0.0,
            "input_tokens": 0,
            "output_tokens": 0,
        },
    )
    item["calls"] += 1
    item["payload_bytes"] += safe_number(row.get("payload_bytes"))
    item["prompt_chars"] += safe_number(row.get("prompt_chars"))
    item["duration_seconds"] = round(
        float(item["duration_seconds"]) + float(safe_number(row.get("duration_seconds"))),
        3,
    )
    item["input_tokens"] += safe_number(
        usage.get("input_tokens") or usage.get("prompt_tokens") or 0
    )
    item["output_tokens"] += safe_number(
        usage.get("output_tokens") or usage.get("completion_tokens") or 0
    )


def safe_number(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def local_markdown_link_targets(markdown: str) -> list[str]:
    targets = []
    for match in re.finditer(r"\[[^\]]+\]\(([^)]+)\)", markdown):
        raw_target = match.group(1).strip()
        if not raw_target:
            continue
        if raw_target.startswith("<") and raw_target.endswith(">"):
            raw_target = raw_target[1:-1].strip()
        lowered = raw_target.lower()
        if lowered.startswith(("http://", "https://", "mailto:", "#")):
            continue
        targets.append(raw_target)
    return targets


def resolve_markdown_target(base_dir: Path, target: str) -> Path | None:
    target = re.split(r"[?#]", target, maxsplit=1)[0].strip()
    if not target:
        return None
    target = target.replace("/", os.sep)
    path = Path(target)
    if not path.is_absolute():
        path = base_dir / path
    return path
