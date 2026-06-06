from __future__ import annotations

from pathlib import Path

from paperlens_core.core_manifest import inspect_core_v2_artifact_set
from paperlens_core.dom import PaperDOM
from paperlens_core.graph import ClaimGraph
from paperlens_core.report.graph_view import GraphReportDraft, render_graph_report_markdown
from paperlens_core.runtime import read_typed_artifact


def write_core_graph_report_view(
    *,
    output_dir: Path,
    data_dir: Path,
    paper_id: str,
    title: str,
    report_name: str,
) -> Path | None:
    manifest = inspect_core_v2_artifact_set(data_dir, paper_id)
    if manifest.get("consumable") is not True:
        return None
    root = data_dir / "core" / "v2" / paper_id
    try:
        dom_envelope = read_typed_artifact(root / "paper_dom.v1.json", expected_type="paper_dom")
        graph_envelope = read_typed_artifact(
            root / "claim_graph.v1.json", expected_type="claim_graph"
        )
        draft_envelope = read_typed_artifact(
            root / "report_draft.v1.json",
            expected_type="graph_report_draft",
        )
        quality_envelope = read_typed_artifact(
            root / "quality_metrics.v1.json",
            expected_type="core_quality_metrics",
        )
    except (FileNotFoundError, ValueError):
        return None
    if not all(
        isinstance(envelope.data, dict)
        for envelope in [dom_envelope, graph_envelope, draft_envelope, quality_envelope]
    ):
        return None
    dom = PaperDOM.model_validate(dom_envelope.data)
    graph = ClaimGraph.model_validate(graph_envelope.data)
    draft = GraphReportDraft.model_validate(draft_envelope.data)
    quality = graph_report_quality_context(
        quality_envelope.data if isinstance(quality_envelope.data, dict) else {},
        manifest=manifest,
    )
    markdown = render_graph_report_markdown(
        title=title or paper_id,
        draft=draft,
        graph=graph,
        dom=dom,
        quality=quality,
    )
    path = output_dir / "papers" / "core_graph" / core_graph_report_filename(report_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def core_graph_report_filename(report_name: str) -> str:
    if report_name.endswith(".md"):
        return report_name[:-3] + ".core_graph.md"
    return report_name + ".core_graph.md"


def graph_report_quality_context(
    quality: dict[str, object],
    *,
    manifest: dict[str, object],
) -> dict[str, object]:
    result = dict(quality)
    for key in [
        "publish_status",
        "artifact_publish_status",
        "current_audit_publish_status",
        "current_audit_error_count",
        "current_audit_warning_count",
        "current_audit_issue_codes",
    ]:
        value = manifest.get(key)
        if value is not None:
            result[key] = value
    return result
