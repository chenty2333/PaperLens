from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol

from paperlens_core.agents.llm import JsonLlmClient
from paperlens_core.config import CoreConfig
from paperlens_core.db import ArtifactDb
from paperlens_core.events import EventWriter, write_json
from paperlens_core.pdf.ingest import scan_pdfs
from paperlens_core.pdf.layout_index import build_layout_index
from paperlens_core.pdf.pymupdf_parser import parse_pdf
from paperlens_core.pdf.quality import parse_quality
from paperlens_core.schemas import PaperRecord, ReviewItem
from paperlens_core.workflow.utils import chunked


class ParseWorkflowContext(Protocol):
    input_dir: Path
    evidence_dir: Path
    data_dir: Path
    config: CoreConfig
    events: EventWriter
    db: ArtifactDb
    papers: list[PaperRecord]

    def checkpoint(self, stage: str) -> None: ...

    def mark_paper_state(
        self,
        paper_id: str,
        stage: str,
        *,
        side_statuses: list[str] | None = None,
        error: str | None = None,
    ) -> None: ...

    def register_file_artifact(
        self,
        path: Path,
        *,
        paper_id: str | None,
        artifact_type: str,
        depends_on: list[str] | None = None,
    ) -> None: ...

    def new_llm_client(self) -> JsonLlmClient: ...

    def llm_enabled(self) -> bool: ...

    def run_vlm_page_mode(
        self,
        *,
        client: JsonLlmClient,
        paper: PaperRecord,
        artifacts: list[Any],
        stage: str,
    ) -> dict[str, Any]: ...


def run_ingest_stage(workflow: ParseWorkflowContext) -> None:
    stage = "stage_00_ingest"
    workflow.checkpoint(stage)
    workflow.events.stage_started(stage, "Scanning PDFs")
    workflow.papers = scan_pdfs(workflow.input_dir)
    active_ids = [paper.paper_id for paper in workflow.papers]
    workflow.db.set_state("active_run_id", workflow.events.run_id)
    workflow.db.set_state("active_input_dir", str(workflow.input_dir))
    workflow.db.set_state("active_paper_ids", active_ids)
    for paper in workflow.papers:
        paper.status = "INGESTED"
        workflow.db.upsert_paper(paper)
        workflow.mark_paper_state(paper.paper_id, stage)
    workflow.events.stage_completed(stage, f"Found {len(workflow.papers)} PDF files")


def run_parse_stage(workflow: ParseWorkflowContext) -> None:
    stage = "stage_01_parse"
    workflow.checkpoint(stage)
    workflow.events.stage_started(stage, "Parsing PDFs with PyMuPDF")
    if not workflow.papers:
        workflow.events.stage_completed(stage, "No PDFs to parse")
        return

    completed = 0
    parsed_by_id: dict[str, PaperRecord] = {}
    max_workers = min(max(1, workflow.config.concurrency), len(workflow.papers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for paper in workflow.papers:
            workflow.checkpoint(stage)
            workflow.events.emit(
                "paper_started",
                stage=stage,
                message=f"Parsing {paper.canonical_title or paper.paper_id}",
                data={"paper_id": paper.paper_id},
            )
            futures[executor.submit(parse_one_paper, workflow, paper)] = paper

        for future in as_completed(futures):
            paper = futures[future]
            completed += 1
            progress = completed / len(workflow.papers)
            workflow.checkpoint(stage)
            try:
                parsed_paper, artifacts, quality, metrics = future.result()
                persist_parse_result(
                    workflow,
                    stage=stage,
                    paper=paper,
                    parsed_paper=parsed_paper,
                    artifacts=artifacts,
                    quality=quality,
                    metrics=metrics,
                    progress=progress,
                )
                parsed_by_id[parsed_paper.paper_id] = parsed_paper
            except Exception as exc:
                paper.status = "FAILED"
                paper.parse_quality = "FAIL"
                workflow.db.upsert_paper(paper)
                workflow.mark_paper_state(paper.paper_id, stage, error=str(exc))
                workflow.events.error(
                    stage,
                    f"Failed to parse {paper.file_path}: {exc}",
                    {"paper_id": paper.paper_id},
                )
    workflow.papers = [parsed_by_id.get(paper.paper_id, paper) for paper in workflow.papers]
    workflow.events.stage_completed(stage, "Parse stage completed")


def parse_one_paper(
    workflow: ParseWorkflowContext,
    paper: PaperRecord,
) -> tuple[PaperRecord, list[Any], str, dict[str, Any]]:
    parsed_paper, artifacts = parse_pdf(
        paper,
        workflow.evidence_dir,
        render_zoom=workflow.config.render_zoom,
    )
    quality, metrics = parse_quality(artifacts)
    return parsed_paper, artifacts, quality, metrics


def persist_parse_result(
    workflow: ParseWorkflowContext,
    *,
    stage: str,
    paper: PaperRecord,
    parsed_paper: PaperRecord,
    artifacts: list[Any],
    quality: str,
    metrics: dict[str, Any],
    progress: float,
) -> None:
    parsed_paper.parse_quality = quality
    parsed_paper.status = "PARSE_VERIFIED"
    workflow.db.upsert_paper(parsed_paper)
    workflow.db.insert_page_artifacts(artifacts)
    layout_index = build_layout_index(parsed_paper, artifacts, metrics)
    layout_path = workflow.data_dir / "artifacts" / "layout" / f"{paper.paper_id}.json"
    write_json(
        layout_path,
        {
            "paper": parsed_paper.model_dump(),
            "parse_quality": quality,
            "metrics": metrics,
            "layout_index": layout_index,
            "pages": [artifact.model_dump() for artifact in artifacts],
        },
    )
    workflow.register_file_artifact(
        layout_path,
        paper_id=paper.paper_id,
        artifact_type="layout_index",
    )
    for artifact in artifacts:
        if artifact.render_path:
            workflow.register_file_artifact(
                Path(artifact.render_path),
                paper_id=paper.paper_id,
                artifact_type="page_render",
            )
    side_statuses = parse_side_statuses(
        workflow,
        paper=paper,
        artifacts=artifacts,
        quality=quality,
        metrics=metrics,
    )
    workflow.mark_paper_state(paper.paper_id, stage, side_statuses=side_statuses)
    workflow.events.emit(
        "paper_completed",
        stage=stage,
        progress=progress,
        message=f"Parsed {paper.paper_id}",
        data={"paper_id": paper.paper_id, "parse_quality": quality},
    )


def parse_side_statuses(
    workflow: ParseWorkflowContext,
    *,
    paper: PaperRecord,
    artifacts: list[Any],
    quality: str,
    metrics: dict[str, Any],
) -> list[str]:
    side_statuses = []
    if quality == "OCR_REQUIRED":
        side_statuses.append("NEED_VISUAL_RECHECK")
        workflow.db.upsert_review_item(
            ReviewItem(
                item_id=f"vlm_scan:{paper.paper_id}",
                paper_id=paper.paper_id,
                item_type="VLM_PAGE_MODE",
                priority=1,
                reason="text extraction weak; route rendered page images to multimodal model",
                payload={"metrics": metrics},
            )
        )
    if quality == "VLM_PAGE_MODE":
        side_statuses.append("NEED_VISUAL_RECHECK")
        workflow.db.upsert_review_item(
            ReviewItem(
                item_id=f"vlm:{paper.paper_id}",
                paper_id=paper.paper_id,
                item_type="VLM_PAGE_MODE",
                priority=1,
                reason="parse_quality=VLM_PAGE_MODE",
                payload={"metrics": metrics},
            )
        )
    visual_pages = [page.page_no for page in artifacts if page.visual_required]
    if visual_pages:
        side_statuses.append("NEED_VISUAL_RECHECK")
        workflow.db.upsert_review_item(
            ReviewItem(
                item_id=f"visual:{paper.paper_id}",
                paper_id=paper.paper_id,
                item_type="NEED_VISUAL_RECHECK",
                priority=2,
                reason="visual_required_pages",
                payload={"pages": visual_pages},
            )
        )
    return side_statuses


def run_parse_verify_stage(workflow: ParseWorkflowContext) -> None:
    stage = "stage_02_parse_verify"
    workflow.checkpoint(stage)
    workflow.events.stage_started(stage, "Verifying parse quality and VLM page enrichment")
    if not workflow.papers:
        workflow.events.stage_completed(stage, "No PDFs to verify")
        return
    client = workflow.new_llm_client() if workflow.llm_enabled() else None
    for paper in workflow.papers:
        artifacts = workflow.db.get_page_artifacts(paper.paper_id)
        visual_pages = visual_pages_for_parse_verification(workflow, paper, artifacts)
        if client and visual_pages:
            visual_results = []
            for batch in chunked(visual_pages, workflow.config.visual_pages_per_call):
                visual_results.append(
                    workflow.run_vlm_page_mode(
                        client=client,
                        paper=paper,
                        artifacts=batch,
                        stage=stage,
                    )
                )
            if visual_results:
                apply_visual_parse_notes(
                    workflow,
                    paper=paper,
                    artifacts=artifacts,
                    visual_results=visual_results,
                )
        side = []
        if paper.parse_quality in {"OCR_REQUIRED", "VLM_PAGE_MODE"}:
            side.append("NEED_VISUAL_RECHECK")
        workflow.mark_paper_state(paper.paper_id, stage, side_statuses=side)
    workflow.events.stage_completed(stage, "Parse verification completed")


def visual_pages_for_parse_verification(
    workflow: ParseWorkflowContext,
    paper: PaperRecord,
    artifacts: list[Any],
) -> list[Any]:
    mode = workflow.config.visual_verification_mode
    if mode == "off":
        return []
    parse_needs_visual = paper.parse_quality in {
        "OCR_REQUIRED",
        "VLM_PAGE_MODE",
        "PASS_WITH_WEAKNESSES",
    }
    if mode == "parse_issues" and not parse_needs_visual:
        return []
    return [
        artifact
        for artifact in artifacts
        if artifact.render_path and (artifact.visual_required or parse_needs_visual)
    ][: workflow.config.visual_verification_max_pages]


def apply_visual_parse_notes(
    workflow: ParseWorkflowContext,
    *,
    paper: PaperRecord,
    artifacts: list[Any],
    visual_results: list[dict[str, Any]],
) -> None:
    all_notes = [
        note
        for visual_result in visual_results
        for note in visual_result.get("page_notes", [])
    ]
    for page in artifacts:
        notes = [note for note in all_notes if note.get("page_no") == page.page_no]
        if notes:
            page.visual_notes = notes
            page.low_confidence_flags = [
                flag for flag in page.low_confidence_flags if flag != "visual_required"
            ]
            page.visual_required = False
    workflow.db.insert_page_artifacts(artifacts)
    paper.parse_quality = (
        "PASS_WITH_WEAKNESSES"
        if paper.parse_quality in {"OCR_REQUIRED", "VLM_PAGE_MODE"}
        else paper.parse_quality
    )
    paper.status = "PARSE_VERIFIED"
    workflow.db.upsert_paper(paper)
