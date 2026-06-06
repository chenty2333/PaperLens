from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from paperlens_core.agents.llm import JsonLlmClient, llm_call_context
from paperlens_core.agents.providers import describe_provider
from paperlens_core.budget import BudgetManager
from paperlens_core.config import CoreConfig
from paperlens_core.control import ControlState
from paperlens_core.db import ArtifactDb
from paperlens_core.events import EventWriter, write_json
from paperlens_core.library import write_paperlens_library
from paperlens_core.pdf.ingest import scan_pdfs
from paperlens_core.pdf.layout_index import build_layout_index
from paperlens_core.pdf.pymupdf_parser import parse_pdf
from paperlens_core.pdf.qa import parse_quality
from paperlens_core.quality_snapshot import write_core_quality_snapshot
from paperlens_core.report import (
    classification_counts,
    compact_reason,
    markdown_title,
    paper_report_filename,
    render_core_graph_report_view,
    render_paperlens_report,
)
from paperlens_core.runtime import (
    llm_cache_path,
    read_llm_cache,
    write_llm_cache,
)
from paperlens_core.library_graph import read_core_v2_graph_summary
from paperlens_core.schemas import (
    ArtifactVersion,
    ClassificationDecision,
    EvidenceRef,
    PaperRecord,
    ReviewItem,
    SkimCard,
)
from paperlens_core.state import transition_state
from paperlens_core.workflow.manifest import (
    summarize_model_calls,
    validate_paperlens_output,
)
from paperlens_core.workflow.stages import (
    WORKFLOW_STAGE_ORDER,
    normalize_workflow_stage,
    resolve_workflow_stages,
)
from paperlens_core.workflow.core_v2 import (
    refresh_core_v2_audit_artifacts,
    run_core_v2_model_observation_tasks,
    write_core_v2_artifacts,
)


class PaperLensWorkflow:
    def __init__(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        config: CoreConfig,
        events: EventWriter,
        control: ControlState,
    ) -> None:
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.config = config
        self.events = events
        self.control = control
        self.internal_dir = output_dir / ".paperlens"
        self.data_dir = self.internal_dir / "data"
        self.cache_dir = Path(os.getenv("PAPERLENS_CACHE_DIR", str(self.internal_dir / "cache")))
        self.evidence_dir = self.internal_dir
        self.db = ArtifactDb(self.internal_dir / "state.sqlite")
        self.papers: list[PaperRecord] = []
        self.skim_cards: list[SkimCard] = []
        self.classifications: list[ClassificationDecision] = []
        self._llm_missing_key_warned = False
        self.budget = BudgetManager(config.budget)

    def new_llm_client(self) -> JsonLlmClient:
        return JsonLlmClient(
            self.config.provider,
            ledger_path=self.data_dir / "model_calls.jsonl",
            run_id=self.events.run_id,
        )

    def run(
        self,
        *,
        from_stage: str | None = None,
        only_stage: str | None = None,
    ) -> dict[str, Any]:
        self.prepare_output()
        selected_stages = resolve_workflow_stages(from_stage=from_stage, only_stage=only_stage)
        provider = describe_provider(self.config.provider)
        self.events.emit(
            "run_started",
            message="PaperLens Core started",
            data={
                "input_dir": str(self.input_dir),
                "output_dir": str(self.output_dir),
                "provider": provider.__dict__,
                "read_mode": self.config.read_mode,
                "stages": selected_stages,
            },
        )
        try:
            if selected_stages[0] != "stage_00_ingest":
                self.load_completed_state_for_stage(selected_stages[0])
            stage_methods = {
                "stage_00_ingest": self.stage_00_ingest,
                "stage_01_parse": self.stage_01_parse,
                "stage_02_parse_verify": self.stage_02_parse_verify,
                "stage_03_skim": self.stage_03_skim,
                "stage_07_normal_read": self.stage_07_normal_read,
                "stage_08_evidence_verify": self.stage_08_evidence_verify,
                "stage_15_export": self.stage_15_export,
                "stage_17_manifest": self.stage_17_manifest,
            }
            manifest: dict[str, Any] | None = None
            for stage in selected_stages:
                result = stage_methods[stage]()
                if stage == "stage_17_manifest":
                    manifest = result
            if manifest is None:
                manifest = {
                    "run_id": self.events.run_id,
                    "input_dir": str(self.input_dir),
                    "output_dir": str(self.output_dir),
                    "mode": "offline_debug" if self.config.offline_debug else "agentic",
                    "read_mode": self.config.read_mode,
                    "partial_run": True,
                    "stages": selected_stages,
                    "budget": self.budget.public_dict(),
                }
                write_json(
                    self.data_dir / "run.json",
                    {
                        "status": "partial_completed",
                        "config": self.config.public_dict(),
                        "manifest": manifest,
                    },
                )
            self.events.emit("run_completed", message="Pipeline completed", data=manifest)
            return manifest
        except Exception as exc:
            self.write_failed_run_json(error=str(exc), stages=selected_stages)
            self.events.emit(
                "run_failed", level="critical", message=str(exc), data={"stages": selected_stages}
            )
            raise
        finally:
            self.db.close()

    def load_completed_state_for_stage(self, stage: str) -> None:
        stage = normalize_workflow_stage(stage) or stage
        index = WORKFLOW_STAGE_ORDER.index(stage)
        active_ids = self.active_paper_ids_from_state()
        if index >= WORKFLOW_STAGE_ORDER.index("stage_01_parse") and not active_ids:
            raise RuntimeError(
                "Cannot resume PaperLens workflow: missing active_paper_ids in run_state. "
                "Rerun from stage_00_ingest or use a clean output directory."
            )
        if index >= WORKFLOW_STAGE_ORDER.index("stage_01_parse"):
            self.papers = self.filter_current_papers(self.db.list_papers(), active_ids)
        if index >= WORKFLOW_STAGE_ORDER.index("stage_03_skim"):
            self.skim_cards = self.filter_current_items(self.db.list_skim_cards(), active_ids)
        if index >= WORKFLOW_STAGE_ORDER.index("stage_03_skim"):
            self.classifications = self.filter_current_items(
                self.db.list_classifications(), active_ids
            )
        self.validate_loaded_state_for_stage(stage)
        self.events.emit(
            "resume_state_loaded",
            stage=stage,
            message=f"Loaded prior pipeline state for {stage}",
            data={
                "papers": len(self.papers),
                "skim_cards": len(self.skim_cards),
                "classifications": len(self.classifications),
            },
        )

    def active_paper_ids_from_state(self) -> set[str]:
        value = self.db.get_state("active_paper_ids", [])
        if not isinstance(value, list):
            return set()
        return {str(item) for item in value if str(item).strip()}

    def filter_current_papers(
        self, papers: list[PaperRecord], active_ids: set[str]
    ) -> list[PaperRecord]:
        if not active_ids:
            return papers
        return [paper for paper in papers if paper.paper_id in active_ids]

    def filter_current_items(self, items: list[Any], active_ids: set[str]) -> list[Any]:
        if not active_ids:
            return items
        return [item for item in items if getattr(item, "paper_id", None) in active_ids]

    def validate_loaded_state_for_stage(self, stage: str) -> None:
        index = WORKFLOW_STAGE_ORDER.index(stage)
        requirements = [
            ("stage_01_parse", self.papers, "paper records"),
            ("stage_07_normal_read", self.skim_cards, "paper maps"),
            ("stage_07_normal_read", self.classifications, "reading metadata"),
        ]
        missing = [
            name
            for required_stage, values, name in requirements
            if index >= WORKFLOW_STAGE_ORDER.index(required_stage) and not values
        ]
        if missing:
            raise RuntimeError(
                f"Cannot resume from {stage}; missing prior state: {', '.join(missing)}"
            )

    def prepare_output(self) -> None:
        for relative in [
            ".paperlens",
            ".paperlens/pages",
            ".paperlens/figures",
            ".paperlens/data",
            ".paperlens/data/artifacts/layout",
            ".paperlens/library",
            ".paperlens/library/index",
            ".paperlens/cache",
            "papers",
        ]:
            (self.output_dir / relative).mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            self.data_dir / "run.json",
            {"status": "running", "config": self.config.public_dict()},
        )

    def write_failed_run_json(self, *, error: str, stages: list[str]) -> None:
        write_json(
            self.data_dir / "run.json",
            {
                "status": "failed",
                "error": error,
                "config": self.config.public_dict(),
                "stages": stages,
                "budget": self.budget.public_dict(),
            },
        )

    def checkpoint(self, stage: str) -> None:
        self.control.wait_if_paused()
        self.control.require_not_cancelled()
        self.db.set_state("last_stage", stage)

    def mark_paper_state(
        self,
        paper_id: str,
        stage: str,
        *,
        side_statuses: list[str] | None = None,
        error: str | None = None,
    ) -> None:
        state = transition_state(
            self.db.get_paper_state(paper_id),
            paper_id=paper_id,
            stage=stage,
            side_statuses=side_statuses,
            error=error,
        )
        self.db.upsert_paper_state(state)

    def register_file_artifact(
        self,
        path: Path,
        *,
        paper_id: str | None,
        artifact_type: str,
        depends_on: list[str] | None = None,
    ) -> None:
        if not path.exists():
            return
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        artifact = ArtifactVersion(
            artifact_id=f"{artifact_type}:{paper_id or 'run'}:{path.name}:{content_hash[:12]}",
            paper_id=paper_id,
            artifact_type=artifact_type,
            path=str(path),
            content_hash=content_hash,
            depends_on=depends_on or [],
        )
        self.db.upsert_artifact_version(artifact)

    def stage_00_ingest(self) -> None:
        stage = "stage_00_ingest"
        self.checkpoint(stage)
        self.events.stage_started(stage, "Scanning PDFs")
        self.papers = scan_pdfs(self.input_dir)
        active_ids = [paper.paper_id for paper in self.papers]
        self.db.set_state("active_run_id", self.events.run_id)
        self.db.set_state("active_input_dir", str(self.input_dir))
        self.db.set_state("active_paper_ids", active_ids)
        for paper in self.papers:
            paper.status = "INGESTED"
            self.db.upsert_paper(paper)
            self.mark_paper_state(paper.paper_id, stage)
        self.events.stage_completed(stage, f"Found {len(self.papers)} PDF files")

    def stage_01_parse(self) -> None:
        stage = "stage_01_parse"
        self.checkpoint(stage)
        self.events.stage_started(stage, "Parsing PDFs with PyMuPDF")
        if not self.papers:
            self.events.stage_completed(stage, "No PDFs to parse")
            return

        def parse_one(paper: PaperRecord) -> tuple[PaperRecord, list[Any], str, dict[str, Any]]:
            parsed_paper, artifacts = parse_pdf(
                paper,
                self.evidence_dir,
                render_zoom=self.config.render_zoom,
            )
            quality, metrics = parse_quality(artifacts)
            return parsed_paper, artifacts, quality, metrics

        completed = 0
        parsed_by_id: dict[str, PaperRecord] = {}
        max_workers = min(max(1, self.config.concurrency), len(self.papers))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for paper in self.papers:
                self.checkpoint(stage)
                self.events.emit(
                    "paper_started",
                    stage=stage,
                    message=f"Parsing {paper.canonical_title or paper.paper_id}",
                    data={"paper_id": paper.paper_id},
                )
                futures[executor.submit(parse_one, paper)] = paper

            for future in as_completed(futures):
                paper = futures[future]
                completed += 1
                progress = completed / len(self.papers)
                self.checkpoint(stage)
                try:
                    parsed_paper, artifacts, quality, metrics = future.result()
                    parsed_paper.parse_quality = quality
                    parsed_paper.status = "PARSE_VERIFIED"
                    parsed_by_id[parsed_paper.paper_id] = parsed_paper
                    self.db.upsert_paper(parsed_paper)
                    self.db.insert_page_artifacts(artifacts)
                    layout_index = build_layout_index(parsed_paper, artifacts, metrics)
                    layout_path = self.data_dir / "artifacts" / "layout" / f"{paper.paper_id}.json"
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
                    self.register_file_artifact(
                        layout_path,
                        paper_id=paper.paper_id,
                        artifact_type="layout_index",
                    )
                    for artifact in artifacts:
                        if artifact.render_path:
                            self.register_file_artifact(
                                Path(artifact.render_path),
                                paper_id=paper.paper_id,
                                artifact_type="page_render",
                            )
                    side_statuses = []
                    if quality == "OCR_REQUIRED":
                        side_statuses.append("NEED_VISUAL_RECHECK")
                        self.db.upsert_review_item(
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
                        self.db.upsert_review_item(
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
                        self.db.upsert_review_item(
                            ReviewItem(
                                item_id=f"visual:{paper.paper_id}",
                                paper_id=paper.paper_id,
                                item_type="NEED_VISUAL_RECHECK",
                                priority=2,
                                reason="visual_required_pages",
                                payload={"pages": visual_pages},
                            )
                        )
                    self.mark_paper_state(paper.paper_id, stage, side_statuses=side_statuses)
                    self.events.emit(
                        "paper_completed",
                        stage=stage,
                        progress=progress,
                        message=f"Parsed {paper.paper_id}",
                        data={"paper_id": paper.paper_id, "parse_quality": quality},
                    )
                except Exception as exc:
                    paper.status = "FAILED"
                    paper.parse_quality = "FAIL"
                    self.db.upsert_paper(paper)
                    self.mark_paper_state(paper.paper_id, stage, error=str(exc))
                    self.events.error(
                        stage,
                        f"Failed to parse {paper.file_path}: {exc}",
                        {"paper_id": paper.paper_id},
                    )
        self.papers = [parsed_by_id.get(paper.paper_id, paper) for paper in self.papers]
        self.events.stage_completed(stage, "Parse stage completed")

    def stage_02_parse_verify(self) -> None:
        stage = "stage_02_parse_verify"
        self.checkpoint(stage)
        self.events.stage_started(stage, "Verifying parse quality and VLM page enrichment")
        if not self.papers:
            self.events.stage_completed(stage, "No PDFs to verify")
            return
        client = self.new_llm_client() if self.llm_enabled() else None
        visual_rows = []
        for paper in self.papers:
            artifacts = self.db.get_page_artifacts(paper.paper_id)
            visual_pages = self.visual_pages_for_parse_verification(paper, artifacts)
            if client and visual_pages:
                visual_results = []
                for batch in chunked(visual_pages, self.config.visual_pages_per_call):
                    visual_results.append(
                        self.run_vlm_page_mode(
                            client=client,
                            paper=paper,
                            artifacts=batch,
                            stage=stage,
                        )
                    )
                if visual_results:
                    visual_rows.extend(visual_results)
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
                                flag
                                for flag in page.low_confidence_flags
                                if flag != "visual_required"
                            ]
                            page.visual_required = False
                    self.db.insert_page_artifacts(artifacts)
                    paper.parse_quality = (
                        "PASS_WITH_WEAKNESSES"
                        if paper.parse_quality in {"OCR_REQUIRED", "VLM_PAGE_MODE"}
                        else paper.parse_quality
                    )
                    paper.status = "PARSE_VERIFIED"
                    self.db.upsert_paper(paper)
            side = []
            if paper.parse_quality in {"OCR_REQUIRED", "VLM_PAGE_MODE"}:
                side.append("NEED_VISUAL_RECHECK")
            self.mark_paper_state(paper.paper_id, stage, side_statuses=side)
        _ = visual_rows
        self.events.stage_completed(stage, "Parse verification completed")

    def visual_pages_for_parse_verification(
        self,
        paper: PaperRecord,
        artifacts: list[Any],
    ) -> list[Any]:
        mode = self.config.visual_verification_mode
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
        ][: self.config.visual_verification_max_pages]

    def run_vlm_page_mode(
        self,
        *,
        client: JsonLlmClient,
        paper: PaperRecord,
        artifacts: list[Any],
        stage: str,
    ) -> dict[str, Any]:
        pages = [artifact.page_no for artifact in artifacts]
        image_paths = [Path(artifact.render_path) for artifact in artifacts if artifact.render_path]
        key_payload = {
            "version": "vlm-page-v1",
            "model": self.config.provider.model,
            "visual_detail": self.config.visual_detail,
            "paper_hash": paper.file_hash,
            "pages": pages,
            "text_hashes": [hash_text(getattr(artifact, "text", "")) for artifact in artifacts],
            "image_hashes": [hash_file_bytes(path) for path in image_paths],
        }
        cache_path = self.cache_path("vlm_page_notes", paper.paper_id, key_payload)
        cached = self.read_cache_payload(cache_path)
        if cached and isinstance(cached.get("data"), dict):
            self.events.emit(
                "cache_hit",
                stage=stage,
                message=f"VLM page cache hit for {paper.paper_id}",
                data={"paper_id": paper.paper_id, "pages": pages, "cache": str(cache_path)},
            )
            data = cached["data"]
            page_notes = data.get("page_notes") if isinstance(data.get("page_notes"), list) else []
            return {
                "paper_id": paper.paper_id,
                "agent_run_id": str(cached.get("agent_run_id") or f"vlm_{paper.paper_id}_cache"),
                "page_notes": page_notes,
                "visual_summary": data.get("visual_summary"),
                "risk_notes": data.get("risk_notes"),
            }
        agent_run_id = f"vlm_{paper.paper_id}_{uuid.uuid4().hex[:8]}"
        self.events.emit(
            "agent_run_started",
            stage=stage,
            message=f"VLM page read {paper.paper_id}",
            data={"paper_id": paper.paper_id, "agent_run_id": agent_run_id},
        )
        with llm_call_context(
            stage=stage,
            paper_id=paper.paper_id,
            operation="vlm_page_read",
            schema_name="paperlens_vlm_page_notes",
            pages=pages,
        ):
            raw = client.invoke_json_with_images(
                system_prompt=VLM_PAGE_READER_SYSTEM_PROMPT,
                user_prompt=build_vlm_page_prompt(paper=paper, artifacts=artifacts),
                image_paths=image_paths,
                schema_name="paperlens_vlm_page_notes",
                schema=VLM_PAGE_NOTES_SCHEMA,
                max_tokens=None,
                detail=self.config.visual_detail,
            )
        self.write_agent_run(
            {
                "agent_run_id": agent_run_id,
                "paper_id": paper.paper_id,
                "stage": stage,
                "provider_kind": self.config.provider.kind,
                "model": self.config.provider.model,
                "endpoint": raw.endpoint,
                "request_id": raw.request_id,
                "usage": raw.usage,
                "status": "PASS",
            }
        )
        self.record_llm_usage(stage, raw.usage)
        self.write_cache_payload(
            cache_path,
            {
                "key": key_payload,
                "data": raw.data,
                "usage": raw.usage,
                "request_id": raw.request_id,
                "endpoint": raw.endpoint,
                "agent_run_id": agent_run_id,
            },
        )
        page_notes = (
            raw.data.get("page_notes") if isinstance(raw.data.get("page_notes"), list) else []
        )
        return {
            "paper_id": paper.paper_id,
            "agent_run_id": agent_run_id,
            "page_notes": page_notes,
            "visual_summary": raw.data.get("visual_summary"),
            "risk_notes": raw.data.get("risk_notes"),
        }

    def stage_03_skim(self) -> None:
        stage = "stage_03_skim"
        self.checkpoint(stage)
        self.events.stage_started(stage, "Building deterministic paper maps")
        active_ids = {paper.paper_id for paper in self.papers}
        existing_skim_by_id = {
            card.paper_id: card
            for card in (self.skim_cards or self.db.list_skim_cards())
            if card.paper_id in active_ids
        }
        existing_decision_by_id = {
            decision.paper_id: decision
            for decision in (self.classifications or self.db.list_classifications())
            if decision.paper_id in active_ids
        }
        self.skim_cards = list(existing_skim_by_id.values())
        self.classifications = list(existing_decision_by_id.values())
        pending: list[tuple[PaperRecord, SkimCard, ClassificationDecision]] = []
        for paper in self.papers:
            if paper.paper_id in existing_skim_by_id and paper.paper_id in existing_decision_by_id:
                self.events.emit(
                    "cache_hit",
                    stage=stage,
                    message=f"Skim/classification already exists for {paper.paper_id}",
                    data={"paper_id": paper.paper_id},
                )
                self.mark_paper_state(paper.paper_id, stage)
                continue
            artifacts = self.db.get_page_artifacts(paper.paper_id)
            card, decision = deterministic_skim_classify(paper, artifacts, self.config.keyword_pool)
            pending.append((paper, card, decision))

        for paper, card, decision in pending:
            self.persist_skim_classification(stage, paper, card, decision)
        self.order_skim_classification_state()
        core_v2_count = self.persist_core_v2_artifacts(stage)
        self.events.stage_completed(
            stage,
            "Paper maps completed",
            {
                "skim_cards": len(self.skim_cards),
                "classifications": len(self.classifications),
                "core_v2_artifacts": core_v2_count,
            },
        )

    def persist_core_v2_artifacts(self, stage: str) -> int:
        skim_by_id = {card.paper_id: card for card in self.skim_cards}
        decision_by_id = {decision.paper_id: decision for decision in self.classifications}
        written_count = 0
        for paper in self.papers:
            layout = load_layout_index(self.data_dir, paper.paper_id)
            if not layout:
                artifacts = self.db.get_page_artifacts(paper.paper_id)
                layout = {"pages": [artifact.model_dump() for artifact in artifacts]}
            paths = write_core_v2_artifacts(
                data_dir=self.data_dir,
                paper=paper,
                layout=layout,
                skim=skim_by_id.get(paper.paper_id),
                decision=decision_by_id.get(paper.paper_id),
            )
            for artifact_type, path in paths.items():
                self.register_file_artifact(
                    path,
                    paper_id=paper.paper_id,
                    artifact_type=f"core_v2_{artifact_type}",
                    depends_on=[f"layout_index:{paper.paper_id}"],
                )
            written_count += 1
            self.events.emit(
                "core_v2_artifacts_written",
                stage=stage,
                message=f"Core v2 artifacts written for {paper.paper_id}",
                data={
                    "paper_id": paper.paper_id,
                    "artifacts": {key: str(path) for key, path in paths.items()},
                },
            )
        return written_count

    def persist_skim_classification(
        self,
        stage: str,
        paper: PaperRecord,
        card: SkimCard,
        decision: ClassificationDecision,
    ) -> None:
        if any(item.paper_id == card.paper_id for item in self.skim_cards):
            self.skim_cards = [
                card if item.paper_id == card.paper_id else item for item in self.skim_cards
            ]
        else:
            self.skim_cards.append(card)
        if any(item.paper_id == decision.paper_id for item in self.classifications):
            self.classifications = [
                decision if item.paper_id == decision.paper_id else item
                for item in self.classifications
            ]
        else:
            self.classifications.append(decision)
        self.db.upsert_skim(card)
        self.db.upsert_classification(decision)
        self.mark_paper_state(paper.paper_id, stage)

    def order_skim_classification_state(self) -> None:
        skim_by_id = {card.paper_id: card for card in self.skim_cards}
        decision_by_id = {decision.paper_id: decision for decision in self.classifications}
        self.skim_cards = [
            skim_by_id[paper.paper_id] for paper in self.papers if paper.paper_id in skim_by_id
        ]
        self.classifications = [
            decision_by_id[paper.paper_id]
            for paper in self.papers
            if paper.paper_id in decision_by_id
        ]

    def llm_enabled(self) -> bool:
        if self.config.offline_debug:
            return False
        self.config.validate_agentic_run()
        return True

    def write_agent_run(self, payload: dict[str, Any]) -> None:
        row = {
            "time": utc_timestamp(),
            "run_id": self.events.run_id,
            **payload,
        }
        path = self.data_dir / "agent_runs.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=str) + "\n")

    def record_llm_usage(self, stage: str, usage: dict[str, Any]) -> None:
        snapshot = self.budget.record_usage(usage)
        self.events.emit(
            "usage_update",
            stage=stage,
            message="Model usage recorded",
            data={
                "input_tokens": snapshot.input_tokens,
                "output_tokens": snapshot.output_tokens,
                "estimated_usd": snapshot.estimated_usd,
                "calls": snapshot.calls,
            },
        )

    def cache_path(self, stage: str, paper_id: str, key_payload: dict[str, Any]) -> Path:
        path = llm_cache_path(self.cache_dir, stage, paper_id, key_payload)
        if path is None:
            raise RuntimeError("PaperLens workflow cache directory is not configured")
        return path

    def read_cache_payload(self, path: Path) -> dict[str, Any] | None:
        return read_llm_cache(path)

    def write_cache_payload(self, path: Path, payload: dict[str, Any]) -> None:
        write_llm_cache(path, payload)

    def stage_07_normal_read(self) -> None:
        stage = "stage_07_normal_read"
        self.checkpoint(stage)
        llm_enabled = self.llm_enabled()
        self.events.stage_started(
            stage,
            "Building core v2 ObservationLog and ClaimGraph from PaperDOM evidence"
            if llm_enabled
            else "Using deterministic core v2 bootstrap artifacts",
        )
        if llm_enabled and self.papers:
            client = self.new_llm_client()
            for paper in self.papers:
                self.control.wait_if_paused()
                self.control.require_not_cancelled()
                self.events.emit(
                    "agent_run_started",
                    stage=stage,
                    message=f"Core v2 observation read {paper.paper_id}",
                    data={"paper_id": paper.paper_id, "read_mode": self.config.read_mode},
                )
                self.run_core_v2_observation_read(
                    client=client,
                    stage=stage,
                    paper=paper,
                )
                self.mark_paper_state(paper.paper_id, stage)
                self.events.emit(
                    "agent_run_completed",
                    stage=stage,
                    message=f"Core v2 observation read completed for {paper.paper_id}",
                    data={"paper_id": paper.paper_id},
                )
        else:
            for paper in self.papers:
                self.mark_paper_state(paper.paper_id, stage)
        self.events.stage_completed(
            stage,
            "Core v2 observation stage completed",
            {"papers": len(self.papers), "llm_enabled": llm_enabled},
        )

    def run_core_v2_observation_read(
        self,
        *,
        client: JsonLlmClient,
        stage: str,
        paper: PaperRecord,
    ) -> None:
        try:
            result = run_core_v2_model_observation_tasks(
                client=client,
                data_dir=self.data_dir,
                paper=paper,
                stage=stage,
                record_usage=self.record_llm_usage,
                record_agent_run=self.write_agent_run,
            )
            for artifact_type, path in result["paths"].items():
                self.register_file_artifact(
                    path,
                    paper_id=paper.paper_id,
                    artifact_type=f"core_v2_model_{artifact_type}",
                    depends_on=[f"core_v2_reading_plan:{paper.paper_id}"],
                )
            self.events.emit(
                "core_v2_observation_read_completed",
                stage=stage,
                message=f"Core v2 observation read completed for {paper.paper_id}",
                data={
                    "paper_id": paper.paper_id,
                    "tasks": result["tasks"],
                    "cards": result["cards"],
                },
            )
        except Exception as exc:
            self.write_agent_run(
                {
                    "agent_run_id": f"core_v2_observe_{paper.paper_id}_failed",
                    "paper_id": paper.paper_id,
                    "stage": stage,
                    "operation": "core_v2_observation_read",
                    "provider_kind": client.config.kind,
                    "model": client.config.model,
                    "status": "FAIL",
                    "error": str(exc),
                }
            )
            raise

    def refresh_core_v2_deterministic_audits(self, stage: str) -> list[dict[str, Any]]:
        skim_by_id = {card.paper_id: card for card in self.skim_cards}
        decision_by_id = {decision.paper_id: decision for decision in self.classifications}
        rows: list[dict[str, Any]] = []
        for paper in self.papers:
            try:
                result = refresh_core_v2_audit_artifacts(
                    data_dir=self.data_dir,
                    paper=paper,
                    skim=skim_by_id.get(paper.paper_id),
                    decision=decision_by_id.get(paper.paper_id),
                )
            except FileNotFoundError:
                if (self.data_dir / "core" / "v2" / paper.paper_id).exists():
                    raise
                continue
            for artifact_type, path in result["paths"].items():
                self.register_file_artifact(
                    path,
                    paper_id=paper.paper_id,
                    artifact_type=f"core_v2_audit_{artifact_type}",
                    depends_on=[
                        f"core_v2_paper_dom:{paper.paper_id}",
                        f"core_v2_claim_graph:{paper.paper_id}",
                    ],
                )
            side_statuses = []
            publish_status = str(result["publish_status"])
            if publish_status != "REVIEWED":
                side_statuses.append(f"CORE_V2_{publish_status}")
            self.mark_paper_state(paper.paper_id, stage, side_statuses=side_statuses)
            row = {
                "paper_id": paper.paper_id,
                "publish_status": publish_status,
                "graph_findings": result["graph_findings"],
                "report_findings": result["report_findings"],
            }
            rows.append(row)
            self.events.emit(
                "core_v2_audit_completed",
                stage=stage,
                message=f"Core v2 deterministic audit completed for {paper.paper_id}",
                data=row,
            )
        return rows

    def stage_08_evidence_verify(self) -> None:
        stage = "stage_08_evidence_verify"
        self.checkpoint(stage)
        self.events.stage_started(stage, "Running deterministic core v2 audit suite")
        core_v2_rows = self.refresh_core_v2_deterministic_audits(stage)
        self.events.stage_completed(
            stage,
            "Core v2 deterministic audit completed",
            {"core_v2_audits": len(core_v2_rows)},
        )

    def stage_15_export(self) -> list[Path]:
        stage = "stage_15_export"
        self.checkpoint(stage)
        self.events.stage_started(stage, "Writing final reading reports")
        active_ids = {paper.paper_id for paper in self.papers}
        report_paths = write_final_report_bundle(
            output_dir=self.output_dir,
            data_dir=self.data_dir,
            papers=self.papers,
            skim_cards=self.skim_cards,
            decisions=self.classifications,
            review_items=[
                item for item in self.db.list_review_items() if item.paper_id in active_ids
            ],
            budget=self.budget.public_dict(),
            budget_provider=self.budget.public_dict,
            config=self.config.public_dict(),
            topic=self.config.topic,
            idea=self.config.idea,
        )
        self.events.stage_completed(
            stage,
            "Final reading reports written",
            {"reports": [str(path) for path in report_paths]},
        )
        return report_paths

    def stage_17_manifest(self) -> dict[str, Any]:
        stage = "stage_17_manifest"
        self.checkpoint(stage)
        self.events.stage_started(stage, "Writing manifest")
        output_validation = validate_paperlens_output(
            self.output_dir,
            expected_report_names={paper_report_filename(paper) for paper in self.papers},
            expected_paper_ids={paper.paper_id for paper in self.papers},
        )
        model_call_summary = summarize_model_calls(self.data_dir / "model_calls.jsonl")
        write_json(self.data_dir / "model_call_summary.json", model_call_summary)
        core_quality_snapshot_path = write_core_quality_snapshot(self.output_dir)
        core_quality_snapshot_payload = json.loads(
            core_quality_snapshot_path.read_text(encoding="utf-8")
        )
        core_quality_snapshot_data = dict_value(core_quality_snapshot_payload.get("data"))
        manifest = {
            "run_id": self.events.run_id,
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "paper_count": len(self.papers),
            "mode": "offline_debug" if self.config.offline_debug else "agentic",
            "read_mode": self.config.read_mode,
            "topic_comparison_enabled": self.config.topic_comparison_enabled,
            "output_language": self.config.output_language,
            "classification_counts": classification_counts(self.classifications),
            "artifacts": {
                "main_report": "PaperLens.md",
                "paper_reports": "papers/",
                "internal_state": ".paperlens/state.sqlite",
                "page_images": ".paperlens/pages/",
                "figure_crops": ".paperlens/figures/",
                "library_records": ".paperlens/library/library_records.jsonl",
                "core_v2": ".paperlens/data/core/v2/",
                "core_quality_snapshot": ".paperlens/data/core_quality_snapshot.v1.json",
                "library_index": ".paperlens/library/index/search_index.json",
                "data": ".paperlens/data/",
                "model_call_summary": ".paperlens/data/model_call_summary.json",
                "output_validation": output_validation,
            },
            "model_calls": model_call_summary,
            "core_quality": {
                "paper_count": core_quality_snapshot_data.get("paper_count"),
                "aggregate": dict_value(core_quality_snapshot_data.get("aggregate")),
            },
            "budget": self.budget.public_dict(),
        }
        write_json(
            self.data_dir / "run.json",
            {
                "status": "completed",
                "config": self.config.public_dict(),
                "manifest": manifest,
            },
        )
        self.events.stage_completed(stage, "Manifest written", manifest)
        return manifest


VLM_PAGE_READER_SYSTEM_PROMPT = """
You are the PaperLens VLMPageReader.
Read rendered PDF page images directly. Extract only visible page facts useful for later
evidence binding: architecture diagrams, tables, plots, captions, section titles, and text
that was likely missed by text extraction. Do not invent content. Return JSON only.
""".strip()


VLM_PAGE_NOTES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["visual_summary", "page_notes", "risk_notes"],
    "properties": {
        "visual_summary": {"type": "string"},
        "risk_notes": {"type": "array", "items": {"type": "string"}},
        "page_notes": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["page_no", "visible_text", "figures", "tables", "evidence_queries"],
                "properties": {
                    "page_no": {"type": "integer", "minimum": 1},
                    "visible_text": {"type": "array", "items": {"type": "string"}},
                    "figures": {"type": "array", "items": {"type": "string"}},
                    "tables": {"type": "array", "items": {"type": "string"}},
                    "evidence_queries": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["keyword", "quote"],
                            "properties": {
                                "keyword": {"type": ["string", "null"]},
                                "quote": {"type": ["string", "null"]},
                            },
                        },
                    },
                },
            },
        },
    },
}


def deterministic_skim_classify(
    paper: PaperRecord,
    artifacts: list[Any],
    keyword_pool: list[str],
) -> tuple[SkimCard, ClassificationDecision]:
    text = "\n".join(page.text for page in artifacts[:3])
    signals = keyword_hits(text + "\n" + (paper.canonical_title or ""), keyword_pool)
    first_ref = first_evidence_ref(paper.paper_id, artifacts, signals)
    card = SkimCard(
        paper_id=paper.paper_id,
        problem=first_sentence(text) or paper.canonical_title,
        method_type=infer_method_type(text),
        system_scope=infer_scope(text),
        evaluation_type=infer_evaluation(text),
        danger_signals=signals,
        evidence_refs=[first_ref] if first_ref else [],
        confidence=min(0.9, 0.35 + 0.12 * len(signals)),
    )
    return card, classify_paper(paper, card)


def build_vlm_page_prompt(*, paper: PaperRecord, artifacts: list[Any]) -> str:
    pages = []
    for artifact in artifacts:
        pages.append(
            {
                "page_no": artifact.page_no,
                "render_path": artifact.render_path,
                "parse_flags": artifact.low_confidence_flags,
                "text_preview": normalize_excerpt(artifact.text or "", limit=900),
                "captions": artifact.captions[:4],
            }
        )
    return "\n\n".join(
        [
            f"paper_id: {paper.paper_id}",
            f"title: {paper.canonical_title or 'unknown'}",
            f"parse_quality: {paper.parse_quality or 'unknown'}",
            "Images are supplied in the same order as this page metadata.",
            "Extract visible page facts, figure/table readings, and evidence queries.",
            json.dumps(pages, ensure_ascii=False),
        ]
    )


def dedupe_artifacts_by_page(artifacts: list[Any]) -> list[Any]:
    seen = set()
    result = []
    for artifact in artifacts:
        page_no = getattr(artifact, "page_no", None)
        if page_no in seen:
            continue
        seen.add(page_no)
        result.append(artifact)
    return result


def hash_file_bytes(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "missing"


def normalize_excerpt(text: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0] + " ..."


def normalize_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def hash_text(value: str) -> str:
    return hashlib.sha256(normalize_for_search(value).encode("utf-8")).hexdigest()[:16]


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    hits = []
    for keyword in keywords:
        if keyword.lower() in lowered:
            hits.append(keyword)
    return sorted(set(hits), key=str.lower)


def first_sentence(text: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return None
    match = re.match(r"(.{30,240}?[.!?])\s", cleaned)
    return match.group(1) if match else cleaned[:240]


def first_evidence_ref(
    paper_id: str,
    artifacts: list[Any],
    signals: list[str],
) -> EvidenceRef | None:
    if not artifacts:
        return None
    needle = signals[0] if signals else None
    for page in artifacts:
        if needle and needle.lower() not in page.text.lower():
            continue
        bbox = page.words[0]["bbox"] if page.words else None
        return EvidenceRef(
            paper_id=paper_id,
            page_no=page.page_no,
            bbox=bbox,
            text_span_id=f"page_{page.page_no}_first_match",
            verification_status="WEAK_VERIFIED",
        )
    page = artifacts[0]
    return EvidenceRef(
        paper_id=paper_id,
        page_no=page.page_no,
        bbox=page.words[0]["bbox"] if page.words else None,
        text_span_id=f"page_{page.page_no}_first_span",
        verification_status="WEAK_VERIFIED",
    )


def infer_method_type(text: str) -> str:
    lowered = text.lower()
    if "system" in lowered or "runtime" in lowered:
        return "system"
    if "analysis" in lowered or "formal" in lowered:
        return "analysis"
    if "survey" in lowered:
        return "survey"
    return "unknown"


def infer_scope(text: str) -> str:
    lowered = text.lower()
    if "webassembly" in lowered or "wasm" in lowered:
        return "webassembly_runtime"
    if "kernel" in lowered:
        return "kernel"
    if "virtual machine" in lowered or "vm" in lowered:
        return "vm"
    return "unknown"


def infer_evaluation(text: str) -> str:
    lowered = text.lower()
    if "benchmark" in lowered or "throughput" in lowered or "latency" in lowered:
        return "performance"
    if "case study" in lowered:
        return "case_study"
    if "proof" in lowered:
        return "formal"
    return "unknown"


def classify_paper(paper: PaperRecord, card: SkimCard) -> ClassificationDecision:
    if paper.parse_quality in {"OCR_REQUIRED", "VLM_PAGE_MODE"}:
        return ClassificationDecision(
            paper_id=paper.paper_id,
            class_label="HOLD",
            confidence=0.3,
            false_negative_risk=0.8,
            reason_codes=[str(paper.parse_quality).lower()],
        )
    if paper.parse_quality == "PASS_WITH_WEAKNESSES" and not card.evidence_refs:
        return ClassificationDecision(
            paper_id=paper.paper_id,
            class_label="HOLD",
            confidence=0.35,
            false_negative_risk=0.7,
            reason_codes=["weak_parse_without_evidence"],
        )
    signal_count = len(card.danger_signals)
    if signal_count >= 3:
        label = "A"
    elif signal_count >= 1:
        label = "B"
    else:
        label = "C"
    return ClassificationDecision(
        paper_id=paper.paper_id,
        class_label=label,
        confidence=min(0.92, 0.45 + 0.15 * signal_count),
        false_negative_risk=max(0.25 if label == "C" else 0.1, 0.75 - 0.16 * signal_count),
        reason_codes=[f"keyword:{signal}" for signal in card.danger_signals]
        or ["no_keyword_signal"],
    )


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    size = max(1, size)
    return [items[index : index + size] for index in range(0, len(items), size)]


def bounded_env_int(name: str, *, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def write_final_report_bundle(
    *,
    output_dir: Path,
    data_dir: Path,
    papers: list[PaperRecord],
    skim_cards: list[SkimCard],
    decisions: list[ClassificationDecision],
    review_items: list[ReviewItem],
    budget: dict[str, Any],
    budget_provider: Any | None = None,
    config: dict[str, Any],
    topic: str | None,
    idea: str | None,
) -> list[Path]:
    formal_run = not bool(config.get("offline_debug"))
    output_language = str(config.get("output_language") or "zh")
    if output_language not in {"zh", "en"}:
        output_language = "zh"
    read_mode = str(config.get("read_mode") or "standard")
    if read_mode != "standard":
        raise ValueError("PaperLens Core currently supports only read_mode='standard'")
    skim_by_id = {card.paper_id: card for card in skim_cards}
    decision_by_id = {decision.paper_id: decision for decision in decisions}
    paper_report_rows: list[dict[str, Any]] = []
    written: list[Path] = []

    for paper in papers:
        skim = skim_by_id.get(paper.paper_id)
        decision = decision_by_id.get(paper.paper_id)
        report_name = paper_report_filename(paper)
        report_path = output_dir / "papers" / report_name
        report_markdown = render_core_graph_report_view(
            data_dir=data_dir,
            paper_id=paper.paper_id,
            title=paper.canonical_title or paper.paper_id,
        )
        if report_markdown is None:
            raise RuntimeError(
                f"Core v2 report is required for {paper.paper_id}; export only publishes "
                "reviewed ClaimGraph reports"
            )
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_markdown, encoding="utf-8")
        written.append(report_path)
        graph_summary = read_core_v2_graph_summary(output_dir, paper.paper_id)
        paper_report_rows.append(
            {
                "paper": paper,
                "skim": skim,
                "decision": decision,
                "report_name": report_name,
                "core_graph_report_name": report_name,
                "report_title": markdown_title(report_markdown) or paper.canonical_title,
                "core_v2_graph_summary": graph_summary,
            }
        )

    final_budget = budget_provider() if budget_provider else budget
    paperlens_report = render_paperlens_report(
        rows=paper_report_rows,
        review_items=review_items,
        budget=final_budget,
        topic=topic,
        idea=idea,
        formal_run=formal_run,
        output_language=output_language,
    )
    main_path = output_dir / "PaperLens.md"
    main_path.write_text(paperlens_report, encoding="utf-8")
    written.append(main_path)
    written.extend(
        write_paperlens_library(
            output_dir=output_dir,
            rows=paper_report_rows,
            topic=topic,
            idea=idea,
        )
    )
    return written


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def load_layout_index(data_dir: Path, paper_id: str) -> dict[str, Any]:
    path = data_dir / "artifacts" / "layout" / f"{paper_id}.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def copy_resources(resources_dir: Path, output_dir: Path) -> None:
    if not resources_dir.exists():
        return
    target = output_dir / "resources_snapshot"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(resources_dir, target)
