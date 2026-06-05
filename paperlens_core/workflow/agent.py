from __future__ import annotations

import html
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

from paperlens_core.agent_loop import AgentLoop, PaperToolRegistry
from paperlens_core.agents.llm import JsonLlmClient, llm_call_context
from paperlens_core.agents.providers import describe_provider
from paperlens_core.budget import BudgetManager
from paperlens_core.config import CoreConfig
from paperlens_core.control import ControlState
from paperlens_core.db import ArtifactDb
from paperlens_core.events import EventWriter, write_json
from paperlens_core.library import write_paperlens_library
from paperlens_core.memory import (
    apply_memory_audit_patch,
    ensure_memory_audit_operation,
    ensure_read_pages_operation,
    fallback_memory_audit,
    memory_v3_pages_read,
    memory_without_audit,
    paper_memory_has_recoverable_content,
    select_central_verification_pages,
    select_high_risk_memory_claims,
)
from paperlens_core.memory_v3 import (
    dict_value,
    list_payload,
    memory_v3_prompt_view,
    safe_int,
    write_paper_memory_v3_file,
)
from paperlens_core.memory_store import (
    MEMORY_PATCH_SET_SCHEMA,
    PaperMemoryStore,
    normalize_memory_patch_set,
)
from paperlens_core.pdf.ingest import scan_pdfs
from paperlens_core.pdf.layout_index import build_layout_index
from paperlens_core.pdf.pymupdf_parser import parse_pdf
from paperlens_core.pdf.qa import parse_quality
from paperlens_core.quality_snapshot import write_core_quality_snapshot
from paperlens_core.reading import select_rolling_read_pages
from paperlens_core.report import (
    classification_counts,
    cluster_rows_by_scope,
    combine_report_and_memory_audits,
    dedupe_evidence_refs,
    describe_rows,
    display_paper_title,
    final_report_audit_acceptable,
    markdown_title,
    novelty_risk,
    paper_report_filename,
    read_decision,
    reading_priority_key,
    render_paperlens_report,
    report_link_lines,
    row_decision,
    row_relation,
    write_core_graph_report_view,
)
from paperlens_core.report.memory_context import (
    build_report_memory_context,
    compact_paper_memory_for_report,
    core_memory_view_dict,
    report_focus_pages,
    report_focus_queries,
)
from paperlens_core.runtime import PaperLensRuntime, context_pack_prompt
from paperlens_core.schemas import (
    ArtifactVersion,
    ClassificationDecision,
    EvidenceRef,
    PaperCard,
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
        self.paper_cards: list[PaperCard] = []
        self.memory_store = PaperMemoryStore(self.data_dir)
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
        if index >= WORKFLOW_STAGE_ORDER.index("stage_07_normal_read"):
            self.paper_cards = self.filter_current_items(self.db.list_paper_cards(), active_ids)
        self.validate_loaded_state_for_stage(stage)
        self.events.emit(
            "resume_state_loaded",
            stage=stage,
            message=f"Loaded prior pipeline state for {stage}",
            data={
                "papers": len(self.papers),
                "skim_cards": len(self.skim_cards),
                "classifications": len(self.classifications),
                "paper_cards": len(self.paper_cards),
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
            ".paperlens/cache/rolling_memory",
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
        self.events.stage_started(stage, "Verifying parse quality and optional VLM page fallback")
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
                    try:
                        visual_results.append(
                            self.run_vlm_page_mode(
                                client=client,
                                paper=paper,
                                artifacts=batch,
                                stage=stage,
                            )
                        )
                    except Exception as exc:
                        if self.require_llm_success():
                            raise
                        self.events.emit(
                            "vlm_page_fallback_failed",
                            stage=stage,
                            level="warning",
                            message=f"VLM page fallback failed for {paper.paper_id}",
                            data={
                                "paper_id": paper.paper_id,
                                "pages": [item.page_no for item in batch],
                                "error": str(exc),
                            },
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
        llm_enabled = False
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
        fallbacks: list[tuple[PaperRecord, list[Any], SkimCard, ClassificationDecision]] = []
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
            fallbacks.append((paper, artifacts, card, decision))

        if llm_enabled and fallbacks:
            max_workers = min(max(1, self.config.concurrency), len(fallbacks))

            def run_skim_task(
                index: int,
            ) -> tuple[int, SkimCard, ClassificationDecision, dict[str, Any]]:
                paper, artifacts, _card, _decision = fallbacks[index]
                card, decision, run_info = self.run_skim_classification(
                    client=self.new_llm_client(),
                    paper=paper,
                    artifacts=artifacts,
                    fallback_card=_card,
                    fallback_decision=_decision,
                )
                return index, card, decision, run_info

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for index, (paper, _artifacts, _card, _decision) in enumerate(fallbacks):
                    self.events.emit(
                        "agent_run_started",
                        stage=stage,
                        message=f"Model skim/classify {paper.paper_id}",
                        data={"paper_id": paper.paper_id},
                    )
                    futures[executor.submit(run_skim_task, index)] = index
                for future in as_completed(futures):
                    index = futures[future]
                    paper, _artifacts, fallback_card, fallback_decision = fallbacks[index]
                    try:
                        _index, card, decision, run_info = future.result()
                        if run_info.get("cache_hit"):
                            self.events.emit(
                                "cache_hit",
                                stage=stage,
                                message=f"Skim/classification cache hit for {paper.paper_id}",
                                data={
                                    "paper_id": paper.paper_id,
                                    "cache": run_info.get("cache"),
                                },
                            )
                        else:
                            self.write_agent_run(
                                {
                                    "agent_run_id": run_info.get("agent_run_id"),
                                    "paper_id": paper.paper_id,
                                    "stage": stage,
                                    "provider_kind": self.config.provider.kind,
                                    "model": self.config.provider.model,
                                    "endpoint": run_info.get("endpoint"),
                                    "request_id": run_info.get("request_id"),
                                    "usage": run_info.get("usage", {}),
                                    "status": "PASS",
                                }
                            )
                            self.events.emit(
                                "agent_run_completed",
                                stage=stage,
                                message=f"Model skim/classify completed for {paper.paper_id}",
                                data={
                                    "paper_id": paper.paper_id,
                                    "agent_run_id": run_info.get("agent_run_id"),
                                    "class_label": decision.class_label,
                                },
                            )
                            self.record_llm_usage(stage, run_info.get("usage", {}))
                        self.persist_skim_classification(stage, paper, card, decision)
                    except Exception as exc:
                        failed_run = {
                            "agent_run_id": f"skim_{paper.paper_id}_failed",
                            "paper_id": paper.paper_id,
                            "stage": stage,
                            "provider_kind": self.config.provider.kind,
                            "model": self.config.provider.model,
                            "status": "FAIL" if self.require_llm_success() else "FALLBACK",
                            "error": str(exc),
                        }
                        self.write_agent_run(failed_run)
                        if self.require_llm_success():
                            raise
                        self.events.emit(
                            "agent_run_fallback",
                            stage=stage,
                            level="warning",
                            message=f"Model skim/classify failed for {paper.paper_id}; using deterministic fallback",
                            data={"paper_id": paper.paper_id, "error": str(exc)},
                        )
                        self.persist_skim_classification(
                            stage,
                            paper,
                            fallback_card,
                            fallback_decision,
                        )

        elif fallbacks:
            for paper, _artifacts, fallback_card, fallback_decision in fallbacks:
                self.persist_skim_classification(stage, paper, fallback_card, fallback_decision)
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

    def run_skim_classification(
        self,
        *,
        client: JsonLlmClient,
        paper: PaperRecord,
        artifacts: list[Any],
        fallback_card: SkimCard,
        fallback_decision: ClassificationDecision,
    ) -> tuple[SkimCard, ClassificationDecision, dict[str, Any]]:
        user_prompt = build_skim_prompt(
            paper=paper,
            artifacts=artifacts,
            keyword_pool=self.config.keyword_pool,
            topic=self.config.topic,
            idea=self.config.idea,
        )
        prompt_pages = [artifact.page_no for artifact in artifacts[:6]]
        key_payload = {
            "version": SKIM_CLASSIFIER_PROMPT_VERSION,
            "model": self.config.provider.model,
            "paper_hash": paper.file_hash,
            "pages": prompt_pages,
            "page_hashes": [hash_text(getattr(artifact, "text", "")) for artifact in artifacts[:6]],
            "prompt_hash": hash_text(SKIM_CLASSIFIER_SYSTEM_PROMPT + "\n" + user_prompt),
            "schema_hash": hash_json_payload(SKIM_CLASSIFICATION_SCHEMA),
        }
        cache_path = self.cache_path("skim_classify", paper.paper_id, key_payload)
        cached = self.read_cache_payload(cache_path)
        if cached and isinstance(cached.get("data"), dict):
            agent_run_id = str(cached.get("agent_run_id") or f"skim_{paper.paper_id}_cache")
            card, decision = llm_skim_classify_to_models(
                paper=paper,
                artifacts=artifacts,
                raw=cached["data"],
                agent_run_id=agent_run_id,
                fallback_card=fallback_card,
                fallback_decision=fallback_decision,
            )
            return (
                card,
                decision,
                {
                    "agent_run_id": agent_run_id,
                    "cache_hit": True,
                    "cache": str(cache_path),
                },
            )

        agent_run_id = f"skim_{paper.paper_id}_{uuid.uuid4().hex[:8]}"
        with llm_call_context(
            stage="stage_03_skim",
            paper_id=paper.paper_id,
            operation="skim_classification",
        ):
            raw = client.invoke_json(
                system_prompt=SKIM_CLASSIFIER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema_name="paperlens_skim_classification",
                schema=SKIM_CLASSIFICATION_SCHEMA,
                max_tokens=None,
            )
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
        card, decision = llm_skim_classify_to_models(
            paper=paper,
            artifacts=artifacts,
            raw=raw.data,
            agent_run_id=agent_run_id,
            fallback_card=fallback_card,
            fallback_decision=fallback_decision,
        )
        return (
            card,
            decision,
            {
                "agent_run_id": agent_run_id,
                "cache_hit": False,
                "usage": raw.usage,
                "request_id": raw.request_id,
                "endpoint": raw.endpoint,
                "cache": str(cache_path),
            },
        )

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

    def require_llm_success(self) -> bool:
        explicit = os.getenv("PAPERLENS_REQUIRE_LLM")
        if explicit is not None:
            return explicit == "1"
        if os.getenv("PAPERLENS_ALLOW_LLM_FALLBACK", "0") == "1":
            return False
        return not self.config.offline_debug and self.config.provider.kind != "none"

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
        key = hashlib.sha256(
            json.dumps(key_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:24]
        safe_stage = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stage)
        safe_paper = re.sub(r"[^a-zA-Z0-9_.-]+", "_", paper_id)
        return self.cache_dir / safe_stage / safe_paper / f"{key}.json"

    def read_cache_payload(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def write_cache_payload(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json(path, payload)

    def stage_07_normal_read(self) -> None:
        stage = "stage_07_normal_read"
        self.checkpoint(stage)
        llm_enabled = self.llm_enabled()
        self.events.stage_started(
            stage,
            "Building audited paper memory and derived PaperCards"
            if llm_enabled
            else "Building deterministic PaperCards for papers that require reading",
        )
        active_ids = {paper.paper_id for paper in self.papers}
        existing_cards = {
            card.paper_id: card
            for card in (self.paper_cards or self.db.list_paper_cards())
            if card.paper_id in active_ids
        }
        self.paper_cards = list(existing_cards.values())
        candidates: list[
            tuple[PaperRecord, SkimCard, ClassificationDecision, list[Any], PaperCard]
        ] = []
        for paper, card, decision in zip(
            self.papers, self.skim_cards, self.classifications, strict=False
        ):
            if not should_run_normal_read(decision):
                continue
            if paper.paper_id in existing_cards:
                self.events.emit(
                    "cache_hit",
                    stage=stage,
                    message=f"PaperCard already exists for {paper.paper_id}",
                    data={"paper_id": paper.paper_id},
                )
                continue
            paper_card = deterministic_paper_card(paper, card, decision)
            artifacts = self.db.get_page_artifacts(paper.paper_id)
            candidates.append((paper, card, decision, artifacts, paper_card))

        if llm_enabled and candidates:
            client = self.new_llm_client()
            for paper, card, decision, artifacts, fallback in candidates:
                self.control.wait_if_paused()
                self.control.require_not_cancelled()
                self.events.emit(
                    "agent_run_started",
                    stage=stage,
                    message=f"Rolling read {paper.paper_id}",
                    data={"paper_id": paper.paper_id, "read_mode": self.config.read_mode},
                )
                try:
                    self.run_core_v2_observation_read(
                        client=client,
                        stage=stage,
                        paper=paper,
                    )
                    paper_card = self.run_rolling_paper_read(
                        client=client,
                        stage=stage,
                        paper=paper,
                        skim=card,
                        decision=decision,
                        artifacts=artifacts,
                        fallback=fallback,
                    )
                    self.persist_paper_card(stage, paper, paper_card)
                    self.events.emit(
                        "agent_run_completed",
                        stage=stage,
                        message=f"Rolling read completed for {paper.paper_id}",
                        data={"paper_id": paper.paper_id},
                    )
                except Exception as exc:
                    failed_run = {
                        "agent_run_id": f"reader_{paper.paper_id}_failed",
                        "paper_id": paper.paper_id,
                        "stage": stage,
                        "provider_kind": self.config.provider.kind,
                        "model": self.config.provider.model,
                        "status": "FAIL" if self.require_llm_success() else "FALLBACK",
                        "error": str(exc),
                    }
                    self.write_agent_run(failed_run)
                    if self.require_llm_success():
                        raise
                    self.events.emit(
                        "agent_run_fallback",
                        stage=stage,
                        level="warning",
                        message=f"Model normal-read failed for {paper.paper_id}; using deterministic PaperCard",
                        data={"paper_id": paper.paper_id, "error": str(exc)},
                    )
                    self.persist_paper_card(stage, paper, fallback)
        elif candidates:
            for paper, _card, _decision, _artifacts, fallback in candidates:
                self.control.wait_if_paused()
                self.control.require_not_cancelled()
                self.persist_paper_card(stage, paper, fallback)
        self.events.stage_completed(
            stage, "PaperCard generation completed", {"paper_cards": len(self.paper_cards)}
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
                    "status": "FAIL" if self.require_llm_success() else "FALLBACK",
                    "error": str(exc),
                }
            )
            if self.require_llm_success():
                raise
            self.events.emit(
                "agent_run_fallback",
                stage=stage,
                level="warning",
                message=(
                    f"Core v2 observation read failed for {paper.paper_id}; "
                    "keeping existing core v2 bootstrap artifacts"
                ),
                data={"paper_id": paper.paper_id, "error": str(exc)},
            )

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

    def persist_paper_card(self, stage: str, paper: PaperRecord, paper_card: PaperCard) -> None:
        if not any(card.paper_id == paper_card.paper_id for card in self.paper_cards):
            self.paper_cards.append(paper_card)
        else:
            self.paper_cards = [
                paper_card if card.paper_id == paper_card.paper_id else card
                for card in self.paper_cards
            ]
        self.db.upsert_paper_card(paper_card)
        self.mark_paper_state(paper.paper_id, stage)

    def run_rolling_paper_read(
        self,
        *,
        client: JsonLlmClient,
        stage: str,
        paper: PaperRecord,
        skim: SkimCard,
        decision: ClassificationDecision,
        artifacts: list[Any],
        fallback: PaperCard,
    ) -> PaperCard:
        selected_pages = select_rolling_read_pages(
            artifacts, skim, decision, read_mode=self.config.read_mode
        )
        if not selected_pages:
            return fallback
        try:
            chunk_size = int(os.getenv("PAPERLENS_ROLLING_CHUNK_PAGES", "3"))
        except ValueError:
            chunk_size = 3
        chunk_size = max(1, min(chunk_size, 4))
        memory = self.memory_store.initialize(
            paper=paper,
            skim=skim,
            decision=decision,
            card=fallback,
            layout=load_layout_index(self.data_dir, paper.paper_id),
            source=stage,
            prefer_existing=False,
        )
        page_chunks = chunked(selected_pages, chunk_size)
        failed_chunks: list[dict[str, Any]] = []
        for chunk_index, page_chunk in enumerate(page_chunks):
            self.control.wait_if_paused()
            self.control.require_not_cancelled()
            pages = [item.page_no for item in page_chunk]
            try:
                patch_set = self.read_rolling_memory_chunk(
                    client=client,
                    stage=stage,
                    paper=paper,
                    skim=skim,
                    decision=decision,
                    memory=memory,
                    artifacts=page_chunk,
                    chunk_index=chunk_index,
                    total_chunks=len(page_chunks),
                )
                memory = self.memory_store.apply_patch_set(
                    paper.paper_id,
                    ensure_read_pages_operation(patch_set, paper_id=paper.paper_id, pages=pages),
                    source=f"rolling_memory_chunk_{chunk_index + 1}",
                )
            except Exception as exc:
                if self.require_llm_success():
                    raise
                failed_chunk = {"pages": pages, "error": str(exc)}
                failed_chunks.append(failed_chunk)
                memory = self.memory_store.apply_patch_set(
                    paper.paper_id,
                    {
                        "paper_id": paper.paper_id,
                        "operations": [
                            {
                                "op": "add_partial_read_failure",
                                "payload": {
                                    "pages": pages,
                                    "error": compact_reason(str(exc), max_chars=280),
                                },
                            },
                            {
                                "op": "add_open_question",
                                "payload": {
                                    "text": "Some selected pages could not be read by the model in this run: "
                                    + ", ".join("p." + str(page) for page in pages)
                                    + "."
                                },
                            },
                        ],
                    },
                    source=f"rolling_memory_chunk_{chunk_index + 1}_failed",
                )
                self.write_agent_run(
                    {
                        "agent_run_id": f"rolling_memory_{paper.paper_id}_chunk_{chunk_index + 1}_failed",
                        "paper_id": paper.paper_id,
                        "stage": stage,
                        "provider_kind": self.config.provider.kind,
                        "model": self.config.provider.model,
                        "status": "FALLBACK",
                        "error": str(exc),
                        "pages": pages,
                    }
                )
                self.events.emit(
                    "agent_run_fallback",
                    stage=stage,
                    level="warning",
                    message=f"Rolling read chunk failed for {paper.paper_id}; keeping prior memory",
                    data={"paper_id": paper.paper_id, "pages": pages, "error": str(exc)},
                )
        _ = failed_chunks
        return paper_card_from_memory_v3(
            paper=paper,
            skim=skim,
            decision=decision,
            memory=memory,
            fallback=fallback,
        )

    def read_rolling_memory_chunk(
        self,
        *,
        client: JsonLlmClient,
        stage: str,
        paper: PaperRecord,
        skim: SkimCard,
        decision: ClassificationDecision,
        memory: dict[str, Any],
        artifacts: list[Any],
        chunk_index: int,
        total_chunks: int,
    ) -> dict[str, Any]:
        pages = [item.page_no for item in artifacts]
        runtime = PaperLensRuntime(artifacts=artifacts)
        agent_context = runtime.build_context_pack(
            stage="rolling_read",
            objective=(
                "Read this page chunk as one streaming step. Preserve continuity through "
                "PaperMemoryV3 and emit only durable MemoryPatch operations."
            ),
            paper_id=paper.paper_id,
            title=paper.canonical_title,
            classification=decision.class_label,
            memory=memory,
            focus_queries=[
                paper.canonical_title or paper.paper_id,
                skim.problem,
                skim.method_type,
                skim.evaluation_type,
            ],
            focus_pages=pages,
            read_artifacts=artifacts,
            output_contract={
                "type": "MemoryPatchSet",
                "rule": (
                    "Patch only facts worth keeping in long-term PaperMemory. If a claim needs "
                    "future verification, add it with lower confidence or an open question."
                ),
            },
            search_limit=3,
            page_text_limit=700,
        ).as_dict()
        key_payload = {
            "version": ROLLING_MEMORY_PROMPT_VERSION,
            "model": self.config.provider.model,
            "paper_hash": paper.file_hash,
            "pages": pages,
            "page_hashes": [hash_text(getattr(item, "text", "")) for item in artifacts],
            "previous_memory_hash": hash_json_payload(memory_v3_prompt_view(memory)),
            "agent_context_hash": hash_json_payload(agent_context),
        }
        cache_path = self.cache_path("rolling_memory", paper.paper_id, key_payload)
        cached = self.read_cache_payload(cache_path)
        if cached and isinstance(cached.get("data"), dict):
            self.events.emit(
                "cache_hit",
                stage=stage,
                message=f"Rolling memory cache hit for {paper.paper_id}",
                data={"paper_id": paper.paper_id, "pages": pages, "cache": str(cache_path)},
            )
            return normalize_memory_patch_set(cached["data"], paper_id=paper.paper_id)
        loop = AgentLoop(
            client=client,
            tools=PaperToolRegistry(
                runtime=runtime,
                paper_id=paper.paper_id,
                title=paper.canonical_title,
                memory=memory,
                layout_pages=artifacts,
            ),
            session_name="rolling_memory",
            objective=(
                "Read the current paper pages and update PaperMemory with durable MemoryPatch "
                "operations. Use tools if you need to inspect page text, figures, or current memory."
            ),
            final_schema_name="paperlens_memory_patch_set",
            final_schema=MEMORY_PATCH_SET_SCHEMA,
            stage=stage,
            paper_id=paper.paper_id,
            trace_path=self.data_dir / "agent_trace.jsonl",
            system_prompt=ROLLING_MEMORY_SYSTEM_PROMPT,
            control_check=self.control.require_not_cancelled,
            pause_check=self.control.wait_if_paused,
        )
        result = loop.run(
            initial_context={
                "paper_id": paper.paper_id,
                "title": paper.canonical_title or "unknown",
                "classification": decision.class_label,
                "pages": pages,
                "agent_context_pack": agent_context,
                "rolling_memory_prompt": build_rolling_memory_prompt(
                    paper=paper,
                    skim=skim,
                    decision=decision,
                    memory=memory,
                    agent_context=agent_context,
                    artifacts=artifacts,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                ),
            }
        )
        self.record_llm_usage(stage, result.usage)
        self.write_agent_run(
            {
                "agent_run_id": f"rolling_memory_{paper.paper_id}_{uuid.uuid4().hex[:8]}",
                "paper_id": paper.paper_id,
                "stage": stage,
                "provider_kind": self.config.provider.kind,
                "model": self.config.provider.model,
                "usage": result.usage,
                "request_ids": result.request_ids,
                "trace_events": len(result.trace),
                "status": "PASS",
            }
        )
        self.write_cache_payload(
            cache_path,
            {
                "key": key_payload,
                "data": result.final,
                "usage": result.usage,
                "request_ids": result.request_ids,
                "endpoint": "agent_loop",
            },
        )
        return normalize_memory_patch_set(result.final, paper_id=paper.paper_id)

    def central_verify_paper_memory(
        self,
        *,
        client: JsonLlmClient,
        stage: str,
        paper: PaperRecord,
        skim: SkimCard,
        decision: ClassificationDecision,
        memory: dict[str, Any],
        all_artifacts: list[Any],
        read_artifacts: list[Any],
    ) -> dict[str, Any]:
        current_memory = dict(memory)
        if current_memory.get("schema_version") == "paper_memory.v3" and not self.memory_store.read(
            paper.paper_id
        ):
            self.memory_store.write(current_memory)
        verification_artifacts = select_central_verification_pages(
            memory=current_memory,
            all_artifacts=all_artifacts,
            read_artifacts=read_artifacts,
        )
        try:
            patch_set = self.verify_paper_memory_once(
                client=client,
                stage=stage,
                paper=paper,
                skim=skim,
                decision=decision,
                memory=current_memory,
                artifacts=verification_artifacts,
            )
            return self.memory_store.apply_patch_set(
                paper.paper_id,
                ensure_memory_audit_operation(
                    patch_set,
                    paper_id=paper.paper_id,
                    phase="central_memory_verify",
                ),
                source="central_memory_verify",
            )
        except Exception as exc:
            if self.require_llm_success() and not paper_memory_has_recoverable_content(
                current_memory
            ):
                raise
            audit = fallback_memory_audit(reason=str(exc), phase="central_memory_verify")
            current_memory = apply_memory_audit_patch(
                self.memory_store, paper.paper_id, audit, source="central_memory_verify_failed"
            )
            self.write_agent_run(
                {
                    "agent_run_id": f"central_memory_verify_{paper.paper_id}_failed",
                    "paper_id": paper.paper_id,
                    "stage": stage,
                    "provider_kind": self.config.provider.kind,
                    "model": self.config.provider.model,
                    "status": "FALLBACK",
                    "error": str(exc),
                }
            )
            self.events.emit(
                "agent_run_fallback",
                stage=stage,
                level="warning",
                message=f"Memory verification failed for {paper.paper_id}; marking memory as weak but usable",
                data={"paper_id": paper.paper_id, "error": str(exc)},
            )
            return current_memory

    def verify_paper_memory_once(
        self,
        *,
        client: JsonLlmClient,
        stage: str,
        paper: PaperRecord,
        skim: SkimCard,
        decision: ClassificationDecision,
        memory: dict[str, Any],
        artifacts: list[Any],
    ) -> dict[str, Any]:
        pages = [item.page_no for item in artifacts]
        runtime = PaperLensRuntime(artifacts=artifacts)
        high_risk_claims = select_high_risk_memory_claims(memory)
        agent_context = runtime.build_context_pack(
            stage="central_memory_verify",
            objective=(
                "Verify the current paper memory once against the paper map and relevant original "
                "page evidence. Return one MemoryPatchSet that both fixes memory and records the "
                "audit boundary."
            ),
            paper_id=paper.paper_id,
            title=paper.canonical_title,
            classification=decision.class_label,
            memory=memory,
            focus_queries=[claim.get("text") for claim in high_risk_claims if claim.get("text")],
            focus_pages=pages,
            read_artifacts=artifacts,
            output_contract={
                "type": "MemoryPatchSet",
                "rule": (
                    "Return a single patch set. Add or link evidence, weaken unsupported claims, "
                    "add missing limitations/open questions, and include exactly one set_memory_audit operation."
                ),
            },
            search_limit=5,
            page_text_limit=1100,
        ).as_dict()
        key_payload = {
            "version": CENTRAL_MEMORY_VERIFY_PROMPT_VERSION,
            "model": self.config.provider.model,
            "paper_hash": paper.file_hash,
            "memory_hash": hash_json_payload(memory_without_audit(memory)),
            "pages": pages,
            "page_hashes": [hash_text(getattr(item, "text", "")) for item in artifacts],
            "high_risk_claims_hash": hash_json_payload(high_risk_claims),
            "agent_context_hash": hash_json_payload(agent_context),
        }
        cache_path = self.cache_path("central_memory_verify", paper.paper_id, key_payload)
        cached = self.read_cache_payload(cache_path)
        if cached and isinstance(cached.get("data"), dict):
            self.events.emit(
                "cache_hit",
                stage=stage,
                message=f"Memory verification cache hit for {paper.paper_id}",
                data={"paper_id": paper.paper_id, "pages": pages, "cache": str(cache_path)},
            )
            return normalize_memory_patch_set(cached["data"], paper_id=paper.paper_id)
        loop = AgentLoop(
            client=client,
            tools=PaperToolRegistry(
                runtime=runtime,
                paper_id=paper.paper_id,
                title=paper.canonical_title,
                memory=memory,
                layout_pages=artifacts,
            ),
            session_name="central_memory_verify",
            objective=(
                "Verify the current PaperMemory against paper-local evidence. Use paper tools "
                "until you can submit one MemoryPatchSet that repairs, weakens, links evidence, "
                "or records open boundaries."
            ),
            final_schema_name="paperlens_memory_patch_set",
            final_schema=MEMORY_PATCH_SET_SCHEMA,
            stage=stage,
            paper_id=paper.paper_id,
            trace_path=self.data_dir / "agent_trace.jsonl",
            system_prompt=CENTRAL_MEMORY_VERIFY_SYSTEM_PROMPT,
            control_check=self.control.require_not_cancelled,
            pause_check=self.control.wait_if_paused,
        )
        result = loop.run(
            initial_context={
                "paper_id": paper.paper_id,
                "title": paper.canonical_title or "unknown",
                "classification": decision.class_label,
                "high_risk_claims": high_risk_claims,
                "verification_pages": pages,
                "agent_context_pack": agent_context,
                "verify_prompt": build_central_memory_verify_prompt(
                    paper=paper,
                    skim=skim,
                    decision=decision,
                    memory=memory,
                    high_risk_claims=high_risk_claims,
                    agent_context=agent_context,
                    artifacts=artifacts,
                ),
            }
        )
        self.record_llm_usage(stage, result.usage)
        self.write_agent_run(
            {
                "agent_run_id": f"central_memory_verify_{paper.paper_id}_{uuid.uuid4().hex[:8]}",
                "paper_id": paper.paper_id,
                "stage": stage,
                "provider_kind": self.config.provider.kind,
                "model": self.config.provider.model,
                "usage": result.usage,
                "request_ids": result.request_ids,
                "trace_events": len(result.trace),
                "status": "PASS",
            }
        )
        patch_set = normalize_memory_patch_set(result.final, paper_id=paper.paper_id)
        self.write_cache_payload(
            cache_path,
            {
                "key": key_payload,
                "data": patch_set,
                "usage": result.usage,
                "request_ids": result.request_ids,
                "endpoint": "agent_loop",
            },
        )
        return patch_set

    def stage_08_evidence_verify(self) -> None:
        stage = "stage_08_evidence_verify"
        self.checkpoint(stage)
        self.events.stage_started(stage, "Verifying paper memory once against local evidence")
        core_v2_rows = self.refresh_core_v2_deterministic_audits(stage)
        client = self.new_llm_client() if self.llm_enabled() else None
        papers_by_id = {paper.paper_id: paper for paper in self.papers}
        skims_by_id = {skim.paper_id: skim for skim in self.skim_cards}
        decisions_by_id = {decision.paper_id: decision for decision in self.classifications}
        rows = []
        verified_cards: list[PaperCard] = []
        for card in self.paper_cards:
            paper = papers_by_id.get(card.paper_id)
            skim = skims_by_id.get(card.paper_id)
            decision = decisions_by_id.get(card.paper_id)
            memory = self.memory_store.read(card.paper_id)
            if client and paper and skim and decision and memory:
                artifacts = self.db.get_page_artifacts(card.paper_id)
                try:
                    memory = self.central_verify_paper_memory(
                        client=client,
                        stage=stage,
                        paper=paper,
                        skim=skim,
                        decision=decision,
                        memory=memory,
                        all_artifacts=artifacts,
                        read_artifacts=[
                            artifact
                            for artifact in artifacts
                            if getattr(artifact, "page_no", None) in memory_v3_pages_read(memory)
                        ],
                    )
                    card = paper_card_from_memory_v3(
                        paper=paper,
                        skim=skim,
                        decision=decision,
                        memory=memory,
                        fallback=card,
                    )
                except Exception as exc:
                    if self.require_llm_success():
                        raise
                    audit = fallback_memory_audit(reason=str(exc), phase="central_memory_verify")
                    apply_memory_audit_patch(
                        self.memory_store,
                        card.paper_id,
                        audit,
                        source="central_memory_verify_failed",
                    )
                    self.events.emit(
                        "agent_run_fallback",
                        stage=stage,
                        level="warning",
                        message=f"Memory verification failed for {card.paper_id}; keeping read memory",
                        data={"paper_id": card.paper_id, "error": str(exc)},
                    )
            status, notes = audit_paper_card_evidence(card)
            card.verification_status = status
            self.db.upsert_paper_card(card)
            verified_cards.append(card)
            side = []
            if status == "NEED_HUMAN_REVIEW":
                side.append("NEED_HUMAN_REVIEW")
                self.db.upsert_review_item(
                    ReviewItem(
                        item_id=f"evidence:{card.paper_id}",
                        paper_id=card.paper_id,
                        item_type="evidence",
                        priority=1,
                        reason=";".join(notes),
                        payload=card.model_dump(),
                    )
                )
            elif status == "PASS_WITH_WEAKNESSES":
                side.append("WEAK_EVIDENCE_BOUNDARY")
                self.db.upsert_review_item(
                    ReviewItem(
                        item_id=f"weak_evidence:{card.paper_id}",
                        paper_id=card.paper_id,
                        item_type="WEAK_EVIDENCE_BOUNDARY",
                        priority=2,
                        reason=";".join(notes),
                        payload=card.model_dump(),
                    )
                )
            self.mark_paper_state(card.paper_id, stage, side_statuses=side)
            rows.append({"paper_id": card.paper_id, "status": status, "notes": notes})
        self.paper_cards = verified_cards
        self.events.stage_completed(
            stage,
            "Evidence audit completed",
            {"paper_cards": len(rows), "core_v2_audits": len(core_v2_rows)},
        )

    def stage_15_export(self) -> list[Path]:
        stage = "stage_15_export"
        self.checkpoint(stage)
        self.events.stage_started(stage, "Writing final reading reports")
        client = None if self.config.offline_debug else self.new_llm_client()
        active_ids = {paper.paper_id for paper in self.papers}
        report_paths = write_final_report_bundle(
            output_dir=self.output_dir,
            data_dir=self.data_dir,
            evidence_dir=self.evidence_dir,
            client=client,
            record_usage=self.record_llm_usage,
            record_agent_run=self.write_agent_run,
            stage=stage,
            papers=self.papers,
            skim_cards=self.skim_cards,
            decisions=self.classifications,
            paper_cards=self.paper_cards,
            review_items=[
                item for item in self.db.list_review_items() if item.paper_id in active_ids
            ],
            budget=self.budget.public_dict(),
            budget_provider=self.budget.public_dict,
            config=self.config.public_dict(),
            topic=self.config.topic,
            idea=self.config.idea,
            cache_dir=self.cache_dir,
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
        write_core_quality_snapshot(self.output_dir)
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
                "paper_memory_v3": ".paperlens/data/memory/v3/",
                "core_v2": ".paperlens/data/core/v2/",
                "core_quality_snapshot": ".paperlens/data/core_quality_snapshot.v1.json",
                "library_index": ".paperlens/library/index/search_index.json",
                "data": ".paperlens/data/",
                "model_call_summary": ".paperlens/data/model_call_summary.json",
                "output_validation": output_validation,
            },
            "model_calls": model_call_summary,
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


SKIM_CLASSIFIER_PROMPT_VERSION = "skim-classifier-v2"


SKIM_CLASSIFIER_SYSTEM_PROMPT = """
You are the PaperLens SkimClassifier.
Grade a research paper's reading value using only the supplied parsed excerpts.

Class labels:
A = high value; keep it prominent and expect follow-up QA or opt-in close reading.
B = useful or plausibly relevant; keep in the standard library.
C = low value for the current goal; only choose when evidence is adequate.
HOLD = insufficient evidence, weak parse, visual uncertainty, or meaningful doubt.

Rules:
- Optimize for false-negative prevention: prefer B/HOLD over C when unsure.
- Judge value from novelty, relevance to user context, methodological quality, evidence strength, reusable ideas, and risk of missing an important paper.
- Do not invent facts that are not in the excerpts.
- Separate author claims from your inference.
- Return evidence_queries that can be found in the supplied text snippets.
- Return only JSON matching the requested schema.
""".strip()


SKIM_CLASSIFICATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["skim", "classification", "evidence_queries"],
    "properties": {
        "skim": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "problem",
                "method_type",
                "system_scope",
                "evaluation_type",
                "danger_signals",
                "confidence",
            ],
            "properties": {
                "problem": {"type": ["string", "null"]},
                "method_type": {"type": ["string", "null"]},
                "system_scope": {"type": ["string", "null"]},
                "evaluation_type": {"type": ["string", "null"]},
                "danger_signals": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
        },
        "classification": {
            "type": "object",
            "additionalProperties": False,
            "required": ["class_label", "confidence", "false_negative_risk", "reason_codes"],
            "properties": {
                "class_label": {"type": "string", "enum": ["A", "B", "C", "HOLD"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "false_negative_risk": {"type": "number", "minimum": 0, "maximum": 1},
                "reason_codes": {"type": "array", "items": {"type": "string"}},
            },
        },
        "evidence_queries": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["page_no", "keyword", "quote"],
                "properties": {
                    "page_no": {"type": ["integer", "null"], "minimum": 1},
                    "keyword": {"type": ["string", "null"]},
                    "quote": {"type": ["string", "null"]},
                },
            },
        },
    },
}


ROLLING_MEMORY_PROMPT_VERSION = "memory-patch-rolling-v2-context"
CENTRAL_MEMORY_VERIFY_PROMPT_VERSION = "central-memory-verify-v2-strict-audit"
REPORT_PLAN_PROMPT_VERSION = "report-plan-v4-complete-capsule-profile"
REPORT_SECTION_PROMPT_VERSION = "report-section-v6-depth-contract"
REPORT_SECTION_AUDIT_PROMPT_VERSION = "report-section-audit-v3-depth-and-boundary"


ROLLING_MEMORY_SYSTEM_PROMPT = """
You are the PaperLens RollingReader.
Read paper pages and improve PaperMemoryV3.
Use tools whenever they help you check text, figures, evidence, or current memory.
Return final_json as one MemoryPatchSet when the memory patch is good enough.
Keep durable paper claims separate from background concepts and open uncertainty.
""".strip()


CENTRAL_MEMORY_VERIFY_SYSTEM_PROMPT = """
You are the PaperLens MemoryVerifier.
Verify PaperMemoryV3 against local paper evidence.
Use tools until you can confidently repair memory, weaken unsupported claims, link evidence,
or record explicit uncertainty.
The memory audit status is authoritative: use NEED_HUMAN_REVIEW only when the capsule should
not be presented as reviewed; never mark NEED_HUMAN_REVIEW as safe_to_generate_capsule=true.
Return final_json as one MemoryPatchSet. Include a memory audit operation when done.
""".strip()


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


def build_skim_prompt(
    *,
    paper: PaperRecord,
    artifacts: list[Any],
    keyword_pool: list[str],
    topic: str | None,
    idea: str | None,
) -> str:
    excerpts = []
    used_chars = 0
    max_chars = 12000
    for page in artifacts[:6]:
        text = normalize_excerpt(page.text, limit=2400)
        if not text:
            continue
        captions = "; ".join(str(caption.get("text") or "") for caption in page.captions[:4])
        visual_flags = ", ".join(page.low_confidence_flags)
        block = (
            f"[page {page.page_no}]\n"
            f"flags: {visual_flags or 'none'}\n"
            f"captions: {captions or 'none'}\n"
            f"text:\n{text}"
        )
        if used_chars + len(block) > max_chars:
            break
        excerpts.append(block)
        used_chars += len(block)
    return "\n\n".join(
        [
            f"paper_id: {paper.paper_id}",
            f"title: {paper.canonical_title or 'unknown'}",
            f"parse_quality: {paper.parse_quality or 'unknown'}",
            "user topic: " + (topic or "not provided"),
            "user idea or claim draft: " + (idea or "not provided"),
            "value-signal keywords: " + ", ".join(keyword_pool),
            (
                "Classify reading value for an automated paper-reading workflow. "
                "A papers should be kept prominent for follow-up QA or opt-in close reading; B papers "
                "belong in the standard library; C papers are lower-priority but still receive the "
                "standard read loop; HOLD papers need review."
            ),
            "Parsed excerpts:",
            "\n\n---\n\n".join(excerpts) if excerpts else "No usable text excerpts.",
        ]
    )


def build_rolling_memory_prompt(
    *,
    paper: PaperRecord,
    skim: SkimCard,
    decision: ClassificationDecision,
    memory: dict[str, Any],
    agent_context: dict[str, Any] | None = None,
    artifacts: list[Any],
    chunk_index: int,
    total_chunks: int,
) -> str:
    page_blocks = []
    for page in artifacts:
        text = normalize_excerpt(page.text, limit=1200)
        if not text:
            continue
        captions = "; ".join(str(caption.get("text") or "") for caption in page.captions[:3])
        visual_notes = getattr(page, "visual_notes", [])[:3]
        page_blocks.append(
            "\n".join(
                [
                    f"[page {page.page_no}]",
                    f"captions: {captions or 'none'}",
                    "visual_notes: " + json.dumps(visual_notes, ensure_ascii=False),
                    f"text:\n{text}",
                ]
            )
        )
    return "\n\n".join(
        [
            f"paper_id: {paper.paper_id}",
            f"title: {paper.canonical_title or 'unknown'}",
            f"classification: {decision.class_label}",
            f"chunk: {chunk_index + 1}/{total_chunks}",
            "skim_card:",
            json.dumps(skim.model_dump(), ensure_ascii=False),
            "agent_context_pack:",
            context_pack_prompt(agent_context),
            "current_paper_memory_v3:",
            json.dumps(memory_v3_prompt_view(memory), ensure_ascii=False),
            "memory_patch_protocol:",
            memory_patch_protocol_text(),
            (
                "Task: return a MemoryPatchSet after reading the current pages. Use add_read_pages "
                "for these pages, then add/upsert only durable knowledge that improves the current "
                "PaperMemoryV3. If these pages change an existing understanding, patch the relevant "
                "field or claim instead of appending a duplicate."
            ),
            (
                "Coverage checklist: preserve enough material for a later Theseus-grade capsule. "
                "Capture the paper's problem frame, core abstraction, mechanism steps, concrete "
                "implementation components/equations/runtime or training path, evaluation setup "
                "and numbers, limitations, concepts a reader needs, and evidence links. Use "
                "upsert_implementation_component for concrete modules, data structures, losses, "
                "runtime components, or algorithm stages that should later become an implementation "
                "detail section."
            ),
            "current_pages:",
            "\n\n---\n\n".join(page_blocks) if page_blocks else "No usable text excerpts.",
        ]
    )


def build_central_memory_verify_prompt(
    *,
    paper: PaperRecord,
    skim: SkimCard,
    decision: ClassificationDecision,
    memory: dict[str, Any],
    high_risk_claims: list[dict[str, Any]],
    agent_context: dict[str, Any] | None = None,
    artifacts: list[Any],
) -> str:
    return "\n\n".join(
        [
            f"paper_id: {paper.paper_id}",
            f"title: {paper.canonical_title or 'unknown'}",
            f"classification: {decision.class_label}",
            "skim_card:",
            json.dumps(skim.model_dump(), ensure_ascii=False),
            "current_paper_memory_v3:",
            json.dumps(memory_v3_prompt_view(memory_without_audit(memory)), ensure_ascii=False),
            "high_risk_claims_to_verify:",
            json.dumps(high_risk_claims, ensure_ascii=False),
            "agent_context_pack:",
            context_pack_prompt(agent_context),
            "memory_patch_protocol:",
            memory_patch_protocol_text(),
            "verification_pages:",
            json.dumps(
                memory_page_excerpt_blocks(artifacts, limit_per_page=2400, max_chars=22000),
                ensure_ascii=False,
            ),
            (
                "Task: verify memory and return a MemoryPatchSet when ready. If a claim is "
                "supported, add/link evidence and mark it checked. If it is too strong, rewrite it "
                "with lower confidence or mark it disputed. If something important is missing, use "
                "tools or record the boundary in set_memory_audit. A complete memory should support "
                "separate capsule sections for orientation, mechanism, implementation path, evaluation, "
                "value/tradeoffs, and limitations when the paper contains that material."
            ),
        ]
    )


def memory_patch_protocol_text() -> str:
    return (
        "Return JSON: {paper_id, rationale, operations:[{op,payload}]}. Useful ops: "
        "add_read_pages {pages}; set_problem_frame {problem,why_it_matters,scope}; "
        "set_core_abstraction {text,evidence_refs,misunderstanding_guard}; "
        "set_mechanism_overview {overview}; upsert_mechanism_step {id?,text}; "
        "upsert_implementation_component {name?,component?,role?,text?,evidence_refs?}; "
        "set_evaluation_summary {summary}; upsert_evaluation_item {id?,text}; "
        "upsert_concept {term,explanation}; set_conceptual_bridge {needed,reader_gap,bridge_text}; "
        "upsert_conceptual_bridge_term {term,explanation,paper_role,provenance}; "
        "upsert_evidence {id?,source_type,page,section?,excerpt_or_caption?,interpretation,reliability}; "
        "upsert_claim {id?,text,type,provenance,confidence,evidence_refs,depends_on?,risk_tags?,critic_status}; "
        "link_claim_evidence {claim_id,evidence_refs}; add_limitation {text}; "
        "add_open_question {text}; set_memory_audit {status,unsupported_claims,missing_items,"
        "repair_instructions,safe_to_generate_capsule,confidence}. "
        "Prefer stable IDs when updating existing entries."
    )


def memory_page_excerpt_blocks(
    artifacts: list[Any],
    *,
    limit_per_page: int,
    max_chars: int,
) -> list[dict[str, Any]]:
    blocks = []
    used_chars = 0
    for page in artifacts:
        block = {
            "page_no": getattr(page, "page_no", None),
            "text": normalize_excerpt(str(getattr(page, "text", "") or ""), limit=limit_per_page),
            "captions": getattr(page, "captions", [])[:5],
            "figures": getattr(page, "figures", [])[:4],
            "tables": getattr(page, "tables", [])[:4],
            "visual_notes": getattr(page, "visual_notes", [])[:4],
            "low_confidence_flags": getattr(page, "low_confidence_flags", []),
        }
        block_text = json.dumps(block, ensure_ascii=False)
        if used_chars + len(block_text) > max_chars:
            break
        blocks.append(block)
        used_chars += len(block_text)
    return blocks


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


def llm_skim_classify_to_models(
    *,
    paper: PaperRecord,
    artifacts: list[Any],
    raw: dict[str, Any],
    agent_run_id: str,
    fallback_card: SkimCard,
    fallback_decision: ClassificationDecision,
) -> tuple[SkimCard, ClassificationDecision]:
    skim = raw.get("skim") if isinstance(raw.get("skim"), dict) else {}
    classification = (
        raw.get("classification") if isinstance(raw.get("classification"), dict) else {}
    )
    signals = normalized_string_list(skim.get("danger_signals"))
    refs = evidence_refs_from_llm(
        paper_id=paper.paper_id,
        artifacts=artifacts,
        queries=raw.get("evidence_queries"),
        fallback_signals=signals,
        agent_run_id=agent_run_id,
    )
    if not refs:
        refs = fallback_card.evidence_refs
    class_label = classification.get("class_label")
    if class_label not in {"A", "B", "C", "HOLD"}:
        class_label = fallback_decision.class_label
    card = SkimCard(
        paper_id=paper.paper_id,
        problem=string_or_none(skim.get("problem")) or fallback_card.problem,
        method_type=string_or_none(skim.get("method_type")) or fallback_card.method_type,
        system_scope=string_or_none(skim.get("system_scope")) or fallback_card.system_scope,
        evaluation_type=string_or_none(skim.get("evaluation_type"))
        or fallback_card.evaluation_type,
        danger_signals=signals or fallback_card.danger_signals,
        evidence_refs=refs,
        confidence=clamp_float(skim.get("confidence"), fallback_card.confidence),
    )
    decision = ClassificationDecision(
        paper_id=paper.paper_id,
        class_label=class_label,
        confidence=clamp_float(classification.get("confidence"), fallback_decision.confidence),
        false_negative_risk=clamp_float(
            classification.get("false_negative_risk"),
            fallback_decision.false_negative_risk,
        ),
        reason_codes=normalized_string_list(classification.get("reason_codes"))
        or fallback_decision.reason_codes,
        audit_status="LLM_PENDING_AUDIT",
    )
    if paper.parse_quality in {"OCR_REQUIRED", "VLM_PAGE_MODE"} and decision.class_label == "C":
        decision.class_label = "HOLD"
        decision.false_negative_risk = max(decision.false_negative_risk, 0.8)
        decision.reason_codes.append(f"{paper.parse_quality.lower()}_guardrail")
    if paper.parse_quality == "PASS_WITH_WEAKNESSES" and decision.class_label == "C":
        decision.class_label = "HOLD"
        decision.false_negative_risk = max(decision.false_negative_risk, 0.7)
        decision.reason_codes.append("weak_parse_c_guardrail")
    if decision.class_label == "C" and not card.evidence_refs:
        decision.false_negative_risk = max(decision.false_negative_risk, 0.55)
        decision.reason_codes.append("c_without_positive_evidence")
    if decision.class_label == "C" and card.danger_signals:
        decision.class_label = "B"
        decision.false_negative_risk = max(decision.false_negative_risk, 0.65)
        decision.reason_codes.append("keyword_anti_leak_guardrail")
    return card, decision


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


def tokenize_memory_query(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9_]{4,}|[\u4e00-\u9fff]{2,}", normalize_for_search(text))
        if token not in {"paper", "memory", "claim", "claims", "missing", "needs", "need"}
    ][:12]


def merge_string_lists(left: list[str], right: list[str], *, limit: int) -> list[str]:
    merged = []
    for item in left + right:
        cleaned = item.strip()
        if cleaned and cleaned not in merged:
            merged.append(cleaned)
        if len(merged) >= limit:
            break
    return merged


def hash_json_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def llm_cache_path(
    cache_dir: Path | None, stage: str, paper_id: str, key_payload: dict[str, Any]
) -> Path | None:
    if cache_dir is None:
        return None
    key = hashlib.sha256(
        json.dumps(key_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]
    safe_stage = re.sub(r"[^a-zA-Z0-9_.-]+", "_", stage)
    safe_paper = re.sub(r"[^a-zA-Z0-9_.-]+", "_", paper_id)
    return cache_dir / safe_stage / safe_paper / f"{key}.json"


def read_llm_cache(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_llm_cache(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def hash_file_bytes(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "missing"


def deterministic_paper_card(
    paper: PaperRecord,
    skim: SkimCard,
    decision: ClassificationDecision,
) -> PaperCard:
    contribution_claims = []
    if skim.problem:
        contribution_claims.append(skim.problem)
    mechanisms = [
        value for value in [skim.method_type, skim.system_scope] if value and value != "unknown"
    ]
    evaluation = (
        [skim.evaluation_type] if skim.evaluation_type and skim.evaluation_type != "unknown" else []
    )
    limitations = [
        "Generated from skim-level verified excerpts; full PaperMemory read not yet performed.",
    ]
    if decision.class_label == "B":
        limitations.append(
            "B-class paper may need follow-up QA verification before strong value or citation claims."
        )
    return PaperCard(
        paper_id=paper.paper_id,
        contribution_claims=contribution_claims or [paper.canonical_title or paper.paper_id],
        mechanisms=mechanisms,
        assumptions=[],
        evaluation=evaluation,
        limitations=limitations,
        relation_to_user_work=(
            f"{decision.class_label}-class paper-value candidate with "
            f"{', '.join(skim.danger_signals) if skim.danger_signals else 'no explicit danger signal'}."
        ),
        evidence_refs=skim.evidence_refs,
        verification_status="PENDING_EVIDENCE_AUDIT",
    )


def should_run_normal_read(decision: ClassificationDecision) -> bool:
    _ = decision
    return True


def paper_card_from_memory_v3(
    *,
    paper: PaperRecord,
    skim: SkimCard,
    decision: ClassificationDecision,
    memory: dict[str, Any],
    fallback: PaperCard,
) -> PaperCard:
    claims = [
        item.get("text", "")
        for item in list_payload(memory.get("claims"))
        if isinstance(item, dict) and string_or_none(item.get("text"))
    ]
    core_abstractions = [
        item.get("text", "")
        for item in list_payload(memory.get("core_abstractions"))
        if isinstance(item, dict) and string_or_none(item.get("text"))
    ]
    contribution_claims = core_abstractions[:1] + claims[:5]
    if not contribution_claims:
        contribution_claims = compact_string_list(
            [
                dict_value(memory.get("problem_frame")).get("problem"),
                fallback.contribution_claims[0] if fallback.contribution_claims else None,
            ],
            limit=6,
            max_chars=220,
        )

    mechanism = dict_value(memory.get("mechanism"))
    mechanisms = normalized_string_list(
        [mechanism.get("overview")]
        + [
            item.get("text")
            for item in list_payload(mechanism.get("steps"))
            if isinstance(item, dict)
        ]
    )
    if not mechanisms:
        mechanisms = fallback.mechanisms

    evaluation_block = dict_value(memory.get("evaluation"))
    evaluation = normalized_string_list(
        [evaluation_block.get("summary")]
        + [
            item.get("text")
            for item in list_payload(evaluation_block.get("items"))
            if isinstance(item, dict)
        ]
    )
    if not evaluation:
        evaluation = fallback.evaluation

    limitations = normalized_string_list(memory.get("limitations")) or fallback.limitations
    audit = dict_value(dict_value(memory.get("audit_trail")).get("memory_audit"))
    for item in normalized_string_list(audit.get("missing_items") if audit else [])[:3]:
        limitations.append(f"Memory audit still wants caution on: {item}")
    limitations = list(dict.fromkeys(limitations))[:6]

    evidence_refs = evidence_refs_from_memory_v3(
        paper_id=paper.paper_id,
        skim=skim,
        memory=memory,
        fallback=fallback.evidence_refs,
    )
    verification_status = memory_backed_card_status(audit, evidence_refs)
    return PaperCard(
        paper_id=paper.paper_id,
        contribution_claims=contribution_claims[:6],
        mechanisms=mechanisms[:6],
        assumptions=fallback.assumptions[:4],
        evaluation=evaluation[:6],
        limitations=limitations,
        relation_to_user_work=(
            f"{decision.class_label}-class paper; PaperCard is derived from PaperMemoryV3."
        ),
        evidence_refs=evidence_refs,
        verification_status=verification_status,
    )


def evidence_refs_from_memory_v3(
    *,
    paper_id: str,
    skim: SkimCard,
    memory: dict[str, Any],
    fallback: list[EvidenceRef],
) -> list[EvidenceRef]:
    refs = list(skim.evidence_refs or fallback)
    seen_pages = {ref.page_no for ref in refs}
    page_numbers: list[int] = []
    evidence_id_to_page = {}
    for item in list_payload(memory.get("evidence")):
        if isinstance(item, dict):
            page_no = safe_int(item.get("page"))
            if page_no and page_no > 0:
                page_numbers.append(page_no)
                if item.get("id"):
                    evidence_id_to_page[str(item.get("id"))] = page_no
    for claim in list_payload(memory.get("claims")):
        if not isinstance(claim, dict):
            continue
        for ref in normalized_string_list(claim.get("evidence_refs")):
            page_no = safe_int(ref) or evidence_id_to_page.get(str(ref))
            if page_no and page_no > 0:
                page_numbers.append(page_no)
    for page_no in page_numbers:
        if page_no in seen_pages:
            continue
        refs.append(
            EvidenceRef(
                paper_id=paper_id,
                page_no=page_no,
                verification_status="KEYWORD_VERIFIED",
            )
        )
        seen_pages.add(page_no)
        if len(refs) >= 8:
            break
    return dedupe_evidence_refs(refs)[:8]


def memory_backed_card_status(audit: dict[str, Any], evidence_refs: list[EvidenceRef]) -> str:
    if not evidence_refs:
        return "NEED_HUMAN_REVIEW"
    status = str(audit.get("status") or "PASS_WITH_WEAKNESSES")
    if status == "PASS":
        return "PASS"
    if status == "PASS_WITH_WEAKNESSES":
        return "PASS_WITH_WEAKNESSES"
    return "NEED_HUMAN_REVIEW"


def audit_paper_card_evidence(card: PaperCard) -> tuple[str, list[str]]:
    hard_notes = []
    weak_notes = []
    if not card.evidence_refs:
        return "NEED_HUMAN_REVIEW", ["missing_evidence_refs"]
    weak = False
    for ref in card.evidence_refs:
        if ref.page_no <= 0:
            hard_notes.append("invalid_page_no")
        if not ref.verification_status or ref.verification_status == "UNVERIFIED":
            hard_notes.append("unverified_ref")
        if ref.verification_status in {"WEAK_VERIFIED", "KEYWORD_VERIFIED"}:
            weak = True
        if ref.bbox is None:
            weak = True
            weak_notes.append("missing_bbox")
    if hard_notes:
        return "NEED_HUMAN_REVIEW", sorted(set(hard_notes + weak_notes))
    if weak:
        return "PASS_WITH_WEAKNESSES", sorted(set(weak_notes or ["skim_level_or_keyword_evidence"]))
    return "PASS", ["text_match_verified"]


def combine_verification_status(left: str, right: str) -> str:
    rank = {"PASS": 0, "PASS_WITH_WEAKNESSES": 1, "NEED_HUMAN_REVIEW": 2}
    return left if rank.get(left, 2) >= rank.get(right, 2) else right


def clean_audit_notes(notes: list[str]) -> list[str]:
    hidden = {"skim_level_or_keyword_evidence", "visual_required_pages"}
    cleaned = []
    for note in notes:
        parts = [part.strip() for part in str(note).split(";") if part.strip()]
        visible = [part for part in parts if part not in hidden]
        if visible:
            cleaned.append("; ".join(visible))
    return cleaned


def evidence_refs_from_llm(
    *,
    paper_id: str,
    artifacts: list[Any],
    queries: Any,
    fallback_signals: list[str],
    agent_run_id: str,
) -> list[EvidenceRef]:
    refs: list[EvidenceRef] = []
    if isinstance(queries, list):
        for index, query in enumerate(queries[:8], start=1):
            if not isinstance(query, dict):
                continue
            quote = string_or_none(query.get("quote"))
            keyword = string_or_none(query.get("keyword"))
            page_no = query.get("page_no") if isinstance(query.get("page_no"), int) else None
            ref = find_evidence_ref(
                paper_id=paper_id,
                artifacts=artifacts,
                quote=quote,
                keyword=keyword,
                page_no=page_no,
                text_span_id=f"llm_query_{index}",
                agent_run_id=agent_run_id,
            )
            if ref:
                refs.append(ref)
    for signal in fallback_signals:
        if len(refs) >= 3:
            break
        ref = find_evidence_ref(
            paper_id=paper_id,
            artifacts=artifacts,
            quote=None,
            keyword=signal,
            page_no=None,
            text_span_id=f"keyword_{slug(signal)}",
            agent_run_id=agent_run_id,
        )
        if ref and not any(
            (ref.page_no, ref.text_span_id) == (old.page_no, old.text_span_id) for old in refs
        ):
            refs.append(ref)
    return refs


def find_evidence_ref(
    *,
    paper_id: str,
    artifacts: list[Any],
    quote: str | None,
    keyword: str | None,
    page_no: int | None,
    text_span_id: str,
    agent_run_id: str,
) -> EvidenceRef | None:
    search_pages = artifacts
    if page_no is not None:
        preferred = [page for page in artifacts if page.page_no == page_no]
        search_pages = preferred + [page for page in artifacts if page.page_no != page_no]
    for page in search_pages:
        text = page.text or ""
        if quote and normalize_for_search(quote) not in normalize_for_search(text):
            continue
        if not quote and keyword and keyword.lower() not in text.lower():
            continue
        bbox = first_matching_bbox(page.words, quote or keyword)
        return EvidenceRef(
            paper_id=paper_id,
            page_no=page.page_no,
            text_span_id=text_span_id,
            bbox=bbox,
            quote_hash=hash_text(quote) if quote else None,
            agent_run_id=agent_run_id,
            verification_status="TEXT_MATCH_VERIFIED" if quote else "KEYWORD_VERIFIED",
        )
    return None


def normalize_excerpt(text: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rsplit(" ", 1)[0] + " ..."


def normalize_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def normalized_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if isinstance(item, str):
            cleaned = re.sub(r"\s+", " ", item).strip()
            if cleaned:
                result.append(cleaned)
    return result[:20]


def string_or_none(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned or None


def clamp_float(value: Any, fallback: float) -> float:
    if isinstance(value, bool):
        return fallback
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback
    return max(0.0, min(1.0, numeric))


def hash_text(value: str) -> str:
    return hashlib.sha256(normalize_for_search(value).encode("utf-8")).hexdigest()[:16]


def slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return cleaned[:48] or "evidence"


def first_matching_bbox(words: list[dict[str, Any]], query: str | None) -> list[float] | None:
    if not words:
        return None
    if not query:
        return words[0].get("bbox")
    tokens = re.findall(r"[A-Za-z0-9_+-]+", query.lower())
    candidates = tokens[:4] or [query.lower()]
    for word in words:
        text = str(word.get("text") or "").strip().lower()
        if any(candidate and candidate in text for candidate in candidates):
            bbox = word.get("bbox")
            return bbox if isinstance(bbox, list) else None
    return words[0].get("bbox")


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


REPORT_PLAN_SYSTEM_PROMPT = """
You are the PaperLens ReportPlanner skill.
Plan a clear knowledge capsule from PaperMemoryV3.
Choose the reading order that best explains this paper. Use tools if you need more grounding.
The report is a derived view; do not invent facts outside memory/evidence.
Plan for a Theseus-grade capsule: complete, sectioned, and detailed enough to teach the paper.
For normal A/B/C papers, cover orientation/background, core mechanism, implementation or
algorithm path, evaluation evidence, value/tradeoffs, and limitations/boundaries. Do not merge
evaluation, value, and limitations into one thin section when the memory has enough material.
Return final_json matching the ReportPlan schema.
""".strip()


REPORT_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "paper_id",
        "grade",
        "read_recommendation",
        "one_line_reason",
        "core_takeaway",
        "sections",
    ],
    "properties": {
        "paper_id": {"type": "string"},
        "grade": {"type": "string", "enum": ["A", "B", "C", "HOLD"]},
        "read_recommendation": {
            "type": "string",
            "enum": ["重点关注", "标准读", "低优先级", "需确认"],
        },
        "one_line_reason": {"type": "string"},
        "core_takeaway": {"type": "string"},
        "sections": {
            "type": "array",
            "minItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "section_id",
                    "title",
                    "purpose",
                    "focus_queries",
                    "claim_ids",
                    "evidence_refs",
                    "target_pages",
                ],
                "properties": {
                    "section_id": {"type": "string"},
                    "section_kind": {
                        "type": "string",
                        "enum": [
                            "orientation",
                            "background",
                            "mechanism",
                            "evaluation",
                            "value",
                            "limits",
                            "other",
                        ],
                    },
                    "title": {"type": "string"},
                    "purpose": {"type": "string"},
                    "focus_queries": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "claim_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "evidence_refs": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "target_pages": {
                        "type": "array",
                        "items": {"type": "integer"},
                    },
                    "detail_questions": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "avoid": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
        },
        "key_visual_pages": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["page_no", "reason"],
                "properties": {
                    "page_no": {"type": "integer"},
                    "reason": {"type": "string"},
                },
            },
        },
        "uncertainty_note": {"type": "string"},
    },
}


REPORT_SECTION_SYSTEM_PROMPT = """
You are the PaperLens ReportComposer skill.
Write the requested report section as connected prose.
Use PaperMemory and tools for grounding. Explain mechanisms and background when useful.
Keep paper claims, interpretation, background knowledge, and evidence limits distinguishable.
Write at Theseus-grade depth: a section should explain why the idea exists, how the mechanism
works, what evidence supports it, and what boundary limits it when those are relevant to the
planned section. Prefer 2-4 compact but substantive paragraphs over a one-paragraph summary.
Return final_json matching the ReportSection schema.
""".strip()


REPORT_SECTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "section_id",
        "title",
        "paragraphs",
        "used_claim_ids",
        "used_evidence_refs",
        "uncertainty_note",
    ],
    "properties": {
        "section_id": {"type": "string"},
        "title": {"type": "string"},
        "paragraphs": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string"},
        },
        "markdown": {"type": "string"},
        "used_claim_ids": {
            "type": "array",
            "items": {"type": "string"},
        },
        "used_evidence_refs": {
            "type": "array",
            "items": {"type": "string"},
        },
        "uncertainty_note": {"type": "string"},
    },
}


REPORT_SECTION_AUDITOR_SYSTEM_PROMPT = """
You are the PaperLens SectionAuditor hook.
Audit one generated section against PaperMemory and paper evidence.
Use tools when a claim needs checking. Prefer explicit evidence boundaries over brittle certainty.
Mark REPAIR when a section is factually unsupported, overclaims, or is too shallow to satisfy
its planned purpose. Missing reader-critical mechanism, evaluation, or limitation context is a
real quality defect, not just a style preference.
Return final_json matching the ReportSectionAudit schema.
""".strip()


REPORT_SECTION_AUDIT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "verdict",
        "unsupported_items",
        "missing_items",
        "repair_instructions",
        "safe_usage_note",
    ],
    "properties": {
        "verdict": {"type": "string", "enum": ["PASS", "PASS_WITH_WEAKNESSES", "REPAIR"]},
        "unsupported_items": {"type": "array", "items": {"type": "string"}},
        "missing_items": {"type": "array", "items": {"type": "string"}},
        "repair_instructions": {"type": "array", "items": {"type": "string"}},
        "safe_usage_note": {"type": "string"},
    },
}


def compose_agentic_paper_report(
    *,
    client: JsonLlmClient,
    data_dir: Path,
    stage: str,
    paper: PaperRecord,
    skim: SkimCard | None,
    decision: ClassificationDecision | None,
    card: PaperCard | None,
    paper_memory: dict[str, Any] | None,
    layout: dict[str, Any],
    topic: str | None,
    idea: str | None,
    output_language: str,
    record_usage: Any,
    record_agent_run: Any,
    read_mode: str = "standard",
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    memory = dict_value(paper_memory)
    plan = generate_report_plan(
        client=client,
        data_dir=data_dir,
        stage=stage,
        paper=paper,
        skim=skim,
        decision=decision,
        card=card,
        paper_memory=memory,
        layout=layout,
        topic=topic,
        idea=idea,
        output_language=output_language,
        record_usage=record_usage,
        record_agent_run=record_agent_run,
        read_mode=read_mode,
        cache_dir=cache_dir,
    )
    sections: list[dict[str, Any]] = []
    section_audits: list[dict[str, Any]] = []
    previous_summaries: list[dict[str, str]] = []
    for raw_section_plan in list_payload(plan.get("sections")):
        if not isinstance(raw_section_plan, dict):
            continue
        section_plan = normalize_report_section_plan(raw_section_plan)
        section = generate_report_section(
            client=client,
            data_dir=data_dir,
            stage=stage,
            paper=paper,
            paper_memory=memory,
            layout=layout,
            plan=plan,
            section_plan=section_plan,
            previous_summaries=previous_summaries,
            output_language=output_language,
            record_usage=record_usage,
            record_agent_run=record_agent_run,
            read_mode=read_mode,
            cache_dir=cache_dir,
        )
        audit = audit_report_section(
            client=client,
            data_dir=data_dir,
            stage=stage,
            paper=paper,
            paper_memory=memory,
            layout=layout,
            plan=plan,
            section_plan=section_plan,
            section=section,
            output_language=output_language,
            record_usage=record_usage,
            record_agent_run=record_agent_run,
            read_mode=read_mode,
            cache_dir=cache_dir,
        )
        if audit.get("verdict") == "REPAIR":
            repaired_section = generate_report_section(
                client=client,
                data_dir=data_dir,
                stage=stage,
                paper=paper,
                paper_memory=memory,
                layout=layout,
                plan=plan,
                section_plan=section_plan,
                previous_summaries=previous_summaries,
                output_language=output_language,
                record_usage=record_usage,
                record_agent_run=record_agent_run,
                read_mode=read_mode,
                cache_dir=cache_dir,
                section_audit=audit,
            )
            repaired_audit = audit_report_section(
                client=client,
                data_dir=data_dir,
                stage=stage,
                paper=paper,
                paper_memory=memory,
                layout=layout,
                plan=plan,
                section_plan=section_plan,
                section=repaired_section,
                output_language=output_language,
                record_usage=record_usage,
                record_agent_run=record_agent_run,
                read_mode=read_mode,
                cache_dir=cache_dir,
            )
            if repaired_audit.get("verdict") != "REPAIR" or report_section_is_more_substantive(
                repaired_section, section
            ):
                section = repaired_section
                audit = repaired_audit
        sections.append(section)
        section_audits.append({"section_id": section.get("section_id"), **audit})
        previous_summaries.append(
            {
                "section_id": str(section.get("section_id") or section_plan.get("section_id")),
                "title": str(section.get("title") or section_plan.get("title")),
                "summary": compact_reason(
                    clean_model_inline_text(section.get("markdown")), max_chars=260
                ),
            }
        )
    report = assemble_agentic_report(
        paper=paper,
        decision=decision,
        plan=plan,
        sections=sections,
        section_audits=section_audits,
        output_language=output_language,
    )
    report_audit = aggregate_section_audits(section_audits)
    return report, report_audit


def generate_report_plan(
    *,
    client: JsonLlmClient,
    data_dir: Path,
    stage: str,
    paper: PaperRecord,
    skim: SkimCard | None,
    decision: ClassificationDecision | None,
    card: PaperCard | None,
    paper_memory: dict[str, Any],
    layout: dict[str, Any],
    topic: str | None,
    idea: str | None,
    output_language: str,
    record_usage: Any,
    record_agent_run: Any,
    read_mode: str,
    cache_dir: Path | None,
) -> dict[str, Any]:
    user_prompt = build_report_plan_prompt(
        paper=paper,
        skim=skim,
        decision=decision,
        card=card,
        paper_memory=paper_memory,
        layout=layout,
        topic=topic,
        idea=idea,
        output_language=output_language,
        read_mode=read_mode,
    )
    key_payload = {
        "version": REPORT_PLAN_PROMPT_VERSION,
        "model": client.config.model,
        "output_language": output_language,
        "read_mode": read_mode,
        "paper_hash": paper.file_hash,
        "prompt_hash": hash_text(REPORT_PLAN_SYSTEM_PROMPT + "\n" + user_prompt),
        "schema_hash": hash_json_payload(REPORT_PLAN_SCHEMA),
    }
    cache_path = llm_cache_path(cache_dir, "report_plans", paper.paper_id, key_payload)
    cached = read_llm_cache(cache_path)
    if cached and isinstance(cached.get("data"), dict):
        record_agent_run(cache_agent_run(client, paper.paper_id, stage, "report_plan", cache_path))
        return normalize_report_plan(
            cached["data"], paper=paper, decision=decision, paper_memory=paper_memory
        )
    with llm_call_context(
        stage=stage,
        paper_id=paper.paper_id,
        operation="report_plan",
        schema_name="paperlens_report_plan",
    ):
        result = AgentLoop(
            client=client,
            tools=PaperToolRegistry(
                runtime=PaperLensRuntime(artifacts=list_payload(layout.get("pages"))),
                paper_id=paper.paper_id,
                title=paper.canonical_title,
                memory=paper_memory,
                layout_pages=list_payload(layout.get("pages")),
            ),
            session_name="report_plan",
            objective="Plan a natural PaperLens knowledge capsule from PaperMemory. Use tools if the plan needs grounding.",
            final_schema_name="paperlens_report_plan",
            final_schema=REPORT_PLAN_SCHEMA,
            stage=stage,
            paper_id=paper.paper_id,
            trace_path=data_dir / "agent_trace.jsonl",
            system_prompt=REPORT_PLAN_SYSTEM_PROMPT,
        ).run(
            initial_context={
                "paper_id": paper.paper_id,
                "title": paper.canonical_title or "unknown",
                "output_language": output_language,
                "read_mode": read_mode,
                "topic": topic,
                "idea": idea,
                "skim_card": compact_skim_for_report(skim),
                "classification": compact_decision_for_report(decision),
                "paper_card": compact_paper_card_for_report(card),
                "paper_memory": compact_paper_memory_for_report(paper_memory),
                "context_prompt": user_prompt,
            }
        )
    record_usage(stage, result.usage)
    record_agent_run(
        {
            "agent_run_id": f"report_plan_{paper.paper_id}_{uuid.uuid4().hex[:8]}",
            "paper_id": paper.paper_id,
            "stage": stage,
            "operation": "report_plan",
            "provider_kind": client.config.kind,
            "model": client.config.model,
            "usage": result.usage,
            "request_ids": result.request_ids,
            "trace_events": len(result.trace),
            "status": "PASS",
        }
    )
    write_llm_cache(
        cache_path,
        {
            "key": key_payload,
            "data": result.final,
            "usage": result.usage,
            "request_ids": result.request_ids,
            "endpoint": "agent_loop",
        },
    )
    return normalize_report_plan(
        result.final, paper=paper, decision=decision, paper_memory=paper_memory
    )


def generate_report_section(
    *,
    client: JsonLlmClient,
    data_dir: Path,
    stage: str,
    paper: PaperRecord,
    paper_memory: dict[str, Any],
    layout: dict[str, Any],
    plan: dict[str, Any],
    section_plan: dict[str, Any],
    previous_summaries: list[dict[str, str]],
    output_language: str,
    record_usage: Any,
    record_agent_run: Any,
    read_mode: str,
    cache_dir: Path | None,
    section_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    user_prompt = build_report_section_prompt(
        paper=paper,
        paper_memory=paper_memory,
        layout=layout,
        plan=plan,
        section_plan=section_plan,
        previous_summaries=previous_summaries,
        output_language=output_language,
        read_mode=read_mode,
        section_audit=section_audit,
    )
    section_id = str(section_plan.get("section_id") or "section")
    key_payload = {
        "version": REPORT_SECTION_PROMPT_VERSION,
        "model": client.config.model,
        "output_language": output_language,
        "read_mode": read_mode,
        "paper_hash": paper.file_hash,
        "section_id": section_id,
        "plan_hash": hash_json_payload(plan),
        "previous_hash": hash_json_payload(previous_summaries),
        "audit_hash": hash_json_payload(section_audit or {}),
        "prompt_hash": hash_text(REPORT_SECTION_SYSTEM_PROMPT + "\n" + user_prompt),
        "schema_hash": hash_json_payload(REPORT_SECTION_SCHEMA),
    }
    cache_path = llm_cache_path(cache_dir, "report_sections", paper.paper_id, key_payload)
    cached = read_llm_cache(cache_path)
    if cached and isinstance(cached.get("data"), dict):
        record_agent_run(
            cache_agent_run(
                client, paper.paper_id, stage, f"report_section_{section_id}", cache_path
            )
        )
        return normalize_report_section(cached["data"], section_plan=section_plan)
    with llm_call_context(
        stage=stage,
        paper_id=paper.paper_id,
        operation="report_section",
        section_id=section_id,
        schema_name="paperlens_report_section",
    ):
        result = AgentLoop(
            client=client,
            tools=PaperToolRegistry(
                runtime=PaperLensRuntime(artifacts=list_payload(layout.get("pages"))),
                paper_id=paper.paper_id,
                title=paper.canonical_title,
                memory=paper_memory,
                layout_pages=list_payload(layout.get("pages")),
            ),
            session_name=f"report_section_{section_id}",
            objective="Write this report section as a clear article fragment from PaperMemory. Use tools if evidence or wording needs grounding.",
            final_schema_name="paperlens_report_section",
            final_schema=REPORT_SECTION_SCHEMA,
            stage=stage,
            paper_id=paper.paper_id,
            trace_path=data_dir / "agent_trace.jsonl",
            system_prompt=REPORT_SECTION_SYSTEM_PROMPT,
        ).run(
            initial_context={
                "paper_id": paper.paper_id,
                "title": paper.canonical_title or "unknown",
                "output_language": output_language,
                "read_mode": read_mode,
                "paper_memory": compact_paper_memory_for_report(paper_memory),
                "report_plan": compact_report_plan(plan),
                "section_to_write": section_plan,
                "previous_section_summaries": previous_summaries[-4:],
                "previous_section_audit": section_audit or {},
                "context_prompt": user_prompt,
            }
        )
    record_usage(stage, result.usage)
    record_agent_run(
        {
            "agent_run_id": f"report_section_{paper.paper_id}_{section_id}_{uuid.uuid4().hex[:8]}",
            "paper_id": paper.paper_id,
            "stage": stage,
            "operation": f"report_section_{section_id}",
            "provider_kind": client.config.kind,
            "model": client.config.model,
            "usage": result.usage,
            "request_ids": result.request_ids,
            "trace_events": len(result.trace),
            "status": "PASS",
        }
    )
    write_llm_cache(
        cache_path,
        {
            "key": key_payload,
            "data": result.final,
            "usage": result.usage,
            "request_ids": result.request_ids,
            "endpoint": "agent_loop",
        },
    )
    return normalize_report_section(result.final, section_plan=section_plan)


def audit_report_section(
    *,
    client: JsonLlmClient,
    data_dir: Path,
    stage: str,
    paper: PaperRecord,
    paper_memory: dict[str, Any],
    layout: dict[str, Any],
    plan: dict[str, Any],
    section_plan: dict[str, Any],
    section: dict[str, Any],
    output_language: str,
    record_usage: Any,
    record_agent_run: Any,
    read_mode: str,
    cache_dir: Path | None,
) -> dict[str, Any]:
    user_prompt = build_report_section_audit_prompt(
        paper=paper,
        paper_memory=paper_memory,
        layout=layout,
        plan=plan,
        section_plan=section_plan,
        section=section,
        output_language=output_language,
        read_mode=read_mode,
    )
    section_id = str(section_plan.get("section_id") or section.get("section_id") or "section")
    key_payload = {
        "version": REPORT_SECTION_AUDIT_PROMPT_VERSION,
        "model": client.config.model,
        "output_language": output_language,
        "read_mode": read_mode,
        "paper_hash": paper.file_hash,
        "section_id": section_id,
        "section_hash": hash_json_payload(section),
        "plan_hash": hash_json_payload(plan),
        "prompt_hash": hash_text(REPORT_SECTION_AUDITOR_SYSTEM_PROMPT + "\n" + user_prompt),
        "schema_hash": hash_json_payload(REPORT_SECTION_AUDIT_SCHEMA),
    }
    cache_path = llm_cache_path(cache_dir, "report_section_audits", paper.paper_id, key_payload)
    cached = read_llm_cache(cache_path)
    if cached and isinstance(cached.get("data"), dict):
        record_agent_run(
            cache_agent_run(
                client, paper.paper_id, stage, f"report_section_audit_{section_id}", cache_path
            )
        )
        return enforce_section_depth_audit(
            normalize_report_section_audit(cached["data"]),
            section=section,
            section_plan=section_plan,
        )
    with llm_call_context(
        stage=stage,
        paper_id=paper.paper_id,
        operation="report_section_audit",
        section_id=section_id,
        schema_name="paperlens_report_section_audit",
    ):
        result = AgentLoop(
            client=client,
            tools=PaperToolRegistry(
                runtime=PaperLensRuntime(artifacts=list_payload(layout.get("pages"))),
                paper_id=paper.paper_id,
                title=paper.canonical_title,
                memory=paper_memory,
                layout_pages=list_payload(layout.get("pages")),
            ),
            session_name=f"report_section_audit_{section_id}",
            objective="Audit one report section against PaperMemory and paper evidence. Use tools when a claim needs checking.",
            final_schema_name="paperlens_report_section_audit",
            final_schema=REPORT_SECTION_AUDIT_SCHEMA,
            stage=stage,
            paper_id=paper.paper_id,
            trace_path=data_dir / "agent_trace.jsonl",
            system_prompt=REPORT_SECTION_AUDITOR_SYSTEM_PROMPT,
        ).run(
            initial_context={
                "paper_id": paper.paper_id,
                "title": paper.canonical_title or "unknown",
                "output_language": output_language,
                "read_mode": read_mode,
                "paper_memory": compact_paper_memory_for_report(paper_memory),
                "report_plan": compact_report_plan(plan),
                "section_plan": section_plan,
                "generated_section": section,
                "context_prompt": user_prompt,
            }
        )
    record_usage(stage, result.usage)
    record_agent_run(
        {
            "agent_run_id": f"report_section_audit_{paper.paper_id}_{section_id}_{uuid.uuid4().hex[:8]}",
            "paper_id": paper.paper_id,
            "stage": stage,
            "operation": f"report_section_audit_{section_id}",
            "provider_kind": client.config.kind,
            "model": client.config.model,
            "usage": result.usage,
            "request_ids": result.request_ids,
            "trace_events": len(result.trace),
            "status": "PASS",
        }
    )
    write_llm_cache(
        cache_path,
        {
            "key": key_payload,
            "data": result.final,
            "usage": result.usage,
            "request_ids": result.request_ids,
            "endpoint": "agent_loop",
        },
    )
    return enforce_section_depth_audit(
        normalize_report_section_audit(result.final),
        section=section,
        section_plan=section_plan,
    )


def build_report_plan_prompt(
    *,
    paper: PaperRecord,
    skim: SkimCard | None,
    decision: ClassificationDecision | None,
    card: PaperCard | None,
    paper_memory: dict[str, Any],
    layout: dict[str, Any],
    topic: str | None,
    idea: str | None,
    output_language: str,
    read_mode: str,
) -> str:
    pages = list_payload(layout.get("pages"))
    runtime = PaperLensRuntime(artifacts=pages)
    memory_for_report = compact_paper_memory_for_report(paper_memory)
    focus_queries = report_focus_queries(paper_memory, paper=paper, skim=skim, card=card)
    context = runtime.build_context_pack(
        stage="report_plan",
        objective=(
            "Plan a streamed PaperLens report. Use PaperMemory as durable state and local "
            "paper-tool observations only to choose section focus and grounding."
        ),
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        classification=decision.class_label if decision else None,
        memory=paper_memory,
        focus_queries=focus_queries,
        focus_pages=report_focus_pages(paper_memory, skim=skim, card=card),
        output_contract={
            "type": "ReportPlan",
            "rule": "Plan section-level generation. Do not write the report in this step.",
        },
        search_limit=4,
        page_text_limit=900,
    ).as_dict()
    return "\n\n".join(
        [
            f"paper_id: {paper.paper_id}",
            f"title: {paper.canonical_title or 'unknown'}",
            f"output_language: {output_language}",
            f"read_mode: {read_mode}",
            "user_topic: " + (topic or "not provided"),
            "user_idea: " + (idea or "not provided"),
            "skim_card:",
            json.dumps(compact_skim_for_report(skim), ensure_ascii=False),
            "classification:",
            json.dumps(compact_decision_for_report(decision), ensure_ascii=False),
            "paper_card:",
            json.dumps(compact_paper_card_for_report(card), ensure_ascii=False),
            "paper_memory:",
            json.dumps(memory_for_report, ensure_ascii=False),
            "agent_context_pack:",
            context_pack_prompt(context),
            (
                "Task: create a report plan for a Standard PaperLens capsule. The plan should let "
                "later section calls explain the paper clearly without any single call writing the "
                "whole report."
            ),
            (
                "Completeness contract: plan 5-7 focused sections for A/B/C papers when evidence "
                "exists. Keep mechanism, implementation path, evaluation, value/tradeoffs, and "
                "limitations as separate sections unless the memory is truly too sparse. Each "
                "section needs concrete focus queries, claim ids, evidence refs, or target pages "
                "when available."
            ),
        ]
    )


def build_report_section_prompt(
    *,
    paper: PaperRecord,
    paper_memory: dict[str, Any],
    layout: dict[str, Any],
    plan: dict[str, Any],
    section_plan: dict[str, Any],
    previous_summaries: list[dict[str, str]],
    output_language: str,
    read_mode: str,
    section_audit: dict[str, Any] | None = None,
) -> str:
    pages = list_payload(layout.get("pages"))
    runtime = PaperLensRuntime(artifacts=pages)
    context = runtime.build_context_pack(
        stage="report_section",
        objective=(
            "Write one report section from PaperMemory and focused paper-tool observations. "
            "Do not write other sections."
        ),
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        classification=str(plan.get("grade") or ""),
        memory=paper_memory,
        focus_queries=list_payload(section_plan.get("focus_queries")),
        focus_pages=list_payload(section_plan.get("target_pages")),
        output_contract={
            "type": "ReportSection",
            "rule": "Return exactly one section body with used claim/evidence ids.",
        },
        search_limit=4,
        page_text_limit=2200 if report_section_is_mechanism(section_plan) else 1400,
    ).as_dict()
    detail_contract = report_section_detail_contract(section_plan)
    parts = [
        f"paper_id: {paper.paper_id}",
        f"title: {paper.canonical_title or 'unknown'}",
        f"output_language: {output_language}",
        f"read_mode: {read_mode}",
        "paper_memory:",
        json.dumps(compact_paper_memory_for_report(paper_memory), ensure_ascii=False),
        "report_plan:",
        json.dumps(compact_report_plan(plan), ensure_ascii=False),
        "section_to_write:",
        json.dumps(section_plan, ensure_ascii=False),
        "section_detail_contract:",
        detail_contract,
        "previous_section_summaries:",
        json.dumps(previous_summaries[-4:], ensure_ascii=False),
        "agent_context_pack:",
        context_pack_prompt(context),
    ]
    if section_audit:
        parts.extend(
            [
                "previous_section_audit:",
                json.dumps(section_audit, ensure_ascii=False),
                (
                    "Task: rewrite only this section to address the audit. Remove unsupported "
                    "claims; add missing context only when memory/evidence supports it."
                ),
            ]
        )
    else:
        parts.append(
            "Task: write only this planned section. Do not include the heading. Produce "
            "2-4 substantive paragraphs when the memory contains enough material; do not return "
            "a thin abstract-style summary."
        )
    return "\n\n".join(parts)


def build_report_section_audit_prompt(
    *,
    paper: PaperRecord,
    paper_memory: dict[str, Any],
    layout: dict[str, Any],
    plan: dict[str, Any],
    section_plan: dict[str, Any],
    section: dict[str, Any],
    output_language: str,
    read_mode: str,
) -> str:
    pages = list_payload(layout.get("pages"))
    runtime = PaperLensRuntime(artifacts=pages)
    context = runtime.build_context_pack(
        stage="report_section_audit",
        objective=(
            "Audit one generated report section against durable PaperMemory and focused local "
            "paper observations."
        ),
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        classification=str(plan.get("grade") or ""),
        memory=paper_memory,
        focus_queries=list_payload(section_plan.get("focus_queries")),
        focus_pages=list_payload(section_plan.get("target_pages")),
        output_contract={
            "type": "ReportSectionAudit",
            "rule": "Return REPAIR only for issues that matter for factuality or reader usefulness.",
        },
        search_limit=4,
        page_text_limit=1200,
    ).as_dict()
    return "\n\n".join(
        [
            f"paper_id: {paper.paper_id}",
            f"title: {paper.canonical_title or 'unknown'}",
            f"output_language: {output_language}",
            f"read_mode: {read_mode}",
            "paper_memory:",
            json.dumps(compact_paper_memory_for_report(paper_memory), ensure_ascii=False),
            "report_plan:",
            json.dumps(compact_report_plan(plan), ensure_ascii=False),
            "section_plan:",
            json.dumps(section_plan, ensure_ascii=False),
            "section_detail_contract:",
            report_section_detail_contract(section_plan),
            "generated_section:",
            json.dumps(section, ensure_ascii=False),
            "agent_context_pack:",
            context_pack_prompt(context),
            (
                "Task: audit this section. Focus on unsupported facts, overclaims, missing "
                "reader-critical context, and whether used_claim_ids/used_evidence_refs match the prose."
            ),
            (
                "Depth contract: if the section does not answer its section_detail_contract or is "
                "too shallow to teach the planned topic, return REPAIR with concrete repair instructions."
            ),
        ]
    )


def normalize_report_plan(
    data: dict[str, Any],
    *,
    paper: PaperRecord,
    decision: ClassificationDecision | None,
    paper_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    grade = str(data.get("grade") or (decision.class_label if decision else "HOLD")).upper()
    if grade not in {"A", "B", "C", "HOLD"}:
        grade = "HOLD"
    recommendation = str(data.get("read_recommendation") or "").strip()
    if recommendation not in {"重点关注", "标准读", "低优先级", "需确认"}:
        recommendation = recommendation_for_grade(grade)
    sections = [
        normalize_report_section_plan(item)
        for item in list_payload(data.get("sections"))
        if isinstance(item, dict)
    ]
    if not sections:
        sections = default_report_sections()
    sections = ensure_report_plan_coverage(
        sections, grade=grade, paper_memory=dict_value(paper_memory)
    )
    return {
        "paper_id": str(data.get("paper_id") or paper.paper_id),
        "grade": grade,
        "read_recommendation": recommendation,
        "one_line_reason": clean_model_inline_text(data.get("one_line_reason")),
        "core_takeaway": clean_model_markdown(data.get("core_takeaway")),
        "sections": sections[:7],
        "key_visual_pages": normalize_key_visual_pages(data.get("key_visual_pages")),
        "uncertainty_note": clean_model_markdown(data.get("uncertainty_note")),
    }


def ensure_report_plan_coverage(
    sections: list[dict[str, Any]], *, grade: str, paper_memory: dict[str, Any]
) -> list[dict[str, Any]]:
    """Supplement thin model plans with a complete PaperLens capsule profile."""
    normalized = [normalize_report_section_plan(section) for section in sections]
    if len(normalized) >= 7:
        return order_report_sections(normalized)[:7]
    desired = desired_report_section_templates(grade=grade, paper_memory=paper_memory)
    for template in desired:
        if len(normalized) >= 7:
            break
        if report_plan_template_is_covered(template, normalized):
            continue
        normalized.append(build_supplemental_report_section(template, paper_memory))
    minimum = minimum_report_sections_for_grade(grade)
    if len(normalized) < minimum:
        for template in default_report_section_templates():
            if len(normalized) >= minimum or len(normalized) >= 7:
                break
            if report_plan_template_is_covered(template, normalized):
                continue
            normalized.append(build_supplemental_report_section(template, paper_memory))
    return order_report_sections(normalized)[:7]


def order_report_sections(sections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = list(enumerate(sections))
    indexed.sort(key=lambda item: (report_section_order_index(item[1]), item[0]))
    return [section for _index, section in indexed]


def report_section_order_index(section: dict[str, Any]) -> int:
    kind = str(section.get("section_kind") or "")
    text = " ".join(
        [
            str(section.get("section_id") or ""),
            str(section.get("title") or ""),
            str(section.get("purpose") or ""),
        ]
    ).lower()
    if kind in {"orientation", "background"}:
        return 0
    if kind == "mechanism" and any(
        token in text
        for token in [
            "implementation",
            "runtime",
            "training",
            "inference",
            "实现",
            "细节",
            "路径",
            "训练",
            "推理",
        ]
    ):
        return 2
    if kind == "mechanism":
        return 1
    if kind == "evaluation":
        return 3
    if kind == "value":
        return 4
    if kind == "limits":
        return 5
    return 6


def minimum_report_sections_for_grade(grade: str) -> int:
    if grade in {"A", "B", "C"}:
        return 5
    return 4


def desired_report_section_templates(
    *, grade: str, paper_memory: dict[str, Any]
) -> list[dict[str, Any]]:
    templates = default_report_section_templates()
    if grade in {"A", "B", "C"} and should_add_implementation_section(paper_memory):
        return templates
    return [template for template in templates if template["section_id"] != "implementation"]


def default_report_section_templates() -> list[dict[str, Any]]:
    return [
        {
            "section_id": "orientation",
            "section_kind": "orientation",
            "title": "论文概览与核心问题",
            "purpose": "Explain the paper's problem, why it matters, and the core abstraction.",
            "coverage_group": "orientation",
            "detail_questions": [
                "What concrete problem or bottleneck is the paper attacking?",
                "Why do prior approaches fail or leave a gap?",
                "What is the paper's core abstraction or thesis?",
                "What should readers not over-claim from this paper?",
            ],
        },
        {
            "section_id": "mechanism",
            "section_kind": "mechanism",
            "title": "核心机制与设计结构",
            "purpose": "Explain the main mechanism in reader order.",
            "coverage_group": "mechanism",
            "detail_questions": [
                "What state, representation, or model view exists before the mechanism?",
                "What new abstraction or component changes that state?",
                "How do the main components connect?",
                "Why should this mechanism improve the target metric or behavior?",
            ],
        },
        {
            "section_id": "implementation",
            "section_kind": "mechanism",
            "title": "关键实现路径与细节",
            "purpose": "Walk through the implementation, algorithm, runtime path, or training/inference path.",
            "coverage_group": "implementation",
            "detail_questions": [
                "What data structures, equations, modules, or runtime components make it concrete?",
                "Walk through one request, object, sample, or inference lifecycle step by step.",
                "Which parameters, losses, schedules, or invariants matter?",
                "Where are the main overheads or fragile assumptions introduced?",
            ],
        },
        {
            "section_id": "evaluation",
            "section_kind": "evaluation",
            "title": "实验评估与证据",
            "purpose": "Explain datasets/workloads, metrics, baselines, headline results, and evidence limits.",
            "coverage_group": "evaluation",
            "detail_questions": [
                "What datasets, workloads, metrics, and baselines are used?",
                "What are the headline quantitative results?",
                "Which ablations or qualitative results support the mechanism?",
                "What exact result boundaries should readers keep in mind?",
            ],
        },
        {
            "section_id": "value",
            "section_kind": "value",
            "title": "价值、适用场景与权衡",
            "purpose": "Explain why the result is useful, where it transfers, and what tradeoff it chooses.",
            "coverage_group": "value",
            "detail_questions": [
                "What practical or conceptual lesson transfers beyond this paper?",
                "Which users, systems, tasks, or research directions benefit most?",
                "What tradeoff does the paper choose instead of optimizing everything?",
                "When would this idea be less useful?",
            ],
        },
        {
            "section_id": "limits",
            "section_kind": "limits",
            "title": "局限性与可信边界",
            "purpose": "State scope, assumptions, missing evidence, and open questions without burying them.",
            "coverage_group": "limits",
            "detail_questions": [
                "What evaluation scope, assumptions, or missing details limit the conclusion?",
                "Which claims require going back to the source before citation?",
                "What deployment, reproducibility, scaling, or generalization risks remain?",
                "What follow-up questions should a reader ask?",
            ],
        },
    ]


def should_add_implementation_section(paper_memory: dict[str, Any]) -> bool:
    core = core_memory_view_dict(paper_memory)
    if core:
        if list_payload(core.get("implementation_nodes")):
            return True
        if any(
            node.get("kind") in {"implementation", "mechanism", "evaluation", "result"}
            for node in list_payload(core.get("fact_nodes"))
        ):
            return True
    memory = paper_memory_v3_dict(paper_memory)
    mechanism = dict_value(memory.get("mechanism"))
    implementation = dict_value(memory.get("implementation_details"))
    if len(list_payload(mechanism.get("steps"))) >= 2:
        return True
    if list_payload(implementation.get("components")):
        return True
    if list_payload(memory.get("figures_tables")):
        return True
    return bool(dict_value(memory.get("evaluation")).get("summary"))


def report_plan_template_is_covered(
    template: dict[str, Any], sections: list[dict[str, Any]]
) -> bool:
    group = str(template.get("coverage_group") or template.get("section_kind") or "")
    if group == "orientation":
        return any(
            section.get("section_kind") in {"orientation", "background"} for section in sections
        )
    if group == "implementation":
        mechanism_sections = [
            section for section in sections if section.get("section_kind") == "mechanism"
        ]
        if len(mechanism_sections) >= 2:
            return True
        return any(
            any(
                token
                in " ".join(
                    [
                        str(section.get("section_id") or ""),
                        str(section.get("title") or ""),
                        str(section.get("purpose") or ""),
                    ]
                ).lower()
                for token in [
                    "implementation",
                    "runtime",
                    "algorithm",
                    "training",
                    "inference",
                    "实现",
                    "细节",
                    "运行",
                    "训练",
                    "推理",
                    "路径",
                ]
            )
            for section in sections
        )
    return any(section.get("section_kind") == group for section in sections)


def build_supplemental_report_section(
    template: dict[str, Any], paper_memory: dict[str, Any]
) -> dict[str, Any]:
    group = str(template.get("coverage_group") or template.get("section_kind") or "other")
    seed = report_section_seed_context(group, paper_memory)
    return normalize_report_section_plan(
        {
            "section_id": template["section_id"],
            "section_kind": template["section_kind"],
            "title": template["title"],
            "purpose": template["purpose"],
            "focus_queries": seed["focus_queries"],
            "claim_ids": seed["claim_ids"],
            "evidence_refs": seed["evidence_refs"],
            "target_pages": seed["target_pages"],
            "detail_questions": template["detail_questions"],
            "avoid": [],
        }
    )


def report_section_seed_context(group: str, paper_memory: dict[str, Any]) -> dict[str, Any]:
    core = core_memory_view_dict(paper_memory)
    if core:
        return core_report_section_seed_context(group, core)
    memory = paper_memory_v3_dict(paper_memory)
    claims = [claim for claim in list_payload(memory.get("claims")) if isinstance(claim, dict)]
    evidence = [item for item in list_payload(memory.get("evidence")) if isinstance(item, dict)]
    claim_ids: list[str] = []
    evidence_refs: list[str] = []
    focus_queries: list[str] = []
    target_pages: list[int] = []

    def add_claim(claim: dict[str, Any]) -> None:
        claim_id = string_or_none(claim.get("id"))
        if claim_id and claim_id not in claim_ids:
            claim_ids.append(claim_id)
        text = string_or_none(claim.get("text"))
        if text:
            focus_queries.append(text)
        for ref in normalized_string_list(claim.get("evidence_refs")):
            if ref not in evidence_refs:
                evidence_refs.append(ref)

    def claim_matches(claim: dict[str, Any]) -> bool:
        text = " ".join(
            [
                str(claim.get("type") or ""),
                str(claim.get("text") or ""),
                " ".join(normalized_string_list(claim.get("risk_tags"))),
            ]
        ).lower()
        if group in {"mechanism", "implementation"}:
            return any(
                token in text
                for token in [
                    "mechanism",
                    "design",
                    "algorithm",
                    "architecture",
                    "implementation",
                    "implication",
                    "机制",
                    "设计",
                    "算法",
                    "架构",
                    "实现",
                ]
            )
        if group == "evaluation":
            return any(
                token in text
                for token in [
                    "evaluation",
                    "comparison",
                    "benchmark",
                    "metric",
                    "performance",
                    "实验",
                    "评估",
                    "基线",
                    "性能",
                ]
            )
        if group == "limits":
            return any(
                token in text
                for token in [
                    "limitation",
                    "scope",
                    "risk",
                    "assumption",
                    "boundary",
                    "局限",
                    "边界",
                    "假设",
                ]
            )
        if group == "value":
            return any(
                token in text
                for token in [
                    "value",
                    "tradeoff",
                    "application",
                    "deployment",
                    "efficiency",
                    "robustness",
                    "价值",
                    "权衡",
                    "适用",
                    "部署",
                ]
            )
        return True

    for claim in claims:
        if claim_matches(claim):
            add_claim(claim)
        if len(claim_ids) >= 6:
            break
    if not claim_ids and group in {"orientation", "value"}:
        for claim in claims[:4]:
            add_claim(claim)

    evidence_by_id = {str(item.get("id")): item for item in evidence if item.get("id")}
    for ref in evidence_refs:
        item = evidence_by_id.get(ref)
        if item:
            page = safe_int(item.get("page"))
            if page and page not in target_pages:
                target_pages.append(page)
    for item in evidence:
        text = " ".join(
            [
                str(item.get("source_type") or ""),
                str(item.get("section") or ""),
                str(item.get("interpretation") or ""),
                str(item.get("excerpt_or_caption") or ""),
            ]
        ).lower()
        if group == "evaluation" and not any(
            token in text for token in ["table", "result", "metric", "baseline", "实验", "评估"]
        ):
            continue
        if group in {"mechanism", "implementation"} and not any(
            token in text
            for token in ["figure", "design", "architecture", "equation", "module", "机制", "架构"]
        ):
            continue
        evidence_id = string_or_none(item.get("id"))
        if evidence_id and evidence_id not in evidence_refs:
            evidence_refs.append(evidence_id)
        page = safe_int(item.get("page"))
        if page and page not in target_pages:
            target_pages.append(page)
        if len(evidence_refs) >= 8 and len(target_pages) >= 4:
            break

    frame = dict_value(memory.get("problem_frame"))
    mechanism = dict_value(memory.get("mechanism"))
    evaluation = dict_value(memory.get("evaluation"))
    if group == "orientation":
        focus_queries.extend([frame.get("problem"), frame.get("why_it_matters")])
        for item in list_payload(memory.get("core_abstractions"))[:2]:
            if isinstance(item, dict):
                focus_queries.append(str(item.get("text") or ""))
    elif group in {"mechanism", "implementation"}:
        focus_queries.append(str(mechanism.get("overview") or ""))
        for step in list_payload(mechanism.get("steps"))[:5]:
            if isinstance(step, dict):
                focus_queries.append(str(step.get("text") or ""))
    elif group == "evaluation":
        focus_queries.append(str(evaluation.get("summary") or ""))
        for item in list_payload(evaluation.get("items"))[:4]:
            if isinstance(item, dict):
                focus_queries.append(str(item.get("text") or ""))
    elif group == "limits":
        focus_queries.extend(normalized_string_list(memory.get("limitations"))[:5])
        focus_queries.extend(normalized_string_list(memory.get("open_questions"))[:4])

    return {
        "focus_queries": compact_string_list(focus_queries, limit=5, max_chars=180),
        "claim_ids": claim_ids[:8],
        "evidence_refs": evidence_refs[:10],
        "target_pages": target_pages[:6],
    }


def core_report_section_seed_context(group: str, core: dict[str, Any]) -> dict[str, Any]:
    fact_nodes = [node for node in list_payload(core.get("fact_nodes")) if isinstance(node, dict)]
    selected = [node for node in fact_nodes if core_fact_matches_group(node, group)]
    if not selected and group in {"orientation", "value"}:
        selected = fact_nodes[:4]
    focus_queries = [node.get("label") for node in selected[:6]]
    claim_ids = [
        string_or_none(node.get("node_id")) or "" for node in selected if node.get("node_id")
    ]
    evidence_refs: list[str] = []
    target_pages: list[int] = []
    for node in selected:
        for evidence_id in normalized_string_list(node.get("evidence_ids")):
            if evidence_id not in evidence_refs:
                evidence_refs.append(evidence_id)
        for source_id in normalized_string_list(node.get("source_ids")):
            if source_id not in evidence_refs:
                evidence_refs.append(source_id)
        for page in list_payload(node.get("pages")):
            page_no = safe_int(page)
            if page_no and page_no not in target_pages:
                target_pages.append(page_no)
    return {
        "focus_queries": compact_string_list(focus_queries, limit=5, max_chars=180),
        "claim_ids": claim_ids[:8],
        "evidence_refs": evidence_refs[:10],
        "target_pages": target_pages[:6],
    }


def core_fact_matches_group(node: dict[str, Any], group: str) -> bool:
    kind = string_or_none(node.get("kind")) or ""
    label = (string_or_none(node.get("label")) or "").lower()
    if group == "orientation":
        return kind in {"problem", "claim", "concept"}
    if group in {"mechanism", "implementation"}:
        return kind in {"mechanism", "implementation"} or any(
            token in label
            for token in ["mechanism", "implementation", "algorithm", "module", "机制", "实现"]
        )
    if group == "evaluation":
        return kind in {"evaluation", "result"}
    if group == "limits":
        return kind == "limitation"
    if group == "value":
        return kind in {"claim", "result", "concept"}
    return True


def paper_memory_v3_dict(memory: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(memory, dict):
        return {}
    if memory.get("schema_version") == "paper_memory.v3":
        return memory
    nested = dict_value(memory.get("paper_memory_v3"))
    return nested if nested.get("schema_version") == "paper_memory.v3" else {}


def normalize_report_section_plan(data: dict[str, Any]) -> dict[str, Any]:
    section_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(data.get("section_id") or "")).strip("_")
    title = clean_model_inline_text(data.get("title"))
    if not section_id:
        section_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", title.lower()).strip("_") or "section"
    section_kind = str(data.get("section_kind") or "").strip()
    if section_kind not in {
        "orientation",
        "background",
        "mechanism",
        "evaluation",
        "value",
        "limits",
        "other",
    }:
        section_kind = infer_report_section_kind(
            section_id=section_id, title=title, purpose=data.get("purpose")
        )
    return {
        "section_id": section_id[:40],
        "section_kind": section_kind,
        "title": title or "正文",
        "purpose": clean_model_inline_text(data.get("purpose")),
        "focus_queries": compact_string_list(data.get("focus_queries"), limit=5, max_chars=180),
        "claim_ids": compact_string_list(data.get("claim_ids"), limit=8, max_chars=40),
        "evidence_refs": compact_string_list(data.get("evidence_refs"), limit=10, max_chars=40),
        "target_pages": [
            page
            for page in (safe_int(value) for value in list_payload(data.get("target_pages")))
            if page
        ],
        "detail_questions": compact_string_list(
            data.get("detail_questions"), limit=8, max_chars=180
        ),
        "avoid": compact_string_list(data.get("avoid"), limit=5, max_chars=160),
    }


def normalize_report_section(
    data: dict[str, Any], *, section_plan: dict[str, Any]
) -> dict[str, Any]:
    paragraphs = compact_string_list(data.get("paragraphs"), limit=12, max_chars=1800)
    markdown_source = "\n\n".join(paragraphs) if paragraphs else data.get("markdown")
    return {
        "section_id": str(data.get("section_id") or section_plan.get("section_id") or "section"),
        "title": clean_model_inline_text(data.get("title"))
        or str(section_plan.get("title") or "正文"),
        "paragraphs": paragraphs,
        "markdown": sanitize_reader_hostile_text(readable_model_body(markdown_source)),
        "used_claim_ids": compact_string_list(data.get("used_claim_ids"), limit=12, max_chars=40),
        "used_evidence_refs": compact_string_list(
            data.get("used_evidence_refs"), limit=16, max_chars=40
        ),
        "uncertainty_note": clean_model_markdown(data.get("uncertainty_note")),
    }


def infer_report_section_kind(*, section_id: str, title: str, purpose: Any) -> str:
    text = " ".join([section_id, title, str(purpose or "")]).lower()
    if any(
        token in text
        for token in [
            "mechanism",
            "algorithm",
            "design",
            "architecture",
            "system",
            "机制",
            "算法",
            "架构",
            "系统",
            "如何工作",
        ]
    ):
        return "mechanism"
    if any(
        token in text for token in ["evaluation", "result", "benchmark", "实验", "评估", "性能"]
    ):
        return "evaluation"
    if any(token in text for token in ["limit", "scope", "boundary", "局限", "边界", "适用"]):
        return "limits"
    if any(
        token in text for token in ["background", "problem", "motivation", "背景", "问题", "动机"]
    ):
        return "background"
    if any(token in text for token in ["value", "transfer", "价值", "启发"]):
        return "value"
    return "other"


def report_section_is_mechanism(section_plan: dict[str, Any]) -> bool:
    if str(section_plan.get("section_kind") or "") == "mechanism":
        return True
    inferred = infer_report_section_kind(
        section_id=str(section_plan.get("section_id") or ""),
        title=str(section_plan.get("title") or ""),
        purpose=section_plan.get("purpose"),
    )
    return inferred == "mechanism"


def report_section_detail_contract(section_plan: dict[str, Any]) -> str:
    questions = compact_string_list(section_plan.get("detail_questions"), limit=8, max_chars=180)
    if report_section_is_mechanism(section_plan):
        defaults = [
            "What state or bottleneck exists before the paper's abstraction is introduced?",
            "What is the new abstraction, and what exactly does it re-map or decouple?",
            "Which data structures, tables, schedulers, allocators, or runtime components make it work?",
            "Walk through one request/object lifecycle step by step.",
            "Explain why each step changes memory, compute, latency, or coordination behavior.",
            "Name the main tradeoffs, overheads, and cases where the mechanism becomes less useful.",
        ]
        merged = merge_string_lists(questions, defaults, limit=10)
        return (
            "Mechanism section contract: answer these questions in connected prose, using only "
            "PaperMemory/evidence and local observations. Do not turn this into a bullet checklist: "
            + json.dumps(merged, ensure_ascii=False)
        )
    section_kind = str(section_plan.get("section_kind") or "")
    if section_kind in {"orientation", "background"}:
        defaults = [
            "Name the paper's concrete problem or bottleneck, not just the broad area.",
            "Explain why prior approaches are insufficient in this paper's framing.",
            "State the paper's core abstraction and the main misunderstanding to avoid.",
            "Keep background concepts separate from claims the paper actually proves.",
        ]
        merged = merge_string_lists(questions, defaults, limit=8)
        return (
            "Orientation section contract: answer these questions in connected prose: "
            + json.dumps(merged, ensure_ascii=False)
        )
    if section_kind == "evaluation":
        defaults = [
            "Describe datasets/workloads, metrics, baselines, and hardware or setting when available.",
            "Report the headline numbers or qualitative findings that directly support the claim.",
            "Explain which result supports which mechanism or value claim.",
            "State what the evaluation does not prove.",
        ]
        merged = merge_string_lists(questions, defaults, limit=8)
        return (
            "Evaluation section contract: answer these questions in connected prose: "
            + json.dumps(merged, ensure_ascii=False)
        )
    if section_kind == "value":
        defaults = [
            "Explain the transferable lesson or product/research value.",
            "Name the scenario where the idea is strongest.",
            "Name the tradeoff the paper chooses and what it gives up.",
            "Avoid generic praise; tie value to evidence and mechanism.",
        ]
        merged = merge_string_lists(questions, defaults, limit=8)
        return "Value section contract: answer these questions in connected prose: " + json.dumps(
            merged, ensure_ascii=False
        )
    if section_kind == "limits":
        defaults = [
            "List the paper's actual evaluation scope and assumptions.",
            "Identify exact numbers, baselines, or implementation details that need source checking.",
            "Explain deployment, reproducibility, scaling, or generalization risks when relevant.",
            "End with open questions that would affect whether a reader should trust or use the result.",
        ]
        merged = merge_string_lists(questions, defaults, limit=8)
        return "Limits section contract: answer these questions in connected prose: " + json.dumps(
            merged, ensure_ascii=False
        )
    if questions:
        return "Section-specific questions to answer: " + json.dumps(questions, ensure_ascii=False)
    return "No additional section-specific detail contract."


def normalize_report_section_audit(data: dict[str, Any]) -> dict[str, Any]:
    verdict = str(data.get("verdict") or "REPAIR").upper()
    if verdict not in {"PASS", "PASS_WITH_WEAKNESSES", "REPAIR"}:
        verdict = "REPAIR"
    unsupported_items = compact_string_list(data.get("unsupported_items"), limit=6, max_chars=240)
    if unsupported_items:
        verdict = "REPAIR"
    return {
        "verdict": verdict,
        "unsupported_items": unsupported_items,
        "missing_items": compact_string_list(data.get("missing_items"), limit=6, max_chars=240),
        "repair_instructions": compact_string_list(
            data.get("repair_instructions"), limit=6, max_chars=240
        ),
        "safe_usage_note": clean_model_markdown(data.get("safe_usage_note")),
    }


def enforce_section_depth_audit(
    audit: dict[str, Any], *, section: dict[str, Any], section_plan: dict[str, Any]
) -> dict[str, Any]:
    issue = section_depth_issue(section, section_plan)
    if not issue:
        return audit
    result = dict(audit)
    result["missing_items"] = compact_string_list(
        list_payload(result.get("missing_items")) + [issue], limit=6, max_chars=240
    )
    result["repair_instructions"] = compact_string_list(
        list_payload(result.get("repair_instructions"))
        + [
            "Rewrite this section with concrete mechanism/evidence/boundary detail from PaperMemory and focused pages."
        ],
        limit=6,
        max_chars=240,
    )
    result["verdict"] = "REPAIR"
    if not result.get("safe_usage_note"):
        result["safe_usage_note"] = "Section is too thin for its planned purpose."
    return result


def section_depth_issue(section: dict[str, Any], section_plan: dict[str, Any]) -> str:
    markdown = readable_model_body(section.get("markdown"))
    normalized = re.sub(r"\s+", "", markdown)
    char_count = len(normalized)
    kind = str(section_plan.get("section_kind") or "other")
    thresholds = {
        "orientation": 420,
        "background": 420,
        "mechanism": 650,
        "evaluation": 560,
        "value": 430,
        "limits": 380,
        "other": 360,
    }
    minimum = thresholds.get(kind, thresholds["other"])
    section_id = section_plan.get("section_id") or section.get("section_id")
    if char_count < minimum:
        return (
            f"Section '{section_id}' is too thin for {kind} coverage "
            f"({char_count} chars; expected at least {minimum})."
        )
    paragraphs = [
        paragraph
        for paragraph in re.split(r"\n\s*\n", markdown)
        if clean_model_inline_text(paragraph)
    ]
    if kind in {"mechanism", "evaluation"} and len(paragraphs) < 2:
        return (
            f"Section '{section_id}' needs at least two substantive paragraphs for {kind} coverage."
        )
    return ""


def report_section_is_more_substantive(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
    candidate_text = re.sub(r"\s+", "", readable_model_body(candidate.get("markdown")))
    current_text = re.sub(r"\s+", "", readable_model_body(current.get("markdown")))
    return len(candidate_text) >= max(len(current_text) + 120, int(len(current_text) * 1.2))


def assemble_agentic_report(
    *,
    paper: PaperRecord,
    decision: ClassificationDecision | None,
    plan: dict[str, Any],
    sections: list[dict[str, Any]],
    section_audits: list[dict[str, Any]],
    output_language: str,
) -> dict[str, Any]:
    grade = str(plan.get("grade") or (decision.class_label if decision else "HOLD")).upper()
    if grade not in {"A", "B", "C", "HOLD"}:
        grade = "HOLD"
    body_parts = []
    for section in sections:
        title = clean_model_inline_text(section.get("title"))
        markdown = readable_model_body(section.get("markdown"))
        if not markdown:
            continue
        if title:
            body_parts.append(f"## {title}\n\n{markdown}")
        else:
            body_parts.append(markdown)
    if not body_parts:
        body_parts.append(
            "模型没有生成可用的分段讲解。"
            if output_language == "zh"
            else "No usable section draft was generated."
        )
    uncertainty_parts = [user_facing_uncertainty_note(plan.get("uncertainty_note"))]
    if any(audit.get("verdict") != "PASS" for audit in section_audits):
        uncertainty_parts.append(
            "部分段落存在证据边界；具体数值、基线、硬件配置和外推结论建议按需追问。"
            if output_language == "zh"
            else "Some sections have evidence boundaries; ask follow-up questions before relying on exact numbers, baselines, hardware setup, or broad extrapolations."
        )
    return {
        "grade": grade,
        "review_status": section_review_status(section_audits, output_language=output_language),
        "read_recommendation": plan.get("read_recommendation") or recommendation_for_grade(grade),
        "one_line_reason": clean_model_inline_text(plan.get("one_line_reason"))
        or compact_reason(
            str(plan.get("core_takeaway") or paper.canonical_title or paper.paper_id)
        ),
        "core_takeaway": clean_model_markdown(plan.get("core_takeaway")),
        "explanation_markdown": "\n\n".join(body_parts),
        "uncertainty_note": "; ".join(item for item in uncertainty_parts if item),
        "key_visual_pages": normalize_key_visual_pages(plan.get("key_visual_pages")),
        "report_plan": plan,
        "section_audits": section_audits,
    }


def aggregate_section_audits(section_audits: list[dict[str, Any]]) -> dict[str, Any]:
    if not section_audits:
        return {
            "verdict": "NEED_HUMAN_REVIEW",
            "unsupported_items": [],
            "missing_items": ["No report sections were generated"],
            "correction_notes": ["ReportComposer produced no section artifacts"],
            "safe_usage_note": "No usable section-level report was generated.",
        }
    if any(audit.get("verdict") in {"REPAIR", "PASS_WITH_WEAKNESSES"} for audit in section_audits):
        verdict = "PASS_WITH_WEAKNESSES"
    else:
        verdict = "PASS"
    return {
        "verdict": verdict,
        "unsupported_items": compact_string_list(
            [
                item
                for audit in section_audits
                for item in list_payload(audit.get("unsupported_items"))
            ],
            limit=5,
            max_chars=240,
        ),
        "missing_items": compact_string_list(
            [item for audit in section_audits for item in list_payload(audit.get("missing_items"))],
            limit=5,
            max_chars=240,
        ),
        "correction_notes": compact_string_list(
            [
                item
                for audit in section_audits
                for item in list_payload(audit.get("repair_instructions"))
            ],
            limit=5,
            max_chars=240,
        ),
        "safe_usage_note": "; ".join(
            item
            for item in compact_string_list(
                [audit.get("safe_usage_note") for audit in section_audits],
                limit=3,
                max_chars=220,
            )
            if item
        ),
    }


def user_facing_uncertainty_note(value: Any) -> str:
    """Remove report-planning/internal memory wording from reader-facing uncertainty."""
    text = clean_model_markdown(value)
    if not text:
        return ""
    internal_markers = (
        "本报告计划",
        "本报告",
        "报告计划",
        "paper memory",
        "papermemory",
        "report plan",
        "memoryv3",
        "将严格遵循",
        "will strictly follow",
        "plan is based",
    )
    chunks = re.split(r"(?:[;；]\s*|\n\s*\n)", text)
    kept: list[str] = []
    for chunk in chunks:
        cleaned = chunk.strip()
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if any(marker in lowered for marker in internal_markers):
            continue
        kept.append(cleaned)
    return "; ".join(kept)


def section_review_status(section_audits: list[dict[str, Any]], *, output_language: str) -> str:
    if not section_audits:
        return "需要人工确认" if output_language == "zh" else "needs human review"
    if any(audit.get("verdict") == "REPAIR" for audit in section_audits):
        return (
            "已分段复核（有未修复边界）"
            if output_language == "zh"
            else "section-audited with unresolved boundaries"
        )
    if any(audit.get("verdict") == "PASS_WITH_WEAKNESSES" for audit in section_audits):
        return (
            "已分段复核（有证据边界）"
            if output_language == "zh"
            else "section-audited with evidence boundaries"
        )
    return "已分段复核" if output_language == "zh" else "section-audited"


def compact_report_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "grade": plan.get("grade"),
        "one_line_reason": plan.get("one_line_reason"),
        "core_takeaway": plan.get("core_takeaway"),
        "sections": [
            {
                "section_id": section.get("section_id"),
                "section_kind": section.get("section_kind"),
                "title": section.get("title"),
                "purpose": section.get("purpose"),
                "claim_ids": section.get("claim_ids"),
                "evidence_refs": section.get("evidence_refs"),
                "detail_questions": section.get("detail_questions"),
            }
            for section in list_payload(plan.get("sections"))
            if isinstance(section, dict)
        ],
    }


def default_report_sections() -> list[dict[str, Any]]:
    return [
        build_supplemental_report_section(template, {})
        for template in default_report_section_templates()
    ]


def cache_agent_run(
    client: JsonLlmClient, paper_id: str, stage: str, prefix: str, cache_path: Path | None
) -> dict[str, Any]:
    return {
        "agent_run_id": f"{prefix}_{paper_id}_cache",
        "paper_id": paper_id,
        "stage": stage,
        "provider_kind": client.config.kind,
        "model": client.config.model,
        "status": "CACHE_HIT",
        "cache": str(cache_path) if cache_path else "",
    }


def model_agent_run(
    client: JsonLlmClient, paper_id: str, stage: str, prefix: str, raw: Any, *, status: str
) -> dict[str, Any]:
    return {
        "agent_run_id": f"{prefix}_{paper_id}_{uuid.uuid4().hex[:8]}",
        "paper_id": paper_id,
        "stage": stage,
        "provider_kind": client.config.kind,
        "model": client.config.model,
        "endpoint": raw.endpoint,
        "request_id": raw.request_id,
        "usage": raw.usage,
        "status": status,
    }


def normalize_key_visual_pages(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    pages = []
    for item in value:
        if not isinstance(item, dict):
            continue
        page_no = item.get("page_no")
        if not isinstance(page_no, int) or page_no <= 0:
            continue
        pages.append(
            {
                "page_no": page_no,
                "reason": compact_reason(
                    clean_model_inline_text(item.get("reason")), max_chars=180
                ),
            }
        )
        if len(pages) >= 3:
            break
    return pages


def recommendation_for_grade(grade: str) -> str:
    return {"A": "重点关注", "B": "标准读", "C": "低优先级", "HOLD": "需确认"}.get(grade, "需确认")


def compact_reason(text: str, *, max_chars: int = 160) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    for mark in "。！？.!?；;，,":
        index = cleaned.rfind(mark, 0, max_chars)
        if index >= 40:
            return cleaned[: index + 1]
    return cleaned[:max_chars].rstrip() + "..."


def clean_model_markdown(value: Any) -> str:
    text = value.strip() if isinstance(value, str) else ""
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n")
    text = repair_markdown_boundaries(text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def repair_markdown_boundaries(text: str) -> str:
    sentence_boundary = r"([。！？!?；;：:])"
    text = re.sub(sentence_boundary + r"\s*(#{1,6}\s+)", r"\1\n\n\2", text)
    text = re.sub(r"([^\n])\s+(#{1,6}\s+)", r"\1\n\n\2", text)
    text = re.sub(
        sentence_boundary + r"\s+((?:\d+\.|[-*])\s+(?:\*\*|[^\s]))",
        r"\1\n\n\2",
        text,
    )
    heading_prefixes = (
        "核心抽象|问题背景|机制|证据|实验|评估|价值|可迁移性|局限|限制|边界|"
        "误解防护|关键图表|来源边界"
    )
    text = re.sub(
        rf"(?m)^(#{{1,6}}\s+(?:{heading_prefixes})[^\s。:：\n]{{0,12}}(?:[:：])?)[ \t]+(?=\S)",
        r"\1\n\n",
        text,
    )
    return text


def readable_model_body(value: Any) -> str:
    text = clean_model_markdown(value)
    if not text:
        return ""
    paragraphs = [
        paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()
    ]
    rewritten = []
    for paragraph in paragraphs:
        if (
            len(paragraph) <= 900
            or "\n" in paragraph
            or paragraph.lstrip().startswith(("-", "*", "1."))
        ):
            rewritten.append(paragraph)
            continue
        sentences = re.findall(
            r".+?(?:[。！？!?；;](?=\s|[\u4e00-\u9fffA-Za-z0-9])|[。！？!?；;]$|$)", paragraph
        )
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
        groups: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            if current and current_len + len(sentence) > 520:
                groups.append("".join(current).strip())
                current = []
                current_len = 0
            current.append(sentence)
            current_len += len(sentence)
        if current:
            groups.append("".join(current).strip())
        rewritten.extend(groups or [paragraph])
    return "\n\n".join(rewritten).strip()


def clean_model_inline_text(value: Any) -> str:
    return re.sub(r"\s+", " ", clean_model_markdown(value)).strip()


def compact_compare_text(text: str) -> str:
    return re.sub(r"\s+", "", text).strip().lower()


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    size = max(1, size)
    return [items[index : index + size] for index in range(0, len(items), size)]


def bounded_env_int(name: str, *, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def compact_skim_for_report(skim: SkimCard | None) -> dict[str, Any]:
    if skim is None:
        return {}
    return {
        "problem": compact_reason(skim.problem, max_chars=240),
        "method_type": skim.method_type,
        "system_scope": skim.system_scope,
        "evaluation_type": skim.evaluation_type,
        "danger_signals": compact_string_list(skim.danger_signals, limit=3, max_chars=120),
        "evidence_pages": [ref.page_no for ref in skim.evidence_refs[:4]],
    }


def compact_decision_for_report(decision: ClassificationDecision | None) -> dict[str, Any]:
    if decision is None:
        return {}
    return {
        "class_label": decision.class_label,
        "confidence": decision.confidence,
        "false_negative_risk": decision.false_negative_risk,
        "reason_codes": compact_string_list(decision.reason_codes, limit=3, max_chars=90),
        "audit_status": decision.audit_status,
        "validation_status": decision.validation_status,
        "validation_notes": compact_string_list(decision.validation_notes, limit=2, max_chars=110),
    }


def compact_paper_card_for_report(card: PaperCard | None) -> dict[str, Any]:
    if card is None:
        return {}
    return {
        "contribution_claims": compact_string_list(
            card.contribution_claims, limit=3, max_chars=170
        ),
        "mechanisms": compact_string_list(card.mechanisms, limit=3, max_chars=170),
        "evaluation": compact_string_list(card.evaluation, limit=2, max_chars=160),
        "limitations": compact_string_list(card.limitations, limit=2, max_chars=160),
        "assumptions": compact_string_list(card.assumptions, limit=2, max_chars=120),
        "verification_status": card.verification_status,
        "evidence_pages": [ref.page_no for ref in card.evidence_refs[:4]],
    }


def compact_string_list(value: Any, *, limit: int, max_chars: int) -> list[str]:
    return [
        compact_reason(item, max_chars=max_chars) for item in normalized_string_list(value)[:limit]
    ]


def fallback_model_paper_report(
    *,
    paper: PaperRecord,
    skim: SkimCard | None,
    decision: ClassificationDecision | None,
    card: PaperCard | None,
    paper_memory: dict[str, Any] | None,
    layout: dict[str, Any],
    topic: str | None,
    idea: str | None,
    reason: str,
    output_language: str = "zh",
) -> dict[str, Any]:
    _ = (paper_memory, layout, topic, idea)
    grade = decision.class_label if decision else "HOLD"
    limitation_note = f"Model final-report generation failed: {reason}"
    body_parts = []
    if output_language == "en":
        if skim and skim.problem:
            body_parts.append(f"This paper is roughly about: {skim.problem}")
        if card and card.contribution_claims:
            body_parts.append(
                "Current reliable signals: " + "; ".join(card.contribution_claims[:3]) + "."
            )
        if card and card.mechanisms:
            body_parts.append(
                "Possible mechanisms include: " + "; ".join(card.mechanisms[:3]) + "."
            )
        body_parts.append(
            "Final explanation generation failed, so this is a conservative fallback rather than a reliable capsule."
        )
        review_status = "needs human review"
    elif skim and skim.problem:
        body_parts.append(f"这篇论文大致在处理：{skim.problem}")
        if card and card.contribution_claims:
            body_parts.append(
                "当前能确定的主要线索是：" + "；".join(card.contribution_claims[:3]) + "。"
            )
        if card and card.mechanisms:
            body_parts.append("可能的关键做法包括：" + "；".join(card.mechanisms[:3]) + "。")
        body_parts.append(
            "但最终讲解生成失败，所以这份报告只能作为保守兜底结果，不能当成可靠论文总结。"
        )
        review_status = "需要人工确认"
    else:
        body_parts.append(
            "但最终讲解生成失败，所以这份报告只能作为保守兜底结果，不能当成可靠论文总结。"
        )
        review_status = "需要人工确认"
    return {
        "grade": grade,
        "review_status": review_status,
        "read_recommendation": recommendation_for_grade(grade),
        "one_line_reason": limitation_note,
        "explanation_markdown": "\n\n".join(body_parts),
        "uncertainty_note": limitation_note,
    }


def write_final_report_bundle(
    *,
    output_dir: Path,
    data_dir: Path,
    evidence_dir: Path,
    client: JsonLlmClient | None,
    record_usage: Any,
    record_agent_run: Any,
    stage: str,
    papers: list[PaperRecord],
    skim_cards: list[SkimCard],
    decisions: list[ClassificationDecision],
    paper_cards: list[PaperCard],
    review_items: list[ReviewItem],
    budget: dict[str, Any],
    budget_provider: Any | None = None,
    config: dict[str, Any],
    topic: str | None,
    idea: str | None,
    cache_dir: Path | None = None,
) -> list[Path]:
    formal_run = not bool(config.get("offline_debug"))
    require_report_success = formal_run and report_generation_must_succeed()
    output_language = str(config.get("output_language") or "zh")
    if output_language not in {"zh", "en"}:
        output_language = "zh"
    read_mode = str(config.get("read_mode") or "standard")
    if read_mode != "standard":
        raise ValueError("PaperLens Core currently supports only read_mode='standard'")
    card_by_id = {card.paper_id: card for card in paper_cards}
    skim_by_id = {card.paper_id: card for card in skim_cards}
    decision_by_id = {decision.paper_id: decision for decision in decisions}
    paper_report_rows: list[dict[str, Any]] = []
    written: list[Path] = []
    memory_store = PaperMemoryStore(data_dir)

    for paper in papers:
        skim = skim_by_id.get(paper.paper_id)
        decision = decision_by_id.get(paper.paper_id)
        card = card_by_id.get(paper.paper_id)
        report_name = paper_report_filename(paper)
        report_path = output_dir / "papers" / report_name
        layout = load_layout_index(data_dir, paper.paper_id)
        paper_memory_v3 = memory_store.initialize(
            paper=paper,
            skim=skim,
            decision=decision,
            card=card,
            layout=layout,
            source="export_prepare",
            prefer_existing=True,
        )
        paper_memory_for_prompt = build_report_memory_context(
            data_dir=data_dir,
            paper_id=paper.paper_id,
            paper_memory_v3=paper_memory_v3,
        )
        model_report = None
        report_audit = None
        if formal_run:
            if client is None:
                raise RuntimeError("Formal report generation requires a model client")
            try:
                model_report, report_audit = compose_agentic_paper_report(
                    client=client,
                    data_dir=data_dir,
                    stage=stage,
                    paper=paper,
                    skim=skim,
                    decision=decision,
                    card=card,
                    paper_memory=paper_memory_for_prompt,
                    layout=layout,
                    topic=topic,
                    idea=idea,
                    output_language=output_language,
                    record_usage=record_usage,
                    record_agent_run=record_agent_run,
                    read_mode=read_mode,
                    cache_dir=cache_dir,
                )
            except Exception as exc:
                record_agent_run(
                    {
                        "agent_run_id": f"final_report_{paper.paper_id}_failed",
                        "paper_id": paper.paper_id,
                        "stage": stage,
                        "provider_kind": client.config.kind,
                        "model": client.config.model,
                        "status": "FALLBACK",
                        "error": str(exc),
                    }
                )
                if require_report_success:
                    raise RuntimeError(
                        f"Final report generation failed for {paper.paper_id}: {exc}"
                    ) from exc
                model_report = fallback_model_paper_report(
                    paper=paper,
                    skim=skim,
                    decision=decision,
                    card=card,
                    paper_memory=paper_memory_for_prompt,
                    layout=layout,
                    topic=topic,
                    idea=idea,
                    reason=str(exc),
                    output_language=output_language,
                )
                report_audit = {
                    "verdict": "NEED_HUMAN_REVIEW",
                    "unsupported_items": [],
                    "missing_items": ["model-generated final explanation was not available"],
                    "correction_notes": [f"report_generation_failed: {exc}"],
                    "safe_usage_note": "This report used a deterministic fallback and needs human review before citation.",
                }
            report_audit = combine_report_and_memory_audits(report_audit, paper_memory_v3)
            if require_report_success and not final_report_audit_acceptable(report_audit):
                safe_usage_note = compact_reason(
                    str((report_audit or {}).get("safe_usage_note") or ""), max_chars=220
                )
                suffix = f" ({safe_usage_note})" if safe_usage_note else ""
                raise RuntimeError(
                    f"Final report audit did not produce a usable report for {paper.paper_id}: "
                    f"{(report_audit or {}).get('verdict')}{suffix}"
                )
        if report_audit is not None:
            paper_memory_v3 = memory_store.apply_patch_set(
                paper.paper_id,
                {
                    "paper_id": paper.paper_id,
                    "operations": [{"op": "set_report_audit", "payload": report_audit}],
                },
                source="export_report_audit",
            )
        written.append(write_paper_memory_v3_file(data_dir, paper_memory_v3))
        report_markdown = render_paper_report(
            paper=paper,
            skim=skim,
            decision=decision,
            card=card,
            layout=layout,
            topic=topic,
            idea=idea,
            formal_run=formal_run,
            model_report=model_report,
            report_audit=report_audit,
            output_dir=output_dir,
            output_language=output_language,
        )
        report_path.write_text(report_markdown, encoding="utf-8")
        written.append(report_path)
        core_graph_report_path = write_core_graph_report_view(
            output_dir=output_dir,
            data_dir=data_dir,
            paper_id=paper.paper_id,
            title=paper.canonical_title or paper.paper_id,
            report_name=report_name,
        )
        if core_graph_report_path is not None:
            written.append(core_graph_report_path)
        paper_report_rows.append(
            {
                "paper": paper,
                "skim": skim,
                "decision": decision,
                "card": card,
                "report_name": report_name,
                "core_graph_report_name": (
                    core_graph_report_path.relative_to(output_dir / "papers").as_posix()
                    if core_graph_report_path is not None
                    else None
                ),
                "report_title": markdown_title(report_markdown) or paper.canonical_title,
                "paper_memory_v3": paper_memory_v3,
                "model_report": model_report,
                "report_audit": report_audit,
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


def report_generation_must_succeed() -> bool:
    explicit = os.getenv("PAPERLENS_REQUIRE_LLM")
    if explicit is not None:
        return explicit == "1"
    return os.getenv("PAPERLENS_ALLOW_LLM_FALLBACK", "0") != "1"


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def render_start_here(
    *,
    rows: list[dict[str, Any]],
    review_items: list[ReviewItem],
    budget: dict[str, Any],
    config: dict[str, Any],
    topic: str | None,
    idea: str | None,
    formal_run: bool,
) -> str:
    ranked = sorted(rows, key=reading_priority_key)
    must_read = [row for row in ranked if row_decision(row).class_label in {"A", "HOLD"}][:10]
    risky = [row for row in ranked if novelty_risk(row) in {"HIGH", "MEDIUM"}][:10]
    provider = config.get("provider", {}) if isinstance(config.get("provider"), dict) else {}
    lines = [
        "# PaperLens",
        "",
        "## What to open",
        "",
        "- Run report: [PaperLens.md](PaperLens.md)",
        "- Per-paper reports: [papers/](papers/)",
        "- Local memory assets: [.paperlens/library/](.paperlens/library/)",
        "",
        "## Run summary",
        "",
        f"- Papers analyzed: {len(rows)}",
        f"- Mode: {'formal agentic audit' if formal_run else 'offline parse/debug diagnostic'}",
        f"- Topic comparison: {'enabled' if topic and idea else 'not enabled'}",
        f"- Provider: {provider.get('kind', 'unknown')}",
        f"- Model: {provider.get('model') or 'not recorded'}",
        f"- Estimated model cost: ${float(budget.get('estimated_usd') or 0):.4f}",
        f"- Human review items: {len(review_items)}",
        "",
    ]
    if topic or idea:
        lines.extend(
            [
                "## User research context",
                "",
                f"- Topic: {topic or 'not provided'}",
                f"- Idea: {idea or 'not provided'}",
                "",
            ]
        )
    lines.extend(["## Read first", ""])
    if formal_run:
        lines.extend(report_link_lines(must_read) or ["- No must-read papers were selected."])
    else:
        lines.append(
            "- Offline debug output is not a formal reading result. Use it only to inspect parsing, rendering, and evidence plumbing."
        )
    lines.extend(["", "## High-value or risky papers", ""])
    if formal_run:
        lines.extend(report_link_lines(risky) or ["- No medium/high risk papers were selected."])
    else:
        lines.append("- Not assessed in offline debug mode.")
    lines.extend(["", "## Needs human confirmation", ""])
    if review_items:
        for item in review_items[:20]:
            paper_id = item.paper_id or "run"
            lines.append(f"- `{paper_id}`: {item.reason}")
    else:
        lines.append("- No open review items.")
    return "\n".join(lines) + "\n"


def render_main_report(
    *,
    rows: list[dict[str, Any]],
    review_items: list[ReviewItem],
    budget: dict[str, Any],
    topic: str | None,
    idea: str | None,
    formal_run: bool,
) -> str:
    ranked = sorted(rows, key=reading_priority_key)
    counts = {"A": 0, "B": 0, "C": 0, "HOLD": 0}
    for row in rows:
        counts[row_decision(row).class_label] += 1
    if not formal_run:
        return render_debug_main_report(
            rows=ranked, review_items=review_items, budget=budget, counts=counts
        )
    lines = [
        "# PaperLens Report",
        "",
        "## 1. Executive Summary",
        "",
        f"- Papers analyzed: {len(rows)}",
        f"- Classification counts: A={counts['A']}, B={counts['B']}, C={counts['C']}, HOLD={counts['HOLD']}",
        f"- Estimated model cost: ${float(budget.get('estimated_usd') or 0):.4f}",
        f"- Report mode: {'formal agentic audit' if formal_run else 'offline parse/debug diagnostic'}",
        f"- Topic comparison: {'enabled' if topic and idea else 'not enabled'}",
        "",
    ]
    lines.extend(["## 2. Recommended Reading Order", ""])
    for index, row in enumerate(ranked, start=1):
        paper = row["paper"]
        decision = row_decision(row)
        lines.append(
            f"{index}. [{paper.paper_id}: {paper.canonical_title or paper.paper_id}](papers/{row['report_name']}) "
            f"- {read_decision(row)}; class {decision.class_label}; novelty risk {novelty_risk(row)}."
        )
    lines.extend(["", "## 3. Must Read Papers", ""])
    must_read = [row for row in ranked if read_decision(row) == "Close read"]
    lines.extend(describe_rows(must_read) or ["No papers require full close reading."])
    lines.extend(["", "## 4. Background Papers", ""])
    background = [row for row in ranked if read_decision(row) in {"Background", "Skip"}]
    lines.extend(describe_rows(background) or ["No background-only papers were identified."])
    lines.extend(["", "## 5. Value / Novelty Risk", ""])
    if topic and idea:
        lines.extend(
            [
                f"- User topic: {topic}",
                f"- User idea: {idea}",
                "",
            ]
        )
    else:
        lines.append(
            "User topic and idea were not provided; this section uses general paper-value and novelty risk."
        )
    for row in [item for item in ranked if novelty_risk(item) in {"HIGH", "MEDIUM"}]:
        paper = row["paper"]
        lines.append(
            f"- `{paper.paper_id}` {paper.canonical_title}: {novelty_risk(row)} risk. {row_relation(row)}"
        )
    if not any(novelty_risk(item) in {"HIGH", "MEDIUM"} for item in ranked):
        lines.append("- No medium/high novelty risk papers were identified.")
    lines.extend(["", "## 6. Paper Value Map", ""])
    for cluster, cluster_rows in cluster_rows_by_scope(ranked).items():
        lines.append(f"### {cluster}")
        for row in cluster_rows:
            paper = row["paper"]
            lines.append(
                f"- [{paper.paper_id}: {paper.canonical_title}](papers/{row['report_name']}) - {row_relation(row)}"
            )
        lines.append("")
    lines.extend(["## 7. Claim Safety Audit", ""])
    for row in ranked:
        card = row.get("card")
        paper = row["paper"]
        if not card:
            continue
        status = "PROBABLY_SAFE" if card.verification_status == "PASS" else "NEEDS_HUMAN_REVIEW"
        lines.append(
            f"- `{paper.paper_id}` {status}: cite only claims backed by listed EvidenceRef entries."
        )
    lines.extend(["", "## 8. Human Review Queue", ""])
    if review_items:
        for item in review_items:
            lines.append(f"- `{item.paper_id or 'run'}` {item.item_type}: {item.reason}")
    else:
        lines.append("- No open human review items.")
    lines.extend(
        [
            "",
            "## 9. Paper Index",
            "",
            "| Paper | Read | Class | Risk | Report |",
            "|---|---|---|---|---|",
        ]
    )
    for row in ranked:
        paper = row["paper"]
        decision = row_decision(row)
        lines.append(
            f"| {paper.paper_id} | {read_decision(row)} | {decision.class_label} | {novelty_risk(row)} | "
            f"[open](papers/{row['report_name']}) |"
        )
    return "\n".join(lines) + "\n"


def render_paper_report(
    *,
    paper: PaperRecord,
    skim: SkimCard | None,
    decision: ClassificationDecision | None,
    card: PaperCard | None,
    layout: dict[str, Any],
    topic: str | None,
    idea: str | None,
    formal_run: bool,
    model_report: dict[str, Any] | None,
    report_audit: dict[str, Any] | None = None,
    output_dir: Path | None = None,
    output_language: str = "zh",
) -> str:
    if not formal_run:
        return render_debug_paper_diagnostic(
            paper=paper,
            skim=skim,
            decision=decision,
            card=card,
            layout=layout,
        )
    if not model_report:
        raise RuntimeError(f"Missing model-generated final report for {paper.paper_id}")
    return render_freeform_paper_report(
        paper=paper,
        decision=decision,
        card=card,
        layout=layout,
        model_report=model_report,
        report_audit=report_audit,
        output_dir=output_dir,
        output_language=output_language,
    )


def render_freeform_paper_report(
    *,
    paper: PaperRecord,
    decision: ClassificationDecision | None,
    card: PaperCard | None = None,
    layout: dict[str, Any] | None = None,
    model_report: dict[str, Any],
    report_audit: dict[str, Any] | None,
    output_dir: Path | None = None,
    output_language: str = "zh",
) -> str:
    grade = string_or_none(model_report.get("grade")) or (
        decision.class_label if decision else "HOLD"
    )
    recommendation = localized_recommendation(
        string_or_none(model_report.get("read_recommendation")) or recommendation_for_grade(grade),
        output_language=output_language,
    )
    review_status = display_review_status(
        model_report, report_audit, output_language=output_language
    )
    reason = compact_reason(
        sanitize_reader_hostile_text(clean_model_inline_text(model_report.get("one_line_reason")))
        or "模型没有给出一句话理由。"
    )
    core_takeaway = sanitize_reader_hostile_text(
        clean_model_markdown(model_report.get("core_takeaway"))
    )
    body = (
        sanitize_reader_hostile_text(readable_model_body(model_report.get("explanation_markdown")))
        or "模型没有给出可用讲解。"
    )
    body = trim_redundant_body_opening(body, core_takeaway)
    uncertainty = sanitize_reader_hostile_text(
        user_facing_uncertainty_note(model_report.get("uncertainty_note"))
    )
    visual_markdown = render_key_visual_crops(
        paper=paper,
        model_report=model_report,
        card=card,
        layout=layout or {},
        output_dir=output_dir,
        output_language=output_language,
    )
    trust_boundary = report_trust_boundary(report_audit, output_language=output_language)
    labels = report_display_labels(output_language)
    lines = [
        f"# {display_paper_title(paper)}",
        "",
        labels["meta"].format(
            grade=grade, review_status=review_status, recommendation=recommendation
        ),
    ]
    if reason and not body_starts_with_reason(body, reason):
        lines.extend(["", f"> {reason}"])
    if core_takeaway:
        lines.extend(["", f"**{labels['core_anchor']}** {core_takeaway}"])
    lines.extend(["", body.strip()])
    if visual_markdown:
        lines.extend(["", visual_markdown])
    if uncertainty:
        lines.extend(["", f"{labels['uncertainty']}：{uncertainty}"])
    if trust_boundary:
        lines.extend(["", f"{labels['trust_boundary']}：{trust_boundary}"])
    return "\n".join(lines).rstrip() + "\n"


def report_display_labels(output_language: str) -> dict[str, str]:
    if output_language == "en":
        return {
            "meta": "Grade: {grade} · Review: {review_status} · Recommendation: {recommendation}",
            "core_anchor": "First hold this abstraction:",
            "uncertainty": "Uncertainty",
            "trust_boundary": "Trust boundary",
        }
    return {
        "meta": "等级：{grade} · 复核：{review_status} · 建议：{recommendation}",
        "core_anchor": "先抓住这个抽象：",
        "uncertainty": "不确定",
        "trust_boundary": "可信边界",
    }


def localized_recommendation(recommendation: str, *, output_language: str) -> str:
    if output_language != "en":
        return recommendation
    return {
        "重点关注": "high priority",
        "标准读": "standard read",
        "低优先级": "lower priority",
        "需确认": "needs confirmation",
    }.get(recommendation, recommendation)


def report_trust_boundary(
    report_audit: dict[str, Any] | None, *, output_language: str = "zh"
) -> str:
    if not report_audit or report_audit.get("verdict") == "PASS":
        return ""
    if report_audit.get("verdict") == "PASS_WITH_WEAKNESSES":
        if output_language == "en":
            return (
                "This capsule passed review with evidence boundaries; ask follow-up questions or "
                "check the source before citing exact numbers, broad extrapolations, or implementation details."
            )
        return "这份胶囊已经过复核，但仍存在证据边界；具体数值、外推结论和实现细节建议按需追问或回到原文核对。"
    if report_audit.get("verdict") == "NEED_HUMAN_REVIEW":
        if output_language == "en":
            return "This capsule did not pass automatic review and should only be used as a reading lead."
        return "这份胶囊未通过自动复核，只能作为阅读线索，不能直接当作可靠结论。"
    return ""


def sanitize_reader_hostile_text(text: str | None) -> str:
    if not text:
        return ""
    replacements = {
        "你给到的片段": "当前自动阅读证据",
        "你给到的摘录": "当前自动阅读证据",
        "你给到": "当前自动阅读证据",
        "你提供的片段": "当前自动阅读证据",
        "你提供的摘录": "当前自动阅读证据",
        "你提供": "当前自动阅读证据",
        "供给的片段": "当前自动阅读证据",
        "供给片段": "当前自动阅读证据",
        "供给的图示": "自动阅读证据中的图示",
        "提供的页面": "当前自动阅读证据",
        "提供的材料": "当前自动阅读证据",
        "提供的证据": "当前自动阅读证据",
        "the supplied excerpts": "the automatic reading evidence",
        "supplied excerpts": "automatic reading evidence",
        "provided excerpts": "automatic reading evidence",
        "provided excerpt": "automatic reading evidence",
        "the user provided": "the current evidence contains",
    }
    cleaned = text
    for old, new in replacements.items():
        cleaned = cleaned.replace(old, new)
    return cleaned


def trim_redundant_body_opening(body: str, core_takeaway: str) -> str:
    if not body or not core_takeaway:
        return body
    paragraphs = body.split("\n\n")
    if not paragraphs:
        return body
    first = paragraphs[0]
    sentences = split_report_sentences(first)
    if len(sentences) < 2:
        return body
    removed = 0
    while sentences and removed < 2 and sentence_overlaps_anchor(sentences[0], core_takeaway):
        sentences.pop(0)
        removed += 1
    if (
        removed
        and sentences
        and clean_model_inline_text(sentences[0]).startswith(
            ("理解了这个", "有了这个", "在这个基础上")
        )
    ):
        sentences.pop(0)
    if not removed or not sentences:
        return body
    paragraphs[0] = "".join(sentences).strip()
    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph.strip())


def split_report_sentences(paragraph_text: str) -> list[str]:
    parts = re.findall(r"[^。！？.!?]+[。！？.!?]?", paragraph_text.strip())
    return [part for part in parts if part]


def sentence_overlaps_anchor(sentence: str, anchor: str) -> bool:
    sentence_chars = {char for char in sentence if "\u4e00" <= char <= "\u9fff"}
    anchor_chars = {char for char in anchor if "\u4e00" <= char <= "\u9fff"}
    if sentence_chars and anchor_chars:
        overlap = len(sentence_chars & anchor_chars) / max(1, len(sentence_chars))
        if overlap >= 0.45:
            return True
    sentence_terms = set(re.findall(r"[A-Za-z0-9_+-]{3,}", sentence.lower()))
    anchor_terms = set(re.findall(r"[A-Za-z0-9_+-]{3,}", anchor.lower()))
    if sentence_terms and anchor_terms:
        return len(sentence_terms & anchor_terms) / max(1, len(sentence_terms)) >= 0.5
    return False


def render_key_visual_crops(
    *,
    paper: PaperRecord,
    model_report: dict[str, Any],
    card: PaperCard | None,
    layout: dict[str, Any],
    output_dir: Path | None,
    output_language: str = "zh",
) -> str:
    if output_dir is None:
        return ""
    pages = select_key_visual_pages(
        paper=paper,
        model_report=model_report,
        card=card,
        layout=layout,
        limit=3,
        output_language=output_language,
    )
    if not pages:
        return ""
    pages_by_no = layout_pages_by_no(layout)
    visuals: list[dict[str, Any]] = []
    for page in pages:
        page_no = page["page_no"]
        layout_page = pages_by_no.get(page_no)
        if not layout_page:
            continue
        bbox = visual_crop_bbox_for_page(layout_page)
        if not bbox:
            continue
        image_path = render_visual_crop(
            output_dir=output_dir,
            paper=paper,
            page_no=page_no,
            bbox=bbox,
            visual_index=len(visuals) + 1,
        )
        if not image_path:
            continue
        fallback_reason = (
            f"Page {page_no} contains key visual evidence."
            if output_language == "en"
            else f"第 {page_no} 页包含关键视觉证据。"
        )
        reason = clean_model_inline_text(page.get("reason")) or fallback_reason
        visuals.append({"page_no": page_no, "reason": reason, "image_path": image_path})
    if not visuals:
        return ""
    lines = ["## Key Figures" if output_language == "en" else "## 关键图表"]
    for visual in visuals:
        page_no = visual["page_no"]
        reason = visual_reader_reason(visual["reason"], output_language=output_language)
        image_path = visual["image_path"]
        caption_prefix = (
            f"Page {page_no} crop" if output_language == "en" else f"第 {page_no} 页裁剪"
        )
        lines.extend(
            [
                "",
                "<figure>",
                f'  <img src="{image_path}" alt="{display_paper_title(paper)} visual crop from page {page_no}" width="720">',
                f"  <figcaption>{caption_prefix}: {reason}</figcaption>",
                "</figure>",
            ]
        )
    return "\n".join(lines)


def layout_pages_by_no(layout: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        page.get("page_no"): page
        for page in layout.get("pages", [])
        if isinstance(page, dict) and isinstance(page.get("page_no"), int)
    }


def visual_crop_bbox_for_page(page: dict[str, Any]) -> list[float] | None:
    page_width = positive_float(page.get("page_width"))
    page_height = positive_float(page.get("page_height"))
    if page_width is None or page_height is None:
        return None
    page_area = page_width * page_height
    if page_area <= 0:
        return None
    base_bboxes: list[list[float]] = []
    captions = [item for item in page.get("captions") or [] if isinstance(item, dict)]
    for kind in ["figures", "tables", "images"]:
        for item in page.get(kind) or []:
            if not isinstance(item, dict):
                continue
            bbox = valid_visual_bbox(item.get("bbox"), page_width, page_height)
            if bbox is None:
                continue
            base_bboxes.append(bbox)
    candidates: list[tuple[float, list[float]]] = []
    for bbox in merge_visual_bbox_groups(base_bboxes, page_width, page_height):
        merged = merge_nearby_caption_bboxes(bbox, captions, page_width, page_height)
        crop = expand_visual_bbox(merged, page_width, page_height)
        if not crop or not visual_bbox_is_reportable(crop, page_area):
            continue
        candidates.append((visual_bbox_area(crop), crop))
    if not candidates:
        for bbox in base_bboxes:
            merged = merge_nearby_caption_bboxes(bbox, captions, page_width, page_height)
            crop = expand_visual_bbox(merged, page_width, page_height)
            if not crop or not visual_bbox_is_reportable(crop, page_area):
                continue
            candidates.append((visual_bbox_area(crop), crop))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def render_visual_crop(
    *,
    output_dir: Path,
    paper: PaperRecord,
    page_no: int,
    bbox: list[float],
    visual_index: int,
) -> str | None:
    pdf_path = Path(paper.file_path)
    if not pdf_path.exists():
        return None
    try:
        from paperlens_core.pdf.pymupdf_parser import require_pymupdf

        fitz = require_pymupdf()
        with fitz.open(pdf_path) as doc:
            if page_no < 1 or page_no > len(doc):
                return None
            page = doc[page_no - 1]
            clipped = [
                max(0.0, min(float(bbox[0]), float(page.rect.width))),
                max(0.0, min(float(bbox[1]), float(page.rect.height))),
                max(0.0, min(float(bbox[2]), float(page.rect.width))),
                max(0.0, min(float(bbox[3]), float(page.rect.height))),
            ]
            if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                return None
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2, 2), clip=fitz.Rect(*clipped), alpha=False
            )
            if pixmap.width < 120 or pixmap.height < 80:
                return None
            figures_dir = output_dir / ".paperlens" / "figures" / paper.paper_id
            figures_dir.mkdir(parents=True, exist_ok=True)
            filename = f"page_{page_no:04d}_visual_{visual_index:02d}.png"
            crop_path = figures_dir / filename
            pixmap.save(crop_path)
            return f"../.paperlens/figures/{paper.paper_id}/{filename}"
    except Exception:
        return None


def positive_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result > 0 else None


def valid_visual_bbox(value: Any, page_width: float, page_height: float) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    bbox = [
        max(0.0, min(bbox[0], page_width)),
        max(0.0, min(bbox[1], page_height)),
        max(0.0, min(bbox[2], page_width)),
        max(0.0, min(bbox[3], page_height)),
    ]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    if (bbox[2] - bbox[0]) < 24 or (bbox[3] - bbox[1]) < 24:
        return None
    page_area = page_width * page_height
    if page_area <= 0 or visual_bbox_area(bbox) / page_area > 0.65:
        return None
    return bbox


def valid_caption_bbox(value: Any, page_width: float, page_height: float) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    bbox = [
        max(0.0, min(bbox[0], page_width)),
        max(0.0, min(bbox[1], page_height)),
        max(0.0, min(bbox[2], page_width)),
        max(0.0, min(bbox[3], page_height)),
    ]
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    if (bbox[2] - bbox[0]) < 12 or (bbox[3] - bbox[1]) < 6:
        return None
    page_area = page_width * page_height
    if page_area <= 0 or visual_bbox_area(bbox) / page_area > 0.35:
        return None
    return bbox


def merge_visual_bbox_groups(
    bboxes: list[list[float]],
    page_width: float,
    page_height: float,
) -> list[list[float]]:
    groups: list[list[float]] = []
    gap = max(24.0, min(page_width, page_height) * 0.06)
    for bbox in sorted(bboxes, key=lambda item: (item[1], item[0])):
        target_index = None
        for index, group in enumerate(groups):
            if visual_bboxes_near(group, bbox, gap=gap):
                target_index = index
                break
        if target_index is None:
            groups.append(bbox)
        else:
            groups[target_index] = union_visual_bbox(groups[target_index], bbox)
    return groups


def merge_nearby_caption_bboxes(
    bbox: list[float],
    captions: list[dict[str, Any]],
    page_width: float,
    page_height: float,
) -> list[float]:
    caption_bboxes = [
        valid_caption_bbox(caption.get("bbox"), page_width, page_height)
        for caption in captions
        if isinstance(caption, dict)
    ]
    caption_bboxes = [caption for caption in caption_bboxes if caption is not None]
    if not caption_bboxes:
        return bbox
    merged = bbox
    for caption in sorted(caption_bboxes, key=lambda item: visual_bbox_distance(bbox, item)):
        if not caption_is_near_visual(bbox, caption, page_width, page_height):
            continue
        candidate = union_visual_bbox(merged, caption)
        page_area = page_width * page_height
        if page_area <= 0 or visual_bbox_area(candidate) / page_area > 0.65:
            continue
        merged = candidate
    return merged


def caption_is_near_visual(
    bbox: list[float],
    caption: list[float],
    page_width: float,
    page_height: float,
) -> bool:
    if visual_bbox_distance(bbox, caption) <= max(140.0, page_height * 0.22):
        return True
    vertical_gap = 0.0
    if caption[1] >= bbox[3]:
        vertical_gap = caption[1] - bbox[3]
    elif bbox[1] >= caption[3]:
        vertical_gap = bbox[1] - caption[3]
    horizontal_overlap = max(0.0, min(bbox[2], caption[2]) - max(bbox[0], caption[0]))
    horizontal_span = max(1.0, min(bbox[2] - bbox[0], caption[2] - caption[0]))
    return (
        vertical_gap <= max(40.0, min(page_width, page_height) * 0.06)
        and horizontal_overlap / horizontal_span >= 0.25
    )


def visual_bboxes_near(left: list[float], right: list[float], *, gap: float) -> bool:
    merged = union_visual_bbox(left, right)
    max_area = max(visual_bbox_area(left), visual_bbox_area(right))
    if max_area and visual_bbox_area(merged) > max_area * 4.5:
        return False
    return not (
        left[2] + gap < right[0]
        or right[2] + gap < left[0]
        or left[3] + gap < right[1]
        or right[3] + gap < left[1]
    )


def expand_visual_bbox(
    bbox: list[float], page_width: float, page_height: float
) -> list[float] | None:
    margin = 10.0
    expanded = [
        max(0.0, bbox[0] - margin),
        max(0.0, bbox[1] - margin),
        min(page_width, bbox[2] + margin),
        min(page_height, bbox[3] + margin),
    ]
    return expanded if expanded[2] > expanded[0] and expanded[3] > expanded[1] else None


def visual_bbox_is_reportable(bbox: list[float], page_area: float) -> bool:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    if width < 36 or height < 36:
        return False
    ratio = visual_bbox_area(bbox) / page_area if page_area else 1.0
    return 0.002 <= ratio <= 0.65


def visual_bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def union_visual_bbox(left: list[float], right: list[float]) -> list[float]:
    return [
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    ]


def visual_bbox_distance(left: list[float], right: list[float]) -> float:
    left_center_x = (left[0] + left[2]) / 2
    right_center_x = (right[0] + right[2]) / 2
    horizontal = abs(left_center_x - right_center_x)
    if right[1] >= left[3]:
        vertical = right[1] - left[3]
    elif left[1] >= right[3]:
        vertical = left[1] - right[3]
    else:
        vertical = 0.0
    return vertical * 4 + horizontal


def visual_reader_reason(reason: str, *, output_language: str = "zh") -> str:
    cleaned = clean_model_inline_text(reason)
    if not cleaned:
        if output_language == "en":
            return "This visual helps ground the paper's key mechanism or evidence."
        return "看这页可以辅助理解论文的关键机制或证据。"
    if re.match(r"^(Figure|Fig\.?|Table)\s+\w+", cleaned, flags=re.IGNORECASE):
        return visual_caption_to_reader_reason(cleaned, output_language=output_language)
    return cleaned


def visual_caption_to_reader_reason(caption: str, *, output_language: str = "zh") -> str:
    cleaned = clean_model_inline_text(caption)
    lowered = cleaned.lower()
    if "architecture" in lowered or "overview" in lowered or "system" in lowered:
        if output_language == "en":
            return "This visual is useful for understanding the system structure and component relationships."
        return "这张图适合用来建立系统结构和组件关系的直觉。"
    if (
        "evaluation" in lowered
        or "result" in lowered
        or "throughput" in lowered
        or "latency" in lowered
    ):
        if output_language == "en":
            return "This visual is useful for checking metrics, baselines, and the boundary of the results."
        return "这张图适合用来核对实验指标、基线对比和结论边界。"
    if "algorithm" in lowered or "example" in lowered or "illustration" in lowered:
        if output_language == "en":
            return "This visual is useful for seeing how the mechanism works step by step."
        return "这张图适合用来理解论文机制如何一步步工作。"
    if output_language == "en":
        return "This visual helps ground the paper's key mechanism or evidence."
    return "这张图适合辅助理解论文的关键机制或证据。"


def select_key_visual_pages(
    *,
    paper: PaperRecord,
    model_report: dict[str, Any],
    card: PaperCard | None,
    layout: dict[str, Any],
    limit: int,
    output_language: str = "zh",
) -> list[dict[str, Any]]:
    pages_by_no = {
        page.get("page_no"): page
        for page in layout.get("pages", [])
        if isinstance(page, dict) and isinstance(page.get("page_no"), int)
    }
    selected: list[dict[str, Any]] = []

    def visual_reason(page_no: int, fallback: str) -> str:
        page = pages_by_no.get(page_no) or {}
        captions = page.get("captions") if isinstance(page.get("captions"), list) else []
        figures = page.get("figures") if isinstance(page.get("figures"), list) else []
        tables = page.get("tables") if isinstance(page.get("tables"), list) else []
        visual_notes = (
            page.get("visual_notes") if isinstance(page.get("visual_notes"), list) else []
        )
        for item in captions + figures + tables + visual_notes:
            if not isinstance(item, dict):
                continue
            text = clean_model_inline_text(
                item.get("text")
                or item.get("caption")
                or item.get("visual_summary")
                or item.get("summary")
            )
            if text:
                return visual_caption_to_reader_reason(text, output_language=output_language)
        return fallback

    def add(page_no: Any, reason: str, *, prefer_reason: bool = False) -> None:
        if not isinstance(page_no, int) or page_no <= 0:
            return
        if any(item["page_no"] == page_no for item in selected):
            return
        selected_reason = (
            compact_reason(clean_model_inline_text(reason), max_chars=180) if prefer_reason else ""
        )
        selected.append(
            {"page_no": page_no, "reason": selected_reason or visual_reason(page_no, reason)}
        )

    for item in model_report.get("key_visual_pages") or []:
        if not isinstance(item, dict):
            continue
        add(
            item.get("page_no"),
            clean_model_inline_text(item.get("reason"))
            or (
                "PaperLens selected this page as useful visual evidence."
                if output_language == "en"
                else "模型认为这页有助于理解论文。"
            ),
            prefer_reason=True,
        )
        if len(selected) >= limit:
            return selected[:limit]

    if card:
        for ref in card.evidence_refs:
            if ref.figure_id or ref.table_id or ref.bbox:
                add(
                    ref.page_no,
                    "This page contains figure/table evidence that supports the capsule."
                    if output_language == "en"
                    else "这页包含支撑正文理解的图表或版面证据。",
                )
            if len(selected) >= limit:
                return selected[:limit]

    scored: list[tuple[int, int]] = []
    for page_no, page in pages_by_no.items():
        score = 0
        if page.get("figures"):
            score += 3
        if page.get("tables"):
            score += 3
        if page.get("captions"):
            score += 2
        if page.get("visual_notes"):
            score += 2
        text = normalize_for_search(str(page.get("text") or ""))
        if any(
            term in text
            for term in [
                "figure",
                "fig.",
                "table",
                "overview",
                "architecture",
                "evaluation",
                "result",
            ]
        ):
            score += 1
        if score:
            scored.append((score, page_no))
    scored.sort(key=lambda item: (-item[0], item[1]))
    for _score, page_no in scored:
        add(
            page_no,
            "This page contains visual or tabular evidence worth viewing with the capsule."
            if output_language == "en"
            else "这页包含图表、表格或关键版面信息，适合和正文一起查看。",
        )
        if len(selected) >= limit:
            break
    return selected[:limit]


def body_starts_with_reason(body: str, reason: str) -> bool:
    body_key = compact_compare_text(body)
    reason_key = compact_compare_text(reason)
    if not body_key or not reason_key:
        return False
    prefix_len = min(max(40, len(reason_key) // 2), len(reason_key))
    return body_key.startswith(reason_key[:prefix_len])


def display_review_status(
    model_report: dict[str, Any],
    report_audit: dict[str, Any] | None,
    output_language: str = "zh",
) -> str:
    verdict = string_or_none(report_audit.get("verdict")) if report_audit else None
    if output_language == "en":
        if verdict == "PASS":
            return "reviewed"
        if verdict == "PASS_WITH_WEAKNESSES":
            return "reviewed with evidence boundaries"
        if verdict == "NEED_HUMAN_REVIEW":
            return "needs review"
        raw = string_or_none(model_report.get("review_status"))
        if raw == "格式归一化":
            return "normalized"
        return raw or "not reviewed"
    if verdict == "PASS":
        return "已复核"
    if verdict == "PASS_WITH_WEAKNESSES":
        return "已复核（有证据边界）"
    if verdict == "NEED_HUMAN_REVIEW":
        return "需复查"
    raw = string_or_none(model_report.get("review_status"))
    if raw == "格式归一化":
        return "已归一化"
    return raw or "未复核"


def render_debug_main_report(
    *,
    rows: list[dict[str, Any]],
    review_items: list[ReviewItem],
    budget: dict[str, Any],
    counts: dict[str, int],
) -> str:
    lines = [
        "# PaperLens Offline Debug Diagnostic",
        "",
        "> This is not a formal PaperLens reading result. It was generated without model reading, visual interpretation, or final report generation.",
        "",
        "## Purpose",
        "",
        "- Verify PDF scanning, parsing, page rendering, layout indexing, EvidenceRef plumbing, SQLite state, and portable sidecar execution.",
        "- Do not use this output for citation, novelty assessment, related-work writing, or reading decisions.",
        "",
        "## Run Summary",
        "",
        f"- Papers parsed: {len(rows)}",
        f"- Initial classification counts: A={counts['A']}, B={counts['B']}, C={counts['C']}, HOLD={counts['HOLD']}",
        f"- Model calls: {budget.get('calls', 0)}",
        "",
        "## Paper Diagnostics",
        "",
        "| Paper | Parse quality | Diagnostic report |",
        "|---|---|---|",
    ]
    for row in rows:
        paper = row["paper"]
        lines.append(
            f"| {paper.paper_id} | {paper.parse_quality or 'unknown'} | [open](papers/{row['report_name']}) |"
        )
    lines.extend(["", "## Review Items", ""])
    if review_items:
        for item in review_items:
            lines.append(f"- `{item.paper_id or 'run'}` {item.item_type}: {item.reason}")
    else:
        lines.append("- No review items were generated.")
    return "\n".join(lines) + "\n"


def render_debug_paper_diagnostic(
    *,
    paper: PaperRecord,
    skim: SkimCard | None,
    decision: ClassificationDecision | None,
    card: PaperCard | None,
    layout: dict[str, Any],
) -> str:
    pages = layout.get("pages") if isinstance(layout.get("pages"), list) else []
    metrics = layout.get("metrics") if isinstance(layout.get("metrics"), dict) else {}
    refs = dedupe_evidence_refs(
        (skim.evidence_refs if skim else []) + (card.evidence_refs if card else [])
    )
    metric_bits = [f"{key}={value}" for key, value in sorted(metrics.items())[:6]]
    metrics_summary = "；".join(metric_bits) if metric_bits else "未记录解析指标"
    lines = [
        f"# {paper.canonical_title or paper.paper_id}",
        "",
        "离线调试模式只检查 PDF 解析、页面渲染和证据链路，不会生成论文价值判断或正式总结。",
        "",
        f"Paper ID：`{paper.paper_id}`；解析质量：`{paper.parse_quality or 'unknown'}`；页数：{len(pages) or paper.page_count}；初始等级：`{decision.class_label if decision else 'missing'}`。",
        "",
        f"解析信号：{metrics_summary}。",
        "",
        f"已连接的证据引用数：{len(refs)}。如果要生成真正的论文讲解，请用模型 provider 运行正式流程。",
    ]
    return "\n".join(lines) + "\n"


def write_html_report(path: Path, markdown: str, title: str) -> None:
    body = markdown_to_html(markdown)
    path.write_text(
        "<!doctype html>\n"
        '<html><head><meta charset="utf-8">'
        f"<title>{html.escape(title)}</title>"
        "<style>"
        "body{font-family:Segoe UI,Arial,sans-serif;line-height:1.55;margin:32px;max-width:1180px;color:#172026}"
        "h1,h2,h3{line-height:1.25} table{border-collapse:collapse;width:100%;margin:12px 0}"
        "th,td{border:1px solid #d8dee4;padding:6px 8px;vertical-align:top} th{background:#f4f6f8}"
        "code{background:#f1f3f5;padding:1px 4px;border-radius:4px} a{color:#0969da}"
        "</style></head><body>" + body + "</body></html>\n",
        encoding="utf-8",
    )


def markdown_to_html(markdown: str) -> str:
    lines = []
    in_list = False
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped:
            if in_list:
                lines.append("</ul>")
                in_list = False
            continue
        if stripped.startswith("# "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h1>{html.escape(stripped[2:])}</h1>")
        elif stripped.startswith("## "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h2>{html.escape(stripped[3:])}</h2>")
        elif stripped.startswith("### "):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<h3>{html.escape(stripped[4:])}</h3>")
        elif stripped.startswith("- "):
            if not in_list:
                lines.append("<ul>")
                in_list = True
            lines.append(f"<li>{inline_markdown_to_html(stripped[2:])}</li>")
        elif stripped.startswith("|"):
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<pre>{html.escape(stripped)}</pre>")
        else:
            if in_list:
                lines.append("</ul>")
                in_list = False
            lines.append(f"<p>{inline_markdown_to_html(stripped)}</p>")
    if in_list:
        lines.append("</ul>")
    return "\n".join(lines)


def inline_markdown_to_html(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    return escaped


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
