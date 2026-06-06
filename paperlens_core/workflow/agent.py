from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from paperlens_core.agents.llm import JsonLlmClient
from paperlens_core.agents.providers import describe_provider
from paperlens_core.budget import BudgetManager
from paperlens_core.config import CoreConfig
from paperlens_core.control import ControlState
from paperlens_core.db import ArtifactDb
from paperlens_core.events import EventWriter, write_json
from paperlens_core.quality_snapshot import write_core_quality_snapshot
from paperlens_core.report import (
    classification_counts,
    paper_report_filename,
)
from paperlens_core.runtime import (
    llm_cache_path,
    read_llm_cache,
    write_llm_cache,
)
from paperlens_core.schemas import (
    ArtifactVersion,
    ClassificationDecision,
    PaperRecord,
    SkimCard,
)
from paperlens_core.workflow.export import write_final_report_bundle
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
from paperlens_core.workflow.parse import (
    run_ingest_stage,
    run_parse_stage,
    run_parse_verify_stage,
)
from paperlens_core.workflow.skim import deterministic_skim_classify
from paperlens_core.workflow.core_v2 import (
    refresh_core_v2_audit_artifacts,
    run_core_v2_model_observation_tasks,
    write_core_v2_artifacts,
)
from paperlens_core.workflow.visual import run_vlm_page_mode as run_visual_vlm_page_mode
from paperlens_core.workflow.utils import (
    dict_value,
    load_layout_index,
    utc_timestamp,
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
        run_ingest_stage(self)

    def stage_01_parse(self) -> None:
        run_parse_stage(self)

    def stage_02_parse_verify(self) -> None:
        run_parse_verify_stage(self)

    def run_vlm_page_mode(
        self,
        *,
        client: JsonLlmClient,
        paper: PaperRecord,
        artifacts: list[Any],
        stage: str,
    ) -> dict[str, Any]:
        return run_visual_vlm_page_mode(
            self,
            client=client,
            paper=paper,
            artifacts=artifacts,
            stage=stage,
        )

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
