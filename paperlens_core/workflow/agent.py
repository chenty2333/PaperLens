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
from paperlens_core.workflow.export import run_export_stage
from paperlens_core.state import transition_state
from paperlens_core.workflow.manifest import run_manifest_stage
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
from paperlens_core.workflow.skim import run_skim_stage
from paperlens_core.workflow.core_v2 import (
    run_core_v2_audit_stage,
    run_core_v2_observation_stage,
)
from paperlens_core.workflow.visual import run_vlm_page_mode as run_visual_vlm_page_mode
from paperlens_core.workflow.utils import utc_timestamp


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
        run_skim_stage(self)

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
        run_core_v2_observation_stage(self)

    def stage_08_evidence_verify(self) -> None:
        run_core_v2_audit_stage(self)

    def stage_15_export(self) -> list[Path]:
        return run_export_stage(self)

    def stage_17_manifest(self) -> dict[str, Any]:
        return run_manifest_stage(self)
