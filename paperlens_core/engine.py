from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable

from paperlens_core.config import CoreConfig, apply_agent_privacy_env, load_config
from paperlens_core.control import ControlState, start_control_listener
from paperlens_core.events import EventWriter
from paperlens_core.library import (
    answer_library_question,
    doctor_library,
    rebuild_library_from_output,
    rebuild_library_index,
    search_library,
)
from paperlens_core.memory_store import PaperMemoryStore
from paperlens_core.memory_v3 import inspect_paper_memory_v3
from paperlens_core.orchestrator import run_pipeline
from paperlens_core.protocol import (
    InspectMemoryRequest,
    LibraryQuestionRequest,
    LibraryBuildRequest,
    LibraryDoctorRequest,
    LibraryRebuildIndexRequest,
    LibrarySearchRequest,
    PaperQuestionRequest,
    RunRequest,
    RunResult,
)
from paperlens_core.qa import answer_question


class PaperLensEngine:
    """Stable Python API for PaperLens Core.

    CLI, desktop sidecars, and future service/plugin entry points should call
    this class instead of reaching into pipeline modules directly.
    """

    def load_core_config(
        self, *, config_path: Path | None, overrides: dict[str, Any]
    ) -> CoreConfig:
        config = load_config(config_path, overrides)
        apply_agent_privacy_env(config)
        return config

    def run_job(
        self,
        request: RunRequest,
        *,
        control: ControlState | None = None,
        event_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> RunResult:
        input_dir = request.input_dir.expanduser().resolve()
        output_dir = request.output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        run_id = request.run_id or f"run_{uuid.uuid4().hex[:12]}"
        events = EventWriter(
            run_id,
            output_dir / ".paperlens" / "data" / "events.jsonl",
            output_dir / ".paperlens" / "data" / "errors.jsonl",
            callback=event_callback,
        )
        control = control or ControlState()
        if request.use_stdin_control:
            start_control_listener(control)
        if not input_dir.exists():
            events.error("startup", f"Input directory does not exist: {input_dir}")
            return RunResult(status="error", data={"reason": "missing_input_dir"})
        config = self.load_core_config(
            config_path=request.config_path,
            overrides=request.config_overrides,
        )
        try:
            config.validate_agentic_run()
        except ValueError as exc:
            events.error("startup", str(exc))
            return RunResult(status="error", data={"reason": str(exc)})
        try:
            manifest = run_pipeline(
                input_dir=input_dir,
                output_dir=output_dir,
                config=config,
                events=events,
                control=control,
                from_stage=request.from_stage,
                only_stage=request.only_stage,
            )
        except Exception as exc:
            events.emit("fatal", level="critical", message=str(exc))
            return RunResult(status="error", data={"reason": str(exc)})
        return RunResult(status="ok" if manifest else "error", data={"manifest": manifest or {}})

    def answer_paper_question(self, request: PaperQuestionRequest) -> dict[str, Any]:
        config = self.load_core_config(
            config_path=request.config_path,
            overrides=request.config_overrides,
        )
        return answer_question(
            output_dir=request.output_dir.expanduser().resolve(),
            config=config,
            paper_id=request.paper_id,
            question=request.question,
        )

    def build_library(self, request: LibraryBuildRequest) -> dict[str, Any]:
        paths = rebuild_library_from_output(request.output_dir.expanduser().resolve())
        return {"written": [str(path) for path in paths]}

    def rebuild_library_index(self, request: LibraryRebuildIndexRequest) -> dict[str, Any]:
        path = rebuild_library_index(request.output_dir.expanduser().resolve())
        return {"written": str(path)}

    def doctor_library(self, request: LibraryDoctorRequest) -> dict[str, Any]:
        return doctor_library(request.output_dir.expanduser().resolve())

    def search_library(self, request: LibrarySearchRequest) -> dict[str, Any]:
        return search_library(
            output_dir=request.output_dir.expanduser().resolve(),
            query=request.query,
            limit=request.limit,
        )

    def answer_library_question(self, request: LibraryQuestionRequest) -> dict[str, Any]:
        config = self.load_core_config(
            config_path=request.config_path,
            overrides=request.config_overrides,
        )
        return answer_library_question(
            output_dir=request.output_dir.expanduser().resolve(),
            config=config,
            question=request.question,
            limit=request.limit,
        )

    def inspect_memory(self, request: InspectMemoryRequest) -> str:
        output_dir = request.output_dir.expanduser().resolve()
        if request.patches:
            data_dir = output_dir / ".paperlens" / "data"
            store = PaperMemoryStore(data_dir)
            paper_id = request.paper_id
            if not paper_id:
                candidates = sorted((data_dir / "memory" / "v3").glob("*.memory_patches.jsonl"))
                if not candidates:
                    raise FileNotFoundError("No PaperMemory patch logs found")
                paper_id = candidates[0].name.split(".", 1)[0]
            return store.patch_log_path(paper_id).read_text(encoding="utf-8")
        return inspect_paper_memory_v3(
            output_dir=output_dir,
            paper_id=request.paper_id,
            section=request.section,
            claim_id=request.claim_id,
        )

