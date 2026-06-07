from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable

from paperlens_core.config import CoreConfig, load_config
from paperlens_core.control import ControlState, start_control_listener
from paperlens_core.events import EventWriter
from paperlens_core.library import (
    answer_library_question,
    doctor_library,
    rebuild_library_from_output,
    rebuild_library_index,
    search_library,
)
from paperlens_core.orchestrator import run_pipeline
from paperlens_core.protocol import (
    LibraryBuildRequest,
    LibraryDoctorRequest,
    LibraryQuestionRequest,
    LibraryRebuildIndexRequest,
    LibrarySearchRequest,
    PaperQuestionRequest,
    RunRequest,
    RunResult,
    WorkspaceCleanupCacheRequest,
    WorkspaceExportRequest,
    WorkspaceImportRequest,
    WorkspaceRequest,
)
from paperlens_core.qa import answer_question
from paperlens_core.storage import WorkspaceStore
from paperlens_core.version import display_version


class PaperLensEngine:
    """Stable Python API for PaperLens Core.

    CLI, desktop sidecars, and future service/plugin entry points should call
    this class instead of reaching into pipeline modules directly.
    """

    def load_core_config(
        self, *, config_path: Path | None, overrides: dict[str, Any]
    ) -> CoreConfig:
        return load_config(config_path, overrides)

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
        WorkspaceStore(output_dir).bootstrap(app_version=display_version())
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
        WorkspaceStore(request.output_dir.expanduser().resolve()).bootstrap(
            app_version=display_version()
        )
        config = self.load_core_config(
            config_path=request.config_path,
            overrides=request.config_overrides,
        )
        return answer_question(
            output_dir=request.output_dir.expanduser().resolve(),
            config=config,
            paper_id=request.paper_id,
            question=request.question,
            chat_history=request.chat_history,
        )

    def build_library(self, request: LibraryBuildRequest) -> dict[str, Any]:
        WorkspaceStore(request.output_dir.expanduser().resolve()).bootstrap(
            app_version=display_version()
        )
        paths = rebuild_library_from_output(request.output_dir.expanduser().resolve())
        return {"written": [str(path) for path in paths]}

    def rebuild_library_index(self, request: LibraryRebuildIndexRequest) -> dict[str, Any]:
        WorkspaceStore(request.output_dir.expanduser().resolve()).bootstrap(
            app_version=display_version()
        )
        path = rebuild_library_index(request.output_dir.expanduser().resolve())
        return {"written": str(path)}

    def doctor_library(self, request: LibraryDoctorRequest) -> dict[str, Any]:
        WorkspaceStore(request.output_dir.expanduser().resolve()).bootstrap(
            app_version=display_version()
        )
        return doctor_library(request.output_dir.expanduser().resolve())

    def search_library(self, request: LibrarySearchRequest) -> dict[str, Any]:
        return search_library(
            output_dir=request.output_dir.expanduser().resolve(),
            query=request.query,
            limit=request.limit,
        )

    def answer_library_question(self, request: LibraryQuestionRequest) -> dict[str, Any]:
        WorkspaceStore(request.output_dir.expanduser().resolve()).bootstrap(
            app_version=display_version()
        )
        config = self.load_core_config(
            config_path=request.config_path,
            overrides=request.config_overrides,
        )
        return answer_library_question(
            output_dir=request.output_dir.expanduser().resolve(),
            config=config,
            question=request.question,
            limit=request.limit,
            chat_history=request.chat_history,
        )

    def migrate_workspace(self, request: WorkspaceRequest) -> dict[str, Any]:
        return WorkspaceStore(request.output_dir.expanduser().resolve()).bootstrap(
            app_version=display_version()
        )

    def doctor_workspace(self, request: WorkspaceRequest, *, repair: bool = False) -> dict[str, Any]:
        return WorkspaceStore(request.output_dir.expanduser().resolve()).doctor(repair=repair)

    def cleanup_workspace_cache(self, request: WorkspaceCleanupCacheRequest) -> dict[str, Any]:
        return WorkspaceStore(request.output_dir.expanduser().resolve()).cleanup_cache(
            max_age_days=request.max_age_days,
            dry_run=request.dry_run,
        )

    def export_workspace(self, request: WorkspaceExportRequest) -> dict[str, Any]:
        return WorkspaceStore(request.output_dir.expanduser().resolve()).export_archive(
            request.archive_path,
            include_cache=request.include_cache,
        )

    def import_workspace(self, request: WorkspaceImportRequest) -> dict[str, Any]:
        return WorkspaceStore(request.output_dir.expanduser().resolve()).import_archive(
            request.archive_path,
            replace=request.replace,
        )
