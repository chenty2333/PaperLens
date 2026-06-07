from __future__ import annotations

import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from paperlens_core.storage import atomic_write_json, atomic_write_jsonl


EventCallback = Callable[[dict[str, Any]], None]


class EventWriter:
    def __init__(
        self,
        run_id: str,
        events_path: Path,
        errors_path: Path,
        callback: EventCallback | None = None,
    ) -> None:
        self.run_id = run_id
        self.events_path = events_path
        self.errors_path = errors_path
        self.callback = callback
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.errors_path.parent.mkdir(parents=True, exist_ok=True)
        self.events_path.touch(exist_ok=True)
        self.errors_path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def emit(
        self,
        event_type: str,
        *,
        stage: str | None = None,
        message: str | None = None,
        level: str = "info",
        progress: float | None = None,
        data: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "type": event_type,
            "run_id": self.run_id,
            "time": datetime.now(timezone.utc).isoformat(),
            "level": level,
            "stage": stage,
            "message": message,
            "progress": progress,
            "data": data or {},
        }
        line = json.dumps(event, ensure_ascii=False, default=str)
        with self._lock:
            try:
                print(line, flush=True)
            except OSError:
                pass
            with self.events_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
            if level in {"error", "critical"}:
                with self.errors_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        if self.callback:
            self.callback(event)

    def stage_started(self, stage: str, message: str) -> None:
        self.emit("stage_started", stage=stage, message=message)

    def stage_completed(self, stage: str, message: str, data: dict[str, Any] | None = None) -> None:
        self.emit("stage_completed", stage=stage, message=message, data=data)

    def error(self, stage: str, message: str, data: dict[str, Any] | None = None) -> None:
        self.emit("error", stage=stage, message=message, level="error", data=data)


def write_json(path: Path, payload: Any) -> None:
    atomic_write_json(path, payload)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_write_jsonl(path, rows)


def emit_fatal(message: str) -> None:
    payload = {
        "type": "fatal",
        "time": datetime.now(timezone.utc).isoformat(),
        "level": "critical",
        "message": message,
        "data": {},
    }
    print(json.dumps(payload, ensure_ascii=False), file=sys.stdout, flush=True)
