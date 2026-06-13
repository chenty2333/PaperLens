from __future__ import annotations

import json
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from paperlens_core.storage import atomic_write_json, atomic_write_jsonl


EventCallback = Callable[[dict[str, Any]], None]
SENSITIVE_KEY_NAMES = {
    "apikey",
    "api_key",
    "accesstoken",
    "authorization",
    "bearer",
    "githubtoken",
    "password",
    "refreshtoken",
    "secret",
    "token",
    "xapikey",
    "x_api_key",
}
SENSITIVE_TEXT_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"(?i)\b(x-api-key|api-key|authorization)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"\b(sk|tp)-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bghp_[0-9A-Za-z_]{20,}\b"),
    re.compile(r"\bgithub_pat_[0-9A-Za-z_]+\b"),
)


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
        event = redact_event(
            {
                "type": event_type,
                "run_id": self.run_id,
                "time": datetime.now(timezone.utc).isoformat(),
                "level": level,
                "stage": stage,
                "message": message,
                "progress": progress,
                "data": data or {},
            }
        )
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
    payload = redact_event(
        {
            "type": "fatal",
            "time": datetime.now(timezone.utc).isoformat(),
            "level": "critical",
            "message": message,
            "data": {},
        }
    )
    print(json.dumps(payload, ensure_ascii=False), file=sys.stdout, flush=True)


def redact_event(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            normalized = normalize_sensitive_key(key)
            redacted[key] = "***" if normalized in SENSITIVE_KEY_NAMES else redact_event(item)
        return redacted
    if isinstance(value, list):
        return [redact_event(item) for item in value]
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value


def normalize_sensitive_key(key: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def redact_sensitive_text(text: str) -> str:
    redacted = text
    for pattern in SENSITIVE_TEXT_PATTERNS:
        redacted = pattern.sub(lambda match: redacted_token(match.group(0)), redacted)
    return redacted


def redacted_token(value: str) -> str:
    prefix = value.split(" ", 1)[0] if value.lower().startswith("bearer ") else ""
    return f"{prefix} ***" if prefix else "***"
