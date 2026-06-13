from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from paperlens_core.storage import atomic_write_json


def hash_json_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()[:24]


def llm_cache_path(
    cache_dir: Path | None, stage: str, paper_id: str, key_payload: dict[str, Any]
) -> Path | None:
    if cache_dir is None:
        return None
    key = hashlib.sha256(_canonical_json(key_payload)).hexdigest()[:24]
    safe_stage = safe_cache_segment(stage)
    safe_paper = safe_cache_segment(paper_id)
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
    atomic_write_json(path, payload)


def safe_cache_segment(value: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value)).strip("_")
    return sanitized or "item"


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
