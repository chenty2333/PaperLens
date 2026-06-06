from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any


def normalize_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def hash_text(value: str) -> str:
    return hashlib.sha256(normalize_for_search(value).encode("utf-8")).hexdigest()[:16]


def chunked(items: list[Any], size: int) -> list[list[Any]]:
    size = max(1, size)
    return [items[index : index + size] for index in range(0, len(items), size)]


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
