from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from paperlens_core._version import __version__ as BUNDLED_VERSION


PACKAGE_NAME = "paperlens-core"


def core_version() -> str:
    try:
        return version(PACKAGE_NAME)
    except PackageNotFoundError:
        return BUNDLED_VERSION or source_tree_version() or "0.0.0"


def display_version() -> str:
    return f"paperlens-core {core_version()}"


def source_tree_version() -> str | None:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if not pyproject.exists():
        return None
    match = re.search(
        r'(?m)^version\s*=\s*"([^"]+)"',
        pyproject.read_text(encoding="utf-8"),
    )
    return match.group(1) if match else None
