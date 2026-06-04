from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from paperlens_core.version import display_version


ROOT = Path(__file__).resolve().parents[1]


def project_version() -> str:
    return json.loads((ROOT / "package.json").read_text(encoding="utf-8"))["version"]


def test_runtime_version_comes_from_project_metadata() -> None:
    assert display_version() == f"paperlens-core {project_version()}"


def test_version_files_are_synchronized() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/sync_version.py", "check"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
