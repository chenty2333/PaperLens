from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARIES = ROOT / "src-tauri" / "binaries"


def host_tuple() -> str:
    return subprocess.check_output(["rustc", "--print", "host-tuple"], text=True).strip()


def main() -> int:
    BINARIES.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--clean",
            "--noconfirm",
            "--onefile",
            "--name",
            "paperlens-core",
            "--hidden-import",
            "agents",
            "--hidden-import",
            "agents.models.openai_provider",
            "--exclude-module",
            "agents.sandbox",
            "--exclude-module",
            "agents.sandbox.memory",
            "--exclude-module",
            "websockets",
            "--exclude-module",
            "websockets.asyncio",
            "--exclude-module",
            "websockets.sync",
            "--exclude-module",
            "uvicorn",
            "--exclude-module",
            "uvloop",
            "--exclude-module",
            "tzdata",
            "--copy-metadata",
            "openai-agents",
            "--copy-metadata",
            "openai",
            "paperlens_core/main.py",
        ],
        cwd=ROOT,
    )
    suffix = ".exe" if sys.platform == "win32" else ""
    source = ROOT / "dist" / f"paperlens-core{suffix}"
    target = BINARIES / f"paperlens-core-{host_tuple()}{suffix}"
    shutil.copy2(source, target)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
