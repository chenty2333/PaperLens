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
    (ROOT / "build").mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--clean",
            "--noconfirm",
            "--specpath",
            str(ROOT / "build"),
            "--onefile",
            "--name",
            "paperlens-core",
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
            "paperlens_core/main.py",
        ],
        cwd=ROOT,
    )
    suffix = ".exe" if sys.platform == "win32" else ""
    source = ROOT / "dist" / f"paperlens-core{suffix}"
    target = BINARIES / f"paperlens-core-{host_tuple()}{suffix}"
    if not source.exists():
        raise SystemExit(f"PyInstaller did not produce {source}")
    if source.stat().st_size < 1_048_576:
        raise SystemExit(f"PyInstaller output is unexpectedly small: {source}")
    shutil.copy2(source, target)
    if target.stat().st_size < 1_048_576:
        raise SystemExit(f"Sidecar copy is unexpectedly small: {target}")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
