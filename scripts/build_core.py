from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BINARIES = ROOT / "src-tauri" / "binaries"
BUILD_ROOT = ROOT / "build"
PYINSTALLER_DIST = BUILD_ROOT / "pyinstaller-dist"
PYINSTALLER_WORK = BUILD_ROOT / "pyinstaller-work"


def host_tuple() -> str:
    return subprocess.check_output(["rustc", "--print", "host-tuple"], text=True).strip()


def main() -> int:
    BINARIES.mkdir(parents=True, exist_ok=True)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "PyInstaller",
            "--clean",
            "--noconfirm",
            "--specpath",
            str(BUILD_ROOT),
            "--distpath",
            str(PYINSTALLER_DIST),
            "--workpath",
            str(PYINSTALLER_WORK),
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
    source = PYINSTALLER_DIST / f"paperlens-core{suffix}"
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
