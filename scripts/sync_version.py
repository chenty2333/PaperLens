from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$")


VERSION_FILES = {
    "package_json": ROOT / "package.json",
    "package_lock": ROOT / "package-lock.json",
    "pyproject": ROOT / "pyproject.toml",
    "uv_lock": ROOT / "uv.lock",
    "tauri_conf": ROOT / "src-tauri" / "tauri.conf.json",
    "cargo_toml": ROOT / "src-tauri" / "Cargo.toml",
    "cargo_lock": ROOT / "src-tauri" / "Cargo.lock",
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sync_version.py",
        description="Keep PaperLens package, Python, Tauri, Rust, and lockfile versions in sync.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check", help="Fail if any PaperLens version field is out of sync")
    subparsers.add_parser("sync", help="Rewrite all PaperLens version fields from package.json")
    bump = subparsers.add_parser("bump", help="Bump package.json and sync all derived versions")
    bump.add_argument("part", choices=["patch", "minor", "major"], help="Version segment to bump")
    args = parser.parse_args(argv)

    if args.command == "check":
        return check_versions()
    if args.command == "sync":
        return sync_versions(read_package_version())
    if args.command == "bump":
        return sync_versions(bumped_version(read_package_version(), args.part))
    raise AssertionError(args.command)


def check_versions() -> int:
    versions = collect_versions()
    expected = versions["package.json"]
    mismatches = {name: value for name, value in versions.items() if value != expected}
    hardcoded = find_hardcoded_core_versions()
    if mismatches or hardcoded:
        print(f"Expected PaperLens version: {expected}")
        for name, value in sorted(mismatches.items()):
            print(f"mismatch: {name}={value}")
        for item in hardcoded:
            print(f"hardcoded runtime version: {item}")
        return 1
    print(f"PaperLens version is synchronized: {expected}")
    return 0


def sync_versions(version: str) -> int:
    validate_version(version)
    write_package_json(version)
    write_package_lock(version)
    replace_project_version(VERSION_FILES["pyproject"], version)
    replace_lock_package_version(VERSION_FILES["uv_lock"], "paperlens-core", version)
    write_tauri_conf(version)
    replace_project_version(VERSION_FILES["cargo_toml"], version)
    replace_lock_package_version(VERSION_FILES["cargo_lock"], "paperlens", version)
    return check_versions()


def collect_versions() -> dict[str, str]:
    package = read_json(VERSION_FILES["package_json"])
    package_lock = read_json(VERSION_FILES["package_lock"])
    tauri_conf = read_json(VERSION_FILES["tauri_conf"])
    packages = package_lock.get("packages") if isinstance(package_lock.get("packages"), dict) else {}
    root_lock = packages.get("") if isinstance(packages.get(""), dict) else {}
    return {
        "package.json": string_value(package.get("version")),
        "package-lock.json": string_value(package_lock.get("version")),
        "package-lock packages.root": string_value(root_lock.get("version")),
        "pyproject.toml": read_project_version(VERSION_FILES["pyproject"]),
        "uv.lock paperlens-core": read_lock_package_version(
            VERSION_FILES["uv_lock"], "paperlens-core"
        ),
        "tauri.conf.json": string_value(tauri_conf.get("version")),
        "src-tauri/Cargo.toml": read_project_version(VERSION_FILES["cargo_toml"]),
        "src-tauri/Cargo.lock paperlens": read_lock_package_version(
            VERSION_FILES["cargo_lock"], "paperlens"
        ),
    }


def read_package_version() -> str:
    version = string_value(read_json(VERSION_FILES["package_json"]).get("version"))
    validate_version(version)
    return version


def bumped_version(version: str, part: str) -> str:
    validate_version(version)
    major, minor, patch = [int(item) for item in version.split(".")]
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"


def validate_version(version: str) -> None:
    if not SEMVER_RE.fullmatch(version):
        raise SystemExit(f"Unsupported PaperLens version: {version!r}")


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_package_json(version: str) -> None:
    data = read_json(VERSION_FILES["package_json"])
    data["version"] = version
    write_json(VERSION_FILES["package_json"], data)


def write_package_lock(version: str) -> None:
    data = read_json(VERSION_FILES["package_lock"])
    data["version"] = version
    packages = data.get("packages")
    if isinstance(packages, dict) and isinstance(packages.get(""), dict):
        packages[""]["version"] = version
    write_json(VERSION_FILES["package_lock"], data)


def write_tauri_conf(version: str) -> None:
    data = read_json(VERSION_FILES["tauri_conf"])
    data["version"] = version
    write_json(VERSION_FILES["tauri_conf"], data)


def read_project_version(path: Path) -> str:
    match = re.search(r'(?m)^version\s*=\s*"([^"]+)"', path.read_text(encoding="utf-8"))
    return match.group(1) if match else ""


def replace_project_version(path: Path, version: str) -> None:
    replace_once(
        path,
        r'(?m)^version\s*=\s*"([^"]+)"',
        lambda _match: f'version = "{version}"',
    )


def read_lock_package_version(path: Path, package_name: str) -> str:
    match = lock_package_pattern(package_name).search(path.read_text(encoding="utf-8"))
    return match.group("version") if match else ""


def replace_lock_package_version(path: Path, package_name: str, version: str) -> None:
    pattern = lock_package_pattern(package_name)
    replace_once(path, pattern, lambda match: f'{match.group("head")}{version}{match.group("tail")}')


def lock_package_pattern(package_name: str) -> re.Pattern[str]:
    escaped = re.escape(package_name)
    return re.compile(
        rf'(?P<head>\[\[package\]\]\s+name = "{escaped}"\s+version = ")'
        rf'(?P<version>[^"]+)'
        rf'(?P<tail>")',
        re.MULTILINE,
    )


def replace_once(path: Path, pattern: str | re.Pattern[str], repl: str | Callable[[re.Match[str]], str]) -> None:
    text = path.read_text(encoding="utf-8")
    updated, count = re.subn(pattern, repl, text, count=1)
    if count != 1:
        raise SystemExit(f"Could not update version in {path.relative_to(ROOT)}")
    path.write_text(updated, encoding="utf-8")


def find_hardcoded_core_versions() -> list[str]:
    pattern = re.compile(r"paperlens-core\s+\d+\.\d+\.\d+")
    hits: list[str] = []
    for path in sorted((ROOT / "paperlens_core").rglob("*.py")):
        if path.name == "version.py":
            continue
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if pattern.search(line):
                hits.append(f"{path.relative_to(ROOT)}:{line_no}")
    return hits


def string_value(value: object) -> str:
    return value if isinstance(value, str) else ""


if __name__ == "__main__":
    sys.exit(main())
