from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a Tauri v2 static updater manifest for a Windows NSIS release."
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--repo", required=True, help="GitHub repository, for example owner/name")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--notes", default="")
    parser.add_argument("--notes-file", type=Path)
    return parser.parse_args()


def find_single(pattern: str, artifact_dir: Path) -> Path:
    matches = sorted(artifact_dir.glob(pattern))
    if not matches:
        raise SystemExit(f"No {pattern} file found in {artifact_dir}")
    if len(matches) > 1:
        names = ", ".join(path.name for path in matches)
        raise SystemExit(f"Expected one {pattern} file in {artifact_dir}, found: {names}")
    return matches[0]


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir.resolve()
    installer_matches = sorted(
        artifact_dir.glob(f"PaperLens_{args.version}_*-setup.exe")
    )
    installer = (
        installer_matches[0]
        if len(installer_matches) == 1
        else find_single("*.exe", artifact_dir)
    )
    signature_path = Path(f"{installer}.sig")
    if not signature_path.exists():
        signature_path = find_single("*.sig", artifact_dir)

    download_url = (
        f"https://github.com/{args.repo}/releases/download/{args.tag}/{installer.name}"
    )
    windows_asset = {
        "signature": signature_path.read_text(encoding="utf-8").strip(),
        "url": download_url,
    }
    notes = args.notes
    if args.notes_file and args.notes_file.exists():
        notes = args.notes_file.read_text(encoding="utf-8").strip()
    manifest = {
        "version": args.version,
        "notes": notes or f"PaperLens {args.tag}",
        "pub_date": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "platforms": {
            "windows-x86_64": windows_asset,
            "windows-x86_64-nsis": windows_asset,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote updater manifest: {args.output}")


if __name__ == "__main__":
    main()
