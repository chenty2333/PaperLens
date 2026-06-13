from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from stat import S_ISLNK
from typing import Any, Iterable


INTERNAL_DIRNAME = ".paperlens"
WORKSPACE_SCHEMA_VERSION = "paperlens.workspace.v1"
STORAGE_SCHEMA_VERSION = 1
EXPORT_SCHEMA_VERSION = "paperlens.workspace_export.v1"
MAX_IMPORT_MANIFEST_BYTES = 1_000_000
MAX_IMPORT_FILE_COUNT = 200_000
MAX_IMPORT_MEMBER_BYTES = 2 * 1024 * 1024 * 1024
MAX_IMPORT_UNCOMPRESSED_BYTES = 50 * 1024 * 1024 * 1024
_JSONL_APPEND_LOCKS: dict[Path, threading.Lock] = {}
_JSONL_APPEND_LOCKS_GUARD = threading.Lock()

REQUIRED_DIRS = (
    INTERNAL_DIRNAME,
    f"{INTERNAL_DIRNAME}/pages",
    f"{INTERNAL_DIRNAME}/figures",
    f"{INTERNAL_DIRNAME}/data",
    f"{INTERNAL_DIRNAME}/data/artifacts/layout",
    f"{INTERNAL_DIRNAME}/data/core/v2",
    f"{INTERNAL_DIRNAME}/library",
    f"{INTERNAL_DIRNAME}/library/index",
    f"{INTERNAL_DIRNAME}/cache",
    "papers",
)

MANAGED_DIR_RELATIVE_PATHS = (
    INTERNAL_DIRNAME,
)
MANAGED_FILE_RELATIVE_PATHS = (
    "PaperLens.md",
    "PaperLens.json",
    "PaperLens_Library.md",
    "PaperLens_Library.json",
)
MANAGED_RELATIVE_PATHS = (*MANAGED_DIR_RELATIVE_PATHS, *MANAGED_FILE_RELATIVE_PATHS)
MANAGED_REPORTS_DIR = "papers"

CRITICAL_JSON_RELATIVE_PATHS = (
    f"{INTERNAL_DIRNAME}/workspace.json",
    f"{INTERNAL_DIRNAME}/data/run.json",
    f"{INTERNAL_DIRNAME}/data/model_call_summary.json",
    f"{INTERNAL_DIRNAME}/data/core_quality_snapshot.v2.json",
    f"{INTERNAL_DIRNAME}/library/index/search_index.json",
    f"{INTERNAL_DIRNAME}/library/index/cross_paper_relations.json",
)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding=encoding,
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            tmp_name = handle.name
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if tmp_name:
            try:
                Path(tmp_name).unlink(missing_ok=True)
            except OSError:
                pass


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows)
    atomic_write_text(path, text, encoding="utf-8")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    with _JSONL_APPEND_LOCKS_GUARD:
        lock = _JSONL_APPEND_LOCKS.setdefault(resolved, threading.Lock())
    line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
    with lock, path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_json_file(path: Path, fallback: Any = None) -> Any:
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


class WorkspaceStore:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir.expanduser().resolve()
        self.internal_dir = self.output_dir / INTERNAL_DIRNAME
        self.data_dir = self.internal_dir / "data"
        self.cache_dir = self.internal_dir / "cache"
        self.library_dir = self.internal_dir / "library"
        self.manifest_path = self.internal_dir / "workspace.json"
        self.migration_log_path = self.data_dir / "migrations.jsonl"
        self.state_db_path = self.internal_dir / "state.sqlite"
        self.recovery_dir = self.internal_dir / "recovery"

    def bootstrap(self, *, app_version: str | None = None) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_layout()
        migration = self.migrate(app_version=app_version)
        manifest = self.read_manifest(recover=True) or self.new_manifest(
            app_version=app_version,
            origin="new",
        )
        manifest["updated_at"] = utc_now()
        manifest["app_version"] = app_version or manifest.get("app_version") or ""
        manifest["layout"] = self.workspace_layout()
        manifest["last_bootstrap"] = {
            "time": utc_now(),
            "migration_status": migration.get("status"),
        }
        self.write_manifest(manifest)
        return migration

    def ensure_layout(self) -> None:
        for relative in REQUIRED_DIRS:
            (self.output_dir / relative).mkdir(parents=True, exist_ok=True)

    def migrate(self, *, app_version: str | None = None) -> dict[str, Any]:
        self.ensure_layout()
        actions: list[str] = []
        repaired: list[str] = []
        manifest = self.read_manifest(recover=True)
        if not manifest:
            origin = "existing" if self.has_existing_workspace_content() else "new"
            manifest = self.new_manifest(app_version=app_version, origin=origin)
            self.write_manifest(manifest)
            self.append_migration(
                {
                    "from_storage_schema_version": 0,
                    "to_storage_schema_version": STORAGE_SCHEMA_VERSION,
                    "action": "create_workspace_manifest",
                    "origin": origin,
                }
            )
            actions.append("create_workspace_manifest")
        else:
            version = int_or_zero(manifest.get("storage_schema_version"))
            if version > STORAGE_SCHEMA_VERSION:
                raise RuntimeError(
                    "This PaperLens build cannot open a newer workspace schema: "
                    f"{version} > {STORAGE_SCHEMA_VERSION}"
                )
            if version < STORAGE_SCHEMA_VERSION:
                manifest["storage_schema_version"] = STORAGE_SCHEMA_VERSION
                manifest["schema_version"] = WORKSPACE_SCHEMA_VERSION
                self.write_manifest(manifest)
                self.append_migration(
                    {
                        "from_storage_schema_version": version,
                        "to_storage_schema_version": STORAGE_SCHEMA_VERSION,
                        "action": "upgrade_workspace_manifest",
                    }
                )
                actions.append("upgrade_workspace_manifest")
        for path in self.critical_json_paths():
            if path == self.manifest_path or not path.exists():
                continue
            if self.json_error(path):
                recovered = self.recover_corrupt_file(path)
                repaired.append(str(recovered.relative_to(self.output_dir)))
        return {
            "status": "ok",
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "storage_schema_version": STORAGE_SCHEMA_VERSION,
            "actions": actions,
            "repaired": repaired,
            "manifest": str(self.manifest_path),
        }

    def new_manifest(self, *, app_version: str | None, origin: str) -> dict[str, Any]:
        now = utc_now()
        return {
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "storage_schema_version": STORAGE_SCHEMA_VERSION,
            "created_at": now,
            "updated_at": now,
            "app_version": app_version or "",
            "origin": origin,
            "layout": self.workspace_layout(),
        }

    def workspace_layout(self) -> dict[str, str]:
        return {
            "root": str(self.output_dir),
            "internal": INTERNAL_DIRNAME,
            "state_db": f"{INTERNAL_DIRNAME}/state.sqlite",
            "data": f"{INTERNAL_DIRNAME}/data",
            "library": f"{INTERNAL_DIRNAME}/library",
            "cache": f"{INTERNAL_DIRNAME}/cache",
            "reports": "papers",
            "main_report": "PaperLens.md",
        }

    def read_manifest(self, *, recover: bool) -> dict[str, Any] | None:
        if not self.manifest_path.exists():
            return None
        try:
            value = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            if recover:
                self.recover_corrupt_file(self.manifest_path)
            return None
        return value if isinstance(value, dict) else None

    def write_manifest(self, manifest: dict[str, Any]) -> None:
        atomic_write_json(self.manifest_path, manifest)

    def append_migration(self, payload: dict[str, Any]) -> None:
        append_jsonl(
            self.migration_log_path,
            {
                "time": utc_now(),
                "schema_version": WORKSPACE_SCHEMA_VERSION,
                **payload,
            },
        )

    def has_existing_workspace_content(self) -> bool:
        return self.has_persisted_workspace_content() or self.state_db_path.exists()

    def has_persisted_workspace_content(self) -> bool:
        if any(
            path.exists()
            for path in [
                self.data_dir / "run.json",
                self.library_dir / "library_records.jsonl",
                self.output_dir / "PaperLens.md",
            ]
        ):
            return True
        papers_dir = self.output_dir / "papers"
        return papers_dir.exists() and any(
            is_managed_report_file(path) for path in papers_dir.glob("*.md")
        )

    def doctor(self, *, repair: bool = False) -> dict[str, Any]:
        repairs: list[str] = []
        issues: list[str] = []
        if repair:
            migration = self.bootstrap()
            repairs.extend(migration.get("actions") or [])
            repairs.extend(migration.get("repaired") or [])
        else:
            manifest = self.read_manifest(recover=False)
            if not self.manifest_path.exists():
                issues.append("missing_workspace_manifest")
            elif not manifest:
                issues.append("invalid_workspace_manifest")
            elif manifest.get("schema_version") != WORKSPACE_SCHEMA_VERSION:
                issues.append("unsupported_workspace_manifest_schema")
            elif int_or_zero(manifest.get("storage_schema_version")) > STORAGE_SCHEMA_VERSION:
                issues.append("newer_workspace_schema")

        for relative in REQUIRED_DIRS:
            path = self.output_dir / relative
            if not path.exists():
                if repair:
                    path.mkdir(parents=True, exist_ok=True)
                    repairs.append(f"created:{relative}")
                else:
                    issues.append(f"missing_dir:{relative}")

        for path in self.critical_json_paths():
            if not path.exists():
                continue
            error = self.json_error(path)
            if not error:
                continue
            relative = str(path.relative_to(self.output_dir)).replace("\\", "/")
            if repair:
                recovered = self.recover_corrupt_file(path)
                repairs.append(f"recovered_corrupt_json:{relative}->{recovered.name}")
            else:
                issues.append(f"invalid_json:{relative}:{error}")

        db_status = self.sqlite_status()
        if db_status != "ok" and (
            db_status != "missing" or self.has_persisted_workspace_content()
        ):
            issues.append(f"state_db:{db_status}")

        library_status = self.library_records_status()
        if library_status["invalid_lines"]:
            issues.append(f"library_records_invalid_lines:{library_status['invalid_lines']}")

        stats = self.stats()
        status = "PASS" if not issues else "FAIL" if any(is_fail_issue(i) for i in issues) else "WARN"
        if repairs and status == "PASS":
            status = "REPAIRED"
        return {
            "status": status,
            "schema_version": WORKSPACE_SCHEMA_VERSION,
            "storage_schema_version": STORAGE_SCHEMA_VERSION,
            "output_dir": str(self.output_dir),
            "manifest": str(self.manifest_path),
            "issues": issues,
            "repairs": repairs,
            "stats": stats,
            "library": library_status,
            "state_db": db_status,
        }

    def critical_json_paths(self) -> list[Path]:
        paths = [self.output_dir / relative for relative in CRITICAL_JSON_RELATIVE_PATHS]
        core_root = self.data_dir / "core" / "v2"
        if core_root.exists():
            paths.extend(sorted(core_root.glob("*/*.json")))
        return paths

    def json_error(self, path: Path) -> str:
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            return exc.__class__.__name__
        except json.JSONDecodeError as exc:
            return f"line_{exc.lineno}_column_{exc.colno}"
        return ""

    def recover_corrupt_file(self, path: Path) -> Path:
        relative = safe_relative_to(path.resolve(), self.output_dir)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        target = self.recovery_dir / stamp / (str(relative).replace("\\", "__").replace("/", "__") + ".corrupt")
        target = unique_sibling_path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
        self.append_migration(
            {
                "action": "recover_corrupt_file",
                "source": str(relative).replace("\\", "/"),
                "target": str(target.relative_to(self.output_dir)).replace("\\", "/"),
            }
        )
        return target

    def sqlite_status(self) -> str:
        if not self.state_db_path.exists():
            return "missing"
        try:
            conn = sqlite3.connect(f"file:{self.state_db_path}?mode=ro", uri=True)
            try:
                row = conn.execute("PRAGMA quick_check").fetchone()
                return "ok" if row and row[0] == "ok" else str(row[0] if row else "unknown")
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return exc.__class__.__name__

    def library_records_status(self) -> dict[str, Any]:
        path = self.library_dir / "library_records.jsonl"
        if not path.exists():
            return {"exists": False, "records": 0, "invalid_lines": 0}
        records = 0
        invalid = 0
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return {"exists": True, "records": 0, "invalid_lines": 1}
        for line in lines:
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                invalid += 1
                continue
            if isinstance(value, dict):
                records += 1
            else:
                invalid += 1
        return {"exists": True, "records": records, "invalid_lines": invalid}

    def cleanup_cache(self, *, max_age_days: int = 30, dry_run: bool = False) -> dict[str, Any]:
        self.ensure_layout()
        cutoff = time.time() - max(0, max_age_days) * 86400
        removed: list[str] = []
        errors: list[str] = []
        bytes_selected = 0
        for path in sorted(self.cache_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = str(path.relative_to(self.output_dir)).replace("\\", "/")
            try:
                stat = path.stat()
            except OSError as exc:
                errors.append(f"{relative}: {exc}")
                continue
            if max_age_days > 0 and stat.st_mtime >= cutoff:
                continue
            bytes_selected += stat.st_size
            if dry_run:
                removed.append(relative)
                continue
            try:
                path.unlink()
                removed.append(relative)
            except OSError as exc:
                errors.append(f"{relative}: {exc}")
        if not dry_run:
            for path in sorted(self.cache_dir.rglob("*"), key=lambda item: len(item.parts), reverse=True):
                if path.is_dir():
                    try:
                        path.rmdir()
                    except OSError:
                        pass
        return {
            "status": "PASS" if not errors else "WARN",
            "dry_run": dry_run,
            "max_age_days": max_age_days,
            "selected": len(removed),
            "bytes": bytes_selected,
            "removed": removed,
            "errors": errors,
        }

    def export_archive(self, archive_path: Path, *, include_cache: bool = False) -> dict[str, Any]:
        self.bootstrap()
        archive_path = archive_path.expanduser().resolve()
        validate_workspace_archive_path(archive_path, self.output_dir)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = archive_path.with_name(f".{archive_path.name}.{os.getpid()}.tmp")
        files = list(self.iter_export_files(include_cache=include_cache, archive_path=archive_path))
        try:
            with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr(
                    "PAPERLENS_EXPORT.json",
                    json.dumps(
                        {
                            "schema_version": EXPORT_SCHEMA_VERSION,
                            "created_at": utc_now(),
                            "storage_schema_version": STORAGE_SCHEMA_VERSION,
                            "source_output_dir": str(self.output_dir),
                            "include_cache": include_cache,
                            "file_count": len(files),
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                )
                for path in files:
                    archive.write(path, path.relative_to(self.output_dir).as_posix())
            os.replace(tmp_path, archive_path)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        return {
            "status": "PASS",
            "archive": str(archive_path),
            "file_count": len(files),
            "bytes": archive_path.stat().st_size,
        }

    def iter_export_files(self, *, include_cache: bool, archive_path: Path) -> Iterable[Path]:
        managed_roots = [self.output_dir / relative for relative in MANAGED_RELATIVE_PATHS]
        for path in [*managed_roots, *self.managed_report_paths()]:
            reject_workspace_symlink(path, self.output_dir)
            if not path.exists():
                continue
            if path.is_file():
                if path.resolve() != archive_path:
                    yield path
                continue
            for child in sorted(path.rglob("*")):
                reject_workspace_symlink(child, self.output_dir)
                if not child.is_file():
                    continue
                if child.resolve() == archive_path:
                    continue
                if not include_cache and self.path_is_under(child, self.cache_dir):
                    continue
                yield child

    def import_archive(self, archive_path: Path, *, replace: bool = False) -> dict[str, Any]:
        archive_path = archive_path.expanduser().resolve()
        validate_workspace_archive_path(archive_path, self.output_dir)
        if not archive_path.exists():
            raise FileNotFoundError(f"Workspace archive not found: {archive_path}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        staging_dir = unique_sibling_path(
            self.output_dir.parent
            / f".{self.output_dir.name}.paperlens-import-{stamp}-{os.getpid()}"
        )
        staging_dir.mkdir(parents=True, exist_ok=False)
        export_manifest: dict[str, Any] = {}
        extracted = 0
        backup_dir: Path | None = None
        try:
            with zipfile.ZipFile(archive_path) as archive:
                export_manifest = self.validate_export_archive(archive)
                managed_infos: list[zipfile.ZipInfo] = []
                total_uncompressed = 0
                for info in archive.infolist():
                    if info.is_dir() or info.filename == "PAPERLENS_EXPORT.json":
                        continue
                    if not archive_member_is_managed(info.filename):
                        raise ValueError(
                            f"Archive member is not managed by PaperLens: {info.filename}"
                        )
                    validate_import_member(info)
                    managed_infos.append(info)
                    if len(managed_infos) > MAX_IMPORT_FILE_COUNT:
                        raise ValueError(
                            "Workspace archive contains too many files: "
                            f"{len(managed_infos)} > {MAX_IMPORT_FILE_COUNT}"
                        )
                    total_uncompressed += info.file_size
                    if total_uncompressed > MAX_IMPORT_UNCOMPRESSED_BYTES:
                        raise ValueError(
                            "Workspace archive is too large after decompression: "
                            f"{total_uncompressed} bytes exceeds "
                            f"{MAX_IMPORT_UNCOMPRESSED_BYTES}"
                        )
                expected_count = int_or_zero(export_manifest.get("file_count"))
                if "file_count" in export_manifest and expected_count != len(managed_infos):
                    raise ValueError(
                        "Workspace archive file count does not match its manifest: "
                        f"{len(managed_infos)} != {expected_count}"
                    )
                for info in managed_infos:
                    target = (staging_dir / info.filename).resolve()
                    safe_relative_to(target, staging_dir)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("wb") as destination:
                        shutil.copyfileobj(source, destination)
                    extracted += 1
            if extracted == 0:
                raise ValueError("Workspace archive does not contain any managed PaperLens files")
            existing = [path for path in self.managed_paths() if path.exists()]
            if existing and not replace:
                names = ", ".join(path.name for path in existing[:6])
                raise RuntimeError(
                    "Output directory already contains PaperLens workspace files. "
                    f"Use replace=True after exporting a backup. Existing: {names}"
                )
            if existing and replace:
                backup_dir = unique_sibling_path(
                    self.output_dir.parent
                    / f"{self.output_dir.name}.paperlens-backup-{stamp}-{os.getpid()}"
                )
                backup_dir.mkdir(parents=True, exist_ok=False)
                move_managed_paths(paths=existing, source_root=self.output_dir, target_root=backup_dir)
            try:
                move_workspace_contents(staging_dir, self.output_dir)
            except Exception:
                if backup_dir:
                    self.rollback_failed_import(backup_dir, stamp)
                raise
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
        try:
            migration = self.bootstrap()
        except Exception:
            if backup_dir:
                self.rollback_failed_import(backup_dir, stamp)
            else:
                self.quarantine_failed_import(stamp)
            raise
        return {
            "status": "PASS",
            "archive": str(archive_path),
            "output_dir": str(self.output_dir),
            "extracted": extracted,
            "backup_dir": str(backup_dir) if backup_dir else "",
            "source_storage_schema_version": export_manifest.get("storage_schema_version"),
            "migration": migration,
        }

    def quarantine_failed_import(self, stamp: str) -> Path:
        failed_dir = self.failed_import_dir(stamp)
        failed_dir.mkdir(parents=True, exist_ok=False)
        move_managed_paths(
            paths=self.managed_paths(),
            source_root=self.output_dir,
            target_root=failed_dir,
        )
        return failed_dir

    def validate_export_archive(self, archive: zipfile.ZipFile) -> dict[str, Any]:
        try:
            manifest_info = archive.getinfo("PAPERLENS_EXPORT.json")
        except KeyError as exc:
            raise ValueError("Not a PaperLens workspace archive: missing PAPERLENS_EXPORT.json") from exc
        if manifest_info.file_size > MAX_IMPORT_MANIFEST_BYTES:
            raise ValueError(
                "PaperLens workspace archive manifest is too large: "
                f"{manifest_info.file_size} bytes"
            )
        try:
            raw = archive.read(manifest_info)
        except RuntimeError as exc:
            raise ValueError("Cannot read PaperLens workspace archive manifest") from exc
        try:
            manifest = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid PaperLens workspace archive manifest") from exc
        if not isinstance(manifest, dict):
            raise ValueError("Invalid PaperLens workspace archive manifest")
        if manifest.get("schema_version") != EXPORT_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported PaperLens workspace archive schema: "
                f"{manifest.get('schema_version')!r}"
            )
        source_schema = int_or_zero(manifest.get("storage_schema_version"))
        if source_schema > STORAGE_SCHEMA_VERSION:
            raise RuntimeError(
                "This PaperLens build cannot import a newer workspace schema: "
                f"{source_schema} > {STORAGE_SCHEMA_VERSION}"
            )
        return manifest

    def rollback_failed_import(self, backup_dir: Path, stamp: str) -> None:
        failed_dir = self.failed_import_dir(stamp)
        failed_dir.mkdir(parents=True, exist_ok=False)
        move_managed_paths(
            paths=self.managed_paths(),
            source_root=self.output_dir,
            target_root=failed_dir,
        )
        move_workspace_contents(backup_dir, self.output_dir)
        shutil.rmtree(backup_dir, ignore_errors=True)

    def failed_import_dir(self, stamp: str) -> Path:
        return unique_sibling_path(
            self.output_dir.parent
            / f"{self.output_dir.name}.paperlens-failed-import-{stamp}-{os.getpid()}"
        )

    def managed_paths(self) -> list[Path]:
        return [
            *[self.output_dir / relative for relative in MANAGED_RELATIVE_PATHS],
            *self.managed_report_paths(),
        ]

    def managed_report_paths(self) -> list[Path]:
        papers_dir = self.output_dir / MANAGED_REPORTS_DIR
        if not papers_dir.exists():
            return []
        return [
            path
            for path in sorted(papers_dir.glob("*.md"))
            if is_managed_report_file(path)
        ]

    def stats(self) -> dict[str, Any]:
        managed_bytes = sum(path_size(path) for path in self.managed_paths() if path.exists())
        return {
            "managed_bytes": managed_bytes,
            "cache_bytes": path_size(self.cache_dir),
            "paper_reports": len(self.managed_report_paths()),
        }

    @staticmethod
    def path_is_under(path: Path, root: Path) -> bool:
        try:
            path.resolve().relative_to(root.resolve())
            return True
        except ValueError:
            return False


def safe_relative_to(path: Path, root: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Path escapes workspace: {path}") from exc


def unique_sibling_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.name}-{index}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Cannot allocate a unique path near: {path}")


def validate_workspace_archive_path(archive_path: Path, output_dir: Path) -> None:
    if archive_path.suffix.lower() != ".zip":
        raise ValueError(f"Workspace archive path must end with .zip: {archive_path}")
    output_dir = output_dir.expanduser().resolve()
    for relative in MANAGED_RELATIVE_PATHS:
        managed = (output_dir / relative).resolve()
        if archive_path == managed:
            raise ValueError(
                f"Workspace archive path cannot overwrite managed PaperLens data: {archive_path}"
            )
        if managed.suffix:
            continue
        try:
            archive_path.relative_to(managed)
        except ValueError:
            continue
        raise ValueError(
            f"Workspace archive path cannot be inside managed PaperLens data: {archive_path}"
        )


def int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def is_fail_issue(issue: str) -> bool:
    return issue.startswith(
        (
            "invalid_workspace_manifest",
            "unsupported_workspace_manifest_schema",
            "newer_workspace_schema",
            "state_db:",
            "invalid_json:",
        )
    )


def archive_member_is_managed(filename: str) -> bool:
    normalized = filename.replace("\\", "/")
    first_part = normalized.split("/", 1)[0]
    if normalized.startswith("/") or ":" in first_part:
        return False
    parts = normalized.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    normalized = "/".join(parts)
    if any(normalized == relative for relative in MANAGED_FILE_RELATIVE_PATHS):
        return True
    if any(normalized.startswith(f"{relative}/") for relative in MANAGED_DIR_RELATIVE_PATHS):
        return True
    return is_managed_report_archive_path(normalized)


def is_managed_report_archive_path(normalized: str) -> bool:
    parts = normalized.split("/")
    if len(parts) != 2 or parts[0] != MANAGED_REPORTS_DIR:
        return False
    name = parts[1]
    return name.startswith("p_") and name.endswith(".md")


def is_managed_report_file(path: Path) -> bool:
    try:
        is_file = path.is_file()
    except OSError:
        return False
    if not is_file or path.is_symlink():
        return False
    name = path.name
    return name.startswith("p_") and name.endswith(".md")


def validate_import_member(info: zipfile.ZipInfo) -> None:
    if info.flag_bits & 0x1:
        raise ValueError(f"Encrypted archive member is not supported: {info.filename}")
    file_type = (info.external_attr >> 16) & 0o170000
    if S_ISLNK(file_type):
        raise ValueError(f"Symlink archive member is not supported: {info.filename}")
    if info.file_size > MAX_IMPORT_MEMBER_BYTES:
        raise ValueError(
            f"Archive member is too large: {info.filename} "
            f"({info.file_size} bytes exceeds {MAX_IMPORT_MEMBER_BYTES})"
        )


def move_managed_paths(
    *,
    paths: Iterable[Path],
    source_root: Path,
    target_root: Path,
) -> None:
    for path in paths:
        if not path.exists():
            continue
        relative = path.relative_to(source_root)
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))


def move_workspace_contents(source_root: Path, target_root: Path) -> None:
    files = [path for path in sorted(source_root.rglob("*")) if path.is_file()]
    for path in files:
        relative = path.relative_to(source_root)
        target = target_root / relative
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing workspace file: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(target))
    shutil.rmtree(source_root, ignore_errors=True)


def path_size(path: Path, *, skip_dirs: set[str] | None = None) -> int:
    skip_dirs = skip_dirs or set()
    if path.is_symlink() or not path.exists():
        return 0
    if path.is_file():
        try:
            return path.stat().st_size
        except OSError:
            return 0
    total = 0
    for child in path.rglob("*"):
        if any(part in skip_dirs for part in child.parts):
            continue
        if child.is_symlink():
            continue
        if child.is_file():
            try:
                total += child.stat().st_size
            except OSError:
                continue
    return total


def reject_workspace_symlink(path: Path, root: Path) -> None:
    if not path.is_symlink():
        return
    try:
        label = str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        label = str(path)
    raise ValueError(f"Refusing to export symlinked workspace path: {label}")
