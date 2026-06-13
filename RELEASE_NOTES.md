# PaperLens 0.2.0

This release turns the local PaperLens workspace into a versioned product data store.

## Local Workspace Safety

- Added a workspace manifest at `.paperlens/workspace.json` with a storage schema version.
- PaperLens now migrates and checks a workspace before reading, asking questions, or rebuilding the library.
- JSON artifacts, reports, library indexes, QA cache files, and typed core artifacts are written atomically to reduce partial-file corruption after crashes or forced shutdowns.
- SQLite-backed paper state access is serialized and safe for service worker threads.
- ClaimGraph and audit lookups now use in-memory indexes for evidence edges and PaperDOM source text.
- Model budget snapshots now preserve explicit zero token fields and are read under lock.
- Corrupt critical JSON files are moved into `.paperlens/recovery/` during repair instead of being overwritten silently.
- Added workspace-level maintenance commands for doctor, migration, cache cleanup, export, and import.

## Install And Upgrade

- Release notes are now reused by the updater manifest and GitHub Release publishing.
- Portable builds include `RELEASE_NOTES.md` beside the app.
- Python sidecar builds now write PyInstaller output under `build/` instead of sharing the frontend `dist/` directory.
- Large artifact hashing now streams files instead of reading whole PDFs or page images into memory.
- Workspace archives can be exported before upgrades and imported into a clean workspace if recovery is needed.
- Workspace archive import now rejects encrypted, abnormally large, or suspiciously dense archives before extraction.
- Workspace archive export/import now rejects symlink entries and verifies the manifest file count before restore.
- Workspace archive paths must be `.zip` files outside PaperLens-managed data paths.
- Desktop startup diagnostics now show the Core launch attempts and searched locations when the bundled sidecar is missing or cannot run.
- Desktop progress streams and report images now authenticate with local HTTP headers instead of URL query tokens.
- Markdown reports are sanitized without raw HTML rendering.
- Runtime events, error records, and model-call ledger entries are redacted before writing to disk or the desktop event stream.
- Completed read jobs and QA answers now clear runtime API keys from in-memory request payloads; retries use the current UI provider settings.
- QA answer state updates and event-stream status reads now use the service lock consistently.
- Local maintenance cleanup no longer clears interface settings until the desktop cleanup command succeeds.
- Workspace cleanup now requires explicit PaperLens workspace markers and removes only generated `p_*.md` reports inside `papers/`, instead of deleting a generic `papers/` folder.
- Local chat history now has bounded storage snapshots and tolerates localStorage quota or availability failures.
- Desktop cleanup removes symbolic links as links instead of following them into external directories.
- Markdown reports open external links through the system browser and no longer load remote image URLs directly.
- Release scripts now use explicit skip branches for optional updater artifacts instead of early success exits.
- Reading prompts and QA fallbacks are now domain-general and no longer expose ClaimGraph, PaperDOM, or source ID internals in reader-facing answers.
- Report section routing now uses domain-general result and ablation cues instead of hardcoded paper-metric names.

## Operator Commands

```bash
paperlens-core workspace doctor --output-dir <workspace>
paperlens-core workspace doctor --output-dir <workspace> --repair
paperlens-core workspace export --output-dir <workspace> --archive <backup.zip>
paperlens-core workspace import --output-dir <workspace> --archive <backup.zip>
paperlens-core workspace cleanup-cache --output-dir <workspace> --max-age-days 30
```
