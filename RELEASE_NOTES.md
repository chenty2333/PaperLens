# PaperLens 0.2.0

This release turns the local PaperLens workspace into a versioned product data store.

## Local Workspace Safety

- Added a workspace manifest at `.paperlens/workspace.json` with a storage schema version.
- PaperLens now migrates and checks a workspace before reading, asking questions, or rebuilding the library.
- JSON artifacts, reports, library indexes, QA cache files, and typed core artifacts are written atomically to reduce partial-file corruption after crashes or forced shutdowns.
- Corrupt critical JSON files are moved into `.paperlens/recovery/` during repair instead of being overwritten silently.
- Added workspace-level maintenance commands for doctor, migration, cache cleanup, export, and import.

## Install And Upgrade

- Release notes are now reused by the updater manifest and GitHub Release publishing.
- Portable builds include `RELEASE_NOTES.md` beside the app.
- Workspace archives can be exported before upgrades and imported into a clean workspace if recovery is needed.
- Workspace archive import now rejects encrypted, abnormally large, or suspiciously dense archives before extraction.
- Desktop startup diagnostics now show the Core launch attempts and searched locations when the bundled sidecar is missing or cannot run.
- Desktop progress streams and report images now authenticate with local HTTP headers instead of URL query tokens.
- Markdown reports are sanitized without raw HTML rendering.

## Operator Commands

```bash
paperlens-core workspace doctor --output-dir <workspace>
paperlens-core workspace doctor --output-dir <workspace> --repair
paperlens-core workspace export --output-dir <workspace> --archive <backup.zip>
paperlens-core workspace import --output-dir <workspace> --archive <backup.zip>
paperlens-core workspace cleanup-cache --output-dir <workspace> --max-age-days 30
```
