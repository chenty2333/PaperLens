# PaperLens

[English](README.md) | [中文](README.zh-CN.md)

[![Windows Installer][ci-badge]][ci-workflow]
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](LICENSE)

PaperLens is a desktop paper-reading agent. It is not a PDF reader, a traditional reference manager, or a generic summarizer. Its core job is to turn papers into auditable knowledge state: a source-anchored PaperDOM, a ClaimGraph, derived PaperMemory views, evidence-bounded QA, and a local graph library of papers that the system has actually read.

## What It Does

- Reads PDFs with text, layout, page images, figure/table hints, and deterministic paper-local tools.
- Builds the paper map once as PaperDOM: sections, source IDs, figures/tables, equations, and key text blocks.
- Builds a deterministic Reading Plan over PaperDOM source IDs and records append-only observations.
- Builds ClaimGraph as the source of truth; PaperMemory is a derived product view, not a model-edited state file.
- Runs deterministic audit checks over graph nodes, edges, source IDs, and numeric grounding.
- Writes the paper report as a readable ClaimGraph view; reports cannot introduce new facts.
- Answers questions from ClaimGraph nodes, PaperDOM source evidence, and library graph records rather than from the rendered report or page-number references.
- Maintains a local graph library containing only papers PaperLens has processed.
- Ships as a Windows Tauri desktop app with a Python `paperlens-core` sidecar.

## Current Product Shape

The app has three surfaces:

- **Library**: previously read papers, title, grade, concepts, and brief.
- **Capsule**: the main Markdown-rendered paper explanation.
- **Chat**: scoped to the current paper or the whole local library.

The core engine can also be used directly through `paperlens-core`.

## Output Layout

```text
output/
  PaperLens.md
  papers/
    <paper_id>_<short_title>.md
  .paperlens/
    workspace.json
    state.sqlite
    cache/
    recovery/
    library/
      library_records.jsonl
      index/
        search_index.json
    pages/
    figures/
    data/
      run.json
      events.jsonl
      migrations.jsonl
      core/
        v2/
          <paper_id>/
            paper_dom.v2.json
            reading_plan.v2.json
            observation_log.v2.json
            claim_graph.v2.json
            relation_candidate_log.v2.json
            audit_findings.v2.json
            quality_metrics.v2.json
            paper_memory_view.v2.json
            report_draft.v2.json
            report_audit_findings.v2.json
            core_manifest.v2.json
```

Open `PaperLens.md` first when using raw output files. The desktop app reads the same output directory and presents the library, capsule, evidence, and chat surfaces.

## Local Workspace Storage

The output directory is a versioned PaperLens workspace. `.paperlens/workspace.json`
declares the storage schema and layout. PaperLens bootstraps or migrates the workspace
before reading papers, rebuilding the library, or answering questions.

Product-level maintenance commands:

```powershell
uv run python -m paperlens_core.main workspace doctor --output-dir output
uv run python -m paperlens_core.main workspace doctor --output-dir output --repair
uv run python -m paperlens_core.main workspace export --output-dir output --archive PaperLens-backup.zip
uv run python -m paperlens_core.main workspace import --output-dir output --archive PaperLens-backup.zip
uv run python -m paperlens_core.main workspace cleanup-cache --output-dir output --max-age-days 30
```

Critical JSON artifacts, reports, library indexes, QA cache files, and typed core
artifacts are written through atomic replacement. If a critical JSON file is corrupt
during repair, PaperLens moves it to `.paperlens/recovery/` instead of silently
overwriting it.

## Install And Run

For releases, download the Windows NSIS installer from GitHub Actions artifacts or tagged GitHub Releases.

The installer is configured as a per-user install:

- no administrator permission by default;
- installs for the current user;
- keeps user-selected paper libraries/output directories outside the app install location;
- asks during uninstall whether local settings and WebView cache should also be removed.

The app does not ship with a model key, model name, or provider URL. Enter those in the UI for each session. API keys are not persisted in local settings.

Before major upgrades, export important workspaces with `paperlens-core workspace export`.
On startup or workspace open, PaperLens migrates the selected workspace in place. If a
replace import is needed, the old managed workspace files are moved to a timestamped
backup directory next to the workspace.

## Development

Requirements:

- Node.js 24
- Python 3.12+
- Rust stable
- `uv`
- NSIS on Windows for installer builds

```powershell
npm ci
uv run --extra dev ruff check .
npm run lint
npm run build
```

Run the desktop app in development:

```powershell
npm run tauri:dev
```

Build the Python sidecar and Windows installer:

```powershell
npm run core:build
npm run tauri:build
```

Build a portable Windows folder and zip:

```powershell
npm run portable:build
```

Bump and synchronize all release versions from one command:

```powershell
npm run version:patch
```

Use `npm run version:minor` or `npm run version:major` for larger releases. CI runs `npm run version:check` so `package.json`, Python metadata, Tauri config, Rust metadata, and lockfiles cannot drift silently.

The installer is written to:

```text
src-tauri/target/release/bundle/nsis/
```

The portable package is written to:

```text
build/PaperLens/
build/PaperLens-<version>-windows-x64-portable.zip
build/PaperLens-<version>-windows-x64-portable.zip.sha256
build/PaperLens-<version>-windows-x64-portable.json
```

## CI

The Windows installer workflow lives at [.github/workflows/windows-installer.yml](.github/workflows/windows-installer.yml).

It runs:

- Linux preflight checks for synchronized versions, Python lint, frontend lint, and frontend build;
- committed-secret guard;
- Python sidecar build;
- Tauri NSIS installer build;
- portable Windows zip packaging;
- installer artifact upload;
- GitHub Release publish when a `v*` tag is pushed, when `main` changes `package.json` version, or when manually dispatched with `publish_release=true`;
- signed updater metadata publish when updater signing secrets are configured.

The workflow intentionally does not require model API keys.

Automatic in-app updates are enabled only for signed release builds. Configure these repository secrets before publishing a self-updating build:

- `PAPERLENS_UPDATER_PUBKEY`: public key generated by the Tauri signer.
- `TAURI_SIGNING_PRIVATE_KEY`: private signing key used only inside GitHub Actions.
- `TAURI_SIGNING_PRIVATE_KEY_PASSWORD`: optional signing key password.

The workflow publishes the NSIS installer, portable zip, portable hash/metadata, installer signature, and `latest.json` to the GitHub Release. By default the app checks `https://github.com/<owner>/<repo>/releases/latest/download/latest.json`; override that with the `PAPERLENS_UPDATER_ENDPOINT` repository variable if needed.

Release notes come from [RELEASE_NOTES.md](RELEASE_NOTES.md) and are reused by the
GitHub Release and signed updater manifest.

## Security And Privacy Defaults

PaperLens is designed to keep keys out of the repository and output artifacts.

The UI does not persist API keys to `localStorage`. Desktop-to-core requests authenticate with local HTTP headers instead of URL query tokens, including live progress streams and report images. Markdown reports are sanitized and raw HTML is not rendered. Generated paper libraries, reports, sidecars, local caches, virtual environments, build output, and `.env` files are ignored by Git. Local model-call accounting stores request sizes, stage names, status, and provider usage metadata, not prompt bodies or API keys.

## Documentation

- [Chinese README](README.zh-CN.md)

## License

PaperLens is released under the [Unlicense](LICENSE).

[ci-badge]: https://github.com/chenty2333/PaperLens/actions/workflows/windows-installer.yml/badge.svg
[ci-workflow]: https://github.com/chenty2333/PaperLens/actions/workflows/windows-installer.yml
