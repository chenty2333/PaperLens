# PaperLens

[English](README.md) | [中文](README.zh-CN.md)

[![Windows Installer][ci-badge]][ci-workflow]
[![License: Unlicense](https://img.shields.io/badge/license-Unlicense-blue.svg)](LICENSE)

PaperLens is a desktop paper-reading agent. It is not a PDF reader, a traditional reference manager, or a generic summarizer. Its core job is to turn papers into auditable knowledge state: a structured PaperMemory, a readable capsule, evidence-bounded QA, and a local library of papers that the system has actually read.

## What It Does

- Reads PDFs with text, layout, page images, figure/table hints, and deterministic paper-local tools.
- Builds PaperMemoryV3 as the source of truth instead of treating the final report as the fact source.
- Mutates memory through auditable MemoryPatch events.
- Runs critic, targeted reread, and repair passes before producing user-facing output.
- Generates a Standard knowledge capsule aimed at helping you avoid reading the original paper unless you choose to.
- Answers questions from memory, evidence, local pages, and library records rather than from the rendered report.
- Maintains a local library containing only papers PaperLens has processed.
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
    state.sqlite
    library/
      paper_memory.jsonl
      index/
        search_index.json
    pages/
    figures/
    data/
      run.json
      events.jsonl
      memory/
        v3/
          <paper_id>.paper_memory.v3.json
          <paper_id>.memory_patches.jsonl
          claim_index.jsonl
          evidence_index.jsonl
```

Open `PaperLens.md` first when using raw output files. The desktop app reads the same output directory and presents the library, capsule, evidence, and chat surfaces.

## Install And Run

For releases, download the Windows NSIS installer from GitHub Actions artifacts or tagged GitHub Releases.

The installer is configured as a per-user install:

- no administrator permission by default;
- installs for the current user;
- keeps user-selected paper libraries/output directories outside the app install location;
- asks during uninstall whether local settings and WebView cache should also be removed.

The app does not ship with a model key, model name, or provider URL. Enter those in the UI for each session. API keys are not persisted in local settings.

## Development

Requirements:

- Node.js 22
- Python 3.12+
- Rust stable
- `uv`
- NSIS on Windows for installer builds

```powershell
npm ci
uv run --extra dev ruff check .
uv run --extra dev pytest
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

The installer is written to:

```text
src-tauri/target/release/bundle/nsis/
```

## CI

The Windows installer workflow lives at [.github/workflows/windows-installer.yml](.github/workflows/windows-installer.yml).

It runs:

- committed-secret guard;
- Python lint/tests;
- frontend lint/build;
- Python sidecar build;
- Tauri NSIS installer build;
- installer artifact upload;
- tagged release publish for `v*` tags.

The workflow intentionally does not require model API keys.

## Security And Privacy Defaults

PaperLens is designed to keep keys out of the repository and output artifacts.

```text
OPENAI_AGENTS_DONT_LOG_MODEL_DATA=1
OPENAI_AGENTS_DONT_LOG_TOOL_DATA=1
OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA=0
```

The UI does not persist API keys to `localStorage`. Generated paper libraries, reports, sidecars, local caches, virtual environments, build output, and `.env` files are ignored by Git.

## Documentation

- [PaperLens Core v1 design](docs/PaperLens_Core_v1.md)
- [Chinese README](README.zh-CN.md)

## License

PaperLens is released under the [Unlicense](LICENSE).

[ci-badge]: https://github.com/chenty2333/PaperLens/actions/workflows/windows-installer.yml/badge.svg
[ci-workflow]: https://github.com/chenty2333/PaperLens/actions/workflows/windows-installer.yml
