PaperLens

Quick start:
  1. Open PaperLens.
  2. Choose a PDF input directory and an output directory.
  3. Enter provider settings, API key, model name, and concurrency.
  4. Start the reading job.

Human-facing output:
  output/
    PaperLens.md
    papers/
      <paper_id>_<short_title>.md

Internal memory:
  output/
    .paperlens/
      workspace.json
      library/
        library_records.jsonl
        index/
          search_index.json
      data/
        migrations.jsonl
      cache/
      recovery/
      pages/
      figures/

Open PaperLens.md first when using raw files. The desktop app reads the same
output directory and presents the library, capsule, evidence, and chat views.

Workspace maintenance:
  paperlens-core workspace doctor --output-dir <output>
  paperlens-core workspace doctor --output-dir <output> --repair
  paperlens-core workspace export --output-dir <output> --archive <backup.zip>
  paperlens-core workspace import --output-dir <output> --archive <backup.zip>
  paperlens-core workspace cleanup-cache --output-dir <output> --max-age-days 30

Security defaults:
  - API keys are accepted at runtime and redacted from run config snapshots.
  - Desktop-to-core requests use local HTTP header auth, not URL query tokens.
  - Markdown reports are sanitized and raw HTML is not rendered.
  - Model-call diagnostics record stage, payload size, usage, status, and request id.
  - Model prompts and responses are not written to the call ledger.
  - Local app settings do not persist the API key.
