PaperLens

Quick start:
  1. Open PaperLens.
  2. Choose a PDF input directory and an output directory.
  3. Enter provider settings, API key, model name, budget, and concurrency.
  4. Start the reading job.

Human-facing output:
  output/
    PaperLens.md
    papers/
      <paper_id>_<short_title>.md

Internal memory:
  output/
    .paperlens/
      library/
        library_records.jsonl
        index/
          search_index.json
      data/
      cache/
      pages/
      figures/

Open PaperLens.md first when using raw files. The desktop app reads the same
output directory and presents the library, capsule, evidence, and chat views.

Security defaults:
  - API keys are accepted at runtime and redacted from run config snapshots.
  - Model-call diagnostics record stage, payload size, usage, status, and request id.
  - Model prompts and responses are not written to the call ledger.
  - Local app settings do not persist the API key.
