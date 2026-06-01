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
  OPENAI_AGENTS_DONT_LOG_MODEL_DATA=1
  OPENAI_AGENTS_DONT_LOG_TOOL_DATA=1
  OPENAI_AGENTS_TRACE_INCLUDE_SENSITIVE_DATA=0

API keys are not written to config, SQLite, JSONL, or CLI args.
