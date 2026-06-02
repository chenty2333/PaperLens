# PaperLens Core v1

PaperLens Core is a paper knowledge-processing agent. It is not a PDF reader,
a classic reference manager, or a generic paper summarizer.

The active workflow is deliberately small:

```text
PDF
-> paper map
-> chunked reading into PaperMemory
-> one central memory verification pass
-> section-by-section capsule writing
-> QA over PaperMemory and local paper evidence
-> library update
```

## Product Principles

1. PaperMemory is the source of truth. Reports, QA, Library, and future skills
   consume memory and evidence, not each other.
2. The paper map is built once by deterministic tools: sections, pages,
   figures/tables, captions, visual hints, and key text blocks.
3. The reader only reads. Chunked reading calls receive the paper map, current
   pages, and current memory; they return MemoryPatch operations and do not
   write report prose.
4. Verification is one central pass. The verifier receives current memory,
   high-risk claims, and relevant original page evidence, then patches memory
   directly and records the evidence boundary.
5. Report writing is streamed at the workflow level. A plan is generated first,
   then sections are written from memory and local evidence; the assembler only
   joins Markdown and renders visuals.
6. QA does not read the rendered report as a fact source. It retrieves memory,
   library records, local page excerpts, and page images when needed.
7. Grade is a presentation and prioritization signal. It does not control claim
   truthfulness and does not skip the standard read loop.

## Core Loop

```text
1. Parse
   Build the paper map once and save it under .paperlens/data/artifacts/layout/.

2. Read
   Select useful pages, read them in small chunks, and apply MemoryPatchSet
   operations to PaperMemory.

3. Verify
   Select high-risk claims and evidence-linked pages. Run one verifier call that
   returns a MemoryPatchSet with claim/evidence fixes plus one memory audit.

4. Compose
   Build a report plan and write each section from PaperMemory and evidence.
   Section audits catch unsupported prose, but there is no whole-report fact
   rewrite path.

5. Ask
   Answer questions from PaperMemory, local evidence, and library records. The
   report is useful orientation, not the truth source.
```

## Runtime Context

Model calls are stateless at the API layer. PaperLens gives them continuity with
a compact context pack:

```text
Always context:
  paper id, title, grade, current memory, known claims, known evidence

Working context:
  objective, focus pages, focus queries, current page chunk, high-risk claims

Tool trace:
  deterministic search/read/figure observations over parsed paper text/captions

Output contract:
  MemoryPatchSet, report plan, report section, section audit, or QA answer
```

This is intentionally narrower than a general agent framework. The runtime owns
state, cache, events, retries, and patch application; the model handles reading,
verification, writing, and answering inside bounded contracts.

## PaperMemory Contract

PaperMemory is internal runtime state, not a second user report. It stores:

- metadata and reading context
- problem frame and core abstraction
- mechanism and implementation details
- evaluation and limitations
- concepts and prerequisite background bridge
- claims, evidence, figures/tables, relations, open questions
- audit trail and user overrides

Every meaningful claim should carry provenance, confidence, evidence refs, and
status. Background concepts must stay in the concept bridge instead of being
promoted to paper claims.

## User Outputs

User-facing output stays small:

```text
PaperLens.md              run index
papers/<paper>.md         one readable capsule per paper
desktop Library           already-read papers only
desktop Chat              current paper or library QA
```

Internal files stay under `.paperlens/` for recovery, QA, evidence lookup, and
library search.
