# PaperLens Core v1

PaperLens Core is a paper knowledge-processing agent. It is not a PDF reader, a classic reference manager, or a generic paper summarizer.

The core workflow is:

```text
PDF / text / page images
-> page index / deterministic paper tools
-> ContextPack for each agent step
-> MemoryPatch log
-> materialized PaperMemoryV3 IR
-> critic / targeted reread / repair
-> ReportPlan / ReportSection / SectionAudit / assembled Standard capsule
-> evidence-bounded QA
-> Library
-> later: Discovery / Reproduction skill
```

The active run is deliberately smaller than a general agent framework:

```text
ingest
-> parse
-> parse verification
-> paper map
-> standard read loop
-> memory/evidence verification
-> report assembly
-> manifest/library
```

PaperLens does not spend separate stages on grade validation or queue construction.
Every imported paper enters the same standard reading loop. Grade is a presentation
signal, not an execution branch or a shortcut around memory construction.

## Product Principles

1. PaperMemoryV3 is the source of truth. Reports, QA, Library, and future reproduction must consume memory and evidence, not each other.
2. Core v1 has one default report view: a Standard knowledge capsule. The report is streamed at the workflow level through `ReportPlan -> ReportSection[] -> SectionAudit[] -> Assemble`; no single model call is responsible for writing the full report, and no hard length floor or regex fact rewrite is allowed. A close-reading view can be added later as an explicit opt-in skill.
3. Grade is a presentation and prioritization signal. It never controls claim truthfulness, and the standard read loop does not skip papers based on grade.
4. Every meaningful claim should carry provenance, confidence, evidence refs, and critic status.
5. Output language is a rendering preference. Chinese and English reports consume the same PaperMemory/library; internal reading prompts may stay English.
6. PaperMemory keeps a fixed concept-introduction layer for necessary background terms. Reports should use that layer organically, without turning into a generic basics section.
7. QA should answer conversationally by default, show concise provenance in the answer, and keep detailed evidence auditable on demand. Challenge and evidence-check questions expand evidence by default.
8. Steer v1 focuses on reading correction and deepening: challenge claim, deepen experiments, explain term, and change lens.
9. Library contains only papers PaperLens has actually read and converted into PaperMemory. External candidates belong to Discovery until imported and processed.
10. Reproduction is a downstream opt-in skill. Core v1 only reserves schema and adapter space for it.

## PaperMemoryV3 Contract

PaperMemoryV3 is the compiler-like intermediate representation for a paper. A final report may be more readable, but it must not introduce unsupported facts that are absent from memory/evidence.

PaperMemoryV3 is runtime state, not an exported summary. The store contract is:

```text
model/read tools produce MemoryPatch events
-> PaperMemoryStore appends patch log
-> PaperMemoryStore materializes PaperMemoryV3
-> reports / QA / library consume the materialized V3 state
```

Formal runs no longer write ad-hoc memory projections. The canonical state is
`paper_memory.v3.json` plus its MemoryPatch log.

## Agent Runtime Context

PaperLens does not put the whole PDF into every model call. Each model call is stateless at the API layer, but the runtime gives it continuity with a compact `ContextPack`:

```text
Always context:
  paper id, title, grade, current PaperMemory view, known claims, known evidence

Working context:
  current objective, focus queries, focus pages, already-read pages, unresolved uncertainty

Tool trace:
  deterministic search/read/figure observations over parsed paper text and captions

Output contract:
  MemoryPatchSet, MemoryCritic audit, QA answer, or another bounded artifact
```

This is the PaperLens equivalent of a lightweight Codex/Claude-Code style loop: the model receives task state and local tool results, decides what is uncertain, and returns a patch or bounded answer. The runtime remains responsible for applying patches, preserving evidence refs, writing the audit trail, and preventing reports or QA from becoming the source of truth.

## Core Loop

PaperLens Core does not expose a general skill registry. The workflow uses a small fixed loop:

```text
Read pages/sections
  -> emit MemoryPatchSet
Ground claims
  -> attach text/figure/table evidence
Critic
  -> identify only high-risk gaps
Targeted repair
  -> patch memory
Compose capsule
  -> plan, section drafts, section audit, assembly
QA
  -> answer from PaperMemory, library records, focused pages, and tool observations
```

The workflow owns job state and stage ordering. Deterministic paper tools retrieve
paper-local evidence; hooks validate patches and section outputs; memory remains
the only durable knowledge state.

Minimum sections:

```text
metadata
reading_context
problem_frame
core_abstractions
mechanism
implementation_details
evaluation
concepts
conceptual_bridge
claims
evidence
figures_tables
limitations
relations
open_questions
audit_trail
user_overrides
```

The most important objects are claims and evidence.

```text
Claim:
  id
  text
  type: motivation | mechanism | evaluation | limitation | comparison | implication
  provenance: explicit | inferred | background | external
  confidence: high | medium | low
  evidence_refs
  depends_on
  risk_tags
  critic_status: unchecked | checked | repaired | disputed

Evidence:
  id
  source_type: text_span | page_image | figure | table | equation
  page
  section
  excerpt_or_caption
  visual_region
  interpretation
  reliability: direct | indirect

Conceptual bridge:
  needed
  reader_gap
  bridge_text
  terms:
    term
    explanation
    paper_role
    provenance: explicit | inferred | background

This layer is for terms a reader must understand before the paper's core idea works, such as KV cache, batch, decode, tail latency, consensus timeout, or cache eviction. It is not a separate "basic concepts" page. Background context is allowed here only when it is clearly marked as background and not promoted into a paper claim.
```

## MemoryPatch Contract

Model reading and repair calls must return a `MemoryPatchSet`, not a full replacement memory object.
The store validates and applies those operations, then writes the materialized V3 state and patch log.

Core operations:

```text
add_read_pages
set_problem_frame
set_core_abstraction
set_mechanism_overview
upsert_mechanism_step
set_evaluation_summary
upsert_evaluation_item
upsert_concept
set_conceptual_bridge
upsert_conceptual_bridge_term
upsert_evidence
upsert_claim
link_claim_evidence
mark_claim_disputed
add_limitation
add_open_question
set_memory_audit
set_report_audit
add_partial_read_failure
add_user_override
```

This makes memory mutation explicit and auditable: rolling read, targeted reread, critic repair,
report audit, QA steer, and user corrections all become patches over the same IR.

## Report Composer

Reports are no longer generated in one full-report call. The export stage now does:

```text
PaperMemoryV3
-> ReportPlanner creates reader-order section plan
-> ReportComposer writes one section per call, with mechanism-specific detail contracts when needed
-> SectionAuditor checks that section against memory/evidence
-> optional section-local rewrite when the section audit requests repair
-> deterministic Markdown assembly
```

The assembler may repair Markdown boundaries and render figure/table crops, but it must not rewrite factual content. Whole-report repair and `paperlens_final_*` protocols are removed from the active workflow.

## Internal State

The active workflow keeps only the durable memory state and its patch log under:

```text
.paperlens/data/memory/v3/
```

Per paper:

```text
<paper_id>.paper_memory.v3.json
<paper_id>.memory_patches.jsonl
```

## QA v2 Direction

QA should behave like a small investigation:

```text
question
-> classify
-> retrieval plan
-> retrieve memory / evidence / pages / figures
-> draft answer
-> QA critic
-> optional targeted reread
-> final answer with provenance
```

Initial question types:

```text
orientation
mechanism
clarification
evidence_check
comparison
implementation
reproduction
library_recall
```

QA traces are written to:

```text
.paperlens/data/qa_trace.jsonl
```

They record the question, inferred question type, selected pages, source attribution, confidence, cache state, and the compact agent context summary used for the answer.

QA does not use the rendered report as a source. The report path can be shown as a user-facing orientation artifact, but answers are grounded in PaperMemoryV3, library records, local page evidence, visual context, and the agent context pack.
