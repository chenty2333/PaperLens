from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

from paperlens_core.agent_loop import AgentLoop, PaperToolRegistry
from paperlens_core.agents.llm import JsonLlmClient, llm_call_context
from paperlens_core.memory_v3 import dict_value, list_payload
from paperlens_core.report.composer_context import (
    compact_decision_for_report,
    compact_paper_card_for_report,
    compact_skim_for_report,
)
from paperlens_core.report.composer_output import (
    aggregate_section_audits,
    assemble_agentic_report,
    enforce_section_depth_audit,
    normalize_report_section,
    normalize_report_section_audit,
    report_section_is_more_substantive,
)
from paperlens_core.report.composer_plan import normalize_report_plan, normalize_report_section_plan
from paperlens_core.report.composer_prompts import (
    REPORT_PLAN_SCHEMA,
    REPORT_PLAN_SYSTEM_PROMPT,
    REPORT_SECTION_AUDITOR_SYSTEM_PROMPT,
    REPORT_SECTION_AUDIT_SCHEMA,
    REPORT_SECTION_SCHEMA,
    REPORT_SECTION_SYSTEM_PROMPT,
    build_report_plan_prompt,
    build_report_section_audit_prompt,
    build_report_section_prompt,
    compact_report_plan,
)
from paperlens_core.report.memory_context import compact_paper_memory_for_report
from paperlens_core.report.text import clean_model_inline_text, compact_reason
from paperlens_core.runtime import (
    PaperLensRuntime,
    hash_json_payload,
    llm_cache_path,
    read_llm_cache,
    write_llm_cache,
)
from paperlens_core.schemas import ClassificationDecision, PaperCard, PaperRecord, SkimCard


REPORT_PLAN_PROMPT_VERSION = "report-plan-v4-complete-capsule-profile"
REPORT_SECTION_PROMPT_VERSION = "report-section-v6-depth-contract"
REPORT_SECTION_AUDIT_PROMPT_VERSION = "report-section-audit-v3-depth-and-boundary"


def normalize_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def hash_text(value: str) -> str:
    return hashlib.sha256(normalize_for_search(value).encode("utf-8")).hexdigest()[:16]


def compose_agentic_paper_report(
    *,
    client: JsonLlmClient,
    data_dir: Path,
    stage: str,
    paper: PaperRecord,
    skim: SkimCard | None,
    decision: ClassificationDecision | None,
    card: PaperCard | None,
    paper_memory: dict[str, Any] | None,
    layout: dict[str, Any],
    topic: str | None,
    idea: str | None,
    output_language: str,
    record_usage: Any,
    record_agent_run: Any,
    read_mode: str = "standard",
    cache_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    memory = dict_value(paper_memory)
    plan = generate_report_plan(
        client=client,
        data_dir=data_dir,
        stage=stage,
        paper=paper,
        skim=skim,
        decision=decision,
        card=card,
        paper_memory=memory,
        layout=layout,
        topic=topic,
        idea=idea,
        output_language=output_language,
        record_usage=record_usage,
        record_agent_run=record_agent_run,
        read_mode=read_mode,
        cache_dir=cache_dir,
    )
    sections: list[dict[str, Any]] = []
    section_audits: list[dict[str, Any]] = []
    previous_summaries: list[dict[str, str]] = []
    for raw_section_plan in list_payload(plan.get("sections")):
        if not isinstance(raw_section_plan, dict):
            continue
        section_plan = normalize_report_section_plan(raw_section_plan)
        section = generate_report_section(
            client=client,
            data_dir=data_dir,
            stage=stage,
            paper=paper,
            paper_memory=memory,
            layout=layout,
            plan=plan,
            section_plan=section_plan,
            previous_summaries=previous_summaries,
            output_language=output_language,
            record_usage=record_usage,
            record_agent_run=record_agent_run,
            read_mode=read_mode,
            cache_dir=cache_dir,
        )
        audit = audit_report_section(
            client=client,
            data_dir=data_dir,
            stage=stage,
            paper=paper,
            paper_memory=memory,
            layout=layout,
            plan=plan,
            section_plan=section_plan,
            section=section,
            output_language=output_language,
            record_usage=record_usage,
            record_agent_run=record_agent_run,
            read_mode=read_mode,
            cache_dir=cache_dir,
        )
        if audit.get("verdict") == "REPAIR":
            repaired_section = generate_report_section(
                client=client,
                data_dir=data_dir,
                stage=stage,
                paper=paper,
                paper_memory=memory,
                layout=layout,
                plan=plan,
                section_plan=section_plan,
                previous_summaries=previous_summaries,
                output_language=output_language,
                record_usage=record_usage,
                record_agent_run=record_agent_run,
                read_mode=read_mode,
                cache_dir=cache_dir,
                section_audit=audit,
            )
            repaired_audit = audit_report_section(
                client=client,
                data_dir=data_dir,
                stage=stage,
                paper=paper,
                paper_memory=memory,
                layout=layout,
                plan=plan,
                section_plan=section_plan,
                section=repaired_section,
                output_language=output_language,
                record_usage=record_usage,
                record_agent_run=record_agent_run,
                read_mode=read_mode,
                cache_dir=cache_dir,
            )
            if repaired_audit.get("verdict") != "REPAIR" or report_section_is_more_substantive(
                repaired_section, section
            ):
                section = repaired_section
                audit = repaired_audit
        sections.append(section)
        section_audits.append({"section_id": section.get("section_id"), **audit})
        previous_summaries.append(
            {
                "section_id": str(section.get("section_id") or section_plan.get("section_id")),
                "title": str(section.get("title") or section_plan.get("title")),
                "summary": compact_reason(
                    clean_model_inline_text(section.get("markdown")), max_chars=260
                ),
            }
        )
    report = assemble_agentic_report(
        paper=paper,
        decision=decision,
        plan=plan,
        sections=sections,
        section_audits=section_audits,
        output_language=output_language,
    )
    report_audit = aggregate_section_audits(section_audits)
    return report, report_audit


def generate_report_plan(
    *,
    client: JsonLlmClient,
    data_dir: Path,
    stage: str,
    paper: PaperRecord,
    skim: SkimCard | None,
    decision: ClassificationDecision | None,
    card: PaperCard | None,
    paper_memory: dict[str, Any],
    layout: dict[str, Any],
    topic: str | None,
    idea: str | None,
    output_language: str,
    record_usage: Any,
    record_agent_run: Any,
    read_mode: str,
    cache_dir: Path | None,
) -> dict[str, Any]:
    user_prompt = build_report_plan_prompt(
        paper=paper,
        skim=skim,
        decision=decision,
        card=card,
        paper_memory=paper_memory,
        layout=layout,
        topic=topic,
        idea=idea,
        output_language=output_language,
        read_mode=read_mode,
    )
    key_payload = {
        "version": REPORT_PLAN_PROMPT_VERSION,
        "model": client.config.model,
        "output_language": output_language,
        "read_mode": read_mode,
        "paper_hash": paper.file_hash,
        "prompt_hash": hash_text(REPORT_PLAN_SYSTEM_PROMPT + "\n" + user_prompt),
        "schema_hash": hash_json_payload(REPORT_PLAN_SCHEMA),
    }
    cache_path = llm_cache_path(cache_dir, "report_plans", paper.paper_id, key_payload)
    cached = read_llm_cache(cache_path)
    if cached and isinstance(cached.get("data"), dict):
        record_agent_run(cache_agent_run(client, paper.paper_id, stage, "report_plan", cache_path))
        return normalize_report_plan(
            cached["data"], paper=paper, decision=decision, paper_memory=paper_memory
        )
    with llm_call_context(
        stage=stage,
        paper_id=paper.paper_id,
        operation="report_plan",
        schema_name="paperlens_report_plan",
    ):
        result = AgentLoop(
            client=client,
            tools=PaperToolRegistry(
                runtime=PaperLensRuntime(artifacts=list_payload(layout.get("pages"))),
                paper_id=paper.paper_id,
                title=paper.canonical_title,
                memory=paper_memory,
                layout_pages=list_payload(layout.get("pages")),
            ),
            session_name="report_plan",
            objective="Plan a natural PaperLens knowledge capsule from PaperMemory. Use tools if the plan needs grounding.",
            final_schema_name="paperlens_report_plan",
            final_schema=REPORT_PLAN_SCHEMA,
            stage=stage,
            paper_id=paper.paper_id,
            trace_path=data_dir / "agent_trace.jsonl",
            system_prompt=REPORT_PLAN_SYSTEM_PROMPT,
        ).run(
            initial_context={
                "paper_id": paper.paper_id,
                "title": paper.canonical_title or "unknown",
                "output_language": output_language,
                "read_mode": read_mode,
                "topic": topic,
                "idea": idea,
                "skim_card": compact_skim_for_report(skim),
                "classification": compact_decision_for_report(decision),
                "paper_card": compact_paper_card_for_report(card),
                "paper_memory": compact_paper_memory_for_report(paper_memory),
                "context_prompt": user_prompt,
            }
        )
    record_usage(stage, result.usage)
    record_agent_run(
        {
            "agent_run_id": f"report_plan_{paper.paper_id}_{uuid.uuid4().hex[:8]}",
            "paper_id": paper.paper_id,
            "stage": stage,
            "operation": "report_plan",
            "provider_kind": client.config.kind,
            "model": client.config.model,
            "usage": result.usage,
            "request_ids": result.request_ids,
            "trace_events": len(result.trace),
            "status": "PASS",
        }
    )
    write_llm_cache(
        cache_path,
        {
            "key": key_payload,
            "data": result.final,
            "usage": result.usage,
            "request_ids": result.request_ids,
            "endpoint": "agent_loop",
        },
    )
    return normalize_report_plan(
        result.final, paper=paper, decision=decision, paper_memory=paper_memory
    )


def generate_report_section(
    *,
    client: JsonLlmClient,
    data_dir: Path,
    stage: str,
    paper: PaperRecord,
    paper_memory: dict[str, Any],
    layout: dict[str, Any],
    plan: dict[str, Any],
    section_plan: dict[str, Any],
    previous_summaries: list[dict[str, str]],
    output_language: str,
    record_usage: Any,
    record_agent_run: Any,
    read_mode: str,
    cache_dir: Path | None,
    section_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    user_prompt = build_report_section_prompt(
        paper=paper,
        paper_memory=paper_memory,
        layout=layout,
        plan=plan,
        section_plan=section_plan,
        previous_summaries=previous_summaries,
        output_language=output_language,
        read_mode=read_mode,
        section_audit=section_audit,
    )
    section_id = str(section_plan.get("section_id") or "section")
    key_payload = {
        "version": REPORT_SECTION_PROMPT_VERSION,
        "model": client.config.model,
        "output_language": output_language,
        "read_mode": read_mode,
        "paper_hash": paper.file_hash,
        "section_id": section_id,
        "plan_hash": hash_json_payload(plan),
        "previous_hash": hash_json_payload(previous_summaries),
        "audit_hash": hash_json_payload(section_audit or {}),
        "prompt_hash": hash_text(REPORT_SECTION_SYSTEM_PROMPT + "\n" + user_prompt),
        "schema_hash": hash_json_payload(REPORT_SECTION_SCHEMA),
    }
    cache_path = llm_cache_path(cache_dir, "report_sections", paper.paper_id, key_payload)
    cached = read_llm_cache(cache_path)
    if cached and isinstance(cached.get("data"), dict):
        record_agent_run(
            cache_agent_run(
                client, paper.paper_id, stage, f"report_section_{section_id}", cache_path
            )
        )
        return normalize_report_section(cached["data"], section_plan=section_plan)
    with llm_call_context(
        stage=stage,
        paper_id=paper.paper_id,
        operation="report_section",
        section_id=section_id,
        schema_name="paperlens_report_section",
    ):
        result = AgentLoop(
            client=client,
            tools=PaperToolRegistry(
                runtime=PaperLensRuntime(artifacts=list_payload(layout.get("pages"))),
                paper_id=paper.paper_id,
                title=paper.canonical_title,
                memory=paper_memory,
                layout_pages=list_payload(layout.get("pages")),
            ),
            session_name=f"report_section_{section_id}",
            objective="Write this report section as a clear article fragment from PaperMemory. Use tools if evidence or wording needs grounding.",
            final_schema_name="paperlens_report_section",
            final_schema=REPORT_SECTION_SCHEMA,
            stage=stage,
            paper_id=paper.paper_id,
            trace_path=data_dir / "agent_trace.jsonl",
            system_prompt=REPORT_SECTION_SYSTEM_PROMPT,
        ).run(
            initial_context={
                "paper_id": paper.paper_id,
                "title": paper.canonical_title or "unknown",
                "output_language": output_language,
                "read_mode": read_mode,
                "paper_memory": compact_paper_memory_for_report(paper_memory),
                "report_plan": compact_report_plan(plan),
                "section_to_write": section_plan,
                "previous_section_summaries": previous_summaries[-4:],
                "previous_section_audit": section_audit or {},
                "context_prompt": user_prompt,
            }
        )
    record_usage(stage, result.usage)
    record_agent_run(
        {
            "agent_run_id": f"report_section_{paper.paper_id}_{section_id}_{uuid.uuid4().hex[:8]}",
            "paper_id": paper.paper_id,
            "stage": stage,
            "operation": f"report_section_{section_id}",
            "provider_kind": client.config.kind,
            "model": client.config.model,
            "usage": result.usage,
            "request_ids": result.request_ids,
            "trace_events": len(result.trace),
            "status": "PASS",
        }
    )
    write_llm_cache(
        cache_path,
        {
            "key": key_payload,
            "data": result.final,
            "usage": result.usage,
            "request_ids": result.request_ids,
            "endpoint": "agent_loop",
        },
    )
    return normalize_report_section(result.final, section_plan=section_plan)


def audit_report_section(
    *,
    client: JsonLlmClient,
    data_dir: Path,
    stage: str,
    paper: PaperRecord,
    paper_memory: dict[str, Any],
    layout: dict[str, Any],
    plan: dict[str, Any],
    section_plan: dict[str, Any],
    section: dict[str, Any],
    output_language: str,
    record_usage: Any,
    record_agent_run: Any,
    read_mode: str,
    cache_dir: Path | None,
) -> dict[str, Any]:
    user_prompt = build_report_section_audit_prompt(
        paper=paper,
        paper_memory=paper_memory,
        layout=layout,
        plan=plan,
        section_plan=section_plan,
        section=section,
        output_language=output_language,
        read_mode=read_mode,
    )
    section_id = str(section_plan.get("section_id") or section.get("section_id") or "section")
    key_payload = {
        "version": REPORT_SECTION_AUDIT_PROMPT_VERSION,
        "model": client.config.model,
        "output_language": output_language,
        "read_mode": read_mode,
        "paper_hash": paper.file_hash,
        "section_id": section_id,
        "section_hash": hash_json_payload(section),
        "plan_hash": hash_json_payload(plan),
        "prompt_hash": hash_text(REPORT_SECTION_AUDITOR_SYSTEM_PROMPT + "\n" + user_prompt),
        "schema_hash": hash_json_payload(REPORT_SECTION_AUDIT_SCHEMA),
    }
    cache_path = llm_cache_path(cache_dir, "report_section_audits", paper.paper_id, key_payload)
    cached = read_llm_cache(cache_path)
    if cached and isinstance(cached.get("data"), dict):
        record_agent_run(
            cache_agent_run(
                client, paper.paper_id, stage, f"report_section_audit_{section_id}", cache_path
            )
        )
        return enforce_section_depth_audit(
            normalize_report_section_audit(cached["data"]),
            section=section,
            section_plan=section_plan,
        )
    with llm_call_context(
        stage=stage,
        paper_id=paper.paper_id,
        operation="report_section_audit",
        section_id=section_id,
        schema_name="paperlens_report_section_audit",
    ):
        result = AgentLoop(
            client=client,
            tools=PaperToolRegistry(
                runtime=PaperLensRuntime(artifacts=list_payload(layout.get("pages"))),
                paper_id=paper.paper_id,
                title=paper.canonical_title,
                memory=paper_memory,
                layout_pages=list_payload(layout.get("pages")),
            ),
            session_name=f"report_section_audit_{section_id}",
            objective="Audit one report section against PaperMemory and paper evidence. Use tools when a claim needs checking.",
            final_schema_name="paperlens_report_section_audit",
            final_schema=REPORT_SECTION_AUDIT_SCHEMA,
            stage=stage,
            paper_id=paper.paper_id,
            trace_path=data_dir / "agent_trace.jsonl",
            system_prompt=REPORT_SECTION_AUDITOR_SYSTEM_PROMPT,
        ).run(
            initial_context={
                "paper_id": paper.paper_id,
                "title": paper.canonical_title or "unknown",
                "output_language": output_language,
                "read_mode": read_mode,
                "paper_memory": compact_paper_memory_for_report(paper_memory),
                "report_plan": compact_report_plan(plan),
                "section_plan": section_plan,
                "generated_section": section,
                "context_prompt": user_prompt,
            }
        )
    record_usage(stage, result.usage)
    record_agent_run(
        {
            "agent_run_id": f"report_section_audit_{paper.paper_id}_{section_id}_{uuid.uuid4().hex[:8]}",
            "paper_id": paper.paper_id,
            "stage": stage,
            "operation": f"report_section_audit_{section_id}",
            "provider_kind": client.config.kind,
            "model": client.config.model,
            "usage": result.usage,
            "request_ids": result.request_ids,
            "trace_events": len(result.trace),
            "status": "PASS",
        }
    )
    write_llm_cache(
        cache_path,
        {
            "key": key_payload,
            "data": result.final,
            "usage": result.usage,
            "request_ids": result.request_ids,
            "endpoint": "agent_loop",
        },
    )
    return enforce_section_depth_audit(
        normalize_report_section_audit(result.final),
        section=section,
        section_plan=section_plan,
    )


def cache_agent_run(
    client: JsonLlmClient, paper_id: str, stage: str, prefix: str, cache_path: Path | None
) -> dict[str, Any]:
    return {
        "agent_run_id": f"{prefix}_{paper_id}_cache",
        "paper_id": paper_id,
        "stage": stage,
        "provider_kind": client.config.kind,
        "model": client.config.model,
        "status": "CACHE_HIT",
        "cache": str(cache_path) if cache_path else "",
    }
