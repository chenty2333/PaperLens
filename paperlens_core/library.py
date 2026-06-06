from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from paperlens_core.agent_loop import AgentLoop, AgentToolObservation, AgentToolRequest
from paperlens_core.agents.llm import JsonLlmClient
from paperlens_core.config import CoreConfig
from paperlens_core.library_graph import (
    build_graph_summary_search_text,
    compact_graph_summary_for_agent,
    compact_graph_summary_for_index,
    first_graph_label,
    graph_node_labels,
    graph_provenance,
    graph_summary_tags,
    normalize_graph_claims,
    normalize_graph_concepts,
    normalize_graph_evidence,
    read_core_v2_graph_summary,
)


APP_NAME = "PaperLens"
INTERNAL_DIRNAME = ".paperlens"
LIBRARY_RECORD_SCHEMA_VERSION = "paperlens.library_record.v1"
LIBRARY_RECORD_FILENAME = "library_records.jsonl"
SEARCH_INDEX_SCHEMA_VERSION = "paperlens_search_index.v1"
LIBRARY_ASK_PROMPT_VERSION = "library-ask-v4-envelope"


SEARCH_QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "内存管理": ("memory", "management"),
    "内存": ("memory",),
    "显存": ("gpu", "memory", "vram"),
    "缓存淘汰": ("cache", "eviction"),
    "缓存": ("cache",),
    "淘汰": ("eviction",),
    "调度": ("scheduling", "scheduler"),
    "系统调度": ("scheduling", "scheduler"),
    "集群": ("cluster",),
    "分布式": ("distributed",),
    "一致性": ("consensus",),
    "远程调用": ("rpc", "remote procedure call"),
    "大模型": ("llm", "large language model"),
    "推理": ("inference", "serving"),
    "训练": ("training",),
    "存储": ("storage",),
    "文件系统": ("file system", "filesystem"),
    "操作系统": ("operating system", "os"),
    "虚拟化": ("virtualization",),
    "验证": ("verification",),
    "测试": ("testing",),
}


LIBRARY_ASK_SYSTEM_PROMPT = """
You are the PaperLens library QA node.
Respect the runtime contract. Return JSON only.
""".strip()


LIBRARY_ASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "answer_markdown",
        "related_papers",
        "cited_source_ids",
        "confidence",
        "source_attribution",
    ],
    "properties": {
        "answer_markdown": {"type": "string"},
        "related_papers": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["paper_id", "title", "report_path", "why_related"],
                "properties": {
                    "paper_id": {"type": "string"},
                    "title": {"type": "string"},
                    "report_path": {"type": "string"},
                    "why_related": {"type": "string"},
                },
            },
        },
        "cited_source_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "source_attribution": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "paper_claims",
                "cross_paper_synthesis",
                "background_context",
                "evidence_limits",
            ],
            "properties": {
                "paper_claims": {"type": "array", "items": {"type": "string"}},
                "cross_paper_synthesis": {"type": "array", "items": {"type": "string"}},
                "background_context": {"type": "array", "items": {"type": "string"}},
                "evidence_limits": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


def write_paperlens_library(
    *,
    output_dir: Path,
    rows: list[dict[str, Any]],
    topic: str | None,
    idea: str | None,
) -> list[Path]:
    records = [
        build_library_record(enrich_row_with_core_v2_graph(output_dir=output_dir, row=row))
        for row in rows
    ]
    return write_library_records(output_dir=output_dir, records=records, topic=topic, idea=idea)


def enrich_row_with_core_v2_graph(*, output_dir: Path, row: dict[str, Any]) -> dict[str, Any]:
    if dict_value(row.get("core_v2_graph_summary")):
        return row
    paper = dump_model(row.get("paper"))
    paper_id = string_or_empty(paper.get("paper_id")) or string_or_empty(row.get("paper_id"))
    if not paper_id:
        return row
    graph_summary = read_core_v2_graph_summary(output_dir, paper_id)
    if not graph_summary:
        return row
    enriched = dict(row)
    enriched["core_v2_graph_summary"] = graph_summary
    return enriched


def rebuild_library_from_output(output_dir: Path) -> list[Path]:
    data_dir = paperlens_data_dir(output_dir)
    rows: list[dict[str, Any]] = []
    for graph_root in sorted((data_dir / "core" / "v2").glob("*")):
        if not graph_root.is_dir():
            continue
        paper_id = graph_root.name
        graph_summary = read_core_v2_graph_summary(output_dir, paper_id)
        if not graph_summary:
            continue
        metadata = dict_value(graph_summary.get("metadata"))
        report_path = first_existing_report(output_dir, paper_id)
        rows.append(
            {
                "paper": {
                    "paper_id": paper_id,
                    "canonical_title": metadata.get("title") or paper_id,
                    "year": metadata.get("year"),
                },
                "decision": {
                    "paper_id": paper_id,
                    "class_label": metadata.get("grade") or "HOLD",
                },
                "report_name": Path(report_path).name if report_path else f"{paper_id}.md",
                "core_graph_report_name": Path(report_path).name if report_path else f"{paper_id}.md",
                "report_title": metadata.get("title") or paper_id,
                "core_v2_graph_summary": graph_summary,
            }
        )
    return write_paperlens_library(
        output_dir=output_dir,
        rows=rows,
        topic=None,
        idea=None,
    )


def write_library_records(
    *,
    output_dir: Path,
    records: list[dict[str, Any]],
    topic: str | None,
    idea: str | None,
) -> list[Path]:
    root = library_dir(output_dir)
    index_root = root / "index"
    root.mkdir(parents=True, exist_ok=True)
    index_root.mkdir(parents=True, exist_ok=True)

    records_path = root / LIBRARY_RECORD_FILENAME
    records_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records
    )
    records_path.write_text(records_text, encoding="utf-8")

    search_index = build_search_index(records)
    index_path = index_root / "search_index.json"
    index_path.write_text(
        json.dumps(search_index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    return [records_path, index_path]


def rebuild_library_index(output_dir: Path) -> Path:
    records = read_library_records(output_dir)
    index_path = library_dir(output_dir) / "index" / "search_index.json"
    write_json(index_path, build_search_index(records))
    return index_path


def doctor_library(output_dir: Path) -> dict[str, Any]:
    records_path = library_dir(output_dir) / LIBRARY_RECORD_FILENAME
    index_path = library_dir(output_dir) / "index" / "search_index.json"
    records = read_library_records(output_dir)
    raw_lines = (
        records_path.read_text(encoding="utf-8").splitlines() if records_path.exists() else []
    )
    unsupported_versions = []
    duplicate_ids = []
    record_issues: dict[str, list[str]] = {}
    seen_ids = set()
    for line in raw_lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            unsupported_versions.append("invalid-json")
            continue
        if not isinstance(value, dict):
            unsupported_versions.append("non-object")
            continue
        version = value.get("schema_version")
        if version != LIBRARY_RECORD_SCHEMA_VERSION:
            unsupported_versions.append(str(version or "missing"))
        paper_id = value.get("paper_id")
        if paper_id in seen_ids:
            duplicate_ids.append(paper_id)
        elif paper_id:
            seen_ids.add(paper_id)
        issues = validate_library_record(value)
        if issues:
            record_issues[str(paper_id or "unknown")] = issues
    return {
        "status": "PASS"
        if records_path.exists()
        and not unsupported_versions
        and not duplicate_ids
        and not record_issues
        else "WARN",
        "records_path": str(records_path),
        "index_path": str(index_path),
        "record_count": len(records),
        "schema_version": LIBRARY_RECORD_SCHEMA_VERSION,
        "unsupported_versions": sorted(set(unsupported_versions)),
        "duplicate_paper_ids": sorted(set(duplicate_ids)),
        "record_issues": record_issues,
        "index_exists": index_path.exists(),
        "can_rebuild_index": records_path.exists(),
    }


def build_library_record(row: dict[str, Any]) -> dict[str, Any]:
    paper = dump_model(row.get("paper"))
    skim = dump_model(row.get("skim"))
    decision = dump_model(row.get("decision"))
    graph_summary = dict_value(row.get("core_v2_graph_summary"))
    if not graph_summary:
        paper_id = string_or_empty(paper.get("paper_id")) or string_or_empty(row.get("paper_id"))
        raise ValueError(f"Library record requires core_v2_graph_summary for {paper_id or 'unknown'}")

    paper_id = string_or_empty(paper.get("paper_id")) or string_or_empty(row.get("paper_id"))
    title = (
        string_or_none(row.get("report_title"))
        or string_or_none(paper.get("canonical_title"))
        or string_or_none(dict_value(graph_summary.get("metadata")).get("title"))
        or paper_id
    )
    report_name = string_or_none(row.get("report_name")) or f"{paper_id}.md"
    grade = (
        string_or_none(decision.get("class_label"))
        or string_or_none(dict_value(graph_summary.get("metadata")).get("grade"))
        or "HOLD"
    )
    graph_brief = first_graph_label(graph_summary, "problem_nodes", "claim_nodes")
    brief = graph_brief or string_or_none(skim.get("problem")) or ""
    concepts = normalize_graph_concepts(graph_summary)
    conceptual_bridge: dict[str, Any] = {"needed": False, "terms": []}
    claims = normalize_graph_claims(graph_summary)
    mechanisms = graph_node_labels(graph_summary, "mechanism_nodes", "implementation_nodes")
    evidence_model = graph_node_labels(graph_summary, "evaluation_nodes", "result_nodes")
    limits = graph_node_labels(graph_summary, "limitation_nodes")
    evidence_items = normalize_graph_evidence(graph_summary)
    problem = first_graph_label(graph_summary, "problem_nodes")
    core_idea = graph_brief or brief
    value = brief
    questions: list[str] = []
    uncertainties = normalized_string_list(
        dict_value(graph_summary.get("quality")).get("unresolved_audit_findings")
    )
    tags = infer_tags(
        title=title,
        concepts=concepts,
        mechanisms=mechanisms,
        claims=claims,
        skim=skim,
    )
    tags = merge_unique(tags, graph_summary_tags(graph_summary))[:10]
    memory = {
        "brief": brief,
        "core_idea": core_idea,
        "problem": problem,
        "mechanism": join_sentences(mechanisms),
        "mechanism_steps": mechanisms,
        "evidence_summary": join_sentences(evidence_model),
        "evidence_items": evidence_items,
        "limits": limits,
        "value": value,
        "concepts": concepts,
        "conceptual_bridge": conceptual_bridge,
        "claims": claims,
        "questions_supported": questions,
        "qa_seed_questions": questions,
        "uncertainties": uncertainties,
        "reader_takeaways": reader_takeaways(
            brief=brief,
            core_idea=core_idea,
            claims=claims,
        ),
    }
    evidence_source_ids = unique_strings(
        item.get("source_id")
        for item in evidence_items
        if isinstance(item, dict) and string_or_none(item.get("source_id"))
    )
    memory_quality = {
        "claim_count": len(claims),
        "concept_count": len(concepts),
        "evidence_item_count": len(memory["evidence_items"]),
        "evidence_source_id_count": len(evidence_source_ids),
        "has_core_abstraction": bool(memory["core_idea"]),
        "graph_publish_status": dict_value(graph_summary.get("quality")).get("publish_status"),
        "graph_evidence_coverage": dict_value(graph_summary.get("quality")).get(
            "evidence_coverage"
        ),
        "graph_reading_required_output_coverage": dict_value(graph_summary.get("quality")).get(
            "reading_required_output_coverage"
        ),
        "graph_fact_node_count": dict_value(graph_summary.get("quality")).get("fact_node_count"),
    }
    outputs = {
        "briefing_md": f"papers/{report_name}",
    }
    core_graph_report_name = string_or_none(row.get("core_graph_report_name"))
    if core_graph_report_name:
        outputs["core_graph_report_md"] = f"papers/{core_graph_report_name}"
    record = {
        "schema_version": LIBRARY_RECORD_SCHEMA_VERSION,
        "paper_id": paper_id,
        "title": title,
        "grade": grade,
        "recommendation": read_recommendation(grade),
        "tags": tags,
        "source": {
            "pdf_sha256": string_or_none(paper.get("file_hash")),
            "original_path": string_or_none(paper.get("file_path")),
            "pages": int_or_none(paper.get("page_count")),
            "venue": string_or_none(paper.get("venue")),
            "year": int_or_none(paper.get("year")),
            "doi": string_or_none(paper.get("doi")),
            "arxiv_id": string_or_none(paper.get("arxiv_id")),
        },
        "model_trace": {
            "created_at": now_iso(),
            "library_record_schema": LIBRARY_RECORD_SCHEMA_VERSION,
            "core_v2_graph_summary_schema": graph_summary.get("schema_version"),
        },
        "memory": memory,
        "graph_summary": graph_summary,
        "quality": memory_quality,
        "provenance": {
            "source_ids": evidence_source_ids,
            "core_v2": graph_provenance(graph_summary),
        },
        "outputs": outputs,
        "search_text": build_record_search_text(
            title=title,
            brief=brief,
            tags=tags,
            memory=memory,
            graph_summary=graph_summary,
        ),
    }
    record["record_hash"] = hash_json_payload(
        {
            "schema_version": record["schema_version"],
            "paper_id": paper_id,
            "source": record["source"],
            "memory": memory,
            "graph_summary": graph_summary,
            "outputs": record["outputs"],
        }
    )
    return record


def build_search_index(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SEARCH_INDEX_SCHEMA_VERSION,
        "derived_from": LIBRARY_RECORD_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "records": [
            {
                "paper_id": record["paper_id"],
                "title": record["title"],
                "grade": record["grade"],
                "tags": record.get("tags", []),
                "report_path": record.get("outputs", {}).get("briefing_md"),
                "graph": compact_graph_summary_for_index(record.get("graph_summary")),
                "record_hash": record.get("record_hash"),
                "tokens": tokenize_for_search(record.get("search_text", ""))[:300],
            }
            for record in records
        ],
    }


def render_library_markdown(
    *,
    records: list[dict[str, Any]],
    topic: str | None,
    idea: str | None,
) -> str:
    ranked = sorted(records, key=library_sort_key)
    lines = [
        "# PaperLens Library",
        "",
        f"已读 {len(records)} 篇论文。这里是入口目录；每篇的完整读后汇报在 `papers/`。",
    ]
    if topic or idea:
        lines.extend(["", "阅读目标：" + "；".join(item for item in [topic, idea] if item)])
    lines.extend(["", "## 论文", ""])
    if not ranked:
        lines.append("暂无论文。")
    for record in ranked:
        report_path = record.get("outputs", {}).get("briefing_md") or ""
        brief = compact_text(record.get("memory", {}).get("brief"), max_chars=140)
        title = record.get("title") or record.get("paper_id")
        lines.append(f"- [{record.get('grade', 'HOLD')}] [{title}](./{report_path}) - {brief}")

    tag_map: dict[str, list[str]] = {}
    for record in ranked:
        for tag in record.get("tags", [])[:6]:
            tag_map.setdefault(str(tag), []).append(
                str(record.get("title") or record.get("paper_id"))
            )
    if tag_map:
        lines.extend(["", "## 标签", ""])
        for tag in sorted(tag_map):
            titles = "；".join(tag_map[tag][:5])
            lines.append(f"- {tag}: {titles}")
    return "\n".join(lines).rstrip() + "\n"


def search_library(
    *,
    output_dir: Path,
    query: str,
    limit: int = 8,
) -> dict[str, Any]:
    records = ensure_library_records(output_dir)
    matches = search_library_records(records, query=query, limit=limit)
    return {"query": query, "matches": matches}


class LibraryToolRegistry:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.title = "PaperLens Library"

    def tool_descriptions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "library.search",
                "description": "Search papers that PaperLens has already read. Returns compact records only.",
                "arguments": {"query": "string", "limit": "optional integer"},
            },
            {
                "name": "library.get_record",
                "description": "Read one local library record, including compact claims, evidence, concepts, outputs, and provenance.",
                "arguments": {"paper_id": "string"},
            },
        ]

    def execute(self, request: AgentToolRequest) -> AgentToolObservation:
        try:
            if request.tool == "library.search":
                result = self._search(request.arguments)
            elif request.tool == "library.get_record":
                result = self._get_record(request.arguments)
            else:
                raise ValueError(f"Unknown tool: {request.tool}")
            return AgentToolObservation(
                id=request.id,
                tool=request.tool,
                arguments=request.arguments,
                result=result,
            )
        except Exception as exc:
            return AgentToolObservation(
                id=request.id,
                tool=request.tool,
                arguments=request.arguments,
                ok=False,
                error=str(exc),
                result={},
            )

    def _search(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = string_or_empty(arguments.get("query"))
        limit = int_or_none(arguments.get("limit")) or 8
        limit = max(1, min(limit, 20))
        matches = search_library_records(self.records, query=query, limit=limit, public=False)
        results = [
            compact_library_record_for_agent(hit["paper"], score=hit["score"], include_memory=False)
            for hit in matches
        ]
        return {
            "tool": "library.search",
            "query": query,
            "results": results,
            "source_ids": library_result_source_ids(results),
        }

    def _get_record(self, arguments: dict[str, Any]) -> dict[str, Any]:
        paper_id = string_or_empty(arguments.get("paper_id"))
        record = find_library_record(self.records, paper_id)
        results = (
            [compact_library_record_for_agent(record, include_memory=True)]
            if record is not None
            else []
        )
        return {
            "tool": "library.get_record",
            "query": paper_id,
            "results": results,
            "source_ids": library_result_source_ids(results),
        }


def answer_library_question(
    *,
    output_dir: Path,
    config: CoreConfig,
    question: str,
    limit: int = 8,
    chat_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    records = ensure_library_records(output_dir)
    matches = search_library_records(records, query=question, limit=limit, public=False)
    if config.offline_debug or config.provider.kind == "none":
        related = [public_search_record(hit["paper"]) for hit in matches]
        cited_source_ids = library_result_source_ids(
            [
                compact_library_record_for_agent(
                    hit["paper"],
                    score=hit["score"],
                    include_memory=False,
                )
                for hit in matches[:8]
            ]
        )
        return {
            "answer_markdown": render_offline_library_answer(question, matches),
            "related_papers": related,
            "cited_source_ids": cited_source_ids[:16],
            "confidence": "low",
            "cache_hit": False,
        }
    config.validate_agentic_run()
    client = JsonLlmClient(
        config.provider,
        ledger_path=output_dir / ".paperlens" / "data" / "model_calls.jsonl",
        run_id=f"library_qa_{hash_text(question)[:12]}",
    )
    user_prompt = build_library_ask_prompt(
        question=question, matches=matches, chat_history=chat_history or []
    )
    cache_path = library_answer_cache_path(
        output_dir,
        {
            "version": LIBRARY_ASK_PROMPT_VERSION,
            "agent_loop": True,
            "model": config.provider.model,
            "question": question,
            "chat_history_hash": hash_json_payload(normalize_chat_history(chat_history or [])),
            "prompt_hash": hash_text(LIBRARY_ASK_SYSTEM_PROMPT + "\n" + user_prompt),
            "schema_hash": hash_json_payload(LIBRARY_ASK_SCHEMA),
            "record_hashes": [hit["paper"].get("record_hash") for hit in matches],
        },
    )
    cached = read_json(cache_path)
    if isinstance(cached.get("data"), dict):
        answer = normalize_library_answer(cached["data"])
        answer["cache_hit"] = True
        answer["usage"] = {}
        return answer
    result = run_library_qa_agent(
        client=client,
        output_dir=output_dir,
        records=records,
        matches=matches,
        question=question,
        chat_history=chat_history or [],
    )
    answer = normalize_library_answer(result.final)
    answer["cache_hit"] = False
    answer["usage"] = result.usage
    write_json(
        cache_path,
        {
            "data": result.final,
            "artifact": result.final_envelope,
            "usage": result.usage,
            "request_ids": result.request_ids,
            "endpoint": "agent_loop",
        },
    )
    return answer


def run_library_qa_agent(
    *,
    client: JsonLlmClient,
    output_dir: Path,
    records: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    question: str,
    chat_history: list[dict[str, Any]],
) -> Any:
    tools = LibraryToolRegistry(records)
    initial_matches = [
        compact_library_record_for_agent(hit["paper"], score=hit["score"], include_memory=False)
        for hit in matches[:8]
    ]
    loop = AgentLoop(
        client=client,
        tools=tools,
        session_name="library_qa",
        objective=(
            "Answer a question over the local PaperLens library. Search or inspect library records "
            "before making paper-specific or cross-paper claims. Use background knowledge for teaching, "
            "but label it separately from local paper evidence."
        ),
        final_artifact_type="library_qa_answer",
        final_data_schema=LIBRARY_ASK_SCHEMA,
        stage="library_qa",
        paper_id="__library__",
        trace_path=paperlens_data_dir(output_dir) / "agent_trace.jsonl",
        system_prompt=LIBRARY_ASK_SYSTEM_PROMPT,
        max_steps=4,
        max_model_calls=4,
        max_tool_calls=8,
        max_tokens=14000,
        timeout_seconds=180.0,
        allowed_tools=("library.search", "library.get_record"),
        input_contract={
            "artifact_type": "library_qa_answer",
            "paper_specific_claims": "must come from graph_summary nodes, claims, or provenance source_ids",
            "report_paths": "navigation only; not evidence",
            "cross_paper_synthesis": "must name which local paper records support it",
            "source_attribution": {
                "paper_claims": "claims from specific local graph records",
                "cross_paper_synthesis": "synthesis across cited local records",
                "background_context": "general knowledge outside local papers",
                "evidence_limits": "missing local graph evidence or uncertainty",
            },
        },
    )
    return loop.run(
        initial_context={
            "question": question,
            "recent_chat_history": normalize_chat_history(chat_history),
            "initial_library_matches": initial_matches,
            "library_record_count": len(records),
        }
    )


def build_library_ask_prompt(
    *,
    question: str,
    matches: list[dict[str, Any]],
    chat_history: list[dict[str, Any]] | None = None,
) -> str:
    compact_records = []
    for hit in matches:
        record = hit["paper"]
        compact_records.append(
            {
                "score": hit["score"],
                "paper_id": record.get("paper_id"),
                "title": record.get("title"),
                "grade": record.get("grade"),
                "tags": record.get("tags"),
                "report_path": record.get("outputs", {}).get("briefing_md"),
                "memory": record.get("memory"),
                "graph_summary": compact_graph_summary_for_agent(record.get("graph_summary")),
                "provenance": record.get("provenance"),
            }
        )
    return "\n\n".join(
        [
            "question:",
            question,
            "recent_chat_history:",
            json.dumps(normalize_chat_history(chat_history or []), ensure_ascii=False),
            "retrieved_library_records:",
            json.dumps(compact_records, ensure_ascii=False),
        ]
    )


def search_library_records(
    records: list[dict[str, Any]],
    *,
    query: str,
    limit: int,
    public: bool = True,
) -> list[dict[str, Any]]:
    query_terms = expand_search_query_terms(query)
    query_tokens = tokenize_for_search(" ".join([query, *query_terms]))
    query_phrases = [
        phrase
        for phrase in [
            normalize_for_search(query),
            *(normalize_for_search(term) for term in query_terms),
        ]
        if len(phrase) >= 2
    ]
    scored = []
    for record in records:
        search_text = normalize_for_search(record.get("search_text", ""))
        title = normalize_for_search(record.get("title", ""))
        tags = normalize_for_search(" ".join(str(tag) for tag in record.get("tags", [])))
        score = 0.0
        for phrase in query_phrases:
            if phrase in search_text:
                score += 8.0
                break
        for token in query_tokens:
            if token in title:
                score += 4.0
            if token in tags:
                score += 3.0
            if token in search_text:
                score += 1.0 + min(2.0, search_text.count(token) * 0.25)
        if score > 0:
            scored.append(
                {
                    "score": round(score, 3),
                    "paper": public_search_record(record) if public else record,
                }
            )
    scored.sort(
        key=lambda item: (
            -item["score"],
            grade_rank(item["paper"].get("grade")),
            item["paper"].get("title", ""),
        )
    )
    return scored[: max(1, min(limit, 20))]


def ensure_library_records(output_dir: Path) -> list[dict[str, Any]]:
    records_path = library_dir(output_dir) / LIBRARY_RECORD_FILENAME
    if not records_path.exists():
        rebuild_library_from_output(output_dir)
    return read_library_records(output_dir)


def read_library_records(output_dir: Path) -> list[dict[str, Any]]:
    path = library_dir(output_dir) / LIBRARY_RECORD_FILENAME
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and value.get("schema_version") == LIBRARY_RECORD_SCHEMA_VERSION:
            records.append(value)
    return records


def validate_library_record(record: dict[str, Any]) -> list[str]:
    issues = []
    if record.get("schema_version") != LIBRARY_RECORD_SCHEMA_VERSION:
        issues.append("unsupported_schema_version")
    for key in ["paper_id", "title", "grade", "memory", "provenance", "outputs", "quality"]:
        if key not in record:
            issues.append(f"missing_{key}")
    memory = dict_value(record.get("memory"))
    required_memory_keys = [
        "brief",
        "core_idea",
        "mechanism_steps",
        "evidence_items",
        "claims",
        "reader_takeaways",
        "qa_seed_questions",
        "uncertainties",
    ]
    for key in required_memory_keys:
        if key not in memory:
            issues.append(f"missing_memory_{key}")
    if not string_or_empty(memory.get("core_idea")):
        issues.append("empty_core_idea")
    if str(record.get("grade")) != "C" and not list_payload(memory.get("claims")):
        issues.append("empty_claims")
    if not isinstance(memory.get("reader_takeaways"), list) or not memory.get("reader_takeaways"):
        issues.append("empty_reader_takeaways")
    if not dict_value(record.get("outputs")).get("briefing_md"):
        issues.append("missing_briefing_output")
    return issues


def public_search_record(record: dict[str, Any]) -> dict[str, Any]:
    memory = dict_value(record.get("memory"))
    return {
        "paper_id": record.get("paper_id"),
        "title": record.get("title"),
        "grade": record.get("grade"),
        "tags": record.get("tags", []),
        "brief": memory.get("brief") or memory.get("core_idea"),
        "core_idea": memory.get("core_idea"),
        "graph_summary": compact_graph_summary_for_index(record.get("graph_summary")),
        "report_path": record.get("outputs", {}).get("briefing_md"),
        "record_hash": record.get("record_hash"),
    }


def find_library_record(records: list[dict[str, Any]], paper_id: str) -> dict[str, Any] | None:
    normalized = paper_id.strip().lower()
    if not normalized:
        return None
    for record in records:
        if str(record.get("paper_id") or "").lower() == normalized:
            return record
    for record in records:
        if str(record.get("title") or "").lower() == normalized:
            return record
    return None


def compact_library_record_for_agent(
    record: dict[str, Any],
    *,
    score: float | None = None,
    include_memory: bool,
) -> dict[str, Any]:
    memory = dict_value(record.get("memory"))
    outputs = dict_value(record.get("outputs"))
    source = dict_value(record.get("source"))
    quality = dict_value(record.get("quality"))
    provenance = dict_value(record.get("provenance"))
    graph_summary = dict_value(record.get("graph_summary"))
    core_v2_provenance = graph_provenance(graph_summary)
    record_source_ids = unique_strings(
        [
            *raw_list_payload(provenance.get("source_ids")),
            *raw_list_payload(core_v2_provenance.get("source_ids")),
        ]
    )
    payload: dict[str, Any] = {
        "paper_id": record.get("paper_id"),
        "title": record.get("title"),
        "grade": record.get("grade"),
        "source_ids": record_source_ids[:24],
        "tags": record.get("tags", [])[:10] if isinstance(record.get("tags"), list) else [],
        "brief": compact_text(memory.get("brief") or memory.get("core_idea"), max_chars=380),
        "core_idea": compact_text(memory.get("core_idea"), max_chars=420),
        "graph_summary": compact_graph_summary_for_agent(graph_summary),
        "report_path": outputs.get("briefing_md"),
        "source": {
            "venue": source.get("venue"),
            "year": source.get("year"),
            "pages": source.get("pages"),
            "doi": source.get("doi"),
            "arxiv_id": source.get("arxiv_id"),
        },
        "quality": {
            "claim_count": quality.get("claim_count"),
            "evidence_item_count": quality.get("evidence_item_count"),
            "graph_publish_status": quality.get("graph_publish_status"),
            "graph_evidence_coverage": quality.get("graph_evidence_coverage"),
            "graph_reading_required_output_coverage": quality.get(
                "graph_reading_required_output_coverage"
            ),
        },
    }
    if score is not None:
        payload["score"] = score
    if include_memory:
        payload["memory"] = {
            "problem": compact_text(memory.get("problem"), max_chars=500),
            "mechanism": compact_text(memory.get("mechanism"), max_chars=900),
            "mechanism_steps": [
                compact_text(item, max_chars=280)
                for item in normalized_string_list(memory.get("mechanism_steps"))[:8]
            ],
            "evidence_summary": compact_text(memory.get("evidence_summary"), max_chars=900),
            "limits": [
                compact_text(item, max_chars=260)
                for item in normalized_string_list(memory.get("limits"))[:8]
            ],
            "concepts": memory.get("concepts", [])[:12]
            if isinstance(memory.get("concepts"), list)
            else [],
            "conceptual_bridge": memory.get("conceptual_bridge")
            if isinstance(memory.get("conceptual_bridge"), dict)
            else {},
            "claims": memory.get("claims", [])[:12]
            if isinstance(memory.get("claims"), list)
            else [],
            "evidence_items": memory.get("evidence_items", [])[:12]
            if isinstance(memory.get("evidence_items"), list)
            else [],
            "uncertainties": [
                compact_text(item, max_chars=260)
                for item in normalized_string_list(memory.get("uncertainties"))[:8]
            ],
        }
        payload["provenance"] = {
            "source_ids": record_source_ids[:24],
            "core_v2": core_v2_provenance,
        }
    return payload


def library_result_source_ids(results: list[dict[str, Any]]) -> list[str]:
    source_ids = []
    for result in results:
        if not isinstance(result, dict):
            continue
        source_ids.extend(raw_list_payload(result.get("source_ids")))
        provenance = dict_value(result.get("provenance"))
        source_ids.extend(raw_list_payload(provenance.get("source_ids")))
        graph_summary = dict_value(result.get("graph_summary"))
        source_ids.extend(raw_list_payload(graph_provenance(graph_summary).get("source_ids")))
    return unique_strings(source_ids)


def render_offline_library_answer(question: str, matches: list[dict[str, Any]]) -> str:
    if not matches:
        return f"本地图索引里没有找到和“{question}”明显相关的论文。"
    lines = [f"离线模式下无法调用模型综合回答。和“{question}”最相关的论文是："]
    for hit in matches[:5]:
        paper = hit["paper"]
        memory = dict_value(paper.get("memory"))
        brief = paper.get("brief") or memory.get("brief") or memory.get("core_idea") or ""
        report_path = paper.get("report_path") or paper.get("outputs", {}).get("briefing_md")
        lines.append(f"- {paper.get('title')}（{paper.get('grade')}）：{brief} [{report_path}]")
    return "\n".join(lines)


def normalize_library_answer(data: dict[str, Any]) -> dict[str, Any]:
    related = []
    raw_related = data.get("related_papers") if isinstance(data.get("related_papers"), list) else []
    for item in raw_related:
        if not isinstance(item, dict):
            continue
        paper_id = string_or_empty(item.get("paper_id"))
        title = string_or_empty(item.get("title"))
        report_path = string_or_empty(item.get("report_path"))
        why_related = string_or_empty(item.get("why_related"))
        if paper_id and title:
            related.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "report_path": report_path,
                    "why_related": why_related,
                }
            )
    confidence = string_or_empty(data.get("confidence")) or "low"
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    answer_markdown = string_or_empty(data.get("answer_markdown"))[:6000]
    cited_source_ids = unique_strings(raw_list_payload(data.get("cited_source_ids")))[:16]
    return {
        "answer_markdown": answer_markdown,
        "related_papers": related[:8],
        "cited_source_ids": cited_source_ids,
        "confidence": confidence,
        "source_attribution": normalize_library_source_attribution(
            data.get("source_attribution"),
            answer=answer_markdown,
            confidence=confidence,
        ),
    }


def normalize_library_source_attribution(
    value: Any,
    *,
    answer: str,
    confidence: str,
) -> dict[str, list[str]]:
    raw = dict_value(value)
    result = {
        "paper_claims": normalized_string_list(raw.get("paper_claims"))[:8],
        "cross_paper_synthesis": normalized_string_list(raw.get("cross_paper_synthesis"))[:8],
        "background_context": normalized_string_list(raw.get("background_context"))[:8],
        "evidence_limits": normalized_string_list(raw.get("evidence_limits"))[:8],
    }
    if not any(result.values()) and answer:
        result["cross_paper_synthesis"] = [answer[:280]]
    if confidence == "low" and not result["evidence_limits"]:
        result["evidence_limits"] = ["本地图索引证据不足或模型未明确给出来源边界。"]
    return result


def library_answer_cache_path(output_dir: Path, key_payload: dict[str, Any]) -> Path:
    key = hash_json_payload(key_payload)[:24]
    return output_dir / INTERNAL_DIRNAME / "cache" / "library_answers" / f"{key}.json"


def library_dir(output_dir: Path) -> Path:
    return output_dir / INTERNAL_DIRNAME / "library"


def paperlens_data_dir(output_dir: Path) -> Path:
    return output_dir / INTERNAL_DIRNAME / "data"


def first_existing_report(output_dir: Path, paper_id: str) -> str:
    papers_dir = output_dir / "papers"
    if not papers_dir.exists():
        return ""
    exact = papers_dir / f"{paper_id}.md"
    if exact.exists():
        return f"papers/{exact.name}"
    matches = sorted(papers_dir.glob(f"{paper_id}_*.md"))
    return f"papers/{matches[0].name}" if matches else ""


def dump_model(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_payload(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def raw_list_payload(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def merge_unique(first: list[str], second: list[str]) -> list[str]:
    result = []
    for item in [*first, *second]:
        text = string_or_empty(item)
        if text and text not in result:
            result.append(text)
    return result


def unique_strings(values: Any) -> list[str]:
    result = []
    for item in values or []:
        text = string_or_none(item)
        if text and text not in result:
            result.append(text)
    return result


def reader_takeaways(
    *,
    brief: str,
    core_idea: str,
    claims: list[dict[str, Any]],
) -> list[str]:
    candidates = [
        core_idea,
        brief,
    ] + [string_or_empty(claim.get("claim")) for claim in claims[:3]]
    takeaways = []
    for candidate in candidates:
        cleaned = compact_text(candidate, max_chars=220)
        if cleaned and cleaned not in takeaways:
            takeaways.append(cleaned)
        if len(takeaways) >= 5:
            break
    return takeaways


def infer_tags(
    *,
    title: str,
    concepts: list[dict[str, str]],
    mechanisms: list[str],
    claims: list[dict[str, Any]],
    skim: dict[str, Any],
) -> list[str]:
    tags = []
    for source in [
        title,
        " ".join(concept["term"] for concept in concepts),
        " ".join(mechanisms),
        " ".join(str(claim.get("claim", "")) for claim in claims),
        string_or_empty(skim.get("method_type")),
        string_or_empty(skim.get("system_scope")),
    ]:
        for token in tokenize_for_search(source):
            if token.isdigit() or len(token) < 3:
                continue
            if token not in tags:
                tags.append(token)
            if len(tags) >= 10:
                return tags
    return tags


def build_record_search_text(
    *,
    title: str,
    brief: str,
    tags: list[str],
    memory: dict[str, Any],
    graph_summary: dict[str, Any] | None = None,
) -> str:
    parts = [title, brief, " ".join(tags)]
    for value in memory.values():
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(value, list):
            parts.append(json.dumps(value, ensure_ascii=False))
        elif isinstance(value, dict):
            parts.append(json.dumps(value, ensure_ascii=False))
    if graph_summary:
        parts.append(build_graph_summary_search_text(graph_summary))
    return "\n".join(parts)


def tokenize_for_search(text: Any) -> list[str]:
    normalized = normalize_for_search(str(text or ""))
    tokens = re.findall(r"[a-z0-9_+.-]{2,}|[\u4e00-\u9fff]{2,}", normalized)
    expanded: list[str] = []
    for token in tokens:
        expanded.append(token)
        if re.fullmatch(r"[\u4e00-\u9fff]+", token):
            expanded.extend(token[index : index + 2] for index in range(max(0, len(token) - 1)))
    return list(dict.fromkeys(item for item in expanded if item))


def expand_search_query_terms(text: Any) -> list[str]:
    normalized = normalize_for_search(str(text or ""))
    expansions: list[str] = []
    for term, aliases in SEARCH_QUERY_EXPANSIONS.items():
        if term in normalized:
            expansions.extend(aliases)
    return list(dict.fromkeys(expansions))


def normalize_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def grade_rank(grade: Any) -> int:
    return {"A": 0, "HOLD": 1, "B": 2, "C": 3}.get(str(grade), 4)


def library_sort_key(record: dict[str, Any]) -> tuple[int, str]:
    return (
        grade_rank(record.get("grade")),
        str(record.get("title") or record.get("paper_id") or ""),
    )


def normalized_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = string_or_empty(item)
        if text and text not in result:
            result.append(text)
    return result


def join_sentences(items: list[str]) -> str:
    return " ".join(item.rstrip(".。") + "。" for item in items if item).strip()


def compact_text(value: Any, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", string_or_empty(value)).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "..."


def compact_compare_text(value: str) -> str:
    return re.sub(r"[\W_]+", "", value.lower())


def read_recommendation(grade: str) -> str:
    return {"A": "重点关注", "B": "标准读", "C": "低优先级", "HOLD": "需确认"}.get(grade, "需确认")


def markdown_title(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_json_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def normalize_chat_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in history[-10:]:
        if not isinstance(item, dict):
            continue
        role = string_or_empty(item.get("role")).lower()
        if role not in {"user", "assistant"}:
            continue
        content = compact_text(item.get("content"), max_chars=1200)
        if content:
            normalized.append({"role": role, "content": content})
    return normalized[-8:]


def string_or_none(value: Any) -> str | None:
    text = string_or_empty(value)
    return text or None


def string_or_empty(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def int_or_none(value: Any) -> int | None:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return None
    return integer if integer > 0 else None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
