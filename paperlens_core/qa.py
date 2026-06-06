from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from paperlens_core.agent_loop import AgentLoop, PaperToolRegistry
from paperlens_core.agents.llm import JsonLlmClient
from paperlens_core.config import CoreConfig
from paperlens_core.core_manifest import inspect_core_v2_artifact_set
from paperlens_core.dom import PaperDOM
from paperlens_core.grounding import text_overlaps_any_reference
from paperlens_core.graph import ClaimGraph
from paperlens_core.library import read_library_records
from paperlens_core.memory_v3 import (
    dict_value,
    memory_v3_prompt_view,
    read_paper_memory_v3,
)
from paperlens_core.runtime import PaperLensRuntime
from paperlens_core.runtime import context_pack_prompt
from paperlens_core.workflow.core_v2 import (
    load_core_v2_dom_and_graph,
)


ASK_SYSTEM_PROMPT = """
You are PaperLens QA.
Answer the user's question clearly. Prefer ClaimGraph/PaperDOM source IDs for paper-specific claims.
For prerequisite/background questions, teach the concept first, then connect it back to the paper.
Separate paper claims, PaperLens inference, background knowledge, and evidence limits.
If a background explanation is useful, label it as background knowledge, not a paper claim.
Use source_attribution to record those boundaries.
Return final matching the QA schema when done.
""".strip()


ASK_PROMPT_VERSION = "qa-agent-v1"
CORE_V2_QA_CONTEXT_VERSION = "paperlens_core_v2_qa_context.v1"
CORE_V2_CONSUMABLE_STATUSES = {"REVIEWED", "REVIEWED_WITH_LIMITS"}


ASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer_markdown", "cited_pages", "confidence", "source_attribution"],
    "properties": {
        "answer_markdown": {"type": "string"},
        "cited_pages": {"type": "array", "items": {"type": "integer", "minimum": 1}},
        "cited_source_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "source_attribution": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "paper_claims",
                "paperlens_inferences",
                "background_context",
                "evidence_limits",
            ],
            "properties": {
                "paper_claims": {"type": "array", "items": {"type": "string"}},
                "paperlens_inferences": {"type": "array", "items": {"type": "string"}},
                "background_context": {"type": "array", "items": {"type": "string"}},
                "evidence_limits": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


def answer_question(
    *,
    output_dir: Path,
    config: CoreConfig,
    paper_id: str | None,
    question: str,
    chat_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report_path = try_resolve_report_path(output_dir, paper_id)
    if paper_id:
        resolved_paper_id = paper_id
    elif report_path:
        resolved_paper_id = paper_id_from_report(report_path)
    else:
        raise FileNotFoundError(f"No paper reports found under {output_dir / 'papers'}")
    layout = load_layout(output_dir, resolved_paper_id)
    paper_memory_v3 = read_paper_memory_v3(output_dir, resolved_paper_id)
    library_record = load_library_record(output_dir, resolved_paper_id)
    core_v2_context = load_core_v2_qa_context(
        output_dir=output_dir,
        paper_id=resolved_paper_id,
        question=question,
    )
    qa_memory = qa_memory_context(
        paper_id=resolved_paper_id,
        paper_memory_v3=paper_memory_v3,
        core_v2_context=core_v2_context,
    )
    augmented_query = memory_augmented_query(question, qa_memory)
    pages = select_relevant_pages(layout, augmented_query, paper_memory_v3=qa_memory)
    pages = merge_core_v2_pages(pages, layout, core_v2_context)
    question_type = classify_question(question)
    layout_pages = [page for page in layout.get("pages", []) if isinstance(page, dict)]
    runtime = PaperLensRuntime(artifacts=layout_pages)
    agent_context = runtime.build_context_pack(
        stage="qa",
        objective=qa_context_objective(core_v2_context),
        paper_id=resolved_paper_id,
        title=string_or_empty(
            (qa_memory.get("metadata") or {}).get("title")
            if isinstance(qa_memory.get("metadata"), dict)
            else None
        ),
        classification=string_or_empty(
            (qa_memory.get("reading_context") or {}).get("grade")
            if isinstance(qa_memory.get("reading_context"), dict)
            else None
        ),
        memory=qa_memory,
        focus_queries=[question, augmented_query],
        focus_pages=[page.get("page_no") for page in pages if isinstance(page.get("page_no"), int)],
        read_artifacts=pages,
        output_contract={
            "type": "QAAnswer",
            "question_type": question_type,
            "rule": (
                "Answer from ClaimGraph/PaperDOM source_ids when available; keep paper claims, "
                "PaperLens inference, background, and evidence limits separate."
            ),
        },
        search_limit=5,
        page_text_limit=1000,
    ).as_dict()
    if config.offline_debug or config.provider.kind == "none":
        answer = offline_qa_answer(
            paper_id=resolved_paper_id,
            report_path=report_path,
            question_type=question_type,
            pages=pages,
            core_v2_context=core_v2_context,
        )
        answer = ground_qa_answer_in_core_v2_context(answer, core_v2_context)
        write_qa_trace(
            output_dir,
            answer,
            question=question,
            selected_pages=pages,
            cache_hit=False,
            agent_context=agent_context,
            core_v2_context=core_v2_context,
        )
        return answer
    config.validate_agentic_run()
    client = JsonLlmClient(
        config.provider,
        ledger_path=output_dir / ".paperlens" / "data" / "model_calls.jsonl",
        run_id=f"qa_{hash_text(resolved_paper_id + ':' + question)[:12]}",
    )
    user_prompt = build_ask_prompt(
        report_path=report_path,
        paper_id=resolved_paper_id,
        question=question,
        chat_history=chat_history or [],
        paper_memory_v3=qa_memory,
        library_record=library_record,
        pages=pages,
        question_type=question_type,
        agent_context=agent_context,
        core_v2_context=core_v2_context,
    )
    image_paths = visual_question_page_images(pages, question)
    cache_path = ask_cache_path(
        output_dir,
        resolved_paper_id,
        {
            "version": ASK_PROMPT_VERSION,
            "model": config.provider.model,
            "visual_detail": config.visual_detail,
            "paper_id": resolved_paper_id,
            "question": question,
            "chat_history_hash": hash_json_payload(normalize_chat_history(chat_history or [])),
            "prompt_hash": hash_text(ASK_SYSTEM_PROMPT + "\n" + user_prompt),
            "schema_hash": hash_json_payload(ASK_SCHEMA),
            "images": [image_cache_fingerprint(path) for path in image_paths],
            "agent_context_hash": hash_json_payload(agent_context),
            "core_v2_context_hash": hash_json_payload(core_v2_context),
        },
    )
    cached = read_ask_cache(cache_path)
    if cached and isinstance(cached.get("data"), dict):
        answer = normalize_answer(cached["data"])
        answer = ground_qa_answer_in_core_v2_context(answer, core_v2_context)
        answer["paper_id"] = resolved_paper_id
        answer["usage"] = {}
        answer["cache_hit"] = True
        answer["question_type"] = question_type
        write_qa_trace(
            output_dir,
            answer,
            question=question,
            selected_pages=pages,
            cache_hit=True,
            agent_context=agent_context,
            core_v2_context=core_v2_context,
        )
        return answer
    result = run_paper_qa_agent(
        client=client,
        output_dir=output_dir,
        paper_id=resolved_paper_id,
        title=string_or_empty(
            (qa_memory.get("metadata") or {}).get("title")
            if isinstance(qa_memory.get("metadata"), dict)
            else None
        ),
        question=question,
        question_type=question_type,
        chat_history=chat_history or [],
        paper_memory_v3=qa_memory,
        library_record=library_record,
        layout_pages=layout_pages,
        selected_pages=pages,
        core_v2_context=core_v2_context,
        agent_context=agent_context,
        user_prompt=user_prompt,
    )
    answer = normalize_answer(result.final)
    answer = ground_qa_answer_in_core_v2_context(answer, core_v2_context)
    answer["paper_id"] = resolved_paper_id
    answer["usage"] = result.usage
    answer["cache_hit"] = False
    answer["question_type"] = question_type
    write_ask_cache(
        cache_path,
        {
            "data": result.final,
            "usage": result.usage,
            "request_ids": result.request_ids,
            "endpoint": "agent_loop",
        },
    )
    write_qa_trace(
        output_dir,
        answer,
        question=question,
        selected_pages=pages,
        cache_hit=False,
        agent_context=agent_context,
        core_v2_context=core_v2_context,
    )
    return answer


def load_core_v2_qa_context(
    *,
    output_dir: Path,
    paper_id: str,
    question: str,
    limit: int = 8,
) -> dict[str, Any]:
    try:
        data_dir = paperlens_data_dir(output_dir)
        dom, graph = load_core_v2_dom_and_graph(data_dir, paper_id)
    except (FileNotFoundError, ValueError):
        return {}
    core_manifest = inspect_core_v2_artifact_set(data_dir, paper_id)
    if not core_manifest.get("consumable"):
        return {
            "schema_version": CORE_V2_QA_CONTEXT_VERSION,
            "paper_id": paper_id,
            "question": question,
            "retrieval_policy": core_v2_non_consumable_policy(core_manifest),
            "answer_source_policy": (
                "Core v2 ClaimGraph is not in a reviewed publish state; do not use it as "
                "paper-claim evidence."
            ),
            "quality": core_v2_quality_context(core_manifest),
            "matches": [],
        }
    matches = search_core_v2_graph(dom=dom, graph=graph, question=question, limit=limit)
    return {
        "schema_version": CORE_V2_QA_CONTEXT_VERSION,
        "paper_id": paper_id,
        "question": question,
        "retrieval_policy": "claim_graph_nodes_with_paper_dom_source_ids",
        "answer_source_policy": (
            "Use graph node IDs and PaperDOM source IDs for paper claims; report Markdown is not "
            "evidence."
        ),
        "quality": core_v2_quality_context(core_manifest),
        "matches": matches,
    }


def core_v2_quality_context(core_manifest: dict[str, Any]) -> dict[str, Any]:
    quality_metrics = core_manifest.get("required_artifacts", {}).get("quality_metrics", {})
    return {
        "status": core_manifest.get("status"),
        "publish_status": core_manifest.get("publish_status"),
        "consumable": core_manifest.get("consumable"),
        "issues": core_manifest.get("issues", []),
        "quality_metrics_artifact": quality_metrics,
    }


def core_v2_non_consumable_policy(core_manifest: dict[str, Any]) -> str:
    issues = set(str(issue) for issue in core_manifest.get("issues", []))
    if "missing:core_manifest.v1.json" in issues:
        return "missing_core_v2_manifest"
    if "missing:quality_metrics.v1.json" in issues:
        return "missing_core_v2_quality_metrics"
    publish_status = str(core_manifest.get("publish_status") or "")
    if publish_status == "BLOCKED":
        return "blocked_by_core_v2_audit"
    return "not_reviewed_by_core_v2_audit"


def qa_context_objective(core_v2_context: dict[str, Any]) -> str:
    if core_v2_context_is_consumable(core_v2_context):
        return (
            "Answer the user's question by grounding paper-specific claims in the reviewed "
            "core v2 ClaimGraph and PaperDOM source IDs. Use legacy PaperMemory only as "
            "supplemental context, and treat report text as orientation, not proof."
        )
    return (
        "Answer the user's question by dynamically grounding it in available PaperMemory and "
        "local paper evidence. Treat report text as orientation, not proof."
    )


def qa_memory_context(
    *,
    paper_id: str,
    paper_memory_v3: dict[str, Any],
    core_v2_context: dict[str, Any],
) -> dict[str, Any]:
    if core_v2_context_is_consumable(core_v2_context):
        return core_v2_qa_memory_view(paper_id=paper_id, core_v2_context=core_v2_context)
    return paper_memory_v3


def core_v2_context_is_consumable(core_v2_context: dict[str, Any]) -> bool:
    return (
        core_v2_context.get("retrieval_policy") == "claim_graph_nodes_with_paper_dom_source_ids"
        and bool(list_of_dicts(core_v2_context.get("matches")))
    )


def core_v2_qa_memory_view(*, paper_id: str, core_v2_context: dict[str, Any]) -> dict[str, Any]:
    claims = []
    evidence_by_id: dict[str, dict[str, Any]] = {}
    for match in list_of_dicts(core_v2_context.get("matches"))[:12]:
        source_ids = unique_strings(
            source_id for source_id in match.get("source_ids", []) if isinstance(source_id, str)
        )
        claims.append(
            {
                "id": match.get("node_id"),
                "text": match.get("label"),
                "type": core_v2_claim_type(str(match.get("kind") or "")),
                "provenance": match.get("provenance") or "explicit",
                "confidence": match.get("confidence") or "medium",
                "critic_status": "checked",
                "evidence_refs": source_ids,
                "source": "core_v2_claim_graph",
            }
        )
        for span in list_of_dicts(match.get("evidence_spans")):
            source_id = str(span.get("source_id") or "").strip()
            if not source_id or source_id in evidence_by_id:
                continue
            evidence_by_id[source_id] = {
                "id": source_id,
                "source_type": span.get("kind") or "paper_dom_source",
                "page": span.get("page_no"),
                "interpretation": span.get("text")
                or span.get("caption")
                or f"PaperDOM source {source_id}",
                "source": "core_v2_paper_dom",
            }
    return {
        "schema_version": "paperlens_core_v2_qa_memory_view.v1",
        "paper_id": paper_id,
        "reading_context": {
            "source_of_truth": "core_v2_claim_graph",
            "retrieval_policy": core_v2_context.get("retrieval_policy"),
            "publish_status": dict_value(core_v2_context.get("quality")).get("publish_status"),
        },
        "claims": [claim for claim in claims if claim.get("id") and claim.get("text")],
        "evidence": list(evidence_by_id.values()),
        "audit_trail": {
            "core_v2_quality": core_v2_context.get("quality"),
            "answer_source_policy": core_v2_context.get("answer_source_policy"),
        },
    }


def core_v2_claim_type(kind: str) -> str:
    if kind in {"mechanism", "implementation"}:
        return "mechanism"
    if kind in {"evaluation", "result"}:
        return "evaluation"
    if kind == "limitation":
        return "limitation"
    if kind == "problem":
        return "motivation"
    return "implication"


def search_core_v2_graph(
    *,
    dom: PaperDOM,
    graph: ClaimGraph,
    question: str,
    limit: int,
) -> list[dict[str, Any]]:
    terms = tokenize(question)
    source_index = core_v2_source_index(dom)
    scored: list[tuple[int, str, dict[str, Any]]] = []
    for node in graph.nodes.values():
        if node.kind == "evidence":
            continue
        evidence_ids = graph.evidence_ids_for(node.node_id)
        evidence_spans = []
        evidence_text = []
        source_ids = []
        for evidence_id in evidence_ids:
            evidence_node = graph.nodes.get(evidence_id)
            source_id = str((evidence_node.payload if evidence_node else {}).get("source_id") or "")
            source = source_index.get(source_id)
            if not source:
                continue
            source_ids.append(source_id)
            evidence_spans.append(source)
            evidence_text.append(json.dumps(source, ensure_ascii=False))
        haystack = normalize_text(
            " ".join([node.kind, node.label, *evidence_text]),
            limit=8000,
        ).lower()
        score = sum(haystack.count(term) for term in terms) if terms else 1
        if score <= 0:
            continue
        scored.append(
            (
                -score,
                node.node_id,
                {
                    "node_id": node.node_id,
                    "kind": node.kind,
                    "label": node.label,
                    "confidence": node.payload.get("confidence"),
                    "provenance": node.payload.get("provenance"),
                    "uncertainty": node.payload.get("uncertainty"),
                    "evidence_ids": evidence_ids,
                    "source_ids": source_ids,
                    "evidence_spans": evidence_spans[:4],
                    "relationships": relationships_for_node(
                        graph=graph,
                        node_id=node.node_id,
                        source_index=source_index,
                    ),
                },
            )
        )
    scored.sort(key=lambda item: (item[0], item[1]))
    if not scored:
        for node in graph.nodes.values():
            if node.kind != "evidence":
                scored.append(
                    (
                        0,
                        node.node_id,
                        {
                            "node_id": node.node_id,
                            "kind": node.kind,
                            "label": node.label,
                            "confidence": node.payload.get("confidence"),
                            "provenance": node.payload.get("provenance"),
                            "uncertainty": node.payload.get("uncertainty"),
                            "evidence_ids": graph.evidence_ids_for(node.node_id),
                            "source_ids": [],
                            "evidence_spans": [],
                            "relationships": relationships_for_node(
                                graph=graph,
                                node_id=node.node_id,
                                source_index=source_index,
                            ),
                        },
                    )
                )
    return [item for _score, _node_id, item in scored[: max(1, limit)]]


def relationships_for_node(
    *,
    graph: ClaimGraph,
    node_id: str,
    source_index: dict[str, dict[str, Any]],
    limit: int = 8,
) -> list[dict[str, Any]]:
    result = []
    for edge in graph.edges:
        if edge.kind == "supported_by":
            continue
        if edge.source_id != node_id and edge.target_id != node_id:
            continue
        source_node = graph.nodes.get(edge.source_id)
        target_node = graph.nodes.get(edge.target_id)
        if source_node is None or target_node is None:
            continue
        result.append(
            {
                "direction": "outgoing" if edge.source_id == node_id else "incoming",
                "kind": edge.kind,
                "source_id": edge.source_id,
                "source_kind": source_node.kind,
                "source_label": source_node.label,
                "target_id": edge.target_id,
                "target_kind": target_node.kind,
                "target_label": target_node.label,
                "source_ids": relationship_source_ids(
                    graph=graph,
                    source_index=source_index,
                    node_ids=[edge.source_id, edge.target_id],
                ),
            }
        )
        if len(result) >= limit:
            break
    return result


def relationship_source_ids(
    *,
    graph: ClaimGraph,
    source_index: dict[str, dict[str, Any]],
    node_ids: list[str],
) -> list[str]:
    result = []
    for node_id in node_ids:
        for evidence_id in graph.evidence_ids_for(node_id):
            evidence_node = graph.nodes.get(evidence_id)
            paper_source_id = str(
                (evidence_node.payload if evidence_node else {}).get("source_id") or ""
            )
            if paper_source_id in source_index and paper_source_id not in result:
                result.append(paper_source_id)
    return result


def core_v2_source_index(dom: PaperDOM) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for span in dom.spans:
        result[span.source_id] = {
            "source_id": span.source_id,
            "kind": span.kind,
            "page_no": span.page_no,
            "section_id": span.section_id,
            "text": normalize_text(span.text, limit=900),
        }
    for figure in dom.figures:
        result[figure.source_id] = {
            "source_id": figure.source_id,
            "kind": figure.kind,
            "page_no": figure.page_no,
            "caption": normalize_text(figure.caption or "", limit=700),
            "bbox": figure.bbox,
        }
    for table in dom.tables:
        result[table.source_id] = {
            "source_id": table.source_id,
            "kind": table.kind,
            "page_no": table.page_no,
            "caption": normalize_text(table.caption or "", limit=700),
            "bbox": table.bbox,
        }
    for equation in dom.equations:
        result[equation.source_id] = {
            "source_id": equation.source_id,
            "kind": equation.kind,
            "page_no": equation.page_no,
            "section_id": equation.section_id,
            "text": normalize_text(equation.latex_or_text, limit=700),
        }
    return result


def merge_core_v2_pages(
    pages: list[dict[str, Any]],
    layout: dict[str, Any],
    core_v2_context: dict[str, Any],
) -> list[dict[str, Any]]:
    wanted_pages = []
    for match in list_of_dicts(core_v2_context.get("matches")):
        for span in list_of_dicts(match.get("evidence_spans")):
            if isinstance(span, dict) and isinstance(span.get("page_no"), int):
                wanted_pages.append(span["page_no"])
    if not wanted_pages:
        return pages
    all_pages = layout.get("pages") if isinstance(layout.get("pages"), list) else []
    by_no = {
        page.get("page_no"): page
        for page in all_pages
        if isinstance(page, dict) and isinstance(page.get("page_no"), int)
    }
    merged = list(pages)
    existing = {page.get("page_no") for page in merged if isinstance(page.get("page_no"), int)}
    for number in wanted_pages:
        page = by_no.get(number)
        if page and number not in existing:
            merged.append(page)
            existing.add(number)
    return merged[:8]


def offline_qa_answer(
    *,
    paper_id: str,
    report_path: Path | None,
    question_type: str,
    pages: list[dict[str, Any]],
    core_v2_context: dict[str, Any],
) -> dict[str, Any]:
    matches = list_of_dicts(core_v2_context.get("matches"))
    if matches:
        claims = [str(item.get("label") or "").strip() for item in matches[:5]]
        source_ids = unique_strings(
            source_id
            for item in matches[:5]
            for source_id in item.get("source_ids", [])
            if isinstance(source_id, str)
        )
        pages_from_sources = unique_ints(
            span.get("page_no")
            for item in matches[:5]
            for span in item.get("evidence_spans", [])
            if isinstance(span, dict)
        )
        lines = ["离线模式下未调用模型；下面是从 ClaimGraph 命中的原文锚定事实："]
        for item in matches[:5]:
            source_hint = ", ".join(item.get("source_ids", [])[:3])
            suffix = f"（source_ids: {source_hint}）" if source_hint else ""
            lines.append(f"- [{item.get('kind')}] {item.get('label')}{suffix}")
            for relation in list_of_dicts(item.get("relationships"))[:3]:
                relation_sources = [
                    source_id
                    for source_id in relation.get("source_ids", [])
                    if isinstance(source_id, str)
                ][:3]
                relation_suffix = (
                    f"（source_ids: {', '.join(relation_sources)}）" if relation_sources else ""
                )
                lines.append(
                    "  - relation: "
                    f"{relation.get('source_id')} --{relation.get('kind')}--> "
                    f"{relation.get('target_id')}{relation_suffix}"
                )
        return {
            "paper_id": paper_id,
            "answer_markdown": "\n".join(lines),
            "cited_pages": pages_from_sources,
            "cited_source_ids": source_ids,
            "confidence": "medium" if source_ids else "low",
            "question_type": question_type,
            "source_attribution": {
                "paper_claims": [claim for claim in claims if claim][:5],
                "paperlens_inferences": [],
                "background_context": [],
                "evidence_limits": ["离线模式只返回 ClaimGraph 检索结果，未调用模型综合推理。"],
            },
        }
    report_hint = f"`{report_path.as_posix()}`" if report_path else "当前论文的 core v2 数据"
    return {
        "paper_id": paper_id,
        "answer_markdown": f"当前是离线模式，不能调用模型回答。可先查看：{report_hint}。",
        "cited_pages": [
            page["page_no"] for page in pages[:3] if isinstance(page.get("page_no"), int)
        ],
        "cited_source_ids": [],
        "confidence": "low",
        "question_type": question_type,
        "source_attribution": {
            "paper_claims": [],
            "paperlens_inferences": ["离线模式只返回可用证据位置，未进行模型问答。"],
            "background_context": [],
            "evidence_limits": ["未调用模型，不能可靠综合回答问题。"],
        },
    }


def list_of_dicts(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def unique_strings(values: Any) -> list[str]:
    result = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in result:
            result.append(text)
    return result


def unique_ints(values: Any) -> list[int]:
    result = []
    for value in values:
        if not isinstance(value, int):
            continue
        if value > 0 and value not in result:
            result.append(value)
    return result


def run_paper_qa_agent(
    *,
    client: JsonLlmClient,
    output_dir: Path,
    paper_id: str,
    title: str,
    question: str,
    question_type: str,
    chat_history: list[dict[str, Any]],
    paper_memory_v3: dict[str, Any],
    library_record: dict[str, Any],
    layout_pages: list[dict[str, Any]],
    selected_pages: list[dict[str, Any]],
    core_v2_context: dict[str, Any],
    agent_context: dict[str, Any],
    user_prompt: str,
) -> Any:
    runtime = PaperLensRuntime(artifacts=layout_pages)
    tools = PaperToolRegistry(
        runtime=runtime,
        paper_id=paper_id,
        title=title,
        memory=paper_memory_v3,
        layout_pages=layout_pages,
    )
    page_hints = [
        {
            "page_no": page.get("page_no"),
            "text": normalize_text(str(page.get("text") or ""), limit=900),
            "captions": (page.get("captions") or [])[:4],
            "visual_notes": (page.get("visual_notes") or [])[:3],
        }
        for page in selected_pages[:6]
    ]
    loop = AgentLoop(
        client=client,
        tools=tools,
        session_name="paper_qa",
        objective=(
            "Answer the current paper question. Use background knowledge freely for teaching, "
            "but use paper tools before making paper-specific claims."
        ),
        final_schema_name="paperlens_paper_question",
        final_schema=ASK_SCHEMA,
        stage="qa",
        paper_id=paper_id,
        trace_path=paperlens_data_dir(output_dir) / "agent_trace.jsonl",
        system_prompt=ASK_SYSTEM_PROMPT,
    )
    return loop.run(
        initial_context={
            "question": question,
            "question_type": question_type,
            "recent_chat_history": normalize_chat_history(chat_history),
            "core_v2_context_priority": core_v2_context_priority(core_v2_context),
            "paper_memory_v3_ir": memory_v3_prompt_view(paper_memory_v3),
            "core_v2_claim_graph_context": core_v2_context,
            "memory_fallback_policy": memory_fallback_policy(core_v2_context),
            "paperlens_library_record": library_record,
            "initial_page_hints": page_hints,
            "context_pack": agent_context,
            "prompt_snapshot": user_prompt,
        }
    )


def ask_cache_path(output_dir: Path, paper_id: str, key_payload: dict[str, Any]) -> Path:
    key = hashlib.sha256(
        json.dumps(key_payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:24]
    safe_paper = re.sub(r"[^a-zA-Z0-9_.-]+", "_", paper_id)
    return output_dir / ".paperlens" / "cache" / "qa_answers" / safe_paper / f"{key}.json"


def read_ask_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def write_ask_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_qa_trace(
    output_dir: Path,
    answer: dict[str, Any],
    *,
    question: str,
    selected_pages: list[dict[str, Any]],
    cache_hit: bool,
    agent_context: dict[str, Any] | None = None,
    core_v2_context: dict[str, Any] | None = None,
) -> None:
    trace_path = paperlens_data_dir(output_dir) / "qa_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "paper_id": answer.get("paper_id"),
        "question": question,
        "question_type": answer.get("question_type"),
        "confidence": answer.get("confidence"),
        "cited_pages": answer.get("cited_pages"),
        "cited_source_ids": answer.get("cited_source_ids") or [],
        "selected_pages": [page.get("page_no") for page in selected_pages[:8]],
        "selected_graph_nodes": [
            match.get("node_id")
            for match in list_of_dicts((core_v2_context or {}).get("matches"))[:8]
        ],
        "source_attribution": answer.get("source_attribution"),
        "cache_hit": cache_hit,
        "agent_context": {
            "stage": (agent_context or {}).get("stage"),
            "objective": (agent_context or {}).get("objective"),
            "focus_pages": ((agent_context or {}).get("working_context") or {}).get("focus_pages"),
            "tool_count": len((agent_context or {}).get("tool_trace") or []),
        },
    }
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def image_cache_fingerprint(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hash_file(path)}


def hash_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return "missing"


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def hash_json_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16]


def resolve_report_path(output_dir: Path, paper_id: str | None) -> Path:
    report = try_resolve_report_path(output_dir, paper_id)
    if report is not None:
        return report
    if paper_id:
        raise FileNotFoundError(f"No paper report found for paper_id={paper_id}")
    raise FileNotFoundError(f"No paper reports found under {output_dir / 'papers'}")


def try_resolve_report_path(output_dir: Path, paper_id: str | None) -> Path | None:
    reports = sorted((output_dir / "papers").glob("*.md"))
    if not reports:
        return None
    if paper_id:
        for report in reports:
            if report.name.startswith(paper_id):
                return report
        return None
    return reports[0]


def paper_id_from_report(path: Path) -> str:
    match = re.match(r"^(p_[A-Za-z0-9]+)", path.name)
    return match.group(1) if match else path.stem.split("_", 1)[0]


def load_layout(output_dir: Path, paper_id: str) -> dict[str, Any]:
    path = paperlens_data_dir(output_dir) / "artifacts" / "layout" / f"{paper_id}.json"
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def load_library_record(output_dir: Path, paper_id: str) -> dict[str, Any]:
    for record in read_library_records(output_dir):
        if record.get("paper_id") == paper_id:
            return record
    return {}


def paperlens_data_dir(output_dir: Path) -> Path:
    data_dir = output_dir / ".paperlens" / "data"
    return data_dir


def select_relevant_pages(
    layout: dict[str, Any],
    question: str,
    *,
    paper_memory_v3: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    pages = layout.get("pages") if isinstance(layout.get("pages"), list) else []
    runtime = PaperLensRuntime(artifacts=[page for page in pages if isinstance(page, dict)])
    search = runtime.search_text(memory_augmented_query(question, paper_memory_v3 or {}), limit=8)
    selected_numbers = [
        item.get("page_no") for item in search.results if isinstance(item.get("page_no"), int)
    ]
    if not selected_numbers:
        selected_numbers = [
            page.get("page_no")
            for page in pages[:3]
            if isinstance(page, dict) and isinstance(page.get("page_no"), int)
        ]
    by_no = {
        page.get("page_no"): page
        for page in pages
        if isinstance(page, dict) and isinstance(page.get("page_no"), int)
    }
    selected = [by_no[number] for number in selected_numbers if number in by_no]
    for page in pages[:3]:
        if not isinstance(page, dict):
            continue
        number = page.get("page_no")
        if isinstance(number, int) and page not in selected:
            selected.append(page)
    return selected[:8]


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_]{3,}|[\u4e00-\u9fff]{2,}", text.lower())
    return list(dict.fromkeys(words))[:32]


def memory_augmented_query(question: str, paper_memory_v3: dict[str, Any]) -> str:
    terms = [question]
    for claim in (
        paper_memory_v3.get("claims") if isinstance(paper_memory_v3.get("claims"), list) else []
    ):
        if not isinstance(claim, dict):
            continue
        text = string_or_empty(claim.get("text"))
        if text and any(token in text.lower() for token in tokenize(question)):
            terms.append(text)
            break
    bridge = paper_memory_v3.get("conceptual_bridge")
    if isinstance(bridge, dict):
        for item in bridge.get("terms") if isinstance(bridge.get("terms"), list) else []:
            if not isinstance(item, dict):
                continue
            term = string_or_empty(item.get("term"))
            role = string_or_empty(item.get("paper_role"))
            if term and term.lower() in question.lower():
                terms.append(" ".join([term, role]))
    evaluation = paper_memory_v3.get("evaluation")
    if isinstance(evaluation, dict) and classify_question(question) == "evidence_check":
        terms.append(string_or_empty(evaluation.get("summary")))
    return " ".join(item for item in terms if item)[:1200]


def classify_question(question: str) -> str:
    normalized = question.lower()
    if any(
        token in normalized
        for token in ["不信", "核对", "真的吗", "真的", "challenge", "verify", "evidence", "证明"]
    ):
        return "evidence_check"
    if any(
        token in normalized
        for token in ["比较", "区别", "相似", "不像", "像在哪里", "vs", "versus", "compare"]
    ):
        return "comparison"
    if any(
        token in normalized for token in ["实现", "代码", "模块", "implementation", "implement"]
    ):
        return "implementation"
    if any(token in normalized for token in ["复现", "reproduce", "artifact"]):
        return "reproduction"
    if any(
        token in normalized
        for token in [
            "基础",
            "常识",
            "背景知识",
            "前置",
            "我不会",
            "不懂",
            "补充常识",
            "直觉",
            "类比",
            "比喻",
            "intuition",
            "background",
            "prerequisite",
        ]
    ):
        return "background_explanation"
    if any(token in normalized for token in ["术语", "是什么意思", "解释", "clarify", "什么是"]):
        return "clarification"
    if any(token in normalized for token in ["怎么做", "机制", "原理", "how", "mechanism"]):
        return "mechanism"
    if any(token in normalized for token in ["以前读过", "哪些论文", "library", "本地库"]):
        return "library_recall"
    return "orientation"


def normalize_chat_history(history: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []
    for item in history[-10:]:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = normalize_text(str(item.get("content") or ""), limit=1200)
        if not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized[-8:]


def build_ask_prompt(
    *,
    report_path: Path | None,
    paper_id: str,
    question: str,
    chat_history: list[dict[str, Any]] | None = None,
    paper_memory_v3: dict[str, Any] | None = None,
    pages: list[dict[str, Any]],
    question_type: str = "orientation",
    library_record: dict[str, Any] | None = None,
    agent_context: dict[str, Any] | None = None,
    core_v2_context: dict[str, Any] | None = None,
) -> str:
    page_blocks = []
    for page in pages:
        page_blocks.append(
            {
                "page_no": page.get("page_no"),
                "text": normalize_text(str(page.get("text") or ""), limit=1800),
                "captions": (page.get("captions") or [])[:6],
                "visual_notes": (page.get("visual_notes") or [])[:4],
            }
        )
    return "\n\n".join(
        [
            f"paper_id: {paper_id}",
            f"report_path: {report_path.as_posix() if report_path else ''}",
            f"question_type: {question_type}",
            f"core_v2_context_priority: {core_v2_context_priority(core_v2_context or {})}",
            "answer_mode:",
            qa_answer_mode_instruction(question_type),
            "recent_chat_history:",
            json.dumps(normalize_chat_history(chat_history or []), ensure_ascii=False),
            "question:",
            question,
            "core_v2_claim_graph_context:",
            json.dumps(core_v2_context or {}, ensure_ascii=False),
            "paper_memory_v3_ir:",
            json.dumps(memory_v3_prompt_view(paper_memory_v3 or {}), ensure_ascii=False),
            "memory_fallback_policy:",
            memory_fallback_policy(core_v2_context or {}),
            "agent_context_pack:",
            context_pack_prompt(agent_context),
            "paperlens_library_record:",
            json.dumps(library_record or {}, ensure_ascii=False),
            "If page images are attached, they follow the same order as the first relevant_page_excerpts that have images.",
            "relevant_page_excerpts:",
            json.dumps(page_blocks, ensure_ascii=False),
            (
                "Answer contract: source_attribution.paper_claims should contain only claims directly "
                "supported by reviewed core v2 ClaimGraph nodes and PaperDOM source IDs when "
                "available; otherwise use PaperMemory evidence/claims or relevant page excerpts "
                "as fallback. "
                "Use agent_context_pack as the active tool/context trace for this question. "
                "When core_v2_claim_graph_context has matches, treat those graph node IDs and "
                "PaperDOM source IDs as the preferred paper evidence. "
                "Do not use the rendered report as a source; it is only a user-facing view. "
                "Use recent_chat_history only to resolve follow-up references and the user's intent; "
                "do not treat previous assistant answers as facts unless supported by memory/evidence. "
                "source_attribution.paperlens_inferences should contain PaperLens synthesis or cautious "
                "interpretation. source_attribution.background_context should contain general field "
                "knowledge that is not asserted by the paper. source_attribution.evidence_limits should "
                "state missing evidence or uncertainty in user-facing wording. Do not blur these "
                "categories. Do not mention implementation context about evidence being supplied "
                "by the system or user."
            ),
        ]
    )


def qa_answer_mode_instruction(question_type: str) -> str:
    if question_type in {"background_explanation", "clarification"}:
        return (
            "Teach the prerequisite concept first in plain language, using general background knowledge "
            "when useful. Then explain how the paper uses that concept. Keep paper claims and background "
            "knowledge explicitly separate."
        )
    if question_type == "implementation":
        return (
            "Use readable implementation-style examples or pseudocode when the user asks for code. "
            "The code can be an explanatory analogy, but label it as such unless the paper provides "
            "actual implementation details."
        )
    return (
        "Answer the paper question directly, using paper memory/evidence for factual claims and "
        "background knowledge only when it helps understanding."
    )


def core_v2_context_priority(core_v2_context: dict[str, Any]) -> str:
    if core_v2_context_is_consumable(core_v2_context):
        return "primary_reviewed_claim_graph"
    if core_v2_context.get("retrieval_policy"):
        return f"unavailable:{core_v2_context.get('retrieval_policy')}"
    return "unavailable:no_core_v2_context"


def memory_fallback_policy(core_v2_context: dict[str, Any]) -> str:
    if core_v2_context_is_consumable(core_v2_context):
        return (
            "supplemental_or_fallback_only; paper-specific factual claims should cite core v2 "
            "ClaimGraph node IDs and PaperDOM source IDs first."
        )
    return "fallback_primary_until_reviewed_core_v2_claim_graph_is_available."


def normalize_answer(data: dict[str, Any]) -> dict[str, Any]:
    pages = first_present(data, ["cited_pages", "pages", "citations", "page_numbers"])
    if not isinstance(pages, list):
        pages = []
    cited_pages = []
    for page in pages:
        if isinstance(page, dict):
            page = page.get("page") or page.get("page_no") or page.get("page_number")
        try:
            value = int(page)
        except (TypeError, ValueError):
            continue
        if value > 0 and value not in cited_pages:
            cited_pages.append(value)
    confidence = str(data.get("confidence") or "low")
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    answer = string_or_empty(
        first_present(
            data, ["answer_markdown", "answer", "response", "content", "markdown", "text"]
        )
    ) or recover_text_answer(data)
    answer = sanitize_qa_text(answer)
    for page in extract_page_citations(answer):
        if page not in cited_pages:
            cited_pages.append(page)
    cited_source_ids = normalized_source_id_list(
        first_present(data, ["cited_source_ids", "source_ids", "paper_dom_source_ids"])
    )
    source_attribution = normalize_source_attribution(data, answer=answer, confidence=confidence)
    return {
        "answer_markdown": answer.strip(),
        "cited_pages": cited_pages,
        "cited_source_ids": cited_source_ids,
        "confidence": confidence,
        "source_attribution": source_attribution,
    }


def ground_qa_answer_in_core_v2_context(
    answer: dict[str, Any],
    core_v2_context: dict[str, Any],
) -> dict[str, Any]:
    if not core_v2_context_is_consumable(core_v2_context):
        return answer
    grounded = dict(answer)
    attribution = normalize_source_attribution(
        {"source_attribution": grounded.get("source_attribution")},
        answer=str(grounded.get("answer_markdown") or ""),
        confidence=str(grounded.get("confidence") or "low"),
    )
    allowed_source_ids = core_v2_context_source_ids(core_v2_context)
    cited_source_ids = [
        source_id
        for source_id in normalized_source_id_list(grounded.get("cited_source_ids"))
        if source_id in allowed_source_ids
    ]
    removed_source_ids = [
        source_id
        for source_id in normalized_source_id_list(grounded.get("cited_source_ids"))
        if source_id not in allowed_source_ids
    ]
    paper_claims = attribution["paper_claims"]
    supported_claims = []
    unsupported_claims = []
    for claim in paper_claims:
        matching_sources = source_ids_for_supported_claim(claim, core_v2_context)
        if matching_sources:
            supported_claims.append(claim)
            for source_id in matching_sources:
                if source_id not in cited_source_ids:
                    cited_source_ids.append(source_id)
        else:
            unsupported_claims.append(claim)
    if removed_source_ids:
        attribution["evidence_limits"].append(removed_qa_source_ids_note(removed_source_ids))
    if unsupported_claims:
        attribution["evidence_limits"].append(unsupported_qa_claims_note(unsupported_claims))
    attribution["paper_claims"] = supported_claims
    grounded["source_attribution"] = attribution
    grounded["cited_source_ids"] = cited_source_ids[:16]
    grounded["cited_pages"] = merge_core_v2_cited_pages(
        normalized_positive_int_list(grounded.get("cited_pages")),
        core_v2_context=core_v2_context,
        cited_source_ids=grounded["cited_source_ids"],
    )
    if removed_source_ids or unsupported_claims:
        grounded["confidence"] = lower_qa_confidence(str(grounded.get("confidence") or "low"))
        if unsupported_claims:
            grounded["answer_markdown"] = guarded_grounded_qa_answer_markdown(
                supported_claims=supported_claims,
                paperlens_inferences=attribution["paperlens_inferences"],
                background_context=attribution["background_context"],
                evidence_limits=attribution["evidence_limits"],
            )
        else:
            grounded["answer_markdown"] = append_qa_evidence_limit_notes(
                str(grounded.get("answer_markdown") or ""),
                [removed_qa_source_ids_note(removed_source_ids)] if removed_source_ids else [],
                [],
            )
    return grounded


def source_ids_for_supported_claim(claim: str, core_v2_context: dict[str, Any]) -> list[str]:
    source_ids = []
    for match in list_of_dicts(core_v2_context.get("matches")):
        label = str(match.get("label") or "")
        if not text_overlaps_any_reference(claim, [label]):
            continue
        for source_id in match.get("source_ids", []):
            if isinstance(source_id, str) and source_id and source_id not in source_ids:
                source_ids.append(source_id)
    return source_ids


def core_v2_context_source_ids(core_v2_context: dict[str, Any]) -> set[str]:
    source_ids = set()
    for match in list_of_dicts(core_v2_context.get("matches")):
        source_ids.update(source_id for source_id in match.get("source_ids", []) if source_id)
        source_ids.update(
            span.get("source_id")
            for span in list_of_dicts(match.get("evidence_spans"))
            if span.get("source_id")
        )
        for relationship in list_of_dicts(match.get("relationships")):
            source_ids.update(
                source_id for source_id in relationship.get("source_ids", []) if source_id
            )
    return {str(source_id).strip() for source_id in source_ids if str(source_id).strip()}


def merge_core_v2_cited_pages(
    pages: list[int],
    *,
    core_v2_context: dict[str, Any],
    cited_source_ids: list[str],
) -> list[int]:
    result = list(pages)
    cited = set(cited_source_ids)
    for match in list_of_dicts(core_v2_context.get("matches")):
        for span in list_of_dicts(match.get("evidence_spans")):
            if span.get("source_id") not in cited:
                continue
            page_no = span.get("page_no")
            if isinstance(page_no, int) and page_no > 0 and page_no not in result:
                result.append(page_no)
    return result[:8]


def normalized_positive_int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        try:
            number = int(item)
        except (TypeError, ValueError):
            continue
        if number > 0 and number not in result:
            result.append(number)
    return result


def lower_qa_confidence(confidence: str) -> str:
    if confidence == "high":
        return "medium"
    if confidence == "medium":
        return "low"
    return "low"


def removed_qa_source_ids_note(source_ids: list[str]) -> str:
    return (
        "Removed QA source IDs that were not present in the reviewed ClaimGraph context: "
        + ", ".join(source_ids[:6])
    )


def unsupported_qa_claims_note(claims: list[str]) -> str:
    return (
        "Removed model-declared paper claims that were not supported by the reviewed "
        "ClaimGraph matches: "
        + " | ".join(claims[:3])
    )


def append_qa_evidence_limit_notes(
    answer_markdown: str,
    removed_source_notes: list[str],
    unsupported_claim_notes: list[str],
) -> str:
    notes = [*removed_source_notes, *unsupported_claim_notes]
    if not notes:
        return answer_markdown
    lines = [answer_markdown.rstrip(), "", "Evidence limits:"]
    lines.extend(f"- {note}" for note in notes)
    return "\n".join(line for line in lines if line is not None).strip()


def guarded_grounded_qa_answer_markdown(
    *,
    supported_claims: list[str],
    paperlens_inferences: list[str],
    background_context: list[str],
    evidence_limits: list[str],
) -> str:
    lines = ["Grounded answer:"]
    if supported_claims:
        lines.extend(f"- {claim}" for claim in supported_claims)
    else:
        lines.append("- Reviewed ClaimGraph context does not support a paper-claim answer.")
    if paperlens_inferences:
        lines.extend(["", "PaperLens inference:"])
        lines.extend(f"- {item}" for item in paperlens_inferences)
    if background_context:
        lines.extend(["", "Background context:"])
        lines.extend(f"- {item}" for item in background_context)
    if evidence_limits:
        lines.extend(["", "Evidence limits:"])
        lines.extend(f"- {item}" for item in evidence_limits)
    return "\n".join(lines).strip()


def normalize_source_attribution(
    data: dict[str, Any],
    *,
    answer: str,
    confidence: str,
) -> dict[str, list[str]]:
    raw = data.get("source_attribution")
    if not isinstance(raw, dict):
        raw = {}
    result = {
        "paper_claims": normalized_string_list(raw.get("paper_claims")),
        "paperlens_inferences": normalized_string_list(raw.get("paperlens_inferences")),
        "background_context": normalized_string_list(raw.get("background_context")),
        "evidence_limits": normalized_string_list(raw.get("evidence_limits")),
    }
    if not any(result.values()) and answer:
        result["paperlens_inferences"] = [answer[:240]]
    if confidence == "low" and not result["evidence_limits"]:
        result["evidence_limits"] = ["回答证据不足或模型返回结果未明确区分来源。"]
    return result


def first_present(data: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = data.get(key)
        if key in data and value is not None and value != "":
            return value
    for value in data.values():
        if isinstance(value, dict):
            nested = first_present(value, keys)
            if nested is not None and nested != "":
                return nested
    return None


def string_or_empty(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalized_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        if not isinstance(item, str):
            continue
        text = sanitize_qa_text(re.sub(r"\s+", " ", item).strip())
        if text and text not in result:
            result.append(text)
        if len(result) >= 6:
            break
    return result


def normalized_source_id_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        text = str(item or "").strip()
        if not text or text in result:
            continue
        if ":" not in text:
            continue
        result.append(text)
        if len(result) >= 16:
            break
    return result


def sanitize_qa_text(text: str) -> str:
    replacements = [
        (r"\bthe supplied excerpts\b", "the automatic reading evidence"),
        (r"\bsupplied excerpts\b", "automatic reading evidence"),
        (r"\bthe supplied evidence\b", "the automatic reading evidence"),
        (r"\bsupplied evidence\b", "automatic reading evidence"),
        (r"\bthe user provided\b", "the indexed paper evidence contains"),
        (r"\buser provided\b", "indexed paper evidence contains"),
        (r"你给到的片段", "自动读取到的证据"),
        (r"你给到的内容", "自动读取到的内容"),
        (r"你给到", "自动读取到"),
        (r"供给的片段", "自动读取到的证据"),
        (r"供给的图示", "自动读取到的图表证据"),
        (r"提供的页面", "自动读取到的证据"),
        (r"提供的材料", "自动读取到的证据"),
        (r"提供的证据", "自动读取到的证据"),
    ]
    cleaned = text
    for pattern, replacement in replacements:
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"\[\[?([^\[\]]{1,24})\]\(#user_content_([A-Za-z0-9_.:-]+)\)\]?",
        r"证据 \2",
        cleaned,
    )
    return cleaned


def recover_text_answer(data: dict[str, Any]) -> str:
    parts: list[str] = []

    def walk(value: Any, key: str = "", depth: int = 0) -> None:
        if depth > 4 or len(parts) >= 4:
            return
        if isinstance(value, str):
            text = value.strip()
            if key not in {"confidence", "paper_id"} and len(text) >= 20:
                parts.append(text)
        elif isinstance(value, list):
            strings = [
                item.strip() for item in value if isinstance(item, str) and len(item.strip()) >= 10
            ]
            if strings:
                parts.append("\n".join(f"- {item}" for item in strings[:5]))
            for item in value:
                if isinstance(item, dict):
                    walk(item, key=key, depth=depth + 1)
        elif isinstance(value, dict):
            for child_key, child_value in value.items():
                walk(child_value, key=str(child_key), depth=depth + 1)

    walk(data)
    return "\n\n".join(dict.fromkeys(parts))[:1800]


def visual_question_page_images(pages: list[dict[str, Any]], question: str) -> list[Path]:
    if not question_needs_visual_context(question):
        return []
    paths = []
    for page in pages:
        raw_path = page.get("render_path")
        if isinstance(raw_path, str) and raw_path:
            path = Path(raw_path)
            if path.exists():
                paths.append(path)
        if len(paths) >= 3:
            break
    return paths


def question_needs_visual_context(question: str) -> bool:
    lowered = question.lower()
    visual_terms = [
        "图",
        "表",
        "figure",
        "fig.",
        "table",
        "chart",
        "plot",
        "diagram",
        "图片",
        "曲线",
        "实验图",
    ]
    return any(term in lowered for term in visual_terms)


def extract_page_citations(text: str) -> list[int]:
    pages = []
    for raw_group in re.findall(r"\bpp?\.\s*([0-9][0-9,\s-]*)", text, flags=re.IGNORECASE):
        for raw_page in re.findall(r"\d+", raw_group):
            value = int(raw_page)
            if value > 0 and value not in pages:
                pages.append(value)
    for pattern in [r"\bpage\s*(\d+)\b", r"第\s*(\d+)\s*页"]:
        for raw_page in re.findall(pattern, text, flags=re.IGNORECASE):
            value = int(raw_page)
            if value > 0 and value not in pages:
                pages.append(value)
    return pages[:8]


def normalize_text(text: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned[:limit]
