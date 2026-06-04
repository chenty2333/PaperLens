from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from paperlens_core.agent_loop import AgentLoop, PaperToolRegistry
from paperlens_core.agents.llm import JsonLlmClient
from paperlens_core.config import CoreConfig
from paperlens_core.library import read_library_records
from paperlens_core.memory_v3 import (
    memory_v3_prompt_view,
    read_paper_memory_v3,
)
from paperlens_core.runtime import PaperLensRuntime
from paperlens_core.runtime import context_pack_prompt


ASK_SYSTEM_PROMPT = """
You are PaperLens QA.
Answer the user's question clearly. Use paper tools whenever the answer needs original-paper evidence.
For prerequisite/background questions, teach the concept first, then connect it back to the paper.
Separate paper claims, PaperLens inference, background knowledge, and evidence limits.
If a background explanation is useful, label it as background knowledge, not a paper claim.
Use source_attribution to record those boundaries.
Return final_json matching the QA schema when done.
""".strip()


ASK_PROMPT_VERSION = "qa-agent-v1"


ASK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["answer_markdown", "cited_pages", "confidence", "source_attribution"],
    "properties": {
        "answer_markdown": {"type": "string"},
        "cited_pages": {"type": "array", "items": {"type": "integer", "minimum": 1}},
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
    report_path = resolve_report_path(output_dir, paper_id)
    resolved_paper_id = paper_id_from_report(report_path)
    layout = load_layout(output_dir, resolved_paper_id)
    paper_memory_v3 = read_paper_memory_v3(output_dir, resolved_paper_id)
    library_record = load_library_record(output_dir, resolved_paper_id)
    pages = select_relevant_pages(layout, question, paper_memory_v3=paper_memory_v3)
    question_type = classify_question(question)
    layout_pages = [page for page in layout.get("pages", []) if isinstance(page, dict)]
    runtime = PaperLensRuntime(artifacts=layout_pages)
    agent_context = runtime.build_context_pack(
        stage="qa",
        objective=(
            "Answer the user's question by dynamically grounding it in PaperMemoryV3 and local "
            "paper evidence. Treat report text as orientation, not proof."
        ),
        paper_id=resolved_paper_id,
        title=string_or_empty(
            (paper_memory_v3.get("metadata") or {}).get("title")
            if isinstance(paper_memory_v3.get("metadata"), dict)
            else None
        ),
        classification=string_or_empty(
            (paper_memory_v3.get("reading_context") or {}).get("grade")
            if isinstance(paper_memory_v3.get("reading_context"), dict)
            else None
        ),
        memory=paper_memory_v3,
        focus_queries=[question, memory_augmented_query(question, paper_memory_v3)],
        focus_pages=[page.get("page_no") for page in pages if isinstance(page.get("page_no"), int)],
        read_artifacts=pages,
        output_contract={
            "type": "QAAnswer",
            "question_type": question_type,
            "rule": (
                "Answer conversationally, but keep paper claims, PaperLens inference, background, "
                "and evidence limits separate in source_attribution."
            ),
        },
        search_limit=5,
        page_text_limit=1000,
    ).as_dict()
    if config.offline_debug or config.provider.kind == "none":
        answer = {
            "paper_id": resolved_paper_id,
            "answer_markdown": (
                "当前是离线模式，不能调用模型回答。你可以先阅读这份报告："
                f"`{report_path.as_posix()}`。"
            ),
            "cited_pages": [page["page_no"] for page in pages[:3]],
            "confidence": "low",
            "question_type": question_type,
            "source_attribution": {
                "paper_claims": [],
                "paperlens_inferences": ["离线模式只返回可用报告位置，未进行模型问答。"],
                "background_context": [],
                "evidence_limits": ["未调用模型，不能可靠综合回答问题。"],
            },
        }
        write_qa_trace(
            output_dir,
            answer,
            question=question,
            selected_pages=pages,
            cache_hit=False,
            agent_context=agent_context,
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
        paper_memory_v3=paper_memory_v3,
        library_record=library_record,
        pages=pages,
        question_type=question_type,
        agent_context=agent_context,
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
        },
    )
    cached = read_ask_cache(cache_path)
    if cached and isinstance(cached.get("data"), dict):
        answer = normalize_answer(cached["data"])
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
        )
        return answer
    result = run_paper_qa_agent(
        client=client,
        output_dir=output_dir,
        paper_id=resolved_paper_id,
        title=string_or_empty(
            (paper_memory_v3.get("metadata") or {}).get("title")
            if isinstance(paper_memory_v3.get("metadata"), dict)
            else None
        ),
        question=question,
        question_type=question_type,
        chat_history=chat_history or [],
        paper_memory_v3=paper_memory_v3,
        library_record=library_record,
        layout_pages=layout_pages,
        selected_pages=pages,
        agent_context=agent_context,
        user_prompt=user_prompt,
    )
    answer = normalize_answer(result.final)
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
    )
    return answer


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
            "paper_memory_v3_ir": memory_v3_prompt_view(paper_memory_v3),
            "paperlens_library_record": library_record,
            "initial_page_hints": page_hints,
            "legacy_context_pack": agent_context,
            "legacy_prompt_for_compatibility": user_prompt,
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
) -> None:
    trace_path = paperlens_data_dir(output_dir) / "qa_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "paper_id": answer.get("paper_id"),
        "question": question,
        "question_type": answer.get("question_type"),
        "confidence": answer.get("confidence"),
        "cited_pages": answer.get("cited_pages"),
        "selected_pages": [page.get("page_no") for page in selected_pages[:8]],
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
    reports = sorted((output_dir / "papers").glob("*.md"))
    if not reports:
        raise FileNotFoundError(f"No paper reports found under {output_dir / 'papers'}")
    if paper_id:
        for report in reports:
            if report.name.startswith(paper_id):
                return report
        raise FileNotFoundError(f"No paper report found for paper_id={paper_id}")
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
    for claim in paper_memory_v3.get("claims") if isinstance(paper_memory_v3.get("claims"), list) else []:
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
    if any(token in normalized for token in ["不信", "核对", "真的吗", "真的", "challenge", "verify", "evidence", "证明"]):
        return "evidence_check"
    if any(token in normalized for token in ["比较", "区别", "相似", "不像", "像在哪里", "vs", "versus", "compare"]):
        return "comparison"
    if any(token in normalized for token in ["实现", "代码", "模块", "implementation", "implement"]):
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
    report_path: Path,
    paper_id: str,
    question: str,
    chat_history: list[dict[str, Any]] | None = None,
    paper_memory_v3: dict[str, Any] | None = None,
    pages: list[dict[str, Any]],
    question_type: str = "orientation",
    library_record: dict[str, Any] | None = None,
    agent_context: dict[str, Any] | None = None,
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
            f"report_path: {report_path.as_posix()}",
            f"question_type: {question_type}",
            "answer_mode:",
            qa_answer_mode_instruction(question_type),
            "recent_chat_history:",
            json.dumps(normalize_chat_history(chat_history or []), ensure_ascii=False),
            "question:",
            question,
            "paper_memory_v3_ir:",
            json.dumps(memory_v3_prompt_view(paper_memory_v3 or {}), ensure_ascii=False),
            "agent_context_pack:",
            context_pack_prompt(agent_context),
            "paperlens_library_record:",
            json.dumps(library_record or {}, ensure_ascii=False),
            "If page images are attached, they follow the same order as the first relevant_page_excerpts that have images.",
            "relevant_page_excerpts:",
            json.dumps(page_blocks, ensure_ascii=False),
            (
                "Answer contract: source_attribution.paper_claims should contain only claims directly "
                "supported by PaperMemoryV3 evidence/claims or relevant page excerpts. "
                "Use agent_context_pack as the active tool/context trace for this question. "
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
    answer = (
        string_or_empty(first_present(data, ["answer_markdown", "answer", "response", "content", "markdown", "text"]))
        or recover_text_answer(data)
    )
    answer = sanitize_qa_text(answer)
    for page in extract_page_citations(answer):
        if page not in cited_pages:
            cited_pages.append(page)
    source_attribution = normalize_source_attribution(data, answer=answer, confidence=confidence)
    return {
        "answer_markdown": answer.strip(),
        "cited_pages": cited_pages,
        "confidence": confidence,
        "source_attribution": source_attribution,
    }


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
            strings = [item.strip() for item in value if isinstance(item, str) and len(item.strip()) >= 10]
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
