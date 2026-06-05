from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from paperlens_core.memory_v3 import dict_value, list_payload, memory_v3_prompt_view


CONTEXT_PACK_SCHEMA_VERSION = "paperlens.context_pack.v1"


@dataclass(frozen=True)
class ToolObservation:
    tool: str
    query: str
    results: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {"tool": self.tool, "query": self.query, "results": self.results}


@dataclass(frozen=True)
class ContextPack:
    """Small, explicit working context for one agent step.

    The model still receives a fresh API context on every call. PaperLens keeps
    continuity by rebuilding this pack from durable memory plus local tool
    observations, then requiring the model to emit a bounded artifact such as a
    MemoryPatchSet or QA answer.
    """

    stage: str
    objective: str
    always: dict[str, Any]
    working: dict[str, Any]
    tool_trace: list[dict[str, Any]]
    output_contract: dict[str, Any]
    budget: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CONTEXT_PACK_SCHEMA_VERSION,
            "stage": self.stage,
            "objective": self.objective,
            "always_context": self.always,
            "working_context": self.working,
            "tool_trace": self.tool_trace,
            "output_contract": self.output_contract,
            "budget": self.budget,
            "contract": (
                "Treat PaperMemory as durable state, page/search results as local tool observations, "
                "and this step's output as a patch or bounded answer. Do not pretend the whole paper "
                "is in context; request or use focused evidence when a claim is uncertain."
            ),
        }


class PaperLensRuntime:
    """Deterministic paper tools for the domain-specific agent runtime."""

    def __init__(self, *, artifacts: Iterable[Any]) -> None:
        self.pages = [page for page in artifacts if page_no(page) is not None]
        self.pages.sort(key=lambda item: page_no(item) or 0)

    def search_text(self, query: str, *, limit: int = 6) -> ToolObservation:
        terms = tokenize(query)
        if not terms:
            return ToolObservation(tool="paper.search_text", query=query, results=[])
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for page in self.pages:
            text = page_text(page)
            captions = page_captions_text(page)
            haystack = normalize_for_search(text + "\n" + captions)
            score = sum(haystack.count(term) for term in terms)
            if score <= 0:
                continue
            number = page_no(page) or 0
            scored.append(
                (
                    score,
                    number,
                    {
                        "page_no": number,
                        "matched_terms": [term for term in terms if term in haystack][:8],
                        "snippet": best_snippet(text, terms, limit=520),
                        "captions": page_captions(page)[:3],
                    },
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1]))
        return ToolObservation(
            tool="paper.search_text",
            query=query,
            results=[item for _score, _page_no, item in scored[:limit]],
        )

    def read_pages(self, page_numbers: Iterable[int], *, text_limit: int = 1400) -> ToolObservation:
        requested = []
        for value in page_numbers:
            try:
                number = int(value)
            except (TypeError, ValueError):
                continue
            if number > 0 and number not in requested:
                requested.append(number)
        by_no = {page_no(page): page for page in self.pages}
        results = []
        for number in requested:
            page = by_no.get(number)
            if page is None:
                continue
            results.append(
                {
                    "page_no": number,
                    "text": compact_text(page_text(page), limit=text_limit),
                    "captions": page_captions(page)[:5],
                    "figures": page_list_field(page, "figures")[:4],
                    "tables": page_list_field(page, "tables")[:4],
                    "visual_notes": page_list_field(page, "visual_notes")[:4],
                }
            )
        return ToolObservation(
            tool="paper.read_pages", query=",".join(map(str, requested)), results=results
        )

    def find_figures(self, query: str, *, limit: int = 4) -> ToolObservation:
        terms = tokenize(query)
        scored: list[tuple[int, int, dict[str, Any]]] = []
        for page in self.pages:
            visual_text = json.dumps(
                {
                    "captions": page_captions(page),
                    "figures": page_list_field(page, "figures"),
                    "tables": page_list_field(page, "tables"),
                    "visual_notes": page_list_field(page, "visual_notes"),
                },
                ensure_ascii=False,
            )
            haystack = normalize_for_search(visual_text)
            score = sum(haystack.count(term) for term in terms)
            if score <= 0 and not any(key in haystack for key in ["figure", "table", "图", "表"]):
                continue
            number = page_no(page) or 0
            if score <= 0:
                score = 1
            scored.append(
                (
                    score,
                    number,
                    {
                        "page_no": number,
                        "captions": page_captions(page)[:5],
                        "figures": page_list_field(page, "figures")[:4],
                        "tables": page_list_field(page, "tables")[:4],
                        "visual_notes": page_list_field(page, "visual_notes")[:4],
                    },
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1]))
        return ToolObservation(
            tool="paper.find_figures",
            query=query,
            results=[item for _score, _page_no, item in scored[:limit]],
        )

    def build_context_pack(
        self,
        *,
        stage: str,
        objective: str,
        paper_id: str,
        title: str | None = None,
        classification: str | None = None,
        memory: dict[str, Any] | None = None,
        focus_queries: Iterable[str] = (),
        focus_pages: Iterable[int] = (),
        read_artifacts: Iterable[Any] = (),
        output_contract: dict[str, Any] | None = None,
        search_limit: int = 4,
        page_text_limit: int = 900,
    ) -> ContextPack:
        memory = memory if isinstance(memory, dict) else {}
        queries = dedupe_queries(focus_queries)
        explicit_pages = unique_pages(focus_pages)
        tool_trace: list[dict[str, Any]] = []
        candidate_pages: list[int] = []
        for query in queries[:5]:
            observation = self.search_text(query, limit=search_limit)
            tool_trace.append(observation.as_dict())
            for result in observation.results[:2]:
                add_page(candidate_pages, result.get("page_no"))
        for page in explicit_pages:
            add_page(candidate_pages, page)
        if candidate_pages:
            tool_trace.append(
                self.read_pages(candidate_pages[:8], text_limit=page_text_limit).as_dict()
            )
        if queries:
            tool_trace.append(self.find_figures(" ".join(queries[:2]), limit=3).as_dict())
        already_read = sorted(
            {
                number
                for number in (page_no(page) for page in read_artifacts)
                if isinstance(number, int)
            }
        )
        base = build_always_context(
            paper_id=paper_id,
            title=title,
            classification=classification,
            memory=memory,
        )
        working = {
            "focus_queries": queries[:5],
            "focus_pages": candidate_pages[:8],
            "already_read_pages": already_read,
            "unread_candidate_pages": [
                page for page in candidate_pages[:8] if page not in set(already_read)
            ],
            "memory_uncertainty": memory_uncertainty(memory),
        }
        return ContextPack(
            stage=stage,
            objective=objective,
            always=base,
            working=working,
            tool_trace=tool_trace,
            output_contract=output_contract
            or {
                "type": "bounded_agent_step",
                "rule": "Return only the requested artifact; write durable facts as MemoryPatch operations.",
            },
            budget={
                "search_queries": min(len(queries), 5),
                "candidate_pages": min(len(candidate_pages), 8),
                "page_text_limit": page_text_limit,
                "whole_paper_in_context": False,
            },
        )

    def audit_context(
        self,
        *,
        memory: dict[str, Any],
        read_artifacts: Iterable[Any],
        audit: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        queries = audit_queries(memory, audit)
        observations: list[dict[str, Any]] = []
        evidence_pages = []
        for query in queries[:5]:
            search = self.search_text(query, limit=4)
            observations.append(search.as_dict())
            for result in search.results[:2]:
                number = result.get("page_no")
                if isinstance(number, int) and number not in evidence_pages:
                    evidence_pages.append(number)
        explicit_pages = pages_from_memory(memory)
        for page in explicit_pages:
            if page not in evidence_pages:
                evidence_pages.append(page)
        read_pages = [page_no(page) for page in read_artifacts if page_no(page) is not None]
        missing_pages = [page for page in evidence_pages if page not in read_pages][:8]
        page_pack = self.read_pages(evidence_pages[:8], text_limit=900)
        figure_query = " ".join(queries[:2]) or str(memory.get("core_abstraction") or "")
        figures = self.find_figures(figure_query, limit=3)
        context_pack = self.build_context_pack(
            stage="central_memory_verify",
            objective=(
                "Verify PaperMemoryV3 like a lightweight paper-reading agent: decide which claims "
                "are grounded, which should be weakened, and which evidence boundaries must remain visible."
            ),
            paper_id=str(memory.get("paper_id") or "unknown"),
            title=str(dict_value(memory.get("metadata")).get("title") or ""),
            classification=str(dict_value(memory.get("reading_context")).get("grade") or ""),
            memory=memory,
            focus_queries=queries[:5],
            focus_pages=evidence_pages[:8],
            read_artifacts=read_artifacts,
            output_contract={
                "type": "MemoryPatchSet",
                "rule": (
                    "Use the local tool trace to fix or bound memory in one verification pass."
                ),
            },
            search_limit=4,
            page_text_limit=900,
        )
        return {
            "runtime_contract": (
                "Deterministic local retrieval over parsed paper text/captions. Use it to verify "
                "where claims might be grounded before finalizing the memory boundary."
            ),
            "agent_context_pack": context_pack.as_dict(),
            "queries": queries[:5],
            "already_read_pages": sorted({page for page in read_pages if isinstance(page, int)}),
            "candidate_unread_pages": missing_pages,
            "observations": observations,
            "evidence_page_pack": page_pack.as_dict(),
            "visual_candidates": figures.as_dict(),
        }


def build_always_context(
    *,
    paper_id: str,
    title: str | None,
    classification: str | None,
    memory: dict[str, Any],
) -> dict[str, Any]:
    prompt_view = memory_v3_prompt_view(memory) if memory else {}
    claims = []
    for claim in list_payload(prompt_view.get("claims"))[:10]:
        if not isinstance(claim, dict):
            continue
        claims.append(
            {
                "id": claim.get("id"),
                "text": compact_text(str(claim.get("text") or ""), limit=240),
                "type": claim.get("type"),
                "provenance": claim.get("provenance"),
                "confidence": claim.get("confidence"),
                "critic_status": claim.get("critic_status"),
                "evidence_refs": claim.get("evidence_refs")
                if isinstance(claim.get("evidence_refs"), list)
                else [],
            }
        )
    evidence = []
    for item in list_payload(prompt_view.get("evidence"))[:12]:
        if not isinstance(item, dict):
            continue
        evidence.append(
            {
                "id": item.get("id"),
                "page": item.get("page") or item.get("page_no"),
                "source_type": item.get("source_type"),
                "reliability": item.get("reliability"),
                "hint": compact_text(
                    str(item.get("interpretation") or item.get("excerpt_or_caption") or ""),
                    limit=180,
                ),
            }
        )
    return {
        "paper_id": paper_id,
        "title": title or dict_value(memory.get("metadata")).get("title") or paper_id,
        "classification": classification or dict_value(memory.get("reading_context")).get("grade"),
        "source_of_truth": "PaperMemoryV3 plus evidence refs; reports are derived views.",
        "current_memory": {
            "schema_version": prompt_view.get("schema_version"),
            "reading_context": prompt_view.get("reading_context"),
            "problem_frame": prompt_view.get("problem_frame"),
            "core_abstractions": prompt_view.get("core_abstractions"),
            "mechanism": prompt_view.get("mechanism"),
            "evaluation": prompt_view.get("evaluation"),
            "conceptual_bridge": prompt_view.get("conceptual_bridge"),
            "limitations": prompt_view.get("limitations"),
            "open_questions": prompt_view.get("open_questions"),
        },
        "known_claims": claims,
        "known_evidence": evidence,
    }


def memory_uncertainty(memory: dict[str, Any]) -> dict[str, Any]:
    audit = dict_value(dict_value(memory.get("audit_trail")).get("memory_audit"))
    claims = list_payload(memory.get("claims"))
    weak_claims = [
        {
            "id": claim.get("id"),
            "text": compact_text(str(claim.get("text") or ""), limit=220),
            "confidence": claim.get("confidence"),
            "critic_status": claim.get("critic_status"),
        }
        for claim in claims
        if isinstance(claim, dict)
        and (
            claim.get("confidence") in {"low", "medium"}
            or claim.get("critic_status") in {"unchecked", "disputed"}
        )
    ][:8]
    return {
        "audit_status": audit.get("status"),
        "missing_items": list_payload(audit.get("missing_items"))[:8],
        "unsupported_claims": list_payload(audit.get("unsupported_claims"))[:6],
        "open_questions": list_payload(memory.get("open_questions"))[:8],
        "weak_claims": weak_claims,
    }


def context_pack_prompt(pack: ContextPack | dict[str, Any] | None) -> str:
    if pack is None:
        return "{}"
    payload = pack.as_dict() if isinstance(pack, ContextPack) else pack
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def unique_pages(values: Iterable[Any]) -> list[int]:
    pages: list[int] = []
    for value in values:
        add_page(pages, value)
    return pages


def audit_queries(memory: dict[str, Any], audit: dict[str, Any] | None) -> list[str]:
    queries: list[str] = []
    problem_frame = (
        memory.get("problem_frame") if isinstance(memory.get("problem_frame"), dict) else {}
    )
    for key in ["problem", "why_it_matters", "scope"]:
        value = problem_frame.get(key)
        if isinstance(value, str) and value.strip():
            queries.append(value)
    for item in (
        memory.get("core_abstractions") if isinstance(memory.get("core_abstractions"), list) else []
    ):
        if isinstance(item, dict) and isinstance(item.get("text"), str):
            queries.append(item["text"])
    mechanism = memory.get("mechanism") if isinstance(memory.get("mechanism"), dict) else {}
    if isinstance(mechanism.get("overview"), str):
        queries.append(mechanism["overview"])
    evaluation = memory.get("evaluation") if isinstance(memory.get("evaluation"), dict) else {}
    if isinstance(evaluation.get("summary"), str):
        queries.append(evaluation["summary"])
    for key in ["core_abstraction", "core_thesis"]:
        value = memory.get(key)
        if isinstance(value, str) and value.strip():
            queries.append(value)
    for item in memory.get("next_focus") if isinstance(memory.get("next_focus"), list) else []:
        if isinstance(item, str) and item.strip():
            queries.append(item)
    for item in (
        memory.get("uncertainties") if isinstance(memory.get("uncertainties"), list) else []
    ):
        if isinstance(item, str) and item.strip():
            queries.append(item)
    claims = memory.get("claims") if isinstance(memory.get("claims"), list) else []
    for item in claims:
        if not isinstance(item, dict):
            continue
        if isinstance(item.get("text"), str):
            queries.append(item["text"])
        elif isinstance(item.get("claim"), str):
            queries.append(item["claim"])
    if audit:
        for item in (
            audit.get("missing_items") if isinstance(audit.get("missing_items"), list) else []
        ):
            if isinstance(item, str) and item.strip():
                queries.append(item)
    return dedupe_queries(queries)


def pages_from_memory(memory: dict[str, Any]) -> list[int]:
    pages: list[int] = []
    reading_context = (
        memory.get("reading_context") if isinstance(memory.get("reading_context"), dict) else {}
    )
    for value in (
        reading_context.get("pages_read")
        if isinstance(reading_context.get("pages_read"), list)
        else []
    ):
        add_page(pages, value)
    evidence_id_to_page: dict[str, int] = {}
    evidence = memory.get("evidence") if isinstance(memory.get("evidence"), list) else []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        page_value = item.get("page") or item.get("page_no")
        add_page(pages, page_value)
        evidence_id = item.get("id")
        if isinstance(evidence_id, str):
            try:
                evidence_id_to_page[evidence_id] = int(page_value)
            except (TypeError, ValueError):
                pass
    for value in memory.get("pages_read") if isinstance(memory.get("pages_read"), list) else []:
        add_page(pages, value)
    claims = memory.get("claims") if isinstance(memory.get("claims"), list) else []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        for value in (
            claim.get("evidence_refs") if isinstance(claim.get("evidence_refs"), list) else []
        ):
            if isinstance(value, str) and value in evidence_id_to_page:
                add_page(pages, evidence_id_to_page[value])
            else:
                add_page(pages, value)
        for value in (
            claim.get("evidence_pages") if isinstance(claim.get("evidence_pages"), list) else []
        ):
            add_page(pages, value)
    return pages


def add_page(pages: list[int], value: Any) -> None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return
    if number > 0 and number not in pages:
        pages.append(number)


def dedupe_queries(values: Iterable[Any]) -> list[str]:
    seen = set()
    queries = []
    for value in values:
        if value is None:
            continue
        query = compact_text(str(value), limit=180)
        key = normalize_for_search(query)
        if not key or key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries


def page_no(page: Any) -> int | None:
    if isinstance(page, dict):
        value = page.get("page_no")
    else:
        value = getattr(page, "page_no", None)
    return value if isinstance(value, int) else None


def page_text(page: Any) -> str:
    if isinstance(page, dict):
        return str(page.get("text") or "")
    return str(getattr(page, "text", "") or "")


def page_list_field(page: Any, name: str) -> list[Any]:
    value = page.get(name) if isinstance(page, dict) else getattr(page, name, [])
    return value if isinstance(value, list) else []


def page_captions(page: Any) -> list[Any]:
    return page_list_field(page, "captions")


def page_captions_text(page: Any) -> str:
    captions = []
    for caption in page_captions(page):
        if isinstance(caption, dict):
            captions.append(str(caption.get("text") or ""))
        else:
            captions.append(str(caption))
    return "\n".join(captions)


def normalize_for_search(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9_]{3,}|[\u4e00-\u9fff]{2,}", text.lower())
    return list(dict.fromkeys(words))[:24]


def compact_text(text: str, *, limit: int) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "..."


def best_snippet(text: str, terms: list[str], *, limit: int) -> str:
    normalized = text or ""
    lower = normalized.lower()
    index = -1
    for term in terms:
        index = lower.find(term)
        if index >= 0:
            break
    if index < 0:
        return compact_text(normalized, limit=limit)
    start = max(0, index - limit // 3)
    end = min(len(normalized), start + limit)
    return compact_text(normalized[start:end], limit=limit)
