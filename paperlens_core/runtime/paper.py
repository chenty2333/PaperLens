from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from paperlens_core.dom import stable_source_id
from paperlens_core.dom.paper_dom import split_paragraphs


CONTEXT_PACK_SCHEMA_VERSION = "paperlens.context_pack.v1"


@dataclass(frozen=True)
class ToolObservation:
    tool: str
    query: str
    results: list[dict[str, Any]]
    source_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        source_ids = self.source_ids or source_ids_from_results(self.results)
        return {
            "tool": self.tool,
            "query": self.query,
            "results": self.results,
            "source_ids": source_ids,
        }


@dataclass(frozen=True)
class ContextPack:
    """Small, explicit working context for one agent step.

    The model still receives a fresh API context on every call. PaperLens keeps
    continuity by rebuilding this pack from typed QA context plus local tool
    observations, then requiring the model to emit a bounded artifact such as a
    QA answer.
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
                "Treat ClaimGraph/PaperDOM source IDs as paper evidence, page/search results as "
                "local tool observations, and this step's output as a bounded answer. Do not "
                "pretend the whole paper is in context; request or use focused evidence when a "
                "claim is uncertain."
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
                        "source_ids": page_text_source_ids(page, terms=terms),
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
                    "source_ids": page_source_ids(page),
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
                        "source_ids": page_visual_source_ids(page),
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
        context: dict[str, Any] | None = None,
        focus_queries: Iterable[str] = (),
        focus_pages: Iterable[int] = (),
        read_artifacts: Iterable[Any] = (),
        output_contract: dict[str, Any] | None = None,
        search_limit: int = 4,
        page_text_limit: int = 900,
    ) -> ContextPack:
        context = context if isinstance(context, dict) else {}
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
            context=context,
        )
        working = {
            "focus_queries": queries[:5],
            "focus_pages": candidate_pages[:8],
            "already_read_pages": already_read,
            "unread_candidate_pages": [
                page for page in candidate_pages[:8] if page not in set(already_read)
            ],
            "context_uncertainty": context_uncertainty(context),
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
                "rule": "Return only the requested typed artifact.",
            },
            budget={
                "search_queries": min(len(queries), 5),
                "candidate_pages": min(len(candidate_pages), 8),
                "page_text_limit": page_text_limit,
                "whole_paper_in_context": False,
            },
        )


def build_always_context(
    *,
    paper_id: str,
    title: str | None,
    classification: str | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    claims = []
    for claim in list_payload(context.get("claims"))[:10]:
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
    for item in list_payload(context.get("evidence"))[:12]:
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
        "title": title or paper_id,
        "classification": classification,
        "source_of_truth": "ClaimGraph nodes plus PaperDOM source IDs; reports are derived views.",
        "qa_context": {
            "schema_version": context.get("schema_version"),
            "reading_context": context.get("reading_context"),
            "audit_trail": context.get("audit_trail"),
        },
        "known_claims": claims,
        "known_evidence": evidence,
    }


def context_uncertainty(context: dict[str, Any]) -> dict[str, Any]:
    audit = dict_value(context.get("audit_trail"))
    claims = list_payload(context.get("claims"))
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
        "evidence_limits": list_payload(audit.get("evidence_limits"))[:8],
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


def page_source_ids(page: Any, *, terms: list[str] | None = None) -> list[str]:
    return dedupe_source_ids(
        [
            *explicit_source_ids(page),
            *page_text_source_ids(page, terms=terms),
            *page_visual_source_ids(page),
        ]
    )


def page_text_source_ids(page: Any, *, terms: list[str] | None = None) -> list[str]:
    terms = terms or []
    source_ids = []
    for index, block in enumerate(page_list_field(page, "blocks"), start=1):
        if not isinstance(block, dict):
            continue
        text = str(block.get("text") or "")
        if terms and not any(term in normalize_for_search(text) for term in terms):
            continue
        source_id = str(block.get("source_id") or block.get("text_span_id") or "").strip()
        if source_id:
            source_ids.append(source_id)
        elif text.strip():
            source_ids.append(generated_source_id(page, "span", index))
    if source_ids:
        return dedupe_source_ids(source_ids)
    for index, paragraph in enumerate(split_paragraphs(page_text(page)), start=1):
        if terms and not any(term in normalize_for_search(paragraph) for term in terms):
            continue
        source_ids.append(generated_source_id(page, "span", index))
    return dedupe_source_ids(source_ids)


def page_visual_source_ids(page: Any) -> list[str]:
    source_ids = []
    for field_name, kind in [("figures", "figure"), ("tables", "table")]:
        for index, item in enumerate(page_list_field(page, field_name), start=1):
            if isinstance(item, dict):
                source_id = str(item.get("source_id") or "").strip()
                source_ids.append(source_id or generated_source_id(page, kind, index))
            elif item is not None:
                source_ids.append(generated_source_id(page, kind, index))
    return dedupe_source_ids(source_ids)


def explicit_source_ids(page: Any) -> list[str]:
    value = page.get("source_ids") if isinstance(page, dict) else getattr(page, "source_ids", [])
    return [str(item).strip() for item in list_payload(value) if str(item).strip()]


def source_ids_from_results(results: list[dict[str, Any]]) -> list[str]:
    source_ids = []
    for result in results:
        if not isinstance(result, dict):
            continue
        source_ids.extend(str(item).strip() for item in list_payload(result.get("source_ids")))
    return dedupe_source_ids(source_ids)


def generated_source_id(page: Any, kind: str, index: int) -> str:
    number = page_no(page) or "unknown"
    return stable_source_id(page_paper_id(page), kind, f"p{number}", index)


def page_paper_id(page: Any) -> str:
    value = page.get("paper_id") if isinstance(page, dict) else getattr(page, "paper_id", None)
    return str(value or "unknown").strip() or "unknown"


def dedupe_source_ids(values: Iterable[Any]) -> list[str]:
    result = []
    for value in values:
        source_id = str(value or "").strip()
        if source_id and source_id not in result:
            result.append(source_id)
    return result


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_payload(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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
