from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from paperlens_core.dom.paper_dom import PaperDOM


class ReadingTaskType(StrEnum):
    ORIENTATION = "orientation"
    CLAIM_INVENTORY = "claim_inventory"
    METHOD_MECHANISM = "method_mechanism"
    IMPLEMENTATION_PATH = "implementation_path"
    EVALUATION_SETUP = "evaluation_setup"
    RESULT_EXTRACTION = "result_extraction"
    LIMITATIONS = "limitations"
    CONCEPT_BRIDGE = "concept_bridge"
    RELATED_POSITIONING = "related_positioning"
    REPRODUCIBILITY = "reproducibility"


class ReadingTask(BaseModel):
    task_id: str
    task_type: ReadingTaskType
    target_source_ids: list[str] = Field(default_factory=list)
    required_outputs: list[str] = Field(default_factory=list)
    allowed_observation_types: list[str] = Field(default_factory=list)
    max_model_calls: int = 1
    max_tokens: int = 16000
    evidence_policy: str = "must_cite_paper_dom_source_ids"


class ReadingPlan(BaseModel):
    schema_version: str = "reading_plan.v1"
    paper_id: str
    tasks: list[ReadingTask]


TASK_KEYWORDS: dict[ReadingTaskType, tuple[str, ...]] = {
    ReadingTaskType.ORIENTATION: ("abstract", "introduction", "problem", "motivation"),
    ReadingTaskType.CLAIM_INVENTORY: ("contribution", "we propose", "novel", "claim"),
    ReadingTaskType.METHOD_MECHANISM: ("method", "approach", "architecture", "algorithm", "design"),
    ReadingTaskType.IMPLEMENTATION_PATH: (
        "implementation",
        "training",
        "module",
        "loss",
        "runtime",
        "system",
    ),
    ReadingTaskType.EVALUATION_SETUP: ("experiment", "evaluation", "dataset", "baseline", "metric"),
    ReadingTaskType.RESULT_EXTRACTION: ("result", "ablation", "table", "figure", "performance"),
    ReadingTaskType.LIMITATIONS: ("limitation", "discussion", "failure", "future work"),
    ReadingTaskType.CONCEPT_BRIDGE: ("background", "preliminary", "definition"),
    ReadingTaskType.RELATED_POSITIONING: ("related work", "prior", "compare", "comparison"),
    ReadingTaskType.REPRODUCIBILITY: ("code", "hardware", "repository", "implementation detail"),
}

ALLOWED_OBSERVATION_TYPES: dict[ReadingTaskType, tuple[str, ...]] = {
    ReadingTaskType.ORIENTATION: ("problem",),
    ReadingTaskType.CLAIM_INVENTORY: ("claim",),
    ReadingTaskType.METHOD_MECHANISM: ("mechanism",),
    ReadingTaskType.IMPLEMENTATION_PATH: ("implementation",),
    ReadingTaskType.EVALUATION_SETUP: ("evaluation",),
    ReadingTaskType.RESULT_EXTRACTION: ("result",),
    ReadingTaskType.LIMITATIONS: ("limitation",),
    ReadingTaskType.CONCEPT_BRIDGE: ("concept",),
    ReadingTaskType.RELATED_POSITIONING: ("claim", "limitation"),
    ReadingTaskType.REPRODUCIBILITY: ("implementation", "limitation"),
}

EQUATION_READING_TASKS = {
    ReadingTaskType.METHOD_MECHANISM,
    ReadingTaskType.IMPLEMENTATION_PATH,
    ReadingTaskType.RESULT_EXTRACTION,
    ReadingTaskType.REPRODUCIBILITY,
}


def build_initial_reading_plan(
    dom: PaperDOM, *, max_sources_per_task: int = 8, max_tokens_per_task: int = 16000
) -> ReadingPlan:
    tasks = []
    for index, task_type in enumerate(ReadingTaskType, start=1):
        targets = select_sources_for_task(dom, task_type, limit=max_sources_per_task)
        tasks.append(
            ReadingTask(
                task_id=f"read_{index:02d}_{task_type.value}",
                task_type=task_type,
                target_source_ids=targets,
                required_outputs=required_outputs_for_task(task_type),
                allowed_observation_types=allowed_observation_types_for_task(task_type),
                max_model_calls=1,
                max_tokens=max_tokens_per_task,
            )
        )
    return ReadingPlan(paper_id=dom.paper_id, tasks=tasks)


def select_sources_for_task(dom: PaperDOM, task_type: ReadingTaskType, *, limit: int) -> list[str]:
    keywords = TASK_KEYWORDS[task_type]
    matches = []
    spans_by_id = {span.source_id: span for span in dom.spans}
    section_titles = {
        section.source_id: normalize_heading_text(section.title) for section in dom.sections
    }
    for section in dom.sections:
        section_text = section.title.lower()
        if any(keyword in section_text for keyword in keywords):
            matches.extend(
                non_heading_section_span_ids(section.span_ids, spans_by_id, section.title)
            )
    for span in dom.spans:
        if is_heading_span(span, section_titles):
            continue
        haystack = span.text.lower()
        if any(keyword in haystack for keyword in keywords):
            matches.append(span.source_id)
    if task_type in EQUATION_READING_TASKS:
        matches.extend(item.source_id for item in dom.equations)
    if task_type == ReadingTaskType.RESULT_EXTRACTION:
        matches.extend(item.source_id for item in [*dom.figures, *dom.tables])
    deduped = []
    for source_id in matches:
        if source_id and source_id not in deduped:
            deduped.append(source_id)
        if len(deduped) >= limit:
            return deduped
    if not deduped:
        fallback_spans = non_heading_spans(dom.spans, section_titles) or dom.spans
        return [span.source_id for span in fallback_spans[: min(limit, 3)]]
    return deduped


def non_heading_section_span_ids(
    span_ids: list[str],
    spans_by_id: dict[str, Any],
    section_title: str,
) -> list[str]:
    title = normalize_heading_text(section_title)
    result = []
    for span_id in span_ids:
        span = spans_by_id.get(span_id)
        text = normalize_heading_text(getattr(span, "text", ""))
        if title and text == title:
            continue
        result.append(span_id)
    return result


def normalize_heading_text(text: str) -> str:
    return " ".join(text.lower().split())


def non_heading_spans(spans: list[Any], section_titles: dict[str, str]) -> list[Any]:
    return [span for span in spans if not is_heading_span(span, section_titles)]


def is_heading_span(span: Any, section_titles: dict[str, str]) -> bool:
    section_id = getattr(span, "section_id", None)
    section_title = section_titles.get(section_id)
    return bool(
        section_title and normalize_heading_text(getattr(span, "text", "")) == section_title
    )


def required_outputs_for_task(task_type: ReadingTaskType) -> list[str]:
    return {
        ReadingTaskType.ORIENTATION: ["problem", "motivation", "scope"],
        ReadingTaskType.CLAIM_INVENTORY: ["claim"],
        ReadingTaskType.METHOD_MECHANISM: ["mechanism"],
        ReadingTaskType.IMPLEMENTATION_PATH: ["implementation"],
        ReadingTaskType.EVALUATION_SETUP: ["evaluation"],
        ReadingTaskType.RESULT_EXTRACTION: ["result"],
        ReadingTaskType.LIMITATIONS: ["limitation"],
        ReadingTaskType.CONCEPT_BRIDGE: ["concept"],
        ReadingTaskType.RELATED_POSITIONING: ["claim", "limitation"],
        ReadingTaskType.REPRODUCIBILITY: ["implementation", "limitation"],
    }[task_type]


def allowed_observation_types_for_task(task_type: ReadingTaskType) -> list[str]:
    return list(ALLOWED_OBSERVATION_TYPES[task_type])
