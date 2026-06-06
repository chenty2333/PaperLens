from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    model_config = ConfigDict(extra="forbid")

    task_id: str
    task_type: ReadingTaskType
    target_source_ids: list[str] = Field(default_factory=list)
    required_outputs: list[str] = Field(default_factory=list)
    allowed_observation_types: list[str] = Field(default_factory=list)
    max_model_calls: int = 1
    max_tokens: int = 16000
    evidence_policy: str = "must_cite_paper_dom_source_ids"

    @field_validator("task_id", "evidence_policy")
    @classmethod
    def nonempty_text(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("reading task text fields cannot be blank")
        return text

    @field_validator("target_source_ids", "required_outputs", "allowed_observation_types")
    @classmethod
    def normalized_string_list(cls, value: list[str]) -> list[str]:
        result = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in result:
                result.append(text)
        return result

    @model_validator(mode="after")
    def validate_task_contract(self) -> "ReadingTask":
        if not self.required_outputs:
            raise ValueError(f"reading task {self.task_id} must declare required_outputs")
        if not self.allowed_observation_types:
            raise ValueError(f"reading task {self.task_id} must declare allowed_observation_types")
        expected_types = set(allowed_observation_types_for_task(self.task_type))
        unknown_types = [
            item for item in self.allowed_observation_types if item not in expected_types
        ]
        if unknown_types:
            raise ValueError(
                f"reading task {self.task_id} has invalid allowed_observation_types: "
                + ", ".join(unknown_types)
            )
        if self.max_model_calls < 1:
            raise ValueError(f"reading task {self.task_id} max_model_calls must be >= 1")
        if self.max_tokens < 1:
            raise ValueError(f"reading task {self.task_id} max_tokens must be >= 1")
        if self.evidence_policy != "must_cite_paper_dom_source_ids":
            raise ValueError(
                f"reading task {self.task_id} evidence_policy must be "
                "must_cite_paper_dom_source_ids"
            )
        return self


class ReadingPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "reading_plan.v2"
    paper_id: str
    tasks: list[ReadingTask]

    @field_validator("paper_id")
    @classmethod
    def nonempty_paper_id(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("reading plan paper_id cannot be blank")
        return text

    @model_validator(mode="after")
    def validate_plan_contract(self) -> "ReadingPlan":
        if not self.tasks:
            raise ValueError("reading plan must contain at least one task")
        task_ids = []
        for task in self.tasks:
            if task.task_id in task_ids:
                raise ValueError(f"reading plan contains duplicate task_id: {task.task_id}")
            task_ids.append(task.task_id)
        return self


TASK_KEYWORDS: dict[ReadingTaskType, tuple[str, ...]] = {
    ReadingTaskType.ORIENTATION: ("abstract", "introduction", "problem", "motivation"),
    ReadingTaskType.CLAIM_INVENTORY: (
        "contribution",
        "contributions",
        "we propose",
        "we present",
        "novel",
        "claim",
        "proposed",
    ),
    ReadingTaskType.METHOD_MECHANISM: (
        "method",
        "methodology",
        "approach",
        "architecture",
        "algorithm",
        "design",
        "framework",
        "module",
        "network",
        "optimization",
        "loss",
    ),
    ReadingTaskType.IMPLEMENTATION_PATH: (
        "implementation",
        "training",
        "module",
        "loss",
        "runtime",
        "system",
    ),
    ReadingTaskType.EVALUATION_SETUP: (
        "experiment",
        "evaluation",
        "dataset",
        "database",
        "baseline",
        "metric",
        "plcc",
        "srocc",
        "krocc",
        "rmse",
    ),
    ReadingTaskType.RESULT_EXTRACTION: (
        "result",
        "ablation",
        "table",
        "figure",
        "performance",
        "outperform",
        "gain",
    ),
    ReadingTaskType.LIMITATIONS: ("limitation", "discussion", "failure", "future work"),
    ReadingTaskType.CONCEPT_BRIDGE: ("background", "preliminary", "definition"),
    ReadingTaskType.RELATED_POSITIONING: ("related work", "prior", "compare", "comparison"),
    ReadingTaskType.REPRODUCIBILITY: (
        "code",
        "hardware",
        "repository",
        "implementation detail",
        "pytorch",
        "gpu",
        "optimizer",
        "learning rate",
        "batch size",
        "epoch",
        "cross-validation",
    ),
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


SECTION_TASK_MAPPING: dict[str, tuple[ReadingTaskType, ...]] = {
    "abstract": (ReadingTaskType.ORIENTATION, ReadingTaskType.CLAIM_INVENTORY),
    "introduction": (ReadingTaskType.ORIENTATION, ReadingTaskType.CLAIM_INVENTORY, ReadingTaskType.RELATED_POSITIONING),
    "background": (ReadingTaskType.CONCEPT_BRIDGE, ReadingTaskType.ORIENTATION),
    "related": (ReadingTaskType.RELATED_POSITIONING,),
    "method": (ReadingTaskType.METHOD_MECHANISM, ReadingTaskType.IMPLEMENTATION_PATH),
    "approach": (ReadingTaskType.METHOD_MECHANISM,),
    "design": (ReadingTaskType.METHOD_MECHANISM, ReadingTaskType.IMPLEMENTATION_PATH),
    "architecture": (ReadingTaskType.METHOD_MECHANISM, ReadingTaskType.IMPLEMENTATION_PATH),
    "implementation": (ReadingTaskType.IMPLEMENTATION_PATH, ReadingTaskType.REPRODUCIBILITY),
    "experiment": (ReadingTaskType.EVALUATION_SETUP,),
    "evaluation": (ReadingTaskType.EVALUATION_SETUP, ReadingTaskType.RESULT_EXTRACTION),
    "result": (ReadingTaskType.RESULT_EXTRACTION,),
    "discussion": (ReadingTaskType.LIMITATIONS,),
    "limitation": (ReadingTaskType.LIMITATIONS,),
    "conclusion": (ReadingTaskType.LIMITATIONS, ReadingTaskType.CLAIM_INVENTORY),
    "reproducibility": (ReadingTaskType.REPRODUCIBILITY,),
    "references": (),
    "bibliography": (),
}

FRONT_MATTER_TASKS: tuple[ReadingTaskType, ...] = (
    ReadingTaskType.ORIENTATION,
    ReadingTaskType.CLAIM_INVENTORY,
    ReadingTaskType.CONCEPT_BRIDGE,
    ReadingTaskType.RELATED_POSITIONING,
)


TASK_DEPENDENCY_ORDER: tuple[ReadingTaskType, ...] = (
    ReadingTaskType.ORIENTATION,
    ReadingTaskType.CONCEPT_BRIDGE,
    ReadingTaskType.RELATED_POSITIONING,
    ReadingTaskType.CLAIM_INVENTORY,
    ReadingTaskType.METHOD_MECHANISM,
    ReadingTaskType.IMPLEMENTATION_PATH,
    ReadingTaskType.EVALUATION_SETUP,
    ReadingTaskType.RESULT_EXTRACTION,
    ReadingTaskType.LIMITATIONS,
    ReadingTaskType.REPRODUCIBILITY,
)


def build_initial_reading_plan(
    dom: PaperDOM, *, max_sources_per_task: int = 8, max_tokens_per_task: int = 16000
) -> ReadingPlan:
    section_task_map = _map_sections_to_tasks(dom)
    tasks: list[ReadingTask] = []
    task_index = 0

    for task_type in TASK_DEPENDENCY_ORDER:
        targets = _select_sources_for_task_type(
            dom, task_type, section_task_map, max_sources_per_task
        )
        if not targets:
            continue
        task_index += 1
        tasks.append(
            ReadingTask(
                task_id=f"read_{task_index:02d}_{task_type.value}",
                task_type=task_type,
                target_source_ids=targets,
                required_outputs=required_outputs_for_task(task_type),
                allowed_observation_types=allowed_observation_types_for_task(task_type),
                max_model_calls=1,
                max_tokens=max_tokens_per_task,
            )
        )

    if not tasks:
        tasks = _fallback_tasks(dom, max_sources_per_task, max_tokens_per_task)

    return ReadingPlan(paper_id=dom.paper_id, tasks=tasks)


def _map_sections_to_tasks(dom: PaperDOM) -> dict[str, list[ReadingTaskType]]:
    result: dict[str, list[ReadingTaskType]] = {}
    for section in dom.sections:
        title_lower = normalize_heading_text(section.title)
        matched: list[ReadingTaskType] = []
        if title_lower in {"front", "front matter"}:
            result[section.source_id] = list(FRONT_MATTER_TASKS)
            continue
        for keyword, task_types in SECTION_TASK_MAPPING.items():
            if keyword in title_lower:
                for task_type in task_types:
                    if task_type not in matched:
                        matched.append(task_type)
        result[section.source_id] = matched
    return result


def _select_sources_for_task_type(
    dom: PaperDOM,
    task_type: ReadingTaskType,
    section_task_map: dict[str, list[ReadingTaskType]],
    limit: int,
) -> list[str]:
    keywords = TASK_KEYWORDS[task_type]
    spans_by_id = {span.source_id: span for span in dom.spans}
    section_titles = {
        section.source_id: normalize_heading_text(section.title) for section in dom.sections
    }
    matches: list[str] = []

    for section in dom.sections:
        if task_type not in section_task_map.get(section.source_id, []):
            continue
        matches.extend(
            non_heading_section_span_ids(section.span_ids, spans_by_id, section.title)
        )

    for span in dom.spans:
        if is_bad_candidate_span(span, section_titles):
            continue
        if span.source_id in matches:
            continue
        haystack = span.text.lower()
        if any(keyword in haystack for keyword in keywords):
            matches.append(span.source_id)

    if task_type in EQUATION_READING_TASKS:
        matches.extend(item.source_id for item in dom.equations)
    if task_type == ReadingTaskType.RESULT_EXTRACTION:
        matches.extend(item.source_id for item in [*dom.figures, *dom.tables])

    matches = prioritize_task_source_ids(matches, spans_by_id, keywords, task_type)
    deduped: list[str] = []
    for source_id in matches:
        if source_id and source_id not in deduped:
            deduped.append(source_id)
        if len(deduped) >= limit:
            return deduped

    if not deduped:
        return _positional_fallback(dom, task_type, limit, section_titles)
    return deduped


def prioritize_task_source_ids(
    source_ids: list[str],
    spans_by_id: dict[str, Any],
    keywords: tuple[str, ...],
    task_type: ReadingTaskType,
) -> list[str]:
    indexed = list(enumerate(source_ids))

    def sort_key(item: tuple[int, str]) -> tuple[float, int]:
        index, source_id = item
        span = spans_by_id.get(source_id)
        if span is None:
            return (0.0, index)
        text = " ".join(str(getattr(span, "text", "") or "").split())
        text_lower = text.lower()
        length = len(text)
        score = min(length, 1800) / 10.0
        if is_bad_candidate_text(text):
            score -= 320.0
        if length < 80:
            score -= 120.0
        if length >= 120:
            score += 80.0
        if any(keyword in text_lower for keyword in keywords):
            score += 180.0
        if task_type == ReadingTaskType.ORIENTATION and (
            "abstract" in text_lower or "introduction" in text_lower
        ):
            score += 240.0
        return (-score, index)

    return [source_id for _, source_id in sorted(indexed, key=sort_key)]


def _positional_fallback(
    dom: PaperDOM,
    task_type: ReadingTaskType,
    limit: int,
    section_titles: dict[str, str],
) -> list[str]:
    spans = non_heading_spans(dom.spans, section_titles) or dom.spans
    start, end = {
        ReadingTaskType.ORIENTATION: (0.0, 0.18),
        ReadingTaskType.CLAIM_INVENTORY: (0.0, 0.28),
        ReadingTaskType.METHOD_MECHANISM: (0.18, 0.58),
        ReadingTaskType.IMPLEMENTATION_PATH: (0.25, 0.68),
        ReadingTaskType.EVALUATION_SETUP: (0.48, 0.82),
        ReadingTaskType.RESULT_EXTRACTION: (0.58, 0.92),
        ReadingTaskType.LIMITATIONS: (0.72, 1.0),
        ReadingTaskType.CONCEPT_BRIDGE: (0.0, 0.32),
        ReadingTaskType.RELATED_POSITIONING: (0.0, 0.4),
        ReadingTaskType.REPRODUCIBILITY: (0.68, 1.0),
    }[task_type]
    candidates: list[str] = [
        span.source_id for span in positional_span_window(spans, start=start, end=end)
    ]
    if task_type in EQUATION_READING_TASKS:
        candidates.extend(item.source_id for item in dom.equations)
    if task_type == ReadingTaskType.RESULT_EXTRACTION:
        candidates.extend(item.source_id for item in [*dom.figures, *dom.tables])
    result: list[str] = []
    for source_id in candidates:
        if source_id and source_id not in result:
            result.append(source_id)
        if len(result) >= limit:
            return result
    return result


def _fallback_tasks(
    dom: PaperDOM, max_sources_per_task: int, max_tokens_per_task: int
) -> list[ReadingTask]:
    section_titles = {
        section.source_id: normalize_heading_text(section.title) for section in dom.sections
    }
    tasks: list[ReadingTask] = []
    for index, task_type in enumerate(TASK_DEPENDENCY_ORDER, start=1):
        targets = _positional_fallback(dom, task_type, max_sources_per_task, section_titles)
        if not targets:
            continue
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
    return tasks


def positional_span_window(spans: list[Any], *, start: float, end: float) -> list[Any]:
    if not spans:
        return []
    count = len(spans)
    start_index = max(0, min(count - 1, int(count * start)))
    end_index = max(start_index + 1, min(count, int(count * end) or 1))
    return spans[start_index:end_index]


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
        if is_bad_candidate_text(getattr(span, "text", "")):
            continue
        result.append(span_id)
    return result


def normalize_heading_text(text: str) -> str:
    return " ".join(text.lower().split())


def non_heading_spans(spans: list[Any], section_titles: dict[str, str]) -> list[Any]:
    return [span for span in spans if not is_bad_candidate_span(span, section_titles)]


def is_heading_span(span: Any, section_titles: dict[str, str]) -> bool:
    section_id = getattr(span, "section_id", None)
    section_title = section_titles.get(section_id)
    return bool(
        section_title and normalize_heading_text(getattr(span, "text", "")) == section_title
    )


def is_bad_candidate_span(span: Any, section_titles: dict[str, str]) -> bool:
    if is_heading_span(span, section_titles):
        return True
    return is_bad_candidate_text(getattr(span, "text", ""))


def is_bad_candidate_text(text: str) -> bool:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) < 40:
        return True
    lowered = cleaned.lower()
    if lowered.isdigit():
        return True
    if lowered.startswith("references") or lowered.startswith("bibliography"):
        return True
    if re_like_reference_entry(cleaned):
        return True
    metadata_markers = (
        "corresponding author",
        "e-mail",
        "email",
        "supported in part",
        "arxiv:",
        "university",
    )
    return any(marker in lowered for marker in metadata_markers)


def re_like_reference_entry(text: str) -> bool:
    stripped = text.strip()
    if stripped.startswith("[") and len(stripped) > 3 and stripped[1:3].strip("]").isdigit():
        return True
    citation_markers = stripped.count("[")
    return citation_markers >= 6 and len(stripped) > 600


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
