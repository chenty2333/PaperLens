from __future__ import annotations

from enum import StrEnum

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


def build_initial_reading_plan(dom: PaperDOM, *, max_sources_per_task: int = 8) -> ReadingPlan:
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
            )
        )
    return ReadingPlan(paper_id=dom.paper_id, tasks=tasks)


def select_sources_for_task(dom: PaperDOM, task_type: ReadingTaskType, *, limit: int) -> list[str]:
    keywords = TASK_KEYWORDS[task_type]
    matches = []
    for section in dom.sections:
        section_text = section.title.lower()
        if any(keyword in section_text for keyword in keywords):
            matches.extend(section.span_ids)
    for span in dom.spans:
        haystack = span.text.lower()
        if any(keyword in haystack for keyword in keywords):
            matches.append(span.source_id)
    if task_type == ReadingTaskType.RESULT_EXTRACTION:
        matches.extend(item.source_id for item in [*dom.figures, *dom.tables])
    deduped = []
    for source_id in matches:
        if source_id and source_id not in deduped:
            deduped.append(source_id)
        if len(deduped) >= limit:
            return deduped
    if not deduped:
        return [span.source_id for span in dom.spans[: min(limit, 3)]]
    return deduped


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
