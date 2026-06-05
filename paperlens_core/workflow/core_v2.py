from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from paperlens_core.audit import audit_claim_graph, compute_core_quality_metrics
from paperlens_core.agents.llm import JsonLlmClient, llm_call_context
from paperlens_core.dom import PaperDOM, PaperSpan, build_paper_dom_from_layout
from paperlens_core.graph import ClaimGraph, graph_from_observations
from paperlens_core.memory import materialize_paper_memory
from paperlens_core.reading import (
    ObservationCard,
    ObservationLog,
    ObservationType,
    ReadingPlan,
    ReadingTask,
    ReadingTaskType,
    build_initial_reading_plan,
    make_observation_id,
)
from paperlens_core.report import audit_report_draft_against_graph, build_report_draft_from_graph
from paperlens_core.runtime import (
    ArtifactEnvelope,
    NodeSpec,
    NodeStatus,
    read_typed_artifact,
    run_finite_node,
    write_typed_artifact,
)
from paperlens_core.schemas import ClassificationDecision, PaperRecord, SkimCard


CORE_V2_SCHEMA_VERSION = "paperlens_core.v2.bootstrap"
CORE_V2_MODEL_OBSERVER_VERSION = "paperlens_core.v2.model_observer"
CORE_V2_AUDIT_SUITE_VERSION = "paperlens_core.v2.audit_suite"

CORE_V2_OBSERVER_SYSTEM_PROMPT = """
You are PaperLens ObservationReader.
Record only source-bound observations from the supplied evidence pack.
Do not write summaries, memory, audits, or report prose.
Return JSON only.
""".strip()

OBSERVATION_CARDS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["artifact_type", "artifact_version", "producer", "data"],
    "properties": {
        "artifact_type": {"type": "string", "enum": ["observation_cards"]},
        "artifact_version": {"type": "string"},
        "producer": {"type": "string"},
        "data": {
            "type": "object",
            "additionalProperties": False,
            "required": ["cards"],
            "properties": {
                "cards": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "observation_type",
                            "statement",
                            "source_ids",
                            "confidence",
                            "provenance",
                            "uncertainty",
                            "extracted_numbers",
                            "proposed_links",
                        ],
                        "properties": {
                            "observation_type": {
                                "type": "string",
                                "enum": [item.value for item in ObservationType],
                            },
                            "statement": {"type": "string"},
                            "source_ids": {"type": "array", "items": {"type": "string"}},
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                            "provenance": {
                                "type": "string",
                                "enum": ["explicit", "inferred", "background"],
                            },
                            "uncertainty": {"type": ["string", "null"]},
                            "extracted_numbers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["text"],
                                    "properties": {"text": {"type": "string"}},
                                },
                            },
                            "proposed_links": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["source_id", "target_id", "kind"],
                                    "properties": {
                                        "source_id": {"type": "string"},
                                        "target_id": {"type": "string"},
                                        "kind": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                }
            },
        },
    },
}


def write_core_v2_artifacts(
    *,
    data_dir: Path,
    paper: PaperRecord,
    layout: dict[str, Any],
    skim: SkimCard | None = None,
    decision: ClassificationDecision | None = None,
) -> dict[str, Path]:
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    reading_plan = build_initial_reading_plan(dom)
    observation_log = bootstrap_observation_log(dom, reading_plan)
    claim_graph = graph_from_observations(paper.paper_id, list(observation_log.cards))
    derived = build_core_v2_derived_views(
        dom=dom,
        claim_graph=claim_graph,
        metadata={
            "title": paper.canonical_title,
            "authors": paper.authors,
            "year": paper.year,
            "grade": decision.class_label if decision else None,
            "skim_problem": skim.problem if skim else None,
            "bootstrap_schema_version": CORE_V2_SCHEMA_VERSION,
        },
    )

    root = data_dir / "core" / "v2" / paper.paper_id
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "paper_dom": root / "paper_dom.v1.json",
        "reading_plan": root / "reading_plan.v1.json",
        "observation_log": root / "observation_log.v1.json",
        "claim_graph": root / "claim_graph.v1.json",
        "audit_findings": root / "audit_findings.v1.json",
        "quality_metrics": root / "quality_metrics.v1.json",
        "paper_memory_view": root / "paper_memory_view.v1.json",
        "report_draft": root / "report_draft.v1.json",
        "report_audit_findings": root / "report_audit_findings.v1.json",
    }
    write_core_v2_envelope(paths["paper_dom"], "paper_dom", paper.paper_id, dom.model_dump())
    write_core_v2_envelope(
        paths["reading_plan"], "reading_plan", paper.paper_id, reading_plan.model_dump()
    )
    write_core_v2_envelope(
        paths["observation_log"],
        "observation_log",
        paper.paper_id,
        observation_log.model_dump(),
        source_ids=sorted(dom.source_ids()),
    )
    write_core_v2_envelope(
        paths["claim_graph"], "claim_graph", paper.paper_id, claim_graph.model_dump()
    )
    write_core_v2_envelope(
        paths["audit_findings"],
        "audit_findings",
        paper.paper_id,
        [finding.model_dump() for finding in derived["audit_findings"]],
    )
    write_core_v2_envelope(
        paths["quality_metrics"],
        "core_quality_metrics",
        paper.paper_id,
        derived["quality_metrics"].model_dump(),
    )
    write_core_v2_envelope(
        paths["paper_memory_view"],
        "paper_memory_view",
        paper.paper_id,
        derived["memory_view"].model_dump(),
    )
    write_core_v2_envelope(
        paths["report_draft"],
        "graph_report_draft",
        paper.paper_id,
        derived["report_draft"].model_dump(),
    )
    write_core_v2_envelope(
        paths["report_audit_findings"],
        "report_audit_findings",
        paper.paper_id,
        [finding.model_dump() for finding in derived["report_audit_findings"]],
    )
    return paths


def run_core_v2_model_observation_tasks(
    *,
    client: JsonLlmClient,
    data_dir: Path,
    paper: PaperRecord,
    stage: str,
    record_usage: Any,
    record_agent_run: Any,
) -> dict[str, Any]:
    dom, reading_plan = load_core_v2_dom_and_plan(data_dir, paper.paper_id)
    log = ObservationLog(paper_id=paper.paper_id)
    total_usage: dict[str, Any] = {}
    request_ids: list[str] = []
    completed_tasks = 0
    for task in reading_plan.tasks:
        if not task.target_source_ids:
            continue
        raw_results: list[Any] = []
        spec = NodeSpec(
            node_id=f"core_v2_observe_{task.task_id}",
            input_artifact_types=("paper_dom", "reading_plan"),
            output_artifact_type="observation_cards",
            allowed_tools=("paper_dom.read_sources",),
            max_steps=1,
            max_model_calls=task.max_model_calls,
            timeout_seconds=180.0,
        )
        inputs = [
            ArtifactEnvelope(
                artifact_type="paper_dom",
                data=dom.model_dump(),
                producer="paperlens_core",
            ),
            ArtifactEnvelope(
                artifact_type="reading_plan",
                data=reading_plan.model_dump(),
                producer="paperlens_core",
            ),
        ]

        def handler(context: Any) -> ArtifactEnvelope:
            context.record_model_call()
            with llm_call_context(
                stage=stage,
                paper_id=paper.paper_id,
                operation="core_v2_observation_task",
                task_id=task.task_id,
                schema_name="paperlens_core_v2_observation_cards",
            ):
                raw = client.invoke_json(
                    system_prompt=CORE_V2_OBSERVER_SYSTEM_PROMPT,
                    user_prompt=build_observation_task_prompt(
                        paper=paper,
                        dom=dom,
                        task=task,
                    ),
                    schema_name="paperlens_core_v2_observation_cards",
                    schema=OBSERVATION_CARDS_SCHEMA,
                    max_tokens=None,
                )
            raw_results.append(raw)
            return ArtifactEnvelope.model_validate(raw.data).require_type("observation_cards")

        node_result = run_finite_node(spec, inputs, handler)
        raw = raw_results[0] if raw_results else None
        usage = dict(getattr(raw, "usage", {}) or {})
        merge_usage(total_usage, usage)
        request_id = getattr(raw, "request_id", None)
        if request_id:
            request_ids.append(request_id)
        record_usage(stage, usage)
        record_agent_run(
            {
                "agent_run_id": f"core_v2_observe_{paper.paper_id}_{task.task_id}",
                "paper_id": paper.paper_id,
                "stage": stage,
                "operation": "core_v2_observation_task",
                "task_id": task.task_id,
                "provider_kind": client.config.kind,
                "model": client.config.model,
                "usage": usage,
                "request_id": request_id,
                "status": node_result.status.value,
                "issues": node_result.issues,
                "model_calls_used": node_result.model_calls_used,
            }
        )
        if node_result.status != NodeStatus.PASS or node_result.output is None:
            raise RuntimeError(
                f"Core v2 observation task failed for {paper.paper_id}/{task.task_id}: "
                + "; ".join(node_result.issues)
            )
        for card in observation_cards_from_model_envelope(
            node_result.output,
            paper_id=paper.paper_id,
            task=task,
            valid_source_ids=dom.source_ids(),
        ):
            log = log.append(card)
        completed_tasks += 1

    paths = write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=reading_plan,
        observation_log=log,
        producer="paperlens_core_v2_model_observer",
    )
    return {
        "paths": paths,
        "cards": len(log.cards),
        "tasks": completed_tasks,
        "usage": total_usage,
        "request_ids": request_ids,
    }


def write_core_v2_from_observation_log(
    *,
    data_dir: Path,
    paper: PaperRecord,
    dom: PaperDOM,
    reading_plan: ReadingPlan,
    observation_log: ObservationLog,
    producer: str,
) -> dict[str, Path]:
    claim_graph = graph_from_observations(paper.paper_id, list(observation_log.cards))
    derived = build_core_v2_derived_views(
        dom=dom,
        claim_graph=claim_graph,
        metadata={
            "title": paper.canonical_title,
            "observer_schema_version": CORE_V2_MODEL_OBSERVER_VERSION,
        },
    )
    root = data_dir / "core" / "v2" / paper.paper_id
    paths = {
        "observation_log": root / "observation_log.v1.json",
        "claim_graph": root / "claim_graph.v1.json",
        "audit_findings": root / "audit_findings.v1.json",
        "quality_metrics": root / "quality_metrics.v1.json",
        "paper_memory_view": root / "paper_memory_view.v1.json",
        "report_draft": root / "report_draft.v1.json",
        "report_audit_findings": root / "report_audit_findings.v1.json",
    }
    write_core_v2_envelope(
        paths["observation_log"],
        "observation_log",
        paper.paper_id,
        observation_log.model_dump(),
        source_ids=sorted(dom.source_ids()),
        producer=producer,
    )
    write_core_v2_envelope(
        paths["claim_graph"],
        "claim_graph",
        paper.paper_id,
        claim_graph.model_dump(),
        producer=producer,
    )
    write_core_v2_envelope(
        paths["audit_findings"],
        "audit_findings",
        paper.paper_id,
        [finding.model_dump() for finding in derived["audit_findings"]],
        producer=producer,
    )
    write_core_v2_envelope(
        paths["quality_metrics"],
        "core_quality_metrics",
        paper.paper_id,
        derived["quality_metrics"].model_dump(),
        producer=producer,
    )
    write_core_v2_envelope(
        paths["paper_memory_view"],
        "paper_memory_view",
        paper.paper_id,
        derived["memory_view"].model_dump(),
        producer=producer,
    )
    write_core_v2_envelope(
        paths["report_draft"],
        "graph_report_draft",
        paper.paper_id,
        derived["report_draft"].model_dump(),
        producer=producer,
    )
    write_core_v2_envelope(
        paths["report_audit_findings"],
        "report_audit_findings",
        paper.paper_id,
        [finding.model_dump() for finding in derived["report_audit_findings"]],
        producer=producer,
    )
    return paths


def refresh_core_v2_audit_artifacts(
    *,
    data_dir: Path,
    paper: PaperRecord,
    skim: SkimCard | None = None,
    decision: ClassificationDecision | None = None,
    producer: str = "paperlens_core_v2_audit_suite",
) -> dict[str, Any]:
    dom, claim_graph = load_core_v2_dom_and_graph(data_dir, paper.paper_id)
    derived = build_core_v2_derived_views(
        dom=dom,
        claim_graph=claim_graph,
        metadata={
            "title": paper.canonical_title,
            "authors": paper.authors,
            "year": paper.year,
            "grade": decision.class_label if decision else None,
            "skim_problem": skim.problem if skim else None,
            "audit_schema_version": CORE_V2_AUDIT_SUITE_VERSION,
        },
    )
    root = data_dir / "core" / "v2" / paper.paper_id
    paths = {
        "audit_findings": root / "audit_findings.v1.json",
        "quality_metrics": root / "quality_metrics.v1.json",
        "paper_memory_view": root / "paper_memory_view.v1.json",
        "report_draft": root / "report_draft.v1.json",
        "report_audit_findings": root / "report_audit_findings.v1.json",
    }
    write_core_v2_envelope(
        paths["audit_findings"],
        "audit_findings",
        paper.paper_id,
        [finding.model_dump() for finding in derived["audit_findings"]],
        producer=producer,
    )
    write_core_v2_envelope(
        paths["quality_metrics"],
        "core_quality_metrics",
        paper.paper_id,
        derived["quality_metrics"].model_dump(),
        producer=producer,
    )
    write_core_v2_envelope(
        paths["paper_memory_view"],
        "paper_memory_view",
        paper.paper_id,
        derived["memory_view"].model_dump(),
        producer=producer,
    )
    write_core_v2_envelope(
        paths["report_draft"],
        "graph_report_draft",
        paper.paper_id,
        derived["report_draft"].model_dump(),
        producer=producer,
    )
    write_core_v2_envelope(
        paths["report_audit_findings"],
        "report_audit_findings",
        paper.paper_id,
        [finding.model_dump() for finding in derived["report_audit_findings"]],
        producer=producer,
    )
    return {
        "paths": paths,
        "graph_findings": len(derived["audit_findings"]),
        "report_findings": len(derived["report_audit_findings"]),
        "publish_status": derived["quality_metrics"].publish_status,
    }


def build_core_v2_derived_views(
    *,
    dom: PaperDOM,
    claim_graph: ClaimGraph,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    audit_findings = audit_claim_graph(claim_graph, dom)
    report_draft = build_report_draft_from_graph(claim_graph)
    report_audit_findings = audit_report_draft_against_graph(report_draft, claim_graph)
    all_findings = [*audit_findings, *report_audit_findings]
    quality_metrics = compute_core_quality_metrics(
        dom=dom,
        graph=claim_graph,
        findings=all_findings,
    )
    memory_view = materialize_paper_memory(
        claim_graph,
        metadata=metadata,
        unresolved_audit_findings=[finding.finding_id for finding in all_findings],
        report_readiness=quality_metrics.publish_status,
    )
    return {
        "audit_findings": audit_findings,
        "quality_metrics": quality_metrics,
        "memory_view": memory_view,
        "report_draft": report_draft,
        "report_audit_findings": report_audit_findings,
    }


def bootstrap_observation_log(dom: PaperDOM, reading_plan: ReadingPlan) -> ObservationLog:
    log = ObservationLog(paper_id=dom.paper_id)
    spans_by_id = {span.source_id: span for span in dom.spans}
    for task in reading_plan.tasks:
        card = bootstrap_observation_for_task(dom, spans_by_id, task)
        if card is not None:
            log = log.append(card)
    return log


def bootstrap_observation_for_task(
    dom: PaperDOM,
    spans_by_id: dict[str, PaperSpan],
    task: ReadingTask,
) -> ObservationCard | None:
    source_id = next((item for item in task.target_source_ids if item in spans_by_id), None)
    if not source_id:
        return None
    span = spans_by_id[source_id]
    statement = observation_statement(task.task_type, span.text)
    if not statement:
        return None
    observation_type = observation_type_for_task(task.task_type)
    observation_id = make_observation_id(
        task_id=task.task_id,
        observation_type=observation_type.value,
        statement=statement,
        source_ids=[source_id],
    )
    return ObservationCard(
        observation_id=observation_id,
        paper_id=dom.paper_id,
        task_id=task.task_id,
        observation_type=observation_type,
        statement=statement,
        source_ids=[source_id],
        confidence="low",
        provenance="explicit",
        uncertainty=(
            "Deterministic bootstrap observation; replace with task-specific model reading "
            "before treating as reviewed knowledge."
        ),
        extracted_numbers=extract_numbers(statement),
    )


def observation_type_for_task(task_type: ReadingTaskType) -> ObservationType:
    return {
        ReadingTaskType.ORIENTATION: ObservationType.PROBLEM,
        ReadingTaskType.CLAIM_INVENTORY: ObservationType.CLAIM,
        ReadingTaskType.METHOD_MECHANISM: ObservationType.MECHANISM,
        ReadingTaskType.IMPLEMENTATION_PATH: ObservationType.IMPLEMENTATION,
        ReadingTaskType.EVALUATION_SETUP: ObservationType.EVALUATION,
        ReadingTaskType.RESULT_EXTRACTION: ObservationType.RESULT,
        ReadingTaskType.LIMITATIONS: ObservationType.LIMITATION,
        ReadingTaskType.CONCEPT_BRIDGE: ObservationType.CONCEPT,
        ReadingTaskType.RELATED_POSITIONING: ObservationType.CLAIM,
        ReadingTaskType.REPRODUCIBILITY: ObservationType.IMPLEMENTATION,
    }[task_type]


def observation_statement(task_type: ReadingTaskType, text: str) -> str:
    sentence = first_sentence(text)
    if not sentence:
        return ""
    prefix = {
        ReadingTaskType.ORIENTATION: "Problem framing evidence: ",
        ReadingTaskType.CLAIM_INVENTORY: "Claim inventory evidence: ",
        ReadingTaskType.METHOD_MECHANISM: "Mechanism evidence: ",
        ReadingTaskType.IMPLEMENTATION_PATH: "Implementation evidence: ",
        ReadingTaskType.EVALUATION_SETUP: "Evaluation setup evidence: ",
        ReadingTaskType.RESULT_EXTRACTION: "Result evidence: ",
        ReadingTaskType.LIMITATIONS: "Limitation evidence: ",
        ReadingTaskType.CONCEPT_BRIDGE: "Concept bridge evidence: ",
        ReadingTaskType.RELATED_POSITIONING: "Related-positioning evidence: ",
        ReadingTaskType.REPRODUCIBILITY: "Reproducibility evidence: ",
    }[task_type]
    return prefix + sentence


def first_sentence(text: str, *, limit: int = 320) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return ""
    match = re.search(r"(.+?[.!?。！？])\s", cleaned + " ")
    sentence = match.group(1) if match else cleaned
    return sentence[:limit].strip()


def extract_numbers(text: str) -> list[dict[str, str]]:
    return [{"text": match.group(0)} for match in re.finditer(r"\b\d+(?:\.\d+)?%?\b", text)][:8]


def write_core_v2_envelope(
    path: Path,
    artifact_type: str,
    paper_id: str,
    data: dict[str, Any] | list[Any],
    *,
    source_ids: list[str] | None = None,
    producer: str = "paperlens_core_v2_bootstrap",
) -> None:
    write_typed_artifact(
        path,
        artifact_type=artifact_type,
        artifact_version="v1",
        data=data,
        producer=producer,
        source_ids=source_ids or [],
        metadata={"paper_id": paper_id, "schema_version": CORE_V2_SCHEMA_VERSION},
    )


def load_core_v2_dom_and_plan(data_dir: Path, paper_id: str) -> tuple[PaperDOM, ReadingPlan]:
    root = data_dir / "core" / "v2" / paper_id
    dom_envelope = read_typed_artifact(root / "paper_dom.v1.json", expected_type="paper_dom")
    plan_envelope = read_typed_artifact(root / "reading_plan.v1.json", expected_type="reading_plan")
    if not isinstance(dom_envelope.data, dict) or not isinstance(plan_envelope.data, dict):
        raise ValueError(f"Core v2 paper_dom/reading_plan artifacts are invalid for {paper_id}")
    return PaperDOM.model_validate(dom_envelope.data), ReadingPlan.model_validate(
        plan_envelope.data
    )


def load_core_v2_dom_and_graph(data_dir: Path, paper_id: str) -> tuple[PaperDOM, ClaimGraph]:
    root = data_dir / "core" / "v2" / paper_id
    dom_envelope = read_typed_artifact(root / "paper_dom.v1.json", expected_type="paper_dom")
    graph_envelope = read_typed_artifact(root / "claim_graph.v1.json", expected_type="claim_graph")
    if not isinstance(dom_envelope.data, dict) or not isinstance(graph_envelope.data, dict):
        raise ValueError(f"Core v2 paper_dom/claim_graph artifacts are invalid for {paper_id}")
    return PaperDOM.model_validate(dom_envelope.data), ClaimGraph.model_validate(
        graph_envelope.data
    )


def build_observation_task_prompt(
    *,
    paper: PaperRecord,
    dom: PaperDOM,
    task: ReadingTask,
) -> str:
    payload = {
        "prompt_version": CORE_V2_MODEL_OBSERVER_VERSION,
        "paper": {
            "paper_id": paper.paper_id,
            "title": paper.canonical_title,
        },
        "task_spec": {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "required_outputs": task.required_outputs,
            "evidence_policy": task.evidence_policy,
            "max_model_calls": task.max_model_calls,
        },
        "evidence_pack": source_pack(dom, task.target_source_ids),
        "output_contract": {
            "artifact_type": "observation_cards",
            "rule": (
                "Return an ArtifactEnvelope with data.cards. Each card must cite source_ids from "
                "the evidence_pack. Do not cite page numbers as evidence. Do not write memory, "
                "audit verdicts, or report prose."
            ),
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def source_pack(dom: PaperDOM, source_ids: list[str]) -> list[dict[str, Any]]:
    by_source: dict[str, dict[str, Any]] = {}
    for section in dom.sections:
        by_source[section.source_id] = {
            "source_id": section.source_id,
            "kind": section.kind,
            "page_no": section.page_no,
            "title": section.title,
        }
    for span in dom.spans:
        by_source[span.source_id] = {
            "source_id": span.source_id,
            "kind": span.kind,
            "page_no": span.page_no,
            "section_id": span.section_id,
            "text": span.text[:1800],
        }
    for figure in dom.figures:
        by_source[figure.source_id] = {
            "source_id": figure.source_id,
            "kind": figure.kind,
            "page_no": figure.page_no,
            "caption": figure.caption,
            "bbox": figure.bbox,
        }
    for table in dom.tables:
        by_source[table.source_id] = {
            "source_id": table.source_id,
            "kind": table.kind,
            "page_no": table.page_no,
            "caption": table.caption,
            "bbox": table.bbox,
        }
    for equation in dom.equations:
        by_source[equation.source_id] = {
            "source_id": equation.source_id,
            "kind": equation.kind,
            "page_no": equation.page_no,
            "section_id": equation.section_id,
            "text": equation.latex_or_text,
        }
    result = []
    for source_id in source_ids:
        if source_id in by_source:
            result.append(by_source[source_id])
    return result


def observation_cards_from_model_envelope(
    envelope: ArtifactEnvelope,
    *,
    paper_id: str,
    task: ReadingTask,
    valid_source_ids: set[str],
) -> list[ObservationCard]:
    payload = envelope.data if isinstance(envelope.data, dict) else {}
    cards = payload.get("cards") if isinstance(payload.get("cards"), list) else []
    result = []
    for item in cards:
        if not isinstance(item, dict):
            continue
        source_ids = clean_model_source_ids(item.get("source_ids"), valid_source_ids)
        if not source_ids:
            raise ValueError(f"Observation card for {task.task_id} did not cite valid source_ids")
        statement = str(item.get("statement") or "").strip()
        observation_type = str(item.get("observation_type") or "").strip()
        if not statement or observation_type not in {kind.value for kind in ObservationType}:
            continue
        observation_id = make_observation_id(
            task_id=task.task_id,
            observation_type=observation_type,
            statement=statement,
            source_ids=source_ids,
        )
        result.append(
            ObservationCard(
                observation_id=observation_id,
                paper_id=paper_id,
                task_id=task.task_id,
                observation_type=ObservationType(observation_type),
                statement=statement,
                source_ids=source_ids,
                confidence=str(item.get("confidence") or "medium"),
                provenance=str(item.get("provenance") or "explicit"),
                uncertainty=none_if_blank(item.get("uncertainty")),
                extracted_numbers=[
                    number
                    for number in list_payload(item.get("extracted_numbers"))
                    if isinstance(number, dict)
                ][:8],
                proposed_links=[
                    link
                    for link in list_payload(item.get("proposed_links"))
                    if isinstance(link, dict)
                ][:8],
            )
        )
    return result


def clean_model_source_ids(value: Any, valid_source_ids: set[str]) -> list[str]:
    result = []
    for item in list_payload(value):
        source_id = str(item or "").strip()
        if source_id in valid_source_ids and source_id not in result:
            result.append(source_id)
    return result


def list_payload(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def none_if_blank(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


def merge_usage(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            previous = target.get(key)
            target[key] = (previous if isinstance(previous, (int, float)) else 0) + value
        elif isinstance(value, dict):
            nested = target.setdefault(key, {})
            if isinstance(nested, dict):
                merge_usage(nested, value)
