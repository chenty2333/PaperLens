from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from paperlens_core.audit import audit_claim_graph, compute_core_quality_metrics
from paperlens_core.dom import PaperDOM, PaperSpan, build_paper_dom_from_layout
from paperlens_core.events import write_json
from paperlens_core.graph import graph_from_observations
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
from paperlens_core.runtime import ArtifactEnvelope
from paperlens_core.schemas import ClassificationDecision, PaperRecord, SkimCard


CORE_V2_SCHEMA_VERSION = "paperlens_core.v2.bootstrap"


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
    audit_findings = audit_claim_graph(claim_graph, dom)
    quality_metrics = compute_core_quality_metrics(
        dom=dom,
        graph=claim_graph,
        findings=audit_findings,
    )
    report_draft = build_report_draft_from_graph(claim_graph)
    report_audit_findings = audit_report_draft_against_graph(report_draft, claim_graph)
    memory_view = materialize_paper_memory(
        claim_graph,
        metadata={
            "title": paper.canonical_title,
            "authors": paper.authors,
            "year": paper.year,
            "grade": decision.class_label if decision else None,
            "skim_problem": skim.problem if skim else None,
            "bootstrap_schema_version": CORE_V2_SCHEMA_VERSION,
        },
        unresolved_audit_findings=[finding.finding_id for finding in audit_findings],
        report_readiness=quality_metrics.publish_status,
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
    write_envelope(paths["paper_dom"], "paper_dom", paper.paper_id, dom.model_dump())
    write_envelope(paths["reading_plan"], "reading_plan", paper.paper_id, reading_plan.model_dump())
    write_envelope(
        paths["observation_log"],
        "observation_log",
        paper.paper_id,
        observation_log.model_dump(),
        source_ids=sorted(dom.source_ids()),
    )
    write_envelope(paths["claim_graph"], "claim_graph", paper.paper_id, claim_graph.model_dump())
    write_envelope(
        paths["audit_findings"],
        "audit_findings",
        paper.paper_id,
        [finding.model_dump() for finding in audit_findings],
    )
    write_envelope(
        paths["quality_metrics"],
        "core_quality_metrics",
        paper.paper_id,
        quality_metrics.model_dump(),
    )
    write_envelope(
        paths["paper_memory_view"],
        "paper_memory_view",
        paper.paper_id,
        memory_view.model_dump(),
    )
    write_envelope(
        paths["report_draft"],
        "graph_report_draft",
        paper.paper_id,
        report_draft.model_dump(),
    )
    write_envelope(
        paths["report_audit_findings"],
        "report_audit_findings",
        paper.paper_id,
        [finding.model_dump() for finding in report_audit_findings],
    )
    return paths


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


def write_envelope(
    path: Path,
    artifact_type: str,
    paper_id: str,
    data: dict[str, Any] | list[Any],
    *,
    source_ids: list[str] | None = None,
) -> None:
    envelope = ArtifactEnvelope(
        artifact_type=artifact_type,
        artifact_version="v1",
        data=data,
        producer="paperlens_core_v2_bootstrap",
        source_ids=source_ids or [],
        metadata={"paper_id": paper_id, "schema_version": CORE_V2_SCHEMA_VERSION},
    )
    write_json(path, json.loads(envelope.model_dump_json()))
