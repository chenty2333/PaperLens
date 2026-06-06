from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Protocol

from paperlens_core.audit import (
    audit_claim_graph,
    audit_claim_graph_from_observation_log,
    audit_observation_log,
    audit_reading_required_outputs,
    audit_relation_candidates,
    compute_core_quality_metrics,
)
from paperlens_core.agents.llm import JsonLlmClient, llm_call_context
from paperlens_core.config import CoreConfig
from paperlens_core.control import ControlState
from paperlens_core.core_manifest import write_core_v2_manifest
from paperlens_core.dom import PaperDOM, PaperSpan, build_paper_dom_from_layout
from paperlens_core.events import EventWriter
from paperlens_core.grounding import text_overlaps_any_reference
from paperlens_core.graph import ClaimGraph, build_claim_graph
from paperlens_core.memory import materialize_paper_memory
from paperlens_core.reading import (
    RELATION_CANDIDATE_KINDS,
    ObservationCard,
    ObservationLog,
    ObservationType,
    ReadingPlan,
    ReadingTask,
    ReadingTaskType,
    RelationCandidate,
    RelationCandidateLog,
    allowed_observation_types_for_task,
    build_initial_reading_plan,
    make_observation_id,
    validate_relation_candidates,
)
from paperlens_core.report import (
    GraphReportDraft,
    audit_report_draft_against_graph,
    build_report_draft_from_graph,
)
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
You are the PaperLens observation node.
Return JSON only.
""".strip()

CORE_V2_REPORT_WRITER_SYSTEM_PROMPT = """
You are the PaperLens report writer node.
Return JSON only.
""".strip()


class CoreV2WorkflowContext(Protocol):
    data_dir: Path
    config: CoreConfig
    events: EventWriter
    control: ControlState
    papers: list[PaperRecord]
    skim_cards: list[SkimCard]
    classifications: list[ClassificationDecision]

    def checkpoint(self, stage: str) -> None: ...

    def llm_enabled(self) -> bool: ...

    def new_llm_client(self) -> JsonLlmClient: ...

    def mark_paper_state(
        self,
        paper_id: str,
        stage: str,
        *,
        side_statuses: list[str] | None = None,
        error: str | None = None,
    ) -> None: ...

    def register_file_artifact(
        self,
        path: Path,
        *,
        paper_id: str | None,
        artifact_type: str,
        depends_on: list[str] | None = None,
    ) -> None: ...

    def write_agent_run(self, payload: dict[str, Any]) -> None: ...

    def record_llm_usage(self, stage: str, usage: dict[str, Any]) -> None: ...


def run_core_v2_observation_stage(workflow: CoreV2WorkflowContext) -> None:
    stage = "stage_07_normal_read"
    workflow.checkpoint(stage)
    llm_enabled = workflow.llm_enabled()
    workflow.events.stage_started(
        stage,
        "Building core v2 ObservationLog and ClaimGraph from PaperDOM evidence"
        if llm_enabled
        else "Using deterministic core v2 bootstrap artifacts",
    )
    if llm_enabled and workflow.papers:
        client = workflow.new_llm_client()
        for paper in workflow.papers:
            workflow.control.wait_if_paused()
            workflow.control.require_not_cancelled()
            workflow.events.emit(
                "agent_run_started",
                stage=stage,
                message=f"Core v2 observation read {paper.paper_id}",
                data={"paper_id": paper.paper_id, "read_mode": workflow.config.read_mode},
            )
            run_core_v2_observation_read(
                workflow,
                client=client,
                stage=stage,
                paper=paper,
            )
            workflow.mark_paper_state(paper.paper_id, stage)
            workflow.events.emit(
                "agent_run_completed",
                stage=stage,
                message=f"Core v2 observation read completed for {paper.paper_id}",
                data={"paper_id": paper.paper_id},
            )
    else:
        for paper in workflow.papers:
            workflow.mark_paper_state(paper.paper_id, stage)
    workflow.events.stage_completed(
        stage,
        "Core v2 observation stage completed",
        {"papers": len(workflow.papers), "llm_enabled": llm_enabled},
    )


def run_core_v2_observation_read(
    workflow: CoreV2WorkflowContext,
    *,
    client: JsonLlmClient,
    stage: str,
    paper: PaperRecord,
) -> None:
    try:
        result = run_core_v2_model_observation_tasks(
            client=client,
            data_dir=workflow.data_dir,
            paper=paper,
            stage=stage,
            output_language=workflow.config.output_language,
            record_usage=workflow.record_llm_usage,
            record_agent_run=workflow.write_agent_run,
        )
        for artifact_type, path in result["paths"].items():
            workflow.register_file_artifact(
                path,
                paper_id=paper.paper_id,
                artifact_type=f"core_v2_model_{artifact_type}",
                depends_on=[f"core_v2_reading_plan:{paper.paper_id}"],
            )
        workflow.events.emit(
            "core_v2_observation_read_completed",
            stage=stage,
            message=f"Core v2 observation read completed for {paper.paper_id}",
            data={
                "paper_id": paper.paper_id,
                "tasks": result["tasks"],
                "cards": result["cards"],
            },
        )
    except Exception as exc:
        workflow.write_agent_run(
            {
                "agent_run_id": f"core_v2_observe_{paper.paper_id}_failed",
                "paper_id": paper.paper_id,
                "stage": stage,
                "operation": "core_v2_observation_read",
                "provider_kind": client.config.kind,
                "model": client.config.model,
                "status": "FAIL",
                "error": str(exc),
            }
        )
        raise


def refresh_core_v2_deterministic_audits(
    workflow: CoreV2WorkflowContext,
    stage: str,
) -> list[dict[str, Any]]:
    skim_by_id = {card.paper_id: card for card in workflow.skim_cards}
    decision_by_id = {decision.paper_id: decision for decision in workflow.classifications}
    rows: list[dict[str, Any]] = []
    for paper in workflow.papers:
        try:
            result = refresh_core_v2_audit_artifacts(
                data_dir=workflow.data_dir,
                paper=paper,
                skim=skim_by_id.get(paper.paper_id),
                decision=decision_by_id.get(paper.paper_id),
            )
        except FileNotFoundError:
            if (workflow.data_dir / "core" / "v2" / paper.paper_id).exists():
                raise
            continue
        for artifact_type, path in result["paths"].items():
            workflow.register_file_artifact(
                path,
                paper_id=paper.paper_id,
                artifact_type=f"core_v2_audit_{artifact_type}",
                depends_on=[
                    f"core_v2_paper_dom:{paper.paper_id}",
                    f"core_v2_claim_graph:{paper.paper_id}",
                ],
            )
        side_statuses = []
        publish_status = str(result["publish_status"])
        if publish_status != "REVIEWED":
            side_statuses.append(f"CORE_V2_{publish_status}")
        workflow.mark_paper_state(paper.paper_id, stage, side_statuses=side_statuses)
        row = {
            "paper_id": paper.paper_id,
            "publish_status": publish_status,
            "graph_findings": result["graph_findings"],
            "report_findings": result["report_findings"],
        }
        rows.append(row)
        workflow.events.emit(
            "core_v2_audit_completed",
            stage=stage,
            message=f"Core v2 deterministic audit completed for {paper.paper_id}",
            data=row,
        )
    return rows


def run_core_v2_audit_stage(workflow: CoreV2WorkflowContext) -> None:
    stage = "stage_08_evidence_verify"
    workflow.checkpoint(stage)
    workflow.events.stage_started(stage, "Running deterministic core v2 audit suite")
    core_v2_rows = refresh_core_v2_deterministic_audits(workflow, stage)
    workflow.events.stage_completed(
        stage,
        "Core v2 deterministic audit completed",
        {"core_v2_audits": len(core_v2_rows)},
    )

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
                            "covered_outputs",
                            "extracted_numbers",
                        ],
                        "properties": {
                            "observation_type": {
                                "type": "string",
                                "enum": [item.value for item in ObservationType],
                            },
                            "statement": {"type": "string"},
                            "source_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                            "provenance": {
                                "type": "string",
                                "enum": ["explicit", "inferred"],
                            },
                            "uncertainty": {"type": ["string", "null"]},
                            "covered_outputs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "evidence_quotes": {
                                "type": "array",
                                "maxItems": 3,
                                "items": {"type": "string"},
                            },
                            "extracted_numbers": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["text"],
                                    "properties": {"text": {"type": "string"}},
                                },
                            },
                        },
                    },
                }
            },
        },
    },
}


RELATION_CANDIDATES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["artifact_type", "artifact_version", "producer", "data"],
    "properties": {
        "artifact_type": {"type": "string", "enum": ["relation_candidates"]},
        "artifact_version": {"type": "string"},
        "producer": {"type": "string"},
        "data": {
            "type": "object",
            "additionalProperties": False,
            "required": ["candidates"],
            "properties": {
                "candidates": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "source_observation_id",
                            "target_observation_id",
                            "kind",
                        ],
                        "properties": {
                            "source_observation_id": {"type": "string"},
                            "target_observation_id": {"type": "string"},
                            "kind": {
                                "type": "string",
                                "enum": sorted(RELATION_CANDIDATE_KINDS),
                            },
                            "confidence": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                            },
                        },
                    },
                }
            },
        },
    },
}


GRAPH_REPORT_DRAFT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["artifact_type", "artifact_version", "producer", "data"],
    "properties": {
        "artifact_type": {"type": "string", "enum": ["graph_report_draft"]},
        "artifact_version": {"type": "string"},
        "producer": {"type": "string"},
        "data": {
            "type": "object",
            "additionalProperties": False,
            "required": ["paper_id", "sections"],
            "properties": {
                "schema_version": {"type": "string"},
                "paper_id": {"type": "string"},
                "sections": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 9,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["section_id", "title", "paragraphs"],
                        "properties": {
                            "section_id": {"type": "string"},
                            "title": {"type": "string"},
                            "paragraphs": {
                                "type": "array",
                                "minItems": 1,
                                "maxItems": 4,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": [
                                        "paragraph_id",
                                        "markdown",
                                        "used_node_ids",
                                        "used_evidence_ids",
                                    ],
                                    "properties": {
                                        "paragraph_id": {"type": "string"},
                                        "markdown": {"type": "string"},
                                        "used_node_ids": {
                                            "type": "array",
                                            "minItems": 1,
                                            "items": {"type": "string"},
                                        },
                                        "used_evidence_ids": {
                                            "type": "array",
                                            "minItems": 1,
                                            "items": {"type": "string"},
                                        },
                                    },
                                },
                            },
                        },
                    },
                },
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
    claim_graph = build_claim_graph(paper.paper_id, list(observation_log.cards))
    derived = build_core_v2_derived_views(
        dom=dom,
        observation_log=observation_log,
        claim_graph=claim_graph,
        reading_plan=reading_plan,
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
        "paper_dom": root / "paper_dom.v2.json",
        "reading_plan": root / "reading_plan.v2.json",
        "observation_log": root / "observation_log.v2.json",
        "claim_graph": root / "claim_graph.v2.json",
        "audit_findings": root / "audit_findings.v2.json",
        "quality_metrics": root / "quality_metrics.v2.json",
        "paper_memory_view": root / "paper_memory_view.v2.json",
        "report_draft": root / "report_draft.v2.json",
        "report_audit_findings": root / "report_audit_findings.v2.json",
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
    paths["core_manifest"] = write_core_v2_manifest(root, paper.paper_id)
    return paths


def run_core_v2_model_observation_tasks(
    *,
    client: JsonLlmClient,
    data_dir: Path,
    paper: PaperRecord,
    stage: str,
    output_language: str,
    record_usage: Any,
    record_agent_run: Any,
) -> dict[str, Any]:
    output_language = "en" if output_language == "en" else "zh"
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
            max_tokens=task.max_tokens,
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
            context.record_tool_call(
                "paper_dom.read_sources",
                source_ids=task.target_source_ids,
            )
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
                        output_language=output_language,
                    ),
                    schema_name="paperlens_core_v2_observation_cards",
                    schema=OBSERVATION_CARDS_SCHEMA,
                    max_tokens=task.max_tokens,
                )
            raw_results.append(raw)
            context.record_token_usage(dict(getattr(raw, "usage", {}) or {}))
            envelope = ArtifactEnvelope.model_validate(raw.data).require_type("observation_cards")
            return envelope.model_copy(
                update={
                    "source_ids": observation_envelope_source_ids(
                        envelope,
                        allowed_source_ids=task.target_source_ids,
                    )
                }
            )

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
                "tool_calls_used": node_result.tool_calls_used,
                "used_tools": node_result.used_tools,
                "tool_source_ids": node_result.tool_source_ids,
                "tokens_used": node_result.tokens_used,
                "token_usage": node_result.token_usage,
            }
        )
        if node_result.status != NodeStatus.PASS or node_result.output is None:
            cards = fallback_observation_cards_from_task(
                dom,
                task,
                reason="; ".join(node_result.issues) or node_result.status.value,
                output_language=output_language,
            )
        else:
            try:
                cards = observation_cards_from_model_envelope(
                    node_result.output,
                    paper_id=paper.paper_id,
                    task=task,
                    allowed_source_ids=node_result.tool_source_ids.get("paper_dom.read_sources", []),
                )
            except ValueError as exc:
                cards = fallback_observation_cards_from_task(
                    dom,
                    task,
                    reason=str(exc),
                    output_language=output_language,
                )
        log = log.append_many(cards)
        completed_tasks += 1

    relation_log = _run_relation_discovery(
        client=client,
        paper=paper,
        observation_log=log,
        stage=stage,
        record_usage=record_usage,
        record_agent_run=record_agent_run,
    ) if len(log.cards) >= 2 else RelationCandidateLog(paper_id=paper.paper_id)
    if relation_log.candidates:
        request_ids.append(f"rel_{paper.paper_id}")

    claim_graph = build_claim_graph(
        paper.paper_id,
        list(log.cards),
        relation_candidates=list(relation_log.candidates) if relation_log else None,
    )
    report_draft = run_core_v2_report_writer(
        client=client,
        paper=paper,
        dom=dom,
        graph=claim_graph,
        output_language=output_language,
        stage=stage,
        record_usage=record_usage,
        record_agent_run=record_agent_run,
    )

    paths = write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=reading_plan,
        observation_log=log,
        relation_log=relation_log,
        report_draft=report_draft,
        output_language=output_language,
        producer="paperlens_core_v2_model_observer",
    )
    return {
        "paths": paths,
        "cards": len(log.cards),
        "tasks": completed_tasks,
        "relation_candidates": len(relation_log.candidates),
        "usage": total_usage,
        "request_ids": request_ids,
    }


def run_core_v2_report_writer(
    *,
    client: JsonLlmClient,
    paper: PaperRecord,
    dom: PaperDOM,
    graph: ClaimGraph,
    output_language: str,
    stage: str,
    record_usage: Any,
    record_agent_run: Any,
) -> GraphReportDraft:
    raw: Any = None
    usage: dict[str, Any] = {}
    output_language = "en" if output_language == "en" else "zh"
    try:
        with llm_call_context(
            stage=stage,
            paper_id=paper.paper_id,
            operation="core_v2_report_writer",
            schema_name="paperlens_core_v2_graph_report_draft",
        ):
            raw = client.invoke_json(
                system_prompt=CORE_V2_REPORT_WRITER_SYSTEM_PROMPT,
                user_prompt=build_graph_report_writer_prompt(
                    paper=paper,
                    dom=dom,
                    graph=graph,
                    output_language=output_language,
                ),
                schema_name="paperlens_core_v2_graph_report_draft",
                schema=GRAPH_REPORT_DRAFT_SCHEMA,
                max_tokens=12000,
            )
        usage = dict(getattr(raw, "usage", {}) or {})
        record_usage(stage, usage)
        envelope = ArtifactEnvelope.model_validate(raw.data).require_type("graph_report_draft")
        if not isinstance(envelope.data, dict):
            raise ValueError("graph_report_draft envelope data must be an object")
        draft = GraphReportDraft.model_validate(envelope.data)
        draft = normalize_model_report_draft(
            draft,
            graph=graph,
            paper_id=paper.paper_id,
            output_language=output_language,
        )
        findings = audit_report_draft_against_graph(draft, graph)
        record_agent_run(
            {
                "agent_run_id": f"core_v2_report_{paper.paper_id}",
                "paper_id": paper.paper_id,
                "stage": stage,
                "operation": "core_v2_report_writer",
                "provider_kind": client.config.kind,
                "model": client.config.model,
                "usage": usage,
                "request_id": getattr(raw, "request_id", None),
                "status": "PASS",
                "report_audit_precheck_findings": len(findings),
            }
        )
        return draft
    except Exception as exc:
        record_agent_run(
            {
                "agent_run_id": f"core_v2_report_{paper.paper_id}",
                "paper_id": paper.paper_id,
                "stage": stage,
                "operation": "core_v2_report_writer",
                "provider_kind": client.config.kind,
                "model": client.config.model,
                "usage": usage,
                "request_id": getattr(raw, "request_id", None),
                "status": "FAIL",
                "issues": [str(exc)],
            }
        )
        return build_report_draft_from_graph(graph, output_language=output_language)


def build_graph_report_writer_prompt(
    *,
    paper: PaperRecord,
    dom: PaperDOM,
    graph: ClaimGraph,
    output_language: str,
) -> str:
    language_rule = (
        "Write all reader-facing markdown in Chinese. Keep technical terms readable; include "
        "English method names only when they are paper terms."
        if output_language != "en"
        else "Write all reader-facing markdown in English."
    )
    payload = {
        "prompt_version": "paperlens_core.v2.graph_report_writer",
        "paper": {
            "paper_id": paper.paper_id,
            "title": dom.title or paper.canonical_title,
        },
        "graph_pack": report_writer_graph_pack(graph, dom),
        "output_contract": {
            "artifact_type": "graph_report_draft",
            "paper_id": paper.paper_id,
            "language_rule": language_rule,
            "reader_report_rule": (
                "The markdown is the final report text for a human reader. Do not mention "
                "ClaimGraph, PaperDOM, source_id, evidence_id, node_id, observation_id, span, "
                "or any internal identifier in markdown."
            ),
            "grounding_rule": (
                "Every paragraph must declare used_node_ids and used_evidence_ids in JSON. "
                "The markdown may explain and connect facts, but must not add factual claims "
                "beyond the labels of the declared graph nodes."
            ),
            "shape_rule": (
                "Use a readable paper-report structure: takeaway, problem, method mechanism, "
                "implementation details, evaluation setup, results, and limitations when present."
            ),
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def report_writer_graph_pack(graph: ClaimGraph, dom: PaperDOM) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for node in graph.nodes.values():
        if node.kind == "evidence":
            continue
        evidence_ids = graph.evidence_ids_for(node.node_id)
        source_ids = []
        for evidence_id in evidence_ids:
            evidence_node = graph.nodes.get(evidence_id)
            source_id = str((evidence_node.payload if evidence_node else {}).get("source_id") or "")
            if source_id and source_id not in source_ids:
                source_ids.append(source_id)
        result.append(
            {
                "node_id": node.node_id,
                "kind": node.kind,
                "label": node.label,
                "evidence_ids": evidence_ids,
                "sources": [report_source_summary(dom, source_id) for source_id in source_ids[:3]],
                "confidence": node.payload.get("confidence"),
                "uncertainty": node.payload.get("uncertainty"),
            }
        )
    return result


def report_source_summary(dom: PaperDOM, source_id: str) -> dict[str, Any]:
    for span in dom.spans:
        if span.source_id == source_id:
            return {
                "source_id": source_id,
                "kind": span.kind,
                "page_no": span.page_no,
                "text": first_sentence(span.text, limit=900),
            }
    for figure in dom.figures:
        if figure.source_id == source_id:
            return {
                "source_id": source_id,
                "kind": figure.kind,
                "page_no": figure.page_no,
                "text": first_sentence(figure.caption or "", limit=700),
            }
    for table in dom.tables:
        if table.source_id == source_id:
            return {
                "source_id": source_id,
                "kind": table.kind,
                "page_no": table.page_no,
                "text": first_sentence(table.caption or "", limit=700),
            }
    for equation in dom.equations:
        if equation.source_id == source_id:
            return {
                "source_id": source_id,
                "kind": equation.kind,
                "page_no": equation.page_no,
                "text": first_sentence(equation.latex_or_text, limit=700),
            }
    return {"source_id": source_id, "kind": "unknown", "page_no": None, "text": ""}


def normalize_model_report_draft(
    draft: GraphReportDraft,
    *,
    graph: ClaimGraph,
    paper_id: str,
    output_language: str,
) -> GraphReportDraft:
    known_fact_nodes = {
        node_id for node_id, node in graph.nodes.items() if node.kind != "evidence"
    }
    known_evidence_nodes = {
        node_id for node_id, node in graph.nodes.items() if node.kind == "evidence"
    }
    sections = []
    for section in draft.sections:
        paragraphs = []
        for paragraph in section.paragraphs:
            node_ids = [
                node_id for node_id in paragraph.used_node_ids if node_id in known_fact_nodes
            ]
            if not node_ids:
                node_ids = infer_report_paragraph_node_ids(
                    paragraph.markdown,
                    section=section,
                    graph=graph,
                )
            evidence_ids = [
                evidence_id
                for evidence_id in paragraph.used_evidence_ids
                if evidence_id in known_evidence_nodes
            ]
            for node_id in node_ids:
                for evidence_id in graph.evidence_ids_for(node_id):
                    if evidence_id in known_evidence_nodes and evidence_id not in evidence_ids:
                        evidence_ids.append(evidence_id)
            markdown = remove_visible_internal_ids(paragraph.markdown)
            if markdown and node_ids and evidence_ids:
                paragraphs.append(
                    paragraph.model_copy(
                        update={
                            "markdown": markdown,
                            "used_node_ids": node_ids,
                            "used_evidence_ids": evidence_ids,
                        }
                    )
                )
        if paragraphs:
            sections.append(section.model_copy(update={"paragraphs": paragraphs}))
    if not sections:
        return build_report_draft_from_graph(graph, output_language=output_language)
    return draft.model_copy(update={"paper_id": paper_id, "sections": sections})


def infer_report_paragraph_node_ids(
    markdown: str,
    *,
    section: Any,
    graph: ClaimGraph,
) -> list[str]:
    kind = report_section_fact_kind(section)
    if not kind:
        return []
    candidates = [node for node in graph.nodes.values() if node.kind == kind]
    matched = [
        node.node_id
        for node in candidates
        if text_overlaps_any_reference(markdown, [node.label])
    ]
    if matched:
        return matched[:4]
    return [node.node_id for node in candidates[:1]]


def report_section_fact_kind(section: Any) -> str | None:
    section_id = str(getattr(section, "section_id", "") or "").lower()
    title = str(getattr(section, "title", "") or "").lower()
    direct = {
        "problem": "problem",
        "motivation": "problem",
        "claim": "claim",
        "claims": "claim",
        "takeaway": "claim",
        "summary": "claim",
        "method": "mechanism",
        "mechanism": "mechanism",
        "implementation": "implementation",
        "evaluation": "evaluation",
        "experiment": "evaluation",
        "results": "result",
        "result": "result",
        "limitations": "limitation",
        "limitation": "limitation",
        "concept": "concept",
        "concept_bridge": "concept",
    }
    if section_id in direct:
        return direct[section_id]
    title_hints = [
        ("问题", "problem"),
        ("动机", "problem"),
        ("主张", "claim"),
        ("贡献", "claim"),
        ("方法", "mechanism"),
        ("机制", "mechanism"),
        ("实现", "implementation"),
        ("实验", "evaluation"),
        ("评估", "evaluation"),
        ("结果", "result"),
        ("结论", "result"),
        ("限制", "limitation"),
        ("边界", "limitation"),
        ("概念", "concept"),
    ]
    for hint, kind in title_hints:
        if hint in title:
            return kind
    return None


def remove_visible_internal_ids(text: str) -> str:
    cleaned_lines = []
    for line in str(text or "").splitlines():
        if any(
            marker in line
            for marker in [
                "ClaimGraph",
                "PaperDOM",
                "source_id",
                "evidence_id",
                "node_id",
                "observation_id",
            ]
        ) or re.search(
            r"\b(?:problem|claim|mechanism|implementation|evaluation|result|limitation|concept):obs_",
            line,
        ) or "evidence:" in line or "span:" in line:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def _run_relation_discovery(
    *,
    client: JsonLlmClient,
    paper: PaperRecord,
    observation_log: ObservationLog,
    stage: str,
    record_usage: Any,
    record_agent_run: Any,
) -> RelationCandidateLog:
    if not observation_log.cards:
        return RelationCandidateLog(paper_id=paper.paper_id)
    raw: Any = None
    usage: dict[str, Any] = {}
    try:
        with llm_call_context(
            stage=stage,
            paper_id=paper.paper_id,
            operation="core_v2_relation_discovery",
            schema_name="paperlens_relation_candidates",
        ):
            raw = client.invoke_json(
                system_prompt=CORE_V2_OBSERVER_SYSTEM_PROMPT,
                user_prompt=_build_relation_discovery_prompt(paper, observation_log),
                schema_name="paperlens_relation_candidates",
                schema=RELATION_CANDIDATES_SCHEMA,
                max_tokens=8000,
            )
        usage = dict(getattr(raw, "usage", {}) or {})
        record_usage(stage, usage)
        envelope = ArtifactEnvelope.model_validate(raw.data).require_type("relation_candidates")
        payload = envelope.data if isinstance(envelope.data, dict) else {}
        raw_candidates = (
            payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
        )
        observation_ids = {card.observation_id for card in observation_log.cards}
        candidates: list[RelationCandidate] = []
        for item in raw_candidates:
            if not isinstance(item, dict):
                continue
            candidates.append(
                RelationCandidate(
                    source_observation_id=str(item.get("source_observation_id") or "").strip(),
                    target_observation_id=str(item.get("target_observation_id") or "").strip(),
                    kind=str(item.get("kind") or "").strip(),
                    confidence=str(item.get("confidence") or "medium"),
                )
            )
        valid = validate_relation_candidates(candidates, observation_ids)
    except Exception as exc:
        record_agent_run(
            {
                "agent_run_id": f"core_v2_rel_{paper.paper_id}",
                "paper_id": paper.paper_id,
                "stage": stage,
                "operation": "core_v2_relation_discovery",
                "provider_kind": client.config.kind,
                "model": client.config.model,
                "usage": usage,
                "request_id": getattr(raw, "request_id", None),
                "status": "FAIL",
                "issues": [str(exc)],
            }
        )
        return RelationCandidateLog(paper_id=paper.paper_id)
    record_agent_run(
        {
            "agent_run_id": f"core_v2_rel_{paper.paper_id}",
            "paper_id": paper.paper_id,
            "stage": stage,
            "operation": "core_v2_relation_discovery",
            "provider_kind": client.config.kind,
            "model": client.config.model,
            "usage": usage,
            "request_id": getattr(raw, "request_id", None),
            "status": "PASS",
        }
    )
    log = RelationCandidateLog(paper_id=paper.paper_id)
    for candidate in valid:
        log = log.append(candidate)
    return log


def _build_relation_discovery_prompt(
    paper: PaperRecord,
    observation_log: ObservationLog,
) -> str:
    import json as _json
    card_summaries = [
        {
            "observation_id": card.observation_id,
            "type": card.observation_type.value,
            "statement": card.statement,
            "source_ids": card.source_ids,
        }
        for card in observation_log.cards
    ]
    return _json.dumps(
        {
            "paper_id": paper.paper_id,
            "title": paper.canonical_title,
            "observations": card_summaries,
            "task": (
                "Review all observations above. Propose semantic relations between them "
                "where you see clear logical connections. Each candidate must reference "
                "existing observation_ids. Valid relation kinds: "
                + ", ".join(sorted(RELATION_CANDIDATE_KINDS))
                + ". Only propose relations you are confident about."
            ),
        },
        ensure_ascii=False,
    )


def write_core_v2_from_observation_log(
    *,
    data_dir: Path,
    paper: PaperRecord,
    dom: PaperDOM,
    reading_plan: ReadingPlan,
    observation_log: ObservationLog,
    relation_log: RelationCandidateLog | None = None,
    report_draft: GraphReportDraft | None = None,
    output_language: str = "zh",
    producer: str,
) -> dict[str, Path]:
    validate_core_v2_write_inputs(
        paper=paper,
        dom=dom,
        reading_plan=reading_plan,
        observation_log=observation_log,
    )
    candidates = list(relation_log.candidates) if relation_log else None
    claim_graph = build_claim_graph(
        paper.paper_id, list(observation_log.cards), relation_candidates=candidates
    )
    derived = build_core_v2_derived_views(
        dom=dom,
        observation_log=observation_log,
        claim_graph=claim_graph,
        reading_plan=reading_plan,
        relation_log=relation_log if relation_log and relation_log.candidates else None,
        report_draft=report_draft,
        output_language=output_language,
        metadata={
            "title": paper.canonical_title,
            "observer_schema_version": CORE_V2_MODEL_OBSERVER_VERSION,
        },
    )
    root = data_dir / "core" / "v2" / paper.paper_id
    write_core_v2_envelope(
        root / "paper_dom.v2.json",
        "paper_dom",
        paper.paper_id,
        dom.model_dump(),
        producer="paperlens_core_v2_input",
    )
    write_core_v2_envelope(
        root / "reading_plan.v2.json",
        "reading_plan",
        paper.paper_id,
        reading_plan.model_dump(),
        producer="paperlens_core_v2_input",
    )
    paths = {
        "observation_log": root / "observation_log.v2.json",
        "claim_graph": root / "claim_graph.v2.json",
        "relation_candidate_log": root / "relation_candidate_log.v2.json",
        "audit_findings": root / "audit_findings.v2.json",
        "quality_metrics": root / "quality_metrics.v2.json",
        "paper_memory_view": root / "paper_memory_view.v2.json",
        "report_draft": root / "report_draft.v2.json",
        "report_audit_findings": root / "report_audit_findings.v2.json",
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
    if relation_log is not None:
        write_core_v2_envelope(
            paths["relation_candidate_log"],
            "relation_candidate_log",
            paper.paper_id,
            relation_log.model_dump(),
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
    paths["core_manifest"] = write_core_v2_manifest(root, paper.paper_id, producer=producer)
    return paths


def validate_core_v2_write_inputs(
    *,
    paper: PaperRecord,
    dom: PaperDOM,
    reading_plan: ReadingPlan,
    observation_log: ObservationLog,
) -> None:
    if dom.paper_id != paper.paper_id:
        raise ValueError(f"Core v2 dom paper_id mismatch: {dom.paper_id} != {paper.paper_id}")
    if reading_plan.paper_id != paper.paper_id:
        raise ValueError(
            f"Core v2 reading_plan paper_id mismatch: {reading_plan.paper_id} != {paper.paper_id}"
        )
    if observation_log.paper_id != paper.paper_id:
        raise ValueError(
            f"Core v2 observation_log paper_id mismatch: "
            f"{observation_log.paper_id} != {paper.paper_id}"
        )
    invalid_plan_sources = sorted(
        {
            source_id
            for task in reading_plan.tasks
            for source_id in task.target_source_ids
            if not dom.source_exists(source_id)
        }
    )
    if invalid_plan_sources:
        raise ValueError(
            "Core v2 reading_plan references source_ids missing from PaperDOM: "
            + ", ".join(invalid_plan_sources[:8])
        )
    invalid_observation_sources = sorted(
        {
            source_id
            for card in observation_log.cards
            for source_id in card.source_ids
            if not dom.source_exists(source_id)
        }
    )
    if invalid_observation_sources:
        raise ValueError(
            "Core v2 observation_log references source_ids missing from PaperDOM: "
            + ", ".join(invalid_observation_sources[:8])
        )


def refresh_core_v2_audit_artifacts(
    *,
    data_dir: Path,
    paper: PaperRecord,
    skim: SkimCard | None = None,
    decision: ClassificationDecision | None = None,
    producer: str = "paperlens_core_v2_audit_suite",
) -> dict[str, Any]:
    dom, reading_plan = load_core_v2_dom_and_plan(data_dir, paper.paper_id)
    observation_log = load_core_v2_observation_log(data_dir, paper.paper_id)
    _, claim_graph = load_core_v2_dom_and_graph(data_dir, paper.paper_id)
    relation_log = _load_relation_candidate_log(data_dir, paper.paper_id)
    report_draft = load_core_v2_report_draft(data_dir, paper.paper_id)
    derived = build_core_v2_derived_views(
        dom=dom,
        observation_log=observation_log,
        claim_graph=claim_graph,
        reading_plan=reading_plan,
        relation_log=relation_log if relation_log and relation_log.candidates else None,
        report_draft=report_draft,
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
        "audit_findings": root / "audit_findings.v2.json",
        "quality_metrics": root / "quality_metrics.v2.json",
        "paper_memory_view": root / "paper_memory_view.v2.json",
        "report_draft": root / "report_draft.v2.json",
        "report_audit_findings": root / "report_audit_findings.v2.json",
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
    paths["core_manifest"] = write_core_v2_manifest(root, paper.paper_id, producer=producer)
    return {
        "paths": paths,
        "graph_findings": len(derived["audit_findings"]),
        "report_findings": len(derived["report_audit_findings"]),
        "publish_status": derived["quality_metrics"].publish_status,
    }


def build_core_v2_derived_views(
    *,
    dom: PaperDOM,
    observation_log: ObservationLog,
    claim_graph: ClaimGraph,
    reading_plan: ReadingPlan | None = None,
    relation_log: RelationCandidateLog | None = None,
    report_draft: GraphReportDraft | None = None,
    output_language: str = "zh",
    metadata: dict[str, Any],
) -> dict[str, Any]:
    observation_findings = (
        audit_observation_log(observation_log, dom, reading_plan)
        if reading_plan is not None
        else []
    )
    relation_candidates = list(relation_log.candidates) if relation_log else None
    relation_findings = (
        audit_relation_candidates(
            relation_log,
            {card.observation_id for card in observation_log.cards},
        )
        if relation_log
        else []
    )
    audit_findings = [
        *observation_findings,
        *audit_claim_graph_from_observation_log(
            claim_graph, observation_log, relation_candidates=relation_candidates
        ),
        *audit_claim_graph(claim_graph, dom),
        *audit_reading_required_outputs(claim_graph, reading_plan),
        *relation_findings,
    ]
    if report_draft is None:
        report_draft = build_report_draft_from_graph(
            claim_graph,
            output_language=output_language,
        )
    else:
        report_draft = normalize_model_report_draft(
            report_draft,
            graph=claim_graph,
            paper_id=claim_graph.paper_id,
            output_language=output_language,
        )
    report_audit_findings = audit_report_draft_against_graph(report_draft, claim_graph)
    all_findings = [*audit_findings, *report_audit_findings]
    quality_metrics = compute_core_quality_metrics(
        dom=dom,
        graph=claim_graph,
        findings=all_findings,
        reading_plan=reading_plan,
    )
    memory_view = materialize_paper_memory(
        claim_graph,
        dom=dom,
        metadata=metadata,
        unresolved_audit_findings=[finding.finding_id for finding in all_findings],
        audit_findings=all_findings,
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


def fallback_observation_cards_from_task(
    dom: PaperDOM,
    task: ReadingTask,
    *,
    reason: str,
    output_language: str = "zh",
) -> list[ObservationCard]:
    spans_by_id = {span.source_id: span for span in dom.spans}
    card = bootstrap_observation_for_task(
        dom,
        spans_by_id,
        task,
        output_language=output_language,
    )
    if card is None:
        raise ValueError(f"Observation task {task.task_id} returned no valid observation cards")
    fallback_note = (
        f"Model observation output was unusable; deterministic fallback used: {reason}"
        if output_language == "en"
        else f"模型观察输出不可用，已使用确定性兜底：{reason}"
    )
    uncertainty = f"{card.uncertainty} {fallback_note}".strip() if card.uncertainty else fallback_note
    return [card.model_copy(update={"uncertainty": uncertainty})]


def bootstrap_observation_for_task(
    dom: PaperDOM,
    spans_by_id: dict[str, PaperSpan],
    task: ReadingTask,
    *,
    output_language: str = "zh",
) -> ObservationCard | None:
    source_id = next((item for item in task.target_source_ids if item in spans_by_id), None)
    if not source_id:
        return None
    span = spans_by_id[source_id]
    statement = observation_statement(task.task_type, span.text, output_language=output_language)
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
            if output_language == "en"
            else "确定性启动观察；在作为已复核知识使用前，需要由任务化模型阅读替换。"
        ),
        covered_outputs=bootstrap_covered_outputs(observation_type, task),
        evidence_quotes=[first_sentence(span.text, limit=220)],
        extracted_numbers=extract_numbers(statement),
    )


def bootstrap_covered_outputs(
    observation_type: ObservationType,
    task: ReadingTask,
) -> list[str]:
    if task.task_type == ReadingTaskType.ORIENTATION and observation_type == ObservationType.PROBLEM:
        return list(task.required_outputs)
    return [observation_type.value] if observation_type.value in task.required_outputs else []


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


def observation_statement(
    task_type: ReadingTaskType,
    text: str,
    *,
    output_language: str = "zh",
) -> str:
    sentence = first_sentence(text)
    if not sentence:
        return ""
    return sentence


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
        artifact_version="v2",
        data=data,
        producer=producer,
        source_ids=source_ids or [],
        metadata={"paper_id": paper_id, "schema_version": CORE_V2_SCHEMA_VERSION},
    )


def load_core_v2_dom_and_plan(data_dir: Path, paper_id: str) -> tuple[PaperDOM, ReadingPlan]:
    root = data_dir / "core" / "v2" / paper_id
    dom_envelope = read_typed_artifact(root / "paper_dom.v2.json", expected_type="paper_dom")
    plan_envelope = read_typed_artifact(root / "reading_plan.v2.json", expected_type="reading_plan")
    if not isinstance(dom_envelope.data, dict) or not isinstance(plan_envelope.data, dict):
        raise ValueError(f"Core v2 paper_dom/reading_plan artifacts are invalid for {paper_id}")
    return PaperDOM.model_validate(dom_envelope.data), ReadingPlan.model_validate(
        plan_envelope.data
    )


def _load_relation_candidate_log(
    data_dir: Path, paper_id: str
) -> RelationCandidateLog | None:
    root = data_dir / "core" / "v2" / paper_id
    path = root / "relation_candidate_log.v2.json"
    if not path.exists():
        return None
    try:
        envelope = read_typed_artifact(path, expected_type="relation_candidate_log")
    except (FileNotFoundError, ValueError):
        return None
    if not isinstance(envelope.data, dict):
        return None
    try:
        return RelationCandidateLog.model_validate(envelope.data)
    except Exception:
        return None


def load_core_v2_observation_log(data_dir: Path, paper_id: str) -> ObservationLog:
    root = data_dir / "core" / "v2" / paper_id
    log_envelope = read_typed_artifact(
        root / "observation_log.v2.json",
        expected_type="observation_log",
    )
    if not isinstance(log_envelope.data, dict):
        raise ValueError(f"Core v2 observation_log artifact is invalid for {paper_id}")
    return ObservationLog.model_validate(log_envelope.data)


def load_core_v2_report_draft(data_dir: Path, paper_id: str) -> GraphReportDraft | None:
    root = data_dir / "core" / "v2" / paper_id
    try:
        draft_envelope = read_typed_artifact(
            root / "report_draft.v2.json",
            expected_type="graph_report_draft",
        )
    except (FileNotFoundError, ValueError):
        return None
    if not isinstance(draft_envelope.data, dict):
        return None
    try:
        return GraphReportDraft.model_validate(draft_envelope.data)
    except Exception:
        return None


def load_core_v2_dom_and_graph(data_dir: Path, paper_id: str) -> tuple[PaperDOM, ClaimGraph]:
    root = data_dir / "core" / "v2" / paper_id
    dom_envelope = read_typed_artifact(root / "paper_dom.v2.json", expected_type="paper_dom")
    graph_envelope = read_typed_artifact(root / "claim_graph.v2.json", expected_type="claim_graph")
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
    output_language: str = "zh",
) -> str:
    output_language = "en" if output_language == "en" else "zh"
    language_rule = (
        "Write statement, uncertainty, and any explanatory text in Chinese. Keep paper terms "
        "or method names in English only when they are original technical terms."
        if output_language != "en"
        else "Write statement, uncertainty, and any explanatory text in English."
    )
    payload = {
        "prompt_version": CORE_V2_MODEL_OBSERVER_VERSION,
        "output_language": output_language,
        "paper": {
            "paper_id": paper.paper_id,
            "title": paper.canonical_title,
        },
        "task_spec": {
            "task_id": task.task_id,
            "task_type": task.task_type.value,
            "required_outputs": task.required_outputs,
            "allowed_observation_types": task_allowed_observation_types(task),
            "task_instruction": observation_task_instruction(
                task.task_type,
                output_language=output_language,
            ),
            "evidence_policy": task.evidence_policy,
            "max_model_calls": task.max_model_calls,
            "max_tokens": task.max_tokens,
        },
        "evidence_pack": source_pack(dom, task.target_source_ids),
        "output_contract": {
            "artifact_type": "observation_cards",
            "allowed_source_ids": task.target_source_ids,
            "language_rule": language_rule,
            "rule": (
                "Return an ArtifactEnvelope with data.cards. Each card must cite source_ids from "
                "output_contract.allowed_source_ids by exact string copy and declare "
                "covered_outputs from task_spec.required_outputs. Across cards, cover every "
                "required_output supported by the evidence. Do not cite page numbers as evidence. "
                "Do not write memory, audit verdicts, or report prose. Statements must be "
                "reader-grade paper facts, not phrases like 'evidence shows', 'I saw', or "
                "task labels. For each card, evidence_quotes must contain 1-3 short exact "
                "substrings copied from the cited sources, preserving original technical terms, "
                "dataset names, metric names, and numeric values when present."
            ),
        },
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def observation_task_instruction(
    task_type: ReadingTaskType,
    *,
    output_language: str,
) -> str:
    zh = output_language != "en"
    if task_type == ReadingTaskType.ORIENTATION:
        return (
            "用中文回答论文要解决什么问题、为什么难、适用的任务/场景范围；可以用一张卡同时覆盖 problem、motivation、scope。"
            if zh
            else "State the problem, why it is difficult, and the task/scope; one card may cover problem, motivation, and scope."
        )
    if task_type == ReadingTaskType.CLAIM_INVENTORY:
        return (
            "抽取论文自己的核心主张/贡献，避免把实现细节或结果反复当成主张。"
            if zh
            else "Extract the paper's own main claims or contributions; do not repeat implementation details as claims."
        )
    if task_type == ReadingTaskType.METHOD_MECHANISM:
        return (
            "解释方法机制链路：关键模块是什么、为什么需要它、它如何缓解论文提出的问题。"
            if zh
            else "Explain the mechanism chain: key modules, why they exist, and how they address the problem."
        )
    if task_type == ReadingTaskType.IMPLEMENTATION_PATH:
        return (
            "抽取可执行实现路径：输入预处理、网络/模块、训练目标、推理流程、重要超参数或尺寸。"
            if zh
            else "Extract implementation facts: preprocessing, networks/modules, objectives, inference flow, and important sizes or hyperparameters."
        )
    if task_type == ReadingTaskType.EVALUATION_SETUP:
        return (
            "只写实验设置事实：数据集、源域/目标域、指标、基线、训练/测试协议；不要写泛泛方法摘要。"
            if zh
            else "Only write evaluation setup facts: datasets, source/target domains, metrics, baselines, and protocol; do not summarize the method."
        )
    if task_type == ReadingTaskType.RESULT_EXTRACTION:
        return (
            "只写实验结果、表格结论、消融结论或关键数值；不要把方法动机、模块列表、摘要内容当成结果。"
            if zh
            else "Only write experimental results, table conclusions, ablations, or key numbers; do not restate method motivation or module lists."
        )
    if task_type == ReadingTaskType.LIMITATIONS:
        return (
            "优先抽取论文显式承认的限制；如果是基于公式/设置推断的边界，provenance 必须为 inferred，并在 statement 中标明是推断。"
            if zh
            else "Prefer explicit limitations; if a boundary is inferred from equations or setup, set provenance to inferred and say it is inferred."
        )
    if task_type == ReadingTaskType.CONCEPT_BRIDGE:
        return (
            "解释读懂论文必须掌握的概念关系，只保留能帮助理解方法或评估的概念。"
            if zh
            else "Explain only concepts needed to understand the method or evaluation."
        )
    if task_type == ReadingTaskType.RELATED_POSITIONING:
        return (
            "说明论文相对已有工作的定位；如果证据中没有限制项，不要编造 limitation。"
            if zh
            else "State positioning against prior work; do not fabricate limitations if evidence does not support them."
        )
    return (
        "抽取复现相关事实；如果证据不足，明确哪些实现细节缺失。"
        if zh
        else "Extract reproducibility facts; state missing implementation details when evidence is insufficient."
    )


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
            "source_span_id": equation.source_span_id,
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
    allowed_source_ids: list[str] | set[str],
) -> list[ObservationCard]:
    payload = envelope.data if isinstance(envelope.data, dict) else {}
    cards = payload.get("cards") if isinstance(payload.get("cards"), list) else []
    allowed_source_id_list = list(dict.fromkeys(str(item) for item in allowed_source_ids if item))
    allowed_source_id_set = set(allowed_source_id_list)
    result = []
    for item in cards:
        if not isinstance(item, dict):
            continue
        source_ids = clean_model_source_ids(item.get("source_ids"), allowed_source_id_set)
        source_fallback = False
        if not source_ids:
            source_ids = allowed_source_id_list
            source_fallback = True
        if not source_ids:
            raise ValueError(f"Observation card for {task.task_id} did not cite valid source_ids")
        statement = clean_model_statement(item.get("statement"))
        observation_type = str(item.get("observation_type") or "").strip()
        if observation_type not in {kind.value for kind in ObservationType}:
            observation_type = infer_observation_type_for_task(item, task)
        if not statement or observation_type not in {kind.value for kind in ObservationType}:
            continue
        allowed_types = task_allowed_observation_types(task)
        if observation_type not in allowed_types:
            if len(allowed_types) == 1:
                observation_type = allowed_types[0]
            else:
                raise ValueError(
                    f"Observation card for {task.task_id} returned disallowed observation_type="
                    f"{observation_type}; allowed={allowed_types}"
                )
        covered_outputs = clean_model_covered_outputs(item.get("covered_outputs"), task)
        inferred_outputs = infer_covered_outputs_from_type(observation_type, task)
        if not any(output in task.required_outputs for output in covered_outputs):
            covered_outputs = inferred_outputs or covered_outputs
        provenance = clean_model_provenance(item.get("provenance"), task.task_id)
        uncertainty = none_if_blank(item.get("uncertainty"))
        if source_fallback:
            fallback_note = (
                "Runtime assigned source_ids from the task evidence pack because the model "
                "omitted source_ids."
            )
            uncertainty = f"{uncertainty} {fallback_note}".strip() if uncertainty else fallback_note
        if not covered_outputs:
            output_note = (
                "Model did not declare covered_outputs; deterministic audit will mark missing "
                "ReadingTask coverage."
            )
            uncertainty = f"{uncertainty} {output_note}".strip() if uncertainty else output_note
        elif inferred_outputs and item.get("covered_outputs") in (None, []):
            output_note = (
                "Runtime inferred covered_outputs from observation_type and ReadingTask spec."
            )
            uncertainty = f"{uncertainty} {output_note}".strip() if uncertainty else output_note
        evidence_quotes = clean_model_evidence_quotes(item.get("evidence_quotes"))
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
                provenance=provenance,
                uncertainty=uncertainty,
                covered_outputs=covered_outputs,
                evidence_quotes=evidence_quotes,
                extracted_numbers=[
                    number
                    for number in list_payload(item.get("extracted_numbers"))
                    if isinstance(number, dict)
                ][:8],
            )
        )
    if not result:
        raise ValueError(f"Observation task {task.task_id} returned no valid observation cards")
    return result



def clean_model_covered_outputs(value: Any, task: ReadingTask) -> list[str]:
    if not isinstance(value, list):
        return []
    allowed = set(task.required_outputs)
    result = []
    for item in value:
        output = str(item or "").strip()
        if not output:
            continue
        if output in allowed and output not in result:
            result.append(output)
    return result


def infer_covered_outputs_from_type(observation_type: str, task: ReadingTask) -> list[str]:
    if task.task_type == ReadingTaskType.ORIENTATION and observation_type == "problem":
        return list(task.required_outputs)
    return [observation_type] if observation_type in task.required_outputs else []


def clean_model_evidence_quotes(value: Any) -> list[str]:
    result = []
    for item in list_payload(value):
        text = " ".join(str(item or "").split()).strip()
        if text and text not in result:
            result.append(text[:260])
        if len(result) >= 3:
            break
    return result


def clean_model_statement(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(
        r"^(问题定位|核心主张|方法机制|实现路径|评估设置|实验结果|限制边界|概念桥接|相关工作定位|可复现性)证据：\s*",
        "",
        text,
    ).strip()


def missing_required_outputs(task: ReadingTask, cards: list[ObservationCard]) -> list[str]:
    covered = {
        output
        for card in cards
        for output in card.covered_outputs
    }
    return [output for output in task.required_outputs if output not in covered]


def infer_observation_type_for_task(item: dict[str, Any], task: ReadingTask) -> str:
    allowed_types = task_allowed_observation_types(task)
    covered_outputs = clean_model_covered_outputs(item.get("covered_outputs"), task)
    for output in covered_outputs:
        if output in allowed_types:
            return output
    if len(allowed_types) == 1:
        return allowed_types[0]
    return ""


def observation_envelope_source_ids(
    envelope: ArtifactEnvelope,
    *,
    allowed_source_ids: list[str] | tuple[str, ...] | set[str] | None = None,
) -> list[str]:
    allowed = set(allowed_source_ids or [])
    data = envelope.data if isinstance(envelope.data, dict) else {}
    cards = data.get("cards") if isinstance(data.get("cards"), list) else []
    result: list[str] = []
    for item in cards:
        if not isinstance(item, dict):
            continue
        source_ids = item.get("source_ids") if isinstance(item.get("source_ids"), list) else []
        for source_id in source_ids:
            text = str(source_id or "").strip()
            if allowed and text not in allowed:
                continue
            if text and text not in result:
                result.append(text)
    return result


def clean_model_provenance(value: Any, task_id: str) -> str:
    provenance = str(value or "explicit").strip()
    if provenance not in {"explicit", "inferred"}:
        raise ValueError(
            f"Observation card for {task_id} returned disallowed provenance={provenance}; "
            "allowed=['explicit', 'inferred']"
        )
    return provenance


def task_allowed_observation_types(task: ReadingTask) -> list[str]:
    return task.allowed_observation_types or allowed_observation_types_for_task(task.task_type)


def clean_model_source_ids(value: Any, allowed_source_ids: set[str]) -> list[str]:
    result = []
    for item in list_payload(value):
        source_id = str(item or "").strip()
        if not source_id:
            continue
        if source_id not in allowed_source_ids:
            continue
        if source_id not in result:
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
