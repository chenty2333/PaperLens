from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from paperlens_core.audit import (
    PublishStatus,
    audit_claim_graph,
    audit_reading_required_outputs,
    compute_core_quality_metrics,
    publish_status_from_findings,
)
from paperlens_core.agent_loop import PaperToolRegistry
from paperlens_core.dom import PaperDOM, PaperSection, PaperSpan, build_paper_dom_from_layout
from paperlens_core.graph import GraphEdge, GraphNode, graph_from_observations
from paperlens_core.library import (
    doctor_library,
    rebuild_library_from_output,
    read_library_records,
    search_library,
    write_paperlens_library,
)
from paperlens_core.memory import materialize_paper_memory
from paperlens_core.qa import (
    answer_question,
    build_ask_prompt,
    core_v2_context_priority,
    ground_qa_answer_in_core_v2_context,
    load_core_v2_qa_context,
    offline_qa_answer,
    qa_memory_context,
)
from paperlens_core.quality_snapshot import write_core_quality_snapshot
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
from paperlens_core.report import (
    GraphReportDraft,
    ReportParagraph,
    ReportSection,
    audit_report_draft_against_graph,
    build_report_draft_from_graph,
    build_report_memory_context,
    compact_paper_memory_for_report,
    report_focus_pages,
    report_focus_queries,
    render_graph_report_markdown,
    write_core_graph_report_view,
)
from paperlens_core.runtime import (
    ArtifactEnvelope,
    NodeSpec,
    NodeStatus,
    PaperLensRuntime,
    run_finite_node,
    write_typed_artifact,
)
from paperlens_core.config import CoreConfig
from paperlens_core.control import ControlState
from paperlens_core.core_manifest import build_core_v2_manifest, write_core_v2_manifest
from paperlens_core.events import EventWriter
from paperlens_core.schemas import (
    ClassificationDecision,
    PageArtifact,
    PaperCard,
    PaperRecord,
    SkimCard,
)
from paperlens_core.workflow.agent import (
    PaperLensWorkflow,
    paper_report_filename,
    write_final_report_bundle,
)
from paperlens_core.workflow.core_v2 import (
    OBSERVATION_CARDS_SCHEMA,
    load_core_v2_dom_and_plan,
    observation_cards_from_model_envelope,
    refresh_core_v2_audit_artifacts,
    run_core_v2_model_observation_tasks,
    source_pack,
    write_core_v2_artifacts,
    write_core_v2_from_observation_log,
)


def sample_dom():
    return build_paper_dom_from_layout(
        paper_id="p_test",
        title="Test Paper",
        layout={
            "pages": [
                {
                    "page_no": 1,
                    "text": "Abstract\n\nWe propose a block table method for faster serving.",
                    "section_candidates": [{"title": "Abstract", "level": 1}],
                    "figures": [],
                    "tables": [],
                },
                {
                    "page_no": 2,
                    "text": "Evaluation\n\nThe method improves latency by 27% on Dataset-A.",
                    "section_candidates": [{"title": "Evaluation", "level": 1}],
                    "figures": [{"caption": "Latency comparison, 27% lower."}],
                    "tables": [{"caption": "Dataset-A latency results"}],
                },
            ]
        },
    )


def reading_plan_subset(dom: PaperDOM, *task_types: ReadingTaskType) -> ReadingPlan:
    full_plan = build_initial_reading_plan(dom)
    selected = [task for task in full_plan.tasks if task.task_type in set(task_types)]
    assert selected
    return ReadingPlan(paper_id=dom.paper_id, tasks=selected)


def reading_task(plan: ReadingPlan, task_type: ReadingTaskType) -> ReadingTask:
    return next(task for task in plan.tasks if task.task_type == task_type)


def test_paper_dom_assigns_stable_source_ids():
    dom = sample_dom()

    assert dom.schema_version == "paper_dom.v1"
    node_count = (
        len(dom.sections) + len(dom.spans) + len(dom.figures) + len(dom.tables) + len(dom.equations)
    )
    assert len(dom.source_ids()) == node_count
    assert {span.page_no for span in dom.spans} == {1, 2}
    assert all(span.source_id.startswith("span:p_test:") for span in dom.spans)
    assert all(section.span_ids for section in dom.sections)
    assert dom.source_exists(dom.spans[0].source_id)
    assert dom.figures[0].source_id.startswith("figure:p_test:")
    assert dom.tables[0].source_id.startswith("table:p_test:")


def test_paper_dom_assigns_unique_equation_and_missing_page_source_ids():
    dom = build_paper_dom_from_layout(
        paper_id="p_math",
        title="Math Paper",
        layout={
            "pages": [
                {
                    "text": "Method\n\nFirst equation $x_i = y_i$.\n\nSecond equation $z_i = x_i$.",
                    "section_candidates": [{"title": "Method", "level": 1}],
                    "figures": [{"caption": "First visual"}],
                },
                {
                    "text": "Appendix\n\nThird equation $a_i = b_i$.",
                    "section_candidates": [{"title": "Appendix", "level": 1}],
                    "figures": [{"caption": "Second visual"}],
                },
            ]
        },
    )

    assert len(dom.equations) == 3
    assert len({item.source_id for item in dom.equations}) == 3
    assert len({item.source_id for item in dom.spans}) == len(dom.spans)
    assert len({item.source_id for item in dom.figures}) == 2
    assert all("p0" not in item.source_id for item in [*dom.spans, *dom.figures, *dom.equations])
    assert "page_missing_number" in dom.parse_warnings


def test_paper_dom_rejects_duplicate_source_ids():
    with pytest.raises(ValueError, match="source_id values must be unique"):
        PaperDOM(
            paper_id="p_test",
            sections=[
                PaperSection(
                    source_id="section:p_test:p1:1",
                    paper_id="p_test",
                    title="Abstract",
                    span_ids=["span:p_test:p1:1"],
                )
            ],
            spans=[
                PaperSpan(
                    source_id="span:p_test:p1:1",
                    paper_id="p_test",
                    section_id="section:p_test:p1:1",
                    text="First span.",
                ),
                PaperSpan(
                    source_id="span:p_test:p1:1",
                    paper_id="p_test",
                    section_id="section:p_test:p1:1",
                    text="Duplicate span.",
                ),
            ],
        )


def test_paper_dom_rejects_cross_paper_source_addresses():
    with pytest.raises(ValueError, match="source_id does not match kind/paper_id"):
        PaperDOM(
            paper_id="p_test",
            sections=[
                PaperSection(
                    source_id="section:p_test:p1:1",
                    paper_id="p_test",
                    title="Abstract",
                )
            ],
            spans=[
                PaperSpan(
                    source_id="span:p_other:p1:1",
                    paper_id="p_test",
                    section_id="section:p_test:p1:1",
                    text="Wrong source address.",
                )
            ],
        )
    with pytest.raises(ValueError, match="node paper_id mismatch"):
        PaperDOM(
            paper_id="p_test",
            sections=[
                PaperSection(
                    source_id="section:p_test:p1:1",
                    paper_id="p_test",
                    title="Abstract",
                )
            ],
            spans=[
                PaperSpan(
                    source_id="span:p_test:p1:1",
                    paper_id="p_other",
                    section_id="section:p_test:p1:1",
                    text="Wrong node paper id.",
                )
            ],
        )


def test_paper_dom_rejects_dangling_internal_source_links():
    with pytest.raises(ValueError, match="section span_ids reference missing spans"):
        PaperDOM(
            paper_id="p_test",
            sections=[
                PaperSection(
                    source_id="section:p_test:p1:1",
                    paper_id="p_test",
                    title="Abstract",
                    span_ids=["span:p_test:p1:missing"],
                )
            ],
        )
    with pytest.raises(ValueError, match="nodes reference missing sections"):
        PaperDOM(
            paper_id="p_test",
            spans=[
                PaperSpan(
                    source_id="span:p_test:p1:1",
                    paper_id="p_test",
                    section_id="section:p_test:p1:missing",
                    text="Dangling section.",
                )
            ],
        )


def test_reading_plan_is_structured_and_source_bound():
    dom = sample_dom()
    plan = build_initial_reading_plan(dom)

    task_types = {task.task_type for task in plan.tasks}
    assert ReadingTaskType.ORIENTATION in task_types
    assert ReadingTaskType.EVALUATION_SETUP in task_types
    assert ReadingTaskType.RESULT_EXTRACTION in task_types
    assert all(task.max_model_calls == 1 for task in plan.tasks)
    assert all(task.max_tokens == 16000 for task in plan.tasks)
    assert all(task.evidence_policy == "must_cite_paper_dom_source_ids" for task in plan.tasks)
    assert all(task.allowed_observation_types for task in plan.tasks)
    assert next(
        task for task in plan.tasks if task.task_type == ReadingTaskType.METHOD_MECHANISM
    ).allowed_observation_types == ["mechanism"]
    assert all(
        dom.source_exists(source_id) for task in plan.tasks for source_id in task.target_source_ids
    )
    orientation_task = next(
        task for task in plan.tasks if task.task_type == ReadingTaskType.ORIENTATION
    )
    assert source_pack(dom, orientation_task.target_source_ids)[0]["text"].startswith("We propose")


def test_reading_plan_targets_equation_sources_for_mechanism_tasks():
    dom = build_paper_dom_from_layout(
        paper_id="p_math",
        title="Math Paper",
        layout={
            "pages": [
                {
                    "page_no": 1,
                    "text": "Method\n\nWe optimize the loss $L = x + y$ with a runtime module.",
                    "section_candidates": [{"title": "Method", "level": 1}],
                }
            ]
        },
    )
    plan = build_initial_reading_plan(dom, max_sources_per_task=8)
    method_task = next(
        task for task in plan.tasks if task.task_type == ReadingTaskType.METHOD_MECHANISM
    )

    assert dom.equations
    assert dom.equations[0].source_id in method_task.target_source_ids
    assert any(
        item["kind"] == "equation" for item in source_pack(dom, method_task.target_source_ids)
    )


def test_reading_task_contract_normalizes_sources_and_requires_outputs():
    task = ReadingTask(
        task_id=" read_01_orientation ",
        task_type=ReadingTaskType.ORIENTATION,
        target_source_ids=[" span:p_test:p1:1 ", "span:p_test:p1:1", ""],
        required_outputs=["problem", " problem ", ""],
        allowed_observation_types=["problem", "problem", ""],
    )

    assert task.task_id == "read_01_orientation"
    assert task.target_source_ids == ["span:p_test:p1:1"]
    assert task.required_outputs == ["problem"]
    assert task.allowed_observation_types == ["problem"]
    with pytest.raises(ValueError, match="must declare required_outputs"):
        ReadingTask(
            task_id="read_bad",
            task_type=ReadingTaskType.ORIENTATION,
            target_source_ids=["span:p_test:p1:1"],
            required_outputs=[],
            allowed_observation_types=["problem"],
        )


def test_reading_task_contract_rejects_invalid_budget_policy_and_observation_types():
    with pytest.raises(ValueError, match="invalid allowed_observation_types"):
        ReadingTask(
            task_id="read_bad_type",
            task_type=ReadingTaskType.METHOD_MECHANISM,
            target_source_ids=["span:p_test:p1:1"],
            required_outputs=["mechanism"],
            allowed_observation_types=["claim"],
        )
    with pytest.raises(ValueError, match="max_model_calls must be >= 1"):
        ReadingTask(
            task_id="read_bad_calls",
            task_type=ReadingTaskType.ORIENTATION,
            target_source_ids=["span:p_test:p1:1"],
            required_outputs=["problem"],
            allowed_observation_types=["problem"],
            max_model_calls=0,
        )
    with pytest.raises(ValueError, match="evidence_policy must be"):
        ReadingTask(
            task_id="read_bad_policy",
            task_type=ReadingTaskType.ORIENTATION,
            target_source_ids=["span:p_test:p1:1"],
            required_outputs=["problem"],
            allowed_observation_types=["problem"],
            evidence_policy="cite_pages",
        )


def test_reading_plan_contract_rejects_duplicate_task_ids():
    first = ReadingTask(
        task_id="read_same",
        task_type=ReadingTaskType.ORIENTATION,
        target_source_ids=["span:p_test:p1:1"],
        required_outputs=["problem"],
        allowed_observation_types=["problem"],
    )
    second = ReadingTask(
        task_id="read_same",
        task_type=ReadingTaskType.CLAIM_INVENTORY,
        target_source_ids=["span:p_test:p1:2"],
        required_outputs=["claim"],
        allowed_observation_types=["claim"],
    )

    with pytest.raises(ValueError, match="duplicate task_id"):
        ReadingPlan(paper_id="p_test", tasks=[first, second])


def test_observation_log_is_append_only_and_requires_sources():
    dom = sample_dom()
    source_id = next(span.source_id for span in dom.spans if "block table method" in span.text)
    observation = ObservationCard(
        observation_id=make_observation_id(
            task_id="read_01_orientation",
            observation_type="claim",
            statement="The paper proposes a block table method.",
            source_ids=[source_id],
        ),
        paper_id="p_test",
        task_id="read_01_orientation",
        observation_type=ObservationType.CLAIM,
        statement="The paper proposes a block table method.",
        source_ids=[source_id],
    )

    log = ObservationLog(paper_id="p_test").append(observation)

    assert len(log.cards) == 1
    with pytest.raises(ValueError, match="duplicate observation_id"):
        log.append(observation)
    with pytest.raises(ValueError, match="at least one PaperDOM source_id"):
        ObservationCard(
            observation_id="obs_bad",
            paper_id="p_test",
            task_id="read_01_orientation",
            observation_type=ObservationType.CLAIM,
            statement="Unsupported claim",
            source_ids=[],
        )


def test_observation_log_append_many_preserves_order_and_default_duplicate_rejection():
    first = ObservationCard(
        observation_id="obs_first",
        paper_id="p_test",
        task_id="read_01_orientation",
        observation_type=ObservationType.CLAIM,
        statement="The paper makes a first claim.",
        source_ids=["span:p_test:p1:1"],
    )
    second = ObservationCard(
        observation_id="obs_second",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper makes a second claim.",
        source_ids=["span:p_test:p1:2"],
    )

    log = ObservationLog(paper_id="p_test").append_many([first, second])

    assert [card.observation_id for card in log.cards] == ["obs_first", "obs_second"]
    with pytest.raises(ValueError, match="duplicate observation_id"):
        log.append_many([first])


def test_observation_log_merge_ignores_identical_duplicates_but_rejects_conflicts():
    first = ObservationCard(
        observation_id="obs_same",
        paper_id="p_test",
        task_id="read_01_orientation",
        observation_type=ObservationType.CLAIM,
        statement="A claim.",
        source_ids=["span:p_test:p1:1"],
        created_at="t1",
    )
    identical_duplicate = ObservationCard(
        observation_id="obs_same",
        paper_id="p_test",
        task_id="read_01_orientation",
        observation_type=ObservationType.CLAIM,
        statement="A claim.",
        source_ids=["span:p_test:p1:1"],
        created_at="t2",
    )
    conflict = ObservationCard(
        observation_id="obs_same",
        paper_id="p_test",
        task_id="read_01_orientation",
        observation_type=ObservationType.CLAIM,
        statement="A different claim.",
        source_ids=["span:p_test:p1:1"],
        created_at="t3",
    )

    log = ObservationLog(paper_id="p_test", cards=(first,))
    merged = log.merge(
        ObservationLog(paper_id="p_test", cards=(identical_duplicate,)),
        on_duplicate="ignore",
    )

    assert len(merged.cards) == 1
    assert merged.cards[0].created_at == "t1"
    with pytest.raises(ValueError, match="conflicting duplicate observation_id"):
        log.merge(ObservationLog(paper_id="p_test", cards=(conflict,)), on_duplicate="ignore")


def test_observation_log_merge_rejects_other_paper():
    log = ObservationLog(paper_id="p_test")
    other = ObservationLog(paper_id="p_other")

    with pytest.raises(ValueError, match="observation paper_id mismatch"):
        log.merge(other)


def test_claim_graph_memory_and_audit_flow_from_observations():
    dom = sample_dom()
    result_span = next(span for span in dom.spans if "27%" in span.text)
    observation = ObservationCard(
        observation_id="obs_result",
        paper_id="p_test",
        task_id="read_06_result_extraction",
        observation_type=ObservationType.RESULT,
        statement="The method improves latency by 27% on Dataset-A.",
        source_ids=[result_span.source_id],
        confidence="high",
        extracted_numbers=[{"text": "27%"}],
    )

    graph = graph_from_observations("p_test", [observation])
    findings = audit_claim_graph(graph, dom)
    memory = materialize_paper_memory(
        graph,
        dom=dom,
        unresolved_audit_findings=[finding.finding_id for finding in findings],
        report_readiness=publish_status_from_findings(findings).value,
    )

    assert findings == []
    assert memory.result_nodes
    assert memory.report_readiness == PublishStatus.REVIEWED
    assert memory.evidence_index[memory.result_nodes[0]]
    assert memory.fact_nodes[0].node_id == memory.result_nodes[0]
    assert memory.fact_nodes[0].source_ids == [result_span.source_id]
    assert memory.fact_nodes[0].pages == [result_span.page_no]
    assert memory.fact_nodes[0].audit_status == PublishStatus.REVIEWED
    assert memory.fact_nodes[0].audit_issue_ids == []
    assert memory.audit_issues == []
    assert memory.evidence_sources[result_span.source_id].excerpt.startswith("The method")
    assert memory.evaluation_matrix[0].node_id == memory.result_nodes[0]
    assert memory.evaluation_matrix[0].extracted_numbers == [{"text": "27%"}]
    assert memory.evaluation_matrix[0].audit_status == PublishStatus.REVIEWED
    metrics = compute_core_quality_metrics(dom=dom, graph=graph, findings=findings)
    assert metrics.evidence_coverage == 1.0
    assert metrics.numeric_fact_node_count == 1
    assert metrics.number_not_located_count == 0
    assert metrics.numeric_locatable_rate == 1.0
    assert metrics.extracted_number_count == 1
    assert metrics.extracted_number_not_located_count == 0
    assert metrics.extracted_number_locatable_rate == 1.0
    assert metrics.unsupported_fact_node_count == 0
    assert metrics.unsupported_fact_node_rate == 0.0
    assert metrics.publish_status == PublishStatus.REVIEWED


def test_quality_metrics_track_reading_required_output_coverage():
    dom = sample_dom()
    reading_plan = build_initial_reading_plan(dom)
    result_task = next(
        task for task in reading_plan.tasks if task.task_type == ReadingTaskType.RESULT_EXTRACTION
    )
    result_span = next(span for span in dom.spans if "27%" in span.text)
    observation = ObservationCard(
        observation_id="obs_result",
        paper_id="p_test",
        task_id=result_task.task_id,
        observation_type=ObservationType.RESULT,
        statement="The method improves latency by 27% on Dataset-A.",
        source_ids=[result_span.source_id],
        confidence="high",
        covered_outputs=result_task.required_outputs,
        extracted_numbers=[{"text": "27%"}],
    )
    graph = graph_from_observations("p_test", [observation])
    findings = [
        *audit_claim_graph(graph, dom),
        *audit_reading_required_outputs(graph, reading_plan),
    ]

    metrics = compute_core_quality_metrics(
        dom=dom,
        graph=graph,
        findings=findings,
        reading_plan=reading_plan,
    )

    assert any(finding.code == "missing_reading_required_output" for finding in findings)
    assert metrics.reading_required_output_count == 14
    assert metrics.reading_required_output_covered_count == 1
    assert metrics.reading_required_output_coverage == 0.0714
    assert "read_01_orientation:problem" in metrics.missing_reading_required_outputs
    assert f"{result_task.task_id}:result" not in metrics.missing_reading_required_outputs
    assert metrics.publish_status == PublishStatus.DRAFT_WEAK


def test_claim_graph_keeps_valid_observation_relationship_edges():
    dom = sample_dom()
    first_source_id = dom.spans[0].source_id
    second_source_id = dom.spans[-1].source_id
    claim = ObservationCard(
        observation_id="obs_claim",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper proposes a block table method.",
        source_ids=[first_source_id],
    )
    mechanism = ObservationCard(
        observation_id="obs_mechanism",
        paper_id="p_test",
        task_id="read_03_method_mechanism",
        observation_type=ObservationType.MECHANISM,
        statement="The block table mechanism organizes serving state.",
        source_ids=[second_source_id],
        proposed_links=[
            {"source_id": "obs_mechanism", "target_id": "obs_claim", "kind": "explain"},
            {"source_id": "obs_mechanism", "target_id": "obs_claim", "kind": "support"},
            {"source_id": "obs_missing", "target_id": "obs_claim", "kind": "explains"},
            {"source_id": "obs_mechanism", "target_id": "obs_claim", "kind": "unknown"},
        ],
    )

    graph = graph_from_observations("p_test", [claim, mechanism])

    relationship_edges = [edge for edge in graph.edges if edge.kind != "supported_by"]
    assert [(edge.source_id, edge.target_id, edge.kind) for edge in relationship_edges] == [
        ("mechanism:obs_mechanism", "claim:obs_claim", "explains")
    ]
    assert relationship_edges[0].payload == {"proposed_by_observation_id": "obs_mechanism"}


def test_claim_graph_rejects_observations_from_other_paper():
    dom = sample_dom()
    observation = ObservationCard(
        observation_id="obs_other",
        paper_id="p_other",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper proposes a block table method.",
        source_ids=[dom.spans[0].source_id],
    )

    with pytest.raises(ValueError, match="observation paper_id mismatch"):
        graph_from_observations("p_test", [observation])


def test_claim_graph_add_node_is_idempotent_but_rejects_conflicting_payloads():
    graph = graph_from_observations("p_test", [])
    node = GraphNode(
        node_id="claim:obs_same",
        kind="claim",
        label="A claim.",
        payload={"observation_id": "obs_same"},
    )
    graph.add_node(node)
    graph.add_node(node.model_copy())

    with pytest.raises(ValueError, match="conflicting graph node_id"):
        graph.add_node(
            GraphNode(
                node_id="claim:obs_same",
                kind="claim",
                label="A different claim.",
                payload={"observation_id": "obs_same"},
            )
        )


def test_claim_graph_ignores_observation_relationship_edges_to_evidence_nodes():
    dom = sample_dom()
    first_source_id = next(
        span.source_id for span in dom.spans if "block table method" in span.text
    )
    second_source_id = next(span.source_id for span in dom.spans if "improves latency" in span.text)
    claim = ObservationCard(
        observation_id="obs_claim",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper proposes a block table method.",
        source_ids=[first_source_id],
    )
    result = ObservationCard(
        observation_id="obs_result",
        paper_id="p_test",
        task_id="read_06_result_extraction",
        observation_type=ObservationType.RESULT,
        statement="The method improves latency by 27% on Dataset-A.",
        source_ids=[second_source_id],
        proposed_links=[
            {
                "source_id": "obs_result",
                "target_id": f"evidence:{first_source_id}",
                "kind": "explains",
            },
            {
                "source_id": f"evidence:{second_source_id}",
                "target_id": "obs_claim",
                "kind": "explains",
            },
        ],
    )

    graph = graph_from_observations("p_test", [claim, result])

    assert [edge for edge in graph.edges if edge.kind != "supported_by"] == []


def test_paper_memory_view_materializes_claim_graph_relationship_edges():
    dom = sample_dom()
    claim = ObservationCard(
        observation_id="obs_claim",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper proposes a block table method.",
        source_ids=[dom.spans[0].source_id],
    )
    mechanism = ObservationCard(
        observation_id="obs_mechanism",
        paper_id="p_test",
        task_id="read_03_method_mechanism",
        observation_type=ObservationType.MECHANISM,
        statement="The block table mechanism organizes serving state.",
        source_ids=[dom.spans[-1].source_id],
        proposed_links=[
            {"source_id": "obs_mechanism", "target_id": "obs_claim", "kind": "explains"}
        ],
    )
    graph = graph_from_observations("p_test", [claim, mechanism])

    memory = materialize_paper_memory(graph)

    assert [edge.model_dump() for edge in memory.relationship_edges] == [
        {
            "source_id": "mechanism:obs_mechanism",
            "target_id": "claim:obs_claim",
            "kind": "explains",
            "payload": {"proposed_by_observation_id": "obs_mechanism"},
        }
    ]


def test_audit_blocks_supported_by_edges_that_do_not_target_evidence():
    dom = sample_dom()
    claim = ObservationCard(
        observation_id="obs_claim",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper proposes a block table method.",
        source_ids=[dom.spans[0].source_id],
    )
    mechanism = ObservationCard(
        observation_id="obs_mechanism",
        paper_id="p_test",
        task_id="read_03_method_mechanism",
        observation_type=ObservationType.MECHANISM,
        statement="The block table mechanism organizes serving state.",
        source_ids=[dom.spans[-1].source_id],
    )
    graph = graph_from_observations("p_test", [claim, mechanism])
    graph.edges.append(
        GraphEdge(
            source_id="claim:obs_claim",
            target_id="mechanism:obs_mechanism",
            kind="supported_by",
        )
    )

    findings = audit_claim_graph(graph, dom)

    assert "mechanism:obs_mechanism" not in graph.evidence_ids_for("claim:obs_claim")
    assert {finding.code for finding in findings} >= {"support_edge_target_not_evidence"}
    assert publish_status_from_findings(findings) == PublishStatus.BLOCKED


def test_audit_blocks_relationship_edges_that_touch_evidence_nodes():
    dom = sample_dom()
    first_source_id = next(
        span.source_id for span in dom.spans if "block table method" in span.text
    )
    second_source_id = next(span.source_id for span in dom.spans if "improves latency" in span.text)
    claim = ObservationCard(
        observation_id="obs_claim",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper proposes a block table method.",
        source_ids=[first_source_id],
    )
    result = ObservationCard(
        observation_id="obs_result",
        paper_id="p_test",
        task_id="read_06_result_extraction",
        observation_type=ObservationType.RESULT,
        statement="The method improves latency by 27% on Dataset-A.",
        source_ids=[second_source_id],
    )
    graph = graph_from_observations("p_test", [claim, result])
    graph.edges.append(
        GraphEdge(
            source_id="claim:obs_claim",
            target_id=f"evidence:{second_source_id}",
            kind="explains",
        )
    )
    graph.edges.append(
        GraphEdge(
            source_id=f"evidence:{first_source_id}",
            target_id="result:obs_result",
            kind="explains",
        )
    )

    findings = audit_claim_graph(graph, dom)

    assert {finding.code for finding in findings} >= {
        "relationship_edge_source_is_evidence",
        "relationship_edge_target_is_evidence",
    }
    assert publish_status_from_findings(findings) == PublishStatus.BLOCKED


def test_audit_blocks_claim_graph_for_other_paper_dom():
    dom = sample_dom()
    graph = graph_from_observations("p_other", [])

    findings = audit_claim_graph(graph, dom)

    assert {finding.code for finding in findings} >= {"claim_graph_paper_id_mismatch"}
    assert publish_status_from_findings(findings) == PublishStatus.BLOCKED


def test_audit_blocks_missing_sources_and_unsupported_fact_nodes():
    dom = sample_dom()
    observation = ObservationCard(
        observation_id="obs_bad_source",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper claims 99% improvement.",
        source_ids=["span:p_test:missing"],
    )

    graph = graph_from_observations("p_test", [observation])
    findings = audit_claim_graph(graph, dom)
    memory = materialize_paper_memory(
        graph,
        dom=dom,
        audit_findings=findings,
        report_readiness=publish_status_from_findings(findings).value,
    )
    claim_node = next(node for node in memory.fact_nodes if node.kind == "claim")

    assert {finding.code for finding in findings} >= {"missing_dom_source"}
    assert claim_node.audit_status == PublishStatus.BLOCKED
    assert claim_node.audit_issue_ids
    assert publish_status_from_findings(findings) == PublishStatus.BLOCKED


def test_audit_blocks_fact_text_unrelated_to_declared_source():
    dom = sample_dom()
    observation = ObservationCard(
        observation_id="obs_mismatch",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper achieves 99% accuracy on ImageNet.",
        source_ids=[dom.spans[0].source_id],
        confidence="high",
    )

    graph = graph_from_observations("p_test", [observation])
    findings = audit_claim_graph(graph, dom)
    metrics = compute_core_quality_metrics(dom=dom, graph=graph, findings=findings)

    assert {finding.code for finding in findings} >= {
        "fact_node_text_not_grounded_in_evidence_source",
        "number_not_located_in_source",
    }
    assert metrics.numeric_fact_node_count == 1
    assert metrics.number_not_located_count == 1
    assert metrics.numeric_locatable_rate == 0.0
    assert publish_status_from_findings(findings) == PublishStatus.BLOCKED


def test_audit_flags_extracted_numbers_not_located_in_declared_sources():
    dom = sample_dom()
    result_span = next(span for span in dom.spans if "27%" in span.text)
    observation = ObservationCard(
        observation_id="obs_result",
        paper_id="p_test",
        task_id="read_06_result_extraction",
        observation_type=ObservationType.RESULT,
        statement="The method improves latency by 27% on Dataset-A.",
        source_ids=[result_span.source_id],
        confidence="high",
        extracted_numbers=[{"text": "99%"}],
    )

    graph = graph_from_observations("p_test", [observation])
    findings = audit_claim_graph(graph, dom)
    memory = materialize_paper_memory(
        graph,
        dom=dom,
        audit_findings=findings,
        report_readiness=publish_status_from_findings(findings).value,
    )
    metrics = compute_core_quality_metrics(dom=dom, graph=graph, findings=findings)
    result_node = next(node for node in memory.fact_nodes if node.kind == "result")

    assert {finding.code for finding in findings} >= {
        "extracted_number_not_located_in_source"
    }
    assert "number_not_located_in_source" not in {finding.code for finding in findings}
    assert result_node.audit_status == PublishStatus.REVIEWED_WITH_LIMITS
    assert result_node.audit_issue_ids == [findings[0].finding_id]
    assert memory.audit_issues_by_node[result_node.node_id] == result_node.audit_issue_ids
    assert metrics.extracted_number_count == 1
    assert metrics.extracted_number_not_located_count == 1
    assert metrics.extracted_number_locatable_rate == 0.0
    assert publish_status_from_findings(findings) == PublishStatus.REVIEWED_WITH_LIMITS


def test_audit_blocks_dangling_claim_graph_edges():
    dom = sample_dom()
    source_id = next(span.source_id for span in dom.spans if "block table method" in span.text)
    observation = ObservationCard(
        observation_id="obs_claim",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper proposes a block table method.",
        source_ids=[source_id],
    )
    graph = graph_from_observations("p_test", [observation])
    graph.edges.append(
        GraphEdge(
            source_id="claim:missing",
            target_id="evidence:missing",
            kind="supported_by",
        )
    )

    findings = audit_claim_graph(graph, dom)

    assert {finding.code for finding in findings} >= {
        "dangling_graph_edge_source",
        "dangling_graph_edge_target",
    }
    assert publish_status_from_findings(findings) == PublishStatus.BLOCKED


def test_report_draft_is_a_claim_graph_view_with_declared_evidence():
    dom = sample_dom()
    source_id = dom.spans[0].source_id
    observation = ObservationCard(
        observation_id="obs_claim",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper proposes a block table method.",
        source_ids=[source_id],
    )
    graph = graph_from_observations("p_test", [observation])

    draft = build_report_draft_from_graph(graph)
    findings = audit_report_draft_against_graph(draft, graph)

    assert findings == []
    paragraph = draft.sections[0].paragraphs[0]
    assert paragraph.used_node_ids
    assert paragraph.used_evidence_ids

    bad = GraphReportDraft(
        paper_id="p_test",
        sections=[
            ReportSection(
                section_id="bad",
                title="Bad",
                paragraphs=[
                    ReportParagraph(
                        paragraph_id="bad_01",
                        markdown="A new unsupported fact.",
                        used_node_ids=[],
                        used_evidence_ids=[],
                    )
                ],
            )
        ],
    )
    bad_findings = audit_report_draft_against_graph(bad, graph)
    assert {finding.code for finding in bad_findings} == {
        "report_paragraph_missing_node_ids",
        "report_paragraph_missing_evidence_ids",
    }

    ungrounded = GraphReportDraft(
        paper_id="p_test",
        sections=[
            ReportSection(
                section_id="bad",
                title="Bad",
                paragraphs=[
                    ReportParagraph(
                        paragraph_id="bad_02",
                        markdown="A new unsupported result about 99% accuracy.",
                        used_node_ids=[paragraph.used_node_ids[0]],
                        used_evidence_ids=[paragraph.used_evidence_ids[0]],
                    )
                ],
            )
        ],
    )
    ungrounded_findings = audit_report_draft_against_graph(ungrounded, graph)
    assert {finding.code for finding in ungrounded_findings} == {
        "report_paragraph_text_not_grounded_in_declared_nodes"
    }


def test_report_audit_rejects_numeric_fact_added_outside_declared_nodes():
    dom = sample_dom()
    result_span = next(span for span in dom.spans if "27%" in span.text)
    observation = ObservationCard(
        observation_id="obs_result",
        paper_id="p_test",
        task_id="read_06_result_extraction",
        observation_type=ObservationType.RESULT,
        statement="The method improves latency by 27% on Dataset-A.",
        source_ids=[result_span.source_id],
    )
    graph = graph_from_observations("p_test", [observation])
    draft = GraphReportDraft(
        paper_id="p_test",
        sections=[
            ReportSection(
                section_id="results",
                title="Results",
                paragraphs=[
                    ReportParagraph(
                        paragraph_id="result_01",
                        markdown=(
                            "The method improves latency by 27% on Dataset-A and reaches "
                            "99% accuracy."
                        ),
                        used_node_ids=["result:obs_result"],
                        used_evidence_ids=[f"evidence:{result_span.source_id}"],
                    )
                ],
            )
        ],
    )

    findings = audit_report_draft_against_graph(draft, graph)

    assert {finding.code for finding in findings} == {
        "report_paragraph_number_not_grounded_in_declared_nodes"
    }
    assert findings[0].source_ids == [result_span.source_id]


def test_report_audit_blocks_draft_for_other_claim_graph():
    dom = sample_dom()
    observation = ObservationCard(
        observation_id="obs_claim",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper proposes a block table method.",
        source_ids=[dom.spans[0].source_id],
    )
    graph = graph_from_observations("p_test", [observation])
    draft = build_report_draft_from_graph(graph).model_copy(update={"paper_id": "p_other"})

    findings = audit_report_draft_against_graph(draft, graph)

    assert {finding.code for finding in findings} >= {"report_draft_paper_id_mismatch"}


def test_report_audit_rejects_declared_node_not_reflected_in_paragraph_text():
    dom = sample_dom()
    first_observation = ObservationCard(
        observation_id="obs_claim_first",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper proposes a block table method.",
        source_ids=[dom.spans[0].source_id],
    )
    second_observation = ObservationCard(
        observation_id="obs_claim_second",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper reports a latency result.",
        source_ids=[dom.spans[-1].source_id],
    )
    graph = graph_from_observations("p_test", [first_observation, second_observation])
    bad = GraphReportDraft(
        paper_id="p_test",
        sections=[
            ReportSection(
                section_id="claims",
                title="Claims",
                paragraphs=[
                    ReportParagraph(
                        paragraph_id="claim_01",
                        markdown="The paper proposes a block table method.",
                        used_node_ids=["claim:obs_claim_first", "claim:obs_claim_second"],
                        used_evidence_ids=[
                            f"evidence:{dom.spans[0].source_id}",
                            f"evidence:{dom.spans[-1].source_id}",
                        ],
                    )
                ],
            )
        ],
    )

    findings = audit_report_draft_against_graph(bad, graph)

    assert {finding.code for finding in findings} == {
        "report_paragraph_declared_node_not_used_in_text"
    }
    assert findings[0].node_id == "claim:obs_claim_second"


def test_graph_report_markdown_declares_nodes_evidence_and_sources():
    dom = sample_dom()
    source_id = next(span.source_id for span in dom.spans if "block table method" in span.text)
    observation = ObservationCard(
        observation_id="obs_claim",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper proposes a block table method.",
        source_ids=[source_id],
    )
    graph = graph_from_observations("p_test", [observation])
    draft = build_report_draft_from_graph(graph)

    markdown = render_graph_report_markdown(
        title="Test Paper",
        draft=draft,
        graph=graph,
        dom=dom,
        quality={"publish_status": PublishStatus.REVIEWED},
    )

    assert "# Test Paper" in markdown
    assert "ClaimGraph nodes: `claim:obs_claim`" in markdown
    assert f"Evidence nodes: `evidence:{source_id}`" in markdown
    assert f"PaperDOM sources: `{source_id}`" in markdown
    assert "The paper proposes a block table method." in markdown


def test_core_graph_report_view_is_materialized_from_typed_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    root = data_dir / "core" / "v2" / "p_test"
    dom = sample_dom()
    source_id = next(span.source_id for span in dom.spans if "block table method" in span.text)
    reading_plan = reading_plan_subset(dom, ReadingTaskType.CLAIM_INVENTORY)
    claim_task = reading_task(reading_plan, ReadingTaskType.CLAIM_INVENTORY)
    observation = ObservationCard(
        observation_id="obs_claim",
        paper_id="p_test",
        task_id=claim_task.task_id,
        observation_type=ObservationType.CLAIM,
        statement="The paper proposes a block table method.",
        source_ids=[source_id],
        covered_outputs=claim_task.required_outputs,
    )
    graph = graph_from_observations("p_test", [observation])
    draft = build_report_draft_from_graph(graph)
    artifact_payloads = {
        "paper_dom.v1.json": ("paper_dom", dom.model_dump(mode="json")),
        "reading_plan.v1.json": ("reading_plan", reading_plan.model_dump(mode="json")),
        "observation_log.v1.json": ("observation_log", {"paper_id": "p_test"}),
        "claim_graph.v1.json": ("claim_graph", graph.model_dump(mode="json")),
        "audit_findings.v1.json": ("audit_findings", []),
        "quality_metrics.v1.json": (
            "core_quality_metrics",
            {"paper_id": "p_test", "publish_status": PublishStatus.REVIEWED},
        ),
        "paper_memory_view.v1.json": (
            "paper_memory_view",
            {"paper_id": "p_test", "schema_version": "paper_memory.view.v1"},
        ),
        "report_draft.v1.json": ("graph_report_draft", draft.model_dump(mode="json")),
        "report_audit_findings.v1.json": ("report_audit_findings", []),
    }
    for filename, (artifact_type, data) in artifact_payloads.items():
        write_typed_artifact(
            root / filename,
            artifact_type=artifact_type,
            data=data,
            producer="unit",
            metadata={"paper_id": "p_test"},
        )
    write_core_v2_manifest(root, "p_test")

    path = write_core_graph_report_view(
        output_dir=output_dir,
        data_dir=data_dir,
        paper_id="p_test",
        title="Test Paper",
        report_name="test.md",
    )

    assert path == output_dir / "papers" / "core_graph" / "test.core_graph.md"
    assert path is not None
    markdown = path.read_text(encoding="utf-8")
    assert "# Test Paper" in markdown
    assert "Publish status: `REVIEWED`" in markdown
    assert f"PaperDOM sources: `{source_id}`" in markdown


def test_core_graph_report_view_uses_manifest_current_publish_status(tmp_path):
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            }
        ]
    }
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    source_id = next(span.source_id for span in dom.spans if "block table method" in span.text)
    reading_plan = reading_plan_subset(dom, ReadingTaskType.CLAIM_INVENTORY)
    claim_task = reading_task(reading_plan, ReadingTaskType.CLAIM_INVENTORY)
    write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=reading_plan,
        observation_log=ObservationLog(paper_id="p_test").append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id=claim_task.task_id,
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[source_id],
                confidence="low",
                covered_outputs=claim_task.required_outputs,
            )
        ),
        producer="unit_test",
    )
    root = data_dir / "core" / "v2" / "p_test"
    write_typed_artifact(
        root / "quality_metrics.v1.json",
        artifact_type="core_quality_metrics",
        data={"paper_id": "p_test", "publish_status": PublishStatus.REVIEWED},
        producer="unit_test_stale_quality",
        metadata={"paper_id": "p_test"},
    )

    path = write_core_graph_report_view(
        output_dir=output_dir,
        data_dir=data_dir,
        paper_id="p_test",
        title="Test Paper",
        report_name="test.md",
    )

    assert path is not None
    markdown = path.read_text(encoding="utf-8")
    assert "Publish status: `REVIEWED_WITH_LIMITS`" in markdown
    assert "Publish status: `REVIEWED`" not in markdown


def test_report_audit_rejects_declared_evidence_not_linked_to_declared_node():
    dom = sample_dom()
    first_source_id = dom.spans[0].source_id
    second_source_id = dom.spans[-1].source_id
    first_observation = ObservationCard(
        observation_id="obs_claim_first",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper proposes a block table method.",
        source_ids=[first_source_id],
    )
    second_observation = ObservationCard(
        observation_id="obs_claim_second",
        paper_id="p_test",
        task_id="read_02_claim_inventory",
        observation_type=ObservationType.CLAIM,
        statement="The paper reports a latency result.",
        source_ids=[second_source_id],
    )
    graph = graph_from_observations("p_test", [first_observation, second_observation])
    bad = GraphReportDraft(
        paper_id="p_test",
        sections=[
            ReportSection(
                section_id="claims",
                title="Claims",
                paragraphs=[
                    ReportParagraph(
                        paragraph_id="claim_01",
                        markdown="The paper proposes a block table method.",
                        used_node_ids=["claim:obs_claim_first"],
                        used_evidence_ids=[f"evidence:{second_source_id}"],
                    )
                ],
            )
        ],
    )

    findings = audit_report_draft_against_graph(bad, graph)

    assert {finding.code for finding in findings} == {
        "report_paragraph_node_missing_declared_evidence",
        "report_paragraph_evidence_not_linked_to_declared_node",
    }
    evidence_finding = next(
        finding
        for finding in findings
        if finding.code == "report_paragraph_evidence_not_linked_to_declared_node"
    )
    assert evidence_finding.source_ids == [second_source_id]


def test_finite_runtime_node_enforces_model_call_budget():
    spec = NodeSpec(
        node_id="read_orientation",
        output_artifact_type="observation_cards",
        max_model_calls=0,
    )

    def handler(context):
        context.record_model_call()
        return ArtifactEnvelope(
            artifact_type="observation_cards",
            producer="unit",
            data=[],
        )

    result = run_finite_node(spec, [], handler)

    assert result.status == NodeStatus.FAIL
    assert result.issues
    assert "max_model_calls=0" in result.issues[0]


def test_artifact_envelope_rejects_wrong_output_type():
    spec = NodeSpec(node_id="node", output_artifact_type="claim_graph")

    result = run_finite_node(
        spec,
        [],
        lambda _context: ArtifactEnvelope(
            artifact_type="observation_cards",
            producer="unit",
            data=[],
        ),
    )

    assert result.status == NodeStatus.FAIL
    assert "Expected artifact_type=claim_graph" in result.issues[0]


def test_finite_runtime_node_requires_declared_output_artifact():
    spec = NodeSpec(node_id="node", output_artifact_type="claim_graph")

    result = run_finite_node(spec, [], lambda _context: None)

    assert result.status == NodeStatus.FAIL
    assert "did not return output_artifact_type=claim_graph" in result.issues[0]


def test_finite_runtime_node_rejects_non_artifact_output():
    spec = NodeSpec(node_id="node")

    result = run_finite_node(spec, [], lambda _context: {"artifact_type": "claim_graph"})

    assert result.status == NodeStatus.FAIL
    assert "returned non-artifact output=dict" in result.issues[0]


def test_finite_runtime_node_rejects_duplicate_input_artifact_types():
    spec = NodeSpec(node_id="node", input_artifact_types=("paper_dom",))
    inputs = [
        ArtifactEnvelope(artifact_type="paper_dom", producer="unit", data={}),
        ArtifactEnvelope(artifact_type="paper_dom", producer="unit", data={}),
    ]

    result = run_finite_node(spec, inputs, lambda _context: None)

    assert result.status == NodeStatus.FAIL
    assert result.issues == ["duplicate_input_artifact:paper_dom"]


def test_finite_runtime_node_enforces_allowed_tools():
    spec = NodeSpec(
        node_id="read_sources",
        output_artifact_type="observation_cards",
        allowed_tools=("paper_dom.read_sources",),
    )

    def handler(context):
        context.record_tool_call(
            "paper_dom.read_sources",
            source_ids=["span:p_test:p1:1", "span:p_test:p1:1", ""],
        )
        return ArtifactEnvelope(
            artifact_type="observation_cards",
            producer="unit",
            data=[],
            source_ids=["span:p_test:p1:1"],
        )

    result = run_finite_node(spec, [], handler)

    assert result.status == NodeStatus.PASS
    assert result.tool_calls_used == 1
    assert result.used_tools == ["paper_dom.read_sources"]
    assert result.tool_source_ids == {"paper_dom.read_sources": ["span:p_test:p1:1"]}
    assert result.output is not None
    assert result.output.source_ids == ["span:p_test:p1:1"]


def test_finite_runtime_node_rejects_output_source_ids_outside_tool_trace():
    spec = NodeSpec(
        node_id="read_sources",
        output_artifact_type="observation_cards",
        allowed_tools=("paper_dom.read_sources",),
    )

    def handler(context):
        context.record_tool_call(
            "paper_dom.read_sources",
            source_ids=["span:p_test:p1:1"],
        )
        return ArtifactEnvelope(
            artifact_type="observation_cards",
            producer="unit",
            data=[],
            source_ids=["span:p_test:p2:9"],
        )

    result = run_finite_node(spec, [], handler)

    assert result.status == NodeStatus.FAIL
    assert "output source_ids outside tool trace: span:p_test:p2:9" in result.issues[0]


def test_finite_runtime_node_rejects_disallowed_tools():
    spec = NodeSpec(node_id="read_sources", allowed_tools=("paper_dom.read_sources",))

    def handler(context):
        context.record_tool_call("filesystem.read")
        return None

    result = run_finite_node(spec, [], handler)

    assert result.status == NodeStatus.FAIL
    assert result.tool_calls_used == 1
    assert "disallowed tool=filesystem.read" in result.issues[0]


def test_finite_runtime_node_requires_tool_source_ids():
    spec = NodeSpec(node_id="read_sources", allowed_tools=("paper_dom.read_sources",))

    def handler(context):
        context.record_tool_call("paper_dom.read_sources")
        return None

    result = run_finite_node(spec, [], handler)

    assert result.status == NodeStatus.FAIL
    assert result.tool_calls_used == 1
    assert "tool=paper_dom.read_sources did not return source_ids" in result.issues[0]


def test_finite_runtime_node_enforces_token_budget():
    spec = NodeSpec(node_id="token_budget", max_tokens=10)

    def handler(context):
        context.record_token_usage({"prompt_tokens": 8, "completion_tokens": 4})
        return None

    result = run_finite_node(spec, [], handler)

    assert result.status == NodeStatus.FAIL
    assert result.tokens_used == 12
    assert result.token_usage == {"prompt_tokens": 8, "completion_tokens": 4}
    assert "exceeded max_tokens=10" in result.issues[0]


def test_finite_runtime_node_enforces_timeout_after_handler_returns():
    spec = NodeSpec(node_id="slow_node", timeout_seconds=1)

    def handler(context):
        context.started_at -= 2
        return ArtifactEnvelope(
            artifact_type="slow_artifact",
            producer="unit",
            data=[],
        )

    result = run_finite_node(spec, [], handler)

    assert result.status == NodeStatus.FAIL
    assert "exceeded timeout_seconds=1" in result.issues[0]


def test_stage03_writes_core_v2_artifact_envelopes(tmp_path):
    output_dir = tmp_path / "out"
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    pipeline = PaperLensWorkflow(
        input_dir=input_dir,
        output_dir=output_dir,
        config=CoreConfig(offline_debug=True),
        events=EventWriter(
            "run_test",
            output_dir / ".paperlens" / "data" / "events.jsonl",
            output_dir / ".paperlens" / "data" / "errors.jsonl",
        ),
        control=ControlState(),
    )
    try:
        pipeline.prepare_output()
        paper = PaperRecord(
            paper_id="p_test",
            file_path="paper.pdf",
            file_hash="hash",
            canonical_title="Test Paper",
            page_count=2,
        )
        pages = [
            PageArtifact(
                paper_id="p_test",
                page_no=1,
                text="Abstract\n\nWe propose a block table method for serving.",
                section_candidates=[{"title": "Abstract", "level": 1}],
            ),
            PageArtifact(
                paper_id="p_test",
                page_no=2,
                text="Evaluation\n\nThe method improves latency by 27% on Dataset-A.",
                section_candidates=[{"title": "Evaluation", "level": 1}],
            ),
        ]
        pipeline.papers = [paper]
        pipeline.db.upsert_paper(paper)
        pipeline.db.insert_page_artifacts(pages)
        layout_path = output_dir / ".paperlens" / "data" / "artifacts" / "layout" / "p_test.json"
        layout_path.parent.mkdir(parents=True, exist_ok=True)
        layout_path.write_text(
            json.dumps({"pages": [page.model_dump() for page in pages]}, ensure_ascii=False),
            encoding="utf-8",
        )

        pipeline.stage_03_skim()

        core_root = output_dir / ".paperlens" / "data" / "core" / "v2" / "p_test"
        dom_envelope = json.loads((core_root / "paper_dom.v1.json").read_text(encoding="utf-8"))
        plan_envelope = json.loads((core_root / "reading_plan.v1.json").read_text(encoding="utf-8"))
        graph_envelope = json.loads((core_root / "claim_graph.v1.json").read_text(encoding="utf-8"))
        metrics_envelope = json.loads(
            (core_root / "quality_metrics.v1.json").read_text(encoding="utf-8")
        )
        memory_envelope = json.loads(
            (core_root / "paper_memory_view.v1.json").read_text(encoding="utf-8")
        )
        report_envelope = json.loads(
            (core_root / "report_draft.v1.json").read_text(encoding="utf-8")
        )
        report_audit_envelope = json.loads(
            (core_root / "report_audit_findings.v1.json").read_text(encoding="utf-8")
        )
        core_manifest_envelope = json.loads(
            (core_root / "core_manifest.v1.json").read_text(encoding="utf-8")
        )

        assert dom_envelope["artifact_type"] == "paper_dom"
        assert plan_envelope["artifact_type"] == "reading_plan"
        assert graph_envelope["artifact_type"] == "claim_graph"
        assert metrics_envelope["artifact_type"] == "core_quality_metrics"
        assert memory_envelope["artifact_type"] == "paper_memory_view"
        assert report_envelope["artifact_type"] == "graph_report_draft"
        assert report_audit_envelope["artifact_type"] == "report_audit_findings"
        assert core_manifest_envelope["artifact_type"] == "core_v2_manifest"
        assert dom_envelope["data"]["spans"]
        assert plan_envelope["data"]["tasks"]
        assert graph_envelope["data"]["nodes"]
        assert report_envelope["data"]["sections"]
        assert report_audit_envelope["data"] == []
        assert memory_envelope["data"]["fact_nodes"]
        assert memory_envelope["data"]["evidence_sources"]
        assert memory_envelope["data"]["evaluation_matrix"]
        assert memory_envelope["data"]["evaluation_matrix"][0]["source_ids"]
        assert metrics_envelope["data"]["fact_node_count"] > 0
        assert metrics_envelope["data"]["publish_status"] == PublishStatus.DRAFT_WEAK
        assert core_manifest_envelope["data"]["status"] == "COMPLETE"
        assert core_manifest_envelope["data"]["publish_status"] == PublishStatus.DRAFT_WEAK
        assert core_manifest_envelope["data"]["consumable"] is False
        assert core_manifest_envelope["data"]["issues"] == []
        assert all(
            item["exists"] for item in core_manifest_envelope["data"]["required_artifacts"].values()
        )
        assert any(
            item.artifact_type == "core_v2_paper_dom"
            for item in pipeline.db.list_artifact_versions()
        )
    finally:
        pipeline.db.close()


def test_core_v2_model_observer_rewrites_observation_graph_artifacts(tmp_path):
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            }
        ]
    }
    write_core_v2_artifacts(
        data_dir=tmp_path,
        paper=paper,
        layout=layout,
    )
    calls = {"count": 0}

    class FakeClient:
        config = SimpleNamespace(kind="openai-compatible", model="fake-model")

        def invoke_json(self, *, user_prompt, **_kwargs):
            calls["count"] += 1
            prompt = json.loads(user_prompt)
            source_id = prompt["evidence_pack"][0]["source_id"]
            task_type = prompt["task_spec"]["task_type"]
            observation_type = prompt["task_spec"]["allowed_observation_types"][0]
            covered_outputs = prompt["task_spec"]["required_outputs"]
            assert _kwargs["max_tokens"] == prompt["task_spec"]["max_tokens"]
            return SimpleNamespace(
                data={
                    "artifact_type": "observation_cards",
                    "artifact_version": "v1",
                    "producer": "fake-model",
                    "data": {
                        "cards": [
                            {
                                "observation_type": observation_type,
                                "statement": (
                                    f"{task_type} observation about the block table method from "
                                    "a source-bound card."
                                ),
                                "source_ids": [source_id],
                                "confidence": "high",
                                "provenance": "explicit",
                                "uncertainty": None,
                                "covered_outputs": covered_outputs,
                                "extracted_numbers": [],
                                "proposed_links": [],
                            }
                        ]
                    },
                },
                usage={"prompt_tokens": 10, "completion_tokens": 2},
                request_id=f"req_{calls['count']}",
            )

    usage_rows = []
    agent_runs = []
    result = run_core_v2_model_observation_tasks(
        client=FakeClient(),
        data_dir=tmp_path,
        paper=paper,
        stage="stage_07_normal_read",
        record_usage=lambda _stage, usage: usage_rows.append(usage),
        record_agent_run=agent_runs.append,
    )

    root = tmp_path / "core" / "v2" / "p_test"
    observation_log = json.loads((root / "observation_log.v1.json").read_text(encoding="utf-8"))
    graph = json.loads((root / "claim_graph.v1.json").read_text(encoding="utf-8"))
    metrics = json.loads((root / "quality_metrics.v1.json").read_text(encoding="utf-8"))
    core_manifest = json.loads((root / "core_manifest.v1.json").read_text(encoding="utf-8"))

    assert result["tasks"] == calls["count"]
    assert result["cards"] == calls["count"]
    assert observation_log["producer"] == "paperlens_core_v2_model_observer"
    assert len(observation_log["data"]["cards"]) == calls["count"]
    assert graph["producer"] == "paperlens_core_v2_model_observer"
    assert metrics["data"]["publish_status"] == PublishStatus.REVIEWED
    assert metrics["data"]["reading_required_output_count"] == 14
    assert metrics["data"]["reading_required_output_covered_count"] == 14
    assert metrics["data"]["reading_required_output_coverage"] == 1.0
    assert metrics["data"]["missing_reading_required_outputs"] == []
    assert core_manifest["artifact_type"] == "core_v2_manifest"
    assert core_manifest["data"]["status"] == "COMPLETE"
    assert core_manifest["data"]["publish_status"] == PublishStatus.REVIEWED
    assert core_manifest["data"]["consumable"] is True
    assert len(usage_rows) == calls["count"]
    assert all(row["status"] == "PASS" for row in agent_runs)
    assert all(row["tool_calls_used"] == 1 for row in agent_runs)
    assert all(row["used_tools"] == ["paper_dom.read_sources"] for row in agent_runs)
    assert all(row["tool_source_ids"].get("paper_dom.read_sources") for row in agent_runs)
    assert all(row["tokens_used"] == 12 for row in agent_runs)
    assert all(
        row["token_usage"] == {"prompt_tokens": 10, "completion_tokens": 2} for row in agent_runs
    )


def test_write_core_v2_from_observation_log_writes_complete_core_inputs(tmp_path):
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            }
        ]
    }
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    reading_plan = build_initial_reading_plan(dom)
    claim_span = next(span for span in dom.spans if "block table method" in span.text)

    paths = write_core_v2_from_observation_log(
        data_dir=tmp_path,
        paper=paper,
        dom=dom,
        reading_plan=reading_plan,
        observation_log=ObservationLog(paper_id="p_test").append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id="read_02_claim_inventory",
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[claim_span.source_id],
            )
        ),
        producer="unit_test",
    )
    loaded_dom, loaded_plan = load_core_v2_dom_and_plan(tmp_path, "p_test")
    metrics = json.loads(paths["quality_metrics"].read_text(encoding="utf-8"))
    findings = json.loads(paths["audit_findings"].read_text(encoding="utf-8"))

    assert (tmp_path / "core" / "v2" / "p_test" / "paper_dom.v1.json").exists()
    assert (tmp_path / "core" / "v2" / "p_test" / "reading_plan.v1.json").exists()
    assert loaded_dom.paper_id == "p_test"
    assert loaded_plan.paper_id == "p_test"
    assert paths["quality_metrics"].exists()
    assert metrics["data"]["publish_status"] == PublishStatus.DRAFT_WEAK
    assert any(finding["code"] == "missing_reading_required_output" for finding in findings["data"])


def test_write_core_v2_from_observation_log_rejects_inconsistent_inputs(tmp_path):
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    dom = sample_dom()
    reading_plan = build_initial_reading_plan(dom)

    with pytest.raises(ValueError, match="observation_log paper_id mismatch"):
        write_core_v2_from_observation_log(
            data_dir=tmp_path,
            paper=paper,
            dom=dom,
            reading_plan=reading_plan,
            observation_log=ObservationLog(paper_id="p_other"),
            producer="unit_test",
        )


def test_core_v2_manifest_reports_incomplete_artifact_sets(tmp_path):
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    write_core_v2_artifacts(
        data_dir=tmp_path,
        paper=paper,
        layout={
            "pages": [
                {
                    "page_no": 1,
                    "text": "Abstract\n\nWe propose a block table method for serving.",
                    "section_candidates": [{"title": "Abstract", "level": 1}],
                }
            ]
        },
    )
    root = tmp_path / "core" / "v2" / "p_test"
    (root / "quality_metrics.v1.json").unlink()

    manifest = build_core_v2_manifest(root, "p_test")

    assert manifest["status"] == "INCOMPLETE"
    assert manifest["publish_status"] is None
    assert manifest["consumable"] is False
    assert "missing:quality_metrics.v1.json" in manifest["issues"]


def test_core_v2_manifest_reaudits_current_artifacts_before_consumable(tmp_path):
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            }
        ]
    }
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    claim_span = next(span for span in dom.spans if "block table method" in span.text)
    narrow_plan = reading_plan_subset(dom, ReadingTaskType.CLAIM_INVENTORY)
    full_plan = build_initial_reading_plan(dom)
    claim_task = reading_task(narrow_plan, ReadingTaskType.CLAIM_INVENTORY)
    write_core_v2_from_observation_log(
        data_dir=tmp_path,
        paper=paper,
        dom=dom,
        reading_plan=narrow_plan,
        observation_log=ObservationLog(paper_id="p_test").append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id=claim_task.task_id,
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[claim_span.source_id],
                covered_outputs=claim_task.required_outputs,
            )
        ),
        producer="unit_test",
    )
    root = tmp_path / "core" / "v2" / "p_test"
    write_typed_artifact(
        root / "reading_plan.v1.json",
        artifact_type="reading_plan",
        data=full_plan.model_dump(),
        producer="unit_test_stale_plan",
        metadata={"paper_id": paper.paper_id},
    )

    manifest = build_core_v2_manifest(root, "p_test")

    assert manifest["status"] == "COMPLETE"
    assert manifest["artifact_publish_status"] == PublishStatus.REVIEWED
    assert manifest["current_audit_publish_status"] == PublishStatus.DRAFT_WEAK
    assert manifest["publish_status"] == PublishStatus.DRAFT_WEAK
    assert manifest["consumable"] is False
    assert "missing_reading_required_output" in manifest["current_audit_issue_codes"]


def test_refresh_core_v2_audit_artifacts_blocks_missing_dom_sources(tmp_path):
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    write_core_v2_artifacts(
        data_dir=tmp_path,
        paper=paper,
        layout={
            "pages": [
                {
                    "page_no": 1,
                    "text": "Abstract\n\nWe propose a block table method for serving.",
                    "section_candidates": [{"title": "Abstract", "level": 1}],
                }
            ]
        },
    )
    root = tmp_path / "core" / "v2" / "p_test"
    graph_path = root / "claim_graph.v1.json"
    graph_envelope = json.loads(graph_path.read_text(encoding="utf-8"))
    evidence_node = next(
        node for node in graph_envelope["data"]["nodes"].values() if node["kind"] == "evidence"
    )
    evidence_node["payload"]["source_id"] = "span:p_test:missing"
    graph_path.write_text(json.dumps(graph_envelope, ensure_ascii=False), encoding="utf-8")

    result = refresh_core_v2_audit_artifacts(data_dir=tmp_path, paper=paper)

    metrics = json.loads((root / "quality_metrics.v1.json").read_text(encoding="utf-8"))
    findings = json.loads((root / "audit_findings.v1.json").read_text(encoding="utf-8"))
    memory = json.loads((root / "paper_memory_view.v1.json").read_text(encoding="utf-8"))
    assert result["publish_status"] == PublishStatus.BLOCKED
    assert metrics["producer"] == "paperlens_core_v2_audit_suite"
    assert metrics["data"]["publish_status"] == PublishStatus.BLOCKED
    assert {finding["code"] for finding in findings["data"]} >= {"missing_dom_source"}
    assert memory["data"]["report_readiness"] == PublishStatus.BLOCKED
    assert memory["data"]["unresolved_audit_findings"]


def test_model_observation_cards_reject_unknown_source_ids():
    envelope = ArtifactEnvelope(
        artifact_type="observation_cards",
        producer="fake",
        data={
            "cards": [
                {
                    "observation_type": "claim",
                    "statement": "Unsupported source id.",
                    "source_ids": ["page:1"],
                    "confidence": "high",
                    "provenance": "explicit",
                    "uncertainty": None,
                    "extracted_numbers": [],
                    "proposed_links": [],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="outside this task evidence pack"):
        observation_cards_from_model_envelope(
            envelope,
            paper_id="p_test",
            task=build_initial_reading_plan(sample_dom()).tasks[0],
            allowed_source_ids=set(
                build_initial_reading_plan(sample_dom()).tasks[0].target_source_ids
            ),
        )


def test_model_observation_cards_reject_empty_task_output():
    task = build_initial_reading_plan(sample_dom()).tasks[0]
    envelope = ArtifactEnvelope(
        artifact_type="observation_cards",
        producer="fake",
        data={"cards": []},
    )

    with pytest.raises(ValueError, match="returned no valid observation cards"):
        observation_cards_from_model_envelope(
            envelope,
            paper_id="p_test",
            task=task,
            allowed_source_ids=set(task.target_source_ids),
        )


def test_model_observation_cards_reject_only_invalid_task_output():
    task = build_initial_reading_plan(sample_dom()).tasks[0]
    envelope = ArtifactEnvelope(
        artifact_type="observation_cards",
        producer="fake",
        data={
            "cards": [
                {
                    "observation_type": task.allowed_observation_types[0],
                    "statement": "",
                    "source_ids": [task.target_source_ids[0]],
                    "confidence": "high",
                    "provenance": "explicit",
                    "uncertainty": None,
                    "extracted_numbers": [],
                    "proposed_links": [],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="returned no valid observation cards"):
        observation_cards_from_model_envelope(
            envelope,
            paper_id="p_test",
            task=task,
            allowed_source_ids=set(task.target_source_ids),
        )


def test_model_observation_cards_require_covered_outputs():
    task = build_initial_reading_plan(sample_dom()).tasks[0]
    envelope = ArtifactEnvelope(
        artifact_type="observation_cards",
        producer="fake",
        data={
            "cards": [
                {
                    "observation_type": task.allowed_observation_types[0],
                    "statement": "A source-bound card without coverage metadata.",
                    "source_ids": [task.target_source_ids[0]],
                    "confidence": "high",
                    "provenance": "explicit",
                    "uncertainty": None,
                    "extracted_numbers": [],
                    "proposed_links": [],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="must declare covered_outputs"):
        observation_cards_from_model_envelope(
            envelope,
            paper_id="p_test",
            task=task,
            allowed_source_ids=set(task.target_source_ids),
        )


def test_model_observation_cards_reject_covered_outputs_outside_required_outputs():
    task = build_initial_reading_plan(sample_dom()).tasks[0]
    envelope = ArtifactEnvelope(
        artifact_type="observation_cards",
        producer="fake",
        data={
            "cards": [
                {
                    "observation_type": task.allowed_observation_types[0],
                    "statement": "A card that claims to cover another task output.",
                    "source_ids": [task.target_source_ids[0]],
                    "confidence": "high",
                    "provenance": "explicit",
                    "uncertainty": None,
                    "covered_outputs": ["claim"],
                    "extracted_numbers": [],
                    "proposed_links": [],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="covered_outputs outside required_outputs: claim"):
        observation_cards_from_model_envelope(
            envelope,
            paper_id="p_test",
            task=task,
            allowed_source_ids=set(task.target_source_ids),
        )


def test_model_observation_cards_reject_incomplete_required_output_coverage():
    task = build_initial_reading_plan(sample_dom()).tasks[0]
    assert set(task.required_outputs) >= {"problem", "motivation", "scope"}
    envelope = ArtifactEnvelope(
        artifact_type="observation_cards",
        producer="fake",
        data={
            "cards": [
                {
                    "observation_type": task.allowed_observation_types[0],
                    "statement": "The paper states a problem.",
                    "source_ids": [task.target_source_ids[0]],
                    "confidence": "high",
                    "provenance": "explicit",
                    "uncertainty": None,
                    "covered_outputs": ["problem"],
                    "extracted_numbers": [],
                    "proposed_links": [],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="did not cover required_outputs: motivation, scope"):
        observation_cards_from_model_envelope(
            envelope,
            paper_id="p_test",
            task=task,
            allowed_source_ids=set(task.target_source_ids),
        )


def test_model_observation_cards_reject_disallowed_observation_type():
    task = next(
        task
        for task in build_initial_reading_plan(sample_dom()).tasks
        if task.task_type == ReadingTaskType.METHOD_MECHANISM
    )
    envelope = ArtifactEnvelope(
        artifact_type="observation_cards",
        producer="fake",
        data={
            "cards": [
                {
                    "observation_type": "claim",
                    "statement": "A claim emitted from a mechanism task.",
                    "source_ids": [task.target_source_ids[0]],
                    "confidence": "high",
                    "provenance": "explicit",
                    "uncertainty": None,
                    "extracted_numbers": [],
                    "proposed_links": [],
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="disallowed observation_type=claim"):
        observation_cards_from_model_envelope(
            envelope,
            paper_id="p_test",
            task=task,
            allowed_source_ids=set(task.target_source_ids),
        )


def test_model_observation_cards_reject_source_ids_outside_task_evidence_pack():
    dom = sample_dom()
    task = build_initial_reading_plan(dom).tasks[0]
    outside_source_id = next(
        source_id for source_id in dom.source_ids() if source_id not in set(task.target_source_ids)
    )
    envelope = ArtifactEnvelope(
        artifact_type="observation_cards",
        producer="fake",
        data={
            "cards": [
                {
                    "observation_type": task.allowed_observation_types[0],
                    "statement": "A card that cites another task source.",
                    "source_ids": [task.target_source_ids[0], outside_source_id],
                    "confidence": "high",
                    "provenance": "explicit",
                    "uncertainty": None,
                    "extracted_numbers": [],
                    "proposed_links": [],
                }
            ]
        },
    )

    assert outside_source_id in dom.source_ids()
    with pytest.raises(ValueError, match="outside this task evidence pack"):
        observation_cards_from_model_envelope(
            envelope,
            paper_id="p_test",
            task=task,
            allowed_source_ids=set(task.target_source_ids),
        )


def test_model_observation_cards_reject_background_provenance():
    dom = sample_dom()
    task = build_initial_reading_plan(dom).tasks[0]
    observation_type = task.allowed_observation_types[0]
    envelope = ArtifactEnvelope(
        artifact_type="observation_cards",
        producer="fake",
        data={
            "cards": [
                {
                    "observation_type": observation_type,
                    "statement": "A card that tries to store background knowledge as a paper fact.",
                    "source_ids": [task.target_source_ids[0]],
                    "confidence": "high",
                    "provenance": "background",
                    "uncertainty": None,
                    "covered_outputs": task.required_outputs,
                    "extracted_numbers": [],
                    "proposed_links": [],
                }
            ]
        },
    )
    provenance_schema = OBSERVATION_CARDS_SCHEMA["properties"]["data"]["properties"]["cards"][
        "items"
    ]["properties"]["provenance"]

    assert provenance_schema["enum"] == ["explicit", "inferred"]
    with pytest.raises(ValueError, match="disallowed provenance=background"):
        observation_cards_from_model_envelope(
            envelope,
            paper_id="p_test",
            task=task,
            allowed_source_ids=set(task.target_source_ids),
        )


def test_observation_card_model_rejects_background_provenance():
    dom = sample_dom()

    with pytest.raises(ValueError):
        ObservationCard(
            observation_id="obs_background",
            paper_id="p_test",
            task_id="task",
            observation_type=ObservationType.CLAIM,
            statement="Background knowledge cannot be stored as a paper observation.",
            source_ids=[dom.spans[0].source_id],
            provenance="background",
        )


def test_stage08_refreshes_core_v2_audits_without_legacy_paper_cards(tmp_path):
    output_dir = tmp_path / "out"
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    pipeline = PaperLensWorkflow(
        input_dir=input_dir,
        output_dir=output_dir,
        config=CoreConfig(offline_debug=True),
        events=EventWriter(
            "run_test",
            output_dir / ".paperlens" / "data" / "events.jsonl",
            output_dir / ".paperlens" / "data" / "errors.jsonl",
        ),
        control=ControlState(),
    )
    try:
        pipeline.prepare_output()
        paper = PaperRecord(
            paper_id="p_test",
            file_path="paper.pdf",
            file_hash="hash",
            canonical_title="Test Paper",
            page_count=1,
        )
        pipeline.papers = [paper]
        pipeline.db.upsert_paper(paper)
        write_core_v2_artifacts(
            data_dir=pipeline.data_dir,
            paper=paper,
            layout={
                "pages": [
                    {
                        "page_no": 1,
                        "text": "Abstract\n\nWe propose a block table method for serving.",
                        "section_candidates": [{"title": "Abstract", "level": 1}],
                    }
                ]
            },
        )

        pipeline.stage_08_evidence_verify()

        root = pipeline.data_dir / "core" / "v2" / "p_test"
        metrics = json.loads((root / "quality_metrics.v1.json").read_text(encoding="utf-8"))
        artifacts = pipeline.db.list_artifact_versions()
        state = pipeline.db.get_paper_state("p_test")
        assert metrics["producer"] == "paperlens_core_v2_audit_suite"
        assert any(item.artifact_type == "core_v2_audit_quality_metrics" for item in artifacts)
        assert state is not None
        assert state.current_stage == "stage_08_evidence_verify"
        assert "CORE_V2_DRAFT_WEAK" in state.side_statuses
    finally:
        pipeline.db.close()


def test_core_v2_qa_reads_claim_graph_without_final_markdown_report(tmp_path):
    output_dir = tmp_path / "out"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for faster serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            },
            {
                "page_no": 2,
                "text": "Method\n\nThe block table mechanism organizes serving state.",
                "section_candidates": [{"title": "Method", "level": 1}],
            },
        ]
    }
    write_core_v2_artifacts(
        data_dir=output_dir / ".paperlens" / "data",
        paper=paper,
        layout=layout,
    )
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    claim_span = next(span for span in dom.spans if "block table method" in span.text)
    mechanism_span = next(span for span in dom.spans if "organizes serving state" in span.text)
    reading_plan = reading_plan_subset(
        dom,
        ReadingTaskType.CLAIM_INVENTORY,
        ReadingTaskType.METHOD_MECHANISM,
    )
    claim_task = reading_task(reading_plan, ReadingTaskType.CLAIM_INVENTORY)
    mechanism_task = reading_task(reading_plan, ReadingTaskType.METHOD_MECHANISM)
    write_core_v2_from_observation_log(
        data_dir=output_dir / ".paperlens" / "data",
        paper=paper,
        dom=dom,
        reading_plan=reading_plan,
        observation_log=ObservationLog(paper_id="p_test")
        .append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id=claim_task.task_id,
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[claim_span.source_id],
                covered_outputs=claim_task.required_outputs,
            )
        )
        .append(
            ObservationCard(
                observation_id="obs_mechanism",
                paper_id="p_test",
                task_id=mechanism_task.task_id,
                observation_type=ObservationType.MECHANISM,
                statement="The block table mechanism organizes serving state.",
                source_ids=[mechanism_span.source_id],
                covered_outputs=mechanism_task.required_outputs,
                proposed_links=[
                    {"source_id": "obs_mechanism", "target_id": "obs_claim", "kind": "explains"}
                ],
            )
        ),
        producer="unit_test",
    )

    context = load_core_v2_qa_context(
        output_dir=output_dir,
        paper_id="p_test",
        question="什么是 block table method？",
        question_type="clarification",
    )
    answer = answer_question(
        output_dir=output_dir,
        config=CoreConfig(offline_debug=True),
        paper_id="p_test",
        question="什么是 block table method？",
    )
    qa_trace = [
        json.loads(line)
        for line in (output_dir / ".paperlens" / "data" / "qa_trace.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    assert context["matches"]
    assert context["question_type"] == "clarification"
    assert context["retrieval_intent"]["policy"] == "classify_question_before_claim_graph_retrieval"
    assert "concept" in context["retrieval_intent"]["preferred_node_kinds"]
    assert context["matches"][0]["source_ids"][0].startswith("span:p_test:")
    assert context["matches"][0]["relationships"][0]["kind"] == "explains"
    assert context["matches"][0]["relationships"][0]["source_ids"]
    assert answer["cited_source_ids"][0].startswith("span:p_test:")
    assert set(answer["cited_pages"]) == {1, 2}
    assert "ClaimGraph" in answer["answer_markdown"]
    assert "relation:" in answer["answer_markdown"]
    assert answer["source_attribution"]["paper_claims"]
    assert qa_trace[-1]["core_v2_question_type"] == "clarification"
    assert qa_trace[-1]["core_v2_retrieval_policy"] == "claim_graph_nodes_with_paper_dom_source_ids"
    assert (
        qa_trace[-1]["core_v2_retrieval_intent"]["policy"]
        == "classify_question_before_claim_graph_retrieval"
    )


def test_core_v2_qa_memory_view_promotes_reviewed_claim_graph_context(tmp_path):
    core_context = {
        "retrieval_policy": "claim_graph_nodes_with_paper_dom_source_ids",
        "answer_source_policy": "Use graph node IDs and PaperDOM source IDs.",
        "quality": {"publish_status": PublishStatus.REVIEWED},
        "matches": [
            {
                "node_id": "claim:obs_claim",
                "kind": "claim",
                "label": "The paper proposes a block table method.",
                "confidence": "high",
                "provenance": "explicit",
                "source_ids": ["span:p_test:p1:1"],
                "evidence_spans": [
                    {
                        "source_id": "span:p_test:p1:1",
                        "kind": "paragraph",
                        "page_no": 1,
                        "text": "We propose a block table method for serving.",
                    }
                ],
            }
        ],
    }
    legacy_memory = {
        "schema_version": "paper_memory.v3",
        "paper_id": "p_test",
        "claims": [{"id": "legacy", "text": "Legacy claim should not be primary."}],
    }

    memory = qa_memory_context(
        paper_id="p_test",
        paper_memory_v3=legacy_memory,
        core_v2_context=core_context,
    )
    prompt = build_ask_prompt(
        report_path=tmp_path / "papers" / "p_test.md",
        paper_id="p_test",
        question="block table 是什么？",
        paper_memory_v3=memory,
        pages=[{"page_no": 1, "text": "page text", "captions": [], "visual_notes": []}],
        question_type="mechanism",
        core_v2_context=core_context,
    )

    assert core_v2_context_priority(core_context) == "primary_reviewed_claim_graph"
    assert memory["schema_version"] == "paperlens_core_v2_qa_memory_view.v1"
    assert memory["reading_context"]["source_of_truth"] == "core_v2_claim_graph"
    assert memory["claims"][0]["id"] == "claim:obs_claim"
    assert memory["claims"][0]["evidence_refs"] == ["span:p_test:p1:1"]
    assert memory["evidence"][0]["id"] == "span:p_test:p1:1"
    assert "core_v2_context_priority: primary_reviewed_claim_graph" in prompt
    assert "memory_fallback_policy:" in prompt
    assert "Legacy claim should not be primary" not in prompt


def test_core_v2_qa_empty_reviewed_graph_match_does_not_fallback_to_legacy_memory(tmp_path):
    core_context = {
        "retrieval_policy": "claim_graph_nodes_with_paper_dom_source_ids",
        "answer_source_policy": "Use graph node IDs and PaperDOM source IDs.",
        "quality": {"publish_status": PublishStatus.REVIEWED},
        "matches": [],
    }
    legacy_memory = {
        "schema_version": "paper_memory.v3",
        "paper_id": "p_test",
        "claims": [{"id": "legacy", "text": "Legacy claim should not answer this question."}],
    }

    memory = qa_memory_context(
        paper_id="p_test",
        paper_memory_v3=legacy_memory,
        core_v2_context=core_context,
    )
    prompt = build_ask_prompt(
        report_path=tmp_path / "papers" / "p_test.md",
        paper_id="p_test",
        question="unmatched question?",
        paper_memory_v3=memory,
        pages=[{"page_no": 1, "text": "page text", "captions": [], "visual_notes": []}],
        question_type="orientation",
        core_v2_context=core_context,
    )
    offline = offline_qa_answer(
        paper_id="p_test",
        report_path=tmp_path / "papers" / "p_test.md",
        question_type="orientation",
        pages=[{"page_no": 1, "text": "page text"}],
        core_v2_context=core_context,
    )

    assert core_v2_context_priority(core_context) == "primary_reviewed_claim_graph"
    assert memory["schema_version"] == "paperlens_core_v2_qa_memory_view.v1"
    assert memory["claims"] == []
    assert "Legacy claim should not answer" not in json.dumps(memory, ensure_ascii=False)
    assert "no_legacy_paper_claim_fallback" in prompt
    assert "Legacy claim should not answer" not in prompt
    assert "没有为当前问题返回可引用" in offline["answer_markdown"]
    assert "p_test.md" not in offline["answer_markdown"]
    assert offline["cited_source_ids"] == []
    assert offline["source_attribution"]["paper_claims"] == []


def test_core_v2_qa_context_returns_no_matches_for_unrelated_question(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            }
        ]
    }
    write_core_v2_artifacts(data_dir=data_dir, paper=paper, layout=layout)
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    claim_span = next(span for span in dom.spans if "block table method" in span.text)
    reading_plan = reading_plan_subset(dom, ReadingTaskType.CLAIM_INVENTORY)
    claim_task = reading_task(reading_plan, ReadingTaskType.CLAIM_INVENTORY)
    write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=reading_plan,
        observation_log=ObservationLog(paper_id="p_test").append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id=claim_task.task_id,
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[claim_span.source_id],
                covered_outputs=claim_task.required_outputs,
            )
        ),
        producer="unit_test",
    )

    context = load_core_v2_qa_context(
        output_dir=output_dir,
        paper_id="p_test",
        question="完全无关的量子猫问题",
        question_type="orientation",
    )
    offline = offline_qa_answer(
        paper_id="p_test",
        report_path=tmp_path / "papers" / "p_test.md",
        question_type="orientation",
        pages=[{"page_no": 1, "text": "page text"}],
        core_v2_context=context,
    )

    assert context["retrieval_policy"] == "claim_graph_nodes_with_paper_dom_source_ids"
    assert context["matches"] == []
    assert offline["cited_source_ids"] == []
    assert offline["source_attribution"]["paper_claims"] == []
    assert "没有为当前问题返回可引用" in offline["answer_markdown"]
    assert "p_test.md" not in offline["answer_markdown"]


def test_core_v2_qa_context_reaudits_stale_reviewed_graph_with_current_reading_gaps(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            }
        ]
    }
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    claim_span = next(span for span in dom.spans if "block table method" in span.text)
    narrow_plan = reading_plan_subset(dom, ReadingTaskType.CLAIM_INVENTORY)
    full_plan = build_initial_reading_plan(dom)
    claim_task = reading_task(narrow_plan, ReadingTaskType.CLAIM_INVENTORY)
    write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=narrow_plan,
        observation_log=ObservationLog(paper_id="p_test").append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id=claim_task.task_id,
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[claim_span.source_id],
                covered_outputs=claim_task.required_outputs,
            )
        ),
        producer="unit_test",
    )
    root = data_dir / "core" / "v2" / "p_test"
    write_typed_artifact(
        root / "reading_plan.v1.json",
        artifact_type="reading_plan",
        data=full_plan.model_dump(),
        producer="unit_test_stale_plan",
        metadata={"paper_id": paper.paper_id},
    )

    context = load_core_v2_qa_context(
        output_dir=output_dir,
        paper_id="p_test",
        question="block table method 是什么？",
    )

    assert context["retrieval_policy"] == "not_reviewed_by_current_graph_audit"
    assert context["quality"]["artifact_publish_status"] == PublishStatus.REVIEWED
    assert context["quality"]["current_audit_publish_status"] == PublishStatus.DRAFT_WEAK
    assert "missing_reading_required_output" in context["quality"]["current_audit_issue_codes"]
    assert context["matches"] == []


def test_core_v2_qa_grounding_filters_unknown_source_ids_and_claims():
    core_context = {
        "retrieval_policy": "claim_graph_nodes_with_paper_dom_source_ids",
        "quality": {"publish_status": PublishStatus.REVIEWED},
        "matches": [
            {
                "node_id": "claim:obs_claim",
                "kind": "claim",
                "label": "The paper proposes a block table method.",
                "source_ids": ["span:p_test:p1:1"],
                "evidence_spans": [
                    {
                        "source_id": "span:p_test:p1:1",
                        "kind": "paragraph",
                        "page_no": 1,
                        "text": "We propose a block table method for serving.",
                    }
                ],
            }
        ],
    }
    answer = {
        "answer_markdown": "The paper proposes a block table method and proves 99% accuracy.",
        "cited_pages": [],
        "cited_source_ids": ["span:p_test:p1:1", "span:p_other:p9:9"],
        "confidence": "high",
        "source_attribution": {
            "paper_claims": [
                "The paper proposes a block table method.",
                "The paper proves 99% accuracy.",
            ],
            "paperlens_inferences": [],
            "background_context": [],
            "evidence_limits": [],
        },
    }

    grounded = ground_qa_answer_in_core_v2_context(answer, core_context)

    assert grounded["cited_source_ids"] == ["span:p_test:p1:1"]
    assert grounded["cited_pages"] == [1]
    assert grounded["confidence"] == "medium"
    assert grounded["source_attribution"]["paper_claims"] == [
        "The paper proposes a block table method."
    ]
    main_answer = grounded["answer_markdown"].split("Evidence limits:")[0]
    assert "The paper proposes a block table method." in main_answer
    assert "99% accuracy" not in main_answer
    assert "Evidence limits:" in grounded["answer_markdown"]
    assert any(
        "Removed QA source IDs" in item
        for item in grounded["source_attribution"]["evidence_limits"]
    )
    assert any(
        "Removed model-declared paper claims" in item
        for item in grounded["source_attribution"]["evidence_limits"]
    )


def test_core_v2_qa_grounding_backfills_source_ids_for_supported_paper_claims():
    core_context = {
        "retrieval_policy": "claim_graph_nodes_with_paper_dom_source_ids",
        "quality": {"publish_status": PublishStatus.REVIEWED},
        "matches": [
            {
                "node_id": "claim:obs_claim",
                "kind": "claim",
                "label": "The paper proposes a block table method.",
                "source_ids": ["span:p_test:p1:1"],
                "evidence_spans": [
                    {
                        "source_id": "span:p_test:p1:1",
                        "kind": "paragraph",
                        "page_no": 1,
                        "text": "We propose a block table method for serving.",
                    }
                ],
            }
        ],
    }
    answer = {
        "answer_markdown": "The paper proposes a block table method.",
        "cited_pages": [],
        "cited_source_ids": [],
        "confidence": "high",
        "source_attribution": {
            "paper_claims": ["The paper proposes a block table method."],
            "paperlens_inferences": [],
            "background_context": [],
            "evidence_limits": [],
        },
    }

    grounded = ground_qa_answer_in_core_v2_context(answer, core_context)

    assert grounded["cited_source_ids"] == ["span:p_test:p1:1"]
    assert grounded["cited_pages"] == [1]
    assert grounded["confidence"] == "high"
    assert grounded["source_attribution"]["evidence_limits"] == []


def test_core_v2_qa_grounding_replaces_fully_unsupported_paper_answer():
    core_context = {
        "retrieval_policy": "claim_graph_nodes_with_paper_dom_source_ids",
        "quality": {"publish_status": PublishStatus.REVIEWED},
        "matches": [
            {
                "node_id": "claim:obs_claim",
                "kind": "claim",
                "label": "The paper proposes a block table method.",
                "source_ids": ["span:p_test:p1:1"],
                "evidence_spans": [
                    {
                        "source_id": "span:p_test:p1:1",
                        "kind": "paragraph",
                        "page_no": 1,
                        "text": "We propose a block table method for serving.",
                    }
                ],
            }
        ],
    }
    answer = {
        "answer_markdown": "The paper proves 99% accuracy.",
        "cited_pages": [],
        "cited_source_ids": [],
        "confidence": "high",
        "source_attribution": {
            "paper_claims": ["The paper proves 99% accuracy."],
            "paperlens_inferences": [],
            "background_context": [],
            "evidence_limits": [],
        },
    }

    grounded = ground_qa_answer_in_core_v2_context(answer, core_context)

    main_answer = grounded["answer_markdown"].split("Evidence limits:")[0]
    assert "Reviewed ClaimGraph context does not support" in main_answer
    assert "99% accuracy" not in main_answer
    assert grounded["source_attribution"]["paper_claims"] == []
    assert grounded["confidence"] == "medium"


def test_export_writes_core_graph_report_view_for_reviewed_core_artifacts(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    (output_dir / "papers").mkdir(parents=True)
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    decision = ClassificationDecision(
        paper_id="p_test",
        class_label="A",
        confidence=0.9,
        false_negative_risk=0.1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for faster serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            }
        ]
    }
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    source_id = next(span.source_id for span in dom.spans if "block table method" in span.text)
    reading_plan = reading_plan_subset(dom, ReadingTaskType.CLAIM_INVENTORY)
    claim_task = reading_task(reading_plan, ReadingTaskType.CLAIM_INVENTORY)
    write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=reading_plan,
        observation_log=ObservationLog(paper_id="p_test").append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id=claim_task.task_id,
                observation_type=ObservationType.CLAIM,
                statement="We propose a block table method for faster serving.",
                source_ids=[source_id],
                covered_outputs=claim_task.required_outputs,
            )
        ),
        producer="unit_test",
    )

    written = write_final_report_bundle(
        output_dir=output_dir,
        data_dir=data_dir,
        evidence_dir=output_dir / ".paperlens",
        client=None,
        record_usage=lambda _stage, _usage: None,
        record_agent_run=lambda _run: None,
        stage="stage_15_export",
        papers=[paper],
        skim_cards=[],
        decisions=[decision],
        paper_cards=[],
        review_items=[],
        budget={},
        config={"offline_debug": True},
        topic=None,
        idea=None,
        cache_dir=output_dir / ".paperlens" / "cache",
    )

    graph_report = output_dir / "papers" / "core_graph" / "p_test_test_paper.core_graph.md"
    index = (output_dir / "PaperLens.md").read_text(encoding="utf-8")
    markdown = graph_report.read_text(encoding="utf-8")
    assert graph_report in written
    assert "[事实图报告](./papers/core_graph/p_test_test_paper.core_graph.md)" in index
    assert "ClaimGraph nodes: `claim:obs_claim`" in markdown
    assert f"Evidence nodes: `evidence:{source_id}`" in markdown
    assert f"PaperDOM sources: `{source_id}`" in markdown


def test_report_memory_context_prefers_reviewed_core_memory_view(tmp_path):
    data_dir = tmp_path / ".paperlens" / "data"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for faster serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            }
        ]
    }
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    source_id = next(span.source_id for span in dom.spans if "block table method" in span.text)
    reading_plan = reading_plan_subset(dom, ReadingTaskType.CLAIM_INVENTORY)
    claim_task = reading_task(reading_plan, ReadingTaskType.CLAIM_INVENTORY)
    write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=reading_plan,
        observation_log=ObservationLog(paper_id="p_test").append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id=claim_task.task_id,
                observation_type=ObservationType.CLAIM,
                statement="We propose a block table method for faster serving.",
                source_ids=[source_id],
                covered_outputs=claim_task.required_outputs,
            )
        ),
        producer="unit_test",
    )

    context = build_report_memory_context(
        data_dir=data_dir,
        paper_id="p_test",
        paper_memory_v3={
            "schema_version": "paper_memory.v3",
            "paper_id": "p_test",
            "claims": [{"id": "legacy", "text": "Legacy claim should be fallback only."}],
        },
    )
    compact = compact_paper_memory_for_report(context)

    assert context["schema_version"] == "paperlens.report_memory_context.v1"
    assert context["source_of_truth"] == "core_v2_paper_memory_view"
    assert compact["source_of_truth"] == "core_v2_paper_memory_view"
    assert compact["core_memory_view"]["fact_nodes"][0]["node_id"] == "claim:obs_claim"
    assert compact["core_memory_view"]["fact_nodes"][0]["source_ids"] == [source_id]
    assert compact["core_memory_view"]["fact_nodes"][0]["pages"] == [1]
    assert compact["core_memory_view"]["fact_nodes"][0]["audit_status"] == PublishStatus.REVIEWED
    assert compact["core_memory_view"]["audit_issues"] == []
    assert report_focus_pages(context, skim=None, card=None) == [1]
    assert any(
        "block table method" in query
        for query in report_focus_queries(context, paper=paper, skim=None, card=None)
    )


def test_agent_memory_tools_search_core_memory_view(tmp_path):
    source_id = "span:p_test:p1:1"
    memory = {
        "schema_version": "paperlens.report_memory_context.v1",
        "core_memory_view": {
            "schema_version": "paper_memory.view.v1",
            "paper_id": "p_test",
            "fact_nodes": [
                {
                    "node_id": "claim:obs_claim",
                    "kind": "claim",
                    "label": "The paper proposes a block table method.",
                    "evidence_ids": [f"evidence:{source_id}"],
                    "source_ids": [source_id],
                    "pages": [1],
                }
            ],
            "evidence_sources": {
                source_id: {
                    "source_id": source_id,
                    "kind": "paragraph",
                    "page_no": 1,
                    "excerpt": "We propose a block table method for faster serving.",
                }
            },
            "evaluation_matrix": [],
            "relationship_edges": [],
        },
    }
    registry = PaperToolRegistry(
        runtime=PaperLensRuntime(artifacts=[]),
        paper_id="p_test",
        memory=memory,
    )

    search = registry._memory_search("block table")
    claim = registry._memory_get_claim("claim:obs_claim")
    evidence = registry._evidence_lookup([source_id])

    assert search["results"][0]["section"] == "core.fact_nodes"
    assert claim["results"][0]["node_id"] == "claim:obs_claim"
    assert evidence["results"][0]["source_id"] == source_id


def test_core_v2_qa_context_does_not_use_blocked_claim_graph(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    write_core_v2_artifacts(
        data_dir=data_dir,
        paper=paper,
        layout={
            "pages": [
                {
                    "page_no": 1,
                    "text": "Abstract\n\nWe propose a block table method for serving.",
                    "section_candidates": [{"title": "Abstract", "level": 1}],
                }
            ]
        },
    )
    graph_path = data_dir / "core" / "v2" / "p_test" / "claim_graph.v1.json"
    graph_envelope = json.loads(graph_path.read_text(encoding="utf-8"))
    evidence_node = next(
        node for node in graph_envelope["data"]["nodes"].values() if node["kind"] == "evidence"
    )
    evidence_node["payload"]["source_id"] = "span:p_test:missing"
    graph_path.write_text(json.dumps(graph_envelope, ensure_ascii=False), encoding="utf-8")
    refresh_core_v2_audit_artifacts(data_dir=data_dir, paper=paper)

    context = load_core_v2_qa_context(
        output_dir=output_dir,
        paper_id="p_test",
        question="block table method 是什么？",
    )

    assert context["retrieval_policy"] == "blocked_by_core_v2_audit"
    assert context["quality"]["publish_status"] == PublishStatus.BLOCKED
    assert context["matches"] == []


def test_core_v2_qa_context_does_not_use_draft_weak_bootstrap_graph(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    write_core_v2_artifacts(
        data_dir=data_dir,
        paper=paper,
        layout={
            "pages": [
                {
                    "page_no": 1,
                    "text": "Abstract\n\nWe propose a block table method for serving.",
                    "section_candidates": [{"title": "Abstract", "level": 1}],
                }
            ]
        },
    )

    context = load_core_v2_qa_context(
        output_dir=output_dir,
        paper_id="p_test",
        question="block table method 是什么？",
    )

    assert context["retrieval_policy"] == "not_reviewed_by_core_v2_audit"
    assert context["quality"]["publish_status"] == PublishStatus.DRAFT_WEAK
    assert context["matches"] == []


def test_core_v2_qa_context_requires_quality_metrics_gate(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            }
        ]
    }
    write_core_v2_artifacts(data_dir=data_dir, paper=paper, layout=layout)
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    claim_span = next(span for span in dom.spans if "block table method" in span.text)
    reading_plan = reading_plan_subset(dom, ReadingTaskType.CLAIM_INVENTORY)
    claim_task = reading_task(reading_plan, ReadingTaskType.CLAIM_INVENTORY)
    write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=reading_plan,
        observation_log=ObservationLog(paper_id="p_test").append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id=claim_task.task_id,
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[claim_span.source_id],
                covered_outputs=claim_task.required_outputs,
            )
        ),
        producer="unit_test",
    )
    (data_dir / "core" / "v2" / "p_test" / "quality_metrics.v1.json").unlink()

    context = load_core_v2_qa_context(
        output_dir=output_dir,
        paper_id="p_test",
        question="block table method 是什么？",
    )

    assert context["retrieval_policy"] == "missing_core_v2_quality_metrics"
    assert context["quality"]["publish_status"] is None
    assert context["matches"] == []


def test_library_rebuild_indexes_core_v2_claim_graph_without_memory_v3(tmp_path):
    output_dir = tmp_path / "out"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=3,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for faster serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            },
            {
                "page_no": 2,
                "text": "Method\n\nThe block table mechanism organizes serving state.",
                "section_candidates": [{"title": "Method", "level": 1}],
            },
            {
                "page_no": 3,
                "text": "Evaluation\n\nThe method improves latency by 27% on Dataset-A.",
                "section_candidates": [{"title": "Evaluation", "level": 1}],
            },
        ]
    }
    write_core_v2_artifacts(
        data_dir=output_dir / ".paperlens" / "data",
        paper=paper,
        layout=layout,
    )
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    claim_span = next(span for span in dom.spans if "block table method" in span.text)
    mechanism_span = next(span for span in dom.spans if "organizes serving state" in span.text)
    result_span = next(span for span in dom.spans if "27%" in span.text)
    reading_plan = reading_plan_subset(
        dom,
        ReadingTaskType.CLAIM_INVENTORY,
        ReadingTaskType.METHOD_MECHANISM,
        ReadingTaskType.RESULT_EXTRACTION,
    )
    claim_task = reading_task(reading_plan, ReadingTaskType.CLAIM_INVENTORY)
    mechanism_task = reading_task(reading_plan, ReadingTaskType.METHOD_MECHANISM)
    result_task = reading_task(reading_plan, ReadingTaskType.RESULT_EXTRACTION)
    write_core_v2_from_observation_log(
        data_dir=output_dir / ".paperlens" / "data",
        paper=paper,
        dom=dom,
        reading_plan=reading_plan,
        observation_log=ObservationLog(paper_id="p_test")
        .append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id=claim_task.task_id,
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[claim_span.source_id],
                covered_outputs=claim_task.required_outputs,
            )
        )
        .append(
            ObservationCard(
                observation_id="obs_mechanism",
                paper_id="p_test",
                task_id=mechanism_task.task_id,
                observation_type=ObservationType.MECHANISM,
                statement="The block table mechanism organizes serving state.",
                source_ids=[mechanism_span.source_id],
                covered_outputs=mechanism_task.required_outputs,
                proposed_links=[
                    {"source_id": "obs_mechanism", "target_id": "obs_claim", "kind": "explains"}
                ],
            )
        )
        .append(
            ObservationCard(
                observation_id="obs_result",
                paper_id="p_test",
                task_id=result_task.task_id,
                observation_type=ObservationType.RESULT,
                statement="The method improves latency by 27% on Dataset-A.",
                source_ids=[result_span.source_id],
                covered_outputs=result_task.required_outputs,
                extracted_numbers=[{"text": "27%"}],
            )
        ),
        producer="unit_test",
    )

    written = rebuild_library_from_output(output_dir)
    records = read_library_records(output_dir)
    result = search_library(output_dir=output_dir, query="block table faster serving", limit=3)
    index = json.loads(
        (output_dir / ".paperlens" / "library" / "index" / "search_index.json").read_text(
            encoding="utf-8"
        )
    )

    assert any(path.name == "library_records.jsonl" for path in written)
    assert records[0]["paper_id"] == "p_test"
    assert records[0]["graph_summary"]["schema_version"] == "paperlens.graph_library_summary.v1"
    assert records[0]["graph_summary"]["relations"][0]["kind"] == "explains"
    assert records[0]["graph_summary"]["relations"][0]["source_id"] == "mechanism:obs_mechanism"
    assert records[0]["graph_summary"]["evaluation_datasets"] == ["Dataset-A"]
    assert "latency" in records[0]["graph_summary"]["evaluation_metrics"]
    assert "27%" in records[0]["graph_summary"]["evaluation_metrics"]
    assert "3" not in records[0]["graph_summary"]["evaluation_metrics"]
    dataset_mention = records[0]["graph_summary"]["evaluation_dataset_mentions"][0]
    metric_mentions = records[0]["graph_summary"]["evaluation_metric_mentions"]
    assert dataset_mention["node_ids"] == ["result:obs_result"]
    assert dataset_mention["source_ids"] == [result_span.source_id]
    assert dataset_mention["pages"] == [3]
    assert any(item["term"] == "27%" and item["source_ids"] == [result_span.source_id] for item in metric_mentions)
    assert records[0]["memory"]["claims"]
    assert records[0]["provenance"]["core_v2"]["source_ids"]
    assert records[0]["quality"]["graph_publish_status"] == PublishStatus.REVIEWED
    assert result["matches"][0]["paper"]["paper_id"] == "p_test"
    assert index["records"][0]["graph"]["node_counts"]["claim"] >= 1
    assert index["records"][0]["graph"]["evaluation_dataset_mentions"][0]["source_ids"] == [
        result_span.source_id
    ]
    assert doctor_library(output_dir)["status"] == "PASS"


def test_library_rebuild_does_not_index_blocked_core_v2_claim_graph(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    write_core_v2_artifacts(
        data_dir=data_dir,
        paper=paper,
        layout={
            "pages": [
                {
                    "page_no": 1,
                    "text": "Abstract\n\nWe propose a block table method for serving.",
                    "section_candidates": [{"title": "Abstract", "level": 1}],
                }
            ]
        },
    )
    graph_path = data_dir / "core" / "v2" / "p_test" / "claim_graph.v1.json"
    graph_envelope = json.loads(graph_path.read_text(encoding="utf-8"))
    evidence_node = next(
        node for node in graph_envelope["data"]["nodes"].values() if node["kind"] == "evidence"
    )
    evidence_node["payload"]["source_id"] = "span:p_test:missing"
    graph_path.write_text(json.dumps(graph_envelope, ensure_ascii=False), encoding="utf-8")
    refresh_core_v2_audit_artifacts(data_dir=data_dir, paper=paper)

    rebuild_library_from_output(output_dir)
    records = read_library_records(output_dir)
    result = search_library(output_dir=output_dir, query="block table serving", limit=3)
    index = json.loads(
        (output_dir / ".paperlens" / "library" / "index" / "search_index.json").read_text(
            encoding="utf-8"
        )
    )

    assert records[0]["quality"]["graph_publish_status"] == PublishStatus.BLOCKED
    assert records[0]["graph_summary"]["graph_access"] == "blocked_by_core_v2_audit"
    assert records[0]["graph_summary"]["claim_nodes"] == []
    assert records[0]["memory"]["claims"] == []
    assert result["matches"] == []
    assert not {"block", "table", "serving"} & set(index["records"][0]["tokens"])


def test_library_rebuild_does_not_index_stale_reviewed_graph_with_current_audit_errors(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            }
        ]
    }
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    claim_span = next(span for span in dom.spans if "block table method" in span.text)
    reading_plan = reading_plan_subset(dom, ReadingTaskType.CLAIM_INVENTORY)
    claim_task = reading_task(reading_plan, ReadingTaskType.CLAIM_INVENTORY)
    write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=reading_plan,
        observation_log=ObservationLog(paper_id="p_test").append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id=claim_task.task_id,
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[claim_span.source_id],
                covered_outputs=claim_task.required_outputs,
            )
        ),
        producer="unit_test",
    )
    graph_path = data_dir / "core" / "v2" / "p_test" / "claim_graph.v1.json"
    graph_envelope = json.loads(graph_path.read_text(encoding="utf-8"))
    evidence_node = next(
        node for node in graph_envelope["data"]["nodes"].values() if node["kind"] == "evidence"
    )
    evidence_node["payload"]["source_id"] = "span:p_test:missing"
    graph_path.write_text(json.dumps(graph_envelope, ensure_ascii=False), encoding="utf-8")

    rebuild_library_from_output(output_dir)
    records = read_library_records(output_dir)
    result = search_library(output_dir=output_dir, query="block table serving", limit=3)

    graph_summary = records[0]["graph_summary"]
    assert graph_summary["quality"]["artifact_publish_status"] == PublishStatus.REVIEWED
    assert graph_summary["quality"]["current_audit_publish_status"] == PublishStatus.BLOCKED
    assert graph_summary["quality"]["publish_status"] == PublishStatus.BLOCKED
    assert graph_summary["quality"]["current_audit_error_count"] >= 1
    assert "missing_dom_source" in graph_summary["quality"]["current_audit_issue_codes"]
    assert graph_summary["graph_access"] == "blocked_by_current_graph_audit"
    assert graph_summary["claim_nodes"] == []
    assert records[0]["memory"]["claims"] == []
    assert result["matches"] == []


def test_library_rebuild_does_not_index_stale_reviewed_graph_with_current_reading_gaps(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            }
        ]
    }
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    claim_span = next(span for span in dom.spans if "block table method" in span.text)
    narrow_plan = reading_plan_subset(dom, ReadingTaskType.CLAIM_INVENTORY)
    full_plan = build_initial_reading_plan(dom)
    claim_task = reading_task(narrow_plan, ReadingTaskType.CLAIM_INVENTORY)
    write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=narrow_plan,
        observation_log=ObservationLog(paper_id="p_test").append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id=claim_task.task_id,
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[claim_span.source_id],
                covered_outputs=claim_task.required_outputs,
            )
        ),
        producer="unit_test",
    )
    root = data_dir / "core" / "v2" / "p_test"
    write_typed_artifact(
        root / "reading_plan.v1.json",
        artifact_type="reading_plan",
        data=full_plan.model_dump(),
        producer="unit_test_stale_plan",
        metadata={"paper_id": paper.paper_id},
    )

    rebuild_library_from_output(output_dir)
    records = read_library_records(output_dir)
    result = search_library(output_dir=output_dir, query="block table serving", limit=3)

    graph_summary = records[0]["graph_summary"]
    assert graph_summary["quality"]["artifact_publish_status"] == PublishStatus.REVIEWED
    assert graph_summary["quality"]["current_audit_publish_status"] == PublishStatus.DRAFT_WEAK
    assert graph_summary["quality"]["publish_status"] == PublishStatus.DRAFT_WEAK
    assert "missing_reading_required_output" in graph_summary["quality"]["current_audit_issue_codes"]
    assert graph_summary["graph_access"] == "not_reviewed_by_current_graph_audit"
    assert graph_summary["claim_nodes"] == []
    assert records[0]["memory"]["claims"] == []
    assert result["matches"] == []


def test_library_rebuild_does_not_index_draft_weak_bootstrap_graph(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    write_core_v2_artifacts(
        data_dir=data_dir,
        paper=paper,
        layout={
            "pages": [
                {
                    "page_no": 1,
                    "text": "Abstract\n\nWe propose a block table method for serving.",
                    "section_candidates": [{"title": "Abstract", "level": 1}],
                }
            ]
        },
    )

    rebuild_library_from_output(output_dir)
    records = read_library_records(output_dir)
    result = search_library(output_dir=output_dir, query="block table serving", limit=3)

    assert records[0]["quality"]["graph_publish_status"] == PublishStatus.DRAFT_WEAK
    assert records[0]["graph_summary"]["graph_access"] == "not_reviewed_by_core_v2_audit"
    assert records[0]["graph_summary"]["claim_nodes"] == []
    assert records[0]["memory"]["claims"] == []
    assert result["matches"] == []


def test_library_rebuild_requires_quality_metrics_gate_for_core_v2_graph(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            }
        ]
    }
    write_core_v2_artifacts(data_dir=data_dir, paper=paper, layout=layout)
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    claim_span = next(span for span in dom.spans if "block table method" in span.text)
    reading_plan = reading_plan_subset(dom, ReadingTaskType.CLAIM_INVENTORY)
    claim_task = reading_task(reading_plan, ReadingTaskType.CLAIM_INVENTORY)
    write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=reading_plan,
        observation_log=ObservationLog(paper_id="p_test").append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id=claim_task.task_id,
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[claim_span.source_id],
                covered_outputs=claim_task.required_outputs,
            )
        ),
        producer="unit_test",
    )
    (data_dir / "core" / "v2" / "p_test" / "quality_metrics.v1.json").unlink()

    rebuild_library_from_output(output_dir)
    records = read_library_records(output_dir)
    result = search_library(output_dir=output_dir, query="block table serving", limit=3)

    assert records[0]["graph_summary"]["graph_access"] == "missing_core_v2_quality_metrics"
    assert records[0]["graph_summary"]["quality"]["publish_status"] is None
    assert (
        records[0]["graph_summary"]["quality"]["memory_report_readiness"] == PublishStatus.REVIEWED
    )
    assert records[0]["memory"]["claims"] == []
    assert result["matches"] == []


def test_core_quality_snapshot_tracks_structural_and_qa_metrics(tmp_path):
    output_dir = tmp_path / "out"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=2,
    )
    write_core_v2_artifacts(
        data_dir=output_dir / ".paperlens" / "data",
        paper=paper,
        layout={
            "pages": [
                {
                    "page_no": 1,
                    "text": "Abstract\n\nWe propose a block table method for faster serving.",
                    "section_candidates": [{"title": "Abstract", "level": 1}],
                },
                {
                    "page_no": 2,
                    "text": "Evaluation\n\nThe method improves latency by 27% on Dataset-A.",
                    "section_candidates": [{"title": "Evaluation", "level": 1}],
                },
            ]
        },
    )
    graph_path = (
        output_dir / ".paperlens" / "data" / "core" / "v2" / "p_test" / "claim_graph.v1.json"
    )
    graph_envelope = json.loads(graph_path.read_text(encoding="utf-8"))
    fact_node = next(
        node for node in graph_envelope["data"]["nodes"].values() if node["kind"] != "evidence"
    )
    fact_node["label"] = f"{fact_node['label']} The reported latency improvement is 27%."
    graph_path.write_text(json.dumps(graph_envelope, ensure_ascii=False), encoding="utf-8")
    qa_trace = output_dir / ".paperlens" / "data" / "qa_trace.jsonl"
    qa_trace.parent.mkdir(parents=True, exist_ok=True)
    qa_trace.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "paper_id": "p_test",
                        "question": "27% 是哪来的？",
                        "cited_source_ids": ["span:p_test:p2:2"],
                        "selected_graph_nodes": ["result:obs"],
                        "cache_hit": False,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "paper_id": "p_test",
                        "question": "选中了图但没有可用引用",
                        "cited_source_ids": [],
                        "selected_graph_nodes": ["claim:obs"],
                        "cache_hit": False,
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "paper_id": "p_test",
                        "question": "离线缓存问题",
                        "cited_source_ids": [],
                        "selected_graph_nodes": [],
                        "cache_hit": True,
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    snapshot_path = write_core_quality_snapshot(output_dir)
    envelope = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot = envelope["data"]
    paper_snapshot = snapshot["papers"][0]

    assert envelope["artifact_type"] == "core_quality_snapshot"
    assert paper_snapshot["paper_id"] == "p_test"
    assert paper_snapshot["evidence_coverage"] == 1.0
    assert paper_snapshot["numeric_fact_node_count"] >= 1
    assert paper_snapshot["numeric_locatable_rate"] == 1.0
    assert "extracted_number_count" in paper_snapshot
    assert "extracted_number_not_located_count" in paper_snapshot
    assert "extracted_number_locatable_rate" in paper_snapshot
    assert paper_snapshot["reading_required_output_count"] == 14
    assert paper_snapshot["reading_required_output_covered_count"] == 0
    assert paper_snapshot["reading_required_output_coverage"] == 0.0
    assert paper_snapshot["missing_reading_required_output_count"] == 14
    assert paper_snapshot["unsupported_fact_node_rate"] == 0.0
    assert paper_snapshot["qa"]["total"] == 3
    assert paper_snapshot["qa"]["graph_hit_count"] == 1
    assert paper_snapshot["qa"]["graph_hit_rate"] == 0.3333
    assert paper_snapshot["qa"]["graph_context_selected_count"] == 2
    assert paper_snapshot["qa"]["graph_context_selected_rate"] == 0.6667
    assert paper_snapshot["qa"]["cache_hit_rate"] == 0.3333
    assert snapshot["aggregate"]["qa_total"] == 3
    assert snapshot["aggregate"]["qa_graph_hit_count"] == 1
    assert snapshot["aggregate"]["qa_graph_context_selected_count"] == 2
    assert snapshot["aggregate"]["qa_cache_hit_rate"] == 0.3333
    assert "average_extracted_number_locatable_rate" in snapshot["aggregate"]
    assert snapshot["aggregate"]["average_reading_required_output_coverage"] == 0.0


def test_core_quality_snapshot_reaudits_stale_reviewed_graph_with_current_reading_gaps(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            }
        ]
    }
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    claim_span = next(span for span in dom.spans if "block table method" in span.text)
    narrow_plan = reading_plan_subset(dom, ReadingTaskType.CLAIM_INVENTORY)
    full_plan = build_initial_reading_plan(dom)
    claim_task = reading_task(narrow_plan, ReadingTaskType.CLAIM_INVENTORY)
    write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=narrow_plan,
        observation_log=ObservationLog(paper_id="p_test").append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id=claim_task.task_id,
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[claim_span.source_id],
                covered_outputs=claim_task.required_outputs,
            )
        ),
        producer="unit_test",
    )
    root = data_dir / "core" / "v2" / "p_test"
    write_typed_artifact(
        root / "reading_plan.v1.json",
        artifact_type="reading_plan",
        data=full_plan.model_dump(),
        producer="unit_test_stale_plan",
        metadata={"paper_id": paper.paper_id},
    )

    snapshot_path = write_core_quality_snapshot(output_dir)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))["data"]
    paper_snapshot = snapshot["papers"][0]

    assert paper_snapshot["artifact_publish_status"] == PublishStatus.REVIEWED
    assert paper_snapshot["current_audit_publish_status"] == PublishStatus.DRAFT_WEAK
    assert paper_snapshot["publish_status"] == PublishStatus.DRAFT_WEAK
    assert "missing_reading_required_output" in paper_snapshot["current_audit_issue_codes"]
    assert paper_snapshot["reading_required_output_count"] == 14
    assert paper_snapshot["reading_required_output_covered_count"] == 1
    assert paper_snapshot["missing_reading_required_output_count"] == 13
    assert snapshot["aggregate"]["draft_weak_paper_count"] == 1


def test_core_quality_snapshot_requires_complete_core_artifact_set(tmp_path):
    output_dir = tmp_path / "out"
    data_dir = output_dir / ".paperlens" / "data"
    paper = PaperRecord(
        paper_id="p_test",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="Test Paper",
        page_count=1,
    )
    layout = {
        "pages": [
            {
                "page_no": 1,
                "text": "Abstract\n\nWe propose a block table method for serving.",
                "section_candidates": [{"title": "Abstract", "level": 1}],
            }
        ]
    }
    dom = build_paper_dom_from_layout(
        paper_id=paper.paper_id,
        title=paper.canonical_title,
        layout=layout,
    )
    claim_span = next(span for span in dom.spans if "block table method" in span.text)
    reading_plan = reading_plan_subset(dom, ReadingTaskType.CLAIM_INVENTORY)
    claim_task = reading_task(reading_plan, ReadingTaskType.CLAIM_INVENTORY)
    write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=reading_plan,
        observation_log=ObservationLog(paper_id="p_test").append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id=claim_task.task_id,
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[claim_span.source_id],
                covered_outputs=claim_task.required_outputs,
            )
        ),
        producer="unit_test",
    )
    (data_dir / "core" / "v2" / "p_test" / "quality_metrics.v1.json").unlink()

    snapshot_path = write_core_quality_snapshot(output_dir)
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))["data"]
    paper_snapshot = snapshot["papers"][0]

    assert paper_snapshot["artifact_set_status"] == "INCOMPLETE"
    assert paper_snapshot["artifact_set_consumable"] is False
    assert "missing:quality_metrics.v1.json" in paper_snapshot["artifact_set_issues"]
    assert paper_snapshot["publish_status"] is None
    assert paper_snapshot["artifact_publish_status"] is None
    assert paper_snapshot["current_audit_publish_status"] is None
    assert snapshot["aggregate"]["blocked_paper_count"] == 0
    assert snapshot["aggregate"]["draft_weak_paper_count"] == 0


def test_stage17_manifest_includes_core_quality_snapshot(tmp_path):
    output_dir = tmp_path / "out"
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    pipeline = PaperLensWorkflow(
        input_dir=input_dir,
        output_dir=output_dir,
        config=CoreConfig(offline_debug=True),
        events=EventWriter(
            "run_test",
            output_dir / ".paperlens" / "data" / "events.jsonl",
            output_dir / ".paperlens" / "data" / "errors.jsonl",
        ),
        control=ControlState(),
    )
    try:
        pipeline.prepare_output()
        paper = PaperRecord(
            paper_id="p_test",
            file_path="paper.pdf",
            file_hash="hash",
            canonical_title="Test Paper",
            page_count=1,
        )
        decision = ClassificationDecision(
            paper_id="p_test",
            class_label="A",
            confidence=0.9,
            false_negative_risk=0.1,
        )
        pipeline.papers = [paper]
        pipeline.classifications = [decision]
        report_name = paper_report_filename(paper)
        (output_dir / "PaperLens.md").write_text("# PaperLens\n\n索引。", encoding="utf-8")
        (output_dir / "papers" / report_name).write_text("# Test Paper\n\n报告。", encoding="utf-8")
        write_paperlens_library(
            output_dir=output_dir,
            rows=[
                {
                    "paper": paper,
                    "decision": decision,
                    "card": PaperCard(
                        paper_id="p_test",
                        contribution_claims=["The paper proposes a block table method."],
                    ),
                    "report_name": report_name,
                    "report_title": "Test Paper",
                    "paper_memory_v3": {},
                    "model_report": {"one_line_reason": "Block table method."},
                    "report_audit": {"verdict": "PASS"},
                }
            ],
            topic=None,
            idea=None,
        )
        memory_dir = output_dir / ".paperlens" / "data" / "memory" / "v3"
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / "p_test.paper_memory.v3.json").write_text("{}", encoding="utf-8")
        write_core_v2_artifacts(
            data_dir=pipeline.data_dir,
            paper=paper,
            layout={
                "pages": [
                    {
                        "page_no": 1,
                        "text": "Abstract\n\nWe propose a block table method.",
                        "section_candidates": [{"title": "Abstract", "level": 1}],
                    }
                ]
            },
        )

        manifest = pipeline.stage_17_manifest()

        snapshot_path = output_dir / ".paperlens" / "data" / "core_quality_snapshot.v1.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert manifest["artifacts"]["core_quality_snapshot"] == (
            ".paperlens/data/core_quality_snapshot.v1.json"
        )
        assert snapshot["artifact_type"] == "core_quality_snapshot"
        assert snapshot["data"]["paper_count"] == 1
    finally:
        pipeline.db.close()


def test_stage07_runs_core_v2_observation_read_before_legacy_rolling(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    pipeline = PaperLensWorkflow(
        input_dir=input_dir,
        output_dir=output_dir,
        config=CoreConfig(offline_debug=False),
        events=EventWriter(
            "run_test",
            output_dir / ".paperlens" / "data" / "events.jsonl",
            output_dir / ".paperlens" / "data" / "errors.jsonl",
        ),
        control=ControlState(),
    )
    try:
        pipeline.prepare_output()
        paper = PaperRecord(
            paper_id="p_test",
            file_path="paper.pdf",
            file_hash="hash",
            canonical_title="Test Paper",
            page_count=1,
        )
        skim = SkimCard(paper_id="p_test", problem="problem")
        decision = ClassificationDecision(
            paper_id="p_test",
            class_label="A",
            confidence=0.9,
            false_negative_risk=0.1,
        )
        page = PageArtifact(
            paper_id="p_test",
            page_no=1,
            text="Abstract\n\nWe propose a block table method.",
        )
        pipeline.papers = [paper]
        pipeline.skim_cards = [skim]
        pipeline.classifications = [decision]
        pipeline.db.upsert_paper(paper)
        pipeline.db.insert_page_artifacts([page])
        calls = []

        monkeypatch.setattr(pipeline, "llm_enabled", lambda: True)
        monkeypatch.setattr(
            pipeline,
            "new_llm_client",
            lambda: SimpleNamespace(config=SimpleNamespace(kind="fake", model="fake-model")),
        )

        def fake_core_v2_read(**kwargs):
            calls.append(("core_v2", kwargs["paper"].paper_id))

        def fake_rolling(**kwargs):
            calls.append(("legacy_rolling", kwargs["paper"].paper_id))
            return PaperCard(
                paper_id=kwargs["paper"].paper_id,
                contribution_claims=["legacy card"],
            )

        monkeypatch.setattr(pipeline, "run_core_v2_observation_read", fake_core_v2_read)
        monkeypatch.setattr(pipeline, "run_rolling_paper_read", fake_rolling)

        pipeline.stage_07_normal_read()

        assert calls == [("core_v2", "p_test"), ("legacy_rolling", "p_test")]
        assert pipeline.paper_cards[0].contribution_claims == ["legacy card"]
    finally:
        pipeline.db.close()
