from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from paperlens_core.audit import (
    PublishStatus,
    audit_claim_graph,
    compute_core_quality_metrics,
    publish_status_from_findings,
)
from paperlens_core.dom import build_paper_dom_from_layout
from paperlens_core.graph import GraphEdge, graph_from_observations
from paperlens_core.library import (
    doctor_library,
    rebuild_library_from_output,
    read_library_records,
    search_library,
    write_paperlens_library,
)
from paperlens_core.memory import materialize_paper_memory
from paperlens_core.qa import answer_question, load_core_v2_qa_context
from paperlens_core.quality_snapshot import write_core_quality_snapshot
from paperlens_core.reading import (
    ObservationCard,
    ObservationLog,
    ObservationType,
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
)
from paperlens_core.runtime import (
    ArtifactEnvelope,
    NodeSpec,
    NodeStatus,
    run_finite_node,
)
from paperlens_core.config import CoreConfig
from paperlens_core.control import ControlState
from paperlens_core.core_manifest import build_core_v2_manifest
from paperlens_core.events import EventWriter
from paperlens_core.schemas import (
    ClassificationDecision,
    PageArtifact,
    PaperCard,
    PaperRecord,
    SkimCard,
)
from paperlens_core.workflow.agent import PaperLensWorkflow, paper_report_filename
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


def test_observation_log_is_append_only_and_requires_sources():
    dom = sample_dom()
    source_id = dom.spans[0].source_id
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
    )

    graph = graph_from_observations("p_test", [observation])
    findings = audit_claim_graph(graph, dom)
    memory = materialize_paper_memory(
        graph,
        unresolved_audit_findings=[finding.finding_id for finding in findings],
        report_readiness=publish_status_from_findings(findings).value,
    )

    assert findings == []
    assert memory.result_nodes
    assert memory.report_readiness == PublishStatus.REVIEWED
    assert memory.evidence_index[memory.result_nodes[0]]
    metrics = compute_core_quality_metrics(dom=dom, graph=graph, findings=findings)
    assert metrics.evidence_coverage == 1.0
    assert metrics.publish_status == PublishStatus.REVIEWED


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

    assert {finding.code for finding in findings} >= {"missing_dom_source"}
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

    assert {finding.code for finding in findings} >= {
        "fact_node_text_not_grounded_in_evidence_source"
    }
    assert publish_status_from_findings(findings) == PublishStatus.BLOCKED


def test_audit_blocks_dangling_claim_graph_edges():
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


def test_finite_runtime_node_enforces_allowed_tools():
    spec = NodeSpec(
        node_id="read_sources",
        output_artifact_type="observation_cards",
        allowed_tools=("paper_dom.read_sources",),
    )

    def handler(context):
        context.record_tool_call("paper_dom.read_sources")
        return ArtifactEnvelope(
            artifact_type="observation_cards",
            producer="unit",
            data=[],
        )

    result = run_finite_node(spec, [], handler)

    assert result.status == NodeStatus.PASS
    assert result.tool_calls_used == 1
    assert result.used_tools == ["paper_dom.read_sources"]


def test_finite_runtime_node_rejects_disallowed_tools():
    spec = NodeSpec(node_id="read_sources", allowed_tools=("paper_dom.read_sources",))

    def handler(context):
        context.record_tool_call("filesystem.read")
        return None

    result = run_finite_node(spec, [], handler)

    assert result.status == NodeStatus.FAIL
    assert result.tool_calls_used == 1
    assert "disallowed tool=filesystem.read" in result.issues[0]


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
        assert report_envelope["artifact_type"] == "graph_report_draft"
        assert report_audit_envelope["artifact_type"] == "report_audit_findings"
        assert core_manifest_envelope["artifact_type"] == "core_v2_manifest"
        assert dom_envelope["data"]["spans"]
        assert plan_envelope["data"]["tasks"]
        assert graph_envelope["data"]["nodes"]
        assert report_envelope["data"]["sections"]
        assert report_audit_envelope["data"] == []
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
    assert core_manifest["artifact_type"] == "core_v2_manifest"
    assert core_manifest["data"]["status"] == "COMPLETE"
    assert core_manifest["data"]["publish_status"] == PublishStatus.REVIEWED
    assert core_manifest["data"]["consumable"] is True
    assert len(usage_rows) == calls["count"]
    assert all(row["status"] == "PASS" for row in agent_runs)
    assert all(row["tool_calls_used"] == 1 for row in agent_runs)
    assert all(row["used_tools"] == ["paper_dom.read_sources"] for row in agent_runs)
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

    assert (tmp_path / "core" / "v2" / "p_test" / "paper_dom.v1.json").exists()
    assert (tmp_path / "core" / "v2" / "p_test" / "reading_plan.v1.json").exists()
    assert loaded_dom.paper_id == "p_test"
    assert loaded_plan.paper_id == "p_test"
    assert paths["quality_metrics"].exists()


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

    with pytest.raises(ValueError, match="did not cite valid source_ids"):
        observation_cards_from_model_envelope(
            envelope,
            paper_id="p_test",
            task=build_initial_reading_plan(sample_dom()).tasks[0],
            valid_source_ids=sample_dom().source_ids(),
        )


def test_model_observation_cards_reject_disallowed_observation_type():
    envelope = ArtifactEnvelope(
        artifact_type="observation_cards",
        producer="fake",
        data={
            "cards": [
                {
                    "observation_type": "claim",
                    "statement": "A claim emitted from a mechanism task.",
                    "source_ids": [sample_dom().spans[0].source_id],
                    "confidence": "high",
                    "provenance": "explicit",
                    "uncertainty": None,
                    "extracted_numbers": [],
                    "proposed_links": [],
                }
            ]
        },
    )
    task = next(
        task
        for task in build_initial_reading_plan(sample_dom()).tasks
        if task.task_type == ReadingTaskType.METHOD_MECHANISM
    )

    with pytest.raises(ValueError, match="disallowed observation_type=claim"):
        observation_cards_from_model_envelope(
            envelope,
            paper_id="p_test",
            task=task,
            valid_source_ids=sample_dom().source_ids(),
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
                    "source_ids": [dom.spans[0].source_id],
                    "confidence": "high",
                    "provenance": "background",
                    "uncertainty": None,
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
            valid_source_ids=dom.source_ids(),
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
    write_core_v2_from_observation_log(
        data_dir=output_dir / ".paperlens" / "data",
        paper=paper,
        dom=dom,
        reading_plan=build_initial_reading_plan(dom),
        observation_log=ObservationLog(paper_id="p_test")
        .append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id="read_02_claim_inventory",
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[claim_span.source_id],
            )
        )
        .append(
            ObservationCard(
                observation_id="obs_mechanism",
                paper_id="p_test",
                task_id="read_03_method_mechanism",
                observation_type=ObservationType.MECHANISM,
                statement="The block table mechanism organizes serving state.",
                source_ids=[mechanism_span.source_id],
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
        question="block table method 是什么？",
    )
    answer = answer_question(
        output_dir=output_dir,
        config=CoreConfig(offline_debug=True),
        paper_id="p_test",
        question="block table method 是什么？",
    )

    assert context["matches"]
    assert context["matches"][0]["source_ids"][0].startswith("span:p_test:")
    assert context["matches"][0]["relationships"][0]["kind"] == "explains"
    assert context["matches"][0]["relationships"][0]["source_ids"]
    assert answer["cited_source_ids"][0].startswith("span:p_test:")
    assert set(answer["cited_pages"]) == {1, 2}
    assert "ClaimGraph" in answer["answer_markdown"]
    assert "relation:" in answer["answer_markdown"]
    assert answer["source_attribution"]["paper_claims"]


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
    write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=build_initial_reading_plan(dom),
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
    write_core_v2_from_observation_log(
        data_dir=output_dir / ".paperlens" / "data",
        paper=paper,
        dom=dom,
        reading_plan=build_initial_reading_plan(dom),
        observation_log=ObservationLog(paper_id="p_test")
        .append(
            ObservationCard(
                observation_id="obs_claim",
                paper_id="p_test",
                task_id="read_02_claim_inventory",
                observation_type=ObservationType.CLAIM,
                statement="The paper proposes a block table method.",
                source_ids=[claim_span.source_id],
            )
        )
        .append(
            ObservationCard(
                observation_id="obs_mechanism",
                paper_id="p_test",
                task_id="read_03_method_mechanism",
                observation_type=ObservationType.MECHANISM,
                statement="The block table mechanism organizes serving state.",
                source_ids=[mechanism_span.source_id],
                proposed_links=[
                    {"source_id": "obs_mechanism", "target_id": "obs_claim", "kind": "explains"}
                ],
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
    assert records[0]["memory"]["claims"]
    assert records[0]["provenance"]["core_v2"]["source_ids"]
    assert records[0]["quality"]["graph_publish_status"] == PublishStatus.REVIEWED
    assert result["matches"][0]["paper"]["paper_id"] == "p_test"
    assert index["records"][0]["graph"]["node_counts"]["claim"] >= 1
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
    write_core_v2_from_observation_log(
        data_dir=data_dir,
        paper=paper,
        dom=dom,
        reading_plan=build_initial_reading_plan(dom),
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
    assert paper_snapshot["unsupported_fact_node_rate"] == 0.0
    assert paper_snapshot["qa"]["total"] == 2
    assert paper_snapshot["qa"]["graph_hit_rate"] == 0.5
    assert paper_snapshot["qa"]["cache_hit_rate"] == 0.5
    assert snapshot["aggregate"]["qa_total"] == 2
    assert snapshot["aggregate"]["qa_cache_hit_rate"] == 0.5


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
