from __future__ import annotations

from paperlens_core.schemas import ClassificationDecision, EvidenceRef, PaperRecord, SkimCard
from paperlens_core.workflow.agent import paper_report_filename as compat_paper_report_filename
from paperlens_core.workflow.report_rows import (
    classification_counts,
    dedupe_evidence_refs,
    novelty_risk,
    paper_report_filename,
    read_decision,
    reading_priority_key,
    row_decision,
    row_relation,
)


def test_report_filename_is_stable_and_agent_compat_entry_matches() -> None:
    paper = PaperRecord(
        paper_id="p_123",
        file_path="paper.pdf",
        file_hash="hash",
        canonical_title="A Geometric Algebra-Informed 3DGS Framework",
    )

    assert paper_report_filename(paper) == ("p_123_a_geometric_algebra_informed_3dgs_framework.md")
    assert compat_paper_report_filename(paper) == paper_report_filename(paper)


def test_missing_classification_falls_back_to_hold() -> None:
    paper = PaperRecord(paper_id="p_test", file_path="paper.pdf", file_hash="hash")

    decision = row_decision({"paper": paper})

    assert decision.class_label == "HOLD"
    assert decision.false_negative_risk == 1.0
    assert decision.reason_codes == ["missing_classification"]


def test_report_row_reading_priority_and_risk_are_decision_driven() -> None:
    paper_a = PaperRecord(paper_id="p_a", file_path="a.pdf", file_hash="hash")
    paper_b = PaperRecord(paper_id="p_b", file_path="b.pdf", file_hash="hash")
    a_decision = ClassificationDecision(
        paper_id="p_a",
        class_label="A",
        confidence=0.9,
        false_negative_risk=0.2,
    )
    b_decision = ClassificationDecision(
        paper_id="p_b",
        class_label="B",
        confidence=0.8,
        false_negative_risk=0.8,
    )
    rows = [
        {"paper": paper_b, "decision": b_decision},
        {"paper": paper_a, "decision": a_decision},
    ]

    assert [row["paper"].paper_id for row in sorted(rows, key=reading_priority_key)] == [
        "p_a",
        "p_b",
    ]
    assert read_decision(rows[0]) == "Selective read"
    assert novelty_risk(rows[0]) == "HIGH"
    assert classification_counts([a_decision, b_decision]) == {"A": 1, "B": 1, "C": 0, "HOLD": 0}


def test_row_relation_and_evidence_dedupe_are_structural() -> None:
    paper = PaperRecord(paper_id="p_test", file_path="paper.pdf", file_hash="hash")
    decision = ClassificationDecision(
        paper_id="p_test",
        class_label="HOLD",
        confidence=0.5,
        false_negative_risk=0.5,
    )
    skim = SkimCard(
        paper_id="p_test",
        method_type="3DGS",
        system_scope="geometry",
        danger_signals=["weak baseline"],
    )
    refs = [
        EvidenceRef(paper_id="p_test", page_no=1, text_span_id="s1"),
        EvidenceRef(paper_id="p_test", page_no=1, text_span_id="s1"),
        EvidenceRef(paper_id="p_test", page_no=2, text_span_id="s2"),
    ]

    relation = row_relation({"paper": paper, "decision": decision, "skim": skim})
    deduped = dedupe_evidence_refs(refs)

    assert "method=3DGS" in relation
    assert "scope=geometry" in relation
    assert [ref.text_span_id for ref in deduped] == ["s1", "s2"]
