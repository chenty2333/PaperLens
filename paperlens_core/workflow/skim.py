from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

from paperlens_core.config import CoreConfig
from paperlens_core.db import ArtifactDb
from paperlens_core.events import EventWriter
from paperlens_core.schemas import ClassificationDecision, PaperRecord, SkimCard
from paperlens_core.workflow.core_v2 import write_core_v2_artifacts
from paperlens_core.workflow.utils import load_layout_index


class SkimWorkflowContext(Protocol):
    data_dir: Path
    config: CoreConfig
    events: EventWriter
    db: ArtifactDb
    papers: list[PaperRecord]
    skim_cards: list[SkimCard]
    classifications: list[ClassificationDecision]

    def checkpoint(self, stage: str) -> None: ...

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


def run_skim_stage(workflow: SkimWorkflowContext) -> None:
    stage = "stage_03_skim"
    workflow.checkpoint(stage)
    workflow.events.stage_started(stage, "Building deterministic paper maps")
    active_ids = {paper.paper_id for paper in workflow.papers}
    existing_skim_by_id = {
        card.paper_id: card
        for card in (workflow.skim_cards or workflow.db.list_skim_cards())
        if card.paper_id in active_ids
    }
    existing_decision_by_id = {
        decision.paper_id: decision
        for decision in (workflow.classifications or workflow.db.list_classifications())
        if decision.paper_id in active_ids
    }
    workflow.skim_cards = list(existing_skim_by_id.values())
    workflow.classifications = list(existing_decision_by_id.values())
    pending: list[tuple[PaperRecord, SkimCard, ClassificationDecision]] = []
    for paper in workflow.papers:
        if paper.paper_id in existing_skim_by_id and paper.paper_id in existing_decision_by_id:
            workflow.events.emit(
                "cache_hit",
                stage=stage,
                message=f"Skim/classification already exists for {paper.paper_id}",
                data={"paper_id": paper.paper_id},
            )
            workflow.mark_paper_state(paper.paper_id, stage)
            continue
        artifacts = workflow.db.get_page_artifacts(paper.paper_id)
        card, decision = deterministic_skim_classify(
            paper,
            artifacts,
            workflow.config.keyword_pool,
        )
        pending.append((paper, card, decision))

    for paper, card, decision in pending:
        persist_skim_classification(workflow, stage, paper, card, decision)
    order_skim_classification_state(workflow)
    core_v2_count = persist_core_v2_artifacts(workflow, stage)
    workflow.events.stage_completed(
        stage,
        "Paper maps completed",
        {
            "skim_cards": len(workflow.skim_cards),
            "classifications": len(workflow.classifications),
            "core_v2_artifacts": core_v2_count,
        },
    )


def persist_core_v2_artifacts(workflow: SkimWorkflowContext, stage: str) -> int:
    skim_by_id = {card.paper_id: card for card in workflow.skim_cards}
    decision_by_id = {decision.paper_id: decision for decision in workflow.classifications}
    written_count = 0
    for paper in workflow.papers:
        layout = load_layout_index(workflow.data_dir, paper.paper_id)
        if not layout:
            artifacts = workflow.db.get_page_artifacts(paper.paper_id)
            layout = {"pages": [artifact.model_dump() for artifact in artifacts]}
        paths = write_core_v2_artifacts(
            data_dir=workflow.data_dir,
            paper=paper,
            layout=layout,
            skim=skim_by_id.get(paper.paper_id),
            decision=decision_by_id.get(paper.paper_id),
        )
        for artifact_type, path in paths.items():
            workflow.register_file_artifact(
                path,
                paper_id=paper.paper_id,
                artifact_type=f"core_v2_{artifact_type}",
                depends_on=[f"layout_index:{paper.paper_id}"],
            )
        written_count += 1
        workflow.events.emit(
            "core_v2_artifacts_written",
            stage=stage,
            message=f"Core v2 artifacts written for {paper.paper_id}",
            data={
                "paper_id": paper.paper_id,
                "artifacts": {key: str(path) for key, path in paths.items()},
            },
        )
    return written_count


def persist_skim_classification(
    workflow: SkimWorkflowContext,
    stage: str,
    paper: PaperRecord,
    card: SkimCard,
    decision: ClassificationDecision,
) -> None:
    if any(item.paper_id == card.paper_id for item in workflow.skim_cards):
        workflow.skim_cards = [
            card if item.paper_id == card.paper_id else item for item in workflow.skim_cards
        ]
    else:
        workflow.skim_cards.append(card)
    if any(item.paper_id == decision.paper_id for item in workflow.classifications):
        workflow.classifications = [
            decision if item.paper_id == decision.paper_id else item
            for item in workflow.classifications
        ]
    else:
        workflow.classifications.append(decision)
    workflow.db.upsert_skim(card)
    workflow.db.upsert_classification(decision)
    workflow.mark_paper_state(paper.paper_id, stage)


def order_skim_classification_state(workflow: SkimWorkflowContext) -> None:
    skim_by_id = {card.paper_id: card for card in workflow.skim_cards}
    decision_by_id = {decision.paper_id: decision for decision in workflow.classifications}
    workflow.skim_cards = [
        skim_by_id[paper.paper_id]
        for paper in workflow.papers
        if paper.paper_id in skim_by_id
    ]
    workflow.classifications = [
        decision_by_id[paper.paper_id]
        for paper in workflow.papers
        if paper.paper_id in decision_by_id
    ]


def deterministic_skim_classify(
    paper: PaperRecord,
    artifacts: list[Any],
    keyword_pool: list[str],
) -> tuple[SkimCard, ClassificationDecision]:
    text = "\n".join(page.text for page in artifacts[:3])
    signals = keyword_hits(text + "\n" + (paper.canonical_title or ""), keyword_pool)
    card = SkimCard(
        paper_id=paper.paper_id,
        problem=first_sentence(text) or paper.canonical_title,
        method_type=infer_method_type(text),
        system_scope=infer_scope(text),
        evaluation_type=infer_evaluation(text),
        danger_signals=signals,
        evidence_source_ids=first_evidence_source_ids(artifacts, signals),
        confidence=min(0.9, 0.35 + 0.12 * len(signals)),
    )
    return card, classify_paper(paper, card)


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    lowered = text.lower()
    hits = []
    for keyword in keywords:
        if keyword.lower() in lowered:
            hits.append(keyword)
    return sorted(set(hits), key=str.lower)


def first_sentence(text: str) -> str | None:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return None
    match = re.match(r"(.{30,240}?[.!?])\s", cleaned)
    return match.group(1) if match else cleaned[:240]


def first_evidence_source_ids(artifacts: list[Any], signals: list[str]) -> list[str]:
    if not artifacts:
        return []
    needle = signals[0] if signals else None
    for page in artifacts:
        if needle and needle.lower() not in page.text.lower():
            continue
        source_id = first_block_source_id(page)
        return [source_id] if source_id else []
    source_id = first_block_source_id(artifacts[0])
    return [source_id] if source_id else []


def first_block_source_id(page: Any) -> str | None:
    for block in getattr(page, "blocks", []) or []:
        if not isinstance(block, dict):
            continue
        source_id = str(block.get("source_id") or "").strip()
        if source_id:
            return source_id
    return None


def infer_method_type(text: str) -> str:
    lowered = text.lower()
    if "system" in lowered or "runtime" in lowered:
        return "system"
    if "analysis" in lowered or "formal" in lowered:
        return "analysis"
    if "survey" in lowered:
        return "survey"
    return "unknown"


def infer_scope(text: str) -> str:
    lowered = text.lower()
    if "webassembly" in lowered or "wasm" in lowered:
        return "webassembly_runtime"
    if "kernel" in lowered:
        return "kernel"
    if "virtual machine" in lowered or "vm" in lowered:
        return "vm"
    return "unknown"


def infer_evaluation(text: str) -> str:
    lowered = text.lower()
    if "benchmark" in lowered or "throughput" in lowered or "latency" in lowered:
        return "performance"
    if "case study" in lowered:
        return "case_study"
    if "proof" in lowered:
        return "formal"
    return "unknown"


def classify_paper(paper: PaperRecord, card: SkimCard) -> ClassificationDecision:
    if paper.parse_quality in {"OCR_REQUIRED", "VLM_PAGE_MODE"}:
        return ClassificationDecision(
            paper_id=paper.paper_id,
            class_label="HOLD",
            confidence=0.3,
            false_negative_risk=0.8,
            reason_codes=[str(paper.parse_quality).lower()],
        )
    if paper.parse_quality == "PASS_WITH_WEAKNESSES" and not card.evidence_source_ids:
        return ClassificationDecision(
            paper_id=paper.paper_id,
            class_label="HOLD",
            confidence=0.35,
            false_negative_risk=0.7,
            reason_codes=["weak_parse_without_evidence"],
        )
    signal_count = len(card.danger_signals)
    if signal_count >= 3:
        label = "A"
    elif signal_count >= 1:
        label = "B"
    else:
        label = "C"
    return ClassificationDecision(
        paper_id=paper.paper_id,
        class_label=label,
        confidence=min(0.92, 0.45 + 0.15 * signal_count),
        false_negative_risk=max(0.25 if label == "C" else 0.1, 0.75 - 0.16 * signal_count),
        reason_codes=[f"keyword:{signal}" for signal in card.danger_signals]
        or ["no_keyword_signal"],
    )
