from __future__ import annotations

from paperlens_core.schemas import PaperState, now_iso


MAIN_STAGE_STATES = {
    "stage_00_ingest": "INGESTED",
    "stage_01_parse": "PARSED",
    "stage_02_parse_verify": "PARSE_VERIFIED",
    "stage_03_skim": "SKIMMED",
    "stage_04_classify": "CLASSIFIED",
    "stage_05_classification_audit": "CLASSIFICATION_AUDITED",
    "stage_06_queue": "QUEUED_FOR_NORMAL_READ",
    "stage_07_normal_read": "NORMAL_READ",
    "stage_08_evidence_verify": "EVIDENCE_VERIFIED",
    "stage_15_export": "REPORT_EXPORTED",
    "stage_17_manifest": "RUN_MANIFEST_READY",
}


def transition_state(
    state: PaperState | None,
    *,
    paper_id: str,
    stage: str,
    side_statuses: list[str] | None = None,
    error: str | None = None,
) -> PaperState:
    next_state = MAIN_STAGE_STATES.get(stage, state.state if state else "NEW")
    merged_side_statuses = list(
        dict.fromkeys((state.side_statuses if state else []) + (side_statuses or []))
    )
    return PaperState(
        paper_id=paper_id,
        state="FAILED" if error else next_state,
        side_statuses=merged_side_statuses,
        current_stage=stage,
        retry_count=state.retry_count if state else 0,
        last_error=error,
        updated_at=now_iso(),
    )
