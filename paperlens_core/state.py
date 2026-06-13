from __future__ import annotations

from paperlens_core.schemas import PaperState, now_iso
from paperlens_core.workflow.stages import WORKFLOW_STAGE_STATES


MAIN_STAGE_STATES = dict(WORKFLOW_STAGE_STATES)


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
