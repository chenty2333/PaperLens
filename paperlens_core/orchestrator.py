from __future__ import annotations

from pathlib import Path

from paperlens_core.config import CoreConfig
from paperlens_core.control import ControlState
from paperlens_core.events import EventWriter
from paperlens_core.workflow import PaperLensWorkflow


def run_pipeline(
    *,
    input_dir: Path,
    output_dir: Path,
    config: CoreConfig,
    events: EventWriter,
    control: ControlState,
    from_stage: str | None = None,
    only_stage: str | None = None,
) -> dict:
    return PaperLensWorkflow(
        input_dir=input_dir,
        output_dir=output_dir,
        config=config,
        events=events,
        control=control,
    ).run(from_stage=from_stage, only_stage=only_stage)
