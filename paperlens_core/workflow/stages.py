from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkflowStage:
    id: str
    state: str
    label_zh: str


WORKFLOW_STAGES: tuple[WorkflowStage, ...] = (
    WorkflowStage("stage_00_ingest", "INGESTED", "导入 PDF"),
    WorkflowStage("stage_01_parse", "PARSED", "解析论文"),
    WorkflowStage("stage_02_parse_verify", "PARSE_VERIFIED", "检查解析质量"),
    WorkflowStage("stage_03_skim", "MAPPED", "建立论文地图"),
    WorkflowStage("stage_07_normal_read", "CORE_GRAPH_BUILT", "构建事实图"),
    WorkflowStage("stage_08_evidence_verify", "CORE_GRAPH_REVIEWED", "审计事实图"),
    WorkflowStage("stage_15_export", "REPORT_EXPORTED", "导出论文报告"),
    WorkflowStage("stage_17_manifest", "RUN_MANIFEST_READY", "保存运行结果"),
)

WORKFLOW_STAGE_ORDER = [stage.id for stage in WORKFLOW_STAGES]
WORKFLOW_STAGE_STATES = {stage.id: stage.state for stage in WORKFLOW_STAGES}
WORKFLOW_STAGE_LABELS_ZH = {stage.id: stage.label_zh for stage in WORKFLOW_STAGES}


def normalize_workflow_stage(stage: str | None) -> str | None:
    if stage is None:
        return None
    if stage not in WORKFLOW_STAGE_ORDER:
        raise ValueError(
            f"Unknown workflow stage '{stage}'. Available stages: {', '.join(WORKFLOW_STAGE_ORDER)}"
        )
    return stage


def resolve_workflow_stages(
    *,
    from_stage: str | None = None,
    only_stage: str | None = None,
) -> list[str]:
    from_stage = normalize_workflow_stage(from_stage)
    only_stage = normalize_workflow_stage(only_stage)
    if from_stage and only_stage:
        raise ValueError("--from-stage and --only-stage cannot be used together")
    if only_stage:
        return [only_stage]
    if from_stage:
        return WORKFLOW_STAGE_ORDER[WORKFLOW_STAGE_ORDER.index(from_stage) :]
    return list(WORKFLOW_STAGE_ORDER)
