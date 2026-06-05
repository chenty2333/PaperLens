from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from paperlens_core.runtime.artifacts import ArtifactEnvelope


class NodeStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"


class NodeExecutionError(RuntimeError):
    pass


class RuntimeBudgetExceeded(NodeExecutionError):
    pass


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    input_artifact_types: tuple[str, ...] = ()
    output_artifact_type: str | None = None
    allowed_tools: tuple[str, ...] = ()
    max_steps: int = 1
    max_model_calls: int = 1
    max_tokens: int | None = None
    timeout_seconds: float = 120.0
    failure_policy: str = "fail"

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("node_id cannot be blank")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_model_calls < 0:
            raise ValueError("max_model_calls must be >= 0")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")


@dataclass
class NodeResult:
    node_id: str
    status: NodeStatus
    output: ArtifactEnvelope | None = None
    steps_used: int = 0
    model_calls_used: int = 0
    elapsed_seconds: float = 0.0
    issues: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NodeContext:
    spec: NodeSpec
    inputs: list[ArtifactEnvelope]
    started_at: float = field(default_factory=time.time)
    steps_used: int = 0
    model_calls_used: int = 0

    def require_step(self) -> None:
        self.steps_used += 1
        self._check_runtime_budget()
        if self.steps_used > self.spec.max_steps:
            raise RuntimeBudgetExceeded(
                f"{self.spec.node_id} exceeded max_steps={self.spec.max_steps}"
            )

    def record_model_call(self) -> None:
        self.model_calls_used += 1
        self._check_runtime_budget()
        if self.model_calls_used > self.spec.max_model_calls:
            raise RuntimeBudgetExceeded(
                f"{self.spec.node_id} exceeded max_model_calls={self.spec.max_model_calls}"
            )

    def _check_runtime_budget(self) -> None:
        elapsed = time.time() - self.started_at
        if elapsed > self.spec.timeout_seconds:
            raise RuntimeBudgetExceeded(
                f"{self.spec.node_id} exceeded timeout_seconds={self.spec.timeout_seconds}"
            )


NodeCallable = Callable[[NodeContext], ArtifactEnvelope | None]


def run_finite_node(
    spec: NodeSpec,
    inputs: list[ArtifactEnvelope],
    handler: NodeCallable,
) -> NodeResult:
    missing = [
        artifact_type
        for artifact_type in spec.input_artifact_types
        if artifact_type not in {artifact.artifact_type for artifact in inputs}
    ]
    if missing:
        return NodeResult(
            node_id=spec.node_id,
            status=NodeStatus.FAIL,
            issues=[f"missing_input_artifact:{item}" for item in missing],
        )

    context = NodeContext(spec=spec, inputs=inputs)
    try:
        context.require_step()
        output = handler(context)
        if spec.output_artifact_type and output is not None:
            output.require_type(spec.output_artifact_type)
        return NodeResult(
            node_id=spec.node_id,
            status=NodeStatus.PASS,
            output=output,
            steps_used=context.steps_used,
            model_calls_used=context.model_calls_used,
            elapsed_seconds=round(time.time() - context.started_at, 3),
        )
    except Exception as exc:
        if spec.failure_policy == "skip":
            status = NodeStatus.SKIP
        else:
            status = NodeStatus.FAIL
        return NodeResult(
            node_id=spec.node_id,
            status=status,
            steps_used=context.steps_used,
            model_calls_used=context.model_calls_used,
            elapsed_seconds=round(time.time() - context.started_at, 3),
            issues=[str(exc)],
        )
