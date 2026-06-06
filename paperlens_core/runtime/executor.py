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
    require_tool_source_ids: bool = True

    def __post_init__(self) -> None:
        if not self.node_id.strip():
            raise ValueError("node_id cannot be blank")
        if any(not tool.strip() for tool in self.allowed_tools):
            raise ValueError("allowed_tools cannot contain blank tool names")
        if self.max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if self.max_model_calls < 0:
            raise ValueError("max_model_calls must be >= 0")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 when set")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be > 0")


@dataclass
class NodeResult:
    node_id: str
    status: NodeStatus
    output: ArtifactEnvelope | None = None
    steps_used: int = 0
    model_calls_used: int = 0
    tool_calls_used: int = 0
    used_tools: list[str] = field(default_factory=list)
    tool_source_ids: dict[str, list[str]] = field(default_factory=dict)
    tokens_used: int = 0
    token_usage: dict[str, Any] = field(default_factory=dict)
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
    tool_calls_used: int = 0
    used_tools: list[str] = field(default_factory=list)
    tool_source_ids: dict[str, list[str]] = field(default_factory=dict)
    tokens_used: int = 0
    token_usage: dict[str, Any] = field(default_factory=dict)

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

    def record_tool_call(
        self,
        tool_name: str,
        *,
        source_ids: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        tool_name = tool_name.strip()
        if not tool_name:
            raise NodeExecutionError("tool_name cannot be blank")
        self.tool_calls_used += 1
        self._check_runtime_budget()
        if tool_name not in self.spec.allowed_tools:
            allowed = ", ".join(self.spec.allowed_tools) or "none"
            raise NodeExecutionError(
                f"{self.spec.node_id} attempted disallowed tool={tool_name}; allowed={allowed}"
            )
        cleaned_source_ids = clean_source_ids(source_ids)
        if self.spec.require_tool_source_ids and not cleaned_source_ids:
            raise NodeExecutionError(
                f"{self.spec.node_id} tool={tool_name} did not return source_ids"
            )
        if tool_name not in self.used_tools:
            self.used_tools.append(tool_name)
        known_source_ids = self.tool_source_ids.setdefault(tool_name, [])
        for source_id in cleaned_source_ids:
            if source_id not in known_source_ids:
                known_source_ids.append(source_id)

    def record_token_usage(self, usage: dict[str, Any]) -> None:
        self._check_runtime_budget()
        usage_tokens = token_count_from_usage(usage)
        self.tokens_used += usage_tokens
        merge_numeric_usage(self.token_usage, usage)
        if self.spec.max_tokens is not None and self.tokens_used > self.spec.max_tokens:
            raise RuntimeBudgetExceeded(
                f"{self.spec.node_id} exceeded max_tokens={self.spec.max_tokens}"
            )

    def enforce_runtime_budget(self) -> None:
        self._check_runtime_budget()

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
    input_counts: dict[str, int] = {}
    for artifact in inputs:
        input_counts[artifact.artifact_type] = input_counts.get(artifact.artifact_type, 0) + 1
    duplicate_inputs = [
        artifact_type for artifact_type, count in sorted(input_counts.items()) if count > 1
    ]
    if duplicate_inputs:
        return NodeResult(
            node_id=spec.node_id,
            status=NodeStatus.FAIL,
            issues=[f"duplicate_input_artifact:{item}" for item in duplicate_inputs],
        )
    missing = [
        artifact_type
        for artifact_type in spec.input_artifact_types
        if artifact_type not in input_counts
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
        context.enforce_runtime_budget()
        if output is not None and not isinstance(output, ArtifactEnvelope):
            raise NodeExecutionError(
                f"{spec.node_id} returned non-artifact output={type(output).__name__}"
            )
        if spec.output_artifact_type and output is None:
            raise NodeExecutionError(
                f"{spec.node_id} did not return output_artifact_type={spec.output_artifact_type}"
            )
        if spec.output_artifact_type and output is not None:
            output.require_type(spec.output_artifact_type)
        if output is not None:
            validate_output_source_ids_from_tool_trace(spec, output, context)
        return NodeResult(
            node_id=spec.node_id,
            status=NodeStatus.PASS,
            output=output,
            steps_used=context.steps_used,
            model_calls_used=context.model_calls_used,
            tool_calls_used=context.tool_calls_used,
            used_tools=list(context.used_tools),
            tool_source_ids={key: list(value) for key, value in context.tool_source_ids.items()},
            tokens_used=context.tokens_used,
            token_usage=dict(context.token_usage),
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
            tool_calls_used=context.tool_calls_used,
            used_tools=list(context.used_tools),
            tool_source_ids={key: list(value) for key, value in context.tool_source_ids.items()},
            tokens_used=context.tokens_used,
            token_usage=dict(context.token_usage),
            elapsed_seconds=round(time.time() - context.started_at, 3),
            issues=[str(exc)],
        )


def token_count_from_usage(usage: dict[str, Any]) -> int:
    total = numeric_int(usage.get("total_tokens"))
    if total is not None:
        return total
    input_tokens = (
        numeric_int(usage.get("input_tokens")) or numeric_int(usage.get("prompt_tokens")) or 0
    )
    output_tokens = (
        numeric_int(usage.get("output_tokens")) or numeric_int(usage.get("completion_tokens")) or 0
    )
    return input_tokens + output_tokens


def numeric_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return None


def merge_numeric_usage(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            previous = target.get(key)
            target[key] = (previous if isinstance(previous, (int, float)) else 0) + value


def validate_output_source_ids_from_tool_trace(
    spec: NodeSpec,
    output: ArtifactEnvelope,
    context: NodeContext,
) -> None:
    if not output.source_ids:
        return
    tool_source_ids = {
        source_id for values in context.tool_source_ids.values() for source_id in values
    }
    unknown = [source_id for source_id in output.source_ids if source_id not in tool_source_ids]
    if unknown:
        raise NodeExecutionError(
            f"{spec.node_id} output source_ids outside tool trace: {', '.join(unknown[:8])}"
        )


def clean_source_ids(value: list[str] | tuple[str, ...] | None) -> list[str]:
    result = []
    for item in value or ():
        source_id = str(item or "").strip()
        if source_id and source_id not in result:
            result.append(source_id)
    return result
