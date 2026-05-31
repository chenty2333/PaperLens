from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import urllib.parse
from pathlib import Path
from typing import Any

from paperlens_core.agents.llm import LlmError, LlmJsonResult, parse_json_text
from paperlens_core.config import ProviderConfig


def invoke_openai_agents_sdk_json(
    *,
    config: ProviderConfig,
    system_prompt: str,
    user_prompt: str,
    schema_name: str,
    max_tokens: int,
) -> LlmJsonResult:
    try:
        from agents import (
            Agent,
            GuardrailFunctionOutput,
            ModelSettings,
            Runner,
            RunConfig,
            SQLiteSession,
            flush_traces,
            function_tool,
            input_guardrail,
            output_guardrail,
            trace,
        )
        from agents.models.openai_provider import OpenAIProvider
    except Exception as exc:  # pragma: no cover - optional dependency path
        raise LlmError(f"OpenAI Agents SDK is unavailable: {exc}") from exc

    os.environ.setdefault("OPENAI_API_KEY", config.resolved_api_key() or "")
    try:
        model_provider = _build_model_provider(OpenAIProvider, config)
        tracing_disabled = config.kind == "openai-compatible"
        tools = _build_policy_tools(function_tool)
        input_contract_guardrail = _build_input_guardrail(input_guardrail, GuardrailFunctionOutput)
        output_json_guardrail = _build_output_guardrail(output_guardrail, GuardrailFunctionOutput)
        auditor_agent = Agent(
            name=f"{schema_name}_auditor",
            handoff_description=(
                "Audit the draft for PaperLens evidence, classification, and JSON-contract risks."
            ),
            instructions=(
                "You are the PaperLens SDK auditor. Check that the output is conservative, "
                "uses only supplied excerpts, preserves EvidenceRef requirements, and returns "
                "only the same JSON object requested by the originating agent."
            ),
            model=config.request_model(),
            tools=tools,
            output_guardrails=[output_json_guardrail],
        )
        agent = Agent(
            name=schema_name,
            instructions=system_prompt
            + "\nYou are running inside the OpenAI Agents SDK with PaperLens policy tools, "
            "an auditor handoff, input/output guardrails, tracing, and a SQLite session. "
            "Use the tools when you need to refresh the evidence/classification policy. "
            "Return only a JSON object matching the requested schema contract.",
            model=config.request_model(),
            tools=tools,
            handoffs=[auditor_agent],
            input_guardrails=[input_contract_guardrail],
            output_guardrails=[output_json_guardrail],
        )
        model_settings = _model_settings(ModelSettings, max_tokens)
        run_config = RunConfig(
            model=config.request_model(),
            model_provider=model_provider,
            trace_include_sensitive_data=False,
            tracing_disabled=tracing_disabled,
            model_settings=model_settings,
            workflow_name="PaperLens Agent Workflow",
            trace_metadata={
                "agent": schema_name,
                "provider": config.kind,
                "model": config.model,
                "paper_id": _extract_paper_id(user_prompt),
            },
        )
        session = _build_session(SQLiteSession, schema_name=schema_name, user_prompt=user_prompt)
        if tracing_disabled:
            result = Runner.run_sync(agent, user_prompt, run_config=run_config, session=session)
        else:
            with trace(
                "PaperLens Agent Workflow",
                metadata={"agent": schema_name, "provider": config.kind, "model": config.model},
            ):
                result = Runner.run_sync(agent, user_prompt, run_config=run_config, session=session)
            flush_traces()
    except TypeError:
        # Older SDK builds may not accept the newest model/session settings.
        tracing_disabled = config.kind == "openai-compatible"
        run_config = RunConfig(
            trace_include_sensitive_data=False,
            tracing_disabled=tracing_disabled,
            model_provider=_build_model_provider(OpenAIProvider, config),
        )
        if tracing_disabled:
            result = Runner.run_sync(agent, user_prompt, run_config=run_config)
        else:
            with trace(
                "PaperLens Agent Workflow",
                metadata={"agent": schema_name, "provider": config.kind, "model": config.model},
            ):
                result = Runner.run_sync(agent, user_prompt, run_config=run_config)
            flush_traces()
    except Exception as exc:
        raise LlmError(f"OpenAI Agents SDK run failed: {exc}") from exc

    output = result.final_output
    if isinstance(output, dict):
        data = output
        text = ""
    elif isinstance(output, str):
        text = output
        data = parse_json_text(text)
    else:
        try:
            data = dict(output)
            text = ""
        except Exception as exc:
            raise LlmError(f"Unexpected Agents SDK output type: {type(output)!r}") from exc
    return LlmJsonResult(
        data=data,
        text=text,
        request_id=_last_request_id(result),
        usage=_usage_from_result(result),
        endpoint="openai-agents-sdk",
    )


def _build_model_provider(openai_provider_type: Any, config: ProviderConfig) -> Any:
    return openai_provider_type(
        api_key=config.resolved_api_key(),
        base_url=_provider_base_url(config),
        use_responses=config.kind == "openai",
    )


def _provider_base_url(config: ProviderConfig) -> str | None:
    if not config.base_url:
        return None
    base_url = config.base_url.rstrip("/")
    parsed = urllib.parse.urlparse(base_url)
    if config.kind == "openai-compatible" and parsed.path in {"", "/"}:
        return base_url + "/v1"
    return base_url


def _build_policy_tools(function_tool: Any) -> list[Any]:
    @function_tool(name_override="paperlens_policy")
    def paperlens_policy() -> str:
        """Return the immutable PaperLens evidence and classification policy."""
        return json.dumps(
            {
                "classification": {
                    "C": "Only use C when the paper is clearly unrelated and parse evidence is adequate.",
                    "HOLD": "Use HOLD for weak parse, OCR/VLM need, missing evidence, or uncertainty.",
                    "false_negative_bias": "Prefer B/HOLD over C when related-work risk is plausible.",
                },
                "evidence": {
                    "important_claims_need_evidence_ref": True,
                    "evidence_ref_fields": ["paper_id", "page_no", "bbox", "quote_hash", "agent_run_id"],
                    "raw_pdf_access": "forbidden for synthesis; use supplied parsed artifacts only",
                },
                "claim_audit": {
                    "risky_claims": "RISKY/UNSUPPORTED claims must not become writing-ready claims.",
                    "author_vs_inference": "Do not present agent inference as author claim.",
                },
            },
            ensure_ascii=False,
        )

    @function_tool(name_override="paperlens_schema_contract")
    def paperlens_schema_contract() -> str:
        """Return the required response contract for PaperLens SDK agents."""
        return (
            "Return one JSON object only. Do not use Markdown fences. Do not invent fields "
            "outside the supplied JSON schema. Use null or conservative empty arrays when "
            "the parsed excerpts do not support a field."
        )

    return [paperlens_policy, paperlens_schema_contract]


def _build_input_guardrail(input_guardrail: Any, guardrail_output_type: Any) -> Any:
    @input_guardrail(name="paperlens_input_contract")
    def paperlens_input_contract(context: Any, agent: Any, agent_input: Any) -> Any:
        text = _input_to_text(agent_input)
        missing = []
        if "paper_id:" not in text:
            missing.append("paper_id")
        if "JSON schema contract:" not in text:
            missing.append("json_schema_contract")
        return guardrail_output_type(
            output_info={"missing": missing, "agent": getattr(agent, "name", None)},
            tripwire_triggered=bool(missing),
        )

    return paperlens_input_contract


def _build_output_guardrail(output_guardrail: Any, guardrail_output_type: Any) -> Any:
    @output_guardrail(name="paperlens_json_output_contract")
    def paperlens_json_output_contract(context: Any, agent: Any, agent_output: Any) -> Any:
        try:
            if isinstance(agent_output, dict):
                valid = True
            elif isinstance(agent_output, str):
                valid = isinstance(parse_json_text(agent_output), dict)
            else:
                valid = isinstance(dict(agent_output), dict)
        except Exception:
            valid = False
        return guardrail_output_type(
            output_info={"valid_json_object": valid, "agent": getattr(agent, "name", None)},
            tripwire_triggered=not valid,
        )

    return paperlens_json_output_contract


def _model_settings(model_settings_type: Any, max_tokens: int) -> Any:
    try:
        return model_settings_type(max_tokens=max_tokens, include_usage=True)
    except TypeError:
        return model_settings_type(max_tokens=max_tokens)


def _build_session(sqlite_session_type: Any, *, schema_name: str, user_prompt: str) -> Any:
    db_path = Path(
        os.getenv("PAPERLENS_AGENT_SESSION_DB")
        or (Path(tempfile.gettempdir()) / "paperlens_agent_sessions.sqlite")
    )
    db_path.parent.mkdir(parents=True, exist_ok=True)
    paper_id = _extract_paper_id(user_prompt) or "unknown"
    prompt_hash = hashlib.sha256(user_prompt.encode("utf-8", errors="ignore")).hexdigest()[:12]
    session_id = f"{schema_name}:{paper_id}:{prompt_hash}"
    return sqlite_session_type(session_id=session_id, db_path=db_path)


def _extract_paper_id(text: str) -> str | None:
    match = re.search(r"(?m)^paper_id:\s*([A-Za-z0-9_.:-]+)", text)
    return match.group(1) if match else None


def _input_to_text(agent_input: Any) -> str:
    if isinstance(agent_input, str):
        return agent_input
    try:
        return json.dumps(agent_input, ensure_ascii=False, default=str)
    except Exception:
        return str(agent_input)


def _last_request_id(result: Any) -> str | None:
    raw_responses = getattr(result, "raw_responses", None) or []
    for response in reversed(raw_responses):
        request_id = getattr(response, "request_id", None)
        if request_id:
            return request_id
    return None


def _usage_from_result(result: Any) -> dict[str, Any]:
    raw_responses = getattr(result, "raw_responses", None) or []
    usage: dict[str, Any] = {
        "requests": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    for response in raw_responses:
        item = getattr(response, "usage", None)
        if not item:
            continue
        usage["requests"] += int(getattr(item, "requests", 0) or 1)
        usage["input_tokens"] += int(getattr(item, "input_tokens", 0) or 0)
        usage["output_tokens"] += int(getattr(item, "output_tokens", 0) or 0)
        usage["total_tokens"] += int(getattr(item, "total_tokens", 0) or 0)
        input_details = getattr(item, "input_tokens_details", None)
        usage["cached_input_tokens"] += int(getattr(input_details, "cached_tokens", 0) or 0)
    if usage["requests"] == 0 and raw_responses:
        usage["requests"] = len(raw_responses)
    if usage["cached_input_tokens"]:
        usage["input_tokens_details"] = {"cached_tokens": usage["cached_input_tokens"]}
    return usage
