from __future__ import annotations

import base64
import contextvars
import http.client
import json
import mimetypes
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from paperlens_core.config import ProviderConfig


class LlmError(RuntimeError):
    pass


@dataclass(frozen=True)
class LlmJsonResult:
    data: dict[str, Any]
    text: str
    request_id: str | None
    usage: dict[str, Any]
    endpoint: str


_CALL_GUARD_LOCK = threading.Lock()
_CALL_GUARD_COUNTS: dict[str, int] = {}
_LAST_CALL_TIME = 0.0
_LEDGER_LOCK = threading.Lock()
_LLM_CONTEXT: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "paperlens_llm_context", default={}
)


@contextmanager
def llm_call_context(**context: Any) -> Any:
    parent = dict(_LLM_CONTEXT.get({}))
    parent.update({key: value for key, value in context.items() if value is not None})
    token = _LLM_CONTEXT.set(parent)
    try:
        yield
    finally:
        _LLM_CONTEXT.reset(token)


class JsonLlmClient:
    def __init__(
        self,
        config: ProviderConfig,
        *,
        ledger_path: Path | str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.config = config
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.run_id = run_id

    def invoke_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int = 1400,
    ) -> LlmJsonResult:
        api_key = self._validate_request_config()
        if self.config.kind in {"openai", "openai-compatible"}:
            if self.config.kind == "openai-compatible":
                return self._openai_chat_completions(
                    api_key=api_key,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    schema_name=schema_name,
                    schema=schema,
                    max_tokens=max_tokens,
                )
            return self._openai_responses(
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_name=schema_name,
                schema=schema,
                max_tokens=max_tokens,
            )
        if self.config.kind in {"anthropic", "anthropic-compatible"}:
            return self._anthropic_messages(
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
                max_tokens=max_tokens,
            )
        raise LlmError(f"Unsupported provider kind: {self.config.kind}")

    def invoke_json_with_images(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[Path],
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int = 1800,
        detail: str = "high",
    ) -> LlmJsonResult:
        if not env_flag("PAPERLENS_ALLOW_IMAGE_INPUTS", default=False):
            raise LlmError(
                "Model image inputs are disabled by PAPERLENS_ALLOW_IMAGE_INPUTS. "
                "PaperLens will not send page images unless this is explicitly enabled."
            )
        api_key = self._validate_request_config()
        image_payloads = [image_data_url(path) for path in image_paths if path.exists()]
        if not image_payloads:
            raise LlmError("No image inputs were available for multimodal request")
        if self.config.kind == "openai":
            return self._openai_responses_with_images(
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_data_urls=image_payloads,
                schema_name=schema_name,
                schema=schema,
                max_tokens=max_tokens,
                detail=detail,
            )
        if self.config.kind == "openai-compatible":
            return self._openai_chat_completions_with_images(
                api_key=api_key,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                image_data_urls=image_payloads,
                schema_name=schema_name,
                schema=schema,
                max_tokens=max_tokens,
                detail=detail,
            )
        raise LlmError(f"Image inputs are not implemented for provider kind: {self.config.kind}")

    def _validate_request_config(self) -> str:
        if self.config.kind == "none":
            raise LlmError("Model reading is disabled")
        if not self.config.model:
            raise LlmError("Missing model name for model reading")
        if self.config.kind in {"openai-compatible", "anthropic-compatible"} and not self.config.base_url:
            raise LlmError(f"Missing base URL for provider {self.config.kind}")
        api_key = self.config.resolved_api_key()
        if not api_key:
            raise LlmError(f"Missing API key for provider {self.config.kind}")
        return api_key

    def _openai_responses(
        self,
        *,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> LlmJsonResult:
        endpoint = f"{self._base_url('https://api.openai.com/v1')}/responses"
        payload = {
            "model": self.config.request_model(),
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": max_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        response, headers = self._post_json(
            endpoint,
            payload,
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        text = extract_openai_response_text(response)
        return LlmJsonResult(
            data=parse_json_text_for_schema(text, schema_name, schema),
            text=text,
            request_id=headers.get("x-request-id"),
            usage=dict(response.get("usage") or {}),
            endpoint=endpoint,
        )

    def _openai_responses_with_images(
        self,
        *,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        image_data_urls: list[str],
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int,
        detail: str,
    ) -> LlmJsonResult:
        endpoint = f"{self._base_url('https://api.openai.com/v1')}/responses"
        content: list[dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]
        for data_url in image_data_urls:
            content.append({"type": "input_image", "image_url": data_url, "detail": detail})
        payload = {
            "model": self.config.request_model(),
            "instructions": system_prompt,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": max_tokens,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        response, headers = self._post_json(
            endpoint,
            payload,
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        text = extract_openai_response_text(response)
        return LlmJsonResult(
            data=parse_json_text_for_schema(text, schema_name, schema),
            text=text,
            request_id=headers.get("x-request-id"),
            usage=dict(response.get("usage") or {}),
            endpoint=endpoint,
        )

    def _openai_chat_completions(
        self,
        *,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> LlmJsonResult:
        endpoint = f"{self._base_url('https://api.openai.com/v1')}/chat/completions"
        payload = {
            "model": self.config.request_model(),
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        payload = self._apply_compatible_chat_options(payload)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            response, response_headers = self._post_json(endpoint, payload, headers)
        except LlmError as exc:
            if "400" not in str(exc) or not env_flag("PAPERLENS_ALLOW_SCHEMA_FALLBACK"):
                raise
            fallback_payload = chat_json_schema_fallback_payload(
                payload=payload,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
            )
            response, response_headers = self._post_json(endpoint, fallback_payload, headers)
        try:
            text, data = parse_chat_completion_json(response, schema_name, schema)
        except LlmError:
            if not env_flag("PAPERLENS_ALLOW_JSON_RETRY"):
                raise
            fallback_payload = chat_json_schema_fallback_payload(
                payload=payload,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=schema,
            )
            retry_payload = expand_completion_budget(
                fallback_payload,
                minimum=json_retry_completion_minimum(schema_name, max_tokens, has_images=False),
            )
            response, response_headers = self._post_json(endpoint, retry_payload, headers)
            text, data = parse_chat_completion_json(response, schema_name, schema)
        return LlmJsonResult(
            data=data,
            text=text,
            request_id=response_headers.get("x-request-id"),
            usage=dict(response.get("usage") or {}),
            endpoint=endpoint,
        )

    def _openai_chat_completions_with_images(
        self,
        *,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        image_data_urls: list[str],
        schema_name: str,
        schema: dict[str, Any],
        max_tokens: int,
        detail: str,
    ) -> LlmJsonResult:
        endpoint = f"{self._base_url('https://api.openai.com/v1')}/chat/completions"
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]
        for data_url in image_data_urls:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url, "detail": detail},
                }
            )
        payload = {
            "model": self.config.request_model(),
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        payload = self._apply_compatible_chat_options(payload)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response, response_headers = self._post_json(endpoint, payload, headers)
        try:
            text, data = parse_chat_completion_json(response, schema_name, schema)
        except LlmError:
            if not env_flag("PAPERLENS_ALLOW_JSON_RETRY"):
                raise
            fallback_payload = chat_json_schema_fallback_payload_with_images(
                payload=payload,
                system_prompt=system_prompt,
                user_content=user_content,
                schema=schema,
            )
            retry_payload = expand_completion_budget(
                fallback_payload,
                minimum=json_retry_completion_minimum(schema_name, max_tokens, has_images=True),
            )
            response, response_headers = self._post_json(endpoint, retry_payload, headers)
            text, data = parse_chat_completion_json(response, schema_name, schema)
        return LlmJsonResult(
            data=data,
            text=text,
            request_id=response_headers.get("x-request-id"),
            usage=dict(response.get("usage") or {}),
            endpoint=endpoint,
        )

    def _anthropic_messages(
        self,
        *,
        api_key: str,
        system_prompt: str,
        user_prompt: str,
        schema: dict[str, Any],
        max_tokens: int,
    ) -> LlmJsonResult:
        endpoint = f"{self._base_url('https://api.anthropic.com/v1')}/messages"
        payload = {
            "model": self.config.request_model(),
            "max_tokens": max_tokens,
            "system": system_prompt
            + "\nReturn only valid JSON. Do not wrap the JSON in Markdown fences.",
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt
                    + "\n\nJSON schema to follow:\n"
                    + json.dumps(schema, ensure_ascii=False),
                }
            ],
        }
        response, headers = self._post_json(
            endpoint,
            payload,
            {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )
        text = extract_anthropic_text(response)
        return LlmJsonResult(
            data=parse_json_text_for_schema(text, "paperlens_anthropic_json", schema),
            text=text,
            request_id=headers.get("request-id"),
            usage=dict(response.get("usage") or {}),
            endpoint=endpoint,
        )

    def _base_url(self, default: str) -> str:
        if self.config.base_url:
            return self.config.base_url.rstrip("/")
        return default.rstrip("/")

    def _apply_compatible_chat_options(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.config.kind != "openai-compatible" or not self._is_mimo_provider():
            return payload
        payload = dict(payload)
        max_tokens = payload.get("max_tokens")
        if max_tokens is not None and env_flag("PAPERLENS_MIMO_USE_MAX_COMPLETION_TOKENS"):
            payload.pop("max_tokens", None)
            payload["max_completion_tokens"] = max_tokens
        thinking = self._mimo_thinking_option(payload)
        if thinking is not None:
            payload["thinking"] = thinking
        else:
            payload.pop("thinking", None)
        return payload

    def _is_mimo_provider(self) -> bool:
        if os.getenv("PAPERLENS_DISABLE_MIMO_COMPAT", "0") == "1":
            return False
        if os.getenv("PAPERLENS_FORCE_MIMO_COMPAT", "0") == "1":
            return True
        model = self.config.model.lower()
        return model.startswith("mimo-")

    def _mimo_thinking_option(self, payload: dict[str, Any]) -> dict[str, str] | None:
        mode = os.getenv("PAPERLENS_MIMO_THINKING", "omit").strip().lower()
        schema_name = compatible_payload_schema_name(payload)
        schema_filter = normalized_csv_env("PAPERLENS_MIMO_THINKING_SCHEMAS")
        if schema_filter and schema_name not in schema_filter:
            mode = "omit"
        if mode in {"1", "true", "yes", "on", "enabled", "enable"}:
            return {"type": "enabled"}
        if mode in {"default", "provider", "omit"}:
            return None
        if mode in {"0", "false", "no", "off", "disabled", "disable"}:
            return {"type": "disabled"}
        return {"type": "disabled"}

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        enforce_request_safety(endpoint, payload, body_bytes)
        retry_codes = {429, 500, 502, 503, 504}
        try:
            attempts = int(os.getenv("PAPERLENS_LLM_RETRIES", str(self.config.max_retries)))
        except ValueError:
            attempts = self.config.max_retries
        attempts = max(1, min(3, attempts))
        last_error: Exception | None = None
        try:
            timeout_seconds = int(os.getenv("PAPERLENS_LLM_TIMEOUT_SECONDS", str(self.config.timeout_seconds)))
        except ValueError:
            timeout_seconds = self.config.timeout_seconds
        timeout_seconds = max(10, min(timeout_seconds, 1800))
        for attempt in range(attempts):
            before_model_attempt(run_id=self.run_id)
            request = urllib.request.Request(
                endpoint,
                data=body_bytes,
                headers=headers,
                method="POST",
            )
            started = time.time()
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                    parsed = json.loads(body)
                    response_headers = {
                        key.lower(): value for key, value in response.headers.items()
                    }
                    write_llm_ledger(
                        endpoint=endpoint,
                        payload=payload,
                        payload_bytes=len(body_bytes),
                        attempt=attempt + 1,
                        attempts=attempts,
                        status="ok",
                        duration_seconds=time.time() - started,
                        response_usage=parsed.get("usage") if isinstance(parsed, dict) else None,
                        request_id=response_headers.get("x-request-id"),
                        ledger_path=self.ledger_path,
                        run_id=self.run_id,
                    )
                    return parsed, response_headers
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = LlmError(f"HTTP {exc.code} from {endpoint}: {body[:1000]}")
                write_llm_ledger(
                    endpoint=endpoint,
                    payload=payload,
                    payload_bytes=len(body_bytes),
                    attempt=attempt + 1,
                    attempts=attempts,
                    status="http_error",
                    duration_seconds=time.time() - started,
                    error=f"HTTP {exc.code}: {body[:300]}",
                    ledger_path=self.ledger_path,
                    run_id=self.run_id,
                )
                if exc.code not in retry_codes or attempt == attempts - 1:
                    raise last_error from exc
                retry_delay = self._retry_delay(attempt, exc.headers)
            except urllib.error.URLError as exc:
                last_error = LlmError(f"Network error calling {endpoint}: {exc}")
                write_llm_ledger(
                    endpoint=endpoint,
                    payload=payload,
                    payload_bytes=len(body_bytes),
                    attempt=attempt + 1,
                    attempts=attempts,
                    status="network_error",
                    duration_seconds=time.time() - started,
                    error=str(exc),
                    ledger_path=self.ledger_path,
                    run_id=self.run_id,
                )
                if attempt == attempts - 1:
                    raise last_error from exc
                retry_delay = self._retry_delay(attempt)
            except (http.client.RemoteDisconnected, ConnectionError, OSError) as exc:
                last_error = LlmError(f"Network connection closed calling {endpoint}: {exc}")
                write_llm_ledger(
                    endpoint=endpoint,
                    payload=payload,
                    payload_bytes=len(body_bytes),
                    attempt=attempt + 1,
                    attempts=attempts,
                    status="connection_error",
                    duration_seconds=time.time() - started,
                    error=str(exc),
                    ledger_path=self.ledger_path,
                    run_id=self.run_id,
                )
                if attempt == attempts - 1:
                    raise last_error from exc
                retry_delay = self._retry_delay(attempt)
            except TimeoutError as exc:
                last_error = LlmError(f"Timeout calling {endpoint}")
                write_llm_ledger(
                    endpoint=endpoint,
                    payload=payload,
                    payload_bytes=len(body_bytes),
                    attempt=attempt + 1,
                    attempts=attempts,
                    status="timeout",
                    duration_seconds=time.time() - started,
                    error="timeout",
                    ledger_path=self.ledger_path,
                    run_id=self.run_id,
                )
                if attempt == attempts - 1:
                    raise last_error from exc
                retry_delay = self._retry_delay(attempt)
            except json.JSONDecodeError as exc:
                write_llm_ledger(
                    endpoint=endpoint,
                    payload=payload,
                    payload_bytes=len(body_bytes),
                    attempt=attempt + 1,
                    attempts=attempts,
                    status="invalid_provider_json",
                    duration_seconds=time.time() - started,
                    error=str(exc),
                    ledger_path=self.ledger_path,
                    run_id=self.run_id,
                )
                raise LlmError(f"Provider returned non-JSON response from {endpoint}") from exc
            time.sleep(retry_delay)
        if last_error:
            raise last_error
        raise LlmError(f"Provider request failed for {endpoint}")

    def _retry_delay(self, attempt: int, headers: Any | None = None) -> float:
        retry_after = None
        if headers is not None:
            try:
                retry_after = headers.get("retry-after") or headers.get("Retry-After")
            except AttributeError:
                retry_after = None
        if retry_after:
            try:
                return min(max(float(retry_after), 1.0), 120.0)
            except ValueError:
                pass
        return min(10.0 * (2**attempt), 90.0)


def env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on", "enabled", "enable"}


def bounded_int_env(name: str, *, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def enforce_request_safety(endpoint: str, payload: dict[str, Any], body_bytes: bytes) -> None:
    max_payload_bytes = bounded_int_env(
        "PAPERLENS_MAX_REQUEST_BYTES", default=1_000_000, minimum=20_000, maximum=20_000_000
    )
    if len(body_bytes) > max_payload_bytes:
        raise LlmError(
            f"Refusing oversized model request: {len(body_bytes)} bytes exceeds "
            f"PAPERLENS_MAX_REQUEST_BYTES={max_payload_bytes} for {endpoint}"
        )

    prompt_chars = payload_text_chars(payload)
    max_prompt_chars = bounded_int_env(
        "PAPERLENS_MAX_PROMPT_CHARS", default=260_000, minimum=20_000, maximum=5_000_000
    )
    if prompt_chars > max_prompt_chars:
        raise LlmError(
            f"Refusing oversized model prompt: {prompt_chars} text chars exceeds "
            f"PAPERLENS_MAX_PROMPT_CHARS={max_prompt_chars}"
        )

    image_count = payload_image_count(payload)
    if image_count and not env_flag("PAPERLENS_ALLOW_IMAGE_INPUTS", default=False):
        raise LlmError("Refusing model request with image inputs; set PAPERLENS_ALLOW_IMAGE_INPUTS=1 to allow it")
    max_images = bounded_int_env("PAPERLENS_MAX_IMAGES_PER_REQUEST", default=1, minimum=0, maximum=20)
    if image_count > max_images:
        raise LlmError(
            f"Refusing model request with {image_count} images; "
            f"PAPERLENS_MAX_IMAGES_PER_REQUEST={max_images}"
        )

    output_limit = payload_completion_limit(payload)
    max_output_limit = bounded_int_env(
        "PAPERLENS_MAX_COMPLETION_TOKENS", default=12_000, minimum=512, maximum=200_000
    )
    if output_limit is not None and output_limit > max_output_limit:
        raise LlmError(
            f"Refusing model request with completion limit {output_limit}; "
            f"PAPERLENS_MAX_COMPLETION_TOKENS={max_output_limit}"
        )


def before_model_attempt(*, run_id: str | None = None) -> None:
    global _LAST_CALL_TIME
    with _CALL_GUARD_LOCK:
        max_calls = bounded_int_env("PAPERLENS_MAX_MODEL_CALLS", default=0, minimum=0, maximum=100_000)
        context = dict(_LLM_CONTEXT.get({}))
        guard_key = str(run_id or context.get("run_id") or "__process__")
        guard_count = _CALL_GUARD_COUNTS.get(guard_key, 0)
        if max_calls and guard_count >= max_calls:
            raise LlmError(
                f"Refusing model request: run has reached PAPERLENS_MAX_MODEL_CALLS={max_calls}"
            )
        min_interval = float_env("PAPERLENS_MIN_SECONDS_BETWEEN_CALLS", default=0.25, minimum=0.0, maximum=60.0)
        now = time.time()
        wait_seconds = min_interval - (now - _LAST_CALL_TIME)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            now = time.time()
        _CALL_GUARD_COUNTS[guard_key] = guard_count + 1
        _LAST_CALL_TIME = now


def float_env(name: str, *, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        value = default
    return max(minimum, min(value, maximum))


def payload_text_chars(value: Any) -> int:
    if isinstance(value, str):
        if value.startswith("data:image/"):
            return 0
        return len(value)
    if isinstance(value, dict):
        return sum(payload_text_chars(item) for item in value.values())
    if isinstance(value, list):
        return sum(payload_text_chars(item) for item in value)
    return 0


def payload_image_count(value: Any) -> int:
    if isinstance(value, str):
        return 1 if value.startswith("data:image/") else 0
    if isinstance(value, dict):
        return sum(payload_image_count(item) for item in value.values())
    if isinstance(value, list):
        return sum(payload_image_count(item) for item in value)
    return 0


def payload_completion_limit(payload: dict[str, Any]) -> int | None:
    for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    return None


def payload_schema_name(payload: dict[str, Any]) -> str:
    response_format = payload.get("response_format")
    if isinstance(response_format, dict):
        json_schema = response_format.get("json_schema")
        if isinstance(json_schema, dict):
            return str(json_schema.get("name") or "")
    text = payload.get("text")
    if isinstance(text, dict):
        fmt = text.get("format")
        if isinstance(fmt, dict):
            return str(fmt.get("name") or "")
    return ""


def write_llm_ledger(
    *,
    endpoint: str,
    payload: dict[str, Any],
    payload_bytes: int,
    attempt: int,
    attempts: int,
    status: str,
    duration_seconds: float,
    response_usage: Any | None = None,
    request_id: str | None = None,
    error: str | None = None,
    ledger_path: Path | str | None = None,
    run_id: str | None = None,
) -> None:
    context = dict(_LLM_CONTEXT.get({}))
    resolved_ledger_path = ledger_path or context.get("ledger_path") or os.getenv("PAPERLENS_LLM_LEDGER")
    if not resolved_ledger_path:
        return
    sample_rate = float_env(
        "PAPERLENS_LLM_LEDGER_SAMPLE_RATE", default=1.0, minimum=0.0, maximum=1.0
    )
    if status == "ok" and sample_rate < 1.0 and random.random() > sample_rate:
        return
    if run_id and "run_id" not in context:
        context["run_id"] = run_id
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "endpoint": endpoint,
        "model": payload.get("model"),
        "schema_name": payload_schema_name(payload),
        "context": context,
        "payload_bytes": payload_bytes,
        "prompt_chars": payload_text_chars(payload),
        "image_count": payload_image_count(payload),
        "completion_limit": payload_completion_limit(payload),
        "attempt": attempt,
        "attempts": attempts,
        "status": status,
        "duration_seconds": round(duration_seconds, 3),
        "request_id": request_id,
        "usage": response_usage if isinstance(response_usage, dict) else {},
    }
    if error:
        record["error"] = error[:500]
    path = Path(resolved_ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LEDGER_LOCK:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def chat_json_schema_fallback_payload(
    *,
    payload: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    fallback_payload = dict(payload)
    fallback_payload["response_format"] = {"type": "json_object"}
    fallback_payload["messages"] = [
        {"role": "system", "content": system_prompt + "\nReturn only valid JSON matching the schema."},
        {
            "role": "user",
            "content": user_prompt
            + "\n\nJSON schema to follow exactly:\n"
            + json.dumps(schema, ensure_ascii=False),
        },
    ]
    return fallback_payload


def chat_json_schema_fallback_payload_with_images(
    *,
    payload: dict[str, Any],
    system_prompt: str,
    user_content: list[dict[str, Any]],
    schema: dict[str, Any],
) -> dict[str, Any]:
    fallback_content = list(user_content)
    fallback_content.append(
        {
            "type": "text",
            "text": "\n\nJSON schema to follow exactly:\n" + json.dumps(schema, ensure_ascii=False),
        }
    )
    fallback_payload = dict(payload)
    fallback_payload["response_format"] = {"type": "json_object"}
    fallback_payload["messages"] = [
        {"role": "system", "content": system_prompt + "\nReturn only valid JSON matching the schema."},
        {"role": "user", "content": fallback_content},
    ]
    return fallback_payload


def extract_openai_response_text(response: dict[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return response["output_text"]
    chunks: list[str] = []
    for item in response.get("output", []) or []:
        for content in item.get("content", []) or []:
            if isinstance(content.get("text"), str):
                chunks.append(content["text"])
            elif isinstance(content.get("refusal"), str):
                chunks.append(content["refusal"])
    if chunks:
        return "\n".join(chunks)
    raise LlmError("OpenAI response did not contain text output")


def extract_chat_completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise LlmError("Chat completion response did not contain choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = [part.get("text") for part in content if isinstance(part, dict) and part.get("text")]
        if chunks:
            return "\n".join(chunks)
    raise LlmError("Chat completion response did not contain text content")


def parse_chat_completion_json(
    response: dict[str, Any],
    schema_name: str,
    schema: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    finish_reason = chat_completion_finish_reason(response)
    text = extract_chat_completion_text(response)
    if finish_reason in {"length", "max_tokens"}:
        raise LlmError(
            f"Model response was truncated before JSON completed (finish_reason={finish_reason})"
        )
    return text, parse_json_text_for_schema(text, schema_name, schema)


def chat_completion_finish_reason(response: dict[str, Any]) -> str | None:
    choices = response.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        return None
    reason = choices[0].get("finish_reason")
    return str(reason) if reason is not None else None


def expand_completion_budget(payload: dict[str, Any], *, minimum: int) -> dict[str, Any]:
    expanded = dict(payload)
    minimum = max(1, min(minimum, 12000))
    for key in ("max_completion_tokens", "max_tokens"):
        value = expanded.get(key)
        if isinstance(value, int):
            expanded[key] = max(value, minimum)
            return expanded
    expanded["max_tokens"] = minimum
    return expanded


def json_retry_completion_minimum(
    schema_name: str, max_tokens: int, *, has_images: bool
) -> int:
    floor = 1800 if has_images else 1600
    if schema_name in {"paperlens_rolling_memory", "paperlens_memory_patch_set"}:
        floor = 4000
    if schema_name in {
        "paperlens_report_plan",
        "paperlens_report_section",
        "paperlens_report_section_audit",
        "paperlens_library_answer",
    }:
        floor = 3000
    return max(max_tokens * 2, floor)


def extract_anthropic_text(response: dict[str, Any]) -> str:
    chunks = []
    for item in response.get("content", []) or []:
        if isinstance(item, dict) and item.get("type") == "text":
            chunks.append(str(item.get("text") or ""))
    if chunks:
        return "\n".join(chunks)
    raise LlmError("Anthropic response did not contain text content")


def image_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def compatible_payload_schema_name(payload: dict[str, Any]) -> str:
    response_format = payload.get("response_format")
    if not isinstance(response_format, dict):
        return ""
    json_schema = response_format.get("json_schema")
    if not isinstance(json_schema, dict):
        return ""
    return str(json_schema.get("name") or "")


def normalized_csv_env(name: str) -> set[str]:
    raw = os.getenv(name, "")
    return {item.strip() for item in raw.split(",") if item.strip()}


def parse_json_text(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        if start < 0:
            raise LlmError("Model response did not contain a JSON object")
        try:
            parsed, _end = json.JSONDecoder().raw_decode(cleaned[start:])
        except json.JSONDecodeError as exc:
            end = cleaned.rfind("}")
            if end <= start:
                raise LlmError("Model response did not contain a JSON object") from exc
            try:
                parsed = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as exc:
                raise LlmError("Model response JSON could not be parsed") from exc
    if not isinstance(parsed, dict):
        raise LlmError("Model response JSON root must be an object")
    return parsed


def parse_json_text_for_schema(
    text: str,
    schema_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    data = parse_json_text(text)
    apply_schema_compatible_defaults(data, schema)
    missing = missing_required_schema_keys(data, schema)
    if missing:
        raise LlmError(f"Provider JSON did not match schema {schema_name}; missing keys: {', '.join(missing)}")
    return data


def apply_schema_compatible_defaults(data: dict[str, Any], schema: dict[str, Any]) -> None:
    properties = schema.get("properties") or {}
    if not isinstance(properties, dict):
        return
    for key in missing_required_schema_keys(data, schema):
        prop = properties.get(key)
        if not isinstance(prop, dict):
            continue
        if schema_property_allows_null(prop):
            data[key] = None
        elif prop.get("type") == "array":
            data[key] = []
        elif prop.get("type") == "object":
            data[key] = {}


def schema_property_allows_null(prop: dict[str, Any]) -> bool:
    prop_type = prop.get("type")
    if prop_type == "null":
        return True
    if isinstance(prop_type, list):
        return "null" in prop_type
    return False


def missing_required_schema_keys(data: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    required = schema.get("required") or []
    if not isinstance(required, list):
        return []
    return [str(key) for key in required if isinstance(key, str) and key not in data]
