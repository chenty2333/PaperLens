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
from typing import Any, Iterable

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
        max_tokens: int | None = None,
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
        max_tokens: int | None = None,
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
        if (
            self.config.kind in {"openai-compatible", "anthropic-compatible"}
            and not self.config.base_url
        ):
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
        max_tokens: int | None,
    ) -> LlmJsonResult:
        endpoint = f"{self._base_url('https://api.openai.com/v1')}/responses"
        payload = {
            "model": self.config.request_model(),
            "instructions": system_prompt,
            "input": user_prompt,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
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
        max_tokens: int | None,
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
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        }
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
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
        max_tokens: int | None,
    ) -> LlmJsonResult:
        endpoint = f"{self._base_url('https://api.openai.com/v1')}/chat/completions"
        payload = {
            "model": self.config.request_model(),
            "temperature": 0,
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
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload = self._apply_compatible_chat_options(payload)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response, response_headers = self._post_json(endpoint, payload, headers)
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
        max_tokens: int | None,
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
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        payload = self._apply_compatible_chat_options(payload)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        response, response_headers = self._post_json(endpoint, payload, headers)
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
        max_tokens: int | None,
    ) -> LlmJsonResult:
        endpoint = f"{self._base_url('https://api.anthropic.com/v1')}/messages"
        payload = {
            "model": self.config.request_model(),
            "max_tokens": anthropic_completion_limit(max_tokens),
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
        payload.pop("_paperlens_schema_name", None)
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
        schema_name = str(
            payload.get("_paperlens_schema_name") or compatible_payload_schema_name(payload)
        )
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
            timeout_seconds = int(
                os.getenv("PAPERLENS_LLM_TIMEOUT_SECONDS", str(self.config.timeout_seconds))
            )
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
        raise LlmError(
            "Refusing model request with image inputs; set PAPERLENS_ALLOW_IMAGE_INPUTS=1 to allow it"
        )
    max_images = bounded_int_env(
        "PAPERLENS_MAX_IMAGES_PER_REQUEST", default=1, minimum=0, maximum=20
    )
    if image_count > max_images:
        raise LlmError(
            f"Refusing model request with {image_count} images; "
            f"PAPERLENS_MAX_IMAGES_PER_REQUEST={max_images}"
        )


def before_model_attempt(*, run_id: str | None = None) -> None:
    global _LAST_CALL_TIME
    with _CALL_GUARD_LOCK:
        max_calls = bounded_int_env(
            "PAPERLENS_MAX_MODEL_CALLS", default=0, minimum=0, maximum=100_000
        )
        context = dict(_LLM_CONTEXT.get({}))
        guard_key = str(run_id or context.get("run_id") or "__process__")
        guard_count = _CALL_GUARD_COUNTS.get(guard_key, 0)
        if max_calls and guard_count >= max_calls:
            raise LlmError(
                f"Refusing model request: run has reached PAPERLENS_MAX_MODEL_CALLS={max_calls}"
            )
        min_interval = float_env(
            "PAPERLENS_MIN_SECONDS_BETWEEN_CALLS", default=0.25, minimum=0.0, maximum=60.0
        )
        now = time.time()
        wait_seconds = min_interval - (now - _LAST_CALL_TIME)
        if wait_seconds > 0:
            time.sleep(wait_seconds)
            now = time.time()
        _CALL_GUARD_COUNTS[guard_key] = guard_count + 1
        _LAST_CALL_TIME = now


def mark_model_attempt_finished() -> None:
    global _LAST_CALL_TIME
    with _CALL_GUARD_LOCK:
        _LAST_CALL_TIME = max(_LAST_CALL_TIME, time.time())


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
    mark_model_attempt_finished()
    context = dict(_LLM_CONTEXT.get({}))
    resolved_ledger_path = (
        ledger_path or context.get("ledger_path") or os.getenv("PAPERLENS_LLM_LEDGER")
    )
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
        chunks = [
            part.get("text") for part in content if isinstance(part, dict) and part.get("text")
        ]
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


def anthropic_completion_limit(max_tokens: int | None) -> int:
    if max_tokens is not None:
        return max(1, int(max_tokens))
    return bounded_int_env(
        "PAPERLENS_ANTHROPIC_COMPLETION_TOKENS",
        default=100_000,
        minimum=1_000,
        maximum=200_000,
    )


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
    candidates = list(iter_json_object_candidates(text))
    if not candidates:
        raise LlmError("Model response did not contain a JSON object")
    return candidates[0]


def parse_json_text_for_schema(
    text: str,
    schema_name: str,
    schema: dict[str, Any],
) -> dict[str, Any]:
    candidates = list(iter_json_object_candidates(text))
    if not candidates:
        raise LlmError("Model response did not contain a JSON object")
    best_missing: list[str] | None = None
    for candidate in candidates:
        data = dict(candidate)
        data = coerce_artifact_envelope_payload(data, schema)
        apply_schema_compatible_defaults(data, schema, schema_name=schema_name)
        missing = missing_required_schema_paths(data, schema)
        if not missing:
            return data
        if best_missing is None or len(missing) < len(best_missing):
            best_missing = missing
    missing_text = ", ".join(best_missing or [])
    raise LlmError(
        f"Provider JSON did not match schema {schema_name}; missing keys: {missing_text}"
    )


def coerce_artifact_envelope_payload(data: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return data
    required = schema.get("required")
    if not isinstance(required, list):
        return data
    envelope_keys = {"artifact_type", "artifact_version", "producer", "data"}
    if not envelope_keys.issubset({str(item) for item in required}):
        return data

    data_schema = properties.get("data")
    artifact_type_schema = properties.get("artifact_type")
    if not isinstance(data_schema, dict) or not isinstance(artifact_type_schema, dict):
        return data
    artifact_type = artifact_type_from_schema(artifact_type_schema)
    if not artifact_type:
        return data
    if isinstance(data.get("artifact_type"), str) and data["artifact_type"].strip():
        artifact_type = data["artifact_type"].strip()
    if isinstance(data.get("data"), dict):
        envelope_data = normalize_artifact_data_payload(data["data"], artifact_type)
        result = dict(data)
        result["artifact_type"] = artifact_type
        result.setdefault("artifact_version", "v2")
        result.setdefault("producer", "paperlens_llm_adapter")
        result["data"] = envelope_data
        return result

    payload = {key: value for key, value in data.items() if key not in envelope_keys}
    payload = normalize_artifact_data_payload(payload, artifact_type)
    if missing_required_schema_keys(payload, data_schema):
        return data
    return {
        "artifact_type": artifact_type,
        "artifact_version": "v2",
        "producer": "paperlens_llm_adapter",
        "data": payload,
    }


def artifact_type_from_schema(schema: dict[str, Any]) -> str | None:
    enum = schema.get("enum")
    if isinstance(enum, list) and len(enum) == 1:
        value = str(enum[0]).strip()
        return value or None
    const = schema.get("const")
    if isinstance(const, str) and const.strip():
        return const.strip()
    return None


def normalize_artifact_data_payload(data: dict[str, Any], artifact_type: str) -> dict[str, Any]:
    if artifact_type == "observation_cards":
        cards = data.get("cards")
        if not isinstance(cards, list):
            cards = data.get("observations")
        if not isinstance(cards, list):
            return data
        return {
            "cards": [
                normalize_observation_card_payload(card)
                for card in cards
                if isinstance(card, dict)
            ]
        }
    if artifact_type == "relation_candidates":
        candidates = data.get("candidates")
        if not isinstance(candidates, list):
            candidates = data.get("relations")
        if not isinstance(candidates, list):
            candidates = data.get("edges")
        if not isinstance(candidates, list):
            return data
        return {
            "candidates": [
                normalize_relation_candidate_payload(candidate)
                for candidate in candidates
                if isinstance(candidate, dict)
            ]
        }
    return data


def normalize_observation_card_payload(card: dict[str, Any]) -> dict[str, Any]:
    statement = first_text_value(
        card,
        (
            "statement",
            "observation",
            "claim",
            "finding",
            "reasoning",
            "description",
            "rationale",
            "explanation",
            "details",
            "detailed_mechanism",
            "analysis",
            "answer",
            "summary",
            "content",
            "text",
            "what_i_saw",
        ),
    )
    source_ids = card.get("source_ids") if isinstance(card.get("source_ids"), list) else []
    if not source_ids:
        source_ids = (
            card.get("cited_source_ids")
            if isinstance(card.get("cited_source_ids"), list)
            else []
        )
    if not source_ids:
        source_ids = (
            card.get("evidence_ids") if isinstance(card.get("evidence_ids"), list) else []
        )
    if not source_ids:
        source_ids = (
            card.get("citation_ids") if isinstance(card.get("citation_ids"), list) else []
        )
    if not source_ids:
        source_ids = (
            card.get("paper_dom_source_ids")
            if isinstance(card.get("paper_dom_source_ids"), list)
            else []
        )
    if not source_ids:
        source_ids = card.get("sources") if isinstance(card.get("sources"), list) else []
    if not source_ids and isinstance(card.get("source_id"), str):
        source_ids = [card.get("source_id")]
    if not source_ids:
        source_ids = source_ids_from_citations(card.get("citations"))
    observation_type = (
        card.get("observation_type")
        or card.get("card_type")
        or card.get("type")
        or observation_type_from_card_id(card.get("card_id"))
        or observation_type_from_outputs(card.get("covered_outputs"))
    )
    return {
        "observation_type": observation_type,
        "statement": statement,
        "source_ids": source_ids,
        "confidence": card.get("confidence") or "medium",
        "provenance": card.get("provenance") or "explicit",
        "uncertainty": card.get("uncertainty") if "uncertainty" in card else None,
        "covered_outputs": (
            card.get("covered_outputs") if isinstance(card.get("covered_outputs"), list) else []
        ),
        "extracted_numbers": (
            card.get("extracted_numbers") if isinstance(card.get("extracted_numbers"), list) else []
        ),
    }


def first_text_value(card: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = card.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


OBSERVATION_TYPE_VALUES = {
    "problem",
    "claim",
    "mechanism",
    "implementation",
    "evaluation",
    "result",
    "limitation",
    "concept",
}


def observation_type_from_card_id(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    prefix = re.split(r"[_:\-\s]+", text, maxsplit=1)[0]
    return prefix if prefix in OBSERVATION_TYPE_VALUES else None


def observation_type_from_outputs(value: Any) -> str | None:
    if not isinstance(value, list):
        return None
    for item in value:
        output = str(item or "").strip().lower()
        if output in OBSERVATION_TYPE_VALUES:
            return output
    return None


def source_ids_from_citations(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    result: list[str] = []
    for item in value:
        if isinstance(item, str):
            source_id = item.strip()
            if source_id and source_id not in result:
                result.append(source_id)
            continue
        if not isinstance(item, dict):
            continue
        source_id = str(item.get("source_id") or "").strip()
        if source_id and source_id not in result:
            result.append(source_id)
    return result


def normalize_relation_candidate_payload(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_observation_id": candidate.get("source_observation_id"),
        "target_observation_id": candidate.get("target_observation_id"),
        "kind": candidate.get("kind"),
        "confidence": candidate.get("confidence") or "medium",
    }


def iter_json_object_candidates(text: str) -> Iterable[dict[str, Any]]:
    cleaned = text.strip()
    seen: set[str] = set()
    for candidate in json_candidate_strings(cleaned):
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            yield parsed


def json_candidate_strings(text: str) -> Iterable[str]:
    fence_pattern = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL | re.IGNORECASE)
    for match in fence_pattern.finditer(text):
        yield match.group(1).strip()

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            _parsed, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        yield text[index : index + end].strip()


def apply_schema_compatible_defaults(
    data: dict[str, Any], schema: dict[str, Any], *, schema_name: str = ""
) -> None:
    if schema_name == "paperlens_paper_question":
        apply_paper_question_defaults(data)
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


def apply_paper_question_defaults(data: dict[str, Any]) -> None:
    if "answer_markdown" not in data:
        data["answer_markdown"] = ""
    if "confidence" not in data or data.get("confidence") not in {"high", "medium", "low"}:
        data["confidence"] = "low"


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


def missing_required_schema_paths(value: Any, schema: dict[str, Any], path: str = "") -> list[str]:
    if not isinstance(schema, dict):
        return []
    missing: list[str] = []
    properties = schema.get("properties")
    if isinstance(value, dict) and isinstance(properties, dict):
        for key in missing_required_schema_keys(value, schema):
            missing.append(join_schema_path(path, key))
        for key, prop_schema in properties.items():
            if key in value and isinstance(prop_schema, dict):
                missing.extend(
                    missing_required_schema_paths(
                        value[key],
                        prop_schema,
                        join_schema_path(path, key),
                    )
                )
    items = schema.get("items")
    if isinstance(value, list) and isinstance(items, dict):
        min_items = schema.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            missing.append(f"{path}:minItems={min_items}" if path else f"minItems={min_items}")
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]" if path else f"[{index}]"
            missing.extend(missing_required_schema_paths(item, items, item_path))
    return missing


def join_schema_path(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key
