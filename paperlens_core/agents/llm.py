from __future__ import annotations

import base64
import http.client
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
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


class JsonLlmClient:
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

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
            sdk_default = "1" if self.config.kind == "openai" else "0"
            if os.getenv("PAPERLENS_USE_AGENTS_SDK", sdk_default) != "0":
                try:
                    from paperlens_core.agents.sdk_runner import invoke_openai_agents_sdk_json

                    return invoke_openai_agents_sdk_json(
                        config=self.config,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt
                        + "\n\nJSON schema contract:\n"
                        + json.dumps(schema, ensure_ascii=False),
                        schema_name=schema_name,
                        max_tokens=max_tokens,
                    )
                except LlmError:
                    if os.getenv("PAPERLENS_REQUIRE_AGENTS_SDK", "0") == "1":
                        raise
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
        try:
            response, response_headers = self._post_json(endpoint, payload, headers)
        except LlmError as exc:
            if "400" not in str(exc):
                raise
            response, response_headers = self._post_json(endpoint, fallback_payload, headers)
        try:
            text, data = parse_chat_completion_json(response, schema_name, schema)
        except LlmError:
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
        response, response_headers = self._post_json(endpoint, payload, headers)
        try:
            text, data = parse_chat_completion_json(response, schema_name, schema)
        except LlmError:
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
        max_tokens = payload.pop("max_tokens", None)
        if max_tokens is not None:
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
        mode = os.getenv("PAPERLENS_MIMO_THINKING", "disabled").strip().lower()
        schema_name = compatible_payload_schema_name(payload)
        schema_filter = normalized_csv_env("PAPERLENS_MIMO_THINKING_SCHEMAS")
        if schema_filter and schema_name not in schema_filter:
            mode = "disabled"
        if mode in {"1", "true", "yes", "on", "enabled", "enable"}:
            return {"type": "enabled"}
        if mode in {"default", "provider", "omit"}:
            return None
        return {"type": "disabled"}

    def _post_json(
        self,
        endpoint: str,
        payload: dict[str, Any],
        headers: dict[str, str],
    ) -> tuple[dict[str, Any], dict[str, str]]:
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        retry_codes = {429, 500, 502, 503, 504}
        try:
            attempts = int(os.getenv("PAPERLENS_LLM_RETRIES", str(self.config.max_retries)))
        except ValueError:
            attempts = self.config.max_retries
        attempts = max(1, min(10, attempts))
        last_error: Exception | None = None
        try:
            timeout_seconds = int(os.getenv("PAPERLENS_LLM_TIMEOUT_SECONDS", str(self.config.timeout_seconds)))
        except ValueError:
            timeout_seconds = self.config.timeout_seconds
        timeout_seconds = max(10, min(timeout_seconds, 1800))
        for attempt in range(attempts):
            request = urllib.request.Request(
                endpoint,
                data=body_bytes,
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    body = response.read().decode("utf-8")
                    return json.loads(body), {
                        key.lower(): value for key, value in response.headers.items()
                    }
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                last_error = LlmError(f"HTTP {exc.code} from {endpoint}: {body[:1000]}")
                if exc.code not in retry_codes or attempt == attempts - 1:
                    raise last_error from exc
                retry_delay = self._retry_delay(attempt, exc.headers)
            except urllib.error.URLError as exc:
                last_error = LlmError(f"Network error calling {endpoint}: {exc}")
                if attempt == attempts - 1:
                    raise last_error from exc
                retry_delay = self._retry_delay(attempt)
            except (http.client.RemoteDisconnected, ConnectionError, OSError) as exc:
                last_error = LlmError(f"Network connection closed calling {endpoint}: {exc}")
                if attempt == attempts - 1:
                    raise last_error from exc
                retry_delay = self._retry_delay(attempt)
            except TimeoutError as exc:
                last_error = LlmError(f"Timeout calling {endpoint}")
                if attempt == attempts - 1:
                    raise last_error from exc
                retry_delay = self._retry_delay(attempt)
            except json.JSONDecodeError as exc:
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
