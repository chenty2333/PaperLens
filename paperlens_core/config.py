from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


ProviderKind = Literal[
    "openai",
    "openai-compatible",
    "anthropic",
    "anthropic-compatible",
    "none",
]

ReadMode = Literal["standard"]


class ProviderConfig(BaseModel):
    kind: ProviderKind = "none"
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str = "PAPERLENS_API_KEY"
    model: str | None = None
    reasoning_model: str | None = None
    timeout_seconds: int = 120
    max_retries: int = 1

    @field_validator("api_key", "base_url", "model", "reasoning_model")
    @classmethod
    def blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @field_validator("model", "reasoning_model")
    @classmethod
    def normalize_model_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if value.lower().startswith("mimo-"):
            return value.lower()
        return value

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        if value := os.getenv(self.api_key_env):
            return value
        if self.kind in {"anthropic", "anthropic-compatible"}:
            return os.getenv("ANTHROPIC_API_KEY")
        return os.getenv("OPENAI_API_KEY")

    def request_model(self) -> str | None:
        return self.model

    def redacted(self) -> dict[str, Any]:
        data = self.model_dump()
        if data.get("api_key"):
            data["api_key"] = "***"
        data["has_api_key"] = bool(self.resolved_api_key())
        return data


class BudgetConfig(BaseModel):
    input_token_usd_per_million: float = 0.75
    cached_input_token_usd_per_million: float = 0.075
    output_token_usd_per_million: float = 4.5


class CoreConfig(BaseModel):
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    concurrency: int = 1
    render_zoom: float = 1.5
    visual_detail: str = "high"
    visual_verification_mode: Literal["parse_issues", "all_marked_pages", "off"] = "parse_issues"
    visual_verification_max_pages: int = 6
    visual_pages_per_call: int = 1
    stage_timeout_seconds: int = 900
    keyword_pool: list[str] = Field(default_factory=list)
    offline_debug: bool = False
    output_language: Literal["zh", "en"] = "zh"
    read_mode: ReadMode = "standard"
    topic: str | None = None
    idea: str | None = None

    @field_validator("concurrency")
    @classmethod
    def sane_concurrency(cls, value: int) -> int:
        return max(1, min(value, 16))

    @field_validator("visual_pages_per_call")
    @classmethod
    def sane_visual_pages_per_call(cls, value: int) -> int:
        return max(1, min(value, 6))

    @field_validator("visual_verification_max_pages")
    @classmethod
    def sane_visual_verification_max_pages(cls, value: int) -> int:
        return max(0, min(value, 50))

    @field_validator("output_language")
    @classmethod
    def sane_output_language(cls, value: str) -> str:
        language = value.strip().lower()
        return language if language in {"en", "zh"} else "zh"

    @field_validator("read_mode", mode="before")
    @classmethod
    def sane_read_mode(cls, value: str | None) -> str:
        mode = str(value or "").strip().lower()
        if mode in {"", "standard"}:
            return "standard"
        raise ValueError("PaperLens Core currently supports only read_mode='standard'")

    @field_validator("topic", "idea")
    @classmethod
    def blank_text_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None

    @property
    def topic_comparison_enabled(self) -> bool:
        return bool(self.topic and self.idea)

    def validate_agentic_run(self) -> None:
        if self.offline_debug:
            return
        if self.provider.kind == "none":
            raise ValueError("Formal PaperLens runs require a model provider")
        if not self.provider.model:
            raise ValueError("Formal PaperLens runs require a model name")
        if (
            self.provider.kind in {"openai-compatible", "anthropic-compatible"}
            and not self.provider.base_url
        ):
            raise ValueError(f"{self.provider.kind} runs require a base URL")
        if not self.provider.resolved_api_key():
            raise ValueError(f"Formal PaperLens runs require an API key for {self.provider.kind}")

    def public_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["provider"] = self.provider.redacted()
        return data


DEFAULT_KEYWORDS = [
    "novel",
    "state-of-the-art",
    "benchmark",
    "ablation",
    "dataset",
    "architecture",
    "algorithm",
    "evaluation",
    "limitation",
    "reproducible",
    "open source",
    "theory",
    "scaling",
    "efficiency",
    "robustness",
    "safety",
    "survey",
    "system",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        elif value is not None:
            result[key] = value
    return result


def drop_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: drop_none_values(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [drop_none_values(item) for item in value]
    return value


def load_config(config_path: Path | None, overrides: dict[str, Any]) -> CoreConfig:
    data: dict[str, Any] = {"keyword_pool": DEFAULT_KEYWORDS}
    if config_path:
        data = deep_merge(data, load_json(config_path))
    data = deep_merge(data, drop_none_values(overrides))
    if not data.get("keyword_pool"):
        data["keyword_pool"] = DEFAULT_KEYWORDS
    return CoreConfig.model_validate(data)
