from __future__ import annotations

from dataclasses import dataclass

from paperlens_core.config import ProviderConfig


@dataclass(frozen=True)
class ProviderCapability:
    kind: str
    responses_api: bool
    messages_api: bool
    notes: str


def describe_provider(config: ProviderConfig) -> ProviderCapability:
    if config.kind in {"openai", "openai-compatible"}:
        notes = (
            "Uses OpenAI Responses API for model reading and audit."
            if config.kind == "openai"
            else "Uses OpenAI-compatible Chat Completions API for model reading and audit."
        )
        return ProviderCapability(
            kind=config.kind,
            responses_api=config.kind == "openai",
            messages_api=config.kind == "openai-compatible",
            notes=notes,
        )
    if config.kind in {"anthropic", "anthropic-compatible"}:
        return ProviderCapability(
            kind=config.kind,
            responses_api=False,
            messages_api=True,
            notes="Uses Anthropic-compatible Messages API for model reading and audit.",
        )
    return ProviderCapability(
        kind="none",
        responses_api=False,
        messages_api=False,
        notes="Offline debug mode; deterministic smoke-test classifiers are used.",
    )
