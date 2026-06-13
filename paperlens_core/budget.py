from __future__ import annotations

from dataclasses import dataclass
import copy
import threading
from typing import Any

from paperlens_core.config import BudgetConfig


@dataclass
class BudgetSnapshot:
    input_tokens: int = 0
    cached_input_tokens: int = 0
    output_tokens: int = 0
    estimated_usd: float = 0.0
    calls: int = 0


class BudgetManager:
    def __init__(self, config: BudgetConfig) -> None:
        self.config = config
        self.snapshot = BudgetSnapshot()
        self._lock = threading.Lock()

    def record_usage(self, usage: dict[str, Any]) -> BudgetSnapshot:
        if not usage:
            with self._lock:
                self.snapshot.calls += 1
                return copy.copy(self.snapshot)
        input_tokens = first_int(usage, "input_tokens", "prompt_tokens") or 0
        prompt_details = usage.get("prompt_tokens_details") or {}
        input_details = usage.get("input_tokens_details") or {}
        cached_input_tokens = first_int(usage, "cached_input_tokens", "cache_read_input_tokens")
        if cached_input_tokens is None:
            cached_input_tokens = first_int(prompt_details, "cached_tokens")
        if cached_input_tokens is None:
            cached_input_tokens = first_int(input_details, "cached_tokens")
        if cached_input_tokens is None:
            cached_input_tokens = 0
        output_tokens = first_int(usage, "output_tokens", "completion_tokens") or 0
        cached_input_tokens = min(cached_input_tokens, input_tokens)
        billable_uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
        with self._lock:
            self.snapshot.input_tokens += input_tokens
            self.snapshot.cached_input_tokens += cached_input_tokens
            self.snapshot.output_tokens += output_tokens
            self.snapshot.calls += 1
            self.snapshot.estimated_usd += (
                billable_uncached_input_tokens * self.config.input_token_usd_per_million
                + cached_input_tokens * self.config.cached_input_token_usd_per_million
                + output_tokens * self.config.output_token_usd_per_million
            ) / 1_000_000
            return copy.copy(self.snapshot)

    def public_dict(self) -> dict[str, Any]:
        with self._lock:
            snapshot = copy.copy(self.snapshot)
        return {
            "input_tokens": snapshot.input_tokens,
            "cached_input_tokens": snapshot.cached_input_tokens,
            "output_tokens": snapshot.output_tokens,
            "estimated_usd": round(snapshot.estimated_usd, 6),
            "calls": snapshot.calls,
        }


def first_int(payload: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return None
