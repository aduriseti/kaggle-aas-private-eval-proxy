"""OpenRouter-backed agent factory.

Reuses the SDK's own `OpenAIResponsesAgent` (it speaks the OpenAI Responses API,
which OpenRouter supports) pointed at OpenRouter — so attacks are evaluated against
the real target models with the real env/tools/scoring, no SDK changes.
"""

from __future__ import annotations

from functools import lru_cache

from private_eval_proxy import _sdk as _bootstrap  # noqa: F401  (side effects: sys.path + env)
from private_eval_proxy._sdk import openrouter_key

from openai import OpenAI

from aicomp_sdk.agents.openai_agent import OpenAIResponsesAgent

OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# OpenRouter returns the exact billed cost in `usage.cost` when usage accounting is
# requested. The SDK's OpenAIResponsesAgent builds the request itself, so we inject the
# flag by wrapping the client's `responses.create` (no SDK edit).
_USAGE_ACCOUNTING = {"usage": {"include": True}}


class _UsageResponses:
    def __init__(self, inner):
        self._inner = inner

    def create(self, **kwargs):
        extra = dict(kwargs.get("extra_body") or {})
        extra.setdefault("usage", _USAGE_ACCOUNTING["usage"])
        kwargs["extra_body"] = extra
        return self._inner.create(**kwargs)


class _UsageAccountingClient:
    """Thin proxy over an OpenAI client that enables OpenRouter usage accounting."""

    def __init__(self, inner):
        self._inner = inner

    @property
    def responses(self):
        return _UsageResponses(self._inner.responses)

# Map short target names -> OpenRouter model ids.
# gpt_oss / gemma are the SDK's stated target variants (gpt_oss_agent.py / gemma4_agent.py).
TARGETS: dict[str, str] = {
    "gpt_oss": "openai/gpt-oss-20b",
    "gpt_oss_120b": "openai/gpt-oss-120b",
    "gemma": "google/gemma-4-26b-a4b-it",
    "gemma_31b": "google/gemma-4-31b-it",
}


@lru_cache(maxsize=1)
def _client():
    # timeout=30s per HTTP request (default is 600s) so a stalled OpenRouter upstream fails fast
    # instead of freezing a whole eval; max_retries=2 (the OpenAI SDK default, made explicit) →
    # worst case ~3 attempts ≈ 90s + backoff before a candidate's rep is abandoned.
    return _UsageAccountingClient(
        OpenAI(base_url=OPENROUTER_BASE, api_key=openrouter_key(), timeout=30.0, max_retries=2)
    )


def resolve_model(target: str) -> str:
    return TARGETS.get(target, target)


def agent_factory(target: str, debug_sink=None):
    """Return a zero-arg factory producing a fresh agent bound to `target`.

    `debug_sink` (optional `AgentDebugSink`) is attached so callers can capture
    per-call cost (tokens / latency). The SDK calls the factory once per replay.
    """
    model = resolve_model(target)
    client = _client()

    def make() -> OpenAIResponsesAgent:
        return OpenAIResponsesAgent(client=client, model=model, debug_sink=debug_sink)

    return make
