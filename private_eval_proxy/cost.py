"""Per-candidate cost accounting via the SDK's agent debug hook (no SDK changes).

The SDK's `OpenAIResponsesAgent` emits an `AgentDebugEvent` per phase; the
`response_received` event carries `response_payload` (with token `usage`) and
`latency_ms`. `CostAccountingSink` implements `AgentDebugSink` and accumulates
model-cost proxies for one candidate replay:

  - n_calls       : model round-trips (tool hops + final reply across messages)
  - input/output/total_tokens : from response usage (proxy for inference time)
  - latency_ms    : measured wall time (noisy; calibration only)

These are proxies for the judge's per-candidate replay time (the SDK only enforces
a fixed time budget, it does not report cost). Tokens are the primary, hardware-
portable proxy; n_calls captures per-round-trip overhead and the multi-hop penalty.
"""

from __future__ import annotations

from dataclasses import dataclass

from private_eval_proxy import _sdk as _bootstrap  # noqa: F401  (sys.path + env)

from aicomp_sdk.agents.debug import AgentDebugEvent


@dataclass
class CostTally:
    n_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0  # OpenRouter-reported billed cost (usage.cost)
    usage_seen: bool = False  # False => provider didn't return token usage
    cost_seen: bool = False  # False => provider didn't report usage.cost

    def as_dict(self) -> dict:
        return {
            "n_calls": self.n_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "latency_s": round(self.latency_ms / 1000.0, 3),
            "cost_usd": self.cost_usd,
            "usage_seen": self.usage_seen,
            "cost_seen": self.cost_seen,
        }


def _extract_usage(payload: object) -> dict | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else None


class CostAccountingSink:
    """AgentDebugSink that tallies model-cost proxies for a single candidate."""

    def __init__(self) -> None:
        self.tally = CostTally()

    def reset(self) -> None:
        self.tally = CostTally()

    def record(self, event: AgentDebugEvent) -> None:
        if event.phase != "response_received":
            return
        self.tally.n_calls += 1
        if event.latency_ms:
            self.tally.latency_ms += float(event.latency_ms)
        usage = _extract_usage(event.response_payload)
        if usage:
            self.tally.usage_seen = True
            # Responses API: input_tokens/output_tokens/total_tokens.
            # Chat Completions style: prompt_tokens/completion_tokens. Accept both.
            inp = int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
            out = int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
            tot = int(usage.get("total_tokens") or (inp + out))
            self.tally.input_tokens += inp
            self.tally.output_tokens += out
            self.tally.total_tokens += tot
            cost = usage.get("cost")
            if cost is not None:
                self.tally.cost_seen = True
                self.tally.cost_usd += float(cost)
