"""Target-model `agent_factory` selection by backend flag.

    openrouter     — the SDK's OpenAIResponsesAgent pointed at OpenRouter (needs internet + an
                     OPENROUTER_API_KEY). Network-bound → safe to run many replays concurrently.
    kaggle_gguf    — the on-Kaggle GGUF/llama-cpp path used by the official local-validation
                     notebook (needs a GPU, `llama-cpp-python`, and the public model datasets;
                     works offline). Single model in GPU memory → run serially.
    deterministic  — the SDK's `VulnerableDeterministicAgent` (no weights, no network) — for
                     fast CI smoke tests of the guard + scorer.

The **judge** backend is an independent axis controlled by `PRIVATE_GUARD_JUDGE_BACKEND`
(openrouter | competition), handled inside `private_judge`. Pair `kaggle_gguf` target with a
`competition` judge for a fully-offline run.
"""

from __future__ import annotations

from functools import lru_cache

from private_eval_proxy import _sdk  # noqa: F401  (side effects: sys.path + env)

VALID_BACKENDS = ("openrouter", "kaggle_gguf", "deterministic")

# Backends that must run serially (a single model occupies the GPU); the runner clamps
# concurrency to 1 for these.
SERIAL_BACKENDS = ("kaggle_gguf", "deterministic")


def make_agent_factory(backend: str, target: str, debug_sink=None):
    """Return a zero-arg agent factory for `target` under `backend`."""
    backend = (backend or "openrouter").strip().lower()
    if backend == "openrouter":
        from private_eval_proxy.models import agent_factory

        return agent_factory(target, debug_sink=debug_sink)
    if backend == "deterministic":
        from aicomp_sdk.agents.deterministic_agent import VulnerableDeterministicAgent

        return lambda: VulnerableDeterministicAgent()
    if backend == "kaggle_gguf":
        server = _load_gguf_server(target)
        return lambda: server._load_agent()
    raise RuntimeError(f"Unknown backend {backend!r}; expected one of {VALID_BACKENDS}")


@lru_cache(maxsize=2)
def _load_gguf_server(target: str):
    """Load (once) a GGUF model server for `target` via the competition's kaggle_evaluation
    package — the same path the official validation notebook uses. Cached so the weights load
    once per target and are reused across replays.
    """
    from kaggle_evaluation.jed_attack_134815 import (  # type: ignore
        gemma_model_server,
        gpt_oss_model_server,
    )
    from kaggle_evaluation.jed_attack_134815.gguf_model_server import (  # type: ignore
        GgufModelServer,
    )

    specs = {"gpt_oss": gpt_oss_model_server.SPEC, "gemma": gemma_model_server.SPEC}
    spec = specs.get(target)
    if spec is None:
        raise RuntimeError(
            f"kaggle_gguf backend supports targets {list(specs)}, not {target!r}"
        )
    server = GgufModelServer(spec)
    server.load_model()
    return server


def default_concurrency(backend: str, requested: int) -> int:
    """Clamp concurrency to 1 for serial backends; otherwise honor the request."""
    return 1 if (backend or "").strip().lower() in SERIAL_BACKENDS else max(1, requested)
