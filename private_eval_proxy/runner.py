"""Pure replay evaluator: run a list of attacks against one env (guard regime), per attack.

This is the package's **replay engine and nothing more**. For each candidate it drives the SDK's
real `_replay_candidate_trace` (the exact judge replay path) against one env:

  - `private` → our stand-in `PrivateGuardrail` (+ POST_TOOL_CALL taint hook) scored with
    `eval_predicates_realistic`;
  - `public`  → the SDK's own `OptimalGuardrail` scored with the vendored `eval_predicates`.

`replay_one` returns the raw `(trace, predicates, finding, cost)` tuple — the same shape the repo's
orchestrator (`harness/evaluate.py`) consumes — so the two share one replay contract. Allocation,
bootstrap CIs, and reporting live in the **repo-level** orchestrator, not here. Replays run
concurrently (OpenRouter is network-bound); serial backends clamp to 1.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from private_eval_proxy import _sdk  # noqa: F401  (side effects: sys.path + env)
from private_eval_proxy import aggregate
from private_eval_proxy.backends import default_concurrency, make_agent_factory
from private_eval_proxy.cost import CostAccountingSink
from private_eval_proxy.private_guard import (
    Guardrail as PrivateGuardrail,
    eval_predicates_realistic,
    install_guard,
)

from aicomp_sdk.attacks import AttackCandidate
from aicomp_sdk.core.cells import cell_signature
from aicomp_sdk.core.env.api import EnvSelection
from aicomp_sdk.core.predicates import eval_predicates
from aicomp_sdk.evaluation.ops import (
    EVALUATION_DEFAULT_MAX_TOOL_HOPS,
    _replay_candidate_trace,
    resolve_fixtures_dir,
)
from aicomp_sdk.guardrails.optimal import Guardrail as OptimalGuardrail
from aicomp_sdk.hooks.registry import HookRegistry
from aicomp_sdk.scoring import score_attack_raw

ENV_SEED = 123  # matches the judge default (ops.py attack_env_seed=123)


# --------------------------------------------------------------------------- replay (the contract)

def replay_one(rec: dict, target: str, *, guard_kind: str, max_hops: int,
               backend: str = "openrouter"):
    """One replay of one candidate against one env. Returns `(trace, predicates, finding, cost)` —
    the shared replay contract the repo orchestrator consumes.

    `guard_kind` selects the env: `private` (our stand-in guard + taint hook + realistic
    predicates) or `public` (SDK `OptimalGuardrail` + vendored predicates). Each call builds its
    own guard + cost sink, so it is independent and thread-safe.
    """
    candidate = AttackCandidate(user_messages=tuple(rec["user_messages"]))
    fixtures = resolve_fixtures_dir()
    sink = CostAccountingSink()
    started = time.perf_counter()

    if guard_kind == "private":
        guard = PrivateGuardrail(debug_sink=sink)

        def hook_factory(_state=guard.state):
            reg = HookRegistry()
            install_guard(reg, _state)
            return reg
    elif guard_kind == "public":
        guard = OptimalGuardrail()
        hook_factory = HookRegistry
    else:
        raise ValueError(f"guard_kind must be 'private' or 'public', got {guard_kind!r}")

    trace = _replay_candidate_trace(
        candidate=candidate,
        index=0,
        seed=ENV_SEED,
        fixtures_dir=fixtures,
        agent_factory=make_agent_factory(backend, target, debug_sink=sink),
        guardrail=guard,
        env_selection=EnvSelection.SANDBOX,
        max_tool_hops=max_hops,
        hook_registry_factory=hook_factory,
    )
    wall_latency = time.perf_counter() - started

    predicates = (
        eval_predicates_realistic(trace) if guard_kind == "private" else eval_predicates(trace)
    )
    finding = None
    if predicates:
        finding = {
            "seed": trace.get("seed"),
            "user_messages": list(trace.get("user_messages", [])),
            "predicates": predicates,
            "cell_signature": cell_signature(
                trace.get("tool_events", []),
                egress_data=trace.get("egress", []),
                user_messages=trace.get("user_messages", []),
            ),
            "score_cell_signature": cell_signature(trace.get("tool_events", [])),
        }

    cost = sink.tally.as_dict()
    cost["wall_latency_s"] = round(wall_latency, 3)
    cost["n_tool_events"] = len(trace.get("tool_events", []))
    cost["n_user_messages"] = len(rec["user_messages"])
    cost["prompt_chars"] = sum(len(m) for m in rec["user_messages"])
    cost["candidate_raw"] = score_attack_raw([finding]) if finding else 0.0
    return trace, predicates, finding, cost


# --------------------------------------------------------------------------- CLI convenience types

@dataclass
class RepResult:
    """Lean per-rep summary derived from the replay tuple — for the standalone CLI only.

    `severity`/`cell` are env-appropriate: a `private` rep is scored with our `scoring_v2`, a
    `public` rep with the SDK's own `score_attack_raw` baseline (so the public numbers stay
    apples-to-apples with the SDK, not our private scorer)."""
    id: str
    fired: bool
    severity: float            # env-appropriate severity (private: scoring_v2; public: SDK raw)
    cell: str | None           # env-appropriate cell hash
    sdk_raw: float             # SDK score_attack_raw([finding]) — always the SDK baseline
    predicates: list[str]
    output_tokens: int
    wall_s: float
    cost_usd: float
    cost_seen: bool            # False ⇒ backend didn't report usage.cost (cost unknown, not $0)


@dataclass
class CandidateStats:
    id: str
    target: str
    guard: str                 # "private" | "public"
    user_messages: list[str]
    reps: int
    fire_rate: float
    mean_sev: float
    cell: str | None
    predicates: list[str]
    mean_output_tokens: float
    mean_wall_s: float
    mean_cost_usd: float
    mean_sdk_raw: float
    cost_reported: bool        # True ⇒ every rep reported usage.cost (mean_cost_usd is real)
    trials_sev: list[float] = field(default_factory=list)
    trials_tokens: list[float] = field(default_factory=list)


def _rep_result(rec: dict, tup, guard_kind: str) -> RepResult:
    m = aggregate.rep_metrics(tup, guard_kind=guard_kind)
    return RepResult(
        id=rec["id"],
        fired=m.fired,
        severity=m.severity,
        cell=m.cell,
        sdk_raw=m.sdk_raw,
        predicates=m.predicates,
        output_tokens=m.output_tokens,
        wall_s=m.wall_s,
        cost_usd=m.cost_usd,
        cost_seen=m.cost_seen,
    )


def _aggregate(rec: dict, target: str, guard_kind: str, reps_out: list[RepResult]) -> CandidateStats:
    if not reps_out:
        # With duplicate ids rejected up-front and reps>=1, this is unreachable; guard loudly so a
        # future caller can't silently produce a fabricated reps=0/fire=0% candidate.
        raise ValueError(f"no replay results for candidate {rec['id']!r} — cannot aggregate")
    n = len(reps_out)
    fired = [r for r in reps_out if r.fired]
    cell = Counter(r.cell for r in fired if r.cell).most_common(1)
    preds = sorted({p for r in fired for p in r.predicates})
    return CandidateStats(
        id=rec["id"],
        target=target,
        guard=guard_kind,
        user_messages=list(rec["user_messages"]),
        reps=n,
        fire_rate=len(fired) / n,
        mean_sev=sum(r.severity for r in reps_out) / n,
        cell=cell[0][0] if cell else None,
        predicates=preds,
        mean_output_tokens=sum(r.output_tokens for r in reps_out) / n,
        mean_wall_s=sum(r.wall_s for r in reps_out) / n,
        mean_cost_usd=sum(r.cost_usd for r in reps_out) / n,
        mean_sdk_raw=sum(r.sdk_raw for r in reps_out) / n,
        cost_reported=all(r.cost_seen for r in reps_out),
        trials_sev=[r.severity for r in reps_out],
        trials_tokens=[float(r.output_tokens) for r in reps_out],
    )


def evaluate(candidates: list[dict], target: str, *, guard_kind: str,
             backend: str = "openrouter", reps: int = 3,
             max_hops: int = EVALUATION_DEFAULT_MAX_TOOL_HOPS,
             concurrency: int = 8) -> list[CandidateStats]:
    """Replay every (candidate × rep) against `target` in env `guard_kind`, concurrently when the
    backend allows it. Returns per-candidate aggregated stats (order preserved). CLI convenience —
    the repo orchestrator calls `replay_one` directly for the raw tuple."""
    if reps < 1:
        raise ValueError(f"reps must be >= 1, got {reps}")
    dups = [cid for cid, c in Counter(rec["id"] for rec in candidates).items() if c > 1]
    if dups:
        raise ValueError(f"duplicate candidate id(s) {dups} — ids must be unique (reps merge by id)")

    conc = default_concurrency(backend, concurrency)
    tasks = [rec for rec in candidates for _ in range(reps)]
    by_id: dict[str, list[RepResult]] = {rec["id"]: [] for rec in candidates}

    def _run(rec):
        tup = replay_one(rec, target, guard_kind=guard_kind, max_hops=max_hops, backend=backend)
        return _rep_result(rec, tup, guard_kind)

    if conc <= 1:
        for rec in tasks:
            rr = _run(rec)
            by_id[rr.id].append(rr)
    else:
        with ThreadPoolExecutor(max_workers=conc) as ex:
            futs = [ex.submit(_run, rec) for rec in tasks]
            for fut in as_completed(futs):
                rr = fut.result()
                by_id[rr.id].append(rr)

    return [_aggregate(rec, target, guard_kind, by_id[rec["id"]]) for rec in candidates]


# --------------------------------------------------------------------------- helpers + CLI

def attack_class_from_candidates(candidates: list[dict]):
    """Wrap a static candidate list as an `AttackAlgorithm` (the submission interface), so the same
    set can drive the SDK's `evaluate_redteam` public baseline AND our private-proxy runner."""
    from aicomp_sdk.attacks import AttackAlgorithmBase

    cands = [AttackCandidate(user_messages=tuple(c["user_messages"])) for c in candidates]

    class AttackAlgorithm(AttackAlgorithmBase):
        def run(self, env, config):  # noqa: ARG002 (static list ignores env/config)
            return list(cands)

    return AttackAlgorithm


def load_candidates(path: Path) -> list[dict]:
    candidates: list[dict] = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        candidates.append(json.loads(line))
    return candidates


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Replay attack candidates against one env (public|private). Pure replay — "
                    "allocation/CI/reporting live in the repo-level orchestrator."
    )
    parser.add_argument("--candidates", default=str(here / "candidates.sample.jsonl"))
    parser.add_argument("--targets", default="gpt_oss,gemma", help="comma-separated short names")
    parser.add_argument("--backend", default="openrouter",
                        choices=["openrouter", "kaggle_gguf", "deterministic"])
    parser.add_argument("--env", required=True, choices=["public", "private"],
                        help="which guard regime to replay against (no default — pick one)")
    parser.add_argument("--reps", type=int, default=3, help="replays per candidate")
    parser.add_argument("--concurrency", type=int, default=8, help="parallel replays (network backends)")
    parser.add_argument("--max-hops", type=int, default=EVALUATION_DEFAULT_MAX_TOOL_HOPS)
    parser.add_argument("--out", default=None, help="optional path to write raw per-attack JSON")
    args = parser.parse_args(argv)
    if args.reps < 1:
        parser.error("--reps must be >= 1")

    candidates = load_candidates(Path(args.candidates))
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]

    # sev̄ is env-appropriate: SDK score_attack_raw baseline for public, scoring_v2 for private.
    metric = "SDK-raw baseline" if args.env == "public" else "scoring_v2 severity"
    results: dict = {"backend": args.backend, "env": args.env, "targets": {}}
    for target in targets:
        print(f"[{target}/{args.env}] replaying {len(candidates)} candidates "
              f"(reps={args.reps}, backend={args.backend}, conc={args.concurrency}); "
              f"sev̄ = {metric} ...")
        stats = evaluate(candidates, target, guard_kind=args.env, backend=args.backend,
                         reps=args.reps, max_hops=args.max_hops, concurrency=args.concurrency)
        for s in stats:
            cost_str = f"${s.mean_cost_usd * s.reps:.5f}" if s.cost_reported else "cost=n/a"
            print(f"  {s.id:26s} fire={s.fire_rate:.0%}  sev̄={s.mean_sev:.1f}  "
                  f"{cost_str}  preds={','.join(s.predicates) or '-'}")
        results["targets"][target] = [dataclasses.asdict(s) for s in stats]

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
