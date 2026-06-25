"""Shared per-rep metric derivation from the replay tuple.

`replay_one` (the contract) returns `(trace, predicates, finding, cost)`. Two consumers reduce that
tuple per rep: the standalone CLI (`runner.evaluate`) and the repo orchestrator
(`harness/evaluate.py:evaluate_target`). They previously each called `scoring_v2.cell_and_severity`
and pulled the SDK raw out of `cost` by hand, so the firing/severity convention could drift. This
module is the single derivation both import.

`rep_metrics` exposes BOTH scorings so each consumer picks the right one:

  - `sev_v2`/`cell_v2` — our private `scoring_v2` (coarse cell + log decay);
  - `sdk_raw`/`sdk_cell` — the SDK's own `score_attack_raw` baseline + SDK cell signature.

The env-appropriate `severity`/`cell`/`fired` are selected by `guard_kind`: a **public** run is the
SDK baseline (so it's apples-to-apples with the SDK), a **private** run is `scoring_v2`.
"""

from __future__ import annotations

from dataclasses import dataclass

from private_eval_proxy import scoring_v2


@dataclass
class RepMetrics:
    """One rep's facts, derived once from the replay tuple."""
    # env-appropriate view (public → SDK baseline, private → scoring_v2)
    cell: str | None
    severity: float
    fired: bool
    # raw components, both always available regardless of env
    sdk_raw: float             # SDK score_attack_raw([finding])
    sdk_cell: str | None       # SDK score cell-signature hash
    sev_v2: float              # scoring_v2 severity (Σ SEVERITY_W over distinct cells)
    cell_v2: str | None        # scoring_v2 coarse (source,sink,predicate) cell hash
    predicates: list[str]
    output_tokens: int
    wall_s: float
    cost_usd: float
    cost_seen: bool            # False ⇒ backend didn't report usage.cost (cost is unknown, not $0)


def rep_metrics(tup, *, guard_kind: str) -> RepMetrics:
    """Derive one rep's metrics from a `(trace, predicates, finding, cost)` replay tuple."""
    trace, predicates, finding, cost = tup
    tool_events = trace.get("tool_events", [])
    cell_v2, sev_v2 = scoring_v2.cell_and_severity(
        tool_events, predicates, trace.get("user_messages", []))
    sdk_raw = cost["candidate_raw"]
    sdk_cell = (finding.get("score_cell_signature") or {}).get("hash") if finding else None

    if guard_kind == "public":
        cell, severity, fired = sdk_cell, sdk_raw, sdk_raw > 0
    else:
        cell, severity, fired = cell_v2, sev_v2, bool(predicates)

    return RepMetrics(
        cell=cell,
        severity=severity,
        fired=fired,
        sdk_raw=sdk_raw,
        sdk_cell=sdk_cell,
        sev_v2=sev_v2,
        cell_v2=cell_v2,
        predicates=[str(p.get("predicate")) for p in predicates],
        output_tokens=cost["output_tokens"],
        wall_s=cost["wall_latency_s"],
        cost_usd=cost["cost_usd"],
        cost_seen=cost["cost_seen"],
    )
