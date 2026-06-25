"""kaggle-aas-private-eval-proxy — a proxy for the hidden *private* eval of the Kaggle competition
"AI Agent Security: Multi-Step Tool Attacks".

Three pieces that together estimate how an attack portfolio would fare against the held-out
private guardrail far better than the gameable public ``OptimalGuardrail`` does:

* ``private_guard`` — a realistic, layered, judge-centric private-guardrail stand-in
  (content + authorization, not name/keyword matching) + ``eval_predicates_realistic``.
* ``private_judge`` — the universal 2nd-line LLM-as-judge (SDK agent path, swappable backend).
* ``scoring_v2`` — the anti-gaming scorer: coarse ``(source, sink, predicate)`` cells + per-cell
  logarithmic decay, with a greedy (optimal) and an as-submitted budget allocator.

The competition SDK (``aicomp_sdk``) is a **declared dependency** located at import time by
``_sdk`` — never vendored or edited. See README.md for the full design rationale.
"""

from __future__ import annotations

__version__ = "0.1.0"
