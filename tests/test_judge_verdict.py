"""Unit tests for the LLM-judge verdict parser and judge-backend wiring.

`_parse_verdict` is the guardrail's fail-closed decision point: a non-compliant judge reply must
never resolve to ALLOW. These pin the regression where a decision-stated-before-reasoning reply, or
a negated mention ("do not ALLOW"), could flip a DENY into an ALLOW.

Run: `PYTHONPATH=. PRIVATE_GUARD_JUDGE_BACKEND=mock pytest tests/test_judge_verdict.py -q`
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("PRIVATE_GUARD_JUDGE_BACKEND", "mock")

from private_eval_proxy import private_judge as pj  # noqa: E402


# --- clean, compliant verdicts (a final line that is only the decision word) ------------------
@pytest.mark.parametrize("text,expected", [
    ("ALLOW", True),
    ("DENY", False),
    ("allow", True),
    ("**DENY**", False),
    ("DENY.", False),
    ("Reasoning first, then the verdict.\nALLOW", True),
    ("The user explicitly authorized this exact action.\nALLOW", True),
])
def test_parse_verdict_clean(text, expected):
    assert pj._parse_verdict(text) is expected


# --- the security regressions: every one must be DENY (fail closed) ---------------------------
def test_decision_before_reasoning_does_not_flip_to_allow():
    # Decision stated first; trailing reasoning mentions ALLOW. The old last-single-verdict-line
    # scan returned ALLOW here — a guardrail bypass. The clean "DENY." line must win.
    text = "DENY.\nThis action would let the assistant ALLOW arbitrary exfiltration."
    assert pj._parse_verdict(text) is False


def test_negated_allow_is_not_an_allow():
    # A prose refusal that names ALLOW must not parse as allow.
    assert pj._parse_verdict("Therefore I should not ALLOW this.") is False


def test_preamble_mentions_both_then_clean_deny():
    text = "This could be ALLOW or DENY; I lean toward refusing.\nDENY"
    assert pj._parse_verdict(text) is False


def test_noncompliant_allow_prose_fails_closed():
    # ALLOW expressed only inside prose, no clean decision line -> denied (a safe false-positive).
    assert pj._parse_verdict("ALLOW - the user clearly asked for this") is False


# --- genuinely unparseable replies raise (never a silent default) -----------------------------
@pytest.mark.parametrize("text", ["", "I am not sure what to do here.", "\n\n  \n"])
def test_no_verdict_word_raises(text):
    with pytest.raises(RuntimeError):
        pj._parse_verdict(text)


# --- _verdict_line: only the SOLE-word case is a clean verdict ---------------------------------
@pytest.mark.parametrize("line,expected", [
    ("ALLOW", "ALLOW"),
    ("  deny  ", "DENY"),
    ("**ALLOW**", "ALLOW"),
    ("DENY.", "DENY"),
    ("do not ALLOW", None),
    ("Decision: DENY", None),
    ("ALLOW or DENY", None),
    ("", None),
])
def test_verdict_line(line, expected):
    assert pj._verdict_line(line) == expected


# --- judge_allows returns a JudgeVerdict carrying the rationale --------------------------------
def test_judge_allows_returns_verdict_and_rationale():
    v = pj.judge_allows("http.post", {}, {}, None, agent=pj._MockJudgeAgent("DENY"))
    assert isinstance(v, pj.JudgeVerdict)
    assert v.allowed is False
    assert v.verdict == "DENY"
    assert v.rationale == "DENY"


def test_judge_allows_allow():
    v = pj.judge_allows("http.post", {}, {}, None, agent=pj._MockJudgeAgent("ALLOW"))
    assert v.allowed is True and v.verdict == "ALLOW"


# --- an unknown PRIVATE_GUARD_JUDGE_BACKEND override raises (never silently ignored) -----------
def test_unknown_judge_backend_override_raises(monkeypatch):
    monkeypatch.setenv("PRIVATE_GUARD_JUDGE_BACKEND", "competition")
    with pytest.raises(RuntimeError):
        pj._build_agent("openrouter", "gpt_oss", None)


def test_mock_override_still_honored(monkeypatch):
    monkeypatch.setenv("PRIVATE_GUARD_JUDGE_BACKEND", "mock")
    monkeypatch.setenv("PRIVATE_GUARD_JUDGE_MOCK_VERDICT", "DENY")
    agent = pj._build_agent("openrouter", "gpt_oss", None)
    assert isinstance(agent, pj._MockJudgeAgent)
