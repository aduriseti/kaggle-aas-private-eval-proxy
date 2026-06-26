"""Make the competition SDK (`aicomp_sdk` + `kaggle_evaluation`) importable, and load
local `env.json` secrets — the package-local replacement for `harness/_bootstrap.py`.

`aicomp-sdk` is a **normal pip dependency** (it's on PyPI), so the usual path is simply *already
importable* — `pip install` puts it on `sys.path` and we leave it alone. This locator only adds
fallbacks for what pip can't cover — chiefly `kaggle_evaluation` (the gguf model server), which
ships **only** in the competition mount. It tries, in order:

  1. ``AICOMP_SDK_DIR`` env var (explicit override) — a dir containing ``aicomp_sdk/``.
  2. **Already importable** (the normal case — ``pip install``-ed) — leave ``sys.path`` alone.
  3. **Kaggle**: globbed from ``/kaggle/input`` (the competition mount provides ``kaggle_evaluation``
     beside ``aicomp_sdk`` for the offline/gguf path).
  4. **Local dev**: walk up from this file for ``competition_sdk/unpacked``.

Never vendored or patched — a pristine SDK from any source.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path


def _already_importable() -> bool:
    try:
        return importlib.util.find_spec("aicomp_sdk") is not None
    except Exception:
        return False


def _find_sdk_root() -> Path | None:
    """Directory to add to ``sys.path`` so ``import aicomp_sdk`` works (or None if already)."""
    override = os.environ.get("AICOMP_SDK_DIR", "").strip()
    if override:
        # A set-but-invalid override is a LOUD error, never a silent fall-through to a different
        # (ambient pip / Kaggle) SDK — mirrors harness/_bootstrap._resolve_sdk_root.
        if not (Path(override) / "aicomp_sdk").is_dir():
            raise RuntimeError(
                f"AICOMP_SDK_DIR={override!r} does not contain an `aicomp_sdk/` package. "
                "Point it at a directory holding aicomp_sdk/, or unset it."
            )
        return Path(override)

    if _already_importable():
        return None

    # Kaggle: the competition dataset root contains aicomp_sdk/ + kaggle_evaluation/.
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for match in sorted(kaggle_input.glob("**/aicomp_sdk")):
            if match.is_dir():
                return match.parent

    # Local dev: competition_sdk/unpacked somewhere above this file.
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "competition_sdk" / "unpacked"
        if (cand / "aicomp_sdk").is_dir():
            return cand
    return None


def _ensure_sdk_on_path() -> None:
    root = _find_sdk_root()
    if root is None:
        if _already_importable():
            return
        raise RuntimeError(
            "Could not locate the competition SDK (`aicomp_sdk`). Set AICOMP_SDK_DIR to a "
            "directory containing `aicomp_sdk/`, run on Kaggle with the competition attached, "
            "or `pip install` the SDK. The package never vendors it."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_env_json() -> None:
    """Load a local ``env.json`` (dev convenience) without clobbering preset env vars.

    Searched upward from CWD then this file. On Kaggle there is no env.json — the notebook
    injects secrets (e.g. via ``UserSecretsClient``) into ``os.environ`` before import.
    """
    seen: set[Path] = set()
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for base in [start, *start.parents]:
            cand = base / "env.json"
            if cand in seen:
                continue
            seen.add(cand)
            if cand.is_file():
                try:
                    raw = cand.read_text()
                except (OSError, UnicodeDecodeError):
                    # Unreadable (permissions), a directory named env.json, or non-UTF-8: skip and
                    # try the next candidate up the tree, as before. Only genuinely malformed JSON
                    # (below) is a user error worth surfacing.
                    continue
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError as exc:
                    # A malformed env.json is a user error to surface, not to swallow and silently
                    # run without secrets (which later fails opaquely on a missing API key).
                    raise RuntimeError(f"env.json at {cand} is not valid JSON: {exc}") from exc
                for k, v in data.items():
                    if v and not os.environ.get(k):
                        os.environ[k] = str(v)
                return


_ensure_sdk_on_path()
_load_env_json()


def openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY missing — set it in env.json (local) or as a Kaggle Secret "
            "named OPENROUTER_API_KEY (notebook reads it via UserSecretsClient)."
        )
    return key
