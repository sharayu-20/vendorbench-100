"""Shared normalization for the LLM track.

Every provider dump (Claude, Gemini, GPT, GLM, Qwen, Llama-4, Nemotron) was
produced with the *same* JSON contract:

    {"status", "ai_manipulated", "probability_real",
     "probability_ai_manipulated", "source_type", "short_summary"}

These helpers map that verdict payload onto the framework's canonical
(prediction, confidence) pair, where ``confidence`` is always P(fake) in [0, 1]
and ``prediction`` is "FAKE" / "REAL". They also derive a stable, collision-free
per-image key so the same picture is comparable across all providers.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_AI_TOKENS = ("ai", "manip", "fake", "synthetic", "generated")
_REAL_TOKENS = ("real", "authentic", "genuine")


def norm_gt(value: str) -> str:
    """Normalize a ground-truth label to 'FAKE' / 'REAL'."""
    v = (value or "").strip().lower()
    if v in ("fake", "ai", "ai_manipulated", "ai_generated"):
        return "FAKE"
    return "FAKE" if v.startswith("fake") else "REAL"


def p_fake(payload: dict[str, Any]) -> float:
    """Extract P(fake) from a verdict payload, clamped to [0, 1].

    Prefers ``probability_ai_manipulated``; falls back to ``1 - probability_real``.
    """
    val = payload.get("probability_ai_manipulated")
    if val is None:
        pr = payload.get("probability_real")
        val = (1.0 - float(pr)) if pr is not None else None
    if val is None:
        # last resort: infer from the boolean / status
        return 1.0 if _is_fake_verdict(payload) else 0.0
    try:
        return max(0.0, min(1.0, float(val)))
    except (TypeError, ValueError):
        return 1.0 if _is_fake_verdict(payload) else 0.0


def _is_fake_verdict(payload: dict[str, Any]) -> bool:
    flag = payload.get("ai_manipulated")
    if isinstance(flag, bool):
        return flag
    status = str(payload.get("status", "")).strip().lower()
    if any(t in status for t in _REAL_TOKENS) and not any(t in status for t in ("ai", "manip", "fake", "synthetic")):
        return False
    return any(t in status for t in _AI_TOKENS)


def prediction(payload: dict[str, Any]) -> str:
    """Derive 'FAKE' / 'REAL' from the verdict payload.

    Uses the explicit boolean/status first; if those are missing/ambiguous,
    falls back to the P(fake) >= 0.5 cut.
    """
    flag = payload.get("ai_manipulated")
    if isinstance(flag, bool):
        return "FAKE" if flag else "REAL"
    status = str(payload.get("status", "")).strip().lower()
    if status:
        if any(t in status for t in _REAL_TOKENS) and not any(t in status for t in ("ai", "manip", "fake", "synthetic")):
            return "REAL"
        if any(t in status for t in _AI_TOKENS):
            return "FAKE"
    return "FAKE" if p_fake(payload) >= 0.5 else "REAL"


def image_num(name: str) -> str:
    """First run of digits in a filename, zero-padded to 3 (e.g. '41' -> '041')."""
    m = re.search(r"(\d+)", Path(name).stem)
    return m.group(1).zfill(3) if m else Path(name).stem


def stem(gt: str, name: str) -> str:
    """Collision-free per-image key shared across providers, e.g. 'fake_041'."""
    return f"{norm_gt(gt).lower()}_{image_num(name)}"


def canonical_filename(gt: str, name: str) -> str:
    """Stable, extension-independent image_filename shared across providers.

    Providers stored the same numbered image under different extensions
    (fake_001.jpg vs .jpeg vs .png), so the key is kept extensionless
    (e.g. 'fake_041') to remain identical for the same picture everywhere.
    """
    return stem(gt, name)
