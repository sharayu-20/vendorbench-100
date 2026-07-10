"""Adapter: Anthropic Claude Opus 4.8 (vision).

Source dump: raw_llm_results_zip/Opus4.8/*.json — one file per image, already in a
near-canonical shape (top-level prediction/confidence + latency_ms + sha256, with
the verdict payload under ``raw_response``). Collected via the Anthropic API
(see raw_llm_results_zip/Opus4.8_scripts/opus.py).
"""

from __future__ import annotations

import json

from benchmark.models import APIResult
from benchmark.llms.base_llm import LLMAdapter


class ClaudeOpus48(LLMAdapter):
    PROVIDER_ID = "claude_opus48"
    DISPLAY_NAME = "Claude Opus 4.8"
    MODEL_ID = "claude-opus-4-8"
    ACCESS = "API"
    RAW = "Opus4.8"

    def load(self) -> list[APIResult]:
        results: list[APIResult] = []
        for jf in sorted(self.raw_path().glob("*.json")):
            d = json.loads(jf.read_text(encoding="utf-8"))
            gt = d.get("ground_truth") or ""
            name = d.get("image_filename") or jf.name
            payload = d.get("raw_response") or {}
            latency = d.get("latency_ms", 0.0)
            sha = d.get("image_sha256", "")
            if not d.get("success", True) or not payload:
                results.append(self._fail(
                    gt=gt, name=name, error=d.get("error_message") or "no verdict",
                    image_path=d.get("image_path", ""), latency_ms=latency))
                continue
            results.append(self._ok(
                gt=gt, name=name, payload=payload,
                image_path=d.get("image_path", ""), latency_ms=latency, sha256=sha))
        return results
