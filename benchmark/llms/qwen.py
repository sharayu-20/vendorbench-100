"""Adapter: Qwen (vision, chat.qwen.ai).

Source dump: raw_llm_results_zip/qwen/qwen_test_results.json — a single object with
``results[]``; each row has ``image_name``, ``category_folder`` (FAKE/REAL),
``success``, ``parsed_response{...}`` (or null), ``elapsed_sec`` and ``error``.
Collected via browser automation (see raw_llm_results_zip/qwen/run_qwen_image_test.py).
"""

from __future__ import annotations

import json

from benchmark.models import APIResult
from benchmark.llms.base_llm import LLMAdapter


class Qwen(LLMAdapter):
    PROVIDER_ID = "qwen"
    DISPLAY_NAME = "Qwen"
    MODEL_ID = "qwen (chat.qwen.ai)"
    ACCESS = "browser"
    RAW = "qwen/qwen_test_results.json"

    def load(self) -> list[APIResult]:
        data = json.loads(self.raw_path().read_text(encoding="utf-8"))
        results: list[APIResult] = []
        for rec in data.get("results", []):
            name = rec.get("image_name") or rec.get("relative_path", "")
            gt = rec.get("category_folder") or ""
            latency = float(rec.get("elapsed_sec") or 0.0) * 1000.0
            payload = rec.get("parsed_response") or {}
            if not rec.get("success", False) or not payload:
                results.append(self._fail(
                    gt=gt, name=name, error=rec.get("error") or "no parsed_response",
                    image_path=rec.get("image_path", ""), latency_ms=latency))
                continue
            results.append(self._ok(
                gt=gt, name=name, payload=payload,
                image_path=rec.get("image_path", ""), latency_ms=latency))
        return results
