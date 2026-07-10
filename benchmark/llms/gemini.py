"""Adapter: Google Gemini (vision).

Source dump: raw_llm_results_zip/gemini/*.json — one file per image, shape
``{file_path, file_name, tested_at, result{...}}``. Ground truth is read from the
``file_path`` (…/fake/… or …/real/…), falling back to the dump filename prefix.
"""

from __future__ import annotations

import json

from benchmark.models import APIResult
from benchmark.llms.base_llm import LLMAdapter


def _gt_from(file_path: str, dump_name: str) -> str:
    p = (file_path or "").lower()
    if "/fake/" in p or "\\fake\\" in p:
        return "FAKE"
    if "/real/" in p or "\\real\\" in p:
        return "REAL"
    return "FAKE" if dump_name.lower().startswith("fake") else "REAL"


class Gemini(LLMAdapter):
    PROVIDER_ID = "gemini"
    DISPLAY_NAME = "Gemini"
    MODEL_ID = "gemini (vision)"
    ACCESS = "API"
    RAW = "gemini"

    def load(self) -> list[APIResult]:
        results: list[APIResult] = []
        for jf in sorted(self.raw_path().glob("*.json")):
            d = json.loads(jf.read_text(encoding="utf-8"))
            name = d.get("file_name") or jf.name
            gt = _gt_from(d.get("file_path", ""), jf.stem)
            payload = d.get("result") or {}
            if not payload:
                results.append(self._fail(gt=gt, name=name, error="no result payload",
                                          image_path=d.get("file_path", "")))
                continue
            results.append(self._ok(gt=gt, name=name, payload=payload,
                                    image_path=d.get("file_path", "")))
        return results
