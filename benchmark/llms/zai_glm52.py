"""Adapter: Z.ai GLM-5.2 (vision).

Source dump: raw_llm_results_zip/z_ai/z_ai_100_images_report.json — a single object
keyed by absolute image path; each value has ``analysis_payload{...}`` and
``folder_source`` (FAKE/REAL). Collected via browser automation
(see raw_llm_results_zip/z_ai/z_ai_test.py).
"""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.models import APIResult
from benchmark.llms.base_llm import LLMAdapter


class ZaiGLM52(LLMAdapter):
    PROVIDER_ID = "zai_glm52"
    DISPLAY_NAME = "Z.ai GLM-5.2"
    MODEL_ID = "glm-5.2"
    ACCESS = "browser"
    RAW = "z_ai/z_ai_100_images_report.json"

    def load(self) -> list[APIResult]:
        data = json.loads(self.raw_path().read_text(encoding="utf-8"))
        results: list[APIResult] = []
        for path, rec in data.items():
            img_path = rec.get("origin_image_path") or path
            name = Path(img_path).name
            gt = rec.get("folder_source") or ""
            payload = rec.get("analysis_payload") or {}
            if not payload:
                results.append(self._fail(gt=gt, name=name, error="no analysis_payload",
                                          image_path=img_path))
                continue
            results.append(self._ok(gt=gt, name=name, payload=payload, image_path=img_path))
        return results
