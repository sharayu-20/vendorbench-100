"""Adapter: Meta Llama-4-Maverick-17B-128E-Instruct (open weights, via NVIDIA NIM).

Source dump: raw_llm_results_zip/4 opensource llms/results_llama4-maverick.json
(summary + results[].analysis). Collected through NVIDIA NIM
(https://integrate.api.nvidia.com/v1) — see the harness ``analyze_images.py``.
"""

from __future__ import annotations

from benchmark.models import APIResult
from benchmark.llms.base_llm import LLMAdapter


class Llama4Maverick(LLMAdapter):
    PROVIDER_ID = "llama4_maverick"
    DISPLAY_NAME = "Llama-4-Maverick-17B-128E"
    MODEL_ID = "meta/llama-4-maverick-17b-128e-instruct"
    ACCESS = "API"  # NVIDIA NIM
    RAW = "4 opensource llms/results_llama4-maverick.json"

    def load(self) -> list[APIResult]:
        return self._load_nim_dump()
