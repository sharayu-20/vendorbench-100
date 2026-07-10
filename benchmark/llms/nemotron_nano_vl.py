"""Adapter: NVIDIA Llama-3.1-Nemotron-Nano-VL-8B-v1 (open weights, via NVIDIA NIM).

Source dump: raw_llm_results_zip/4 opensource llms/results_nemotron-nano-vl.json
(summary + results[].analysis). 6 of 100 responses were unparseable and are
emitted as success=False (excluded from Acc/F1/AUC, counted under coverage) — see
the harness ``analyze_images.py``.
"""

from __future__ import annotations

from benchmark.models import APIResult
from benchmark.llms.base_llm import LLMAdapter


class NemotronNanoVL(LLMAdapter):
    PROVIDER_ID = "nemotron_nano_vl"
    DISPLAY_NAME = "Nemotron-Nano-VL-8B"
    MODEL_ID = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"
    ACCESS = "API"  # NVIDIA NIM
    RAW = "4 opensource llms/results_nemotron-nano-vl.json"

    def load(self) -> list[APIResult]:
        return self._load_nim_dump()
