"""Ordered registry of vision-LLM adapters for the LLM track.

Mirrors ``opensource_models/registry.py``: an ordered id list plus lazy
``adapter(id)`` / ``enabled_ids()`` / ``info(id)`` factories. Each adapter reads a
pre-collected result dump and re-emits canonical ``APIResult`` objects.

Roster (5 hosted + 2 open-weight):
  gemini · claude_opus48 · gpt_openai · zai_glm52 · qwen | llama4_maverick · nemotron_nano_vl
"""

from __future__ import annotations

from benchmark.llms.base_llm import LLMAdapter
from benchmark.llms.claude_opus48 import ClaudeOpus48
from benchmark.llms.gemini import Gemini
from benchmark.llms.gpt_openai import GPTOpenAI
from benchmark.llms.zai_glm52 import ZaiGLM52
from benchmark.llms.qwen import Qwen
from benchmark.llms.llama4_maverick import Llama4Maverick
from benchmark.llms.nemotron_nano_vl import NemotronNanoVL

_ADAPTERS: dict[str, type[LLMAdapter]] = {
    a.PROVIDER_ID: a
    for a in (
        Gemini,
        ClaudeOpus48,
        GPTOpenAI,
        ZaiGLM52,
        Qwen,
        Llama4Maverick,
        NemotronNanoVL,
    )
}

LLM_IDS: list[str] = list(_ADAPTERS.keys())


def enabled_ids() -> list[str]:
    return list(LLM_IDS)


def adapter(provider_id: str) -> LLMAdapter:
    return _ADAPTERS[provider_id]()


def info(provider_id: str) -> dict[str, str]:
    a = _ADAPTERS[provider_id]
    return {
        "id": a.PROVIDER_ID,
        "name": a.DISPLAY_NAME,
        "model_id": a.MODEL_ID,
        "access": a.ACCESS,
        "raw": a.RAW,
    }
