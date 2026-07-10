# Vision LLMs (LLM track)

**7 models (5 hosted + 2 open-weight), July 2026.** Same 100-image `Source/` set
(79 fake / 21 real) as every other track, screened **zero-shot** with one shared
forensic JSON-verdict prompt (`status`, `ai_manipulated`, `probability_real`,
`probability_ai_manipulated`, `source_type`, `short_summary`).

**As built:** results were collected out-of-band (some via vendor API, some via
browser automation) and live under `raw_llm_results_zip/`. Each model has an
**adapter** in `benchmark/llms/<id>.py` that normalizes its dump into the canonical
`APIResult` schema — this track imports evidence, it does not re-call the models.

## Roster

| # | ID | Model | Access | Model id / slug | Cover |
|---|---|---|---|---|---|
| 1 | `claude_opus48` | Anthropic Claude Opus 4.8 | API | `claude-opus-4-8` | 100/100 |
| 2 | `gemini` | Google Gemini (vision) | API | `gemini` | 100/100 |
| 3 | `gpt_openai` | OpenAI GPT (vision) | API | `openai-gpt` | 100/100 |
| 4 | `zai_glm52` | Z.ai GLM-5.2 (Zhipu AI) | browser | `glm-5.2` | 100/100 |
| 5 | `qwen` | Qwen (Alibaba) | browser (chat.qwen.ai) | `qwen` | 100/100 |
| 6 | `llama4_maverick` | Meta Llama-4-Maverick-17B-128E | NVIDIA NIM | `meta/llama-4-maverick-17b-128e-instruct` | 100/100 |
| 7 | `nemotron_nano_vl` | NVIDIA Nemotron-Nano-VL-8B | NVIDIA NIM | `nvidia/llama-3.1-nemotron-nano-vl-8b-v1` | 94/100 |

## Results (ROC-AUC, this run)

| Rank | Model | AUC | Acc@0.5 | F1 | Spec | Note |
|---|---|---|---|---|---|---|
| 1 | `claude_opus48` | **0.791** | 0.49 | 0.523 | 1.00 | best ranker; zero false positives |
| 2 | `gemini` | 0.700 | 0.61 | 0.678 | 0.95 | best balanced operating point |
| 3 | `qwen` | 0.678 | 0.47 | 0.505 | 0.95 | conservative |
| 4 | `nemotron_nano_vl` | 0.632 | 0.39 | 0.374 | 0.95 | best per-parameter; 6 abstentions |
| 5 | `gpt_openai` | 0.621 | 0.37 | 0.364 | 0.90 | conservative |
| 6 | `llama4_maverick` | 0.533 | 0.34 | 0.283 | 1.00 | near-chance; most conservative |
| 7 | `zai_glm52` | 0.470 | 0.71 | 0.830 | 0.00 | fake-everything; base-rate trap |

**Read AUC + specificity, not accuracy** — the set is 79% fake, so "call everything
fake" (GLM-5.2) posts high accuracy with zero real skill. Abstentions are excluded
from rates and reported as coverage.

## Reproduce

```
.venv-research/bin/python scripts/import_llms.py          # dumps -> logs/llms/<id>/*.json + reports/llm_summary.json
.venv-research/bin/python scripts/report_llms.py          # -> reports/llm-benchmark/benchmark_report.html
.venv-research/bin/python scripts/report_per_model_llms.py # -> reports/llm-benchmark/models/<id>.md
```

**Reports:** [`../reports/llm-benchmark/MODELS_REPORT.md`](../reports/llm-benchmark/MODELS_REPORT.md)
· per-model files in `reports/llm-benchmark/models/`.
