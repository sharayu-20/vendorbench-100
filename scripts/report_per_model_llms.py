"""Generate one Markdown report per vision-LLM (read-only, regenerable).

Reads the canonical per-image logs (logs/llms/<id>/*.json) written by
scripts/import_llms.py, recomputes metrics with the shared pipeline, and merges
them with an embedded narrative (what the model is, how it was prompted, why it
was picked, and an analysis of its behaviour on our set). Emits:

    reports/llm-benchmark/models/<id>.md      (one file per LLM)

Prose mirrors reports/llm-benchmark/MODELS_REPORT.md so the per-model files and
the combined report stay in sync. No API calls.

Run:  .venv-research/bin/python scripts/report_per_model_llms.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.models import APIResult, BenchmarkRun  # noqa: E402
from benchmark.metrics import compute_all_metrics  # noqa: E402

LOGS_DIR = ROOT / "logs" / "llms"
OUT_DIR = ROOT / "reports" / "llm-benchmark" / "models"

_KEEP = {
    "provider", "image_path", "image_filename", "ground_truth", "prediction",
    "confidence", "raw_response", "latency_ms", "timestamp", "image_sha256",
    "success", "error_message", "retry_count",
}

# Shared detection protocol (identical prompt / JSON contract across all 7).
_PROTOCOL = (
    "Every provider was run zero-shot with the **same forensic prompt**: a system "
    "instruction to act as an image-forensics expert and return *only* a raw JSON "
    "verdict with fixed keys — `status` (`real`/`ai_manipulated`), `ai_manipulated` "
    "(bool), `probability_real`, `probability_ai_manipulated`, `source_type`, "
    "`short_summary`. We map **P(fake) = `probability_ai_manipulated`** (clamped to "
    "[0,1]) and take `FAKE` when `ai_manipulated` is true. No fine-tuning, no "
    "few-shot examples, one image per call.\n\n"
    "**No label leakage:** images were shown with **neutral numeric filenames** "
    "(`001.jpg`, …) split only by folder — the model never saw label-bearing names like "
    "`fake_001`/`real_001`, so it cannot infer the answer from the filename or metadata. "
    "The `fake_NNN`/`real_NNN` ids in this report are **post-hoc join keys** (never shown "
    "to the model); ground truth comes from the `fake`/`real` source folder."
)

# ---------------------------------------------------------------------------
# Per-LLM narrative. links = list of (label, url). result/bias reference the
# real computed metrics below and were written against this exact run.
# ---------------------------------------------------------------------------
DOCS: dict[str, dict] = {
    "claude_opus48": {
        "name": "Claude Opus 4.8", "vendor": "Anthropic", "kind": "Hosted frontier VLM",
        "access": "Anthropic Messages API (`claude-opus-4-8`, max_tokens 1024, JSON system prompt)",
        "one": "Anthropic's largest multimodal model; best ranking (AUC) of the seven and the only one that never false-flagged a real image.",
        "links": [("Anthropic — Claude", "https://www.anthropic.com/claude")],
        "bio": "Claude Opus 4.8 is Anthropic's flagship multimodal model — a proprietary, "
               "instruction-tuned frontier system with native image understanding. It was queried "
               "through the official Messages API with the image sent as base64 plus the shared "
               "forensic system prompt; responses are JSON verdicts parsed straight into our schema. "
               "This is the same client that was validated in the commercial-API track, so its "
               "normalization is battle-tested.",
        "why": "Included as the strongest general-purpose reasoning VLM available, to establish the "
               "ceiling for what a frontier model can do on deepfake screening with no task-specific "
               "training.",
        "result": "**Top of the LLM leaderboard by ROC-AUC (0.791).** Its confidence *ranking* "
                  "separates fake from real better than any other LLM here. Yet at the fixed 0.5 cut "
                  "it looks weak (Acc 0.49, F1 0.52) because it is strongly **conservative**: 51 of 79 "
                  "fakes were scored below 0.5 (called REAL) — but it produced **zero false positives** "
                  "(TP 28 / FP 0 / TN 21 / FN 51), i.e. it never wrongly accused a genuine photo. A "
                  "lower decision threshold would convert much of that high AUC into accuracy.",
        "bias": "Precision-first / recall-shy. Perfect specificity (21/21 reals correct) at the cost of "
                "under-calling AI images. Reads as a cautious expert that only says \"fake\" when very sure.",
        "limits": "Proprietary and closed-weight; latency ~9 s/image; verdict probabilities are "
                  "systematically under-confident for fakes, so the 0.5 threshold under-sells it — AUC "
                  "is the fair metric here.",
    },
    "gemini": {
        "name": "Gemini", "vendor": "Google DeepMind", "kind": "Hosted frontier VLM",
        "access": "Google Gemini API (JSON verdict, `result{...}` schema)",
        "one": "Google's multimodal model; the best-balanced LLM here — highest accuracy/F1 with strong ranking and near-perfect specificity.",
        "links": [("Google DeepMind — Gemini", "https://deepmind.google/technologies/gemini/")],
        "bio": "Gemini is Google DeepMind's proprietary multimodal family with strong native vision. "
               "It was called through its API with the shared forensic prompt, returning the same JSON "
               "verdict contract as the other hosted models.",
        "why": "Included as the principal frontier competitor to GPT/Claude and one of the most widely "
               "deployed multimodal APIs.",
        "result": "**Best all-round LLM at the 0.5 threshold: Acc 0.61, F1 0.68**, with a solid ROC-AUC "
                  "of 0.700 (2nd). It caught 41/79 fakes with just **one** false positive "
                  "(TP 41 / FP 1 / TN 20 / FN 38). It is less conservative than Claude — willing to "
                  "commit to \"fake\" more often — which pays off in thresholded accuracy while keeping "
                  "specificity high (20/21).",
        "bias": "The most balanced operating point of the seven: good recall without sacrificing "
                "specificity. Closest to usable out-of-the-box at a 0.5 cut.",
        "limits": "Proprietary/closed; no latency captured in this dump; still misses ~half the fakes, "
                  "so not a substitute for a purpose-built detector.",
    },
    "gpt_openai": {
        "name": "GPT (OpenAI)", "vendor": "OpenAI", "kind": "Hosted frontier VLM",
        "access": "OpenAI API (JSON verdict, `result{...}` schema)",
        "one": "OpenAI's multimodal model; moderate ranking (AUC 0.62) and very conservative at 0.5.",
        "links": [("OpenAI", "https://openai.com/")],
        "bio": "OpenAI's proprietary multimodal GPT model, queried through the OpenAI API with the "
               "shared forensic prompt and JSON verdict contract.",
        "why": "Included as the most recognizable frontier VLM and a natural reference point.",
        "result": "**ROC-AUC 0.621 (5th).** Like Claude and Qwen it is very conservative at 0.5 "
                  "(Acc 0.37, F1 0.36): it called REAL on 61 of 79 fakes (TP 18 / FP 2 / TN 19 / FN 61). "
                  "Its confidence ranking is only modestly better than chance on this generation-heavy "
                  "set, and its calibrated probabilities skew low for fakes.",
        "bias": "Cautious/recall-shy but with slightly weaker separation than Claude or Gemini — it "
                "both under-calls fakes *and* ranks them less cleanly.",
        "limits": "Proprietary/closed; no latency in dump; mid-pack ranking suggests limited transfer "
                  "to diffusion/face-swap artefacts under zero-shot prompting.",
    },
    "qwen": {
        "name": "Qwen", "vendor": "Alibaba (Qwen team)", "kind": "Hosted VLM (web UI)",
        "access": "Browser automation of chat.qwen.ai (screenshots + parsed JSON)",
        "one": "Alibaba's multimodal chat model, driven through its web UI; strong ranking (AUC 0.68), conservative at 0.5.",
        "links": [("Qwen Chat", "https://chat.qwen.ai/"), ("QwenLM (GitHub)", "https://github.com/QwenLM")],
        "bio": "Qwen is Alibaba's multimodal model family. Because no batch API was used here, it was "
               "driven via **browser automation of chat.qwen.ai** — each image uploaded through the web "
               "UI, the reply screenshotted and its JSON block parsed (see "
               "`raw_llm_results_zip/qwen/run_qwen_image_test.py`). All 100 images completed.",
        "why": "Included as a leading non-US multimodal system and to test whether a web-UI model "
               "matches the API frontier on forensic screening.",
        "result": "**ROC-AUC 0.678 (3rd)** — competitive ranking, above GPT and the open-weight models. "
                  "At 0.5 it is conservative (Acc 0.47, F1 0.51): 27/79 fakes caught, one false positive "
                  "(TP 27 / FP 1 / TN 20 / FN 52). Behaviour closely mirrors Claude's cautious profile.",
        "bias": "Recall-shy with high specificity; its probability scale under-weights fakes, so it "
                "ranks better (AUC) than it classifies at 0.5.",
        "limits": "Collected via browser automation (slower, screenshot-dependent, harder to reproduce "
                  "than an API); web-UI model version is not pinned in the dump.",
    },
    "zai_glm52": {
        "name": "Z.ai GLM-5.2", "vendor": "Zhipu AI / Z.ai", "kind": "Hosted VLM (web UI)",
        "access": "Browser automation of Z.ai GLM-5.2 (parsed JSON payload)",
        "one": "Zhipu's GLM multimodal model; a fake-everything responder — highest raw accuracy but the only sub-chance AUC.",
        "links": [("Z.ai", "https://z.ai/"), ("GLM (GitHub)", "https://github.com/THUDM/GLM")],
        "bio": "GLM-5.2 is Zhipu AI's (Z.ai) multimodal model, driven through browser automation "
               "(see `raw_llm_results_zip/z_ai/z_ai_test.py`). It returned the shared JSON verdict for "
               "all 100 images.",
        "why": "Included as another leading non-US multimodal system and a diversity check on the "
               "roster.",
        "result": "**A cautionary tale about base rates.** GLM-5.2 posts the *highest* headline accuracy "
                  "(0.71) and F1 (0.83) — but only because it labels almost **everything** as fake: "
                  "TP 71 / FP 21 / TN 0 / FN 8. It flagged **all 21 real images** as fake (specificity "
                  "0.00). Its ROC-AUC is **0.470 — below chance** — meaning its confidence ordering is "
                  "essentially uninformative. On a set that is 79% fake, a \"call everything fake\" "
                  "policy scores high on accuracy/F1, which is exactly why AUC and specificity, not "
                  "accuracy, are the honest metrics here.",
        "bias": "Extreme fake-positive bias (near-zero specificity). High accuracy is a base-rate "
                "artifact, not detection skill.",
        "limits": "Browser-collected; unusable as-is for screening (would flag every genuine photo); "
                  "model version not pinned.",
    },
    "llama4_maverick": {
        "name": "Llama-4-Maverick-17B-128E", "vendor": "Meta (via NVIDIA NIM)", "kind": "Open-weight MoE VLM",
        "access": "NVIDIA NIM OpenAI-compatible API (`meta/llama-4-maverick-17b-128e-instruct`, temp 1.0)",
        "one": "Meta's open-weight mixture-of-experts multimodal model; near-chance ranking and the most conservative of all seven.",
        "links": [("Meta — Llama", "https://ai.meta.com/llama/"),
                  ("NVIDIA build (NIM)", "https://build.nvidia.com/meta/llama-4-maverick-17b-128e-instruct")],
        "bio": "Llama-4-Maverick is Meta's **open-weight** 17B-active / 128-expert mixture-of-experts "
               "multimodal model, served here through NVIDIA NIM's OpenAI-compatible endpoint "
               "(`analyze_images.py`, temperature 1.0). All 100 images completed.",
        "why": "Included as the flagship open-weight multimodal model — the reference for what a freely "
               "downloadable system achieves versus the closed frontier.",
        "result": "**ROC-AUC 0.533 — barely above chance** and last-but-one. Extremely conservative at "
                  "0.5 (Acc 0.34, F1 0.28): it called REAL on 66 of 79 fakes with **zero** false "
                  "positives (TP 13 / FP 0 / TN 21 / FN 66). It rarely commits to \"fake,\" and when it "
                  "doesn't, its ranking barely orders the set.",
        "bias": "Maximally recall-shy: perfect specificity, but it misses the large majority of fakes "
                "and provides little usable ranking signal.",
        "limits": "Open weights are a plus for reproducibility, but zero-shot forensic performance is "
                  "near-random on this distribution; would need fine-tuning to be useful.",
    },
    "nemotron_nano_vl": {
        "name": "Nemotron-Nano-VL-8B", "vendor": "NVIDIA (via NVIDIA NIM)", "kind": "Open-weight compact VLM",
        "access": "NVIDIA NIM OpenAI-compatible API (`nvidia/llama-3.1-nemotron-nano-vl-8b-v1`, temp 1.0)",
        "one": "NVIDIA's compact 8B open-weight VLM; punches above its size on ranking (AUC 0.63) but abstained on 6 images.",
        "links": [("NVIDIA build (NIM)", "https://build.nvidia.com/nvidia/llama-3.1-nemotron-nano-vl-8b-v1"),
                  ("HF model card", "https://huggingface.co/nvidia/llama-3.1-nemotron-nano-vl-8b-v1")],
        "bio": "Nemotron-Nano-VL-8B is NVIDIA's **compact 8B open-weight** vision-language model "
               "(Llama-3.1 based), served through NVIDIA NIM. Six of 100 responses were unparseable "
               "(e.g. a server-side image-decode 500), so it answered **94/100**; those abstentions are "
               "excluded from Acc/F1/AUC and reported as coverage.",
        "why": "Included as the small/efficient open-weight option — to see how an 8B model compares to "
               "17B+ MoE and frontier systems on the same task.",
        "result": "**ROC-AUC 0.632 (4th)** on 94 answered images — notably ahead of the much larger "
                  "Llama-4-Maverick, and even GPT, on ranking. Conservative at 0.5 (Acc 0.39, F1 0.37): "
                  "17/79 fakes caught, one false positive (TP 17 / FP 1 / TN 20 / FN 56). A strong "
                  "size-for-performance showing.",
        "bias": "Recall-shy like the other open models, but with better separation per parameter; also "
                "the only model with real abstentions (coverage 94%).",
        "limits": "6% of images failed server-side; small model with a low ceiling; still misses most "
                  "fakes at 0.5 despite decent ranking.",
    },
}

_TIER = "LLM (zero-shot vision)"


def load_results() -> list[APIResult]:
    results: list[APIResult] = []
    for model_dir in sorted(LOGS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for jf in sorted(model_dir.glob("*.json")):
            d = json.loads(jf.read_text(encoding="utf-8"))
            results.append(APIResult(**{k: v for k, v in d.items() if k in _KEEP}))
    return results


def _fmt(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def build_run(results: list[APIResult]) -> BenchmarkRun:
    seen: dict[str, str] = {r.image_filename: r.ground_truth for r in results}
    n_fake = sum(1 for gt in seen.values() if gt == "FAKE")
    run = BenchmarkRun(
        run_id="per-model-llm", start_time="", end_time="",
        dataset_size=len(seen), fake_count=n_fake, real_count=len(seen) - n_fake,
        providers=sorted({r.provider for r in results}), results=results,
    )
    return compute_all_metrics(run)


def per_image_rows(results: list[APIResult], provider: str) -> list[dict]:
    rows = [r for r in results if r.provider == provider]
    rows.sort(key=lambda r: (r.confidence if r.confidence is not None else -1), reverse=True)
    out = []
    for r in rows:
        conf = r.confidence if r.confidence is not None else 0.0
        pred = r.prediction or ("FAKE" if conf >= 0.5 else "REAL")
        correct = (pred == r.ground_truth) if r.success else False
        out.append({"img": r.image_filename, "gt": r.ground_truth, "conf": conf,
                    "pred": pred, "ok": correct, "success": r.success})
    return out


def render(mid: str, doc: dict, m, rank: int, total: int, rows: list[dict], generated: str) -> str:
    links = " · ".join(f"[{lbl}]({url})" for lbl, url in doc["links"])
    answered = [r for r in rows if r["success"]]
    misses = [r for r in answered if not r["ok"]]
    fp = [r for r in misses if r["gt"] == "REAL"]   # real called fake
    fn = [r for r in misses if r["gt"] == "FAKE"]   # fake called real
    abstain = [r for r in rows if not r["success"]]
    cover = f"{m.successful}/{m.total_images}"

    def tbl(rs: list[dict]) -> str:
        lines = ["| # | Image | Truth | P(fake) | Predicted | Correct |",
                 "|---|---|---|---|---|---|"]
        for i, r in enumerate(rs, 1):
            if not r["success"]:
                mark, pred = "⚠️", "ERROR"
            else:
                mark, pred = ("✅" if r["ok"] else "❌"), r["pred"]
            lines.append(f"| {i} | `{r['img']}` | {r['gt']} | {_fmt(r['conf'])} | {pred} | {mark} |")
        return "\n".join(lines)

    md = f"""# {doc['name']} — `{mid}`

**{_TIER}** · **Vendor:** {doc['vendor']} · **Type:** {doc['kind']} · **Rank:** {rank} / {total} by ROC-AUC

> {doc['one']}

**Access:** {doc['access']}
**Links:** {links}
**Combined report:** [`../MODELS_REPORT.md`](../MODELS_REPORT.md) · **Catalogue:** [`../../../plan/LLMS.md`](../../../plan/LLMS.md)

## Metrics — 100-image `Source/` set (79 fake / 21 real)

| Metric | Value |
|---|---|
| ROC-AUC | **{_fmt(m.roc_auc)}** |
| Accuracy @0.5 | {_fmt(m.accuracy)} |
| F1 @0.5 | {_fmt(m.f1_score)} |
| Precision | {_fmt(m.precision)} |
| Recall (fake) | {_fmt(m.recall)} |
| Specificity (real) | {_fmt(m.specificity)} |
| MCC | {_fmt(m.mcc)} |
| TP / FP / TN / FN | {m.true_positives} / {m.false_positives} / {m.true_negatives} / {m.false_negatives} |
| Coverage (answered) | {cover} |
| Mean latency | {_fmt(m.latency_mean_ms, 1)} ms |

*ROC-AUC is threshold-free (ranking quality); Acc/F1/Prec/Rec/Spec are at a fixed 0.5 cut. On a 79%-fake set, accuracy and F1 are inflated by the base rate — AUC and specificity are the honest read. Abstentions are excluded from all rates and shown as coverage.*

## Overview

{doc['bio']}

## Detection protocol

{_PROTOCOL}

## Why we included it

{doc['why']}

## Our benchmark result — analysis

{doc['result']}

## Behavioural profile / bias

{doc['bias']}

## Limitations & caveats

{doc['limits']}

## Errors & misclassifications @0.5

- **Abstentions (unparseable / API error): {len(abstain)}** — {', '.join('`'+r['img']+'`' for r in abstain) if abstain else '_none_'}
- **False positives (real called fake): {len(fp)}** — {', '.join('`'+r['img']+'`' for r in fp) if fp else '_none_'}
- **False negatives (fake called real): {len(fn)}** — {len(fn)} images{' (e.g. ' + ', '.join('`'+r['img']+'`' for r in fn[:8]) + ('…' if len(fn) > 8 else '') + ')' if fn else ''}

<details>
<summary><b>Full {len(rows)}-image scores (sorted by P(fake), high → low)</b></summary>

{tbl(rows)}

</details>

---
*Generated {generated} by `scripts/report_per_model_llms.py` from `logs/llms/{mid}/`. Metrics recomputed with the shared `benchmark/metrics.py` pipeline.*
"""
    return md


def main() -> None:
    if not LOGS_DIR.exists():
        sys.exit(f"No logs at {LOGS_DIR} — run scripts/import_llms.py first.")
    results = load_results()
    if not results:
        sys.exit(f"No per-image logs under {LOGS_DIR}.")

    run = build_run(results)
    metrics = {m.provider: m for m in run.provider_metrics}
    ranked = sorted(run.provider_metrics, key=lambda m: m.roc_auc, reverse=True)
    rank_of = {m.provider: i for i, m in enumerate(ranked, 1)}
    total = len(ranked)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for mid, doc in DOCS.items():
        if mid not in metrics:
            print(f"  WARN no logs for {mid}, skipping")
            continue
        rows = per_image_rows(results, mid)
        md = render(mid, doc, metrics[mid], rank_of[mid], total, rows, generated)
        (OUT_DIR / f"{mid}.md").write_text(md, encoding="utf-8")
        written.append((rank_of[mid], mid))

    for mid in metrics:
        if mid not in DOCS:
            print(f"  WARN logs for {mid} but no narrative in DOCS")

    print(f"Wrote {len(written)} per-model LLM reports -> {OUT_DIR}")
    for rank, mid in sorted(written):
        print(f"  #{rank}  {mid}")


if __name__ == "__main__":
    main()
