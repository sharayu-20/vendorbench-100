"""Import pre-collected vision-LLM result dumps into canonical per-image logs.

For each provider in benchmark/llms/registry.py:
  * read its raw dump (raw_llm_results_zip/…),
  * normalize every verdict to the framework's APIResult schema,
  * write logs/llms/<provider_id>/<image_stem>.json.

Then recompute per-provider metrics (Acc / F1 / ROC-AUC + coverage) and save a
summary. No live API calls — the dumps are the source of truth.

Run:  .venv-research/bin/python scripts/import_llms.py
      .venv-research/bin/python scripts/import_llms.py --providers gemini qwen
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.models import BenchmarkRun  # noqa: E402
from benchmark.metrics import compute_all_metrics  # noqa: E402
from benchmark.llms.registry import enabled_ids, adapter, info  # noqa: E402

LOGS_DIR = ROOT / "logs" / "llms"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", nargs="*", help="only these provider ids")
    args = ap.parse_args()

    ids = args.providers if args.providers else enabled_ids()
    run = BenchmarkRun(
        run_id="llms-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        start_time=datetime.now(timezone.utc).isoformat(),
        providers=list(ids),
    )

    print(f"Importing {len(ids)} LLM providers -> {LOGS_DIR}\n")
    for pid in ids:
        meta = info(pid)
        results = adapter(pid).load()
        out_dir = LOGS_DIR / pid
        out_dir.mkdir(parents=True, exist_ok=True)
        for r in results:
            stem = Path(r.image_filename).stem
            (out_dir / f"{stem}.json").write_text(
                json.dumps(r.to_dict(), indent=2, default=str), encoding="utf-8")
        run.results.extend(results)
        ok = sum(1 for r in results if r.success)
        print(f"  {pid:18s} {meta['access']:8s} {len(results):3d} imgs  "
              f"({ok} ok / {len(results) - ok} fail)  <- {meta['raw']}")

    # dataset stats
    seen = {r.image_filename: r.ground_truth for r in run.results}
    n_fake = sum(1 for gt in seen.values() if gt == "FAKE")
    run.dataset_size = len(seen)
    run.fake_count = n_fake
    run.real_count = len(seen) - n_fake
    run.end_time = datetime.now(timezone.utc).isoformat()
    run = compute_all_metrics(run)

    print("\n" + "=" * 82)
    print(f"  {'provider':20s} {'Acc':>6s} {'F1':>6s} {'AUC':>6s} {'cover':>7s} {'TP/FP/TN/FN':>14s}")
    print("=" * 82)
    summary: dict[str, dict] = {}
    for pm in sorted(run.provider_metrics, key=lambda m: -m.roc_auc):
        cover = f"{pm.successful}/{pm.total_images}"
        print(f"  {pm.provider:20s} {pm.accuracy:6.3f} {pm.f1_score:6.3f} {pm.roc_auc:6.3f} "
              f"{cover:>7s} {f'{pm.true_positives}/{pm.false_positives}/{pm.true_negatives}/{pm.false_negatives}':>14s}")
        summary[pm.provider] = {
            "display": info(pm.provider)["name"], "model_id": info(pm.provider)["model_id"],
            "access": info(pm.provider)["access"],
            "accuracy": round(pm.accuracy, 4), "f1": round(pm.f1_score, 4),
            "roc_auc": round(pm.roc_auc, 4), "mcc": round(pm.mcc, 4),
            "precision": round(pm.precision, 4), "recall": round(pm.recall, 4),
            "specificity": round(pm.specificity, 4),
            "TP": pm.true_positives, "FP": pm.false_positives,
            "TN": pm.true_negatives, "FN": pm.false_negatives,
            "total": pm.total_images, "successful": pm.successful, "failed": pm.failed,
            "coverage": round(pm.successful / pm.total_images, 4) if pm.total_images else 0.0,
            "latency_mean_ms": round(pm.latency_mean_ms, 1),
        }
    print("=" * 82)

    out = ROOT / "reports" / "llm_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_id": run.run_id, "dataset_size": run.dataset_size,
        "fake": n_fake, "real": run.real_count,
        "failure_policy": "abstentions excluded from Acc/F1/AUC; coverage reported separately",
        "providers": summary,
    }, indent=2), encoding="utf-8")
    print(f"\nSummary -> {out}")
    print(f"Per-image logs -> logs/llms/<provider_id>/<image>.json")


if __name__ == "__main__":
    main()
