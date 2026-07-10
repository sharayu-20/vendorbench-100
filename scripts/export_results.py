"""Export the published `results/` layer from the raw per-image logs.

Aggregates the scattered evidence in `logs/<track>/<id>/*.json` into the two
shareable artifacts the plan calls "results" — one per track:

    results/<track>-benchmark/summary.json    per-provider metrics (ranked by ROC-AUC)
    results/<track>-benchmark/per-image.csv    one row per (provider, image)

This is the `logs -> aggregate -> results` step. Read-only w.r.t. logs; no
inference or API calls. Metrics use the shared `benchmark/metrics.py` pipeline.

Run:  .venv-research/bin/python scripts/export_results.py
      .venv-research/bin/python scripts/export_results.py --tracks llms
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.models import APIResult, BenchmarkRun  # noqa: E402
from benchmark.metrics import compute_all_metrics  # noqa: E402

LOGS_DIR = ROOT / "logs"
RESULTS_DIR = ROOT / "results"

# logs subdir  ->  published results/reports dir name
TRACKS = {
    "commercial_apis": "commercial-benchmark",
    "opensource_models": "opensource-benchmark",
    "llms": "llm-benchmark",
}

_KEEP = {
    "provider", "image_path", "image_filename", "ground_truth", "prediction",
    "confidence", "raw_response", "latency_ms", "timestamp", "image_sha256",
    "success", "error_message", "retry_count",
}


def load_track(track: str) -> list[APIResult]:
    base = LOGS_DIR / track
    results: list[APIResult] = []
    if not base.exists():
        return results
    for provider_dir in sorted(base.iterdir()):
        if not provider_dir.is_dir():
            continue
        for jf in sorted(provider_dir.glob("*.json")):
            d = json.loads(jf.read_text(encoding="utf-8"))
            results.append(APIResult(**{k: v for k, v in d.items() if k in _KEEP}))
    return results


def export(track: str, out_name: str) -> str | None:
    results = load_track(track)
    if not results:
        print(f"  [skip] no logs for {track}")
        return None

    seen = {r.image_filename: r.ground_truth for r in results}
    n_fake = sum(1 for gt in seen.values() if gt == "FAKE")
    run = BenchmarkRun(
        run_id=f"{out_name}-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        dataset_size=len(seen), fake_count=n_fake, real_count=len(seen) - n_fake,
        providers=sorted({r.provider for r in results}), results=results,
    )
    run = compute_all_metrics(run)
    ranked = sorted(run.provider_metrics, key=lambda m: m.roc_auc, reverse=True)

    out_dir = RESULTS_DIR / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # summary.json
    summary = {
        "track": track, "run_id": run.run_id,
        "dataset_size": run.dataset_size, "fake": n_fake, "real": run.real_count,
        "providers": {},
    }
    for m in ranked:
        summary["providers"][m.provider] = {
            "roc_auc": round(m.roc_auc, 4), "accuracy": round(m.accuracy, 4),
            "f1": round(m.f1_score, 4), "precision": round(m.precision, 4),
            "recall": round(m.recall, 4), "specificity": round(m.specificity, 4),
            "mcc": round(m.mcc, 4),
            "TP": m.true_positives, "FP": m.false_positives,
            "TN": m.true_negatives, "FN": m.false_negatives,
            "total": m.total_images, "successful": m.successful, "failed": m.failed,
            "coverage": round(m.successful / m.total_images, 4) if m.total_images else 0.0,
            "latency_mean_ms": round(m.latency_mean_ms, 1),
        }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # per-image.csv
    rank_of = {m.provider: i for i, m in enumerate(ranked, 1)}
    rows = sorted(results, key=lambda r: (rank_of.get(r.provider, 999), r.image_filename))
    with open(out_dir / "per-image.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["provider", "image_filename", "ground_truth", "prediction",
                    "confidence", "success", "correct", "latency_ms"])
        for r in rows:
            correct = int(r.success and r.prediction == r.ground_truth)
            w.writerow([r.provider, r.image_filename, r.ground_truth, r.prediction,
                        f"{r.confidence:.4f}", int(r.success), correct,
                        f"{r.latency_ms:.1f}"])

    print(f"  {track:18s} -> results/{out_name}/  "
          f"({len(ranked)} providers, {len(results)} rows)")
    return out_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="*", choices=list(TRACKS.keys()),
                    help="only these logs tracks (default: all)")
    args = ap.parse_args()
    tracks = args.tracks if args.tracks else list(TRACKS.keys())

    print(f"Exporting results from logs -> {RESULTS_DIR}\n")
    for track in tracks:
        export(track, TRACKS[track])
    print("\nDone. Each results/<track>-benchmark/ has summary.json + per-image.csv.")


if __name__ == "__main__":
    main()
