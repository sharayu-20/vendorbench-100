"""Build the open-source track report from stored per-image logs.

Read-only: reconstructs a BenchmarkRun from logs/opensource_models/<id>/*.json
(the exact evidence written by run_research.py), recomputes metrics, sorts models
by ROC-AUC, and renders the shared HTML report. No GPU / no re-inference.

Run:  .venv-research/bin/python scripts/report_research.py
Out:  reports/opensource-benchmark/benchmark_report.html
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
from benchmark.report_html import generate_html_report  # noqa: E402

LOGS_DIR = ROOT / "logs" / "opensource_models"
OUT_DIR = ROOT / "reports" / "opensource-benchmark"

_KEEP = {
    "provider", "image_path", "image_filename", "ground_truth", "prediction",
    "confidence", "raw_response", "latency_ms", "timestamp", "image_sha256",
    "success", "error_message", "retry_count",
}


def load_results() -> list[APIResult]:
    results: list[APIResult] = []
    for model_dir in sorted(LOGS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for jf in sorted(model_dir.glob("*.json")):
            d = json.loads(jf.read_text(encoding="utf-8"))
            results.append(APIResult(**{k: v for k, v in d.items() if k in _KEEP}))
    return results


def main() -> None:
    if not LOGS_DIR.exists():
        sys.exit(f"No logs at {LOGS_DIR} — run scripts/run_research.py first.")

    results = load_results()
    if not results:
        sys.exit(f"No per-image logs found under {LOGS_DIR}.")

    # dataset stats from the distinct images seen
    seen: dict[str, str] = {}
    for r in results:
        seen[r.image_filename] = r.ground_truth
    n_fake = sum(1 for gt in seen.values() if gt == "FAKE")
    n_real = len(seen) - n_fake

    providers = sorted({r.provider for r in results})
    run = BenchmarkRun(
        run_id="opensource-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        start_time=datetime.now(timezone.utc).isoformat(),
        end_time=datetime.now(timezone.utc).isoformat(),
        dataset_size=len(seen),
        fake_count=n_fake,
        real_count=n_real,
        providers=providers,
        results=results,
    )

    run = compute_all_metrics(run)
    # rank models best-first by ROC-AUC (drives tables + charts order)
    run.provider_metrics.sort(key=lambda m: m.roc_auc, reverse=True)

    print(f"Models: {len(run.provider_metrics)}  |  images: {len(seen)} "
          f"({n_fake} fake, {n_real} real)  |  results: {len(results)}")
    for pm in run.provider_metrics:
        print(f"  {pm.provider:22s} AUC={pm.roc_auc:.3f}  Acc={pm.accuracy:.3f}  "
              f"F1={pm.f1_score:.3f}")

    html_path = generate_html_report(run, str(OUT_DIR))
    print(f"\nHTML report -> {html_path}")


if __name__ == "__main__":
    main()
