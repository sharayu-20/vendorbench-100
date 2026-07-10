"""CLI entry point for the AI Detection API Benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmark.config import load_config
from benchmark.runner import run_benchmark, scan_dataset, _load_evidence_results, _build_clients
from benchmark.metrics import compute_all_metrics
from benchmark.report_html import generate_html_report
from benchmark.report_pdf import generate_pdf_report
from benchmark.models import BenchmarkRun, APIResult


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def cmd_run(args: argparse.Namespace) -> None:
    """Run the full benchmark pipeline."""
    config = load_config(args.config)
    print("\n" + "="*60)
    print("  AI Detection API Benchmark")
    print("="*60)
    print(f"  Config:   {args.config or 'config.env'}")
    print(f"  Dataset:  {config.settings.dataset_dir}")
    print(f"  Resume:   {not args.no_resume}")
    print(f"  Delay:    {config.settings.request_delay}s between requests")
    print("="*60)

    run = run_benchmark(config, resume=not args.no_resume)
    run = compute_all_metrics(run)

    print("\n" + "="*60)
    print("  Generating Reports")
    print("="*60)

    html_path = generate_html_report(run, config.settings.reports_dir)
    print(f"  HTML: {html_path}")

    pdf_path = generate_pdf_report(run, config.settings.reports_dir)
    print(f"  PDF:  {pdf_path}")

    _save_run_summary(run, config.settings.reports_dir)

    print("\n" + "="*60)
    print("  Benchmark Complete!")
    print("="*60)
    for pm in run.provider_metrics:
        print(f"  {pm.provider:20s}  Acc={pm.accuracy:.1%}  F1={pm.f1_score:.1%}  AUC={pm.roc_auc:.3f}")
    print("="*60 + "\n")


def cmd_report(args: argparse.Namespace) -> None:
    """Generate reports from existing evidence logs (no API calls)."""
    config = load_config(args.config)

    samples = scan_dataset(config.settings.dataset_dir)
    clients = _build_clients(config)

    run = BenchmarkRun(
        run_id="report-only",
        start_time=datetime.now(timezone.utc).isoformat(),
        dataset_size=len(samples),
        fake_count=sum(1 for s in samples if s.ground_truth == "FAKE"),
        real_count=sum(1 for s in samples if s.ground_truth == "REAL"),
        providers=[c.PROVIDER_NAME for c in clients],
    )

    run.results = _load_evidence_results(config.settings.logs_dir, clients, samples, [])
    if not run.results:
        print("[ERROR] No evidence logs found. Run the benchmark first.")
        sys.exit(1)

    print(f"Loaded {len(run.results)} results from evidence logs")
    run.end_time = datetime.now(timezone.utc).isoformat()
    run = compute_all_metrics(run)

    html_path = generate_html_report(run, config.settings.reports_dir)
    print(f"HTML: {html_path}")

    pdf_path = generate_pdf_report(run, config.settings.reports_dir)
    print(f"PDF:  {pdf_path}")

    _save_run_summary(run, config.settings.reports_dir)

    for c in clients:
        c.close()


def _save_run_summary(run: BenchmarkRun, reports_dir: str) -> None:
    """Save a machine-readable JSON summary of the run."""
    out = Path(reports_dir) / "run_summary.json"
    summary = {
        "run_id": run.run_id,
        "start_time": run.start_time,
        "end_time": run.end_time,
        "dataset_size": run.dataset_size,
        "fake_count": run.fake_count,
        "real_count": run.real_count,
        "providers": run.providers,
        "total_results": len(run.results),
        "metrics": {},
    }
    for pm in run.provider_metrics:
        summary["metrics"][pm.provider] = {
            "accuracy": round(pm.accuracy, 4),
            "precision": round(pm.precision, 4),
            "recall": round(pm.recall, 4),
            "specificity": round(pm.specificity, 4),
            "f1_score": round(pm.f1_score, 4),
            "mcc": round(pm.mcc, 4),
            "roc_auc": round(pm.roc_auc, 4),
            "latency_mean_ms": round(pm.latency_mean_ms, 1),
            "latency_median_ms": round(pm.latency_median_ms, 1),
            "latency_p95_ms": round(pm.latency_p95_ms, 1),
            "successful": pm.successful,
            "failed": pm.failed,
            "confusion_matrix": {
                "TP": pm.true_positives,
                "FP": pm.false_positives,
                "TN": pm.true_negatives,
                "FN": pm.false_negatives,
            },
        }
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI Detection API Benchmark - Compare detection providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to config.env file")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = sub.add_parser("run", help="Run full benchmark (API calls + reports)")
    run_parser.add_argument("--no-resume", action="store_true", help="Start fresh, ignore checkpoint")

    sub.add_parser("report", help="Generate reports from existing evidence logs")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "report":
        cmd_report(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
