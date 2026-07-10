"""Run the open-source (local GPU) detectors over the Source/ image set.

For each enabled model in benchmark/opensource_models/registry.py:
  * load it onto the GPU (one model at a time),
  * run every image -> write logs/opensource_models/<id>/<stem>.json,
  * free GPU memory, move to the next model.

Then compute per-model metrics (Acc / F1 / ROC-AUC) and save a summary.
Any model with AUC < 0.5 is flagged: its label->P(fake) mapping is reversed.

Run:
  .venv-research/bin/python scripts/run_research.py                 # all enabled, Source/
  .venv-research/bin/python scripts/run_research.py --limit 5       # quick smoke over 5 imgs
  .venv-research/bin/python scripts/run_research.py --models ateeqq_siglip2 wvolf_vit
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tqdm import tqdm  # noqa: E402

from benchmark.models import ImageSample, BenchmarkRun  # noqa: E402
from benchmark.metrics import compute_all_metrics  # noqa: E402
from benchmark.opensource_models.registry import enabled_ids, build, info  # noqa: E402

_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif")


def scan_source(dataset_dir: str) -> list[ImageSample]:
    """Case-insensitive scan for real/fake subfolders."""
    base = Path(dataset_dir)
    samples: list[ImageSample] = []
    for gt, names in (("FAKE", ("fake", "FAKE")), ("REAL", ("real", "REAL"))):
        for name in names:
            folder = base / name
            if folder.exists():
                for p in sorted(folder.iterdir()):
                    if p.is_file() and p.suffix.lower() in _EXTS:
                        samples.append(ImageSample(path=str(p), ground_truth=gt))
                break
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=str(ROOT / "Source"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="max images (0 = all)")
    ap.add_argument("--models", nargs="*", help="only these model ids")
    args = ap.parse_args()

    samples = scan_source(args.dataset)
    if args.limit:
        fakes = [s for s in samples if s.ground_truth == "FAKE"][: args.limit]
        reals = [s for s in samples if s.ground_truth == "REAL"][: args.limit]
        samples = fakes + reals
    n_fake = sum(1 for s in samples if s.ground_truth == "FAKE")
    n_real = len(samples) - n_fake
    print(f"Dataset: {args.dataset}  ->  {len(samples)} images ({n_fake} fake, {n_real} real)")

    ids = args.models if args.models else enabled_ids()
    print(f"Models: {len(ids)} enabled\n")

    run = BenchmarkRun(
        run_id="research-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"),
        start_time=datetime.now(timezone.utc).isoformat(),
        dataset_size=len(samples),
        fake_count=n_fake,
        real_count=n_real,
        providers=list(ids),
    )

    for mid in ids:
        meta = info(mid)
        print(f"[load] {mid}  ({meta['hf_repo']})")
        try:
            client = build(mid, device=args.device, logs_dir=str(ROOT / "logs"))
        except Exception as exc:  # noqa: BLE001
            print(f"  [SKIP] load failed: {type(exc).__name__}: {exc}")
            continue

        for sample in tqdm(samples, desc=f"  {mid:22s}", unit="img", ncols=90):
            run.results.append(client.detect(sample))
        client.unload()

    run.end_time = datetime.now(timezone.utc).isoformat()
    run = compute_all_metrics(run)

    # ---- summary ----
    print("\n" + "=" * 74)
    print(f"  {'model':24s} {'Acc':>6s} {'F1':>6s} {'AUC':>6s} {'ok?':>10s}")
    print("=" * 74)
    flipped: list[str] = []
    summary: dict[str, dict] = {}
    for pm in sorted(run.provider_metrics, key=lambda m: -m.roc_auc):
        flag = ""
        if pm.roc_auc and pm.roc_auc < 0.5:
            flag = "FLIP MAP"
            flipped.append(pm.provider)
        print(f"  {pm.provider:24s} {pm.accuracy:6.3f} {pm.f1_score:6.3f} {pm.roc_auc:6.3f} {flag:>10s}")
        summary[pm.provider] = {
            "accuracy": round(pm.accuracy, 4), "f1": round(pm.f1_score, 4),
            "roc_auc": round(pm.roc_auc, 4), "mcc": round(pm.mcc, 4),
            "precision": round(pm.precision, 4), "recall": round(pm.recall, 4),
            "TP": pm.true_positives, "FP": pm.false_positives,
            "TN": pm.true_negatives, "FN": pm.false_negatives,
            "successful": pm.successful, "failed": pm.failed,
            "latency_mean_ms": round(pm.latency_mean_ms, 1),
        }
    print("=" * 74)
    if flipped:
        print("\n[!] AUC < 0.5 (label->P(fake) reversed, flip mapping in the model's script):")
        for m in flipped:
            print("   -", m)

    out = ROOT / "reports" / "research_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "run_id": run.run_id, "dataset": args.dataset,
        "dataset_size": run.dataset_size, "fake": n_fake, "real": n_real,
        "models": summary, "flip_recommended": flipped,
    }, indent=2), encoding="utf-8")
    print(f"\nSummary -> {out}")
    print(f"Per-image logs -> logs/opensource_models/<model_id>/<image>.json")


if __name__ == "__main__":
    main()
