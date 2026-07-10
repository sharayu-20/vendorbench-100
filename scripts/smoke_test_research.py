"""Smoke test: run every enabled open-source model on 1 real + 1 fake image.

Validates that each model downloads, loads, runs, and that its label->P(fake)
mapping is not inverted (fake image should score HIGHER than the real image).

Run:  .venv-research/bin/python scripts/smoke_test_research.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.models import ImageSample  # noqa: E402
from benchmark.opensource_models.registry import enabled_ids, build  # noqa: E402


_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
# fully-AI-GENERATED fakes (not face-swaps) — the fair test for generation detectors
_GEN_HINTS = ("DALLE", "SDXL", "SD_", "MIDJOURNEY", "MJ", "FLUX", "IMAGEN", "FIREFLY", "GEMINI", "IDEOGRAM")


def _imgs(sub: str) -> list[Path]:
    d = ROOT / "Source" / sub
    return sorted(p for p in d.iterdir() if p.suffix.lower() in _EXTS)


def _sample(sub: str, n: int, prefer_generated: bool = False) -> list[str]:
    imgs = _imgs(sub)
    if prefer_generated:
        gen = [p for p in imgs if any(h in p.name.upper() for h in _GEN_HINTS)]
        imgs = (gen or imgs)
    return [str(p) for p in imgs[:n]]


def main() -> None:
    device = "cuda:0"
    n = 5
    reals = [ImageSample(path=p, ground_truth="REAL") for p in _sample("real", n)]
    fakes = [ImageSample(path=p, ground_truth="FAKE") for p in _sample("fake", n, prefer_generated=True)]
    print(f"reals: {[Path(s.path).name for s in reals]}")
    print(f"fakes: {[Path(s.path).name for s in fakes]}\n")

    ids = enabled_ids()
    print(f"{len(ids)} enabled models  (mean P(fake) over {n} images each)\n")
    header = f"{'model_id':24s} {'mean|real':>10s} {'mean|fake':>10s}  verdict"
    print(header)
    print("-" * len(header))

    flags: list[str] = []
    for mid in ids:
        try:
            client = build(mid, device=device, logs_dir=str(ROOT / "logs_smoke"))
            rc = [client.detect(s) for s in reals]
            fc = [client.detect(s) for s in fakes]
            errs = [x.error_message for x in rc + fc if not x.success]
            if errs:
                print(f"{mid:24s}  ERROR: {errs[0]}")
                flags.append(f"{mid}: inference error")
            else:
                mr = sum(x.confidence for x in rc) / len(rc)
                mf = sum(x.confidence for x in fc) / len(fc)
                ok = mf > mr
                verdict = "ok" if ok else "*** CHECK MAPPING ***"
                print(f"{mid:24s} {mr:10.4f} {mf:10.4f}  {verdict}")
                if not ok:
                    flags.append(f"{mid}: mean fake({mf:.3f}) <= mean real({mr:.3f})")
            client.unload()
        except Exception as exc:  # noqa: BLE001
            print(f"{mid:24s}  LOAD/RUN FAIL: {type(exc).__name__}: {exc}")
            flags.append(f"{mid}: {type(exc).__name__}")

    print("\n" + "=" * 60)
    if flags:
        print("REVIEW NEEDED:")
        for fl in flags:
            print("  -", fl)
    else:
        print("All models loaded and mappings look correct (fake > real).")


if __name__ == "__main__":
    main()
