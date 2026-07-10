"""Download every enabled open-source model into models/<id>/.

Populates a per-model subfolder with the real weight/config files (pulled from
the HF cache if already present). After this, the benchmark loads models from
the local models/<id>/ directory (offline).

Run:  .venv-research/bin/python scripts/download_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.opensource_models.registry import enabled_ids, info  # noqa: E402


def main() -> None:
    from huggingface_hub import snapshot_download

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)

    ids = enabled_ids()
    print(f"Downloading {len(ids)} models into {models_dir}/\n")
    for mid in ids:
        repo = info(mid)["hf_repo"]
        dest = models_dir / mid
        print(f"[{mid}]  {repo}  ->  models/{mid}/")
        try:
            snapshot_download(
                repo_id=repo,
                local_dir=str(dest),
                ignore_patterns=["*.msgpack", "*.h5", "*.onnx", "*.tflite", "*.pth"],
            )
            size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
            print(f"   done ({size/1e6:.0f} MB)")
        except Exception as exc:  # noqa: BLE001
            print(f"   FAILED: {type(exc).__name__}: {exc}")

    print("\nAll model folders:")
    for d in sorted(models_dir.iterdir()):
        if d.is_dir():
            n = sum(1 for _ in d.rglob("*") if _.is_file())
            print(f"  models/{d.name}/  ({n} files)")


if __name__ == "__main__":
    main()
