"""ash_flux_vit — ash12321/flux-detector-vit (Tier A, ViT).

Generic labels (LABEL_0 / LABEL_1); index-based mapping: fake = index 1
(confirmed directionally correct on the smoke/full run). FLUX.1 specialist.
"""

from pathlib import Path

from benchmark.opensource_models.hf_classifier import HFClassifierClient

MODEL_ID = "ash_flux_vit"
HF_REPO = "ash12321/flux-detector-vit"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
FAKE_INDEX = [1]
IMAGE_SIZE = None
ENABLED = True


def build(device: str = "cuda:0", logs_dir: str = "logs") -> HFClassifierClient:
    entry = {"id": MODEL_ID, "hf_repo": HF_REPO, "model_path": LOCAL_DIR,
             "fake_index": FAKE_INDEX}
    if IMAGE_SIZE:
        entry["image_size"] = IMAGE_SIZE
    return HFClassifierClient(entry, device=device, logs_dir=logs_dir)
