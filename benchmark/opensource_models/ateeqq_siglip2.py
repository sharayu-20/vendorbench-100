"""ateeqq_siglip2 — Ateeqq/ai-vs-human-image-detector (Tier A, SigLIP2).

id2label: 0=ai, 1=hum  ->  P(fake) = softmax["ai"].
"""

from pathlib import Path

from benchmark.opensource_models.hf_classifier import HFClassifierClient

MODEL_ID = "ateeqq_siglip2"
HF_REPO = "Ateeqq/ai-vs-human-image-detector"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
FAKE_LABELS = ["ai"]
IMAGE_SIZE = None
ENABLED = True


def build(device: str = "cuda:0", logs_dir: str = "logs") -> HFClassifierClient:
    entry = {"id": MODEL_ID, "hf_repo": HF_REPO, "model_path": LOCAL_DIR,
             "fake_labels": FAKE_LABELS}
    if IMAGE_SIZE:
        entry["image_size"] = IMAGE_SIZE
    return HFClassifierClient(entry, device=device, logs_dir=logs_dir)
