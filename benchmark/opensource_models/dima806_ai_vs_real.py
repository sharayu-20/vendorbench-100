"""dima806_ai_vs_real — dima806/ai_vs_real_image_detection (Tier A, ViT).

id2label: 0=REAL, 1=FAKE  ->  P(fake) = softmax["FAKE"].
Classic CIFAKE-style baseline.
"""

from pathlib import Path

from benchmark.opensource_models.hf_classifier import HFClassifierClient

MODEL_ID = "dima806_ai_vs_real"
HF_REPO = "dima806/ai_vs_real_image_detection"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
FAKE_LABELS = ["FAKE"]
IMAGE_SIZE = None
ENABLED = True


def build(device: str = "cuda:0", logs_dir: str = "logs") -> HFClassifierClient:
    entry = {"id": MODEL_ID, "hf_repo": HF_REPO, "model_path": LOCAL_DIR,
             "fake_labels": FAKE_LABELS}
    if IMAGE_SIZE:
        entry["image_size"] = IMAGE_SIZE
    return HFClassifierClient(entry, device=device, logs_dir=logs_dir)
