"""date3k2_vit — date3k2/vit-real-fake-classification-v3 (Tier A, ViT).

id2label: 0=Fake, 1=Real  ->  P(fake) = softmax["Fake"].
Fine-tuned google/vit-base-patch16-224; reported 98% eval accuracy.
"""

from pathlib import Path

from benchmark.opensource_models.hf_classifier import HFClassifierClient

MODEL_ID = "date3k2_vit"
HF_REPO = "date3k2/vit-real-fake-classification-v3"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
FAKE_LABELS = ["Fake"]
IMAGE_SIZE = None
ENABLED = True


def build(device: str = "cuda:0", logs_dir: str = "logs") -> HFClassifierClient:
    entry = {"id": MODEL_ID, "hf_repo": HF_REPO, "model_path": LOCAL_DIR,
             "fake_labels": FAKE_LABELS}
    if IMAGE_SIZE:
        entry["image_size"] = IMAGE_SIZE
    return HFClassifierClient(entry, device=device, logs_dir=logs_dir)
