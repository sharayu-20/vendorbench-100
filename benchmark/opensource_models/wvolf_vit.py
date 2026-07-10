"""wvolf_vit — Wvolf/ViT_Deepfake_Detection (Tier A, ViT).

id2label: 0=Real, 1=Fake  ->  P(fake) = softmax["Fake"].
Face-focused MSc deepfake detector (Solent Univ.), reported 98.7% on its test set.
"""

from pathlib import Path

from benchmark.opensource_models.hf_classifier import HFClassifierClient

MODEL_ID = "wvolf_vit"
HF_REPO = "Wvolf/ViT_Deepfake_Detection"
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
