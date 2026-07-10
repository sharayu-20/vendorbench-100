"""aidfr_real_v2 — prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0 (Tier A, SigLIP2, 3-class).

id2label: 0=Artificial, 1=Deepfake, 2=Real.
P(fake) = softmax["Artificial"] + softmax["Deepfake"]  (= 1 - P(Real)).
"""

from pathlib import Path

from benchmark.opensource_models.hf_classifier import HFClassifierClient

MODEL_ID = "aidfr_real_v2"
HF_REPO = "prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
FAKE_LABELS = ["Artificial", "Deepfake"]
IMAGE_SIZE = None
ENABLED = True


def build(device: str = "cuda:0", logs_dir: str = "logs") -> HFClassifierClient:
    entry = {"id": MODEL_ID, "hf_repo": HF_REPO, "model_path": LOCAL_DIR,
             "fake_labels": FAKE_LABELS}
    if IMAGE_SIZE:
        entry["image_size"] = IMAGE_SIZE
    return HFClassifierClient(entry, device=device, logs_dir=logs_dir)
