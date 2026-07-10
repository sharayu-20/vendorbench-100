"""king1oo1_deepguard — king1oo1/deepfake-model (Tier A, SigLIP2).

id2label: 0=Real, 1=Fake  ->  P(fake) = softmax["Fake"].
DeepGuard: SigLIP2 trained on ~330k images across 5 sources.
"""

from pathlib import Path

from benchmark.opensource_models.hf_classifier import HFClassifierClient

MODEL_ID = "king1oo1_deepguard"
HF_REPO = "king1oo1/deepfake-model"
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
