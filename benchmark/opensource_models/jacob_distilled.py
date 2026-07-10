"""jacob_distilled — jacoballessio/ai-image-detect-distilled (Tier A, distilled ViT).

id2label: 0=fake, 1=real  ->  P(fake) = softmax["fake"].
Three MJ/SD-vs-real detectors distilled into one small ViT (~11.8M params).
"""

from pathlib import Path

from benchmark.opensource_models.hf_classifier import HFClassifierClient

MODEL_ID = "jacob_distilled"
HF_REPO = "jacoballessio/ai-image-detect-distilled"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
FAKE_LABELS = ["fake"]
IMAGE_SIZE = None
ENABLED = True


def build(device: str = "cuda:0", logs_dir: str = "logs") -> HFClassifierClient:
    entry = {"id": MODEL_ID, "hf_repo": HF_REPO, "model_path": LOCAL_DIR,
             "fake_labels": FAKE_LABELS}
    if IMAGE_SIZE:
        entry["image_size"] = IMAGE_SIZE
    return HFClassifierClient(entry, device=device, logs_dir=logs_dir)
