"""sadra_sdxl_face — SadraCoding/SDXL-Deepfake-Detector (Tier A, ViT/Swin).

id2label: 0=artificial, 1=human  ->  P(fake) = softmax["artificial"].
Face-focused (fine-tuned from Organika/sdxl-detector on 140k real/fake faces).
Docs confirm artificial=0. Note: face-only -> weaker on non-face imagery.
"""

from pathlib import Path

from benchmark.opensource_models.hf_classifier import HFClassifierClient

MODEL_ID = "sadra_sdxl_face"
HF_REPO = "SadraCoding/SDXL-Deepfake-Detector"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
FAKE_LABELS = ["artificial"]
IMAGE_SIZE = None
ENABLED = True


def build(device: str = "cuda:0", logs_dir: str = "logs") -> HFClassifierClient:
    entry = {"id": MODEL_ID, "hf_repo": HF_REPO, "model_path": LOCAL_DIR,
             "fake_labels": FAKE_LABELS}
    if IMAGE_SIZE:
        entry["image_size"] = IMAGE_SIZE
    return HFClassifierClient(entry, device=device, logs_dir=logs_dir)
