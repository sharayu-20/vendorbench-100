"""opensight_commfor — aiwithoutborders-xyz/OpenSight-CommunityForensics-Deepfake-ViT.

Generic labels (LABEL_0 / LABEL_1); index-based mapping: fake = index 1.
Preprocessor reports 440px but the model's pos-embeds require 384px -> force 384.
"""

from pathlib import Path

from benchmark.opensource_models.hf_classifier import HFClassifierClient

MODEL_ID = "opensight_commfor"
HF_REPO = "aiwithoutborders-xyz/OpenSight-CommunityForensics-Deepfake-ViT"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
FAKE_INDEX = [1]
IMAGE_SIZE = 384
ENABLED = True


def build(device: str = "cuda:0", logs_dir: str = "logs") -> HFClassifierClient:
    entry = {"id": MODEL_ID, "hf_repo": HF_REPO, "model_path": LOCAL_DIR,
             "fake_index": FAKE_INDEX, "image_size": IMAGE_SIZE}
    return HFClassifierClient(entry, device=device, logs_dir=logs_dir)
