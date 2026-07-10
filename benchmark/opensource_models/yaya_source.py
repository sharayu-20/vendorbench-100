"""yaya_source — yaya36095/ai-source-detector (Tier A, ViT, multi-class source ID).

WARNING — defective upstream repo:
  * model.safetensors (1.1 GB) is a corrupt/mislabeled RoBERTa (text) checkpoint;
    loading it silently random-initialises the ViT. We force the real weights via
    use_safetensors=False (pytorch_model.bin, 343 MB).
  * config.json declares 5 labels but the checkpoint head is 6-class -> we override
    num_labels=6 so the real classifier loads (strict, no re-init).

id2label (documented, 5 of 6): 0=stable_diffusion, 1=midjourney, 2=dalle,
3=real, 4=other_ai (index 5 unnamed). Mapping: P(fake) = sum over every
non-real index = 1 - P(real@3).

NOTE: even with the correct weights this model does NOT separate real vs fake on
the Source set (AUC ~0.48 across every class interpretation). Kept for honest
coverage, but treat its score as non-discriminative on this data.
"""

from pathlib import Path

from benchmark.opensource_models.hf_classifier import HFClassifierClient

MODEL_ID = "yaya_source"
HF_REPO = "yaya36095/ai-source-detector"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
REAL_INDEX = 3
NUM_LABELS = 6
FAKE_INDEX = [i for i in range(NUM_LABELS) if i != REAL_INDEX]  # 0,1,2,4,5
IMAGE_SIZE = None
ENABLED = True


def build(device: str = "cuda:0", logs_dir: str = "logs") -> HFClassifierClient:
    entry = {
        "id": MODEL_ID,
        "hf_repo": HF_REPO,
        "model_path": LOCAL_DIR,
        "fake_index": FAKE_INDEX,
        "use_safetensors": False,
        "num_labels": NUM_LABELS,
    }
    if IMAGE_SIZE:
        entry["image_size"] = IMAGE_SIZE
    return HFClassifierClient(entry, device=device, logs_dir=logs_dir)
