"""declip — DeCLIP (WACV 2025, Bitdefender, Tier C, custom).

"DeCLIP: Decoding CLIP Representations for Deepfake Localization" (arXiv
2409.08849). A frozen CLIP ViT-L/14 whose intermediate-layer features feed a
convolutional decoder that predicts a per-patch manipulation map. Although the
model is trained for localization, the repo's own detection path
(validate.py:validate) turns it into a detector by:  P(fake) = mean(sigmoid(map)).
We use that exact reduction — it is DeCLIP's official detection score, not an
ad-hoc one.

We reuse the repo's `models.clip_models.CLIPModelLocalisation` (vendor/declip,
with its vendored CLIP under models/clip). feature_layer + decoder_type are read
straight from the checkpoint. Weights load with strict=False (the frozen CLIP
backbone is re-created by clip.load). Test transform mirrors the repo's
data/datasets.py: Resize((224,224)) -> ToTensor -> CLIP-normalize.

Checkpoint: the LDM-trained ViT checkpoint (paper reports LDM training
generalizes best) from the public GCS bucket bitdefender_ml_artifacts/declip.

Perf note: on the 100-image Source set this scores ROC-AUC 0.275 (below chance).
Not a mapping bug — DeCLIP is a *localization* detector for partial manipulations
(e.g. LDM inpainting); it fires on manipulated regions, so fully-synthetic images
(DALL·E / SDXL) and face-swaps produce near-empty maps and low mean scores. We
keep the repo's own detection reduction rather than flipping to game AUC.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from benchmark.opensource_models.base_research import BaseResearchClient

MODEL_ID = "declip"
HF_REPO = "bit-ml/DeCLIP"  # GitHub; weights on GCS bitdefender_ml_artifacts
VENDOR_DIR = str(Path(__file__).resolve().parents[2] / "vendor" / "declip")
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
CKPT_NAME = "ViT_layer20_conv-20_ldm.pth"
ARCH = "ViT-L/14"
IMAGE_SIZE = 224
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
ENABLED = True


class DeclipClient(BaseResearchClient):
    def _load(self) -> None:
        import torch
        import torchvision.transforms as T

        self._torch = torch
        if VENDOR_DIR not in sys.path:
            sys.path.insert(0, VENDOR_DIR)
        from models.clip_models import CLIPModelLocalisation

        state = torch.load(str(Path(LOCAL_DIR) / CKPT_NAME), map_location="cpu")
        feature_layer = state.get("feature_layer", "layer20")
        decoder_type = state.get("decoder_type", "conv-20")
        self._meta = {"feature_layer": feature_layer, "decoder_type": decoder_type}

        self.model = CLIPModelLocalisation(
            ARCH, intermidiate_layer_output=feature_layer, decoder_type=decoder_type,
        )
        self.model.load_state_dict(state["model"], strict=False)
        self.model.to(self.device).eval()

        self._transform = T.Compose([
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])

    def _infer(self, image: Any) -> tuple[float, dict[str, Any]]:
        torch = self._torch
        x = self._transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            patch_map = torch.sigmoid(self.model(x))  # (1, N) per-patch P(fake)
            p_fake = float(torch.mean(patch_map, dim=1)[0].item())
        raw = {
            "loader": "declip.models.CLIPModelLocalisation",
            "feature_layer": self._meta["feature_layer"],
            "decoder_type": self._meta["decoder_type"],
            "n_patches": int(patch_map.shape[1]),
            "prob_fake": round(p_fake, 6),
            "label_mapping": "P(fake) = mean(sigmoid(localization_map))  (repo detection path)",
        }
        return p_fake, raw


def build(device: str = "cuda:0", logs_dir: str = "logs") -> DeclipClient:
    return DeclipClient(MODEL_ID, HF_REPO, device=device, logs_dir=logs_dir)
