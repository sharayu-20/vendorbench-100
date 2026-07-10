"""drct — DRCT (ICML 2024 Spotlight, Tier C, custom).

"Diffusion Reconstruction Contrastive Training towards Universal Detection of
Diffusion Generated Images" (icml.cc/virtual/2024/poster/33086). The detector is
a ConvNeXt-Base(in22k) backbone -> 1024-d embedding -> 2-class head, trained with
margin-based contrastive loss over real / fake / diffusion-reconstructed images.

We reuse the repo's own `network.models.get_models` (vendor/drct) to build the
exact ContrastiveModels architecture and load the ModelScope checkpoint
(convnext_base_in22k, DRCT-2M, DR=SDv1) with strict=True. Test transform mirrors
the repo's create_val_transforms(is_crop=True): pad-if-small -> CenterCrop(224)
-> ImageNet normalize. Output: P(fake) = softmax(logits)[1]  (0=real, 1=fake),
matching the repo's eval (`1 - softmax[:,0]`).

Checkpoint source: modelscope BokingChen/DRCT-2M/pretrained.zip (4.2 GB bundle;
we extract only the ConvNeXt DRCT-2M/sdv14 checkpoint).

Perf note: ROC-AUC 0.866 on the 100-image Source set — the best Tier C model;
the diffusion-reconstruction training generalizes well to this diffusion-heavy
set. (Accuracy at the fixed 0.5 threshold is low only because its scores are
peaky/miscalibrated — ranking is strong.)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from benchmark.opensource_models.base_research import BaseResearchClient

MODEL_ID = "drct"
HF_REPO = "BokingChen/DRCT-2M"  # ModelScope dataset (weights in pretrained.zip)
VENDOR_DIR = str(Path(__file__).resolve().parents[2] / "vendor" / "drct")
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
CKPT_NAME = "convnext_base_in22k_drct2m_sdv14.pth"  # extracted from pretrained.zip
MODEL_NAME = "convnext_base_in22k"
EMBEDDING_SIZE = 1024
IMAGE_SIZE = 224
FAKE_INDEX = 1
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
ENABLED = True


class DrctClient(BaseResearchClient):
    def _load(self) -> None:
        import torch
        import torchvision.transforms as T

        self._torch = torch
        if VENDOR_DIR not in sys.path:
            sys.path.insert(0, VENDOR_DIR)
        from network.models import get_models

        self.model = get_models(
            model_name=MODEL_NAME, num_classes=2, pretrained=False,
            embedding_size=EMBEDDING_SIZE,
        )
        state = torch.load(str(Path(LOCAL_DIR) / CKPT_NAME), map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

        # create_val_transforms(is_crop=True): pad-if-small -> CenterCrop -> IN-norm
        self._transform = T.Compose([
            T.Lambda(self._pad_if_small),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])

    @staticmethod
    def _pad_if_small(img):
        from PIL import Image

        w, h = img.size
        if w >= IMAGE_SIZE and h >= IMAGE_SIZE:
            return img
        nw, nh = max(w, IMAGE_SIZE), max(h, IMAGE_SIZE)
        canvas = Image.new("RGB", (nw, nh), (0, 0, 0))
        canvas.paste(img, ((nw - w) // 2, (nh - h) // 2))
        return canvas

    def _infer(self, image: Any) -> tuple[float, dict[str, Any]]:
        torch = self._torch
        x = self._transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].float().cpu().tolist()
        p_fake = float(probs[FAKE_INDEX])
        raw = {
            "loader": "drct.network.models.get_models (ContrastiveModels convnext_base_in22k)",
            "probs": {"real": round(probs[0], 6), "fake": round(probs[1], 6)},
            "prob_fake": round(p_fake, 6),
            "label_mapping": "P(fake) = softmax(logits)[1]  (0=real, 1=fake)",
        }
        return p_fake, raw


def build(device: str = "cuda:0", logs_dir: str = "logs") -> DrctClient:
    return DrctClient(MODEL_ID, HF_REPO, device=device, logs_dir=logs_dir)
