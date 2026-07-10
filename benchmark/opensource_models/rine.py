"""rine — RINE (ECCV 2024, Tier C, custom).

"Leveraging Representations from Intermediate Encoder-blocks for Synthetic
Image Detection" (arXiv 2402.19091). A frozen OpenAI CLIP ViT-L/14 whose n
intermediate CLS tokens (one per transformer block, grabbed via ln_2 hooks) are
projected, weighted by a Trainable Importance Estimator, summed, and passed to a
small head. Only the trainable part ships in the repo's ckpt/ (CLIP is excluded
and re-downloaded via clip.load).

We reuse the repo's own `src.models.Model` (vendor/rine) and reproduce its
`get_our_trained_model`/`get_transforms` logic (ckpt loaded with strict=False;
the only "missing" keys are the frozen clip.* backbone). Default = the 4-class
ProGAN checkpoint used by the repo's demo. Output: P(fake) = sigmoid(logit).

Requires the OpenAI `clip` package; clip.load("ViT-L/14") pulls the (public)
CLIP backbone on first use.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from benchmark.opensource_models.base_research import BaseResearchClient

MODEL_ID = "rine"
HF_REPO = "mever-team/rine"  # GitHub; weights ship in-repo (ckpt/)
VENDOR_DIR = str(Path(__file__).resolve().parents[2] / "vendor" / "rine")
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
NCLS = 4  # repo demo default (4-class ProGAN); nproj=2, proj_dim=1024
CKPT_NAME = "model_4class_trainable.pth"
IMAGE_SIZE = 224
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
ENABLED = True

# (nproj, proj_dim) per ncls, from src.utils.get_our_trained_model.
_NCLS_CFG = {1: (4, 1024), 2: (4, 128), 4: (2, 1024), "ldm": (4, 1024)}


class RineClient(BaseResearchClient):
    def _load(self) -> None:
        import torch
        import torchvision.transforms as T

        self._torch = torch
        if VENDOR_DIR not in sys.path:
            sys.path.insert(0, VENDOR_DIR)
        from src.models import Model  # OpenAI CLIP ViT-L/14 + trainable head

        nproj, proj_dim = _NCLS_CFG[NCLS]
        self.model = Model(
            backbone=("ViT-L/14", 1024), nproj=nproj, proj_dim=proj_dim,
            device=self.device,
        )
        state = torch.load(str(Path(LOCAL_DIR) / CKPT_NAME), map_location=self.device)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        # ckpt intentionally omits the frozen backbone -> all "missing" must be clip.*
        bad = [k for k in missing if not k.startswith("clip.")]
        if bad or unexpected:
            raise RuntimeError(
                f"rine load mismatch: unexpected={unexpected[:3]} "
                f"non-clip-missing={bad[:3]}"
            )
        self.model.to(self.device).eval()

        # repo test transform (transforms_test_1): CenterCrop -> ToTensor -> CLIP-norm
        self._transform = T.Compose([
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])

    def _infer(self, image: Any) -> tuple[float, dict[str, Any]]:
        torch = self._torch
        x = self._transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(x)[0].flatten()[0]
            p_fake = float(torch.sigmoid(logit).item())
        raw = {
            "loader": "rine.src.models.Model (ncls=4)",
            "logit": round(float(logit.item()), 6),
            "prob_fake": round(p_fake, 6),
            "label_mapping": "P(fake) = sigmoid(logit)",
        }
        return p_fake, raw


def build(device: str = "cuda:0", logs_dir: str = "logs") -> RineClient:
    return RineClient(MODEL_ID, HF_REPO, device=device, logs_dir=logs_dir)
