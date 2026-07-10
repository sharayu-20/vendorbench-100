"""c2p_clip — C2P-CLIP (AAAI 2025, Tier C, custom).

Category Common Prompt CLIP: a frozen CLIP-ViT-L/14 vision encoder + visual
projection with a single-logit linear head (`model.fc`). At test time the text
tower is unused — the prompt was only injected during training — so inference is
just: CLIP image embed -> L2-normalize -> fc -> sigmoid = P(fake).

We reproduce the repo's own `C2P_CLIP` nn.Module (vendor/c2p_clip/scripts/
inference.py) and its test transform (vendor/c2p_clip/data/datasets.py, the
no_resize=False / no_crop=False branch: tile-up-if-small -> CenterCrop(224) ->
ToTensor -> CLIP-normalize). The checkpoint is a full strict=True state_dict
(1.2 GB, mirrored on HF siddharthksah/deepsafe-weights). The bare
openai/clip-vit-large-patch14 is pulled once to build the architecture, then
fully overwritten by the checkpoint.

Ref: https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection (arXiv 2408.09647)

Perf note: on the 100-image Source set this scores ROC-AUC 0.385 (below chance).
The mapping is NOT inverted — it's the repo's own convention (verified: real_001
correctly scores 0.03). C2P-CLIP was trained on ProGAN/GAN (ForenSynths) and
underperforms on this diffusion + face-swap distribution. We keep the faithful
mapping + the repo's exact test transform rather than flipping to game AUC.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from benchmark.opensource_models.base_research import BaseResearchClient

MODEL_ID = "c2p_clip"
HF_REPO = "chuangchuangtan/C2P-CLIP-DeepfakeDetection"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
CLIP_BACKBONE = "openai/clip-vit-large-patch14"
CKPT_NAME = "C2P_CLIP_release_20240901.pth"
IMAGE_SIZE = 224
# CLIP normalization constants (from the repo's data/datasets.py test transform).
_CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
_CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
ENABLED = True


def _translate_duplicate(img, crop_size: int):
    """Repo's tile-up-if-smaller-than-crop helper (no downscaling)."""
    from PIL import Image

    if min(img.size) < crop_size:
        w, h = img.size
        nw, nh = w * math.ceil(crop_size / w), h * math.ceil(crop_size / h)
        tiled = Image.new("RGB", (nw, nh))
        for i in range(0, nw, w):
            for j in range(0, nh, h):
                tiled.paste(img, (i, j))
        return tiled
    return img


def _build_c2p_clip(clip_name: str):
    """Faithful copy of scripts/inference.py:C2P_CLIP (num_classes=1)."""
    import torch
    import torch.nn as nn
    from transformers import CLIPModel

    class C2P_CLIP(nn.Module):
        def __init__(self, name: str, num_classes: int = 1) -> None:
            super().__init__()
            self.model = CLIPModel.from_pretrained(name)
            del self.model.text_model
            del self.model.text_projection
            del self.model.logit_scale
            self.model.vision_model.requires_grad_(False)
            self.model.visual_projection.requires_grad_(False)
            self.model.fc = nn.Linear(768, num_classes)

        def encode_image(self, img):
            vision_outputs = self.model.vision_model(pixel_values=img)
            pooled_output = vision_outputs[1]
            return self.model.visual_projection(pooled_output)

        def forward(self, img):
            embeds = self.encode_image(img)
            embeds = embeds / embeds.norm(p=2, dim=-1, keepdim=True)
            return self.model.fc(embeds)

    return C2P_CLIP(clip_name, num_classes=1)


class C2PClipClient(BaseResearchClient):
    def _load(self) -> None:
        import torch
        import torchvision.transforms as T

        self._torch = torch
        self.model = _build_c2p_clip(CLIP_BACKBONE)
        state = torch.load(str(Path(LOCAL_DIR) / CKPT_NAME), map_location="cpu")
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

        self._transform = T.Compose([
            T.Lambda(lambda im: _translate_duplicate(im, IMAGE_SIZE)),
            T.CenterCrop(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])

    def _infer(self, image: Any) -> tuple[float, dict[str, Any]]:
        torch = self._torch
        x = self._transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(x).flatten()[0]
            p_fake = float(torch.sigmoid(logit).item())
        raw = {
            "loader": "c2p_clip.C2P_CLIP (repo inference.py)",
            "logit": round(float(logit.item()), 6),
            "prob_fake": round(p_fake, 6),
            "label_mapping": "P(fake) = sigmoid(single_logit)",
        }
        return p_fake, raw


def build(device: str = "cuda:0", logs_dir: str = "logs") -> C2PClipClient:
    return C2PClipClient(MODEL_ID, HF_REPO, device=device, logs_dir=logs_dir)
