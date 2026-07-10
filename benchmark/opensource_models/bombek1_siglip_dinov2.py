"""bombek1_siglip_dinov2 — Bombek1/ai-image-detector-siglip-dinov2 (Tier B, custom).

SigLIP2-so400m + DINOv2-large ensemble with LoRA adapters, loaded from the repo's
own model.py building blocks (create_model_with_lora + create_transforms). The two
frozen backbones (SigLIP2 via transformers, DINOv2 via timm) download to the HF
cache on first load.

Compat note: the repo's own AIImageDetector does a strict load that breaks on
transformers >=5 — the checkpoint stores SigLIP keys as
`siglip.base_model.model.vision_model.*` but transformers 5.x's SiglipVisionModel
flattened that to `siglip.base_model.model.*`. We remap those keys on load
(verified: 0 missing / 0 unexpected).

Output: P(fake) = sigmoid(logit) = P(AI-generated).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from benchmark.opensource_models.base_research import BaseResearchClient

MODEL_ID = "bombek1_siglip_dinov2"
HF_REPO = "Bombek1/ai-image-detector-siglip-dinov2"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
ENABLED = True

_VISION_PREFIX_OLD = "siglip.base_model.model.vision_model."
_VISION_PREFIX_NEW = "siglip.base_model.model."


def _load_repo_module(local_dir: str):
    path = Path(local_dir) / "model.py"
    spec = importlib.util.spec_from_file_location("bombek1_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Bombek1Client(BaseResearchClient):
    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor

        self._torch = torch
        mod = _load_repo_module(LOCAL_DIR)
        weights = str(Path(LOCAL_DIR) / "pytorch_model.pt")
        ck = torch.load(weights, map_location=self.device, weights_only=False)
        c = ck["config"]

        self.model = mod.create_model_with_lora(
            siglip_model_name=c["siglip_model"],
            dinov2_model_name=c["dinov2_model"],
            image_size=c["image_size"],
            lora_rank=c["lora_rank"],
            lora_alpha=c["lora_alpha"],
            lora_dropout=c["lora_dropout"],
        )
        remapped = {
            k.replace(_VISION_PREFIX_OLD, _VISION_PREFIX_NEW): v
            for k, v in ck["model_state_dict"].items()
        }
        missing, unexpected = self.model.load_state_dict(remapped, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"bombek1 load mismatch: {len(missing)} missing / "
                f"{len(unexpected)} unexpected keys"
            )
        self.model.to(self.device).eval()
        self._proc = AutoProcessor.from_pretrained(c["siglip_model"])
        self._tf = mod.create_transforms(c["image_size"])

    def _infer(self, image: Any) -> tuple[float, dict[str, Any]]:
        torch = self._torch
        siglip_px = self._proc(images=image, return_tensors="pt")["pixel_values"].to(self.device)
        dino_px = self._tf(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(siglip_px, dino_px)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            p_fake = float(torch.sigmoid(logits).flatten()[0].item())
        raw = {
            "loader": "bombek1_model.create_model_with_lora (+vision_model key remap)",
            "prob_fake": round(p_fake, 6),
            "label_mapping": "P(fake) = sigmoid(logit) = P(AI)",
        }
        return p_fake, raw


def build(device: str = "cuda:0", logs_dir: str = "logs") -> Bombek1Client:
    return Bombek1Client(MODEL_ID, HF_REPO, device=device, logs_dir=logs_dir)
