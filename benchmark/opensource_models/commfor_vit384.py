"""Custom loader for Community Forensics ViT-384 (OwensLab/commfor-model-384).

Not a standard HF auto-model: a timm `vit_small_patch16_384` backbone with a
single-logit head, saved via PyTorchModelHubMixin. Weights are prefixed `vit.`.
P(fake) = sigmoid(logit).  Ref: arXiv 2411.04125.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.opensource_models.base_research import BaseResearchClient

MODEL_ID = "commfor_vit384"
HF_REPO = "OwensLab/commfor-model-384"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
ENABLED = True


class CommforVit384Client(BaseResearchClient):
    def __init__(self, entry: dict[str, Any], device: str = "cuda:0",
                 logs_dir: str = "logs") -> None:
        self._local = entry.get("model_path")
        super().__init__(entry["id"], entry["hf_repo"], device=device, logs_dir=logs_dir)

    def _load(self) -> None:
        import os
        import timm
        import torch
        from safetensors.torch import load_file

        self._torch = torch
        self.model = timm.create_model(
            "vit_small_patch16_384", pretrained=False, num_classes=1
        )
        local_wt = os.path.join(self._local, "model.safetensors") if self._local else None
        if local_wt and os.path.exists(local_wt):
            weights = local_wt
        else:
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(self.hf_repo, "model.safetensors")
        state = load_file(weights)
        state = {
            (k[len("vit."):] if k.startswith("vit.") else k): v
            for k, v in state.items()
        }
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            # tolerate head/pos naming diffs but surface them
            import logging
            logging.getLogger(__name__).warning(
                "commfor_vit384 load: missing=%s unexpected=%s", missing, unexpected
            )
        self.model.to(self.device).eval()

        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.processor = None
        self._transform = timm.data.create_transform(**cfg)

    def _infer(self, image: Any) -> tuple[float, dict[str, Any]]:
        torch = self._torch
        x = self._transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logit = self.model(x).squeeze()
            p_fake = float(torch.sigmoid(logit).item())
        raw = {
            "loader": "custom_timm_vit_small_384",
            "logit": float(logit.item()),
            "prob_fake": round(p_fake, 6),
            "label_mapping": "P(fake) = sigmoid(single_logit)",
        }
        return p_fake, raw


def build(device: str = "cuda:0", logs_dir: str = "logs") -> CommforVit384Client:
    entry = {"id": MODEL_ID, "hf_repo": HF_REPO, "model_path": LOCAL_DIR}
    return CommforVit384Client(entry, device=device, logs_dir=logs_dir)
