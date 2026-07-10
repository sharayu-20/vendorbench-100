"""gend_dinov3_l — yermandy/GenD_DINOv3_L (Tier B, WACV 2026, custom transformers).

GenD DINOv3-L linear-probe detector. Loaded via the repo's modeling_gend.GenD
(a PreTrainedModel); the DINOv3-L backbone facebook/dinov3-vitl16-pretrain-lvd1689m
is pulled at load time. 2-class head -> softmax; P(fake) = softmax[FAKE_INDEX].

Requires an HF_TOKEN: the DINOv3 backbone repo
facebook/dinov3-vitl16-pretrain-lvd1689m is GATED. Before the first run:
  1. Accept the license at https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m
  2. export HF_TOKEN=<your token>   (or `huggingface-cli login`)
Once the backbone is cached, later runs work offline. FAKE_INDEX=1 verified
correct on the smoke test (P(fake) higher on fakes).

Loading notes (transformers 5.x): we build GenD(config) directly instead of
GenD.from_pretrained (which wraps init in a meta-device context that breaks the
repo's nested AutoModel.from_pretrained), and remap the DINOv3 layer keys
(backbone.layer.N -> backbone.model.layer.N).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from benchmark.opensource_models.base_research import BaseResearchClient

MODEL_ID = "gend_dinov3_l"
HF_REPO = "yermandy/GenD_DINOv3_L"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
FAKE_INDEX = 1
ENABLED = True  # requires HF_TOKEN with access to the gated DINOv3 backbone


def _load_repo_module(local_dir: str):
    path = Path(local_dir) / "modeling_gend.py"
    spec = importlib.util.spec_from_file_location("gend_modeling", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class GendDinov3Client(BaseResearchClient):
    def _load(self) -> None:
        import json
        import torch
        from safetensors.torch import load_file

        self._torch = torch
        mod = _load_repo_module(LOCAL_DIR)
        # Build directly (not GenD.from_pretrained): transformers 5.x wraps
        # from_pretrained in a meta-device context, which breaks GenD's habit of
        # calling AutoModel.from_pretrained(backbone) inside its own __init__.
        cfg = json.loads((Path(LOCAL_DIR) / "config.json").read_text())
        config = mod.GenDConfig(backbone=cfg["backbone"], head=cfg["head"])
        model = mod.GenD(config)  # downloads/loads the DINOv3-L backbone structure
        state = load_file(str(Path(LOCAL_DIR) / "model.safetensors"))
        # transformers 5.x nests DINOv3 blocks under `.model`; the checkpoint
        # (transformers 4.56) stored them flat -> remap the layer keys.
        state = {
            k.replace(
                "feature_extractor.backbone.layer.",
                "feature_extractor.backbone.model.layer.",
            ): v
            for k, v in state.items()
        }
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"gend load mismatch: {len(missing)} missing / "
                f"{len(unexpected)} unexpected (e.g. {(missing or unexpected)[:3]})"
            )
        self.model = model.to(self.device).eval()

    def _infer(self, image: Any) -> tuple[float, dict[str, Any]]:
        torch = self._torch
        tensor = self.model.feature_extractor.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)[0]
            probs = torch.softmax(logits, dim=-1).float().cpu().tolist()
        p_fake = float(probs[FAKE_INDEX])
        raw = {
            "loader": "gend_modeling.GenD",
            "probs": [round(p, 6) for p in probs],
            "fake_index": FAKE_INDEX,
            "prob_fake": round(p_fake, 6),
            "label_mapping": f"P(fake) = softmax[{FAKE_INDEX}]",
        }
        return p_fake, raw


def build(device: str = "cuda:0", logs_dir: str = "logs") -> GendDinov3Client:
    return GendDinov3Client(MODEL_ID, HF_REPO, device=device, logs_dir=logs_dir)
