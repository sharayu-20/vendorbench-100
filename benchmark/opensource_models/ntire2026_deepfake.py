"""ntire2026_deepfake — NTIRE 2026 Robust Deepfake Detection (Ant Intl, Tier C).

Winner-grade ensemble from the CVPR 2026 / NTIRE Robust Deepfake Detection
Challenge (HF scarlettss/NTIRE2026_deepfake). Two DINOv3-ViT-7B (`dinov3_vit`,
hidden=4096, 40 layers) detectors, each a full end-to-end fine-tune (backbone
weights baked into the .pt — no gated backbone needed):

  * v135 "ViT-CLS":     DINOv3 -> pooler_output(4096) -> Linear(4096,1)
  * v156 "ViT-Attnpool": DINOv3 -> patch tokens -> AttentionPooling -> Linear(4096,1)

We reproduce the repo's two MFF_MoE variants (infer_step1.py / infer_step2.py)
and its submit.py ensemble:  P(fake) = 0.35*sigmoid(v135) + 0.65*sigmoid(v156).
Preprocessing = the DINOv3 AutoProcessor forced to 384x384. Both experts are
loaded in fp16 (~14 GB each) so the pair fits on one H100.

Weights (~54 GB) live under models/ntire2026_deepfake/; the DINOv3 config +
processor live in vendor/ntire2026/. AutoModel.from_config (not from_pretrained)
builds the architecture, then the .pt checkpoints fill every weight.

Load note (transformers 5.x): the checkpoints (tf 4.56) store the DINOv3
transformer blocks as `experts.layer.N.*`, but tf 5.x nests them under
`experts.model.layer.N.*` (embeddings / final norm stay at `experts.*`). We
remap adaptively against the live model's keys (verified 0 missing / 0
unexpected). Full-set ROC-AUC 0.750 — top of Tier C.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.opensource_models.base_research import BaseResearchClient

MODEL_ID = "ntire2026_deepfake"
HF_REPO = "scarlettss/NTIRE2026_deepfake"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
DINOV3_DIR = str(Path(__file__).resolve().parents[2] / "vendor" / "ntire2026"
                 / "dinov3-vit7b16-pretrain-lvd1689m")
CKPT_V135 = "deepfake_v135_epoch_1.pt"
CKPT_V156 = "deepfake_v156_epoch_1.pt"
W_V135, W_V156 = 0.35, 0.65
IMAGE_SIZE = 384
ENABLED = True


def _remap_experts(state: dict, target: set) -> dict:
    """transformers 5.x nests the DINOv3 transformer blocks one level deeper
    (`experts.layer.N` -> `experts.model.layer.N`) while leaving embeddings /
    final norm at `experts.*`. Remap adaptively against the live model's keys:
    keep a key as-is if it already matches, else try inserting `.model.`."""
    out = {}
    for k, v in state.items():
        if k in target:
            out[k] = v
            continue
        alt = "experts.model." + k[len("experts."):] if k.startswith("experts.") else k
        out[alt if alt in target else k] = v
    return out


def _load_ckpt(model, path, torch):
    target = set(model.state_dict().keys())
    state = _remap_experts(torch.load(path, map_location="cpu"), target)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"ntire load mismatch for {path}: {len(missing)} missing / "
            f"{len(unexpected)} unexpected (e.g. {(missing or unexpected)[:3]})"
        )
    return model


def _build_experts_config():
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(DINOV3_DIR, local_files_only=True)


def _make_v135(torch):
    """infer_step1.py MFF_MoE: pooler_output -> Linear(4096,1)."""
    import torch.nn as nn
    from transformers import AutoModel

    class MFF_MoE_CLS(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = AutoModel.from_config(_build_experts_config())
            self.head = nn.Linear(4096, 1)

        def forward(self, x):
            emb = self.experts(pixel_values=x).pooler_output
            return self.head(emb).squeeze(dim=-1)

    return MFF_MoE_CLS()


def _make_v156(torch):
    """infer_step2.py MFF_MoE: patch tokens -> AttentionPooling -> Linear."""
    import torch.nn as nn
    from transformers import AutoModel

    class AttentionPooling(nn.Module):
        def __init__(self, embed_dim: int) -> None:
            super().__init__()
            self.embed_dim = embed_dim
            self.attention_net = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.Tanh(),
                nn.Linear(embed_dim // 2, 1),
            )

        def forward(self, x):
            attn_weights = torch.softmax(self.attention_net(x), dim=1)
            return torch.sum(x * attn_weights, dim=1)

    class MFF_MoE_Attn(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = AutoModel.from_config(_build_experts_config())
            self.embed_dim = self.experts.config.hidden_size
            self.num_register_tokens = self.experts.config.num_register_tokens
            self.pooling = AttentionPooling(self.embed_dim)
            self.head = nn.Linear(self.embed_dim, 1)

        def forward(self, x):
            out = self.experts(pixel_values=x)
            start = 1 + self.num_register_tokens
            patches = out.last_hidden_state[:, start:, :]
            return self.head(self.pooling(patches)).squeeze(dim=-1)

    return MFF_MoE_Attn()


class Ntire2026Client(BaseResearchClient):
    def _load(self) -> None:
        import torch
        from transformers import AutoProcessor

        self._torch = torch
        self._proc = AutoProcessor.from_pretrained(DINOV3_DIR)
        self._proc.size = {"height": IMAGE_SIZE, "width": IMAGE_SIZE}

        self.m135 = _load_ckpt(_make_v135(torch), str(Path(LOCAL_DIR) / CKPT_V135), torch)
        self.m135.half().to(self.device).eval()

        self.m156 = _load_ckpt(_make_v156(torch), str(Path(LOCAL_DIR) / CKPT_V156), torch)
        self.m156.half().to(self.device).eval()

    def _infer(self, image: Any) -> tuple[float, dict[str, Any]]:
        torch = self._torch
        px = self._proc(images=[image], return_tensors="pt").pixel_values.to(self.device).half()
        with torch.no_grad():
            s135 = torch.sigmoid(self.m135(px)).flatten()[0].item()
            s156 = torch.sigmoid(self.m156(px)).flatten()[0].item()
        p_fake = float(W_V135 * s135 + W_V156 * s156)
        raw = {
            "loader": "ntire2026 MFF_MoE ensemble (DINOv3-ViT7B x2)",
            "score_v135": round(float(s135), 6),
            "score_v156": round(float(s156), 6),
            "weights": {"v135": W_V135, "v156": W_V156},
            "prob_fake": round(p_fake, 6),
            "label_mapping": "P(fake) = 0.35*sigmoid(v135) + 0.65*sigmoid(v156)",
        }
        return p_fake, raw

    def unload(self) -> None:
        self.m135 = None
        self.m156 = None
        super().unload()


def build(device: str = "cuda:0", logs_dir: str = "logs") -> Ntire2026Client:
    return Ntire2026Client(MODEL_ID, HF_REPO, device=device, logs_dir=logs_dir)
