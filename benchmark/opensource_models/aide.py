"""aide — meet4150/AIDE_image_detector (Tier B, ICLR 2025, custom).

AIDE hybrid: SRM/DCT frequency branch (ResNet encoders) + OpenCLIP ConvNeXt-XXL
semantic branch + MLP. Loaded via the repo's own inference.py (load_model +
build_aide_input_from_pil) with its 5-view DCT+RGB input ([5,3,256,256] per image).
config id2label: 0=real, 1=fake  ->  P(fake) = softmax[1].

The repo ships `inference.py`, `models/` and `data/` packages; we add the local
model folder to sys.path so its relative imports resolve, then reuse its code.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from benchmark.opensource_models.base_research import BaseResearchClient

MODEL_ID = "aide"
HF_REPO = "meet4150/AIDE_image_detector"
LOCAL_DIR = str(Path(__file__).resolve().parents[2] / "models" / MODEL_ID)
FAKE_INDEX = 1
ENABLED = True


def _load_inference_module(local_dir: str):
    # inference.py does `from data.dct import ...` / `from models import AIDE`,
    # both relative to the repo root -> make it importable.
    if local_dir not in sys.path:
        sys.path.insert(0, local_dir)
    path = Path(local_dir) / "inference.py"
    spec = importlib.util.spec_from_file_location("aide_inference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class AideClient(BaseResearchClient):
    def _load(self) -> None:
        import torch
        self._torch = torch
        self._inf = _load_inference_module(LOCAL_DIR)
        self.model = self._inf.load_model(LOCAL_DIR, device=self.device)
        self._dct = self._inf.DCT_base_Rec_Module()

    def _infer(self, image: Any) -> tuple[float, dict[str, Any]]:
        torch = self._torch
        batch = (
            self._inf.build_aide_input_from_pil(image, self._dct)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.inference_mode():
            logits = self.model(batch)
            probs = torch.softmax(logits, dim=-1)[0].float().cpu().tolist()
        p_fake = float(probs[FAKE_INDEX])
        raw = {
            "loader": "aide_inference.AIDE",
            "probs": {"real": round(probs[0], 6), "fake": round(probs[1], 6)},
            "fake_index": FAKE_INDEX,
            "prob_fake": round(p_fake, 6),
            "label_mapping": "P(fake) = softmax[1]  (0=real, 1=fake)",
        }
        return p_fake, raw


def build(device: str = "cuda:0", logs_dir: str = "logs") -> AideClient:
    return AideClient(MODEL_ID, HF_REPO, device=device, logs_dir=logs_dir)
