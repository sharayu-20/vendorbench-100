"""Registry for the open-source track.

The per-model scripts in this package are the source of truth: each exposes
MODEL_ID, HF_REPO, LOCAL_DIR, ENABLED and a build(device, logs_dir) factory.
This module just enumerates them (in run order) and imports on demand so only
one model is instantiated / on the GPU at a time.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

# Ordered list of per-model modules (Tier A + custom commfor).
MODEL_IDS: list[str] = [
    "wvolf_vit",
    "yaya_source",
    "organika_sdxl",
    "ateeqq_siglip2",
    "haywoodsloan_deploy",
    "dima806_ai_vs_real",
    "jacob_distilled",
    "nahrawy_aiornot",
    "king1oo1_deepguard",
    "sadra_sdxl_face",
    "aidfr_real_v2",
    "date3k2_vit",
    "ummmaybe_vit",
    "opensight_commfor",
    "ash_flux_vit",
    "commfor_vit384",
    # Tier B — custom loaders (Phase 2)
    "bombek1_siglip_dinov2",
    "gend_dinov3_l",
    "aide",
    # Tier C — latest, external repos (Phase 3)
    "c2p_clip",
    "rine",
    "drct",
    "declip",
    "ntire2026_deepfake",
]


def get_module(model_id: str) -> ModuleType:
    return importlib.import_module(f"benchmark.opensource_models.{model_id}")


def enabled_ids() -> list[str]:
    return [mid for mid in MODEL_IDS if getattr(get_module(mid), "ENABLED", True)]


def info(model_id: str) -> dict[str, Any]:
    m = get_module(model_id)
    return {
        "id": getattr(m, "MODEL_ID", model_id),
        "hf_repo": getattr(m, "HF_REPO", ""),
        "local_dir": getattr(m, "LOCAL_DIR", ""),
        "fake_labels": getattr(m, "FAKE_LABELS", None),
        "fake_index": getattr(m, "FAKE_INDEX", None),
        "image_size": getattr(m, "IMAGE_SIZE", None),
        "enabled": getattr(m, "ENABLED", True),
    }


def build(model_id: str, device: str = "cuda:0", logs_dir: str = "logs"):
    """Instantiate one model's client (loads weights onto the GPU)."""
    return get_module(model_id).build(device=device, logs_dir=logs_dir)
