"""Introspect the 16 Tier-A HF models' label schemes.

Downloads ONLY each model's config (+ preprocessor) — no weights — and prints
id2label, label order, and expected image size. Output is used to hand-build the
label -> P(fake) mapping inside each benchmark/opensource_models/<id>.py script.

Run:  .venv-research/bin/python scripts/introspect_models.py
"""

from __future__ import annotations

import json
import traceback

# id -> HF repo  (Tier A, MODELS.md #1-16)
TIER_A = {
    "commfor_vit384": "OwensLab/commfor-model-384",
    "opensight_commfor": "aiwithoutborders-xyz/OpenSight-CommunityForensics-Deepfake-ViT",
    "wvolf_vit": "Wvolf/ViT_Deepfake_Detection",
    "yaya_source": "yaya36095/ai-source-detector",
    "organika_sdxl": "Organika/sdxl-detector",
    "ateeqq_siglip2": "Ateeqq/ai-vs-human-image-detector",
    "haywoodsloan_deploy": "haywoodsloan/ai-image-detector-deploy",
    "dima806_ai_vs_real": "dima806/ai_vs_real_image_detection",
    "jacob_distilled": "jacoballessio/ai-image-detect-distilled",
    "nahrawy_aiornot": "Nahrawy/AIorNot",
    "king1oo1_deepguard": "king1oo1/deepfake-model",
    "sadra_sdxl_face": "SadraCoding/SDXL-Deepfake-Detector",
    "aidfr_real_v2": "prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0",
    "ash_flux_vit": "ash12321/flux-detector-vit",
    "date3k2_vit": "date3k2/vit-real-fake-classification-v3",
    "ummmaybe_vit": "umm-maybe/AI-image-detector",
}


def _image_size(repo: str) -> object:
    """Best-effort read of expected input size from the preprocessor config."""
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(repo, "preprocessor_config.json")
        with open(path, "r", encoding="utf-8") as f:
            pp = json.load(f)
        return pp.get("size") or pp.get("crop_size") or "?"
    except Exception:
        return "?"


def main() -> None:
    from transformers import AutoConfig

    summary: dict[str, dict] = {}
    for model_id, repo in TIER_A.items():
        print("=" * 78)
        print(f"{model_id}   <-  {repo}")
        try:
            cfg = AutoConfig.from_pretrained(repo, trust_remote_code=True)
            id2label = getattr(cfg, "id2label", None) or {}
            arch = (getattr(cfg, "architectures", None) or ["?"])[0]
            size = _image_size(repo)
            print(f"  arch        : {arch}")
            print(f"  num_labels  : {len(id2label)}")
            print(f"  id2label    : {id2label}")
            print(f"  image_size  : {size}")
            summary[model_id] = {
                "repo": repo,
                "arch": arch,
                "id2label": {str(k): v for k, v in id2label.items()},
                "image_size": size,
            }
        except Exception as exc:
            print(f"  [ERROR] {type(exc).__name__}: {exc}")
            traceback.print_exc()
            summary[model_id] = {"repo": repo, "error": f"{type(exc).__name__}: {exc}"}

    out = "scripts/introspect_result.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n" + "=" * 78)
    print(f"JSON SUMMARY written to {out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
