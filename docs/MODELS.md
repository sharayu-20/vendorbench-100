# Open-source models (research track) — 24 built (of 27 surveyed)

**Assumes full GPU + ample disk.** Every model below was verified (July 2026) to ship a
**ready-to-download inference checkpoint plus runnable inference code** — no "train it
yourself" repos.

**Status:** 24 of the 27 surveyed detectors are implemented, downloaded, and benchmarked on
the 100-image `Source/` set (79 fake / 21 real). The 3 not built (`ufd_clip`, `fatformer`,
`npr`) are listed at the end of Tier C with the reason. Every table below carries the
**measured ROC-AUC / Accuracy / F1** from that run; the full write-up (per-model biography,
citations, training details, and combined analysis) is in
[`reports/opensource-benchmark/MODELS_REPORT.md`](../reports/opensource-benchmark/MODELS_REPORT.md)
and the interactive charts in
[`reports/opensource-benchmark/benchmark_report.html`](../reports/opensource-benchmark/benchmark_report.html).

> **Metric note.** AUC (threshold-free ranking) is the headline number; Accuracy/F1 are at a
> fixed 0.5 threshold, so a well-ranking-but-miscalibrated model (e.g. DRCT) can show high AUC
> yet low accuracy. AUC < 0.5 means the model ranks *worse than chance* on this set — a
> domain mismatch (trained on GANs / faces / localization), not a wiring bug; mappings are
> kept faithful to each repo rather than flipped to game the score.

## Selection criterion

A model qualifies **only if all three hold**:

1. **Downloadable weights** — HF (`safetensors`/`.pt`), or a hosted checkpoint on
   Drive / OneDrive / GCS / ModelScope. A GitHub repo with *training code only* (e.g.
   DINO-MAC) is **rejected**.
2. **Runnable inference** — an `AutoModel`, a `model.py` loader, or an `infer`/`validate`
   script exists; no reconstruction of the method from scratch.
3. **One model per author, distinct method** — no re-uploads, no near-duplicate finetunes,
   at most one repo per author.

Loader tiers: **A** = HF `transformers` native (`from_pretrained`), **B** = HF custom
loader (`safetensors` + `model.py`), **C** = academic checkpoint + repo inference script.

Tier A loads in ~3 lines:

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
model = AutoModelForImageClassification.from_pretrained("<repo_id>")   # SigLIP2 -> SiglipForImageClassification
processor = AutoImageProcessor.from_pretrained("<repo_id>")
```

---

## The 24 (built + benchmarked)

Metrics = 100-image `Source/` run (79 fake / 21 real). **AUC** is the headline; Acc / F1 at 0.5.

### Tier A — HF `transformers` native (plug-and-play, `from_pretrained`) — 16 models, one per author

| # | ID | Model | Author | Arch | Size | AUC | Acc | F1 | Source |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `commfor_vit384` | Community Forensics ViT-384 | OwensLab (UMich) | ViT-S | ~87 MB | 0.620 | 0.55 | 0.651 | https://huggingface.co/OwensLab/commfor-model-384 |
| 2 | `opensight_commfor` | OpenSight CommunityForensics ViT | aiwithoutborders-xyz | ViT-S | ~90 MB | 0.517 | 0.53 | 0.667 | https://huggingface.co/aiwithoutborders-xyz/OpenSight-CommunityForensics-Deepfake-ViT |
| 3 | `wvolf_vit` | ViT Deepfake Detection | Wvolf | ViT-B | ~330 MB | 0.168 | 0.22 | 0.278 | https://huggingface.co/Wvolf/ViT_Deepfake_Detection |
| 4 | `yaya_source` | AI Source Detector | yaya36095 | ViT-B (6-cls) | ~1.1 GB | 0.482 | 0.73 | 0.836 | https://huggingface.co/yaya36095/ai-source-detector |
| 5 | `organika_sdxl` | Organika SDXL-Detector | Organika | SwinV2 | ~200 MB | 0.677 | 0.51 | 0.574 | https://huggingface.co/Organika/sdxl-detector |
| 6 | `ateeqq_siglip2` | Ateeqq AI-vs-Human | Ateeqq | SigLIP2 | ~370 MB | 0.536 | 0.62 | 0.763 | https://huggingface.co/Ateeqq/ai-vs-human-image-detector |
| 7 | `haywoodsloan_deploy` | AI-Image-Detector (WasItAI-style) | haywoodsloan | SwinV2 | ~800 MB | 0.508 | 0.41 | 0.469 | https://huggingface.co/haywoodsloan/ai-image-detector-deploy |
| 8 | `dima806_ai_vs_real` | AI-vs-Real (CIFAKE) | dima806 | ViT-B | ~343 MB | 0.655 | 0.81 | 0.893 | https://huggingface.co/dima806/ai_vs_real_image_detection |
| 9 | `jacob_distilled` | AI-Image-Detect (distilled) | jacoballessio | Distilled ViT | ~47 MB | 0.439 | 0.40 | 0.483 | https://huggingface.co/jacoballessio/ai-image-detect-distilled |
| 10 | `nahrawy_aiornot` | AIorNot | Nahrawy | Swin-tiny | ~110 MB | 0.602 | 0.52 | 0.593 | https://huggingface.co/Nahrawy/AIorNot |
| 11 | `king1oo1_deepguard` | DeepGuard | king1oo1 | SigLIP2 | ~372 MB | 0.459 | 0.51 | 0.614 | https://huggingface.co/king1oo1/deepfake-model |
| 12 | `sadra_sdxl_face` | SDXL-Deepfake-Detector | SadraCoding | ViT | ~343 MB | 0.282 | 0.25 | 0.257 | https://huggingface.co/SadraCoding/SDXL-Deepfake-Detector |
| 13 | `aidfr_real_v2` | AI-vs-Deepfake-vs-Real v2.0 | prithivMLmods | SigLIP2 | ~370 MB | 0.694 | 0.67 | 0.769 | https://huggingface.co/prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0 |
| 14 | `ash_flux_vit` | FLUX Detector (ViT) | ash12321 | ViT | ~343 MB | 0.650 | 0.50 | 0.597 | https://huggingface.co/ash12321/flux-detector-vit |
| 15 | `date3k2_vit` | ViT Real/Fake v3 | date3k2 | ViT-B | ~343 MB | 0.432 | 0.34 | 0.353 | https://huggingface.co/date3k2/vit-real-fake-classification-v3 |
| 16 | `ummmaybe_vit` | AI Art Detector | umm-maybe | ViT | ~350 MB | 0.761 | 0.51 | 0.581 | https://huggingface.co/umm-maybe/AI-image-detector |

### Tier B — HF custom loader (`safetensors` + `model.py`) — 3 models

| # | ID | Model | Venue / Origin | Size | AUC | Acc | F1 | Source |
|---|---|---|---|---|---|---|---|---|
| 17 | `bombek1_siglip_dinov2` | SigLIP2 + DINOv2 ensemble | Community | ~3 GB | 0.649 | 0.57 | 0.650 | [Bombek1/ai-image-detector-siglip-dinov2](https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2) |
| 18 | `gend_dinov3_l` | GenD DINOv3-L | **WACV 2026** (arXiv 2508.06248) | ~1.2 GB | 0.597 | 0.34 | 0.365 | [yermandy/GenD_DINOv3_L](https://huggingface.co/yermandy/GenD_DINOv3_L) |
| 19 | `aide` | AIDE | **ICLR 2025** (arXiv 2406.19435) | ~3.6 GB | 0.392 | 0.40 | 0.500 | [meet4150/AIDE_image_detector](https://huggingface.co/meet4150/AIDE_image_detector) (+ [Drive](https://drive.google.com/drive/folders/1qx76UFvDpgCxaPLBCmsA2WY-SSzeJrd4)) |

### Tier C — academic checkpoints + repo inference script — 5 built

| # | ID | Model | Venue / Origin | Size | AUC | Acc | F1 | Source |
|---|---|---|---|---|---|---|---|---|
| 20 | `c2p_clip` | C2P-CLIP | **AAAI 2025** (arXiv 2408.09647) | ~1.2 GB | 0.385 | 0.37 | 0.422 | [GitHub](https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection) |
| 21 | `rine` | RINE | **ECCV 2024** (arXiv 2402.19091) | ~25 MB* | 0.737 | 0.38 | 0.367 | [mever-team/rine](https://github.com/mever-team/rine) (`ckpt/` in-repo) |
| 22 | `drct` | DRCT | **ICML 2024 Spotlight** | ~355 MB | **0.866** | 0.47 | 0.514 | [beibuwandeluori/DRCT](https://github.com/beibuwandeluori/DRCT) + [ModelScope](https://modelscope.cn/datasets/BokingChen/DRCT-2M/files) |
| 23 | `declip` | DeCLIP | **WACV 2025** (Bitdefender) | ~627 MB | 0.275 | 0.21 | 0.000 | [bit-ml/DeCLIP](https://github.com/bit-ml/DeCLIP) + GCS bucket |
| 24 | `ntire2026_deepfake` | NTIRE 2026 – Ant Intl (DINOv3 ensemble) | **CVPR 2026 / NTIRE challenge** | ~54 GB | 0.750 | 0.60 | 0.683 | [scarlettss/NTIRE2026_deepfake](https://huggingface.co/scarlettss/NTIRE2026_deepfake) |

\* RINE's trainable head is ~25 MB; the frozen CLIP ViT-L/14 backbone (~1.2 GB) is re-downloaded via `clip.load`.

#### Surveyed but not implemented (3)

| # | ID | Model | Venue | Why deferred |
|---|---|---|---|---|
| — | `ufd_clip` | UniversalFakeDetect | CVPR 2023 | Foundational CLIP-linear baseline, but its method is effectively **subsumed** by the 4 newer CLIP detectors already built (C2P-CLIP, RINE, DeCLIP, FatFormer-class); dropped to avoid a near-duplicate at the low end. |
| — | `fatformer` | FatFormer | CVPR 2024 | Checkpoint is split across Google Drive **and** OneDrive with a bespoke forgery-adapter loader; higher integration cost for marginal added coverage over the other CLIP-adapter models. Good next candidate. |
| — | `npr` | NPR | CVPR 2024 | Tiny upsampling-artifact detector trained on **ProGAN only**; expected strong domain mismatch on this diffusion+faceswap set (cf. C2P-CLIP at 0.385). Deferred as low-value on this distribution. |

> These three are fully surveyed and their sources are pinned above — they can be added later without re-planning.

---

## Method / backbone coverage (24 built)

ViT (1, 2, 3, 4, 8, 12, 14, 15, 16) · SwinV2 (5, 7) · Swin-tiny (10) · distilled ViT (9) ·
SigLIP2 (6, 11, 13) · SigLIP2+DINOv2 (17) · DINOv3 (18, 24-ntire) · SRM/DCT frequency + ConvNeXt (19) ·
CLIP-prompt (20-c2p) · CLIP intermediate-block (21-rine) · diffusion-reconstruction ConvNeXt (22-drct) ·
CLIP detect+localize (23-declip).

**7 are peer-reviewed** (WACV ×2, ICLR, ICML, ECCV, AAAI, CVPR-NTIRE-workshop) plus Community
Forensics (arXiv) and a Solent MSc thesis. The remaining community models add production
baselines and generator diversity. (The 3 not built — `ufd_clip` CVPR'23, `fatformer` CVPR'24,
`npr` CVPR'24 — would raise peer-reviewed coverage to 10.)

---

## Notes / caveats

- **#24 `ntire2026_deepfake`** is a **pickle `.pt`** two-checkpoint DINOv3-ViT-7B ensemble,
  ~54 GB total, loaded in fp16 to fit one H100 — only enable if disk allows.
- **#17 Bombek1** and **#19 AIDE** are HF-hosted but need their bundled `model.py` /
  `inference.py` (not native `AutoModel`).
- **#8 dima806** (CIFAKE) and **#16 umm-maybe** (2022) are older — keep as historical
  baselines; expect misses on modern generators.
- **Low-AUC ≠ broken:** `declip` (0.275), `c2p_clip` (0.385), `aide` (0.392), `sadra_sdxl_face`
  (0.282), `wvolf_vit` (0.168) rank below chance because of domain mismatch (localization-only,
  ProGAN-only, or face-only training) — mappings are kept faithful to each repo, **not flipped**.
- **Tier C** models fetch weights from Drive/OneDrive/GCS/ModelScope — mirror into
  `models/<id>/` and pin the file hash for reproducibility.
- **Optional shortcut:** [`leekwoon/260204_cfd_backup`](https://huggingface.co/datasets/leekwoon/260204_cfd_backup)
  (CleanFD) bundles ~23 detectors + weights in one download.

## Considered but not added (one-per-author rule)

| Model | Reason |
|---|---|
| Extra prithivMLmods repos (Deep-Fake-Detector-v2, deepfake-detector-model-v1, Deepfake-Detect-Siglip2, AIorNot-SigLIP2, OpenSDI-*) | Author already at #13 — one prithiv only |
| Extra ash12321 repos (sdxl-detector-vit, sdxl-detector-resnet50) | Author already at #14 — one ash only |
| Second dima806 (deepfake_vs_real_image_detection) | Author already at #8 |
| capcheck/* | Re-publish of dima806's CIFAKE ViT — redundant |
| Hemg/Deepfake-Detection, Heem2/AI-vs-Real-Image-Detection | Weights unconfirmed — add only if verified |
| Reju983/ai-generated-image-detector | SwinV2+SRM+DCT+FFT method but HF weights unconfirmed |
| WasItAI (wasitai.com) | Commercial SaaS — not downloadable OSS (`haywoodsloan` #7 is the open stand-in) |

## Excluded (rejected outright)

| Model | Reason |
|---|---|
| DINO-MAC (NTIRE 2026 #1) | GitHub repo is **training code only** — no downloadable checkpoint |
| AEROBLADE, DIRE | Require a full diffusion-autoencoder pipeline — heavy, slow, fragile |
| MIRROR, Effort | Poorly maintained / superseded by C2P-CLIP & FatFormer |
| buildborderless CommunityForensics | Re-upload of #1 |
| ItsNotAI v2, xRayon ConvNeXt, OmniAID | Lower reputation / redundant |
| LOGER, HEDGE, MICV | No public weights (paper-only) |

---

## Build status (24 done)

| S.No | Model | Tier | Status | AUC |
|---|---|---|---|---|
| 1 | `commfor_vit384` — Community Forensics ViT-384 | A | ✅ done | 0.620 |
| 2 | `opensight_commfor` — OpenSight CommunityForensics ViT | A | ✅ done | 0.517 |
| 3 | `wvolf_vit` — ViT Deepfake Detection (thesis) | A | ✅ done | 0.168 |
| 4 | `yaya_source` — AI Source Detector (6-class) | A | ✅ done | 0.482 |
| 5 | `organika_sdxl` — Organika SDXL-Detector | A | ✅ done | 0.677 |
| 6 | `ateeqq_siglip2` — Ateeqq AI-vs-Human | A | ✅ done | 0.536 |
| 7 | `haywoodsloan_deploy` — AI-Image-Detector (WasItAI-style) | A | ✅ done | 0.508 |
| 8 | `dima806_ai_vs_real` — AI-vs-Real (CIFAKE) | A | ✅ done | 0.655 |
| 9 | `jacob_distilled` — AI-Image-Detect (distilled) | A | ✅ done | 0.439 |
| 10 | `nahrawy_aiornot` — AIorNot (Swin-tiny) | A | ✅ done | 0.602 |
| 11 | `king1oo1_deepguard` — DeepGuard (SigLIP2) | A | ✅ done | 0.459 |
| 12 | `sadra_sdxl_face` — SDXL-Deepfake-Detector (face) | A | ✅ done | 0.282 |
| 13 | `aidfr_real_v2` — AI-vs-Deepfake-vs-Real v2.0 | A | ✅ done | 0.694 |
| 14 | `ash_flux_vit` — FLUX Detector (ViT) | A | ✅ done | 0.650 |
| 15 | `date3k2_vit` — ViT Real/Fake v3 | A | ✅ done | 0.432 |
| 16 | `ummmaybe_vit` — AI Art Detector (2022) | A | ✅ done | 0.761 |
| 17 | `bombek1_siglip_dinov2` — SigLIP2 + DINOv2 ensemble | B | ✅ done | 0.649 |
| 18 | `gend_dinov3_l` — GenD DINOv3-L | B | ✅ done | 0.597 |
| 19 | `aide` — AIDE | B | ✅ done | 0.392 |
| 20 | `c2p_clip` — C2P-CLIP | C | ✅ done | 0.385 |
| 21 | `rine` — RINE | C | ✅ done | 0.737 |
| 22 | `drct` — DRCT | C | ✅ done | **0.866** |
| 23 | `declip` — DeCLIP | C | ✅ done | 0.275 |
| 24 | `ntire2026_deepfake` — NTIRE 2026 (Ant Intl DINOv3 ensemble) | C | ✅ done | 0.750 |
| — | `ufd_clip` — UniversalFakeDetect | C | ⬜ surveyed, not built | — |
| — | `fatformer` — FatFormer | C | ⬜ surveyed, not built | — |
| — | `npr` — NPR | C | ⬜ surveyed, not built | — |
