"""Generate one Markdown report per open-source model (+ regenerable, read-only).

Reads the stored per-image logs (logs/opensource_models/<id>/*.json), recomputes
metrics with the shared pipeline, and merges them with an embedded narrative
(biography / citations / architecture + training / HF trend / result). Emits:

    reports/opensource-benchmark/models/<id>.md      (one file per model)

Narrative prose mirrors reports/opensource-benchmark/MODELS_REPORT.md so the
per-model files and the combined report stay in sync. No GPU / no re-inference.

Run:  .venv-research/bin/python scripts/report_per_model.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from benchmark.models import APIResult, BenchmarkRun  # noqa: E402
from benchmark.metrics import compute_all_metrics  # noqa: E402

LOGS_DIR = ROOT / "logs" / "opensource_models"
OUT_DIR = ROOT / "reports" / "opensource-benchmark" / "models"

_KEEP = {
    "provider", "image_path", "image_filename", "ground_truth", "prediction",
    "confidence", "raw_response", "latency_ms", "timestamp", "image_sha256",
    "success", "error_message", "retry_count",
}

# ---------------------------------------------------------------------------
# Narrative (mirrors MODELS_REPORT.md §3). links = list of (label, url).
# ---------------------------------------------------------------------------
DOCS: dict[str, dict] = {
    "commfor_vit384": {
        "name": "Community Forensics ViT-384", "tier": "A", "arch": "ViT-S/16 @384", "size": "~87 MB",
        "one": "UMich generator-diversity detector — millions of images from 4,800+ generators.",
        "links": [("HF", "https://huggingface.co/OwensLab/commfor-model-384"),
                  ("Paper: Community Forensics (arXiv 2411.04125)", "https://arxiv.org/abs/2411.04125"),
                  ("Dataset", "https://huggingface.co/datasets/OwensLab/CommunityForensics")],
        "bio": "The official checkpoint from *Community Forensics: Using Thousands of Generators to Train "
               "Fake Image Detectors* (Jeongsoo Park & Andrew Owens, University of Michigan; arXiv 2411.04125). "
               "Its central argument is that a detector's ability to **generalize** comes primarily from the "
               "*diversity of generators* it sees in training — not from a bespoke architecture — so the authors "
               "assembled the largest generator-diverse corpus to date and trained a plain ViT on it. The "
               "weights are published via `PyTorchModelHubMixin`.",
        "how": "A deliberately vanilla **ViT-Small/16 at 384px** with a single binary real/fake head — no SRM "
               "filters, no DCT branch, no CLIP trunk. The forward pass is an ordinary image-classification "
               "softmax; the design bet is that breadth of training data, rather than an architectural inductive "
               "bias, is what makes a detector robust. We load it through the repo's mixin and read `logits[1]` "
               "as P(fake).",
        "train": "Trained on the **Community Forensics** dataset — ~**2.7M images spanning 4,803 generators** "
                 "(15+ model families, >1.15 TB on disk) — with AdamW (lr 5e-5), bf16 mixed precision, batch 32, "
                 "10 epochs. The authors report (and self-label as *unverified*, for lack of eval compute over a "
                 "1.4 TB test set) roughly **0.992 AUC / 97.2% accuracy** in-domain. License MIT.",
        "trend": "Newer (Aug 2025) and low-traffic on the Hub (**698 downloads / 1 like**). It was chosen for "
                 "**method importance rather than popularity**: it is the reference implementation of the "
                 "generator-diversity thesis and the most methodologically interesting Tier-A entry.",
        "result": "**ROC-AUC 0.620 (#10 / 24).** An honest mid-pack ranking — the broad generator training does "
                  "transfer to our diffusion fakes, but the small ViT-S backbone caps the ceiling. At the 0.5 "
                  "threshold it keeps precision high (0.84) while recall is only 0.53, i.e. it misses the subtler "
                  "face-swap fakes rather than raising false alarms on reals.",
        "limits": "ViT-S is a small backbone with a limited ceiling; there is no explicit face-swap modelling; "
                  "and the headline numbers are the authors' own *unverified* figures. The weights are effectively "
                  "a sibling of `opensight_commfor` (#2), which repackages the same model — expect correlated "
                  "behaviour between the two.",
    },
    "opensight_commfor": {
        "name": "OpenSight CommunityForensics ViT", "tier": "A", "arch": "timm ViT-S/16 @384", "size": "~90 MB",
        "one": "Community-Forensics method repackaged for drop-in deployment under the OpenSight framework.",
        "links": [("HF", "https://huggingface.co/aiwithoutborders-xyz/OpenSight-CommunityForensics-Deepfake-ViT"),
                  ("Base: timm ViT-S/16-384", "https://huggingface.co/timm/vit_small_patch16_384.augreg_in21k_ft_in1k")],
        "bio": "A Hugging-Face-compatibility **repackage of the *same* Community-Forensics ViT-S/16-384** "
               "(Park & Owens; 2.7M images / 4,803 generators), uploaded by *AI Without Borders* for community "
               "validation under their upcoming \"OpenSight\" adaptive-detection framework. It is not a new "
               "model — it is essentially #1's weights wrapped for one-line `transformers` / `AutoModel` "
               "inference.",
        "how": "Identical **timm ViT-S/16 @384** (augreg in21k->in1k pretrain) backbone + binary head. The only "
               "delta from `commfor_vit384` is the HF inference wrapper and its default preprocessing, not the "
               "trained weights.",
        "train": "Same **Community Forensics** corpus (2.7M images, 4,803 generators, >1.15 TB); AdamW 5e-5, "
                 "bf16, 10 epochs, batch 32. Card reports (unverified) **0.992 AUC / 97.2% accuracy / 2.1% false "
                 "positive rate** in-domain. License MIT.",
        "trend": "**771,110 downloads / 15 likes** — among the single most-downloaded deepfake detectors on the "
                 "Hub. That reach is precisely why it's included: it is a de-facto production dependency for many "
                 "downstream tools, so its real-world behaviour matters.",
        "result": "**ROC-AUC 0.517 (#14 / 24)** — essentially chance, and notably **below its sibling "
                  "`commfor_vit384` (0.620)** even though the trained weights are the same. That gap is a clean "
                  "demonstration that preprocessing/packaging alone can shift results materially on this "
                  "distribution, and a reminder that download count is not a proxy for accuracy.",
        "limits": "A near-duplicate of #1 (kept only because it is a distinct, extremely popular artifact); its "
                  "packaging under-performs the official release here. Extreme popularity should not be read as "
                  "an accuracy signal.",
    },
    "wvolf_vit": {
        "name": "ViT Deepfake Detection", "tier": "A", "arch": "ViT-B", "size": "~330 MB",
        "one": "Solent MSc-thesis ViT (~98.7% on its own split), widely forked into HF Spaces.",
        "links": [("HF", "https://huggingface.co/Wvolf/ViT_Deepfake_Detection")],
        "bio": "A ViT trained by **Rudolf Enyimba** in partial fulfilment of an **MSc in Artificial "
               "Intelligence & Data Science at Solent University**. The card frames it explicitly as a **face** "
               "deepfake detector (\"upload a face image\") and reports **98.70%** accuracy on its own held-out "
               "test set. There is no accompanying paper; the model is widely forked into demo Spaces.",
        "how": "A **ViT-Base** binary classifier applied to face imagery — a straightforward supervised "
               "fine-tune with no forensic, frequency, or foundation-model components. `logits[1]` = P(fake).",
        "train": "Fine-tuned on a **face-centric real/fake** dataset (not specified on the card); reported "
                 "98.7% in-domain test accuracy.",
        "trend": "**2,684 downloads / 14 likes** with heavy Spaces reuse — a popular community demo. Included to "
                 "represent the large, real-world class of \"face-deepfake ViT\" models that practitioners "
                 "actually deploy.",
        "result": "**ROC-AUC 0.168 (#24 / 24, last).** An AUC this far below 0.5 means the model **ranks our set "
                  "inversely**: the cues it learned on aligned face-deepfakes anti-correlate with the artefacts in "
                  "our generation-heavy fakes (DALL·E / SDXL / video-model frames). On faces it may perform as "
                  "advertised, but on this distribution it is actively misleading. The mapping is kept faithful "
                  "(not flipped to game the metric).",
        "limits": "Face-only training, undocumented dataset, and no method novelty. It is the single worst "
                  "generaliser in the pool and the clearest \"trained on the wrong distribution\" cautionary case "
                  "— useful precisely because it shows how badly a strong in-domain face detector can transfer.",
    },
    "yaya_source": {
        "name": "AI Source Detector", "tier": "A", "arch": "ViT-B (6-class)", "size": "~1.1 GB",
        "one": "Source-attribution model (SD/MJ/DALL·E/real/other) repurposed as a binary detector.",
        "links": [("HF", "https://huggingface.co/yaya36095/ai-source-detector")],
        "bio": "yaya36095's **source-attribution** model: rather than a binary real/fake head it predicts "
               "*which generator* produced an image (the card lists 5 classes — Stable Diffusion / Midjourney / "
               "DALL·E / real / other-AI). We repurpose it as a detector via **P(fake) = 1 − P(real)**. It is "
               "included to test whether an *attribution* model doubles as a *detector*.",
        "how": "ViT-Base/16-224 (86M params). The shipped `model.safetensors` turned out to be a **corrupt "
               "RoBERTa checkpoint**, while the correct weights live in `pytorch_model.bin` with a **6-class** "
               "head (one more than the card documents). We therefore force `use_safetensors=False` + "
               "`num_labels=6` and compute P(fake) as `1 − softmax[real_index]`.",
        "train": "~**52k images** across Stable Diffusion (12k), Midjourney (10.5k), DALL·E-3 (9.4k), real "
                 "(11.8k) and other-AI (8.2k), 80/10/10 split; AdamW (lr 3e-5, wd 0.01), 10 epochs, single "
                 "RTX 4090. Reported **91.6% test top-1 / 0.914 macro-F1** — but as a 5-way *attribution* task, "
                 "not binary detection. License Apache-2.0.",
        "trend": "Very low traffic (**19 downloads / 2 likes**). Chosen for **task diversity** — it is the only "
                 "generator-attribution model in the pool — not for popularity.",
        "result": "**ROC-AUC 0.482 (#16 / 24)** — non-discriminative as a binary detector. Its 0.73 accuracy is "
                  "an artefact of the 79% fake prior (it labels most things fake), not genuine skill. The takeaway "
                  "is that a good source-attribution head does **not** automatically yield a good real/fake "
                  "ranker.",
        "limits": "Optimised for a 5-way attribution objective; the corrupt safetensors required a manual `.bin` "
                  "load; the training set (~52k) is small and generator-balanced rather than detection-balanced.",
    },
    "organika_sdxl": {
        "name": "Organika SDXL-Detector", "tier": "A", "arch": "SwinV2", "size": "~200 MB",
        "one": "The community's de-facto SDXL baseline — the first model most people reach for.",
        "links": [("HF", "https://huggingface.co/Organika/sdxl-detector"),
                  ("Train data", "https://huggingface.co/datasets/Colby/autotrain-data-sdxl-detection")],
        "bio": "Organika's SDXL detector — the community's **de-facto first-choice** for screening "
               "Stable-Diffusion-XL imagery. Built by **fine-tuning the umm-maybe AI-art detector (#16) on "
               "Wikimedia↔SDXL image pairs**, it is the direct descendant of the oldest model in this pool and "
               "the parent of `sadra_sdxl_face` (#12) — a small family tree of fine-tunes.",
        "how": "A **SwinV2** hierarchical-attention backbone with a binary head, produced via HF AutoTrain. "
               "`logits[1]` = P(fake).",
        "train": "Trained on **Wikimedia real images paired with SDXL renders**, where each SDXL image is "
                 "generated from a BLIP caption of the matching Wikimedia photo — so real and fake differ only in "
                 "origin, not content. Reported validation **AUC 0.998 / accuracy 98.1% / F1 0.973**. The card "
                 "explicitly warns performance drops on non-SDXL and older generators (e.g. VQGAN+CLIP). License "
                 "CC-BY-NC-3.0 (non-commercial).",
        "trend": "**152,685 downloads / 57 likes** — a top-tier popularity signal and effectively a mandatory "
                 "baseline for any SDXL-era comparison.",
        "result": "**ROC-AUC 0.677 (#6 / 24)** — the strongest of the single-generator baselines, because its "
                  "SDXL training overlaps our diffusion-heavy fakes. Precision 0.917 with recall 0.42 at 0.5: "
                  "confident when it fires, but miscalibrated toward misses. Weakest on face-swaps and non-SDXL "
                  "video-model frames (VEO/SORA).",
        "limits": "SDXL-centric — degrades on other generators by the author's own admission; CC-BY-NC blocks "
                  "commercial use; miscalibrated at the default threshold.",
    },
    "ateeqq_siglip2": {
        "name": "Ateeqq AI-vs-Human", "tier": "A", "arch": "SigLIP2", "size": "~370 MB",
        "one": "Popular SigLIP2 binary AI-vs-human classifier (~120k-image train).",
        "links": [("HF", "https://huggingface.co/Ateeqq/ai-vs-human-image-detector")],
        "bio": "Ateeqq's binary **AI-vs-human** image classifier built on Google's SigLIP2 vision-language "
               "encoder — one of the more polished community SigLIP2 detectors, with a public write-up of its "
               "fine-tuning recipe.",
        "how": "**SigLIP2** backbone via `SiglipForImageClassification` with a 2-class head (`ai` / `hum`); "
               "P(fake) = softmax(`ai`).",
        "train": "**60,000 AI + 60,000 human** images (120k total), 5 epochs. Reported **test accuracy 0.9923 / "
                 "macro-F1 0.9923** in-domain — though the card itself notes users have reported **overfitting**. "
                 "License Apache-2.0.",
        "trend": "**13,318 downloads / 60 likes** — one of the better-liked modern SigLIP2 detectors, chosen as "
                 "the popular SigLIP2 binary representative.",
        "result": "**ROC-AUC 0.536 (#13 / 24)** with a **negative MCC (−0.187)**: it correctly flags 61/79 fakes "
                  "but only 1/21 reals — it over-predicts \"ai\" almost indiscriminately, so its ranking barely "
                  "clears chance. A concrete example of the overfitting the card warns about: strong in-domain "
                  "accuracy, poor real-image specificity out-of-domain.",
        "limits": "Documented overfitting; very low real-image specificity on this set; training data described "
                  "only at a high level (120k AI/human).",
    },
    "haywoodsloan_deploy": {
        "name": "AI-Image-Detector (WasItAI-style)", "tier": "A", "arch": "SwinV2", "size": "~800 MB",
        "one": "Open stand-in for the WasItAI service (reports F1 0.988 on its own data).",
        "links": [("HF", "https://huggingface.co/haywoodsloan/ai-image-detector-deploy")],
        "bio": "haywoodsloan's deployment-oriented detector — the open, downloadable stand-in for the "
               "**WasItAI** web service (which is closed SaaS). Popular in production tooling with ~19 dependent "
               "Spaces.",
        "how": "A **SwinV2** binary classifier packaged (via AutoTrain) for straightforward deployment; "
               "`logits[1]` = P(fake).",
        "train": "AutoTrain run over an (undisclosed) real/fake corpus; reported in-domain **AUC 0.995 / F1 "
                 "0.988 / accuracy 98.1% / recall 0.993**.",
        "trend": "**20,761 downloads / 21 likes** — a high-traffic production baseline and the closest open "
                 "proxy for a widely-used commercial detector.",
        "result": "**ROC-AUC 0.508 (#15 / 24)** — chance-level on this set. Near-perfect in-domain numbers do "
                  "not transfer to our mixed diffusion + face-swap distribution; another \"popular but "
                  "distribution-limited\" data point alongside `opensight_commfor`.",
        "limits": "Undisclosed training data; strong in-domain metrics that collapse out-of-domain; heaviest "
                  "Tier-A latency (~31 ms).",
    },
    "dima806_ai_vs_real": {
        "name": "AI-vs-Real (CIFAKE)", "tier": "A", "arch": "ViT-B", "size": "~343 MB",
        "one": "The classic CIFAKE ViT baseline (~2yr old; concept drift).",
        "links": [("HF", "https://huggingface.co/dima806/ai_vs_real_image_detection"),
                  ("Base: ViT-B/16-in21k", "https://huggingface.co/google/vit-base-patch16-224-in21k")],
        "bio": "dima806's **CIFAKE** classifier — one of the most-cited 2023 reference detectors (the author "
               "publishes dozens of HF classifiers). The model card is unusually candid: it explicitly warns of "
               "**concept drift** and recommends retraining or lowering the threshold before production use.",
        "how": "**ViT-Base/16** (in21k pretrain) fine-tuned with a binary head; `logits[FAKE]` = P(fake).",
        "train": "Trained on **CIFAKE** — 60k real (CIFAR-10) vs 60k Stable-Diffusion-generated images, all at "
                 "**32×32 px**. Reported **98.25% accuracy / 0.982 F1** on a 48k test split. License Apache-2.0. "
                 "Roughly two years old.",
        "trend": "**3,758 downloads / 20 likes** — included as the canonical CIFAKE historical baseline that "
                 "many later works benchmark against.",
        "result": "**ROC-AUC 0.655 (#7 / 24)**, but note **recall 1.00 with only 2/21 true negatives** — it "
                  "labels almost everything fake. Its 0.81 accuracy is therefore the 79% class prior, not skill; "
                  "the AUC (which still shows real ranking signal despite 32-px training) is the trustworthy "
                  "figure. Exactly the behaviour the author's drift warning predicts.",
        "limits": "Trained at 32×32 px; behaves like a near-constant \"fake\" predictor at 0.5; two years stale "
                  "relative to current generators.",
    },
    "jacob_distilled": {
        "name": "AI-Image-Detect (distilled)", "tier": "A", "arch": "Distilled ViT", "size": "~47 MB",
        "one": "Three detectors knowledge-distilled into one tiny/fast student.",
        "links": [("HF", "https://huggingface.co/jacoballessio/ai-image-detect-distilled")],
        "bio": "jacoballessio's efficiency-first detector: **three specialist teachers** (Midjourney-vs-real, "
               "Stable-Diffusion-vs-real, and SD-finetunes-vs-real) knowledge-distilled into a single tiny "
               "student, aiming for fast, generalisable detection of internet imagery.",
        "how": "The three teachers are distilled into an **11.8M-parameter ViT** (~47 MB — the smallest model in "
               "the pool). The design deliberately trades capacity for speed and portability; `logits[1]` = "
               "P(fake).",
        "train": "Real images from **Google Open Images**, paired with **BLIP-matched** Stable-Diffusion "
                 "generations and BLIP-matched Midjourney images (so real/fake differ mainly in origin). Reported "
                 "**74% validation / 72% real-world accuracy**, claimed +5 pts over other popular detectors on "
                 "both. License MIT.",
        "trend": "**394 downloads / 6 likes** — chosen as the \"tiny/fast distilled\" reference (smallest & "
                 "second-fastest model here at ~12 ms).",
        "result": "**ROC-AUC 0.439 (#18 / 24)** — below chance. The aggressive compression to 11.8M params "
                  "appears to have discarded the fine-grained cues needed on our distribution; even in-domain it "
                  "only reaches 74%. A clean illustration of the capacity/robustness trade-off.",
        "limits": "Capacity-limited (11.8M params); only ~74% even in-domain; the distillation teachers were "
                  "GAN/SD-era, predating current generators.",
    },
    "nahrawy_aiornot": {
        "name": "AIorNot", "tier": "A", "arch": "Swin-tiny", "size": "~110 MB",
        "one": "Lightweight binary Real/AI model from the 2023 HF \"AIorNot\" competition.",
        "links": [("HF", "https://huggingface.co/Nahrawy/AIorNot"),
                  ("Competition data", "https://huggingface.co/datasets/competitions/aiornot")],
        "bio": "Nahrawy's entry to the 2023 Hugging Face **\"AIorNot\"** competition — a compact binary Real/AI "
               "classifier that became a popular lightweight default (34 dependent Spaces).",
        "how": "**swin-tiny-patch4-window7-224** fine-tuned with a 2-class head (labels `Real` / `AI`); "
               "P(fake) = softmax(`AI`).",
        "train": "Fine-tuned on the **`competitions/aiornot`** dataset (the competition's real vs AI-generated "
                 "images). License Apache-2.0.",
        "trend": "**3,058 downloads / 13 likes** across 34 Spaces — chosen as a genuinely lightweight, widely "
                 "reused option.",
        "result": "**ROC-AUC 0.602 (#11 / 24)** — a good result for a Swin-tiny at only 17 ms: it out-ranks "
                  "several much larger models and offers one of the better speed/quality trade-offs in the pool.",
        "limits": "Small backbone with a modest ceiling; trained on a single 2023 competition distribution, so "
                  "newer generators are out-of-domain.",
    },
    "king1oo1_deepguard": {
        "name": "DeepGuard", "tier": "A", "arch": "SigLIP2", "size": "~372 MB",
        "one": "2026 SigLIP2 detector trained across five aggregated deepfake datasets.",
        "links": [("HF", "https://huggingface.co/king1oo1/deepfake-model"),
                  ("Base: SigLIP2-base", "https://huggingface.co/google/siglip2-base-patch16-224")],
        "bio": "king1oo1's **DeepGuard** — the inference engine behind a HF media-forensics Space. A 2026 "
               "SigLIP2 fine-tune deliberately trained across **five aggregated deepfake datasets** to cover many "
               "forgery types at once (faces, GAN, and modern diffusion).",
        "how": "**google/siglip2-base-patch16-224** + binary head, trained with a **progressive-unfreezing** "
               "schedule (head → top-6 transformer blocks → all layers), AdamW + cosine annealing, "
               "cross-entropy, and flip/rotate/jitter augmentation. `logits[1]` = P(fake).",
        "train": "A **balanced 40k slice (20k real / 20k fake)** sampled from five sources: a 190k faceswap set, "
                 "StyleGAN2 \"hard\" faces, GRAVEX-200K (FaceForensics++/DFDC/Celeb-DF/SD), DeepDetect-2025 "
                 "(DALL·E-3/MJ/SD3) and a SUT set (MJ v6/Flux/SDXL). Reported **78.5% accuracy / AUC > 0.86**. "
                 "License Apache-2.0.",
        "trend": "Recent (Apr 2026), **323 downloads / 1 like** — chosen as a fresh, deliberately multi-source "
                 "SigLIP2 detector.",
        "result": "**ROC-AUC 0.459 (#17 / 24)** — slightly below chance. Its training mix is **heavily "
                  "face/faceswap-weighted**, which biases it away from our generation-heavy fakes; the "
                  "\"cover-everything\" 40k sample is broad but shallow per-domain.",
        "limits": "Face/faceswap-heavy training; only 40k images sampled from the five sources; recent with low "
                  "adoption and limited independent validation.",
    },
    "sadra_sdxl_face": {
        "name": "SDXL-Deepfake-Detector", "tier": "A", "arch": "ViT", "size": "~343 MB",
        "one": "Face-focused detector (140k real/fake faces), reports ~97% on faces.",
        "links": [("HF", "https://huggingface.co/SadraCoding/SDXL-Deepfake-Detector"),
                  ("Train data", "https://huggingface.co/datasets/xhlulu/140k-real-and-fake-faces")],
        "bio": "SadraCoding's **face-specialist** detector, built as a **fine-tune of `Organika/sdxl-detector` "
               "(#5)** on a large real/fake face dataset — an ethics-motivated, offline, open tool. Reports ~97% "
               "on faces.",
        "how": "A **ViT** (labels `0 = artificial`, `1 = human` — note the reversed order vs most models, which "
               "our client accounts for). Fine-tuned on a single RTX 3060; `P(fake) = softmax(artificial)`.",
        "train": "**xhlulu 140k-real-and-fake-faces** — 70k real **FFHQ** faces vs 70k **StyleGAN**-generated "
                 "faces, aligned frontal crops. Reported **97% accuracy** on that face benchmark. License MIT.",
        "trend": "**104 downloads / 3 likes** — included as the deliberate \"face-specialist\" data point "
                 "(trained on GAN faces, not diffusion scenes).",
        "result": "**ROC-AUC 0.282 (#22 / 24)** — well below chance. It was trained on **aligned StyleGAN face "
                  "crops**, so it misreads full-scene diffusion generations (DALL·E/SDXL landscapes, video-model "
                  "frames); the card itself lists non-frontal, low-res and diffusion imagery as out-of-scope. A "
                  "textbook domain-mismatch result — not a wiring bug.",
        "limits": "Frontal-face-only; fakes are StyleGAN (GAN), not diffusion; explicitly poor on low-res, "
                  "occluded, or non-face imagery.",
    },
    "aidfr_real_v2": {
        "name": "AI-vs-Deepfake-vs-Real v2.0", "tier": "A", "arch": "SigLIP2 (3-class)", "size": "~370 MB",
        "one": "prithivMLmods 3-way classifier (AI / deepfake / real); synthetic classes merged to P(fake).",
        "links": [("HF", "https://huggingface.co/prithivMLmods/AI-vs-Deepfake-vs-Real-v2.0"),
                  ("Base: SigLIP2-base", "https://huggingface.co/google/siglip2-base-patch16-224")],
        "bio": "prithivMLmods' distinctive **three-way** classifier that separates **fully AI-generated**, "
               "**deepfake** (real content manipulated), and **real** images — a finer taxonomy than the usual "
               "binary. We collapse the two synthetic classes into P(fake) = 1 − P(Real).",
        "how": "**google/siglip2-base-patch16-224** with a **3-class** head (`label2id`: Artificial 0 / Deepfake "
               "1 / Real 2). We compute P(fake) = softmax(0) + softmax(1). The explicit AI-vs-deepfake split "
               "gives the head a richer decision boundary than a single fake bucket.",
        "train": "Fine-tuned on a 3-class AI/deepfake/real dataset (not enumerated on the card); the card's "
                 "training log shows **eval accuracy 0.9916** after one epoch. License Apache-2.0.",
        "trend": "**103 downloads / 4 likes** — the single prithivMLmods pick (author publishes many detector "
                 "variants; the one-per-author rule keeps just this one), chosen for its distinctive 3-way "
                 "framing.",
        "result": "**ROC-AUC 0.694 (#5 / 24) at 0.67 accuracy** — the **best ranking-plus-calibration balance in "
                  "Tier A** and 5th overall. The explicit separation of *generated* from *deepfake* appears to "
                  "help it distinguish full synthesis from face-swaps better than plain binary SigLIP2 models.",
        "limits": "Training data undisclosed; the card shows only a single-epoch log, so reported numbers are "
                  "thin; still miscalibrated enough that a threshold sweep would help.",
    },
    "ash_flux_vit": {
        "name": "FLUX Detector (ViT)", "tier": "A", "arch": "ViT", "size": "~343 MB",
        "one": "Generator-specialist for FLUX.1 imagery (reports 99.85% on FLUX).",
        "links": [("HF", "https://huggingface.co/ash12321/flux-detector-vit"),
                  ("Train: WikiArt", "https://huggingface.co/datasets/huggan/wikiart"),
                  ("Train: FLUX-1-dev-10k", "https://huggingface.co/datasets/ash12321/flux-1-dev-generated-10k")],
        "bio": "ash12321's **generator-specialist**: a detector purpose-built for **FLUX.1-dev** (Black Forest "
               "Labs) imagery, reporting a near-perfect **99.85% accuracy / AUC 1.0 with zero false positives** "
               "in-domain. Included to probe how a single-generator specialist transfers.",
        "how": "**ViT-Base/16-224** (uses Google's ViT image processor) with a binary head; ships a `model.py` "
               "helper. `logits[1]` = P(FLUX/fake).",
        "train": "Real = **`huggan/wikiart`** paintings; fake = **`ash12321/flux-1-dev-generated-10k`** (10k FLUX "
                 "renders), with disjoint train/val/test splits. Reported **test accuracy 0.9985, precision 1.0, "
                 "AUC 1.0**. License Apache-2.0.",
        "trend": "Brand-new (Dec 2025), **13 downloads / 1 like** — the single ash12321 pick (one-per-author), "
                 "chosen specifically for FLUX specialisation, a generator absent from most other training sets.",
        "result": "**ROC-AUC 0.650 (#8 / 24)** — a pleasant surprise: despite being FLUX-only, it transfers "
                  "reasonably to our broader diffusion fakes, landing upper-mid. Evidence that FLUX-era artefacts "
                  "share signal with other modern diffusion models.",
        "limits": "Single-generator (FLUX) training; the \"real\" class is **WikiArt paintings**, a narrow and "
                  "unusual real distribution that may inflate in-domain numbers and skew calibration on photos.",
    },
    "date3k2_vit": {
        "name": "ViT Real/Fake v3", "tier": "A", "arch": "ViT-B", "size": "~343 MB",
        "one": "Minimal ViT real/fake classifier (v3), ~98% on its own eval.",
        "links": [("HF", "https://huggingface.co/date3k2/vit-real-fake-classification-v3"),
                  ("Base: ViT-B/16", "https://huggingface.co/google/vit-base-patch16-224")],
        "bio": "date3k2's third-iteration real/fake ViT — a minimal, auto-generated model card (\"training and "
               "evaluation data: more information needed\") reporting ~98% on its own eval. Included as a plain, "
               "generic ViT reference point.",
        "how": "**google/vit-base-patch16-224** fine-tuned with a binary head. `logits[1]` = P(fake).",
        "train": "**Dataset undisclosed** (the auto-card omits it). Hyperparameters: AdamW (lr 5e-5), batch 32, "
                 "5 epochs, linear schedule with 10% warmup. Reported **98.17% accuracy / 0.983 F1** on its "
                 "eval split. License Apache-2.0.",
        "trend": "**3 downloads / 0 likes** — the lowest-traffic model here; included precisely as a neutral, "
                 "no-frills ViT baseline to contrast with the method-driven entries.",
        "result": "**ROC-AUC 0.432 (#19 / 24)** — below chance. A generic ViT with an undisclosed (and evidently "
                  "mismatched) training distribution has no particular reason to rank our fakes correctly, and it "
                  "doesn't.",
        "limits": "Completely undocumented training data; auto-generated card; below-chance ranking with no "
                  "method novelty to fall back on.",
    },
    "ummmaybe_vit": {
        "name": "AI Art Detector", "tier": "A", "arch": "ViT", "size": "~350 MB",
        "one": "One of the earliest open AI-art detectors (Oct 2022) — the report's biggest surprise.",
        "links": [("HF", "https://huggingface.co/umm-maybe/AI-image-detector")],
        "bio": "Matthew Maybe's **October-2022** proof-of-concept ViT for detecting AI-generated **art** — one "
               "of the earliest open detectors, predating Midjourney v5, SDXL and DALL·E-3. It is the **ancestor "
               "that `Organika/sdxl-detector` (#5) later fine-tuned**, and the most-liked model in this pool. The "
               "card is candid that it targets *artistic* images and is not a face-deepfake or general-photo "
               "detector.",
        "how": "An AutoTrain binary **ViT** classifier; `logits[1]` = P(fake). No frequency or foundation-model "
               "components — a straightforward 2022-era image classifier.",
        "train": "Trained on **AI-art vs real images scraped from Reddit links (2022)** — so the training data "
                 "predates all modern generators. Reported validation **AUC 0.980 / accuracy 94.2%**. License "
                 "CC-BY-4.0 (with a no-derivatives clause forbidding use to evade detection).",
        "trend": "**19,445 downloads / 97 likes — the most-liked model in the entire pool**. Kept as the "
                 "historical baseline and popularity anchor.",
        "result": "**ROC-AUC 0.761 (#2 / 24) — the report's biggest surprise.** The 2022 \"art detector\" "
                  "out-ranks every peer-reviewed CLIP method here. The likely explanation: its broad early-art "
                  "training captured *generic* synthesis cues (over-smoothness, texture statistics) that remain "
                  "predictive even for generators it never saw. A reminder that recency and peer review don't "
                  "guarantee real-world ranking.",
        "limits": "Trained on 2022 data with no modern generators; art-oriented (may stumble on photographic or "
                  "face content); ND licence restricts derivative use.",
    },
    "bombek1_siglip_dinov2": {
        "name": "SigLIP2 + DINOv2 ensemble", "tier": "B", "arch": "SigLIP2 + DINOv2", "size": "~3 GB",
        "one": "Community two-backbone ensemble designed to be quality-agnostic (JPEG/blur/noise robust).",
        "links": [("HF", "https://huggingface.co/Bombek1/ai-image-detector-siglip-dinov2"),
                  ("Train data: OpenFake", "https://huggingface.co/datasets/nebula-9000/OpenFake")],
        "bio": "Bombek1's **dual-encoder ensemble**, explicitly engineered to be **quality-agnostic** — to hold "
               "up under JPEG compression, blur and noise, which defeat many detectors on real web images. It "
               "fuses a semantic backbone (SigLIP2) with a self-supervised one (DINOv2), reporting a near-perfect "
               "0.9997 validation AUC and strong cross-dataset generalisation.",
        "how": "`EnsembleAIDetector` (~740M params, only ~8M trainable via **LoRA**): **SigLIP2-SO400M/14-384** "
               "(LoRA r=32 on q/v → 1152-d) ⊕ **DINOv2-Large/14** (LoRA r=32 on qkv → 1024-d), concatenated to "
               "2176-d → LayerNorm → MLP(2176→512→256→1) → sigmoid = P(AI). We manually remap the SigLIP keys and "
               "load with `strict=False` to bridge a transformers-5.x version drift.",
        "train": "**OpenFake** (~95k train / 5k val, 392×392), 5 epochs, **focal loss** (γ=2, α=0.25), AdamW "
                 "(2e-4 head / 5e-5 LoRA), cosine schedule. Crucially, training uses **aggressive degradations** "
                 "(JPEG q30–95, Gaussian blur/noise, 50% downscale-and-restore, colour jitter) to force "
                 "robustness. OpenFake spans **25+ generators** (SD 1.5/2.1/XL/3.5, FLUX, DALL·E-3, MJ, Imagen, "
                 "plus StyleGAN/ProGAN/BigGAN). Reported val **AUC 0.9997**, cross-dataset ~97%. License MIT.",
        "trend": "**282 downloads / 8 likes** (Jan 2026) — chosen for its **robustness angle**: the "
                 "degradation-heavy training makes it a strong candidate for compressed, real-world inputs, a gap "
                 "most Tier-A models ignore.",
        "result": "**ROC-AUC 0.649 (#9 / 24) — best in Tier B**, with high precision (0.909). The SigLIP2+DINOv2 "
                  "fusion gives steady, well-behaved ranking. It sits below its self-reported 0.9997 because our "
                  "face-swap-heavy fakes differ from OpenFake's generator mix — but it degrades gracefully rather "
                  "than collapsing.",
        "limits": "Heavy (~3 GB, ~67 ms). Card notes ~9% false positives on cluttered COCO-style photos and "
                  "~50% accuracy on sub-128px images (e.g. CIFAKE); designed for photographs, not illustrations "
                  "or screenshots.",
    },
    "gend_dinov3_l": {
        "name": "GenD DINOv3-L", "tier": "B", "arch": "DINOv3-ViT-L/16 + head", "size": "~1.2 GB",
        "one": "WACV 2026 study of DINOv3 self-supervised features for cross-generator generalization.",
        "links": [("HF", "https://huggingface.co/yermandy/GenD_DINOv3_L"),
                  ("Paper: GenD (arXiv 2508.06248), WACV 2026", "https://arxiv.org/abs/2508.06248")],
        "bio": "The \"GenD (DINO)\" model (Table 2) from **\"Deepfake Detection that Generalizes Across "
               "Benchmarks\"** (Yermandy et al., **WACV 2026**, arXiv 2508.06248). The paper asks a pointed "
               "question — *which frozen foundation backbone generalises best across detection benchmarks?* — and "
               "argues that a lightly-adapted **DINOv3** encoder transfers across generators better than "
               "task-specific designs.",
        "how": "A **gated DINOv3-ViT-L/16** backbone with a lightweight detection head; softmax → P(fake). "
               "Because the DINOv3 backbone is gated, loading needs an `HF_TOKEN`; and because GenD's own "
               "`from_pretrained` clashes with transformers-5.x meta-device init, we build `GenD(config)` "
               "explicitly, then `load_state_dict` with a DINOv3 layer-key remap.",
        "train": "Trained for **cross-benchmark generalisation** on standard deepfake/synthetic benchmarks per "
                 "the paper's protocol (FaceForensics-style real/fake plus generated sets); the emphasis is on "
                 "transfer, not fitting any single generator. License MIT.",
        "trend": "**573 downloads / 0 likes** (Nov 2025) — chosen as the **peer-reviewed DINOv3** representative, "
                 "to test whether self-supervised backbones generalise as the paper claims.",
        "result": "**ROC-AUC 0.597 (#12 / 24)** — modestly above chance, so the generalisation claim holds "
                  "*partially* on our out-of-distribution set. But at the 0.5 threshold it badly **under-calls "
                  "fakes** (recall 0.24, MCC ≈ 0): good relative ranking, poor default calibration here.",
        "limits": "Requires a token for the gated backbone; strong ranking but needs re-thresholding; evaluated "
                  "here far from its training benchmarks.",
    },
    "aide": {
        "name": "AIDE", "tier": "B", "arch": "Frequency (SRM/DCT) + semantic hybrid", "size": "~3.6 GB",
        "one": "ICLR 2025 hybrid combining low-level frequency forensics with high-level semantics.",
        "links": [("HF mirror", "https://huggingface.co/meet4150/AIDE_image_detector"),
                  ("Paper: AIDE (arXiv 2406.19435), ICLR 2025", "https://arxiv.org/abs/2406.19435")],
        "bio": "AIDE, from **\"A Sanity Check for AI-generated Image Detection\"** (Yan et al., **ICLR 2025**, "
               "arXiv 2406.19435). The paper's premise is that robust detection needs *both* low-level forensic "
               "artefacts *and* high-level semantics, and it builds an explicit two-expert model to prove it. "
               "This HF checkpoint is a community-repackaged `checkpoint-19` from a multi-source training run.",
        "how": "A genuine dual-branch design: (1) a **frequency expert** — a fixed **30-filter SRM high-pass "
               "bank** feeding two ResNet-50 branches over **four DCT-reconstructed views** "
               "(`DCT_base_Rec`, window 32 / stride 16); (2) a **semantic expert** — a frozen **OpenCLIP "
               "ConvNeXt-XXL** trunk. The ConvNeXt embedding (3072→256) is concatenated with the averaged "
               "forensic embedding (2048) → 2304-d → MLP(2304→1024→2). Requires the repo's custom `inference.py` "
               "preprocessing (five stacked views). `0=real, 1=fake`.",
        "train": "This checkpoint (54.4M trainable params) is a **multi-source run**: base-lr 5e-4, 20 epochs, "
                 "batch 8, RandAug, label smoothing 0.1. Reported **77.9% validation top-1 (best 78.6% @ ep17)** "
                 "in-domain. The upstream AIDE benchmarks on diffusion-heavy academic sets. License MIT.",
        "trend": "Mirror has **10 downloads** — chosen for its **ICLR-2025 peer review** and the distinctive, "
                 "genuinely-hybrid frequency+semantic architecture (nothing else in the pool works this way).",
        "result": "**ROC-AUC 0.392 (#20 / 24)** — below chance. Its **SRM/DCT frequency prior is tuned to "
                  "specific generator artefacts**; on our mixed diffusion + face-swap set those forensic cues "
                  "mislead more than they help, dragging the semantic branch down. Even in-domain it only reaches "
                  "~78%, so the gap is unsurprising.",
        "limits": "Heavy (~3.6 GB, ~63 ms); the frequency prior is brittle across unseen generators; this is a "
                  "single community `checkpoint-19`, not the authors' best released model.",
    },
    "c2p_clip": {
        "name": "C2P-CLIP", "tier": "C", "arch": "Frozen CLIP ViT-L/14 + linear head", "size": "~1.2 GB",
        "one": "AAAI 2025 — injects a learned category prompt into CLIP's text side during training.",
        "links": [("GitHub", "https://github.com/chuangchuangtan/C2P-CLIP-DeepfakeDetection"),
                  ("Paper: C2P-CLIP (arXiv 2408.09647), AAAI 2025", "https://arxiv.org/abs/2408.09647")],
        "bio": "**C2P-CLIP** — \"Category Common Prompt CLIP\" (Tan et al., **AAAI 2025**, arXiv 2408.09647). "
               "The idea is to improve CLIP-based detection by injecting a learned **category (real/fake) prompt** "
               "into CLIP's text encoder *during training*, which reshapes the image encoder's features so that "
               "the real/fake boundary is crisper — without adding any inference-time cost.",
        "how": "A **frozen CLIP ViT-L/14** image encoder + visual projection + a single-logit linear head "
               "(`model.fc`). At inference the text tower is **unused** (the prompt only shaped training): image "
               "embed → L2-normalise → `fc` → sigmoid = P(fake). We reproduce the repo's `C2P_CLIP` module and "
               "its exact test transform (tile-up-if-small → CenterCrop 224 → CLIP-normalise) and load the 1.2 GB "
               "strict state-dict.",
        "train": "Trained on **ProGAN / ForenSynths** (the standard GAN-detection protocol: ProGAN fakes vs "
                 "LSUN reals). The paper reports **~98% accuracy on unseen GANs**, a strong cross-GAN result.",
        "trend": "GitHub-hosted (weights mirrored on our `deepsafe-weights`). Chosen as the **AAAI-2025 "
                 "CLIP-prompt SOTA** — a recent, well-cited representative of the prompt-tuning approach to "
                 "detection.",
        "result": "**ROC-AUC 0.385 (#21 / 24)** — below chance, but verified **NOT inverted** (`real_001` → "
                  "0.03, correct). This is a pure **GAN→diffusion domain gap**: the prompt-shaped CLIP features "
                  "keyed on GAN fingerprints that our diffusion / face-swap fakes simply don't carry. We keep the "
                  "repo's faithful mapping rather than flipping it to game AUC.",
        "limits": "Trained on GAN imagery (ProGAN); underperforms on diffusion and face-swaps; strong only "
                  "within its GAN-detection lineage.",
    },
    "rine": {
        "name": "RINE", "tier": "C", "arch": "Frozen CLIP ViT-L/14 intermediate blocks + head", "size": "~25 MB (head)",
        "one": "ECCV 2024 — uses CLIP's intermediate encoder blocks, not just the final CLS token.",
        "links": [("GitHub", "https://github.com/mever-team/rine"),
                  ("Paper: RINE (arXiv 2402.19091), ECCV 2024", "https://arxiv.org/abs/2402.19091")],
        "bio": "**RINE** — \"Leveraging Representations from **INtermediate Encoder-blocks** for Synthetic Image "
               "Detection\" (Koutlis & Papadopoulos, **ECCV 2024**, arXiv 2402.19091). Its insight: CLIP's *final* "
               "layer captures high-level semantics, but the **intermediate** blocks carry the low-level detail "
               "that actually distinguishes synthetic images — so it should use all of them, weighted.",
        "how": "Over a **frozen OpenAI CLIP ViT-L/14**, RINE grabs the **CLS token from every transformer "
               "block** (via ln_2 hooks), projects each with a learnable linear map into a forgery-aware space, "
               "weights the blocks with a **Trainable Importance Estimator**, sums, and feeds a small head → "
               "sigmoid. Only the **6.3M-parameter trainable part** ships (~25 MB); `clip.load` re-pulls the "
               "backbone; we load with `strict=False` (only the frozen `clip.*` weights are \"missing\").",
        "train": "Trained **only on ProGAN** (1/2/4 object classes) from ForenSynths, yet evaluated on **20 test "
                 "datasets** (GAN, diffusion, deepfake, DALL·E, …) where it reports a **+10.6% absolute** average "
                 "improvement over the prior SOTA — while training for just **one epoch (~8 min)**. We use the "
                 "repo's default 4-class checkpoint.",
        "trend": "GitHub-hosted (weights ship in the repo's `ckpt/`). Chosen as the **ECCV-2024 "
                 "intermediate-block** method — a strong, minimal, and unusually generalisable CLIP detector.",
        "result": "**ROC-AUC 0.737 (#4 / 24) — the best CLIP-based detector here.** Despite GAN-only training, "
                  "its intermediate-block features transfer to our diffusion / face-swap fakes far better than "
                  "C2P-CLIP (0.385) or DeCLIP (0.275) — direct empirical support for the paper's central claim. "
                  "At 0.5 it under-fires (recall 0.23), so its excellent ranking needs re-thresholding.",
        "limits": "GAN-trained (ProGAN); miscalibrated at the default threshold; needs the OpenAI `clip` package "
                  "and a backbone download at first use.",
    },
    "drct": {
        "name": "DRCT", "tier": "C", "arch": "ConvNeXt-Base (in22k) + 2-class head", "size": "~355 MB",
        "one": "ICML 2024 Spotlight — diffusion-reconstruction contrastive training. Best model overall.",
        "links": [("GitHub", "https://github.com/beibuwandeluori/DRCT"),
                  ("ModelScope weights", "https://modelscope.cn/datasets/BokingChen/DRCT-2M/files")],
        "bio": "**DRCT** — \"Diffusion Reconstruction Contrastive Training\" (Chen et al., **ICML 2024 "
               "Spotlight**). It targets *universal* diffusion detection with a clever training signal: pass any "
               "image through a diffusion autoencoder to make a near-identical **reconstruction**, then teach the "
               "model to separate originals from reconstructions. This forces it to key on the **subtle trace any "
               "diffusion pass leaves**, rather than generator-specific texture.",
        "how": "A **ConvNeXt-Base (IN-22k)** backbone → 1024-d embedding → 2-class head, trained with a "
               "**margin-based contrastive loss** over real / fake / diffusion-reconstructed triples. We rebuild "
               "the repo's `ContrastiveModels` via `network.models.get_models`, apply its test transform "
               "(pad-if-small → CenterCrop 224 → ImageNet-normalise), and take `P(fake) = softmax[1]`.",
        "train": "Trained on **DRCT-2M** — real images plus generations from ~16 diffusion families and their "
                 "SDv1 reconstructions. The checkpoint used is `convnext_base_in22k`, DRCT-2M / DR=SDv1 (a 355 MB "
                 "file extracted from a 4.2 GB ModelScope bundle).",
        "trend": "GitHub + ModelScope hosted. Chosen as the **ICML-2024 Spotlight** diffusion-reconstruction "
                 "method — the most conceptually targeted model for a diffusion-heavy test set.",
        "result": "**ROC-AUC 0.866 — #1 of all 24.** Its objective is the closest match to our distribution: it "
                  "nails diffusion fakes (HG / PICSI / SORA → P≈1.0) with precision 0.933. The low 0.47 accuracy "
                  "is **pure miscalibration** (peaky scores), not weak ranking — and its 51 false-negatives are "
                  "concentrated in **face-swaps** (DEEPCAM / DF40 / PDNE → P≈0), which carry no diffusion trace "
                  "for it to fingerprint.",
        "limits": "A **face-swap blind spot** (no diffusion pass to detect); scores are peaky and need "
                  "per-deployment threshold tuning to convert its strong ranking into good accuracy.",
    },
    "declip": {
        "name": "DeCLIP", "tier": "C", "arch": "Frozen CLIP ViT-L/14 + conv decoder (localization)", "size": "~627 MB",
        "one": "WACV 2025 (Bitdefender) — a localization model; detection = mean(sigmoid(map)).",
        "links": [("GitHub", "https://github.com/bit-ml/DeCLIP"),
                  ("Paper: DeCLIP (arXiv 2409.08849), WACV 2025", "https://arxiv.org/abs/2409.08849")],
        "bio": "**DeCLIP** — \"Decoding CLIP Representations for Deepfake Localization\" (Smeu et al., **WACV "
               "2025**, Bitdefender, arXiv 2409.08849). Unlike every other model here, DeCLIP is fundamentally a "
               "**localization** method: it predicts *where* an image was manipulated (a per-patch heatmap), "
               "aimed at partial edits like LDM inpainting. The repo also exposes an official detection score "
               "derived from that map.",
        "how": "A **frozen CLIP ViT-L/14** whose intermediate features feed a **convolutional decoder** that "
               "outputs a per-patch manipulation map. Detection uses the repo's own reduction "
               "**P(fake) = mean(sigmoid(map))** (its `validate()` path). We rebuild `CLIPModelLocalisation` with "
               "the vendored CLIP, read feature-layer/decoder from the ckpt, and load `strict=False`.",
        "train": "Trained on **LDM-inpainting localization** data (the paper finds LDM training generalises "
                 "best); we use that LDM ViT checkpoint (627 MB) from Bitdefender's public GCS bucket.",
        "trend": "GCS-hosted. Chosen as the **WACV-2025 detection-plus-localization** representative — the only "
                 "localization-first model in the pool, included deliberately to test that paradigm.",
        "result": "**ROC-AUC 0.275 (#23 / 24)**, with F1 / precision / recall all **0 at 0.5** (it never crosses "
                  "the threshold). This is expected, not a bug: a **region localizer** produces near-empty maps "
                  "on **fully-synthetic** images (DALL·E / SDXL) and face-swaps — there is no localized edit to "
                  "find — so mean-map scores stay uniformly low. We keep the repo's own detection reduction "
                  "rather than flipping it.",
        "limits": "Designed for **partial / inpainting** manipulation, not whole-image generation — a poor fit "
                  "as a global detector on this set; strong only where a localized edit exists.",
    },
    "ntire2026_deepfake": {
        "name": "NTIRE 2026 DINOv3 ensemble", "tier": "C", "arch": "2x DINOv3-ViT-7B (fp16 ensemble)", "size": "~54 GB",
        "one": "CVPR 2026 / NTIRE challenge-grade robustness ensemble (Ant International).",
        "links": [("HF", "https://huggingface.co/scarlettss/NTIRE2026_deepfake"),
                  ("Challenge dataset", "https://huggingface.co/datasets/scarlettss/NTIRERobustDeepfakeDetectionChallengeCVPR2026")],
        "bio": "**Ant International's** entry to the **CVPR 2026 NTIRE Robust Deepfake Detection Challenge** — a "
               "challenge-grade ensemble explicitly engineered to survive real-world **degradations** "
               "(compression, resize, noise). It represents the current \"win-a-benchmark, spend-the-compute\" "
               "end of the spectrum, in deliberate contrast to the lightweight community models.",
        "how": "An ensemble of **two full end-to-end DINOv3-ViT-7B** detectors (hidden 4096, 40 layers): "
               "**v135 \"ViT-CLS\"** (pooler_output → Linear) and **v156 \"ViT-Attnpool\"** (patch tokens → "
               "AttentionPooling → Linear). Final score **P(fake) = 0.35·σ(v135) + 0.65·σ(v156)**. Both experts "
               "load in **fp16 (~14 GB each)** so the pair fits one H100; the DINOv3 weights are baked into the "
               "`.pt` (no gated download). We adaptively remap `experts.[model.]layer.N` keys for transformers-5.x "
               "(verified 0 missing / 0 unexpected).",
        "train": "Trained on the **NTIRE Robust Deepfake Detection Challenge (CVPR 2026)** dataset, with the "
                 "challenge's degradation-augmentation regime; the whole design goal is robustness under "
                 "corruption rather than clean-image peak accuracy.",
        "trend": "Fresh challenge upload (Mar 2026), **0 downloads / 1 like** — chosen as the **CVPR-2026, "
                 "challenge-grade, robustness-oriented** flagship and the largest model in the study.",
        "result": "**ROC-AUC 0.750 (#3 / 24) at 0.60 accuracy** — the **best ranking-plus-calibration balance of "
                  "any top model**. Massive-scale end-to-end DINOv3-7B fine-tuning yields robust, well-behaved "
                  "scores across our varied fakes; it is the most *balanced* strong detector in the pool, even if "
                  "DRCT out-ranks it on pure AUC.",
        "limits": "Enormous and slow (~54 GB on disk, ~73 ms/img — the slowest here); tuned to a specific "
                  "challenge distribution; heavy operational cost for a +0.75 AUC.",
    },
}


def load_results() -> list[APIResult]:
    results: list[APIResult] = []
    for model_dir in sorted(LOGS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for jf in sorted(model_dir.glob("*.json")):
            d = json.loads(jf.read_text(encoding="utf-8"))
            results.append(APIResult(**{k: v for k, v in d.items() if k in _KEEP}))
    return results


def _fmt(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def build_run(results: list[APIResult]) -> BenchmarkRun:
    seen: dict[str, str] = {r.image_filename: r.ground_truth for r in results}
    n_fake = sum(1 for gt in seen.values() if gt == "FAKE")
    run = BenchmarkRun(
        run_id="per-model", start_time="", end_time="",
        dataset_size=len(seen), fake_count=n_fake, real_count=len(seen) - n_fake,
        providers=sorted({r.provider for r in results}), results=results,
    )
    return compute_all_metrics(run)


def per_image_rows(results: list[APIResult], provider: str) -> list[dict]:
    rows = [r for r in results if r.provider == provider]
    rows.sort(key=lambda r: (r.confidence if r.confidence is not None else -1), reverse=True)
    out = []
    for r in rows:
        conf = r.confidence if r.confidence is not None else 0.0
        pred = r.prediction or ("FAKE" if conf >= 0.5 else "REAL")
        correct = (pred == r.ground_truth)
        out.append({"img": r.image_filename, "gt": r.ground_truth, "conf": conf,
                    "pred": pred, "ok": correct})
    return out


def render(mid: str, doc: dict, m, rank: int, total: int, rows: list[dict], generated: str) -> str:
    tier = {"A": "A — HF `transformers`-native",
            "B": "B — HF custom loader",
            "C": "C — academic repo"}[doc["tier"]]
    links = " · ".join(f"[{lbl}]({url})" for lbl, url in doc["links"])
    misses = [r for r in rows if not r["ok"]]
    fp = [r for r in misses if r["gt"] == "REAL"]   # real called fake
    fn = [r for r in misses if r["gt"] == "FAKE"]   # fake called real

    def tbl(rs: list[dict]) -> str:
        lines = ["| # | Image | Truth | P(fake) | Predicted | Correct |",
                 "|---|---|---|---|---|---|"]
        for i, r in enumerate(rs, 1):
            mark = "✅" if r["ok"] else "❌"
            lines.append(f"| {i} | `{r['img']}` | {r['gt']} | {_fmt(r['conf'])} | {r['pred']} | {mark} |")
        return "\n".join(lines)

    md = f"""# {doc['name']} — `{mid}`

**Tier {tier}** · **Arch:** {doc['arch']} · **Size:** {doc['size']} · **Rank:** {rank} / {total} by ROC-AUC

> {doc['one']}

**Links:** {links}
**Combined report:** [`../MODELS_REPORT.md`](../MODELS_REPORT.md) · **Catalogue:** [`../../../plan/MODELS.md`](../../../plan/MODELS.md)

## Metrics — 100-image `Source/` set (79 fake / 21 real)

| Metric | Value |
|---|---|
| ROC-AUC | **{_fmt(m.roc_auc)}** |
| Accuracy @0.5 | {_fmt(m.accuracy)} |
| F1 @0.5 | {_fmt(m.f1_score)} |
| Precision | {_fmt(m.precision)} |
| Recall | {_fmt(m.recall)} |
| MCC | {_fmt(m.mcc)} |
| TP / FP / TN / FN | {m.true_positives} / {m.false_positives} / {m.true_negatives} / {m.false_negatives} |
| Mean latency | {_fmt(m.latency_mean_ms, 1)} ms |
| Inferences (ok / fail) | {m.successful} / {m.failed} |

*AUC is threshold-free (ranking); Acc/F1/Prec/Rec are at a fixed 0.5 cut. AUC < 0.5 = ranks below chance on this distribution (domain mismatch), not a bug — mappings are kept faithful to the source repo.*

## Overview

{doc['bio']}

## How it works

{doc['how']}

## Training data & recipe

{doc['train']}

## Why we included it (HF trend / selection)

{doc['trend']}

## Our benchmark result — analysis

{doc['result']}

## Limitations & caveats

{doc['limits']}

## Misclassifications @0.5 ({len(misses)} of {len(rows)})

- **False positives (real called fake): {len(fp)}** — {', '.join('`'+r['img']+'`' for r in fp) if fp else '_none_'}
- **False negatives (fake called real): {len(fn)}** — {len(fn)} images{' (top misses: ' + ', '.join('`'+r['img']+'`' for r in fn[:8]) + ('…' if len(fn) > 8 else '') + ')' if fn else ''}

<details>
<summary><b>Full {len(rows)}-image scores (sorted by P(fake), high → low)</b></summary>

{tbl(rows)}

</details>

---
*Generated {generated} by `scripts/report_per_model.py` from `logs/opensource_models/{mid}/`. HF download/like counts are live API values from 2026-07-03.*
"""
    return md


def main() -> None:
    if not LOGS_DIR.exists():
        sys.exit(f"No logs at {LOGS_DIR} — run scripts/run_research.py first.")
    results = load_results()
    if not results:
        sys.exit(f"No per-image logs under {LOGS_DIR}.")

    run = build_run(results)
    metrics = {m.provider: m for m in run.provider_metrics}
    ranked = sorted(run.provider_metrics, key=lambda m: m.roc_auc, reverse=True)
    rank_of = {m.provider: i for i, m in enumerate(ranked, 1)}
    total = len(ranked)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for mid, doc in DOCS.items():
        if mid not in metrics:
            print(f"  WARN no logs for {mid}, skipping")
            continue
        rows = per_image_rows(results, mid)
        md = render(mid, doc, metrics[mid], rank_of[mid], total, rows, generated)
        out = OUT_DIR / f"{mid}.md"
        out.write_text(md, encoding="utf-8")
        written.append((rank_of[mid], mid))

    # any model in logs but missing from DOCS?
    for mid in metrics:
        if mid not in DOCS:
            print(f"  WARN logs for {mid} but no narrative in DOCS")

    print(f"Wrote {len(written)} per-model reports -> {OUT_DIR}")
    for rank, mid in sorted(written):
        print(f"  #{rank:2d}  {mid}")


if __name__ == "__main__":
    main()
