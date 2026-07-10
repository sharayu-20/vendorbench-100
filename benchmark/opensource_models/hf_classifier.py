"""Generic loader for HF `transformers` image-classification detectors (Tier A).

One class runs any of the 16 Tier-A models. Per-model behaviour comes entirely
from the entry dict built by that model's script in this package
(hf_repo/model_path + fake_labels / fake_index + optional overrides). P(fake) is
the summed softmax mass over the classes designated "fake".
"""

from __future__ import annotations

from typing import Any

from benchmark.opensource_models.base_research import BaseResearchClient


class HFClassifierClient(BaseResearchClient):
    def __init__(self, entry: dict[str, Any], device: str = "cuda:0",
                 logs_dir: str = "logs") -> None:
        self._fake_labels = entry.get("fake_labels")
        self._fake_index = entry.get("fake_index")
        self._image_size = entry.get("image_size")  # optional override
        # some repos ship a corrupt/mismatched .safetensors -> pin the weight file
        self._use_safetensors = entry.get("use_safetensors")  # None = auto
        # override head size when config num_labels != checkpoint classifier
        self._num_labels = entry.get("num_labels")
        # prefer local models/<id>/ weights; fall back to HF repo id
        self._src = entry.get("model_path") or entry["hf_repo"]
        if not self._fake_labels and self._fake_index is None:
            raise ValueError(f"{entry['id']}: needs fake_labels or fake_index")
        super().__init__(entry["id"], entry["hf_repo"], device=device, logs_dir=logs_dir)

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForImageClassification, AutoImageProcessor

        load_kwargs: dict[str, Any] = {}
        if self._use_safetensors is not None:
            load_kwargs["use_safetensors"] = self._use_safetensors
        if self._num_labels is not None:
            from transformers import AutoConfig
            cfg0 = AutoConfig.from_pretrained(self._src)
            cfg0.num_labels = int(self._num_labels)
            load_kwargs["config"] = cfg0

        self.model = (
            AutoModelForImageClassification.from_pretrained(self._src, **load_kwargs)
            .to(self.device)
            .eval()
        )
        self._torch = torch
        cfg = self.model.config
        arch = (getattr(cfg, "architectures", None) or [""])[0].lower()
        size = int(self._image_size or getattr(cfg, "image_size", 224) or 224)

        # Preferred: the model's own AutoImageProcessor. Fall back to a manual
        # torchvision transform when the repo ships no valid preprocessor_config.
        self._manual = None
        try:
            self.processor = AutoImageProcessor.from_pretrained(self._src, use_fast=True)
            if self._image_size:  # force size override (e.g. opensight 440->384)
                self.processor.size = {"height": size, "width": size}
                if getattr(self.processor, "crop_size", None):
                    self.processor.crop_size = {"height": size, "width": size}
        except Exception:  # noqa: BLE001
            from torchvision import transforms
            # Swin family uses ImageNet stats; ViT/SigLIP default to 0.5.
            if "swin" in arch:
                mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            else:
                mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
            self.processor = None
            self._manual = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])
            self._manual_meta = {"size": size, "mean": mean, "std": std, "fallback": True}

        # normalize id2label keys to int
        raw_map = self.model.config.id2label or {}
        self.id2label = {int(k): str(v) for k, v in raw_map.items()}

        if self._fake_index is not None:
            self._fake_idx = {int(i) for i in self._fake_index}
        else:
            wanted = {s.lower() for s in self._fake_labels}
            self._fake_idx = {
                i for i, lbl in self.id2label.items() if lbl.lower() in wanted
            }
        if not self._fake_idx:
            raise ValueError(
                f"{self.model_id}: no class matched fake_labels={self._fake_labels} "
                f"in id2label={self.id2label}"
            )

    def _infer(self, image: Any) -> tuple[float, dict[str, Any]]:
        torch = self._torch
        if self.processor is not None:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        else:
            pixel_values = self._manual(image).unsqueeze(0).to(self.device)
            inputs = {"pixel_values": pixel_values}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0]
            probs = torch.softmax(logits, dim=-1).float().cpu().tolist()

        p_fake = float(sum(probs[i] for i in self._fake_idx if i < len(probs)))
        top_idx = int(max(range(len(probs)), key=lambda i: probs[i]))

        raw = {
            "loader": "hf_classifier",
            "id2label": self.id2label,
            "probs": {self.id2label.get(i, str(i)): round(p, 6) for i, p in enumerate(probs)},
            "fake_classes": sorted(self._fake_idx),
            "label_mapping": (
                f"fake_labels={self._fake_labels}" if self._fake_labels
                else f"fake_index={self._fake_index}"
            ),
            "predicted_label_raw": self.id2label.get(top_idx, str(top_idx)),
            "prob_fake": round(p_fake, 6),
        }
        if self._manual is not None:
            raw["preprocess"] = getattr(self, "_manual_meta", {"fallback": True})
        return p_fake, raw
