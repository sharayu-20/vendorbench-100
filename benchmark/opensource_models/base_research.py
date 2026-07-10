"""Base client for local (on-GPU) open-source detectors.

Subclasses the commercial `BaseClient` so results flow into the exact same
`APIResult` schema, evidence logs, metrics, and reports. The only difference:
`detect()` runs a local model instead of an HTTP call (no retry / no network).

Evidence is written to  logs/opensource_models/<model_id>/<image_stem>.json
"""

from __future__ import annotations

import time
import logging
from typing import Any

from benchmark.base_client import BaseClient
from benchmark.models import ImageSample, APIResult

logger = logging.getLogger(__name__)


class BaseResearchClient(BaseClient):
    """Local-inference detector. Subclasses implement `_load()` and `_infer()`."""

    TRACK = "opensource_models"

    def __init__(self, model_id: str, hf_repo: str, device: str = "cuda:0",
                 logs_dir: str = "logs", **_ignored: Any) -> None:
        # PROVIDER_NAME drives the log folder; set before BaseClient.__init__.
        self.PROVIDER_NAME = model_id
        self.model_id = model_id
        self.hf_repo = hf_repo
        self.device = device
        super().__init__(timeout=0, max_retries=1, logs_dir=logs_dir)
        self._load()

    # --- subclass contract ---------------------------------------------------
    def _load(self) -> None:
        """Load model + preprocessor onto self.device (once)."""
        raise NotImplementedError

    def _infer(self, image: "Any") -> tuple[float, dict[str, Any]]:
        """Run one PIL image → (p_fake in [0,1], raw_response dict)."""
        raise NotImplementedError

    # --- unused HTTP hooks (satisfy the ABC; never called) -------------------
    def _call_api(self, sample: ImageSample) -> dict[str, Any]:
        raise NotImplementedError("research clients use local inference")

    def _normalize(self, raw: dict[str, Any]) -> tuple[str, float]:
        raise NotImplementedError("research clients use local inference")

    # --- local-inference detect() -------------------------------------------
    def detect(self, sample: ImageSample) -> APIResult:
        from PIL import Image

        start = time.perf_counter()
        try:
            image = Image.open(sample.path).convert("RGB")
            p_fake, raw = self._infer(image)
            elapsed_ms = (time.perf_counter() - start) * 1000
            raw.setdefault("model_id", self.model_id)
            raw.setdefault("hf_repo", self.hf_repo)
            raw.setdefault("device", self.device)
            raw["inference_ms"] = round(elapsed_ms, 2)
            result = APIResult(
                provider=self.PROVIDER_NAME,
                image_path=sample.path,
                image_filename=sample.filename,
                ground_truth=sample.ground_truth,
                prediction="FAKE" if p_fake >= 0.5 else "REAL",
                confidence=round(float(p_fake), 6),
                raw_response=raw,
                latency_ms=round(elapsed_ms, 2),
                image_sha256=sample.sha256,
                success=True,
                retry_count=0,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("Inference failed for %s on %s: %s",
                         self.PROVIDER_NAME, sample.filename, exc)
            result = APIResult(
                provider=self.PROVIDER_NAME,
                image_path=sample.path,
                image_filename=sample.filename,
                ground_truth=sample.ground_truth,
                prediction="ERROR",
                confidence=0.0,
                raw_response={"error": str(exc)},
                latency_ms=round(elapsed_ms, 2),
                image_sha256=sample.sha256,
                success=False,
                error_message=str(exc),
                retry_count=0,
            )

        self._save_evidence(result)
        return result

    def unload(self) -> None:
        """Free GPU memory after a model finishes its run."""
        try:
            import torch
            for attr in ("model", "processor"):
                if hasattr(self, attr):
                    setattr(self, attr, None)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
