"""TruthScan - AI Image Detection client using official SDK.

Uses the truthscan-image-detector-client SDK which handles the full
presign -> upload -> detect -> poll workflow automatically.
"""

from __future__ import annotations

import logging
from typing import Any

from benchmark.models import ImageSample
from benchmark.base_client import BaseClient, TransientAPIError

logger = logging.getLogger(__name__)


class TruthScanClient(BaseClient):
    PROVIDER_NAME = "truthscan"
    TRACK = "commercial_apis"

    def __init__(self, config, **kwargs) -> None:
        super().__init__(**kwargs)
        self._api_key = config.api_key

    def _call_api(self, sample: ImageSample) -> dict[str, Any]:
        from truthscan.image_detection import ImageDetectionClient

        client = ImageDetectionClient(api_key=self._api_key)
        try:
            result = client.detect(
                image=sample.path,
                max_poll_attempts=60,
                poll_interval_seconds=2.0,
            )
            if isinstance(result, dict):
                return result
            return {"raw": str(result), "result": result.__dict__ if hasattr(result, "__dict__") else str(result)}
        except Exception as exc:
            err_msg = str(exc).lower()
            if "rate" in err_msg or "429" in err_msg or "limit" in err_msg:
                from benchmark.base_client import RateLimitError
                raise RateLimitError(f"TruthScan rate limit: {exc}") from exc
            if "500" in err_msg or "502" in err_msg or "503" in err_msg:
                raise TransientAPIError(f"TruthScan server error: {exc}") from exc
            raise

    def _normalize(self, raw: dict[str, Any]) -> tuple[str, float]:
        """Normalize TruthScan response.

        Actual response structure:
        {
          "status": "done",
          "result": 3.82,              # raw ML score
          "result_details": {
            "final_result": "Digitally Edited" | "AI Generated" | "Real" | ...,
            "confidence": 3.82,        # raw score (not 0-1)
            "ml_model": ["Digitally Edited", 3.82],
            ...
          }
        }

        Classification mapping:
          "AI Generated", "Digitally Edited" -> FAKE
          "Real", "Human" -> REAL

        The confidence score is a raw float (not 0-1), so we
        normalize it: score >= 2.0 -> FAKE, else REAL.
        We also clamp to 0-1 for reporting using min(score/5, 1.0).
        """
        details = raw.get("result_details", {}) or {}
        final_result = ""
        if isinstance(details, dict):
            final_result = str(details.get("final_result", "")).lower()

        raw_score = 0.0
        if isinstance(details, dict) and "confidence" in details:
            raw_score = float(details["confidence"])
        elif "result" in raw and isinstance(raw["result"], (int, float)):
            raw_score = float(raw["result"])

        fake_labels = ("ai generated", "digitally edited", "ai edited", "fake", "manipulated")
        real_labels = ("real", "human", "authentic", "original")

        if any(lbl in final_result for lbl in fake_labels):
            prediction = "FAKE"
        elif any(lbl in final_result for lbl in real_labels):
            prediction = "REAL"
        else:
            prediction = "FAKE" if raw_score >= 2.0 else "REAL"

        confidence = min(raw_score / 5.0, 1.0) if raw_score > 0 else 0.0

        return prediction, round(confidence, 4)
