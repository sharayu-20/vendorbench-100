"""Reality Defender - AI content detection client using official SDK."""

from __future__ import annotations

import logging
from typing import Any

from benchmark.models import ImageSample
from benchmark.base_client import BaseClient, TransientAPIError

logger = logging.getLogger(__name__)


class RealityDefenderClient(BaseClient):
    PROVIDER_NAME = "reality_defender"
    TRACK = "commercial_apis"

    def __init__(self, config, **kwargs) -> None:
        super().__init__(**kwargs)
        self._keys = config.keys
        self._key_index = 0

    def _next_key(self) -> str:
        key = self._keys[self._key_index % len(self._keys)]
        self._key_index += 1
        return key

    def _call_api(self, sample: ImageSample) -> dict[str, Any]:
        from realitydefender import RealityDefender

        api_key = self._next_key()
        client = RealityDefender(api_key=api_key)

        try:
            result = client.detect_file(sample.path)
            if isinstance(result, dict):
                return result
            return {"result": str(result)}
        except Exception as exc:
            err_msg = str(exc).lower()
            if "rate" in err_msg or "429" in err_msg or "limit" in err_msg:
                from benchmark.base_client import RateLimitError
                raise RateLimitError(f"Reality Defender rate limit: {exc}") from exc
            if "500" in err_msg or "502" in err_msg or "503" in err_msg:
                raise TransientAPIError(f"Reality Defender server error: {exc}") from exc
            raise

    def _normalize(self, raw: dict[str, Any]) -> tuple[str, float]:
        """Normalize Reality Defender response.

        The SDK returns a result dict with detection scores.
        We look for common keys: 'score', 'fake_probability', 'ai_generated', 'is_fake'.
        """
        confidence = 0.0

        for key in ("fake_probability", "ai_generated", "score", "fake_score"):
            if key in raw:
                confidence = float(raw[key])
                break

        if "result" in raw and isinstance(raw["result"], dict):
            inner = raw["result"]
            for key in ("fake_probability", "ai_generated", "score", "fake_score"):
                if key in inner:
                    confidence = float(inner[key])
                    break

        if "is_fake" in raw:
            is_fake = raw["is_fake"]
            if isinstance(is_fake, bool):
                return ("FAKE" if is_fake else "REAL"), (1.0 if is_fake else 0.0)

        prediction = "FAKE" if confidence >= 0.5 else "REAL"
        return prediction, confidence
