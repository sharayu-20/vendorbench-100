"""SightEngine - AI-Generated Image + Deepfake Detection client.

Uses models=genai,deepfake in a single call to detect both fully
AI-generated images and face-swapped deepfakes. Confidence is
max(ai_generated, deepfake) to catch both manipulation types.
"""

from __future__ import annotations

from typing import Any

from benchmark.models import ImageSample
from benchmark.base_client import BaseClient


class SightEngineClient(BaseClient):
    PROVIDER_NAME = "sightengine"
    TRACK = "commercial_apis"

    def __init__(self, config, **kwargs) -> None:
        super().__init__(**kwargs)
        self._endpoint = config.endpoint
        self._models = config.models
        self._keys = config.keys
        self._key_index = 0

    def _next_key(self) -> dict[str, str]:
        key = self._keys[self._key_index % len(self._keys)]
        self._key_index += 1
        return key

    def _call_api(self, sample: ImageSample) -> dict[str, Any]:
        key = self._next_key()
        with open(sample.path, "rb") as f:
            resp = self._session.post(
                self._endpoint,
                files={"media": (sample.filename, f)},
                data={
                    "models": self._models,
                    "api_user": key["api_user"],
                    "api_secret": key["api_secret"],
                },
                timeout=self.timeout,
            )

        self._check_http_status(resp)
        return resp.json()

    def _normalize(self, raw: dict[str, Any]) -> tuple[str, float]:
        """Normalize SightEngine response.

        With models=genai,deepfake the response contains:
          {
            "type": {
              "ai_generated": 0.99,
              "deepfake": 0.01
            },
            ...
          }

        We use max(ai_generated, deepfake) as the final confidence
        so both fully AI-generated images and face-swapped deepfakes
        are caught.
        """
        type_scores = raw.get("type", {})
        ai_gen = float(type_scores.get("ai_generated", 0.0))
        deepfake = float(type_scores.get("deepfake", 0.0))

        confidence = max(ai_gen, deepfake)
        prediction = "FAKE" if confidence >= 0.5 else "REAL"
        return prediction, confidence
