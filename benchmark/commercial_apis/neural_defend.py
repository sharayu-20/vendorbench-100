"""Neural Defend - DeepScan unified endpoint client.

Uses the /test/unified endpoint which runs liveness, faceswap, and deepfake
checks in a single call and returns a prediction_tag for final classification.
"""

from __future__ import annotations

import mimetypes
from typing import Any

from benchmark.models import ImageSample
from benchmark.base_client import BaseClient


class NeuralDefendClient(BaseClient):
    PROVIDER_NAME = "neural_defend"
    TRACK = "commercial_apis"

    def __init__(self, config, **kwargs) -> None:
        super().__init__(**kwargs)
        self._endpoint = config.endpoint
        self._api_key = config.api_key

    def _call_api(self, sample: ImageSample) -> dict[str, Any]:
        mime_type = mimetypes.guess_type(sample.path)[0] or "application/octet-stream"
        with open(sample.path, "rb") as f:
            files = {"file": (sample.filename, f, mime_type)}
            resp = self._session.post(
                self._endpoint,
                headers={"x-api-key": self._api_key},
                files=files,
                timeout=self.timeout,
            )

        self._check_http_status(resp)
        return resp.json()

    def _normalize(self, raw: dict[str, Any]) -> tuple[str, float]:
        """Normalize Neural Defend unified response.

        Response structure:
        {
          "unified_analysis": {
            "prediction_tag": "FACESWAP" | "DEEPFAKE" | "REAL" | ...,
            "faceswap_check": {"confidence": 0.99, "prediction": "..."},
            "deepfake_check": {"confidence": 0.99, "prediction": "..."},
            "liveness_check": {"confidence": 0.95, "prediction": "Real"},
            "face_confidence": 0.745,
            ...
          }
        }

        prediction_tag is the final classification output.
        """
        analysis = raw.get("unified_analysis", raw)

        tag = str(analysis.get("prediction_tag", "")).upper()

        fake_tags = ("FACESWAP", "DEEPFAKE", "FAKE", "MANIPULATED", "AI_GENERATED")
        real_tags = ("REAL", "GENUINE", "AUTHENTIC")

        if tag in fake_tags:
            prediction = "FAKE"
        elif tag in real_tags:
            prediction = "REAL"
        else:
            prediction = "FAKE" if tag and tag not in real_tags else "REAL"

        confidence = 0.0
        faceswap = analysis.get("faceswap_check", {}) or {}
        deepfake = analysis.get("deepfake_check", {}) or {}

        fs_conf = float(faceswap.get("confidence", 0) or 0)
        df_conf = float(deepfake.get("confidence", 0) or 0)
        confidence = max(fs_conf, df_conf)

        if confidence == 0.0 and prediction == "FAKE":
            confidence = float(analysis.get("face_confidence", 0.75))
        elif confidence == 0.0 and prediction == "REAL":
            confidence = 1.0 - float(analysis.get("face_confidence", 0.25))

        return prediction, round(confidence, 4)
