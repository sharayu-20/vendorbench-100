"""The Hive AI - AI-Generated and Deepfake Content Detection client."""

from __future__ import annotations

import base64
import json
import mimetypes
from typing import Any

from benchmark.models import ImageSample
from benchmark.base_client import BaseClient, RateLimitError, TransientAPIError


class HiveClient(BaseClient):
    PROVIDER_NAME = "hive"
    TRACK = "commercial_apis"

    def __init__(self, config, **kwargs) -> None:
        super().__init__(**kwargs)
        self._endpoint = config.endpoint
        self._keys = config.keys
        self._key_index = 0

    def _next_key(self) -> dict[str, str]:
        key = self._keys[self._key_index % len(self._keys)]
        self._key_index += 1
        return key

    def _call_api(self, sample: ImageSample) -> dict[str, Any]:
        key = self._next_key()

        mime_type = mimetypes.guess_type(sample.path)[0] or "image/jpeg"
        with open(sample.path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        media_base64 = f"data:{mime_type};base64,{b64}"

        payload = {
            "media_metadata": True,
            "input": [{"media_base64": media_base64}],
        }

        resp = self._session.post(
            self._endpoint,
            headers={
                "Authorization": f"Bearer {key['secret_key']}",
                "Content-Type": "application/json",
            },
            data=json.dumps(payload),
            timeout=self.timeout,
        )
        self._check_http_status(resp)
        return resp.json()

    def _normalize(self, raw: dict[str, Any]) -> tuple[str, float]:
        """Extract AI-generated confidence from Hive response.

        Hive returns output[0].classes[] with entries like:
        - {"class": "ai_generated", "value": 0.99}
        - {"class": "not_ai_generated", "value": 0.01}
        """
        try:
            output = raw["output"][0]
            classes = output.get("classes", [])
            ai_score = 0.0
            for cls in classes:
                if cls.get("class") == "ai_generated":
                    ai_score = float(cls["value"])
                    break
            prediction = "FAKE" if ai_score >= 0.5 else "REAL"
            return prediction, ai_score
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected Hive response structure: {exc}") from exc
