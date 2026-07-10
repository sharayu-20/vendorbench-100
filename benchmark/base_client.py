"""Abstract base client with retry, timeout, and structured evidence logging."""

from __future__ import annotations

import json
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
    before_sleep_log,
)

from benchmark.models import ImageSample, APIResult

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised on 429 / rate-limit responses to trigger tenacity retry."""


class TransientAPIError(Exception):
    """Raised on 5xx responses to trigger tenacity retry."""


class BaseClient(ABC):
    """Abstract base for all API provider clients."""

    PROVIDER_NAME: str = ""
    TRACK: str = ""  # commercial_apis | opensource_models | llms

    def __init__(
        self,
        timeout: int = 120,
        max_retries: int = 3,
        logs_dir: str = "logs",
    ) -> None:
        self.timeout = timeout
        self.max_retries = max_retries
        self.logs_dir = Path(logs_dir) / self.TRACK / self.PROVIDER_NAME
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()

    def detect(self, sample: ImageSample) -> APIResult:
        """Run detection with retry logic and evidence logging."""
        retry_count = 0
        start = time.perf_counter()

        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential_jitter(initial=2, max=30, jitter=2),
            retry=retry_if_exception_type((RateLimitError, TransientAPIError)),
            before_sleep=before_sleep_log(logger, logging.WARNING),
            reraise=True,
        )
        def _call() -> dict[str, Any]:
            nonlocal retry_count
            retry_count += 1
            return self._call_api(sample)

        try:
            raw_response = _call()
            elapsed_ms = (time.perf_counter() - start) * 1000
            prediction, confidence = self._normalize(raw_response)
            result = APIResult(
                provider=self.PROVIDER_NAME,
                image_path=sample.path,
                image_filename=sample.filename,
                ground_truth=sample.ground_truth,
                prediction=prediction,
                confidence=confidence,
                raw_response=raw_response,
                latency_ms=round(elapsed_ms, 2),
                image_sha256=sample.sha256,
                success=True,
                retry_count=max(0, retry_count - 1),
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error("API call failed for %s on %s: %s", self.PROVIDER_NAME, sample.filename, exc)
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
                retry_count=max(0, retry_count - 1),
            )

        self._save_evidence(result)
        return result

    @abstractmethod
    def _call_api(self, sample: ImageSample) -> dict[str, Any]:
        """Execute the raw API call. Must raise RateLimitError or TransientAPIError for retries."""

    @abstractmethod
    def _normalize(self, raw: dict[str, Any]) -> tuple[str, float]:
        """Normalize raw API response to (prediction, confidence).
        prediction: "FAKE" or "REAL"
        confidence: 0.0-1.0 probability of being fake
        """

    def _save_evidence(self, result: APIResult) -> None:
        """Persist individual result as JSON evidence log."""
        stem = Path(result.image_filename).stem
        log_path = self.logs_dir / f"{stem}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        logger.debug("Evidence saved: %s", log_path)

    def _check_http_status(self, resp: requests.Response) -> None:
        """Raise retriable errors for 429 and 5xx status codes."""
        if resp.status_code == 429:
            raise RateLimitError(f"Rate limited (429) from {self.PROVIDER_NAME}")
        if resp.status_code >= 500:
            raise TransientAPIError(f"Server error ({resp.status_code}) from {self.PROVIDER_NAME}")
        resp.raise_for_status()

    def close(self) -> None:
        self._session.close()
