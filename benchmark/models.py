"""Data models for the benchmark framework."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


@dataclass
class ImageSample:
    path: str
    ground_truth: str  # "FAKE" or "REAL"
    sha256: str = ""

    def __post_init__(self) -> None:
        if not self.sha256:
            self.sha256 = self._compute_hash()

    def _compute_hash(self) -> str:
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    @property
    def filename(self) -> str:
        return Path(self.path).name


@dataclass
class APIResult:
    provider: str
    image_path: str
    image_filename: str
    ground_truth: str
    prediction: str          # "FAKE" or "REAL"
    confidence: float        # 0.0-1.0 probability of being fake/AI-generated
    raw_response: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    timestamp: str = ""
    image_sha256: str = ""
    success: bool = True
    error_message: Optional[str] = None
    retry_count: int = 0

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProviderMetrics:
    provider: str
    total_images: int = 0
    successful: int = 0
    failed: int = 0
    true_positives: int = 0   # correctly predicted FAKE
    false_positives: int = 0  # predicted FAKE but was REAL
    true_negatives: int = 0   # correctly predicted REAL
    false_negatives: int = 0  # predicted REAL but was FAKE
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    specificity: float = 0.0
    f1_score: float = 0.0
    mcc: float = 0.0
    roc_auc: float = 0.0
    latency_mean_ms: float = 0.0
    latency_median_ms: float = 0.0
    latency_p95_ms: float = 0.0
    latency_p99_ms: float = 0.0
    confidences: list[float] = field(default_factory=list)
    ground_truths: list[int] = field(default_factory=list)  # 1=FAKE, 0=REAL


@dataclass
class BenchmarkRun:
    run_id: str = ""
    start_time: str = ""
    end_time: str = ""
    dataset_size: int = 0
    fake_count: int = 0
    real_count: int = 0
    providers: list[str] = field(default_factory=list)
    results: list[APIResult] = field(default_factory=list)
    provider_metrics: list[ProviderMetrics] = field(default_factory=list)
