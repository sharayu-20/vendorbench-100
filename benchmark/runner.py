"""Benchmark orchestrator with parallel providers, sequential images, checkpoint/resume."""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

from benchmark.config import AppConfig
from benchmark.models import ImageSample, APIResult, BenchmarkRun
from benchmark.base_client import BaseClient
from benchmark.commercial_apis.hive import HiveClient
from benchmark.commercial_apis.reality_defender import RealityDefenderClient
from benchmark.commercial_apis.neural_defend import NeuralDefendClient
from benchmark.commercial_apis.sightengine import SightEngineClient
from benchmark.commercial_apis.truthscan import TruthScanClient

logger = logging.getLogger(__name__)


def scan_dataset(dataset_dir: str) -> list[ImageSample]:
    """Scan dataset directory for FAKE and REAL image samples."""
    samples: list[ImageSample] = []
    base = Path(dataset_dir)

    for label_dir in ("FAKE", "REAL"):
        folder = base / label_dir
        if not folder.exists():
            logger.warning("Dataset folder not found: %s", folder)
            continue

        for img_path in sorted(folder.iterdir()):
            if img_path.is_file() and img_path.suffix.lower() in (
                ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".gif",
            ):
                samples.append(ImageSample(
                    path=str(img_path),
                    ground_truth=label_dir,
                ))

    logger.info("Dataset scan complete: %d images (%d FAKE, %d REAL)",
                len(samples),
                sum(1 for s in samples if s.ground_truth == "FAKE"),
                sum(1 for s in samples if s.ground_truth == "REAL"))
    return samples


def _build_clients(config: AppConfig) -> list[BaseClient]:
    """Instantiate only the clients that have valid credentials."""
    clients: list[BaseClient] = []
    common = {
        "timeout": config.settings.request_timeout,
        "max_retries": config.settings.max_retries,
        "logs_dir": config.settings.logs_dir,
    }

    if config.hive.keys:
        clients.append(HiveClient(config.hive, **common))
    else:
        logger.warning("Skipping Hive: no API keys configured")

    if config.reality_defender.keys:
        clients.append(RealityDefenderClient(config.reality_defender, **common))
    else:
        logger.warning("Skipping Reality Defender: no API keys configured")

    if config.neural_defend.api_key:
        clients.append(NeuralDefendClient(config.neural_defend, **common))
    else:
        logger.warning("Skipping Neural Defend: no API key configured")

    if config.sightengine.keys:
        clients.append(SightEngineClient(config.sightengine, **common))
    else:
        logger.warning("Skipping SightEngine: no API keys configured")

    if config.truthscan.api_key:
        clients.append(TruthScanClient(config.truthscan, **common))
    else:
        logger.warning("Skipping TruthScan: no API key configured")

    return clients


class CheckpointManager:
    """Thread-safe checkpoint state for resume capability."""

    def __init__(self, path: str) -> None:
        self._path = Path(path)
        self._completed: set[str] = set()
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._completed = set(data.get("completed", []))
            logger.info("Checkpoint loaded: %d completed entries", len(self._completed))

    def _key(self, provider: str, image_path: str) -> str:
        return f"{provider}::{Path(image_path).name}"

    def is_done(self, provider: str, image_path: str) -> bool:
        return self._key(provider, image_path) in self._completed

    def mark_done(self, provider: str, image_path: str) -> None:
        with self._lock:
            self._completed.add(self._key(provider, image_path))
            self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump({"completed": sorted(self._completed)}, f, indent=2)


def _has_evidence_logs(logs_dir: str, track: str, provider: str, sample_count: int) -> bool:
    """Check if a provider already has all evidence logs (completed in prior run)."""
    provider_dir = Path(logs_dir) / track / provider
    if not provider_dir.exists():
        return False
    log_count = len(list(provider_dir.glob("*.json")))
    return log_count >= sample_count


def _run_provider(
    client: BaseClient,
    samples: list[ImageSample],
    checkpoint: CheckpointManager,
    delay: float,
    results_lock: threading.Lock,
    all_results: list[APIResult],
) -> None:
    """Run a single provider against all images (one at a time). Runs in its own thread."""
    provider = client.PROVIDER_NAME
    skipped = 0

    progress = tqdm(
        samples,
        desc=f"  {provider:20s}",
        unit="img",
        ncols=100,
        position=None,
    )

    for sample in progress:
        if checkpoint.is_done(provider, sample.path):
            skipped += 1
            progress.set_postfix(skip=skipped)
            continue

        result = client.detect(sample)

        with results_lock:
            all_results.append(result)

        status = "OK" if result.success else "ERR"
        progress.set_postfix(
            s=status,
            p=result.prediction[:1],
            c=f"{result.confidence:.2f}",
        )

        checkpoint.mark_done(provider, sample.path)
        time.sleep(delay)

    if skipped:
        logger.info("%s: skipped %d already-completed images (resume)", provider, skipped)

    client.close()
    print(f"  [DONE] {provider} finished ({len(samples) - skipped} processed, {skipped} skipped)")


def run_benchmark(config: AppConfig, resume: bool = True) -> BenchmarkRun:
    """Execute the full benchmark: all providers run in parallel, each processes images sequentially."""
    run = BenchmarkRun(
        run_id=str(uuid.uuid4())[:8],
        start_time=datetime.now(timezone.utc).isoformat(),
    )

    samples = scan_dataset(config.settings.dataset_dir)
    run.dataset_size = len(samples)
    run.fake_count = sum(1 for s in samples if s.ground_truth == "FAKE")
    run.real_count = sum(1 for s in samples if s.ground_truth == "REAL")

    clients = _build_clients(config)
    run.providers = [c.PROVIDER_NAME for c in clients]

    checkpoint = CheckpointManager(config.settings.checkpoint_path)

    if resume:
        active_clients = []
        for c in clients:
            if _has_evidence_logs(config.settings.logs_dir, c.TRACK, c.PROVIDER_NAME, len(samples)):
                print(f"  [SKIP] {c.PROVIDER_NAME}: all {len(samples)} evidence logs found, skipping")
            else:
                active_clients.append(c)
    else:
        active_clients = clients

    total_calls = len(active_clients) * len(samples)
    logger.info("Starting benchmark: %d providers in PARALLEL x %d images = up to %d API calls",
                len(active_clients), len(samples), total_calls)

    print(f"\n  Running {len(active_clients)} providers in parallel (1 image at a time per provider)")
    print(f"  Delay between requests: {config.settings.request_delay}s\n")

    all_results: list[APIResult] = []
    results_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=len(active_clients)) as executor:
        futures = []
        for client in active_clients:
            fut = executor.submit(
                _run_provider,
                client, samples, checkpoint,
                config.settings.request_delay,
                results_lock, all_results,
            )
            futures.append((client.PROVIDER_NAME, fut))

        for provider_name, fut in futures:
            try:
                fut.result()
            except Exception as exc:
                logger.error("Provider %s crashed: %s", provider_name, exc)

    # Recover results from evidence logs (for skipped/resumed providers)
    skipped_clients = [c for c in clients if c not in active_clients]
    if skipped_clients or resume:
        all_results.extend(_load_evidence_results(
            config.settings.logs_dir, clients, samples, all_results,
        ))

    run.results = all_results
    run.end_time = datetime.now(timezone.utc).isoformat()
    return run


def _load_evidence_results(
    logs_dir: str,
    clients: list[BaseClient],
    samples: list[ImageSample],
    already_collected: list[APIResult],
) -> list[APIResult]:
    """Reload results from evidence log files for resumed runs."""
    existing_keys = {
        f"{r.provider}::{r.image_filename}" for r in already_collected
    }
    recovered: list[APIResult] = []
    logs_base = Path(logs_dir)

    for client in clients:
        provider_dir = logs_base / client.TRACK / client.PROVIDER_NAME
        if not provider_dir.exists():
            continue
        for log_file in provider_dir.glob("*.json"):
            try:
                with open(log_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                key = f"{data['provider']}::{data['image_filename']}"
                if key not in existing_keys:
                    recovered.append(APIResult(**{
                        k: v for k, v in data.items()
                        if k in APIResult.__dataclass_fields__
                    }))
                    existing_keys.add(key)
            except (json.JSONDecodeError, KeyError, TypeError):
                continue

    if recovered:
        logger.info("Recovered %d results from evidence logs", len(recovered))
    return recovered
