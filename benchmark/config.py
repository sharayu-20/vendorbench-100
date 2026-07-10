"""Configuration loader with validation and API key rotation pools."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@dataclass
class HiveConfig:
    endpoint: str = "https://api.thehive.ai/api/v3/hive/ai-generated-and-deepfake-content-detection"
    keys: list[dict[str, str]] = field(default_factory=list)
    _index: int = field(default=0, repr=False)

    def next_key(self) -> dict[str, str]:
        key = self.keys[self._index % len(self.keys)]
        self._index += 1
        return key


@dataclass
class RealityDefenderConfig:
    keys: list[str] = field(default_factory=list)
    _index: int = field(default=0, repr=False)

    def next_key(self) -> str:
        key = self.keys[self._index % len(self.keys)]
        self._index += 1
        return key


@dataclass
class NeuralDefendConfig:
    endpoint: str = ""
    api_key: str = ""


@dataclass
class SightEngineConfig:
    endpoint: str = "https://api.sightengine.com/1.0/check.json"
    models: str = "genai,deepfake"
    keys: list[dict[str, str]] = field(default_factory=list)
    _index: int = field(default=0, repr=False)

    def next_key(self) -> dict[str, str]:
        key = self.keys[self._index % len(self.keys)]
        self._index += 1
        return key


@dataclass
class TruthScanConfig:
    api_key: str = ""


@dataclass
class BenchmarkSettings:
    request_delay: float = 1.5
    max_retries: int = 3
    request_timeout: int = 120
    dataset_dir: str = ""
    logs_dir: str = ""
    reports_dir: str = ""
    checkpoint_path: str = ""


@dataclass
class AppConfig:
    hive: HiveConfig = field(default_factory=HiveConfig)
    reality_defender: RealityDefenderConfig = field(default_factory=RealityDefenderConfig)
    neural_defend: NeuralDefendConfig = field(default_factory=NeuralDefendConfig)
    sightengine: SightEngineConfig = field(default_factory=SightEngineConfig)
    truthscan: TruthScanConfig = field(default_factory=TruthScanConfig)
    settings: BenchmarkSettings = field(default_factory=BenchmarkSettings)


def _require(key: str) -> str:
    val = os.getenv(key, "").strip()
    if not val:
        print(f"[CONFIG ERROR] Missing required env var: {key}", file=sys.stderr)
        sys.exit(1)
    return val


def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def load_config(env_path: Optional[str] = None) -> AppConfig:
    """Load and validate all configuration from the .env file."""
    root = Path(__file__).resolve().parent.parent
    if env_path is None:
        env_path = str(root / "config.env")

    if not Path(env_path).exists():
        print(f"[CONFIG ERROR] config.env not found at {env_path}", file=sys.stderr)
        sys.exit(1)

    load_dotenv(env_path, override=True)

    hive_keys = []
    for i in range(1, 6):
        ak = _optional(f"HIVE_API_KEY_{i}")
        sk = _optional(f"HIVE_SECRET_KEY_{i}")
        if ak and sk:
            hive_keys.append({"access_key": ak, "secret_key": sk})
    if not hive_keys:
        print("[CONFIG WARNING] No Hive API keys found, Hive client will be skipped", file=sys.stderr)

    rd_keys = []
    for i in range(1, 7):
        k = _optional(f"RD_API_KEY_{i}")
        if k:
            rd_keys.append(k)
    if not rd_keys:
        print("[CONFIG WARNING] No Reality Defender API keys found, RD client will be skipped", file=sys.stderr)

    se_keys = []
    for i in range(1, 4):
        u = _optional(f"SIGHTENGINE_API_USER_{i}")
        s = _optional(f"SIGHTENGINE_API_SECRET_{i}")
        if u and s:
            se_keys.append({"api_user": u, "api_secret": s})
    if not se_keys:
        print("[CONFIG WARNING] No SightEngine keys found, SE client will be skipped", file=sys.stderr)

    return AppConfig(
        hive=HiveConfig(keys=hive_keys),
        reality_defender=RealityDefenderConfig(keys=rd_keys),
        neural_defend=NeuralDefendConfig(
            endpoint=_optional("NEURAL_DEFEND_ENDPOINT", "https://deepscan.neuraldefend.com/test/image"),
            api_key=_optional("NEURAL_DEFEND_API_KEY"),
        ),
        sightengine=SightEngineConfig(keys=se_keys),
        truthscan=TruthScanConfig(api_key=_optional("TRUTHSCAN_API_KEY")),
        settings=BenchmarkSettings(
            request_delay=float(_optional("REQUEST_DELAY_SECONDS", "1.5")),
            max_retries=int(_optional("MAX_RETRIES", "3")),
            request_timeout=int(_optional("REQUEST_TIMEOUT_SECONDS", "120")),
            dataset_dir=str(root / "dataset"),
            logs_dir=str(root / "logs"),
            reports_dir=str(root / "reports"),
            checkpoint_path=str(root / "checkpoint.json"),
        ),
    )
