"""Base adapter for the LLM track.

Unlike the commercial-API and open-source tracks (which call a model live inside
``BaseClient.detect``), the vision-LLM results were collected out-of-band — some
via API, some via browser automation (Qwen, Z.ai). Those runs are the source of
truth, so each provider here is an **adapter** that reads its pre-collected dump
and re-emits it in the framework's canonical ``APIResult`` shape at
``logs/llms/<provider_id>/<image_stem>.json`` — the exact schema ``metrics.py``
and the HTML report already consume.

Failure policy: an unparseable / refused response is emitted with
``success=False`` and ``prediction="ERROR"``. ``metrics.py`` then excludes it
from Acc/F1/AUC and counts it under ``failed`` (coverage is surfaced in reports),
identical to the other tracks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.models import APIResult
from benchmark.llms import normalize as N

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "raw_llm_results_zip"


class LLMAdapter:
    """One adapter per vision-LLM provider."""

    PROVIDER_ID: str = ""       # log folder + report id, e.g. "claude_opus48"
    DISPLAY_NAME: str = ""      # human label, e.g. "Claude Opus 4.8"
    MODEL_ID: str = ""          # slug / api model id
    ACCESS: str = "API"         # "API" | "browser"
    RAW: str = ""               # raw dump path relative to RAW_DIR

    def load(self) -> list[APIResult]:  # pragma: no cover - interface
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    def _ok(
        self,
        *,
        gt: str,
        name: str,
        payload: dict[str, Any],
        image_path: str = "",
        latency_ms: float = 0.0,
        sha256: str = "",
        raw: dict[str, Any] | None = None,
    ) -> APIResult:
        ground = N.norm_gt(gt)
        conf = N.p_fake(payload)
        return APIResult(
            provider=self.PROVIDER_ID,
            image_path=image_path or name,
            image_filename=N.canonical_filename(ground, name),
            ground_truth=ground,
            prediction=N.prediction(payload),
            confidence=conf,
            raw_response={
                "model_id": self.MODEL_ID,
                "access": self.ACCESS,
                "verdict": payload,
                **({"source": raw} if raw is not None else {}),
            },
            latency_ms=round(float(latency_ms or 0.0), 2),
            image_sha256=sha256,
            success=True,
        )

    def _fail(
        self,
        *,
        gt: str,
        name: str,
        error: str,
        image_path: str = "",
        latency_ms: float = 0.0,
        raw: dict[str, Any] | None = None,
    ) -> APIResult:
        ground = N.norm_gt(gt)
        return APIResult(
            provider=self.PROVIDER_ID,
            image_path=image_path or name,
            image_filename=N.canonical_filename(ground, name),
            ground_truth=ground,
            prediction="ERROR",
            confidence=0.0,
            raw_response={
                "model_id": self.MODEL_ID,
                "access": self.ACCESS,
                "error": error,
                **({"source": raw} if raw is not None else {}),
            },
            latency_ms=round(float(latency_ms or 0.0), 2),
            success=False,
            error_message=error,
        )

    def raw_path(self) -> Path:
        return RAW_DIR / self.RAW

    def _load_nim_dump(self) -> list[APIResult]:
        """Load an NVIDIA-NIM style dump: {summary, results[{filename,
        ground_truth_category, analysis|null, error}]}. Shared by the two
        open-weight adapters (Llama-4-Maverick, Nemotron-Nano-VL)."""
        import json

        data = json.loads(self.raw_path().read_text(encoding="utf-8"))
        out: list[APIResult] = []
        for rec in data.get("results", []):
            name = rec.get("filename", "")
            gt = rec.get("ground_truth_category") or ""
            payload = rec.get("analysis")
            if not payload:
                out.append(self._fail(
                    gt=gt, name=name, error=rec.get("error") or "no analysis"))
                continue
            out.append(self._ok(gt=gt, name=name, payload=payload))
        return out
