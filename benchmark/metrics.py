"""Metrics computation: confusion matrix, accuracy, precision, recall, F1, MCC, ROC-AUC, latency stats."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from benchmark.models import APIResult, ProviderMetrics, BenchmarkRun

logger = logging.getLogger(__name__)


def _to_binary(label: str) -> int:
    """FAKE=1 (positive), REAL=0 (negative)."""
    return 1 if label == "FAKE" else 0


def compute_provider_metrics(provider: str, results: list[APIResult]) -> ProviderMetrics:
    """Calculate all classification and latency metrics for a single provider."""
    successful = [r for r in results if r.success and r.prediction in ("FAKE", "REAL")]
    failed = [r for r in results if not r.success]

    metrics = ProviderMetrics(
        provider=provider,
        total_images=len(results),
        successful=len(successful),
        failed=len(failed),
    )

    if not successful:
        logger.warning("No successful results for %s, skipping metrics", provider)
        return metrics

    y_true = np.array([_to_binary(r.ground_truth) for r in successful])
    y_pred = np.array([_to_binary(r.prediction) for r in successful])
    y_conf = np.array([r.confidence for r in successful])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    metrics.true_positives = int(tp)
    metrics.false_positives = int(fp)
    metrics.true_negatives = int(tn)
    metrics.false_negatives = int(fn)

    metrics.accuracy = float(accuracy_score(y_true, y_pred))
    metrics.precision = float(precision_score(y_true, y_pred, zero_division=0))
    metrics.recall = float(recall_score(y_true, y_pred, zero_division=0))
    metrics.f1_score = float(f1_score(y_true, y_pred, zero_division=0))
    metrics.mcc = float(matthews_corrcoef(y_true, y_pred))

    if tn + fp > 0:
        metrics.specificity = float(tn / (tn + fp))

    if len(np.unique(y_true)) == 2:
        try:
            metrics.roc_auc = float(roc_auc_score(y_true, y_conf))
        except ValueError:
            metrics.roc_auc = 0.0
    else:
        metrics.roc_auc = 0.0

    metrics.confidences = y_conf.tolist()
    metrics.ground_truths = y_true.tolist()

    latencies = np.array([r.latency_ms for r in successful])
    metrics.latency_mean_ms = float(np.mean(latencies))
    metrics.latency_median_ms = float(np.median(latencies))
    metrics.latency_p95_ms = float(np.percentile(latencies, 95))
    metrics.latency_p99_ms = float(np.percentile(latencies, 99))

    return metrics


def compute_roc_data(metrics: ProviderMetrics) -> dict[str, Any]:
    """Compute ROC curve points for a provider."""
    if not metrics.confidences or not metrics.ground_truths:
        return {"fpr": [], "tpr": [], "auc": 0.0}

    y_true = np.array(metrics.ground_truths)
    y_scores = np.array(metrics.confidences)

    if len(np.unique(y_true)) < 2:
        return {"fpr": [], "tpr": [], "auc": 0.0}

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": metrics.roc_auc,
    }


def compute_all_metrics(run: BenchmarkRun) -> BenchmarkRun:
    """Compute metrics for all providers and attach to the run."""
    provider_results: dict[str, list[APIResult]] = {}
    for result in run.results:
        provider_results.setdefault(result.provider, []).append(result)

    run.provider_metrics = []
    for provider, results in provider_results.items():
        pm = compute_provider_metrics(provider, results)
        run.provider_metrics.append(pm)
        logger.info(
            "%s: Acc=%.3f Prec=%.3f Rec=%.3f F1=%.3f MCC=%.3f AUC=%.3f",
            provider, pm.accuracy, pm.precision, pm.recall,
            pm.f1_score, pm.mcc, pm.roc_auc,
        )

    return run
