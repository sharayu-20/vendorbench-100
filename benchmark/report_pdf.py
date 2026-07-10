"""PDF report generator using matplotlib for static charts and weasyprint for rendering."""

from __future__ import annotations

import io
import base64
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np
from jinja2 import Template

from benchmark.models import BenchmarkRun, ProviderMetrics
from benchmark.metrics import compute_roc_data

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="muted")


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _make_bar_chart(metrics_list: list[ProviderMetrics]) -> str:
    providers = [m.provider for m in metrics_list]
    x = np.arange(len(providers))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5*width, [m.accuracy*100 for m in metrics_list], width, label="Accuracy")
    ax.bar(x - 0.5*width, [m.precision*100 for m in metrics_list], width, label="Precision")
    ax.bar(x + 0.5*width, [m.recall*100 for m in metrics_list], width, label="Recall")
    ax.bar(x + 1.5*width, [m.f1_score*100 for m in metrics_list], width, label="F1-Score")

    ax.set_ylabel("Score (%)")
    ax.set_title("Provider Comparison: Classification Metrics")
    ax.set_xticks(x)
    ax.set_xticklabels(providers, rotation=15, ha="right")
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    return _fig_to_base64(fig)


def _make_roc_chart(metrics_list: list[ProviderMetrics]) -> str:
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = plt.cm.tab10.colors

    for i, pm in enumerate(metrics_list):
        roc = compute_roc_data(pm)
        if roc["fpr"]:
            ax.plot(roc["fpr"], roc["tpr"],
                    color=colors[i % len(colors)],
                    lw=2, label=f"{pm.provider} (AUC={roc['auc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    return _fig_to_base64(fig)


def _make_confusion_matrices(metrics_list: list[ProviderMetrics]) -> str:
    n = len(metrics_list)
    if n == 0:
        return ""
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4.5*rows), squeeze=False)

    for idx, pm in enumerate(metrics_list):
        r, c = divmod(idx, cols)
        ax = axes[r][c]
        cm = np.array([[pm.true_negatives, pm.false_positives],
                       [pm.false_negatives, pm.true_positives]])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Pred REAL", "Pred FAKE"],
                    yticklabels=["Actual REAL", "Actual FAKE"],
                    cbar=False, linewidths=1, linecolor="white")
        ax.set_title(pm.provider, fontsize=12, fontweight="bold")

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.tight_layout(pad=2)
    return _fig_to_base64(fig)


def _make_latency_chart(metrics_list: list[ProviderMetrics]) -> str:
    providers = [m.provider for m in metrics_list]
    fig, ax = plt.subplots(figsize=(10, 4))

    x = np.arange(len(providers))
    width = 0.2
    ax.bar(x - width, [m.latency_mean_ms for m in metrics_list], width, label="Mean")
    ax.bar(x, [m.latency_median_ms for m in metrics_list], width, label="Median")
    ax.bar(x + width, [m.latency_p95_ms for m in metrics_list], width, label="P95")

    ax.set_ylabel("Latency (ms)")
    ax.set_title("API Response Latency")
    ax.set_xticks(x)
    ax.set_xticklabels(providers, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    return _fig_to_base64(fig)


PDF_TEMPLATE = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<style>
  @page { size: A4 landscape; margin: 1.5cm; }
  body { font-family: 'Segoe UI', Arial, sans-serif; font-size: 11pt; color: #212529; line-height: 1.5; }
  h1 { font-size: 22pt; color: #1a1a2e; margin-bottom: 4px; }
  h2 { font-size: 14pt; color: #0f3460; margin-top: 24px; border-bottom: 2px solid #0d6efd; padding-bottom: 4px; page-break-after: avoid; }
  h3 { font-size: 12pt; margin-top: 16px; }
  .subtitle { font-size: 11pt; color: #6c757d; margin-bottom: 16px; }
  table { width: 100%; border-collapse: collapse; font-size: 9pt; margin: 8px 0 16px; }
  th, td { padding: 6px 8px; border: 1px solid #dee2e6; text-align: left; }
  th { background: #f0f4ff; font-weight: 600; }
  .chart-img { max-width: 100%; height: auto; margin: 8px 0; }
  .page-break { page-break-before: always; }
  .badge { padding: 1px 6px; border-radius: 8px; font-size: 8pt; font-weight: 600; }
  .badge-fake { background: #fde8e8; color: #dc3545; }
  .badge-real { background: #d4edda; color: #198754; }
  .badge-correct { background: #d4edda; color: #198754; }
  .badge-wrong { background: #fde8e8; color: #dc3545; }
  .badge-error { background: #fff3cd; color: #856404; }
  .summary-grid { display: flex; gap: 16px; margin: 12px 0; }
  .summary-box { flex: 1; text-align: center; background: #f0f4ff; border-radius: 8px; padding: 12px; }
  .summary-box .val { font-size: 18pt; font-weight: 700; color: #0d6efd; }
  .summary-box .lbl { font-size: 9pt; color: #6c757d; }
</style>
</head>
<body>

<h1>AI Detection API Benchmark Report</h1>
<p class="subtitle">{{ report_date }} | Run ID: {{ run_id }} | {{ dataset_size }} images ({{ fake_count }} fake, {{ real_count }} real)</p>

<div class="summary-grid">
  <div class="summary-box"><div class="val">{{ dataset_size }}</div><div class="lbl">Images</div></div>
  <div class="summary-box"><div class="val">{{ num_providers }}</div><div class="lbl">Providers</div></div>
  <div class="summary-box"><div class="val">{{ total_calls }}</div><div class="lbl">API Calls</div></div>
  {% if best %}<div class="summary-box"><div class="val">{{ "%.1f%%"|format(best.f1_score*100) }}</div><div class="lbl">Best F1 ({{ best.provider }})</div></div>{% endif %}
</div>

<h2>Provider Comparison</h2>
<table>
  <tr><th>Provider</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>Specificity</th><th>F1</th><th>MCC</th><th>AUC</th><th>Avg Latency</th></tr>
  {% for pm in provider_metrics %}
  <tr>
    <td><strong>{{ pm.provider }}</strong></td>
    <td>{{ "%.1f%%"|format(pm.accuracy*100) }}</td>
    <td>{{ "%.1f%%"|format(pm.precision*100) }}</td>
    <td>{{ "%.1f%%"|format(pm.recall*100) }}</td>
    <td>{{ "%.1f%%"|format(pm.specificity*100) }}</td>
    <td>{{ "%.1f%%"|format(pm.f1_score*100) }}</td>
    <td>{{ "%.3f"|format(pm.mcc) }}</td>
    <td>{{ "%.3f"|format(pm.roc_auc) }}</td>
    <td>{{ "%.0f ms"|format(pm.latency_mean_ms) }}</td>
  </tr>
  {% endfor %}
</table>

<h2>Classification Metrics</h2>
<img class="chart-img" src="data:image/png;base64,{{ bar_chart }}">

<div class="page-break"></div>

<h2>ROC Curves</h2>
<img class="chart-img" src="data:image/png;base64,{{ roc_chart }}" style="max-width: 70%;">

<h2>Confusion Matrices</h2>
<img class="chart-img" src="data:image/png;base64,{{ cm_chart }}">

<div class="page-break"></div>

<h2>Latency Analysis</h2>
<img class="chart-img" src="data:image/png;base64,{{ latency_chart }}">

<h2>Per-Image Results</h2>
<table>
  <tr><th>Image</th><th>Provider</th><th>Truth</th><th>Prediction</th><th>Confidence</th><th>Result</th><th>Latency</th></tr>
  {% for r in results %}
  <tr>
    <td>{{ r.image_filename }}</td>
    <td>{{ r.provider }}</td>
    <td><span class="badge badge-{{ r.ground_truth|lower }}">{{ r.ground_truth }}</span></td>
    <td><span class="badge badge-{{ r.prediction|lower if r.prediction in ['FAKE','REAL'] else 'error' }}">{{ r.prediction }}</span></td>
    <td>{{ "%.2f"|format(r.confidence) }}</td>
    <td>{% if r.prediction == 'ERROR' %}<span class="badge badge-error">ERR</span>{% elif r.prediction == r.ground_truth %}<span class="badge badge-correct">OK</span>{% else %}<span class="badge badge-wrong">MISS</span>{% endif %}</td>
    <td>{{ "%.0f"|format(r.latency_ms) }}</td>
  </tr>
  {% endfor %}
</table>

<div class="page-break"></div>
<h2>Methodology</h2>
<ul>
  <li><strong>Dataset:</strong> {{ fake_count }} fake images (AI-generated, face-swapped, deepfakes) and {{ real_count }} real images.</li>
  <li><strong>Processing:</strong> Images sent one at a time to each provider sequentially.</li>
  <li><strong>Retry:</strong> Exponential backoff with jitter on 429/5xx errors.</li>
  <li><strong>Key Rotation:</strong> Multiple API keys rotated per provider to distribute load.</li>
  <li><strong>Threshold:</strong> Confidence &ge; 0.5 classified as FAKE; &lt; 0.5 as REAL.</li>
  <li><strong>Metrics:</strong> scikit-learn for confusion matrix, ROC-AUC, F1, MCC.</li>
  <li><strong>Evidence:</strong> Individual JSON log per image per API stored in logs/ directory.</li>
</ul>

</body>
</html>"""


def generate_pdf_report(run: BenchmarkRun, output_dir: str) -> str:
    """Generate a PDF benchmark report with matplotlib charts."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    bar_chart = _make_bar_chart(run.provider_metrics)
    roc_chart = _make_roc_chart(run.provider_metrics)
    cm_chart = _make_confusion_matrices(run.provider_metrics)
    latency_chart = _make_latency_chart(run.provider_metrics)

    best = max(run.provider_metrics, key=lambda m: m.f1_score) if run.provider_metrics else None
    sorted_results = sorted(run.results, key=lambda r: (r.provider, r.image_filename))
    from datetime import datetime, timezone
    report_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    template = Template(PDF_TEMPLATE)
    html = template.render(
        run_id=run.run_id,
        report_date=report_date,
        dataset_size=run.dataset_size,
        fake_count=run.fake_count,
        real_count=run.real_count,
        num_providers=len(run.provider_metrics),
        total_calls=len(run.results),
        best=best,
        provider_metrics=run.provider_metrics,
        results=sorted_results,
        bar_chart=bar_chart,
        roc_chart=roc_chart,
        cm_chart=cm_chart,
        latency_chart=latency_chart,
    )

    pdf_path = out / "benchmark_report.pdf"
    try:
        from weasyprint import HTML
        HTML(string=html).write_pdf(str(pdf_path))
        logger.info("PDF report generated: %s", pdf_path)
    except ImportError:
        logger.warning("weasyprint not installed, saving HTML-for-PDF instead")
        fallback = out / "benchmark_report_for_pdf.html"
        fallback.write_text(html, encoding="utf-8")
        return str(fallback)
    except Exception as exc:
        logger.error("PDF generation failed: %s. Saving HTML fallback.", exc)
        fallback = out / "benchmark_report_for_pdf.html"
        fallback.write_text(html, encoding="utf-8")
        return str(fallback)

    return str(pdf_path)
