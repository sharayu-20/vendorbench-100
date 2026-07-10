"""HTML report generator using Jinja2 with Chart.js visualizations."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Template

from benchmark.models import BenchmarkRun, ProviderMetrics
from benchmark.metrics import compute_roc_data

logger = logging.getLogger(__name__)

REPORT_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Detection API Benchmark Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  :root {
    --bg: #f8f9fa; --card: #ffffff; --text: #212529; --muted: #6c757d;
    --border: #dee2e6; --primary: #0d6efd; --success: #198754;
    --danger: #dc3545; --warning: #ffc107;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); line-height: 1.6; }
  .container { max-width: 1400px; margin: 0 auto; padding: 24px; }
  header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); color: #fff; padding: 40px 0; margin-bottom: 32px; }
  header .container { display: flex; flex-direction: column; gap: 8px; }
  header h1 { font-size: 2rem; font-weight: 700; }
  header p { opacity: 0.85; font-size: 1.05rem; }
  .meta-bar { display: flex; gap: 24px; flex-wrap: wrap; margin-top: 12px; font-size: 0.9rem; opacity: 0.75; }
  .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.04); }
  .card h2 { font-size: 1.3rem; margin-bottom: 16px; padding-bottom: 8px; border-bottom: 2px solid var(--primary); }
  .card h3 { font-size: 1.1rem; margin: 16px 0 8px; }
  .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
  .stat-box { text-align: center; padding: 20px; border-radius: 8px; background: var(--bg); }
  .stat-box .value { font-size: 2rem; font-weight: 700; color: var(--primary); }
  .stat-box .label { font-size: 0.85rem; color: var(--muted); margin-top: 4px; }
  .chart-container { position: relative; width: 100%; max-height: 400px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th, td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--border); }
  th { background: var(--bg); font-weight: 600; position: sticky; top: 0; }
  tr:hover { background: #f0f4ff; }
  .badge { display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: 600; }
  .badge-fake { background: #fde8e8; color: var(--danger); }
  .badge-real { background: #d4edda; color: var(--success); }
  .badge-error { background: #fff3cd; color: #856404; }
  .badge-correct { background: #d4edda; color: var(--success); }
  .badge-wrong { background: #fde8e8; color: var(--danger); }
  .cm-grid { display: grid; grid-template-columns: auto 1fr 1fr; grid-template-rows: auto 1fr 1fr; gap: 2px; max-width: 300px; margin: 8px auto; }
  .cm-cell { padding: 16px; text-align: center; font-weight: 700; font-size: 1.1rem; border-radius: 4px; }
  .cm-header { font-size: 0.8rem; font-weight: 600; color: var(--muted); display: flex; align-items: center; justify-content: center; }
  .cm-tp { background: #c3e6cb; color: #155724; }
  .cm-tn { background: #c3e6cb; color: #155724; }
  .cm-fp { background: #f5c6cb; color: #721c24; }
  .cm-fn { background: #f5c6cb; color: #721c24; }
  .methodology { font-size: 0.95rem; }
  .methodology ul { padding-left: 20px; margin: 8px 0; }
  .methodology li { margin: 4px 0; }
  .scrollable-table { max-height: 600px; overflow-y: auto; border: 1px solid var(--border); border-radius: 8px; }
  .filter-bar { display: flex; gap: 12px; margin-bottom: 12px; flex-wrap: wrap; }
  .filter-bar select, .filter-bar input { padding: 6px 12px; border: 1px solid var(--border); border-radius: 6px; font-size: 0.9rem; }
  @media (max-width: 768px) { .grid-2, .grid-3 { grid-template-columns: 1fr; } }
  @media print {
    header { background: #1a1a2e !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
    .card { break-inside: avoid; }
    .no-print { display: none; }
  }
</style>
</head>
<body>

<header>
<div class="container">
  <h1>AI Detection API Benchmark Report</h1>
  <p>Comparative analysis of {{ providers|length }} AI image detection providers across {{ dataset_size }} images</p>
  <div class="meta-bar">
    <span>Run ID: {{ run_id }}</span>
    <span>Date: {{ report_date }}</span>
    <span>Dataset: {{ fake_count }} Fake + {{ real_count }} Real</span>
  </div>
</div>
</header>

<div class="container">

<!-- Executive Summary -->
<div class="card">
  <h2>Executive Summary</h2>
  <div class="grid-3">
    <div class="stat-box">
      <div class="value">{{ dataset_size }}</div>
      <div class="label">Total Images Tested</div>
    </div>
    <div class="stat-box">
      <div class="value">{{ providers|length }}</div>
      <div class="label">API Providers</div>
    </div>
    <div class="stat-box">
      <div class="value">{{ total_api_calls }}</div>
      <div class="label">Total API Calls</div>
    </div>
  </div>
  {% if best_provider %}
  <p style="margin-top:16px; font-size:1rem;">
    <strong>Top Performer:</strong> {{ best_provider.provider }} with <strong>{{ "%.1f"|format(best_provider.f1_score * 100) }}%</strong> F1-Score,
    <strong>{{ "%.1f"|format(best_provider.accuracy * 100) }}%</strong> Accuracy, and
    <strong>{{ "%.1f"|format(best_provider.roc_auc * 100) }}%</strong> ROC-AUC.
  </p>
  {% endif %}
</div>

<!-- Comparison Table -->
<div class="card">
  <h2>Provider Comparison</h2>
  <div class="scrollable-table">
  <table>
    <thead>
      <tr>
        <th>Provider</th><th>Accuracy</th><th>Precision</th><th>Recall</th>
        <th>Specificity</th><th>F1-Score</th><th>MCC</th><th>ROC-AUC</th>
        <th>Avg Latency</th><th>Success Rate</th>
      </tr>
    </thead>
    <tbody>
    {% for pm in provider_metrics %}
      <tr>
        <td><strong>{{ pm.provider }}</strong></td>
        <td>{{ "%.1f%%"|format(pm.accuracy * 100) }}</td>
        <td>{{ "%.1f%%"|format(pm.precision * 100) }}</td>
        <td>{{ "%.1f%%"|format(pm.recall * 100) }}</td>
        <td>{{ "%.1f%%"|format(pm.specificity * 100) }}</td>
        <td>{{ "%.1f%%"|format(pm.f1_score * 100) }}</td>
        <td>{{ "%.3f"|format(pm.mcc) }}</td>
        <td>{{ "%.3f"|format(pm.roc_auc) }}</td>
        <td>{{ "%.0f ms"|format(pm.latency_mean_ms) }}</td>
        <td>{{ "%.0f%%"|format(pm.successful / pm.total_images * 100 if pm.total_images else 0) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  </div>
</div>

<!-- Charts Row -->
<div class="grid-2">

  <!-- F1 / Accuracy Bar Chart -->
  <div class="card">
    <h2>Accuracy, Precision, Recall, F1</h2>
    <div class="chart-container">
      <canvas id="metricsChart"></canvas>
    </div>
  </div>

  <!-- ROC Curves -->
  <div class="card">
    <h2>ROC Curves</h2>
    <div class="chart-container">
      <canvas id="rocChart"></canvas>
    </div>
  </div>

</div>

<!-- Confusion Matrices -->
<div class="card">
  <h2>Confusion Matrices</h2>
  <div class="grid-{{ [providers|length, 3]|min }}" style="gap: 32px;">
  {% for pm in provider_metrics %}
    <div>
      <h3 style="text-align:center;">{{ pm.provider }}</h3>
      <div class="cm-grid">
        <div class="cm-header"></div>
        <div class="cm-header">Pred REAL</div>
        <div class="cm-header">Pred FAKE</div>
        <div class="cm-header">Actual REAL</div>
        <div class="cm-cell cm-tn">{{ pm.true_negatives }}<br><small>TN</small></div>
        <div class="cm-cell cm-fp">{{ pm.false_positives }}<br><small>FP</small></div>
        <div class="cm-header">Actual FAKE</div>
        <div class="cm-cell cm-fn">{{ pm.false_negatives }}<br><small>FN</small></div>
        <div class="cm-cell cm-tp">{{ pm.true_positives }}<br><small>TP</small></div>
      </div>
    </div>
  {% endfor %}
  </div>
</div>

<!-- Latency Comparison -->
<div class="card">
  <h2>Latency Analysis (ms)</h2>
  <table>
    <thead>
      <tr><th>Provider</th><th>Mean</th><th>Median</th><th>P95</th><th>P99</th></tr>
    </thead>
    <tbody>
    {% for pm in provider_metrics %}
      <tr>
        <td><strong>{{ pm.provider }}</strong></td>
        <td>{{ "%.0f"|format(pm.latency_mean_ms) }}</td>
        <td>{{ "%.0f"|format(pm.latency_median_ms) }}</td>
        <td>{{ "%.0f"|format(pm.latency_p95_ms) }}</td>
        <td>{{ "%.0f"|format(pm.latency_p99_ms) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>

<!-- Per-Image Results -->
<div class="card">
  <h2>Per-Image Results</h2>
  <div class="filter-bar no-print">
    <select id="filterProvider" onchange="filterTable()">
      <option value="">All Providers</option>
      {% for p in providers %}<option value="{{ p }}">{{ p }}</option>{% endfor %}
    </select>
    <select id="filterGT" onchange="filterTable()">
      <option value="">All Labels</option>
      <option value="FAKE">FAKE</option>
      <option value="REAL">REAL</option>
    </select>
    <select id="filterCorrect" onchange="filterTable()">
      <option value="">All Results</option>
      <option value="correct">Correct</option>
      <option value="wrong">Wrong</option>
      <option value="error">Error</option>
    </select>
    <input type="text" id="filterFilename" placeholder="Search filename..." oninput="filterTable()">
  </div>
  <div class="scrollable-table">
  <table id="resultsTable">
    <thead>
      <tr>
        <th>Image</th><th>Provider</th><th>Ground Truth</th><th>Prediction</th>
        <th>Confidence</th><th>Result</th><th>Latency (ms)</th>
      </tr>
    </thead>
    <tbody>
    {% for r in results %}
      <tr data-provider="{{ r.provider }}" data-gt="{{ r.ground_truth }}"
          data-correct="{{ 'correct' if r.prediction == r.ground_truth else ('error' if r.prediction == 'ERROR' else 'wrong') }}">
        <td title="{{ r.image_path }}">{{ r.image_filename }}</td>
        <td>{{ r.provider }}</td>
        <td><span class="badge badge-{{ r.ground_truth|lower }}">{{ r.ground_truth }}</span></td>
        <td><span class="badge badge-{{ r.prediction|lower if r.prediction in ['FAKE','REAL'] else 'error' }}">{{ r.prediction }}</span></td>
        <td>{{ "%.2f"|format(r.confidence) }}</td>
        <td>
          {% if r.prediction == 'ERROR' %}<span class="badge badge-error">ERROR</span>
          {% elif r.prediction == r.ground_truth %}<span class="badge badge-correct">CORRECT</span>
          {% else %}<span class="badge badge-wrong">WRONG</span>{% endif %}
        </td>
        <td>{{ "%.0f"|format(r.latency_ms) }}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
  </div>
</div>

<!-- Methodology -->
<div class="card methodology">
  <h2>Methodology</h2>
  <ul>
    <li><strong>Dataset:</strong> {{ fake_count }} fake images (AI-generated, face-swapped, deepfakes) and {{ real_count }} real images.</li>
    <li><strong>Processing:</strong> Images sent one at a time to avoid rate limits; sequential per provider.</li>
    <li><strong>Retry:</strong> Exponential backoff with jitter on 429/5xx errors (max {{ max_retries }} retries).</li>
    <li><strong>Key Rotation:</strong> Multiple API keys rotated per provider to distribute load.</li>
    <li><strong>Threshold:</strong> Confidence &ge; 0.5 classified as FAKE; &lt; 0.5 as REAL.</li>
    <li><strong>Metrics:</strong> scikit-learn v1.4+ for confusion matrix, ROC-AUC, F1, MCC.</li>
    <li><strong>Evidence:</strong> Individual JSON log per image per API stored in <code>logs/</code> directory.</li>
  </ul>
</div>

</div>

<script>
const COLORS = ['#0d6efd','#198754','#dc3545','#ffc107','#6f42c1','#20c997','#fd7e14'];
const providers = {{ providers_json }};
const metricsData = {{ metrics_json }};
const rocData = {{ roc_json }};

// Metrics bar chart
new Chart(document.getElementById('metricsChart'), {
  type: 'bar',
  data: {
    labels: providers,
    datasets: [
      { label: 'Accuracy', data: metricsData.map(m => (m.accuracy*100).toFixed(1)), backgroundColor: COLORS[0]+'cc' },
      { label: 'Precision', data: metricsData.map(m => (m.precision*100).toFixed(1)), backgroundColor: COLORS[1]+'cc' },
      { label: 'Recall', data: metricsData.map(m => (m.recall*100).toFixed(1)), backgroundColor: COLORS[2]+'cc' },
      { label: 'F1-Score', data: metricsData.map(m => (m.f1_score*100).toFixed(1)), backgroundColor: COLORS[3]+'cc' },
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    scales: { y: { beginAtZero: true, max: 100, title: { display: true, text: 'Score (%)' } } },
    plugins: { legend: { position: 'bottom' } }
  }
});

// ROC chart
const rocDatasets = [];
rocData.forEach((r, i) => {
  if (r.fpr.length > 0) {
    rocDatasets.push({
      label: providers[i] + ' (AUC=' + r.auc.toFixed(3) + ')',
      data: r.fpr.map((x, j) => ({ x: x, y: r.tpr[j] })),
      borderColor: COLORS[i % COLORS.length],
      backgroundColor: 'transparent',
      pointRadius: 0, borderWidth: 2, tension: 0.1
    });
  }
});
rocDatasets.push({
  label: 'Random', data: [{x:0,y:0},{x:1,y:1}],
  borderColor: '#ccc', borderDash: [5,5], pointRadius: 0, borderWidth: 1
});
new Chart(document.getElementById('rocChart'), {
  type: 'scatter',
  data: { datasets: rocDatasets },
  options: {
    responsive: true, maintainAspectRatio: false, showLine: true,
    scales: {
      x: { min: 0, max: 1, title: { display: true, text: 'False Positive Rate' } },
      y: { min: 0, max: 1, title: { display: true, text: 'True Positive Rate' } }
    },
    plugins: { legend: { position: 'bottom' } }
  }
});

// Filter table
function filterTable() {
  const prov = document.getElementById('filterProvider').value;
  const gt = document.getElementById('filterGT').value;
  const corr = document.getElementById('filterCorrect').value;
  const fname = document.getElementById('filterFilename').value.toLowerCase();
  document.querySelectorAll('#resultsTable tbody tr').forEach(row => {
    let show = true;
    if (prov && row.dataset.provider !== prov) show = false;
    if (gt && row.dataset.gt !== gt) show = false;
    if (corr && row.dataset.correct !== corr) show = false;
    if (fname && !row.children[0].textContent.toLowerCase().includes(fname)) show = false;
    row.style.display = show ? '' : 'none';
  });
}
</script>
</body>
</html>"""


def generate_html_report(run: BenchmarkRun, output_dir: str) -> str:
    """Generate a self-contained HTML benchmark report."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    providers = [pm.provider for pm in run.provider_metrics]

    metrics_for_json = []
    for pm in run.provider_metrics:
        metrics_for_json.append({
            "accuracy": pm.accuracy,
            "precision": pm.precision,
            "recall": pm.recall,
            "f1_score": pm.f1_score,
            "mcc": pm.mcc,
            "roc_auc": pm.roc_auc,
        })

    roc_for_json = [compute_roc_data(pm) for pm in run.provider_metrics]

    best = max(run.provider_metrics, key=lambda m: m.f1_score) if run.provider_metrics else None

    sorted_results = sorted(run.results, key=lambda r: (r.provider, r.image_filename))

    template = Template(REPORT_TEMPLATE)
    html = template.render(
        run_id=run.run_id,
        report_date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        dataset_size=run.dataset_size,
        fake_count=run.fake_count,
        real_count=run.real_count,
        providers=providers,
        provider_metrics=run.provider_metrics,
        best_provider=best,
        total_api_calls=len(run.results),
        results=sorted_results,
        max_retries=3,
        providers_json=json.dumps(providers),
        metrics_json=json.dumps(metrics_for_json),
        roc_json=json.dumps(roc_for_json),
    )

    report_path = out / "benchmark_report.html"
    report_path.write_text(html, encoding="utf-8")
    logger.info("HTML report generated: %s", report_path)
    return str(report_path)
