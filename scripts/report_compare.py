#!/usr/bin/env python3
"""Cross-track comparative report: commercial APIs vs LLMs vs open-source models.

Reads only the three published, git-tracked summaries

    results/commercial-benchmark/summary.json
    results/llm-benchmark/summary.json
    results/opensource-benchmark/summary.json

(no recomputation, no per-image reprocessing) and emits into
``reports/comparative-benchmark/``:

    COMPARISON_REPORT.md     unified leaderboard + per-track aggregates + findings
    comparison_report.html   self-contained (inline SVG charts, no CDN) leaderboard

All three tracks were scored on the same 100-image corpus (79 fake / 21 real)
with the same metric schema and the same abstention policy (failed calls are
excluded from Accuracy/F1/AUC and reported via coverage), so the numbers are
directly comparable. Models are ranked by MCC (operating-point quality), with
ROC-AUC as the tiebreaker.
"""
from __future__ import annotations

import html
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = ROOT / "reports" / "comparative-benchmark"

# track key in summary.json -> (short label, css class, colour)
TRACKS = [
    ("commercial-benchmark", "commercial_apis", "Commercial API", "commercial", "#6366f1"),
    ("llm-benchmark", "llms", "LLM", "llm", "#ec4899"),
    ("opensource-benchmark", "opensource_models", "Open-source", "opensource", "#10b981"),
]

METRIC_COLS = [
    ("mcc", "MCC", 3),
    ("roc_auc", "ROC-AUC", 3),
    ("accuracy", "Accuracy", 3),
    ("f1", "F1", 3),
    ("precision", "Precision", 3),
    ("recall", "Recall", 3),
    ("specificity", "Specificity", 3),
    ("coverage", "Coverage", 2),
]


def load_rows() -> tuple[list[dict], dict]:
    rows: list[dict] = []
    meta: dict = {}
    for folder, track_key, label, css, colour in TRACKS:
        path = RESULTS / folder / "summary.json"
        if not path.exists():
            print(f"WARNING: missing {path}, skipping")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        meta.setdefault("dataset_size", data.get("dataset_size"))
        meta.setdefault("fake", data.get("fake"))
        meta.setdefault("real", data.get("real"))
        for model, m in data["providers"].items():
            row = dict(m)
            row["model"] = model
            row["track_key"] = track_key
            row["track_label"] = label
            row["track_css"] = css
            row["track_colour"] = colour
            rows.append(row)
    # rank by MCC desc, tiebreak ROC-AUC desc
    rows.sort(key=lambda r: (r.get("mcc", -9), r.get("roc_auc", 0)), reverse=True)
    for i, r in enumerate(rows, start=1):
        r["rank"] = i
    return rows, meta


def track_aggregates(rows: list[dict]) -> list[dict]:
    aggs = []
    for _folder, track_key, label, css, colour in TRACKS:
        trows = [r for r in rows if r["track_key"] == track_key]
        if not trows:
            continue
        best_mcc = max(trows, key=lambda r: r["mcc"])
        best_auc = max(trows, key=lambda r: r["roc_auc"])
        aggs.append({
            "track_key": track_key,
            "label": label,
            "css": css,
            "colour": colour,
            "count": len(trows),
            "best_mcc_val": best_mcc["mcc"],
            "best_mcc_model": best_mcc["model"],
            "median_mcc": statistics.median(r["mcc"] for r in trows),
            "best_auc_val": best_auc["roc_auc"],
            "best_auc_model": best_auc["model"],
            "median_auc": statistics.median(r["roc_auc"] for r in trows),
            "mean_coverage": statistics.mean(r["coverage"] for r in trows),
        })
    return aggs


def commercial_latency_rows(rows: list[dict]) -> list[dict]:
    """Commercial-API rows that have a measured latency, fastest first."""
    crows = [r for r in rows if r["track_key"] == "commercial_apis" and r.get("latency_mean_ms")]
    return sorted(crows, key=lambda r: r["latency_mean_ms"])


def fmt(v, nd) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def lat(v) -> str:
    if not v:  # 0.0 or None -> not measured (e.g. browser-collected LLMs)
        return "n/a"
    if v >= 1000:
        return f"{v/1000:.1f} s"
    return f"{v:.0f} ms"


def findings(rows: list[dict], aggs: list[dict]) -> list[str]:
    out = []
    top = rows[0]
    top_auc = max(rows, key=lambda r: r["roc_auc"])
    out.append(
        f"**Best operating point (MCC):** `{top['model']}` ({top['track_label']}) at "
        f"MCC {top['mcc']:.3f}, accuracy {top['accuracy']:.3f}, F1 {top['f1']:.3f}."
    )
    out.append(
        f"**Best ranking power (ROC-AUC):** `{top_auc['model']}` ({top_auc['track_label']}) at "
        f"AUC {top_auc['roc_auc']:.3f}."
    )
    # AUC vs operating-point divergence
    high_auc_low_mcc = [r for r in rows if r["roc_auc"] >= 0.80 and r["mcc"] <= 0.10]
    if high_auc_low_mcc:
        names = ", ".join(f"`{r['model']}` (AUC {r['roc_auc']:.2f}, MCC {r['mcc']:.2f})" for r in high_auc_low_mcc)
        out.append(
            "**Strong separability, bad default threshold:** " + names +
            " — these rank fakes above reals well but their shipped decision threshold is mis-calibrated "
            "(e.g. flagging almost everything fake), so accuracy/MCC collapse."
        )
    high_mcc_low_auc = [r for r in rows if r["mcc"] >= 0.55 and r["roc_auc"] < 0.60]
    if high_mcc_low_auc:
        names = ", ".join(f"`{r['model']}` (MCC {r['mcc']:.2f}, AUC {r['roc_auc']:.2f})" for r in high_mcc_low_auc)
        out.append(
            "**Good decisions, uninformative scores:** " + names +
            " — the hard label is accurate but the confidence score barely separates classes, so AUC looks weak. "
            "Reads as a near-binary output rather than a calibrated probability."
        )
    # per-track best line
    parts = [f"{a['label']} `{a['best_mcc_model']}` (MCC {a['best_mcc_val']:.3f})" for a in aggs]
    out.append("**Best per track (MCC):** " + "; ".join(parts) + ".")
    # abstentions
    abst = [r for r in rows if r["coverage"] < 1.0]
    if abst:
        names = ", ".join(f"`{r['model']}` ({r['coverage']*100:.0f}%)" for r in abst)
        out.append("**Below-full coverage (abstentions/failures, excluded from Acc/F1/AUC):** " + names + ".")
    # track medians
    med = "; ".join(f"{a['label']} {a['median_mcc']:.3f}" for a in aggs)
    out.append("**Median MCC by track:** " + med + ".")
    return out


# ---------------------------------------------------------------- Markdown ----

def build_md(rows: list[dict], aggs: list[dict], meta: dict) -> str:
    ds, fk, rl = meta["dataset_size"], meta["fake"], meta["real"]
    L = []
    L.append("# Comparative Benchmark — Commercial APIs vs LLMs vs Open-source")
    L.append("")
    L.append(
        f"All three tracks scored on the **same {ds}-image corpus** ({fk} fake / {rl} real), "
        f"identical metric schema, identical abstention policy. **{len(rows)} models total** — "
        f"{aggs_count(aggs, 'Commercial API')} commercial APIs, {aggs_count(aggs, 'LLM')} LLMs, "
        f"{aggs_count(aggs, 'Open-source')} open-source detectors. Ranked by **MCC** "
        "(operating-point quality on an imbalanced set), tiebreak **ROC-AUC**."
    )
    L.append("")
    L.append(
        "> **No label leakage:** models saw neutral numeric filenames (`001.jpg`, …) split only by "
        "folder; `fake_*`/`real_*` ids are post-hoc join keys only. **Abstention policy:** failed/"
        "unparseable calls are excluded from Accuracy/F1/AUC and reported via **Coverage**. Latency "
        "shows `n/a` where it was not measured (e.g. browser-collected LLM runs)."
    )
    L.append("")
    L.append("## Why MCC first")
    L.append("")
    L.append(
        "With a 79/21 split, accuracy is easy to game by always guessing *fake*. **MCC** balances all "
        "four confusion cells and is the fairest single operating-point score; **ROC-AUC** is shown "
        "alongside because it is threshold-free and measures pure ranking power. The two often "
        "disagree — that gap is the most interesting story in this data (see Key findings)."
    )
    L.append("")
    L.append("## Key findings")
    L.append("")
    for f in findings(rows, aggs):
        L.append(f"- {f}")
    L.append("")
    L.append("## Per-track summary")
    L.append("")
    L.append("| Track | Models | Best MCC | Median MCC | Best ROC-AUC | Median ROC-AUC | Mean coverage |")
    L.append("|---|---:|---|---:|---|---:|---:|")
    for a in aggs:
        L.append(
            f"| {a['label']} | {a['count']} | {a['best_mcc_val']:.3f} (`{a['best_mcc_model']}`) | "
            f"{a['median_mcc']:.3f} | {a['best_auc_val']:.3f} (`{a['best_auc_model']}`) | "
            f"{a['median_auc']:.3f} | {a['mean_coverage']*100:.0f}% |"
        )
    L.append("")
    # Latency — commercial APIs only (cross-track latency is not comparable)
    crows = commercial_latency_rows(rows)
    if crows:
        L.append("## Latency — commercial APIs only")
        L.append("")
        L.append(
            "Latency is only compared across the **commercial APIs** (network round-trip to a hosted "
            "service). Open-source detectors run on our own GPU, where throughput depends on local "
            "hardware/batching, and the browser-collected LLM runs were not timed — so those latencies "
            "are **not comparable** and are omitted here."
        )
        L.append("")
        L.append("| Provider | Mean latency |")
        L.append("|---|---:|")
        for r in crows:
            L.append(f"| `{r['model']}` | {lat(r['latency_mean_ms'])} |")
        L.append("")

    L.append(f"## Unified leaderboard — all {len(rows)} models")
    L.append("")
    header = "| # | Model | Track | " + " | ".join(lbl for _k, lbl, _n in METRIC_COLS) + " | TP | FP | TN | FN |"
    sep = "|---:|---|---|" + "|".join(["---:"] * len(METRIC_COLS)) + "|---:|---:|---:|---:|"
    L.append(header)
    L.append(sep)
    for r in rows:
        metrics = " | ".join(fmt(r.get(k), nd) for k, _lbl, nd in METRIC_COLS)
        L.append(
            f"| {r['rank']} | `{r['model']}` | {r['track_label']} | {metrics} | "
            f"{r['TP']} | {r['FP']} | {r['TN']} | {r['FN']} |"
        )
    L.append("")
    L.append("---")
    L.append("")
    L.append(
        "Per-track deep dives: [`../commercial-benchmark/`](../commercial-benchmark/), "
        "[`../llm-benchmark/MODELS_REPORT.md`](../llm-benchmark/MODELS_REPORT.md), "
        "[`../opensource-benchmark/MODELS_REPORT.md`](../opensource-benchmark/MODELS_REPORT.md). "
        "Regenerate with `python scripts/report_compare.py`."
    )
    L.append("")
    return "\n".join(L)


def aggs_count(aggs: list[dict], label: str) -> int:
    for a in aggs:
        if a["label"] == label:
            return a["count"]
    return 0


# ------------------------------------------------------------------- HTML -----

def svg_mcc_bars(rows: list[dict]) -> str:
    """Horizontal MCC bars with a zero baseline, coloured by track."""
    vals = [r["mcc"] for r in rows]
    dmin, dmax = min(min(vals), 0.0), max(max(vals), 0.0)
    span = dmax - dmin or 1.0
    label_w, plot_w = 210, 640
    row_h, pad_top = 20, 28
    height = pad_top + len(rows) * row_h + 12
    total_w = label_w + plot_w + 60
    zero_x = label_w + (0.0 - dmin) / span * plot_w

    def x_of(v):
        return label_w + (v - dmin) / span * plot_w

    parts = [f'<svg viewBox="0 0 {total_w} {height}" width="100%" role="img" '
             f'aria-label="MCC by model" font-family="system-ui,Segoe UI,Roboto,sans-serif">']
    # axis gridlines at a few MCC values
    for gv in [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8]:
        if gv < dmin - 1e-9 or gv > dmax + 1e-9:
            continue
        gx = x_of(gv)
        parts.append(f'<line x1="{gx:.1f}" y1="{pad_top-6}" x2="{gx:.1f}" y2="{height-8}" '
                     f'stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{gx:.1f}" y="{pad_top-10}" font-size="10" fill="#9ca3af" '
                     f'text-anchor="middle">{gv:.1f}</text>')
    parts.append(f'<line x1="{zero_x:.1f}" y1="{pad_top-6}" x2="{zero_x:.1f}" y2="{height-8}" '
                 f'stroke="#9ca3af" stroke-width="1.2"/>')
    for i, r in enumerate(rows):
        y = pad_top + i * row_h
        bx = x_of(r["mcc"])
        x0, x1 = (zero_x, bx) if r["mcc"] >= 0 else (bx, zero_x)
        w = max(1.0, x1 - x0)
        parts.append(f'<title>{html.escape(r["model"])} — MCC {r["mcc"]:.3f}</title>')
        parts.append(f'<rect x="{x0:.1f}" y="{y+3}" width="{w:.1f}" height="{row_h-8}" '
                     f'rx="2" fill="{r["track_colour"]}"><title>{html.escape(r["model"])} '
                     f'({r["track_label"]}) MCC {r["mcc"]:.3f}</title></rect>')
        parts.append(f'<text x="{label_w-6}" y="{y+row_h-6}" font-size="11" fill="#374151" '
                     f'text-anchor="end">{html.escape(r["model"])}</text>')
        vx = bx + (4 if r["mcc"] >= 0 else -4)
        anchor = "start" if r["mcc"] >= 0 else "end"
        parts.append(f'<text x="{vx:.1f}" y="{y+row_h-6}" font-size="10" fill="#6b7280" '
                     f'text-anchor="{anchor}">{r["mcc"]:.2f}</text>')
    parts.append("</svg>")
    return "".join(parts)


def svg_scatter(rows: list[dict]) -> str:
    """ROC-AUC (x) vs MCC (y) scatter, coloured by track."""
    W, H = 720, 420
    ml, mr, mt, mb = 60, 20, 20, 46
    pw, ph = W - ml - mr, H - mt - mb
    xmin, xmax = 0.4, 0.95
    ymin, ymax = -0.25, 0.95

    def px(v):
        v = min(max(v, xmin), xmax)
        return ml + (v - xmin) / (xmax - xmin) * pw

    def py(v):
        v = min(max(v, ymin), ymax)
        return mt + (1 - (v - ymin) / (ymax - ymin)) * ph

    parts = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="ROC-AUC vs MCC" '
             f'font-family="system-ui,Segoe UI,Roboto,sans-serif">']
    # grid + axes
    for gx in [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        X = px(gx)
        parts.append(f'<line x1="{X:.1f}" y1="{mt}" x2="{X:.1f}" y2="{mt+ph}" stroke="#eef0f3"/>')
        parts.append(f'<text x="{X:.1f}" y="{mt+ph+16}" font-size="10" fill="#9ca3af" text-anchor="middle">{gx:.1f}</text>')
    for gy in [-0.2, 0.0, 0.2, 0.4, 0.6, 0.8]:
        Y = py(gy)
        parts.append(f'<line x1="{ml}" y1="{Y:.1f}" x2="{ml+pw}" y2="{Y:.1f}" stroke="#eef0f3"/>')
        parts.append(f'<text x="{ml-8}" y="{Y+3:.1f}" font-size="10" fill="#9ca3af" text-anchor="end">{gy:.1f}</text>')
    # zero MCC emphasis
    parts.append(f'<line x1="{ml}" y1="{py(0):.1f}" x2="{ml+pw}" y2="{py(0):.1f}" stroke="#d1d5db" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{ml+pw/2:.0f}" y="{H-6}" font-size="11" fill="#6b7280" text-anchor="middle">ROC-AUC (ranking power) \u2192</text>')
    parts.append(f'<text x="16" y="{mt+ph/2:.0f}" font-size="11" fill="#6b7280" text-anchor="middle" '
                 f'transform="rotate(-90 16 {mt+ph/2:.0f})">MCC (decision quality) \u2192</text>')
    for r in rows:
        X, Y = px(r["roc_auc"]), py(r["mcc"])
        parts.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="5.5" fill="{r["track_colour"]}" '
                     f'fill-opacity="0.82" stroke="#fff" stroke-width="1"><title>{html.escape(r["model"])} '
                     f'({r["track_label"]}) — AUC {r["roc_auc"]:.3f}, MCC {r["mcc"]:.3f}</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def svg_latency_bars(crows: list[dict]) -> str:
    """Horizontal latency bars for commercial APIs (fastest first)."""
    maxv = max(r["latency_mean_ms"] for r in crows)
    label_w, plot_w = 170, 520
    row_h, pad_top = 30, 10
    height = pad_top + len(crows) * row_h + 8
    total_w = label_w + plot_w + 90
    parts = [f'<svg viewBox="0 0 {total_w} {height}" width="100%" role="img" '
             f'aria-label="Commercial API latency" font-family="system-ui,Segoe UI,Roboto,sans-serif">']
    for i, r in enumerate(crows):
        y = pad_top + i * row_h
        w = max(2.0, r["latency_mean_ms"] / maxv * plot_w)
        parts.append(f'<rect x="{label_w}" y="{y+4}" width="{w:.1f}" height="{row_h-12}" rx="3" '
                     f'fill="{r["track_colour"]}"><title>{html.escape(r["model"])} — '
                     f'{lat(r["latency_mean_ms"])}</title></rect>')
        parts.append(f'<text x="{label_w-8}" y="{y+row_h-9}" font-size="12" fill="#374151" '
                     f'text-anchor="end">{html.escape(r["model"])}</text>')
        parts.append(f'<text x="{label_w+w+6:.1f}" y="{y+row_h-9}" font-size="11" fill="#6b7280" '
                     f'text-anchor="start">{lat(r["latency_mean_ms"])}</text>')
    parts.append("</svg>")
    return "".join(parts)


def build_html(rows: list[dict], aggs: list[dict], meta: dict) -> str:
    ds, fk, rl = meta["dataset_size"], meta["fake"], meta["real"]
    legend = "".join(
        f'<span class="badge {a["css"]}">{html.escape(a["label"])} · {a["count"]}</span>' for a in aggs
    )
    cards = "".join(
        f'''<div class="card" style="border-top:4px solid {a['colour']}">
              <h3>{html.escape(a['label'])}</h3>
              <div class="big">{a['best_mcc_val']:.3f}</div>
              <div class="sub">best MCC · <code>{html.escape(a['best_mcc_model'])}</code></div>
              <table class="mini">
                <tr><td>Models</td><td>{a['count']}</td></tr>
                <tr><td>Median MCC</td><td>{a['median_mcc']:.3f}</td></tr>
                <tr><td>Best ROC-AUC</td><td>{a['best_auc_val']:.3f} <code>{html.escape(a['best_auc_model'])}</code></td></tr>
                <tr><td>Median ROC-AUC</td><td>{a['median_auc']:.3f}</td></tr>
                <tr><td>Mean coverage</td><td>{a['mean_coverage']*100:.0f}%</td></tr>
              </table>
            </div>''' for a in aggs
    )
    find_items = "".join(f"<li>{md_inline_to_html(f)}</li>" for f in findings(rows, aggs))

    # latency section — commercial APIs only
    crows = commercial_latency_rows(rows)
    latency_html = ""
    if crows:
        trows = "".join(
            f'<tr><td><code>{html.escape(r["model"])}</code></td>'
            f'<td style="text-align:right">{lat(r["latency_mean_ms"])}</td></tr>' for r in crows
        )
        latency_html = f'''
  <h2>Latency — commercial APIs only</h2>
  <div class="panel">
    <p style="margin:0 0 12px;color:#374151;font-size:13px">Compared for <b>commercial APIs</b> only
    (network round-trip to a hosted service). Open-source detectors run on our own GPU, where
    throughput depends on local hardware/batching, and browser-collected LLM runs were not timed \u2014
    so those latencies are <b>not comparable</b> and are omitted.</p>
    {svg_latency_bars(crows)}
    <table class="mini" style="max-width:340px;margin-top:12px">
      <tr><td><b>Provider</b></td><td style="text-align:right"><b>Mean latency</b></td></tr>
      {trows}
    </table>
  </div>'''

    # leaderboard table
    head_cells = "".join(f'<th data-k="{k}" class="num">{lbl}</th>' for k, lbl, _n in METRIC_COLS)
    body = []
    for r in rows:
        metric_tds = "".join(
            f'<td class="num" data-v="{(r.get(k) if r.get(k) is not None else -9)}">{fmt(r.get(k), nd)}</td>'
            for k, _lbl, nd in METRIC_COLS
        )
        body.append(
            f'<tr><td class="num">{r["rank"]}</td>'
            f'<td class="model"><code>{html.escape(r["model"])}</code></td>'
            f'<td><span class="badge {r["track_css"]}">{html.escape(r["track_label"])}</span></td>'
            f'{metric_tds}'
            f'<td class="num muted">{r["TP"]}/{r["FP"]}/{r["TN"]}/{r["FN"]}</td></tr>'
        )
    rows_html = "\n".join(body)

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Comparative Benchmark — Commercial / LLM / Open-source</title>
<style>
  :root {{ --commercial:#6366f1; --llm:#ec4899; --opensource:#10b981; }}
  * {{ box-sizing:border-box; }}
  body {{ font-family:system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif; margin:0;
          color:#111827; background:#f8fafc; line-height:1.5; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:32px 20px 64px; }}
  h1 {{ font-size:26px; margin:0 0 6px; }}
  h2 {{ font-size:19px; margin:36px 0 12px; border-bottom:1px solid #e5e7eb; padding-bottom:6px; }}
  h3 {{ margin:0 0 8px; font-size:15px; }}
  p.lede {{ color:#374151; max-width:900px; }}
  .note {{ background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; padding:12px 14px;
           font-size:13px; color:#1e3a8a; max-width:900px; }}
  .badge {{ display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px; font-weight:600;
            color:#fff; margin-right:6px; }}
  .badge.commercial {{ background:var(--commercial); }}
  .badge.llm {{ background:var(--llm); }}
  .badge.opensource {{ background:var(--opensource); }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(240px,1fr)); gap:16px; }}
  .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:16px 18px;
           box-shadow:0 1px 2px rgba(0,0,0,.04); }}
  .card .big {{ font-size:30px; font-weight:700; }}
  .card .sub {{ color:#6b7280; font-size:13px; margin-bottom:10px; }}
  table.mini {{ width:100%; border-collapse:collapse; font-size:13px; }}
  table.mini td {{ padding:3px 0; border-top:1px solid #f1f5f9; }}
  table.mini td:last-child {{ text-align:right; color:#374151; }}
  .panel {{ background:#fff; border:1px solid #e5e7eb; border-radius:12px; padding:16px 18px; }}
  .chart-grid {{ display:grid; grid-template-columns:1fr; gap:20px; }}
  @media(min-width:900px){{ .chart-grid.two {{ grid-template-columns:1fr 1fr; }} }}
  ul.find li {{ margin:6px 0; }}
  table.lb {{ width:100%; border-collapse:collapse; font-size:13px; background:#fff;
              border:1px solid #e5e7eb; border-radius:12px; overflow:hidden; }}
  table.lb th, table.lb td {{ padding:7px 10px; border-bottom:1px solid #f1f5f9; text-align:left; }}
  table.lb th {{ background:#f9fafb; position:sticky; top:0; cursor:pointer; user-select:none; font-size:12px;
                 color:#374151; white-space:nowrap; }}
  table.lb th.num, table.lb td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
  table.lb tr:hover td {{ background:#fafafa; }}
  td.model code {{ font-size:12px; }}
  td.muted {{ color:#9ca3af; }}
  code {{ background:#f3f4f6; padding:1px 5px; border-radius:4px; font-size:12px; }}
  .foot {{ color:#6b7280; font-size:13px; margin-top:28px; }}
  caption {{ text-align:left; color:#6b7280; font-size:12px; padding:8px 2px; caption-side:bottom; }}
</style></head>
<body><div class="wrap">
  <h1>Comparative Benchmark</h1>
  <p class="lede">Commercial detection APIs vs vision LLMs vs open-source detectors, all scored on the
  <b>same {ds} images ({fk} fake / {rl} real)</b> with one metric schema. {len(rows)} models total.
  Ranked by <b>MCC</b> (operating-point quality), tiebreak <b>ROC-AUC</b>.</p>
  <p style="margin:10px 0 14px">{legend}</p>
  <div class="note"><b>No label leakage</b> — models saw neutral numeric filenames only.
  <b>Abstention policy</b> — failed/unparseable calls are excluded from Accuracy/F1/AUC and shown as
  <b>Coverage</b>. Latency is <code>n/a</code> where not measured (e.g. browser-collected LLM runs).</div>

  <h2>Per-track summary</h2>
  <div class="cards">{cards}</div>

  <h2>MCC vs ROC-AUC — decision quality vs ranking power</h2>
  <div class="chart-grid two">
    <div class="panel"><h3>MCC by model</h3>{svg_mcc_bars(rows)}
      <div style="font-size:12px;color:#6b7280;margin-top:6px">Ranked best→worst. Bars left of the line are negative MCC (worse than chance at the shipped threshold).</div>
    </div>
    <div class="panel"><h3>ROC-AUC vs MCC</h3>{svg_scatter(rows)}
      <div style="font-size:12px;color:#6b7280;margin-top:6px">Top-right = strong ranking <i>and</i> good threshold. High-x/low-y points separate classes well but ship a mis-calibrated threshold.</div>
    </div>
  </div>

  <h2>Key findings</h2>
  <ul class="find">{find_items}</ul>
{latency_html}
  <h2>Unified leaderboard — all {len(rows)} models</h2>
  <table class="lb" id="lb">
    <thead><tr>
      <th class="num" data-k="rank">#</th><th data-k="model">Model</th><th data-k="track_label">Track</th>
      {head_cells}
      <th class="num">TP/FP/TN/FN</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
    <caption>Sorted by MCC (desc). Click any numeric header to re-sort; click again to reverse.</caption>
  </table>

  <div class="foot">Generated by <code>scripts/report_compare.py</code> from <code>results/*/summary.json</code>.
  Per-track deep dives live in <code>reports/commercial-benchmark/</code>,
  <code>reports/llm-benchmark/</code>, <code>reports/opensource-benchmark/</code>.</div>
</div>
<script>
(function() {{
  var tbl = document.getElementById('lb');
  var tb = tbl.tBodies[0];
  tbl.querySelectorAll('th').forEach(function(th, idx) {{
    if (!th.classList.contains('num')) return;
    var asc = false;
    th.addEventListener('click', function() {{
      asc = !asc;
      var rowsArr = Array.prototype.slice.call(tb.rows);
      rowsArr.sort(function(a, b) {{
        var av = parseFloat(a.cells[idx].dataset.v != null ? a.cells[idx].dataset.v : a.cells[idx].textContent);
        var bv = parseFloat(b.cells[idx].dataset.v != null ? b.cells[idx].dataset.v : b.cells[idx].textContent);
        if (isNaN(av)) av = -Infinity; if (isNaN(bv)) bv = -Infinity;
        return asc ? av - bv : bv - av;
      }});
      rowsArr.forEach(function(r) {{ tb.appendChild(r); }});
    }});
  }});
}})();
</script>
</body></html>
"""


def md_inline_to_html(s: str) -> str:
    """Minimal **bold** and `code` -> HTML for the findings list."""
    import re
    s = html.escape(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
    return s


def main() -> None:
    rows, meta = load_rows()
    if not rows:
        raise SystemExit("no track summaries found under results/")
    aggs = track_aggregates(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "COMPARISON_REPORT.md").write_text(build_md(rows, aggs, meta), encoding="utf-8")
    (OUT / "comparison_report.html").write_text(build_html(rows, aggs, meta), encoding="utf-8")
    print(f"models={len(rows)} tracks={len(aggs)} -> {OUT.relative_to(ROOT)}/"
          f" (COMPARISON_REPORT.md, comparison_report.html)")
    print("top5 by MCC:", ", ".join(f"{r['model']}={r['mcc']:.3f}" for r in rows[:5]))


if __name__ == "__main__":
    main()
