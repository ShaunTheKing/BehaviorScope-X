#!/usr/bin/env python3
"""
build_kpt_eval_figures.py
==========================
Generate manuscript-grade figures from kpt_eval_fast_results_raw.csv.

Outputs (Plotly HTML, embedded so they render offline + embed in Quarto):
  - per_keypoint_pck.html       : grouped bar chart of PCK@0.05/0.10/0.20 per body part
  - per_keypoint_rmse.html      : bar chart of mean RMSE per body part (with error bars)
  - rmse_distribution.html      : overlaid histograms of per-keypoint pixel error
  - pck_by_class_heatmap.html   : heatmap of PCK@0.10 across (behavior class x body part)
  - body_length_distribution.html: histogram of per-window body length (sanity check)
  - oks_distribution.html       : histogram of mean OKS per window

Usage:
    python scripts/build_kpt_eval_figures.py ^
        --raw_csv     data/manuscript_v4/keypoint_eval/kpt_eval_fast_results_raw.csv ^
        --summary     data/manuscript_v4/keypoint_eval/kpt_eval_fast_results_summary.json ^
        --out_dir     data/manuscript_v4/keypoint_eval/figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

KPT_NAMES = ["nose", "ear_1", "ear_2", "neck", "hip_1", "hip_2", "tail_base"]
KPT_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
              "#9467bd", "#8c564b", "#e377c2"]

# Quality thresholds (PCK@0.10) for highlighting
PCK_GOOD = 0.85
PCK_OK   = 0.75


def _save(fig: go.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn", full_html=True)
    print(f"  -> {out_path}")


def _per_keypoint_pck(df: pd.DataFrame, out: Path) -> None:
    """Grouped bar chart: PCK@0.05/0.10/0.20 for each keypoint."""
    matched = df[df["matched"]].copy()
    rows = []
    for k, name in enumerate(KPT_NAMES):
        for thr_label, col_prefix in [("@0.05", "pck05"), ("@0.10", "pck10"), ("@0.20", "pck20")]:
            vals = matched[f"{col_prefix}_kpt{k}"].dropna()
            if len(vals) == 0:
                continue
            rows.append({"keypoint": name, "threshold": f"PCK{thr_label}",
                         "value": float(vals.mean()), "n": int(len(vals))})
    pck_df = pd.DataFrame(rows)

    fig = go.Figure()
    for thr in ["PCK@0.05", "PCK@0.10", "PCK@0.20"]:
        sub = pck_df[pck_df["threshold"] == thr]
        fig.add_trace(go.Bar(
            name=thr,
            x=sub["keypoint"], y=sub["value"],
            text=[f"{v:.3f}" for v in sub["value"]],
            textposition="outside",
            hovertemplate="%{x}<br>%{y:.3f}<extra>" + thr + "</extra>",
        ))
    fig.update_layout(
        title="Per-keypoint PCK (Percentage of Correct Keypoints)",
        xaxis_title="Body part",
        yaxis_title="Fraction correct",
        yaxis=dict(range=[0, 1.05]),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=460,
    )
    _save(fig, out)


def _per_keypoint_rmse(df: pd.DataFrame, out: Path) -> None:
    """Bar chart: mean RMSE per keypoint with std error bars."""
    matched = df[df["matched"]].copy()
    rmse_means, rmse_stds, kpt_labels = [], [], []
    for k, name in enumerate(KPT_NAMES):
        vals = matched[f"rmse_kpt{k}"].dropna().values
        if len(vals) == 0:
            continue
        kpt_labels.append(name)
        rmse_means.append(float(np.sqrt(np.nanmean(vals ** 2))))  # quadratic mean
        rmse_stds.append(float(np.nanstd(vals)))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=kpt_labels, y=rmse_means,
        error_y=dict(type="data", array=rmse_stds),
        marker_color=KPT_COLORS[:len(kpt_labels)],
        text=[f"{m:.1f} px" for m in rmse_means],
        textposition="outside",
        hovertemplate="%{x}<br>RMSE = %{y:.2f} px<extra></extra>",
    ))
    fig.update_layout(
        title="Per-keypoint RMSE (pixels, lower is better)",
        xaxis_title="Body part",
        yaxis_title="RMSE (px)",
        template="plotly_white",
        height=460,
        showlegend=False,
    )
    _save(fig, out)


def _rmse_distribution(df: pd.DataFrame, out: Path) -> None:
    """Overlaid histograms of pixel error, one trace per keypoint.
    Pre-binned with np.histogram so the HTML stays small even for ~1M rows."""
    matched = df[df["matched"]].copy()
    # global clip range so all keypoints share the same x axis
    all_vals = np.concatenate([matched[f"rmse_kpt{k}"].dropna().values
                               for k in range(len(KPT_NAMES))])
    if len(all_vals) == 0:
        return
    x_max = float(np.percentile(all_vals, 99))
    n_bins = 60
    edges = np.linspace(0.0, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig = go.Figure()
    for k, name in enumerate(KPT_NAMES):
        vals = matched[f"rmse_kpt{k}"].dropna().values
        if len(vals) == 0:
            continue
        counts, _ = np.histogram(vals[vals < x_max], bins=edges)
        fig.add_trace(go.Bar(
            x=centers, y=counts, name=name,
            marker_color=KPT_COLORS[k], opacity=0.55,
            width=(edges[1] - edges[0]),
            hovertemplate="%{x:.1f} px<br>n=%{y}<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        title="Per-keypoint pixel-error distribution (clipped at 99th percentile)",
        xaxis_title="Distance to GT keypoint (pixels)",
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_white",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    _save(fig, out)


def _pck_by_class_heatmap(df: pd.DataFrame, out: Path) -> None:
    """Heatmap of PCK@0.10 across (behavior class x body part)."""
    matched = df[df["matched"]].copy()
    classes = sorted(matched["class_name"].dropna().unique().tolist())
    z = np.zeros((len(classes), len(KPT_NAMES)), dtype=float)
    n  = np.zeros_like(z, dtype=int)
    for ci, c in enumerate(classes):
        sub = matched[matched["class_name"] == c]
        for k in range(len(KPT_NAMES)):
            vals = sub[f"pck10_kpt{k}"].dropna().values
            z[ci, k] = float(np.mean(vals)) if len(vals) else float("nan")
            n[ci, k] = len(vals)

    text = [[f"{z[i,j]:.3f}<br>n={n[i,j]:,}" for j in range(z.shape[1])]
            for i in range(z.shape[0])]

    fig = go.Figure(go.Heatmap(
        z=z, x=KPT_NAMES, y=classes,
        colorscale="RdYlGn", zmin=0.6, zmax=1.0,
        text=text, texttemplate="%{text}",
        hovertemplate="class=%{y}<br>kpt=%{x}<br>PCK@0.10=%{z:.3f}<extra></extra>",
        colorbar=dict(title="PCK@0.10"),
    ))
    fig.update_layout(
        title="PCK@0.10 by behavior class and body part",
        xaxis_title="Body part",
        yaxis_title="Behavior class",
        template="plotly_white",
        height=420,
    )
    _save(fig, out)


def _binned_bar(values: np.ndarray, n_bins: int, color: str,
                hover_unit: str) -> go.Bar:
    """Pre-bin values with numpy and return a small go.Bar trace."""
    counts, edges = np.histogram(values, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return go.Bar(
        x=centers, y=counts, marker_color=color,
        width=(edges[1] - edges[0]),
        hovertemplate="%{x:" + hover_unit + "}<br>n=%{y}<extra></extra>",
    )


def _body_length_distribution(df: pd.DataFrame, out: Path) -> None:
    """Histogram of per-frame body length (sanity check on PCK normalisation).
    Pre-binned to keep HTML small."""
    bls = df["body_len_px"].dropna().values
    bls = bls[(bls > 5) & (bls < np.percentile(bls, 99.5))]
    if len(bls) == 0:
        return
    fig = go.Figure(_binned_bar(bls, 80, "#1f77b4", ".1f"))
    fig.update_layout(
        title=f"Body length distribution (neck-to-tail, px) — median={np.median(bls):.1f}",
        xaxis_title="Body length (pixels)",
        yaxis_title="Count",
        template="plotly_white",
        height=380,
        showlegend=False,
    )
    _save(fig, out)


def _oks_distribution(df: pd.DataFrame, out: Path) -> None:
    """Histogram of OKS per matched window-frame-animal. Pre-binned."""
    oks = df.loc[df["matched"], "oks"].dropna().values
    if len(oks) == 0:
        return
    fig = go.Figure(_binned_bar(oks, 60, "#2ca02c", ".3f"))
    fig.add_vline(x=float(np.mean(oks)), line_dash="dash", line_color="#d62728",
                  annotation_text=f"mean={np.mean(oks):.3f}", annotation_position="top right")
    fig.update_layout(
        title=f"Object Keypoint Similarity (OKS) distribution — n={len(oks):,}",
        xaxis_title="OKS (0=worst, 1=perfect)",
        yaxis_title="Count",
        template="plotly_white",
        height=380,
        showlegend=False,
    )
    _save(fig, out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw_csv",  required=True)
    p.add_argument("--summary",  required=True)
    p.add_argument("--out_dir",  required=True)
    args = p.parse_args()

    raw_csv = Path(args.raw_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[figs] Loading {raw_csv}  ({raw_csv.stat().st_size/1e6:.1f} MB) ...")
    df = pd.read_csv(raw_csv)
    print(f"[figs] Loaded {len(df):,} rows; matched={df['matched'].sum():,}")

    with open(args.summary) as f:
        summary = json.load(f)
    print(f"[figs] Summary: detection_rate={summary.get('detection_rate'):.3f}  "
          f"mean_oks={summary.get('mean_oks'):.3f}  "
          f"PCK@0.10={summary.get('overall_pck@0.10'):.3f}")

    print("\n[figs] Building figures:")
    _per_keypoint_pck(df,           out_dir / "per_keypoint_pck.html")
    _per_keypoint_rmse(df,          out_dir / "per_keypoint_rmse.html")
    _rmse_distribution(df,          out_dir / "rmse_distribution.html")
    _pck_by_class_heatmap(df,       out_dir / "pck_by_class_heatmap.html")
    _body_length_distribution(df,   out_dir / "body_length_distribution.html")
    _oks_distribution(df,           out_dir / "oks_distribution.html")
    print("\n[figs] Done.")


if __name__ == "__main__":
    main()
