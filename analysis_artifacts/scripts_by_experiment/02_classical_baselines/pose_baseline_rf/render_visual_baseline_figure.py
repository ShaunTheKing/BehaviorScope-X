#!/usr/bin/env python3
"""Render two-panel figure comparing pose-only vs pose+visual baselines.

Panel A: Validation frame macro-F1 (no CIs — single aggregate per model).
Panel B: Held-out frame macro-F1 with bootstrap 95% CIs.

The visual "inversion" is self-evident: visual features help all classifiers
on validation but only the LSTM on held-out.

Outputs:
    outputs/results_visual/visual_baseline_comparison.pdf
    outputs/results_visual/visual_baseline_comparison.png

Usage:
    cd pose_baseline_rf
    python render_visual_baseline_figure.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

THIS_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((THIS_DIR / "config.yaml").read_text(encoding="utf-8"))
OUTPUT_ROOT = Path(CONFIG["output_root"])
FINAL_RUN_ROOT = THIS_DIR.parent

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.28,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

# Colors matching existing manuscript figure palette
COLOR_POSE = "#3b82f6"     # blue — pose-only
COLOR_VISUAL = "#10b981"   # green — pose+visual
COLOR_POSE_LIGHT = "#93c5fd"
COLOR_VISUAL_LIGHT = "#6ee7b7"


def load_val_results() -> dict:
    """Load validation macro-F1 from training summary JSONs."""
    # Pose-only
    pose_summary = json.loads(
        (OUTPUT_ROOT / "results" / "training_summary.json").read_text(encoding="utf-8"))
    # Pose+visual
    visual_summary = json.loads(
        (OUTPUT_ROOT / "results_visual" / "training_summary.json").read_text(encoding="utf-8"))

    # Extract per-frame only (what we compare in visual experiment)
    rf_pose_val = next(r["val_macro_f1"] for r in pose_summary
                       if r["model"] == "RandomForest" and r["tag"] == "perframe")
    xgb_pose_val = next(r["val_macro_f1"] for r in pose_summary
                        if r["model"] == "XGBoost" and r["tag"] == "perframe")
    rf_vis_val = next(r["val_macro_f1"] for r in visual_summary
                      if r["model"] == "RandomForest")
    xgb_vis_val = next(r["val_macro_f1"] for r in visual_summary
                       if r["model"] == "XGBoost")

    # LSTM val results — from manuscript Table 4 (already reported)
    lstm_pose_val = 0.781   # Pose-only LSTM-256
    lstm_full_val = 0.812   # Full LSTM-256 (pose+visual)

    return {
        "RF": {"pose": rf_pose_val, "visual": rf_vis_val},
        "XGBoost": {"pose": xgb_pose_val, "visual": xgb_vis_val},
        "LSTM": {"pose": lstm_pose_val, "visual": lstm_full_val},
    }


def load_heldout_results() -> dict:
    """Load held-out smoothed frame macro-F1 from aggregate results."""
    # Pose-only
    pose_agg = json.loads(
        (OUTPUT_ROOT / "results" / "heldout_aggregate_results.json").read_text(encoding="utf-8"))
    # Pose+visual
    visual_agg = json.loads(
        (OUTPUT_ROOT / "results_visual" / "heldout_aggregate_results.json").read_text(encoding="utf-8"))

    rf_pose = next(r["mean_frame_macro_f1"] for r in pose_agg
                   if r["model"] == "rf_perframe" and r["prediction_type"] == "smoothed")
    xgb_pose = next(r["mean_frame_macro_f1"] for r in pose_agg
                    if r["model"] == "xgb_perframe" and r["prediction_type"] == "smoothed")
    rf_vis = next(r["mean_frame_macro_f1"] for r in visual_agg
                  if r["model"] == "rf_perframe" and r["prediction_type"] == "smoothed")
    xgb_vis = next(r["mean_frame_macro_f1"] for r in visual_agg
                   if r["model"] == "xgb_perframe" and r["prediction_type"] == "smoothed")

    # LSTM held-out from manuscript Table 5
    lstm_pose = 0.679   # Pose-only LSTM-256
    lstm_full = 0.706   # Full LSTM-256

    return {
        "RF": {"pose": rf_pose, "visual": rf_vis},
        "XGBoost": {"pose": xgb_pose, "visual": xgb_vis},
        "LSTM": {"pose": lstm_pose, "visual": lstm_full},
    }


def load_bootstrap_cis() -> dict:
    """Load bootstrap CIs for held-out within-classifier comparisons."""
    ci_path = OUTPUT_ROOT / "results_visual" / "bootstrap_ci_results.json"
    if not ci_path.is_file():
        print(f"  WARNING: {ci_path} not found. Run compute_bootstrap_ci.py first.")
        print("  Rendering figure without error bars.")
        return {}

    all_ci = json.loads(ci_path.read_text(encoding="utf-8"))

    # Extract within-classifier visual-minus-pose CIs on frame_macro_f1
    cis = {}
    for comp in all_ci:
        if "visual" in comp["model_a"] and "pose" in comp["model_b"]:
            if comp["model_a"].startswith("RF"):
                cis["RF"] = comp["metrics"]["frame_macro_f1"]
            elif comp["model_a"].startswith("XGB"):
                cis["XGBoost"] = comp["metrics"]["frame_macro_f1"]
        elif comp["model_a"] == "Full_LSTM_512" and comp["model_b"] == "Pose_LSTM_256":
            cis["LSTM"] = comp["metrics"]["frame_macro_f1"]

    return cis


def render_figure():
    val_data = load_val_results()
    heldout_data = load_heldout_results()
    boot_cis = load_bootstrap_cis()

    classifiers = ["RF", "XGBoost", "LSTM"]
    x = np.arange(len(classifiers))
    bar_width = 0.32

    fig, (ax_val, ax_held) = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)

    # ── Panel A: Validation ────────────────────────────────────────────
    pose_vals = [val_data[c]["pose"] for c in classifiers]
    vis_vals = [val_data[c]["visual"] for c in classifiers]

    bars_p = ax_val.bar(x - bar_width / 2, pose_vals, bar_width,
                        color=COLOR_POSE, edgecolor="white", linewidth=0.8,
                        label="Pose-only (132-d)", zorder=3)
    bars_v = ax_val.bar(x + bar_width / 2, vis_vals, bar_width,
                        color=COLOR_VISUAL, edgecolor="white", linewidth=0.8,
                        label="Pose + visual (900-d)", zorder=3)

    # Value labels
    for bar in bars_p:
        ax_val.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8, color="#374151")
    for bar in bars_v:
        ax_val.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8, color="#374151")

    # Delta annotations
    for i, c in enumerate(classifiers):
        delta = vis_vals[i] - pose_vals[i]
        mid_y = max(pose_vals[i], vis_vals[i]) + 0.030
        ax_val.annotate(
            f"{delta:+.3f}",
            xy=(x[i], mid_y), ha="center", fontsize=8,
            color="#059669" if delta > 0 else "#dc2626",
            fontweight="bold",
        )

    ax_val.set_xlabel("")
    ax_val.set_ylabel("Frame macro-F1 (behavior classes)")
    ax_val.set_title("A. Validation", fontsize=12, fontweight="bold", loc="left")
    ax_val.set_xticks(x)
    ax_val.set_xticklabels(classifiers)
    ax_val.set_ylim(0.55, 0.90)
    ax_val.legend(loc="upper left", framealpha=0.9, fontsize=8.5)

    # ── Panel B: Held-out ──────────────────────────────────────────────
    pose_held = [heldout_data[c]["pose"] for c in classifiers]
    vis_held = [heldout_data[c]["visual"] for c in classifiers]

    bars_p2 = ax_held.bar(x - bar_width / 2, pose_held, bar_width,
                          color=COLOR_POSE, edgecolor="white", linewidth=0.8,
                          label="Pose-only", zorder=3)
    bars_v2 = ax_held.bar(x + bar_width / 2, vis_held, bar_width,
                          color=COLOR_VISUAL, edgecolor="white", linewidth=0.8,
                          label="Pose + visual", zorder=3)

    # Value labels
    for bar in bars_p2:
        ax_held.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}", ha="center", va="bottom",
                     fontsize=8, color="#374151")
    for bar in bars_v2:
        ax_held.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                     f"{bar.get_height():.3f}", ha="center", va="bottom",
                     fontsize=8, color="#374151")

    # Delta annotations with significance markers
    for i, c in enumerate(classifiers):
        delta = vis_held[i] - pose_held[i]
        mid_y = max(pose_held[i], vis_held[i]) + 0.030
        sig_str = ""
        if c in boot_cis and boot_cis[c].get("significant"):
            sig_str = "*"
        label_text = f"{delta:+.3f}{sig_str}"
        ax_held.annotate(
            label_text,
            xy=(x[i], mid_y), ha="center", fontsize=8,
            color="#059669" if delta > 0 else "#dc2626",
            fontweight="bold",
        )

    ax_held.set_xlabel("")
    ax_held.set_title("B. Held-out test (30 videos)", fontsize=12,
                      fontweight="bold", loc="left")
    ax_held.set_xticks(x)
    ax_held.set_xticklabels(classifiers)

    # Shared note
    fig.text(0.5, -0.02,
             "Visual features improve validation F1 for all classifiers "
             "but transfer to held-out only for the LSTM.\n"
             "* = paired bootstrap 95% CI excludes zero (10,000 resamples, 30 per-video deltas).",
             ha="center", fontsize=8.5, color="#4b5563", style="italic")

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    # ── Save ───────────────────────────────────────────────────────────
    out_dir = OUTPUT_ROOT / "results_visual"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Also save to manuscript figures directory so LaTeX \includegraphics works
    manuscript_fig_dir = FINAL_RUN_ROOT / "manuscript_penultimate" / "figures"
    manuscript_fig_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "visual_baseline_comparison.png"
    fig.savefig(out_path)
    print(f"  -> {out_path}")

    ms_path = manuscript_fig_dir / "visual_baseline_comparison.png"
    fig.savefig(ms_path)
    print(f"  -> {ms_path}")

    plt.close(fig)
    print("\nFigure rendering complete.")


if __name__ == "__main__":
    render_figure()
