#!/usr/bin/env python3
"""Compute paired bootstrap 95% CIs for pose-only vs pose+visual baselines.

Uses per-video summary CSVs from RF/XGBoost held-out evaluation (30 videos).
Also loads the full LSTM per-video results for cross-classifier comparison.

Outputs:
    outputs/results_visual/bootstrap_ci_results.json

Usage:
    cd pose_baseline_rf
    python compute_bootstrap_ci.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

THIS_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((THIS_DIR / "config.yaml").read_text(encoding="utf-8"))
OUTPUT_ROOT = Path(CONFIG["output_root"])
FINAL_RUN_ROOT = THIS_DIR.parent

N_BOOTSTRAP = 10_000
SEED = 42
ALPHA = 0.05  # two-sided → 2.5th and 97.5th percentiles


# ─── Data loaders ──────────────────────────────────────────────────────────

def load_baseline_csv(csv_path: Path, pred_type: str = "smoothed") -> pd.DataFrame:
    """Load RF/XGBoost per-video summary CSV, filter to smoothed rows."""
    df = pd.read_csv(csv_path)
    df = df[df["prediction_type"] == pred_type].copy()
    df = df.rename(columns={
        "mars_split": "split",
        "frame_macro_f1": "frame_f1",
        "bout_macro_f1_iou25": "bout_f1_25",
        "bout_macro_f1_iou50": "bout_f1_50",
    })
    return df[["split", "video_id", "frame_f1", "bout_f1_25", "bout_f1_50"]].copy()


def load_lstm_csv(csv_path: Path, pred_type: str = "smoothed_frames") -> pd.DataFrame:
    """Load LSTM per-video summary CSV, filter to smoothed rows."""
    df = pd.read_csv(csv_path)
    df = df[df["prediction_type"] == pred_type].copy()
    df = df.rename(columns={
        "frame_macro_f1_present_behaviors": "frame_f1",
        "bout_macro_f1_iou25_present_behaviors": "bout_f1_25",
        "bout_macro_f1_iou50_present_behaviors": "bout_f1_50",
    })
    return df[["split", "video_id", "frame_f1", "bout_f1_25", "bout_f1_50"]].copy()


# ─── Bootstrap CI ──────────────────────────────────────────────────────────

def paired_bootstrap_ci(
    deltas: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    seed: int = SEED,
) -> dict:
    """Paired bootstrap 95% CI over per-video deltas.

    Returns dict with mean_delta, ci_lo, ci_hi, significant (CI excludes 0).
    """
    rng = np.random.RandomState(seed)
    n = len(deltas)
    boot_means = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means[b] = deltas[idx].mean()
    ci_lo = float(np.percentile(boot_means, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    mean_delta = float(deltas.mean())
    significant = (ci_lo > 0) or (ci_hi < 0)
    return {
        "mean_delta": round(mean_delta, 4),
        "ci_lo": round(ci_lo, 4),
        "ci_hi": round(ci_hi, 4),
        "significant": significant,
        "n_videos": int(n),
    }


def compute_comparison(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
) -> dict:
    """Compute paired bootstrap CIs for A minus B across all metrics."""
    merged = pd.merge(df_a, df_b, on="video_id", suffixes=("_a", "_b"))
    assert len(merged) == 30, f"Expected 30 videos, got {len(merged)}"

    metrics = ["frame_f1", "bout_f1_25", "bout_f1_50"]
    metric_labels = {
        "frame_f1": "frame_macro_f1",
        "bout_f1_25": "bout_f1_iou25",
        "bout_f1_50": "bout_f1_iou50",
    }

    result = {
        "comparison": f"{label_a} minus {label_b}",
        "model_a": label_a,
        "model_b": label_b,
        "metrics": {},
    }
    for m in metrics:
        deltas = merged[f"{m}_a"].values - merged[f"{m}_b"].values
        ci = paired_bootstrap_ci(deltas)
        result["metrics"][metric_labels[m]] = ci

    return result


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Paired bootstrap CIs — pose+visual vs pose-only baselines")
    print("=" * 70)

    # ── Load all per-video results ─────────────────────────────────────
    # Pose-only baselines
    rf_pose = load_baseline_csv(
        OUTPUT_ROOT / "predictions" / "rf_perframe" / "per_video_summary.csv")
    xgb_pose = load_baseline_csv(
        OUTPUT_ROOT / "predictions" / "xgb_perframe" / "per_video_summary.csv")

    # Pose+visual baselines
    rf_visual = load_baseline_csv(
        OUTPUT_ROOT / "predictions_visual" / "rf_perframe" / "per_video_summary.csv")
    xgb_visual = load_baseline_csv(
        OUTPUT_ROOT / "predictions_visual" / "xgb_perframe" / "per_video_summary.csv")

    # LSTM models
    lstm_full = load_lstm_csv(
        FINAL_RUN_ROOT / "outputs" / "test_eval" / "full_lstm512" / "per_video_summary.csv")
    lstm_pose = load_lstm_csv(
        FINAL_RUN_ROOT / "outputs" / "test_eval" / "pose_relations_only_lstm256" / "per_video_summary.csv")

    print(f"\nLoaded per-video results:")
    for name, df in [("RF pose-only", rf_pose), ("RF pose+visual", rf_visual),
                     ("XGB pose-only", xgb_pose), ("XGB pose+visual", xgb_visual),
                     ("LSTM full", lstm_full), ("LSTM pose-only", lstm_pose)]:
        print(f"  {name:<20s}: {len(df)} videos, "
              f"mean frame-F1={df['frame_f1'].mean():.3f}")

    # ── Compute all comparisons ────────────────────────────────────────
    comparisons = []

    # 1. Pose+visual vs pose-only within each classifier
    print("\n── Within-classifier: pose+visual vs pose-only ──")
    comparisons.append(compute_comparison(
        rf_visual, rf_pose, "RF_perframe_visual", "RF_perframe_pose"))
    comparisons.append(compute_comparison(
        xgb_visual, xgb_pose, "XGB_perframe_visual", "XGB_perframe_pose"))

    # 2. Pose+visual baselines vs full LSTM
    print("── Pose+visual baselines vs full LSTM ──")
    comparisons.append(compute_comparison(
        rf_visual, lstm_full, "RF_perframe_visual", "Full_LSTM_512"))
    comparisons.append(compute_comparison(
        xgb_visual, lstm_full, "XGB_perframe_visual", "Full_LSTM_512"))

    # 3. Pose+visual baselines vs pose-only LSTM
    print("── Pose+visual baselines vs pose-only LSTM ──")
    comparisons.append(compute_comparison(
        rf_visual, lstm_pose, "RF_perframe_visual", "Pose_LSTM_256"))
    comparisons.append(compute_comparison(
        xgb_visual, lstm_pose, "XGB_perframe_visual", "Pose_LSTM_256"))

    # 4. Full LSTM (visual) vs pose-only LSTM (for reference / manuscript)
    print("── LSTM: full vs pose-only ──")
    comparisons.append(compute_comparison(
        lstm_full, lstm_pose, "Full_LSTM_512", "Pose_LSTM_256"))

    # ── Print results ──────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"{'Comparison':<45s} {'Metric':<16s} {'Delta':>7s} {'95% CI':>20s} {'Sig?':>5s}")
    print("-" * 95)
    for comp in comparisons:
        for metric_name, ci in comp["metrics"].items():
            sig_str = "***" if ci["significant"] else "n.s."
            print(f"{comp['comparison']:<45s} {metric_name:<16s} "
                  f"{ci['mean_delta']:+.4f} [{ci['ci_lo']:+.4f}, {ci['ci_hi']:+.4f}] "
                  f"{sig_str:>5s}")
        print()

    # ── Save ───────────────────────────────────────────────────────────
    out_dir = OUTPUT_ROOT / "results_visual"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bootstrap_ci_results.json"
    out_path.write_text(json.dumps(comparisons, indent=2), encoding="utf-8")
    print(f"\n  -> {out_path}")

    # ── Summary table for manuscript ───────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Key findings for manuscript:")
    print("=" * 70)

    # Extract key numbers
    rf_vs = comparisons[0]  # RF visual minus RF pose
    xgb_vs = comparisons[1]  # XGB visual minus XGB pose
    lstm_vs = comparisons[6]  # LSTM full minus LSTM pose

    for label, comp in [("RF (visual−pose)", rf_vs),
                        ("XGB (visual−pose)", xgb_vs),
                        ("LSTM (full−pose)", lstm_vs)]:
        ff1 = comp["metrics"]["frame_macro_f1"]
        print(f"  {label}: frame-F1 delta = {ff1['mean_delta']:+.3f} "
              f"CI [{ff1['ci_lo']:+.3f}, {ff1['ci_hi']:+.3f}] "
              f"{'SIG' if ff1['significant'] else 'n.s.'}")

    print("\nBootstrap CI computation complete.")


if __name__ == "__main__":
    main()
