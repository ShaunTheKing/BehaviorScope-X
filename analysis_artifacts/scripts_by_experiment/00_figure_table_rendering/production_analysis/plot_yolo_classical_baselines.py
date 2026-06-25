"""Create production YOLO/SPPF RF/XGBoost baseline figure."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
STYLE_DIR = THIS_DIR.parent / "current_analysis"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import (  # noqa: E402
    CLASSICAL_MODEL_COLORS,
    NEUTRAL,
    apply_style,
    panel_label,
    save_figure,
)


BACKEND = "mars_yolo_sppf"
FEATURE_ORDER = ["pose_frame", "pose_visual_frame_pca", "pose_window", "pose_visual_window_pca"]
FEATURE_LABELS = {
    "pose_frame": "Pose\nframe",
    "pose_visual_frame_pca": "Pose+visual PCA\nframe",
    "pose_window": "Pose\nwindow",
    "pose_visual_window_pca": "Pose+visual PCA\nwindow",
}
MODEL_LABELS = {"rf": "RF", "xgb": "XGBoost"}
MINUS = "\N{MINUS SIGN}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("outputs/controlled_comparison_runs/run_20260528_205727/manuscript_figures/interim_qc/classical_rf_xgb"),
    )
    parser.add_argument(
        "--suite_root",
        type=Path,
        default=Path("outputs/controlled_comparison_runs/run_20260528_205727"),
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=Path("Manuscript_penultimate/figures/production/classical_baselines"),
    )
    parser.add_argument(
        "--table_dir",
        type=Path,
        default=Path("Manuscript_penultimate/tables/production/classical_baselines"),
    )
    return parser.parse_args()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def mean_sd(df: pd.DataFrame, group_cols: list[str], value_cols: list[str]) -> pd.DataFrame:
    grouped = df.groupby(group_cols, as_index=False)
    out = grouped.agg(**{f"{col}_mean": (col, "mean") for col in value_cols})
    sd = grouped.agg(**{f"{col}_sd": (col, "std") for col in value_cols})
    for col in sd.columns:
        if col not in group_cols:
            out[col] = sd[col]
    return out


def load_data(source_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training = read_csv(source_dir / "source_training_seed_level.csv")
    heldout = read_csv(source_dir / "source_heldout_video_mean_seed_level.csv")
    per_behavior = read_csv(source_dir / "source_per_behavior_seed_level.csv")
    training = training[training["backend"].eq(BACKEND)].copy()
    heldout = heldout[heldout["backend"].eq(BACKEND)].copy()
    per_behavior = per_behavior[per_behavior["backend"].eq(BACKEND)].copy()
    return training, heldout, per_behavior


def jitter(model: str) -> float:
    return -0.11 if model == "rf" else 0.11


def draw_validation_scatter(ax, training: pd.DataFrame, heldout: pd.DataFrame) -> None:
    merged = training.merge(
        heldout,
        on=["backend", "model", "feature_set", "seed"],
        validate="one_to_one",
        suffixes=("_val", "_heldout"),
    )
    for model in ["rf", "xgb"]:
        sub = merged[merged["model"].eq(model)]
        ax.scatter(
            sub["val_behavior_macro_f1"],
            sub["frame_macro_f1"],
            s=38,
            color=CLASSICAL_MODEL_COLORS[model],
            edgecolor=NEUTRAL["dark"],
            linewidth=0.45,
            alpha=0.88,
            label=MODEL_LABELS[model],
        )
    lo = min(float(merged["val_behavior_macro_f1"].min()), float(merged["frame_macro_f1"].min())) - 0.01
    hi = max(float(merged["val_behavior_macro_f1"].max()), float(merged["frame_macro_f1"].max())) + 0.01
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=NEUTRAL["mid"], linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Validation behavior macro-F1")
    ax.set_ylabel("Held-out frame macro-F1")
    ax.set_title("Validation-to-held-out relationship", fontsize=11)
    ax.legend(frameon=False, loc="lower right")


def draw_feature_performance(ax, summary: pd.DataFrame) -> None:
    x = np.arange(len(FEATURE_ORDER))
    for model in ["rf", "xgb"]:
        sub = summary[summary["model"].eq(model)].set_index("feature_set").loc[FEATURE_ORDER].reset_index()
        xpos = x + jitter(model)
        ax.errorbar(
            xpos,
            sub["frame_macro_f1_mean"],
            yerr=sub["frame_macro_f1_sd"].fillna(0),
            fmt="o",
            color=CLASSICAL_MODEL_COLORS[model],
            capsize=3,
            linewidth=1.5,
            markersize=5,
            label=MODEL_LABELS[model],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([FEATURE_LABELS[f] for f in FEATURE_ORDER])
    ax.set_ylabel("Held-out frame macro-F1")
    ax.set_title("Controlled feature access", fontsize=11)
    ax.legend(frameon=False, loc="best")


def draw_bout_thresholds(ax, summary: pd.DataFrame) -> None:
    thresholds = [0.10, 0.25, 0.50]
    metric_cols = ["bout_f1_iou10_mean", "bout_f1_iou25_mean", "bout_f1_iou50_mean"]
    line_styles = {"pose_window": "-", "pose_visual_window_pca": "--"}
    for model in ["rf", "xgb"]:
        for feature_set in ["pose_window", "pose_visual_window_pca"]:
            row = summary[
                summary["model"].eq(model) & summary["feature_set"].eq(feature_set)
            ].iloc[0]
            values = [float(row[col]) for col in metric_cols]
            ax.plot(
                thresholds,
                values,
                marker="o",
                linewidth=2.0,
                linestyle=line_styles[feature_set],
                color=CLASSICAL_MODEL_COLORS[model],
                label=f"{MODEL_LABELS[model]} {FEATURE_LABELS[feature_set].replace(chr(10), ' ')}",
            )
    ax.set_xticks(thresholds)
    ax.set_xlabel("Temporal IoU threshold")
    ax.set_ylabel("Held-out bout macro-F1")
    ax.set_title("Bout recovery by window feature set", fontsize=11)
    ax.legend(frameon=False, loc="best", fontsize=7)


def draw_feature_deltas(ax, summary: pd.DataFrame) -> None:
    rows = []
    indexed = summary.set_index(["model", "feature_set"])
    for model in ["rf", "xgb"]:
        window_gain = (
            indexed.loc[(model, "pose_window"), "frame_macro_f1_mean"]
            - indexed.loc[(model, "pose_frame"), "frame_macro_f1_mean"]
        )
        visual_window_gain = (
            indexed.loc[(model, "pose_visual_window_pca"), "frame_macro_f1_mean"]
            - indexed.loc[(model, "pose_window"), "frame_macro_f1_mean"]
        )
        rows.extend(
            [
                {"model": model, "contrast": f"Window {MINUS} frame\n(pose)", "delta": window_gain},
                {"model": model, "contrast": f"Visual PCA {MINUS} pose\n(window)", "delta": visual_window_gain},
            ]
        )
    delta = pd.DataFrame(rows)
    contrasts = list(delta["contrast"].drop_duplicates())
    x = np.arange(len(contrasts))
    for model in ["rf", "xgb"]:
        sub = delta[delta["model"].eq(model)].set_index("contrast").loc[contrasts].reset_index()
        ax.bar(
            x + jitter(model),
            sub["delta"],
            width=0.20,
            color=CLASSICAL_MODEL_COLORS[model],
            alpha=0.82,
            label=MODEL_LABELS[model],
        )
    ax.axhline(0, color=NEUTRAL["mid"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(contrasts)
    ax.set_ylabel("Delta held-out frame macro-F1")
    ax.set_title("Feature-control deltas", fontsize=11)
    ax.legend(frameon=False, loc="best")
    return delta


def compute_bout_window_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    indexed = summary.set_index(["model", "feature_set"])
    metrics = [
        ("TIoU 0.10", "bout_f1_iou10_mean"),
        ("TIoU 0.25", "bout_f1_iou25_mean"),
        ("TIoU 0.50", "bout_f1_iou50_mean"),
    ]
    for model in ["rf", "xgb"]:
        for label, col in metrics:
            rows.append(
                {
                    "model": model,
                    "metric": label,
                    "delta_pose_visual_window_pca_minus_pose_window": (
                        indexed.loc[(model, "pose_visual_window_pca"), col]
                        - indexed.loc[(model, "pose_window"), col]
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_provenance(args: argparse.Namespace, delta: pd.DataFrame, summary: pd.DataFrame) -> None:
    matrix_path = args.suite_root / "classical" / "matrices" / "mars_yolo_sppf" / "matrix_build_summary.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.is_file() else {}
    rows = [
        {"quantity": "backend", "value": "YOLO/SPPF"},
        {"quantity": "classical_models", "value": "Random Forest; XGBoost"},
        {"quantity": "feature_sets", "value": "; ".join(FEATURE_ORDER)},
        {"quantity": "visual_pca_fit_scope", "value": "training split only"},
        {"quantity": "pose_relation_dim_per_frame", "value": matrix.get("pose_relation_dim_per_frame", "")},
        {"quantity": "pose_visual_pca_dim_per_frame", "value": matrix.get("pose_visual_pca_dim_per_frame", "")},
        {"quantity": "window_size_frames", "value": matrix.get("train_manifest_window_size", "")},
    ]
    pd.DataFrame(rows).to_csv(args.table_dir / "fig7_yolo_classical_pca_provenance.csv", index=False)
    delta.to_csv(args.table_dir / "fig7_yolo_classical_feature_deltas.csv", index=False)
    compute_bout_window_deltas(summary).to_csv(
        args.table_dir / "fig7_yolo_classical_bout_window_deltas.csv", index=False
    )
    summary.to_csv(args.table_dir / "fig7_yolo_classical_baselines_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    training, heldout, _per_behavior = load_data(args.source_dir)
    summary = mean_sd(
        heldout,
        ["backend", "model", "feature_set"],
        ["frame_macro_f1", "bout_f1_iou10", "bout_f1_iou25", "bout_f1_iou50"],
    )
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()
    draw_validation_scatter(ax_a, training, heldout)
    draw_feature_performance(ax_b, summary)
    draw_bout_thresholds(ax_c, summary)
    delta = draw_feature_deltas(ax_d, summary)
    for label, ax in zip(["A", "B", "C", "D"], [ax_a, ax_b, ax_c, ax_d]):
        panel_label(ax, label, x=-0.14, y=1.12)
    save_figure(fig, args.figure_dir / "fig7_yolo_classical_baselines")
    plt.close(fig)
    write_provenance(args, delta, summary)
    print(f"Wrote {args.figure_dir / 'fig7_yolo_classical_baselines.png'}")


if __name__ == "__main__":
    main()
