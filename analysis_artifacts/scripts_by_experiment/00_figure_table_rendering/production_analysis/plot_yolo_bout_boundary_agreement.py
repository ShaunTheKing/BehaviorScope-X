"""Plot YOLO/SPPF MARS bout-boundary agreement from production tables."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
STYLE_DIR = THIS_DIR.parent / "current_analysis"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import (  # noqa: E402
    BEHAVIOR_COLORS,
    HEAD_COLORS,
    STREAM_COLORS,
    STREAM_LABELS,
    STREAM_SHORT_LABELS,
    apply_style,
    save_figure,
    title_with_panel,
)


BEHAVIOR_CLASSES = ["attack", "investigation", "mount"]
STREAM_ORDER = [
    "full",
    "no_group_visual",
    "pose_relations_only",
    "no_relations",
    "animal_visual_only",
    "visual_only",
]
THRESHOLDS = [0.10, 0.25, 0.50]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table_dir",
        type=Path,
        default=Path("Manuscript_penultimate/tables/production/yolo_sppf_main"),
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=Path("Manuscript_penultimate/figures/production/yolo_sppf_main"),
    )
    parser.add_argument("--prediction_type", default="smoothed_frames")
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--focus_head", default="attention", choices=["lstm", "attention"])
    parser.add_argument("--focus_capacity", type=int, default=256)
    return parser.parse_args()


def threshold_label(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def stream_order_index(values: pd.Series) -> pd.Series:
    order = {name: i for i, name in enumerate(STREAM_ORDER)}
    return values.map(order)


def load_tables(table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_class = pd.read_csv(table_dir / "yolo_lstm_attention_per_class_long.csv")
    matches = pd.read_csv(table_dir / "yolo_lstm_attention_bout_matches_long.csv.gz")
    best = pd.read_csv(table_dir / "yolo_lstm_attention_best_capacity_by_stream_head.csv")
    return per_class, matches, best


def aggregate_bout_counts(per_class: pd.DataFrame, prediction_type: str) -> pd.DataFrame:
    df = per_class[
        per_class["prediction_type"].eq(prediction_type)
        & per_class["class"].isin(BEHAVIOR_CLASSES)
    ].copy()
    group_cols = ["stream", "head", "capacity", "seed", "class"]
    rows = []
    for keys, sub in df.groupby(group_cols):
        base = dict(zip(group_cols, keys))
        for thr in THRESHOLDS:
            suffix = threshold_label(thr).replace(".", "")
            # Column names are iou10, iou25, iou50.
            suffix = {0.10: "iou10", 0.25: "iou25", 0.50: "iou50"}[thr]
            tp = float(sub[f"bout_tp_{suffix}"].sum())
            fp = float(sub[f"bout_fp_{suffix}"].sum())
            fn = float(sub[f"bout_fn_{suffix}"].sum())
            precision = tp / (tp + fp) if tp + fp > 0 else np.nan
            recall = tp / (tp + fn) if tp + fn > 0 else np.nan
            f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan
            rows.append(
                {
                    **base,
                    "threshold": thr,
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "bout_precision": precision,
                    "bout_recall": recall,
                    "bout_f1": f1,
                }
            )
    return pd.DataFrame(rows)


def summarize_boundary_matches(matches: pd.DataFrame, prediction_type: str) -> pd.DataFrame:
    df = matches[
        matches["prediction_type"].eq(prediction_type)
        & matches["class"].isin(BEHAVIOR_CLASSES)
    ].copy()
    df["pred_duration_frames"] = df["pred_end"] - df["pred_start"]
    df["gt_duration_frames"] = df["gt_end"] - df["gt_start"]
    df["duration_error_frames"] = df["pred_duration_frames"] - df["gt_duration_frames"]
    df["abs_duration_error_frames"] = df["duration_error_frames"].abs()
    df["abs_start_offset_frames"] = df["start_offset_frames"].abs()
    df["abs_end_offset_frames"] = df["end_offset_frames"].abs()
    df["abs_mean_boundary_offset_frames"] = (
        df["abs_start_offset_frames"] + df["abs_end_offset_frames"]
    ) / 2.0
    group_cols = ["stream", "head", "capacity", "seed", "class", "threshold"]
    return (
        df.groupby(group_cols, as_index=False)
        .agg(
            matched_bouts=("tiou", "size"),
            median_tiou=("tiou", "median"),
            mean_tiou=("tiou", "mean"),
            median_abs_start_offset_frames=("abs_start_offset_frames", "median"),
            median_abs_end_offset_frames=("abs_end_offset_frames", "median"),
            median_abs_mean_boundary_offset_frames=("abs_mean_boundary_offset_frames", "median"),
            q75_abs_mean_boundary_offset_frames=("abs_mean_boundary_offset_frames", lambda x: np.nanpercentile(x, 75)),
            median_duration_error_frames=("duration_error_frames", "median"),
            median_abs_duration_error_frames=("abs_duration_error_frames", "median"),
            median_pred_duration_frames=("pred_duration_frames", "median"),
            median_gt_duration_frames=("gt_duration_frames", "median"),
        )
    )


def seed_mean_bout_counts(counts: pd.DataFrame) -> pd.DataFrame:
    return (
        counts.groupby(["stream", "head", "capacity", "class", "threshold"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            bout_precision_mean=("bout_precision", "mean"),
            bout_precision_sd=("bout_precision", "std"),
            bout_recall_mean=("bout_recall", "mean"),
            bout_recall_sd=("bout_recall", "std"),
            bout_f1_mean=("bout_f1", "mean"),
            bout_f1_sd=("bout_f1", "std"),
            tp_mean=("tp", "mean"),
            fp_mean=("fp", "mean"),
            fn_mean=("fn", "mean"),
        )
    )


def seed_mean_boundary(boundary: pd.DataFrame) -> pd.DataFrame:
    return (
        boundary.groupby(["stream", "head", "capacity", "class", "threshold"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            matched_bouts_mean=("matched_bouts", "mean"),
            median_tiou_mean=("median_tiou", "mean"),
            median_tiou_sd=("median_tiou", "std"),
            median_abs_start_offset_frames_mean=("median_abs_start_offset_frames", "mean"),
            median_abs_start_offset_frames_sd=("median_abs_start_offset_frames", "std"),
            median_abs_end_offset_frames_mean=("median_abs_end_offset_frames", "mean"),
            median_abs_end_offset_frames_sd=("median_abs_end_offset_frames", "std"),
            median_abs_mean_boundary_offset_frames_mean=(
                "median_abs_mean_boundary_offset_frames",
                "mean",
            ),
            median_abs_mean_boundary_offset_frames_sd=(
                "median_abs_mean_boundary_offset_frames",
                "std",
            ),
            median_duration_error_frames_mean=("median_duration_error_frames", "mean"),
            median_duration_error_frames_sd=("median_duration_error_frames", "std"),
            median_abs_duration_error_frames_mean=("median_abs_duration_error_frames", "mean"),
            median_abs_duration_error_frames_sd=("median_abs_duration_error_frames", "std"),
            median_pred_duration_frames_mean=("median_pred_duration_frames", "mean"),
            median_pred_duration_frames_sd=("median_pred_duration_frames", "std"),
            median_gt_duration_frames_mean=("median_gt_duration_frames", "mean"),
            median_gt_duration_frames_sd=("median_gt_duration_frames", "std"),
        )
    )


def best_stream_head_capacity(best: pd.DataFrame, head: str) -> pd.DataFrame:
    sub = best[best["head"].eq(head)].copy()
    sub["capacity"] = sub["capacity"].astype(int)
    return sub[["stream", "head", "capacity"]]


def plot_figure(
    counts_summary: pd.DataFrame,
    boundary_summary: pd.DataFrame,
    best: pd.DataFrame,
    figure_dir: Path,
    threshold: float,
    focus_head: str,
    focus_capacity: int,
) -> None:
    apply_style()
    fig = plt.figure(figsize=(13.5, 10.8), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.95])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, :])

    # Panel A: full-stream macro bout F1 across TIoU thresholds, LSTM vs attention.
    full = counts_summary[
        counts_summary["stream"].eq("full")
        & counts_summary["capacity"].eq(focus_capacity)
    ].copy()
    full_macro = (
        full.groupby(["head", "threshold"], as_index=False)
        .agg(f1=("bout_f1_mean", "mean"), f1_sd=("bout_f1_mean", "std"))
        .sort_values(["head", "threshold"])
    )
    for head in ["lstm", "attention"]:
        sub = full_macro[full_macro["head"].eq(head)]
        ax_a.plot(
            sub["threshold"],
            sub["f1"],
            marker="o",
            linewidth=2.0,
            color=HEAD_COLORS[head],
            label=head.upper() if head == "lstm" else "Attention",
        )
    ax_a.set_xticks(THRESHOLDS)
    ax_a.set_xlabel("TIoU threshold")
    ax_a.set_ylabel("Bout F1")
    title_with_panel(ax_a, "A", f"Full-stream bout agreement ({focus_capacity})")
    ax_a.legend(frameon=False)

    # Panels B-D: class-specific LSTM vs attention comparisons.
    focus_boundary = boundary_summary[
        boundary_summary["stream"].eq("full")
        & boundary_summary["capacity"].eq(focus_capacity)
        & np.isclose(boundary_summary["threshold"], threshold)
    ].copy()
    focus_counts = counts_summary[
        counts_summary["stream"].eq("full")
        & counts_summary["capacity"].eq(focus_capacity)
        & np.isclose(counts_summary["threshold"], threshold)
    ].copy()
    focus_boundary["class"] = pd.Categorical(focus_boundary["class"], BEHAVIOR_CLASSES, ordered=True)
    focus_counts["class"] = pd.Categorical(focus_counts["class"], BEHAVIOR_CLASSES, ordered=True)
    focus_boundary = focus_boundary.sort_values("class")
    focus_counts = focus_counts.sort_values("class")
    x = np.arange(len(BEHAVIOR_CLASSES))
    width = 0.36

    def grouped_head_bars(ax, df: pd.DataFrame, value_col: str, err_col: str, ylabel: str, panel: str, title: str) -> None:
        for offset, head in [(-width / 2, "lstm"), (width / 2, "attention")]:
            sub = (
                df[df["head"].eq(head)]
                .sort_values("class")
                .set_index("class")
                .reindex(BEHAVIOR_CLASSES)
                .reset_index()
            )
            ax.bar(
                x + offset,
                sub[value_col].to_numpy(),
                width,
                yerr=sub[err_col].fillna(0).to_numpy() if err_col in sub.columns else None,
                color=HEAD_COLORS[head],
                capsize=3,
                alpha=0.9,
                edgecolor="white",
                linewidth=0.7,
                label=head.upper() if head == "lstm" else "Attention",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([c.capitalize() for c in BEHAVIOR_CLASSES], rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        title_with_panel(ax, panel, title)
        ax.legend(frameon=False, ncol=2)

    grouped_head_bars(
        ax_b,
        focus_counts,
        "bout_f1_mean",
        "bout_f1_sd",
        "Bout F1",
        "B",
        f"Class-wise bout F1 at TIoU {threshold:g}",
    )
    ax_b.set_ylim(0, 1.0)

    grouped_head_bars(
        ax_c,
        focus_boundary,
        "median_abs_mean_boundary_offset_frames_mean",
        "median_abs_mean_boundary_offset_frames_sd",
        "Median boundary offset (frames)",
        "C",
        f"Class-wise boundary offset at TIoU {threshold:g}",
    )

    grouped_head_bars(
        ax_d,
        focus_boundary,
        "median_tiou_mean",
        "median_tiou_sd",
        "Median matched-bout TIoU",
        "D",
        f"Class-wise matched-bout overlap at TIoU {threshold:g}",
    )
    ax_d.set_ylim(0, 1.0)

    grouped_head_bars(
        ax_e,
        focus_boundary,
        "median_duration_error_frames_mean",
        "median_duration_error_frames_sd",
        "Predicted − GT duration (frames)",
        "E",
        f"Matched-bout duration agreement at TIoU {threshold:g}",
    )
    ax_e.axhline(0, color="#5F6773", linewidth=1.0)

    save_figure(fig, figure_dir / "supp_yolo_bout_boundary_head_comparison")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table_dir = args.table_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)

    per_class, matches, best = load_tables(table_dir)
    counts = aggregate_bout_counts(per_class, args.prediction_type)
    boundary = summarize_boundary_matches(matches, args.prediction_type)
    counts_summary = seed_mean_bout_counts(counts)
    boundary_summary = seed_mean_boundary(boundary)

    full_head_comparison = counts_summary[
        counts_summary["stream"].eq("full")
        & counts_summary["capacity"].eq(args.focus_capacity)
        & np.isclose(counts_summary["threshold"], args.threshold)
        & counts_summary["class"].isin(BEHAVIOR_CLASSES)
    ].merge(
        boundary_summary[
            boundary_summary["stream"].eq("full")
            & boundary_summary["capacity"].eq(args.focus_capacity)
            & np.isclose(boundary_summary["threshold"], args.threshold)
            & boundary_summary["class"].isin(BEHAVIOR_CLASSES)
        ],
        on=["stream", "head", "capacity", "class", "threshold", "n_seeds"],
        how="left",
    )

    counts.to_csv(table_dir / "yolo_bout_boundary_counts_by_seed.csv", index=False)
    boundary.to_csv(table_dir / "yolo_bout_boundary_matches_by_seed.csv", index=False)
    counts_summary.to_csv(table_dir / "yolo_bout_boundary_counts_seed_summary.csv", index=False)
    boundary_summary.to_csv(table_dir / "yolo_bout_boundary_match_seed_summary.csv", index=False)
    full_head_comparison.to_csv(
        table_dir / "yolo_bout_boundary_full256_lstm_attention_class_summary.csv",
        index=False,
    )

    plot_figure(
        counts_summary,
        boundary_summary,
        best,
        figure_dir,
        args.threshold,
        args.focus_head,
        args.focus_capacity,
    )
    print(f"Wrote bout-boundary tables to {table_dir}")
    print(f"Wrote bout-boundary figure to {figure_dir}")


if __name__ == "__main__":
    main()
