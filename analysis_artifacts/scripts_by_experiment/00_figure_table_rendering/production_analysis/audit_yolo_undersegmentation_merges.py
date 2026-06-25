"""Audit whether YOLO/SPPF predictions merge multiple GT bouts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_DIR = SCRIPT_DIR.parent / "current_analysis"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import apply_style, save_figure  # noqa: E402


BEHAVIORS = ["attack", "investigation", "mount"]
TRACK_ORDER = [
    "Attention-256",
    "Attention-512",
    "Attention-896",
    "LSTM-256",
    "LSTM-512",
    "LSTM-896",
]
TRACK_COLORS = {
    "Attention-256": "#10B981",
    "Attention-512": "#34D399",
    "Attention-896": "#6EE7B7",
    "LSTM-256": "#2271B2",
    "LSTM-512": "#60A5FA",
    "LSTM-896": "#93C5FD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--smoothed_all_bouts_csv",
        type=Path,
        default=Path(
            "Manuscript_penultimate/tables/production/yolo_sppf_main/"
            "supp_yolo_bout_fragmentation_audit_all_capacities_all_bouts_long.csv"
        ),
    )
    parser.add_argument(
        "--raw_all_bouts_csv",
        type=Path,
        default=Path(
            "Manuscript_penultimate/tables/production/yolo_sppf_main/"
            "supp_yolo_bout_fragmentation_audit_all_capacities_raw_windows_all_bouts_long.csv"
        ),
    )
    parser.add_argument("--output_name", default="fig9_yolo_bout_undersegmentation_merge_audit")
    return parser.parse_args()


def load_bouts(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["class"].isin(BEHAVIORS)].copy()
    for col in ["start_frame", "end_frame", "duration_frames", "capacity", "seed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_comparison_bouts(smoothed_path: Path, raw_path: Path) -> pd.DataFrame:
    smoothed = load_bouts(smoothed_path)
    raw_pred = load_bouts(raw_path)
    gt = smoothed[smoothed["track"].eq("Ground truth")].copy()
    gt_raw = gt.copy()
    gt_raw["prediction_type"] = "raw_windows"
    raw = pd.concat([gt_raw, raw_pred], ignore_index=True)
    return pd.concat([smoothed, raw], ignore_index=True)


def build_pred_merge_detail(all_bouts: pd.DataFrame) -> pd.DataFrame:
    gt = all_bouts[all_bouts["track"].eq("Ground truth")].copy()
    pred = all_bouts[~all_bouts["track"].eq("Ground truth")].copy()
    rows = []
    gt_groups = {
        key: sub[["start_frame", "end_frame"]].to_numpy(dtype=np.int64)
        for key, sub in gt.groupby(["prediction_type", "split", "video_id", "class"], sort=False)
    }
    for key, sub in pred.groupby(["prediction_type", "split", "video_id", "class"], sort=False):
        gt_intervals = gt_groups.get(key)
        if gt_intervals is None or len(gt_intervals) == 0:
            n_gt = np.zeros(len(sub), dtype=np.int64)
            overlap_frames = np.zeros(len(sub), dtype=np.int64)
        else:
            pred_starts = sub["start_frame"].to_numpy(dtype=np.int64)[:, None]
            pred_ends = sub["end_frame"].to_numpy(dtype=np.int64)[:, None]
            gt_starts = gt_intervals[:, 0][None, :]
            gt_ends = gt_intervals[:, 1][None, :]
            starts = np.maximum(pred_starts, gt_starts)
            ends = np.minimum(pred_ends, gt_ends)
            overlap = np.maximum(0, ends - starts + 1)
            n_gt = (overlap > 0).sum(axis=1).astype(np.int64)
            overlap_frames = overlap.sum(axis=1).astype(np.int64)
        detail = sub.copy()
        detail["pred_start_frame"] = detail["start_frame"].astype(int)
        detail["pred_end_frame"] = detail["end_frame"].astype(int)
        detail["pred_duration_frames"] = detail["duration_frames"].astype(int)
        detail["overlapping_gt_bouts"] = n_gt
        detail["overlap_gt_frames"] = overlap_frames
        detail["merges_multiple_gt_bouts"] = detail["overlapping_gt_bouts"] > 1
        rows.append(
            detail[
                [
                    "prediction_type",
                    "track",
                    "head",
                    "capacity",
                    "seed",
                    "model_name",
                    "split",
                    "video_id",
                    "class",
                    "pred_start_frame",
                    "pred_end_frame",
                    "pred_duration_frames",
                    "overlapping_gt_bouts",
                    "overlap_gt_frames",
                    "merges_multiple_gt_bouts",
                ]
            ]
        )
    return pd.concat(rows, ignore_index=True)


def summarize(all_bouts: pd.DataFrame, merge_detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_counts = (
        all_bouts[all_bouts["track"].eq("Ground truth")]
        .groupby(["prediction_type", "class"], as_index=False)
        .agg(
            gt_bouts=("duration_frames", "size"),
            gt_median_duration_frames=("duration_frames", "median"),
        )
    )
    seed_summary = (
        merge_detail.groupby(["prediction_type", "track", "head", "capacity", "seed", "class"], as_index=False)
        .agg(
            pred_bouts=("pred_duration_frames", "size"),
            pred_median_duration_frames=("pred_duration_frames", "median"),
            fraction_pred_bouts_with_any_gt=("overlapping_gt_bouts", lambda x: float(np.mean(np.asarray(x) > 0))),
            fraction_pred_bouts_merging_gt=("merges_multiple_gt_bouts", "mean"),
            mean_overlapping_gt_bouts=("overlapping_gt_bouts", "mean"),
            median_overlapping_gt_bouts=("overlapping_gt_bouts", "median"),
        )
        .merge(gt_counts, on=["prediction_type", "class"], validate="many_to_one")
    )
    seed_summary["bout_count_ratio_pred_gt"] = seed_summary["pred_bouts"] / seed_summary["gt_bouts"]
    seed_summary["median_duration_ratio_pred_gt"] = (
        seed_summary["pred_median_duration_frames"] / seed_summary["gt_median_duration_frames"]
    )
    model_summary = (
        seed_summary.groupby(["prediction_type", "track", "head", "capacity", "class"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            pred_bouts_mean=("pred_bouts", "mean"),
            pred_bouts_sd=("pred_bouts", "std"),
            bout_count_ratio_pred_gt_mean=("bout_count_ratio_pred_gt", "mean"),
            bout_count_ratio_pred_gt_sd=("bout_count_ratio_pred_gt", "std"),
            pred_median_duration_frames_mean=("pred_median_duration_frames", "mean"),
            pred_median_duration_frames_sd=("pred_median_duration_frames", "std"),
            median_duration_ratio_pred_gt_mean=("median_duration_ratio_pred_gt", "mean"),
            median_duration_ratio_pred_gt_sd=("median_duration_ratio_pred_gt", "std"),
            fraction_pred_bouts_with_any_gt_mean=("fraction_pred_bouts_with_any_gt", "mean"),
            fraction_pred_bouts_with_any_gt_sd=("fraction_pred_bouts_with_any_gt", "std"),
            fraction_pred_bouts_merging_gt_mean=("fraction_pred_bouts_merging_gt", "mean"),
            fraction_pred_bouts_merging_gt_sd=("fraction_pred_bouts_merging_gt", "std"),
            mean_overlapping_gt_bouts_mean=("mean_overlapping_gt_bouts", "mean"),
            mean_overlapping_gt_bouts_sd=("mean_overlapping_gt_bouts", "std"),
            gt_bouts=("gt_bouts", "first"),
            gt_median_duration_frames=("gt_median_duration_frames", "first"),
        )
    )
    return seed_summary, model_summary


def plot(model_summary: pd.DataFrame, figure_dir: Path, output_name: str) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.90, bottom=0.20, hspace=0.48, wspace=0.24)
    axes = axes.ravel()
    x = np.arange(len(TRACK_ORDER))

    panel_specs = [
        (
            "bout_count_ratio_pred_gt_mean",
            "bout_count_ratio_pred_gt_sd",
            "Predicted / GT bout count",
            "A. Bout-count recovery",
            1.0,
        ),
        (
            "median_duration_ratio_pred_gt_mean",
            "median_duration_ratio_pred_gt_sd",
            "Predicted / GT median duration",
            "B. Bout-duration inflation",
            1.0,
        ),
        (
            "fraction_pred_bouts_merging_gt_mean",
            "fraction_pred_bouts_merging_gt_sd",
            "Fraction of predicted bouts",
            "C. Predicted bouts spanning multiple GT bouts",
            0.0,
        ),
        (
            "mean_overlapping_gt_bouts_mean",
            "mean_overlapping_gt_bouts_sd",
            "Mean overlapping GT bouts per prediction",
            "D. GT bouts represented per predicted bout",
            1.0,
        ),
    ]
    behavior_colors = {"attack": "#D55E00", "investigation": "#0072B2", "mount": "#009E73"}
    behavior_handles = None
    behavior_labels = None
    for ax, (value_col, err_col, ylabel, title, reference) in zip(axes, panel_specs):
        for prediction_type, linestyle, marker, alpha in [
            ("smoothed_frames", "-", "o", 0.12),
            ("raw_windows", "--", "s", 0.06),
        ]:
            for behavior in BEHAVIORS:
                sub = (
                    model_summary[
                        model_summary["class"].eq(behavior)
                        & model_summary["prediction_type"].eq(prediction_type)
                    ]
                    .set_index("track")
                    .reindex(TRACK_ORDER)
                    .reset_index()
                )
                label = behavior.capitalize() if prediction_type == "smoothed_frames" else None
                ax.plot(
                    x,
                    sub[value_col],
                    marker=marker,
                    linestyle=linestyle,
                    linewidth=2.0,
                    color=behavior_colors[behavior],
                    label=label,
                )
                ax.fill_between(
                    x,
                    sub[value_col] - sub[err_col].fillna(0),
                    sub[value_col] + sub[err_col].fillna(0),
                    color=behavior_colors[behavior],
                    alpha=alpha,
                )
        ax.axhline(reference, color="#6B7280", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(TRACK_ORDER, rotation=30, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        if behavior_handles is None:
            behavior_handles, behavior_labels = ax.get_legend_handles_labels()

    fig.suptitle("Under-segmentation audit across YOLO/SPPF full-stream models", fontsize=13)
    if behavior_handles is not None:
        style_handles = [
            plt.Line2D([0], [0], color="#374151", linestyle="-", marker="o", linewidth=2, label="Smoothed"),
            plt.Line2D([0], [0], color="#374151", linestyle="--", marker="s", linewidth=2, label="Raw"),
        ]
        fig.legend(
            behavior_handles + style_handles,
            behavior_labels + ["Smoothed", "Raw"],
            loc="lower center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.035),
        )
    save_figure(fig, figure_dir / output_name)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    all_bouts = load_comparison_bouts(args.smoothed_all_bouts_csv, args.raw_all_bouts_csv)
    merge_detail = build_pred_merge_detail(all_bouts)
    seed_summary, model_summary = summarize(all_bouts, merge_detail)
    merge_detail.to_csv(args.table_dir / f"{args.output_name}_prediction_overlap_detail.csv", index=False)
    seed_summary.to_csv(args.table_dir / f"{args.output_name}_seed_summary.csv", index=False)
    model_summary.to_csv(args.table_dir / f"{args.output_name}_model_summary.csv", index=False)
    plot(model_summary, args.figure_dir, args.output_name)
    print(model_summary.to_string(index=False))
    print(f"Wrote {args.figure_dir / (args.output_name + '.png')}")


if __name__ == "__main__":
    main()
