"""Duration-dependent bout-recall operating envelope for YOLO/SPPF ethograms."""
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

from behaviorscope_figure_style import (  # noqa: E402
    BEHAVIOR_COLORS,
    HEAD_COLORS,
    apply_style,
    save_figure,
)


BEHAVIORS = ["attack", "investigation", "mount"]
BEHAVIOR_LABELS = {
    "attack": "Attack",
    "investigation": "Investigation",
    "mount": "Mount",
}
FOCUS_MODELS = [
    {
        "track": "Attention-256",
        "head": "attention",
        "capacity": 256,
        "color": HEAD_COLORS["attention"],
    },
    {
        "track": "LSTM-256",
        "head": "lstm",
        "capacity": 256,
        "color": "#1D4ED8",
    },
]
PREDICTION_LABELS = {
    "smoothed_frames": "Smoothed",
    "raw_windows": "Raw",
}
PREDICTION_LINESTYLES = {
    "smoothed_frames": "-",
    "raw_windows": "--",
}
PREDICTION_MARKERS = {
    "smoothed_frames": "o",
    "raw_windows": "s",
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
    parser.add_argument("--output_name", default="fig11_yolo_operating_envelope")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--primary_threshold", type=float, default=0.25)
    parser.add_argument("--strict_threshold", type=float, default=0.50)
    parser.add_argument("--reliable_recall", type=float, default=0.80)
    parser.add_argument("--min_bin_gt_bouts", type=int, default=10)
    return parser.parse_args()


def duration_bin_edges_frames(fps: float) -> np.ndarray:
    # Seconds: 0, 0.25, 0.5, 1, 2, 4, 8, infinity.
    seconds = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, np.inf])
    frames = seconds * fps
    frames[-1] = np.inf
    return frames


def duration_bin_labels() -> list[str]:
    return ["<0.25", "0.25-0.5", "0.5-1", "1-2", "2-4", "4-8", ">=8"]


def duration_bin_centers_seconds() -> list[float]:
    return [0.18, 0.35, 0.70, 1.4, 2.8, 5.6, 9.0]


def focus_model_filter(df: pd.DataFrame) -> pd.Series:
    mask = df["stream"].eq("full")
    combo_mask = pd.Series(False, index=df.index)
    for spec in FOCUS_MODELS:
        combo_mask |= df["head"].eq(spec["head"]) & df["capacity"].eq(spec["capacity"])
    return mask & combo_mask & df["class"].isin(BEHAVIORS)


def load_gt(table_dir: Path, fps: float) -> pd.DataFrame:
    cols = [
        "split",
        "video_id",
        "class",
        "start_frame",
        "end_frame",
        "duration_frames",
        "stream",
        "head",
        "capacity",
        "seed",
        "model_name",
    ]
    gt = pd.read_csv(table_dir / "yolo_lstm_attention_ground_truth_bouts_long.csv.gz", usecols=cols)
    gt = gt[focus_model_filter(gt)].copy()
    for col in ["start_frame", "end_frame", "duration_frames", "capacity", "seed"]:
        gt[col] = pd.to_numeric(gt[col], errors="coerce")
    gt = gt.sort_values(["model_name", "split", "video_id", "class", "start_frame", "end_frame"]).copy()
    gt["gt_bout_index"] = gt.groupby(["model_name", "split", "video_id", "class"]).cumcount()
    gt["duration_s"] = gt["duration_frames"] / fps
    gt["duration_bin"] = pd.cut(
        gt["duration_frames"],
        bins=duration_bin_edges_frames(fps),
        labels=duration_bin_labels(),
        include_lowest=True,
        right=False,
    )
    gt["track"] = gt["head"].map({"attention": "Attention", "lstm": "LSTM"}) + "-" + gt["capacity"].astype(int).astype(str)
    return gt


def load_matches(table_dir: Path) -> pd.DataFrame:
    cols = [
        "threshold",
        "class",
        "gt_bout_index",
        "split",
        "video_id",
        "prediction_type",
        "stream",
        "head",
        "capacity",
        "seed",
        "model_name",
        "tiou",
    ]
    matches = pd.read_csv(table_dir / "yolo_lstm_attention_bout_matches_long.csv.gz", usecols=cols)
    matches = matches[focus_model_filter(matches)].copy()
    matches["threshold"] = pd.to_numeric(matches["threshold"], errors="coerce")
    matches["gt_bout_index"] = pd.to_numeric(matches["gt_bout_index"], errors="coerce").astype(int)
    matches["capacity"] = pd.to_numeric(matches["capacity"], errors="coerce").astype(int)
    matches["seed"] = pd.to_numeric(matches["seed"], errors="coerce").astype(int)
    matches = matches[matches["prediction_type"].isin(PREDICTION_LABELS)].copy()
    return matches


def build_gt_recall_table(gt: pd.DataFrame, matches: pd.DataFrame, thresholds: list[float]) -> pd.DataFrame:
    eval_rows = []
    for prediction_type in PREDICTION_LABELS:
        for threshold in thresholds:
            base = gt.copy()
            base["prediction_type"] = prediction_type
            base["threshold"] = threshold
            eval_rows.append(base)
    recall = pd.concat(eval_rows, ignore_index=True)
    key_cols = [
        "model_name",
        "split",
        "video_id",
        "class",
        "gt_bout_index",
        "prediction_type",
        "threshold",
    ]
    matched_keys = (
        matches[matches["threshold"].isin(thresholds)][key_cols]
        .drop_duplicates()
        .assign(matched=1)
    )
    recall = recall.merge(matched_keys, on=key_cols, how="left")
    recall["matched"] = recall["matched"].fillna(0).astype(int)
    return recall


def summarize_by_seed(recall: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_bin = (
        recall.groupby(
            ["track", "head", "capacity", "seed", "prediction_type", "threshold", "class", "duration_bin"],
            observed=True,
            as_index=False,
        )
        .agg(gt_bouts=("matched", "size"), matched_bouts=("matched", "sum"), recall=("matched", "mean"))
    )
    model_bin = (
        seed_bin.groupby(["track", "head", "capacity", "prediction_type", "threshold", "class", "duration_bin"], observed=True, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            gt_bouts_mean=("gt_bouts", "mean"),
            matched_bouts_mean=("matched_bouts", "mean"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
        )
    )
    seed_class = (
        recall.groupby(["track", "head", "capacity", "seed", "prediction_type", "threshold", "class"], as_index=False)
        .agg(gt_bouts=("matched", "size"), matched_bouts=("matched", "sum"), recall=("matched", "mean"))
    )
    class_summary = (
        seed_class.groupby(["track", "head", "capacity", "prediction_type", "threshold", "class"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            gt_bouts_mean=("gt_bouts", "mean"),
            matched_bouts_mean=("matched_bouts", "mean"),
            recall_mean=("recall", "mean"),
            recall_sd=("recall", "std"),
        )
    )
    return seed_bin, model_bin, class_summary


def reliable_duration_table(model_bin: pd.DataFrame, fps: float, target_recall: float) -> pd.DataFrame:
    centers = dict(zip(duration_bin_labels(), duration_bin_centers_seconds()))
    rows = []
    for keys, sub in model_bin.groupby(["track", "prediction_type", "threshold", "class"], observed=True):
        sub = sub.sort_values("duration_bin")
        candidates = sub[sub["recall_mean"] >= target_recall].copy()
        reliable_bin = str(candidates.iloc[0]["duration_bin"]) if not candidates.empty else "not reached"
        reliable_s = centers.get(reliable_bin, np.nan)
        rows.append(
            {
                "track": keys[0],
                "prediction_type": keys[1],
                "threshold": keys[2],
                "class": keys[3],
                "target_recall": target_recall,
                "first_duration_bin_reaching_target": reliable_bin,
                "approx_duration_s": reliable_s,
            }
        )
    return pd.DataFrame(rows)


def plot_operating_envelope(
    model_bin: pd.DataFrame,
    class_summary: pd.DataFrame,
    reliable: pd.DataFrame,
    figure_dir: Path,
    output_name: str,
    primary_threshold: float,
    strict_threshold: float,
    target_recall: float,
    min_bin_gt_bouts: int,
) -> None:
    apply_style()
    fig = plt.figure(figsize=(14.0, 10.0), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, top=0.91, bottom=0.10, hspace=0.34, wspace=0.28)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    centers = np.array(duration_bin_centers_seconds())
    labels = duration_bin_labels()

    for ax, spec, panel in zip(axes[:2], FOCUS_MODELS, ["A", "B"]):
        track = spec["track"]
        for prediction_type in ["smoothed_frames", "raw_windows"]:
            for behavior in BEHAVIORS:
                sub = (
                    model_bin[
                        model_bin["track"].eq(track)
                        & model_bin["prediction_type"].eq(prediction_type)
                        & np.isclose(model_bin["threshold"], primary_threshold)
                        & model_bin["class"].eq(behavior)
                    ]
                    .set_index("duration_bin")
                    .reindex(labels)
                    .reset_index()
                )
                low_count = sub["gt_bouts_mean"].fillna(0) < min_bin_gt_bouts
                sub.loc[low_count, ["recall_mean", "recall_sd"]] = np.nan
                ax.plot(
                    centers,
                    sub["recall_mean"],
                    linestyle=PREDICTION_LINESTYLES[prediction_type],
                    marker=PREDICTION_MARKERS[prediction_type],
                    color=BEHAVIOR_COLORS[behavior],
                    linewidth=2.0,
                    label=f"{BEHAVIOR_LABELS[behavior]} ({PREDICTION_LABELS[prediction_type]})",
                )
                ax.fill_between(
                    centers,
                    sub["recall_mean"] - sub["recall_sd"].fillna(0),
                    sub["recall_mean"] + sub["recall_sd"].fillna(0),
                    color=BEHAVIOR_COLORS[behavior],
                    alpha=0.08,
                )
        ax.axhline(target_recall, color="#6B7280", linewidth=1.0)
        ax.set_xscale("log")
        ax.set_xticks(centers)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Ground-truth bout duration (s)")
        ax.set_ylabel(f"Bout recall at TIoU {primary_threshold:g}")
        ax.set_title(f"{panel}. Operating envelope: {track}", loc="left")
        if panel == "A":
            ax.legend(frameon=False, fontsize=7, ncol=2)

    # Panel C: strictness effect for smoothed outputs at class level.
    ax = axes[2]
    x = np.arange(len(BEHAVIORS))
    width = 0.18
    offsets = [-1.5, -0.5, 0.5, 1.5]
    bars = [
        ("Attention-256", primary_threshold, "#10B981", "Attn 0.25"),
        ("Attention-256", strict_threshold, "#A7F3D0", "Attn 0.50"),
        ("LSTM-256", primary_threshold, "#1D4ED8", "LSTM 0.25"),
        ("LSTM-256", strict_threshold, "#93C5FD", "LSTM 0.50"),
    ]
    for offset, (track, threshold, color, label) in zip(offsets, bars):
        sub = (
            class_summary[
                class_summary["track"].eq(track)
                & class_summary["prediction_type"].eq("smoothed_frames")
                & np.isclose(class_summary["threshold"], threshold)
            ]
            .set_index("class")
            .reindex(BEHAVIORS)
            .reset_index()
        )
        ax.bar(
            x + offset * width,
            sub["recall_mean"],
            width,
            yerr=sub["recall_sd"].fillna(0),
            color=color,
            edgecolor="white",
            linewidth=0.7,
            capsize=2,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([BEHAVIOR_LABELS[b] for b in BEHAVIORS])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Bout recall")
    ax.set_title("C. Boundary strictness reduces recall", loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=2)

    # Panel D: ground-truth duration burden.
    ax = axes[3]
    gt_dist = model_bin[
        model_bin["track"].eq("Attention-256")
        & model_bin["prediction_type"].eq("smoothed_frames")
        & np.isclose(model_bin["threshold"], primary_threshold)
    ].copy()
    for behavior in BEHAVIORS:
        sub = gt_dist[gt_dist["class"].eq(behavior)].set_index("duration_bin").reindex(labels).reset_index()
        fractions = sub["gt_bouts_mean"] / sub["gt_bouts_mean"].sum()
        ax.plot(
            centers,
            fractions,
            marker="o",
            linewidth=2.0,
            color=BEHAVIOR_COLORS[behavior],
            label=BEHAVIOR_LABELS[behavior],
        )
    ax.set_xscale("log")
    ax.set_xticks(centers)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, None)
    ax.set_xlabel("Ground-truth bout duration (s)")
    ax.set_ylabel("Fraction of GT bouts")
    ax.set_title("D. Ground-truth short-bout burden", loc="left")
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("Operating envelope: bout recovery depends on biological duration scale", fontsize=14)
    save_figure(fig, figure_dir / output_name)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    thresholds = [args.primary_threshold, args.strict_threshold]
    gt = load_gt(args.table_dir, args.fps)
    matches = load_matches(args.table_dir)
    recall = build_gt_recall_table(gt, matches, thresholds)
    seed_bin, model_bin, class_summary = summarize_by_seed(recall)
    reliable = reliable_duration_table(model_bin, args.fps, args.reliable_recall)

    recall.to_csv(args.table_dir / f"{args.output_name}_gt_bout_level.csv", index=False)
    seed_bin.to_csv(args.table_dir / f"{args.output_name}_duration_bin_seed_summary.csv", index=False)
    model_bin.to_csv(args.table_dir / f"{args.output_name}_duration_bin_model_summary.csv", index=False)
    class_summary.to_csv(args.table_dir / f"{args.output_name}_class_recall_summary.csv", index=False)
    reliable.to_csv(args.table_dir / f"{args.output_name}_reliable_duration_summary.csv", index=False)

    plot_operating_envelope(
        model_bin,
        class_summary,
        reliable,
        args.figure_dir,
        args.output_name,
        args.primary_threshold,
        args.strict_threshold,
        args.reliable_recall,
        args.min_bin_gt_bouts,
    )
    print("Reliable duration summary:")
    print(
        reliable[
            reliable["prediction_type"].eq("smoothed_frames")
            & reliable["threshold"].eq(args.primary_threshold)
        ].to_string(index=False)
    )
    print(f"\nWrote {args.figure_dir / (args.output_name + '.png')}")


if __name__ == "__main__":
    main()
