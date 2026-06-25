"""Audit bout fragmentation across YOLO/SPPF full-stream capacities."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_DIR = SCRIPT_DIR.parent / "current_analysis"
sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import apply_style, save_figure  # noqa: E402


BEHAVIORS = ["attack", "investigation", "mount"]
MODEL_RE = re.compile(r"^yolo_sppf_full_(?P<head>lstm|attention)(?P<capacity>\d+)_seed(?P<seed>\d+)$")
HEAD_ORDER = ["attention", "lstm"]
CAPACITY_ORDER = [256, 512, 896]
TRACK_ORDER = ["Ground truth"] + [
    f"{head.capitalize() if head == 'attention' else 'LSTM'}-{capacity}"
    for head in HEAD_ORDER
    for capacity in CAPACITY_ORDER
]
TRACK_COLORS = {
    "Ground truth": "#6B7280",
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
        default=Path("Manuscript_penultimate/figures/production/supplement"),
    )
    parser.add_argument("--prediction_type", default="smoothed_frames")
    parser.add_argument("--reference_model", default="yolo_sppf_full_attention256_seed42")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output_name", default="supp_yolo_bout_fragmentation_audit_all_capacities")
    return parser.parse_args()


def load_inputs(table_dir: Path, prediction_type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_video = pd.read_csv(table_dir / "yolo_lstm_attention_per_video_long.csv")
    per_video = per_video[per_video["prediction_type"].eq(prediction_type)].copy()
    cols = [
        "split",
        "video_id",
        "prediction_type",
        "class",
        "start_frame",
        "end_frame",
        "duration_frames",
        "model_name",
    ]
    gt = pd.read_csv(table_dir / "yolo_lstm_attention_ground_truth_bouts_long.csv.gz", usecols=cols)
    pred = pd.read_csv(table_dir / "yolo_lstm_attention_predicted_bouts_long.csv.gz", usecols=cols)
    gt = gt[gt["prediction_type"].eq(prediction_type)].copy()
    pred = pred[pred["prediction_type"].eq(prediction_type)].copy()
    for df in [gt, pred]:
        df["start_frame"] = pd.to_numeric(df["start_frame"], errors="coerce").astype(int)
        df["end_frame"] = pd.to_numeric(df["end_frame"], errors="coerce").astype(int)
        df["duration_frames"] = pd.to_numeric(df["duration_frames"], errors="coerce").astype(int)
    return per_video, gt, pred


def video_units(per_video: pd.DataFrame, reference_model: str) -> pd.DataFrame:
    units = (
        per_video[per_video["model_name"].eq(reference_model)][["split", "video_id", "n_frames"]]
        .drop_duplicates()
        .copy()
    )
    if units.empty:
        raise ValueError(f"No held-out video units found for reference model {reference_model}")
    return units


def discover_full_models(per_video: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for model_name in sorted(per_video["model_name"].dropna().unique()):
        match = MODEL_RE.match(str(model_name))
        if not match:
            continue
        head = match.group("head")
        capacity = int(match.group("capacity"))
        seed = int(match.group("seed"))
        if capacity not in CAPACITY_ORDER:
            continue
        label = f"{head.capitalize() if head == 'attention' else 'LSTM'}-{capacity}"
        rows.append(
            {
                "track": label,
                "head": head,
                "capacity": capacity,
                "seed": seed,
                "model_name": str(model_name),
            }
        )
    rows = sorted(rows, key=lambda r: (HEAD_ORDER.index(r["head"]), r["capacity"], r["seed"]))
    if not rows:
        raise ValueError("No full-stream YOLO/SPPF LSTM/attention models were found in per-video table.")
    return rows


def track_bouts(table: pd.DataFrame, split: str, video_id: str, model_name: str | None) -> pd.DataFrame:
    sub = table[table["split"].eq(split) & table["video_id"].eq(video_id)].copy()
    if model_name is None:
        sub = sub.drop_duplicates(["split", "video_id", "class", "start_frame", "end_frame"]).copy()
    else:
        sub = sub[sub["model_name"].eq(model_name)].copy()
    return sub[sub["class"].isin(BEHAVIORS)].sort_values(["start_frame", "end_frame", "class"]).reset_index(drop=True)


def consecutive_transition_rows(meta: dict, bouts: pd.DataFrame, fps: float) -> list[dict]:
    rows: list[dict] = []
    if len(bouts) < 2:
        return rows
    ordered = bouts.sort_values(["start_frame", "end_frame", "class"]).reset_index(drop=True)
    for idx in range(len(ordered) - 1):
        cur = ordered.iloc[idx]
        nxt = ordered.iloc[idx + 1]
        gap = int(nxt["start_frame"]) - int(cur["end_frame"]) - 1
        rows.append(
            {
                **meta,
                "from_behavior": str(cur["class"]),
                "to_behavior": str(nxt["class"]),
                "same_class": bool(cur["class"] == nxt["class"]),
                "gap_frames": gap,
                "gap_s": gap / fps,
                "current_duration_frames": int(cur["duration_frames"]),
                "next_duration_frames": int(nxt["duration_frames"]),
            }
        )
    return rows


def interval_union_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals = sorted(intervals)
    total = 0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end + 1:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start + 1
            cur_start, cur_end = start, end
    total += cur_end - cur_start + 1
    return total


def gt_overlap_rows(gt: pd.DataFrame, pred: pd.DataFrame, meta: dict) -> list[dict]:
    rows: list[dict] = []
    for _, gt_bout in gt.iterrows():
        cls = str(gt_bout["class"])
        same_class_preds = pred[pred["class"].eq(cls)].copy()
        intervals: list[tuple[int, int]] = []
        overlapping = 0
        for _, pr in same_class_preds.iterrows():
            start = max(int(gt_bout["start_frame"]), int(pr["start_frame"]))
            end = min(int(gt_bout["end_frame"]), int(pr["end_frame"]))
            if end >= start:
                overlapping += 1
                intervals.append((start, end))
        covered = interval_union_length(intervals)
        gt_len = int(gt_bout["end_frame"]) - int(gt_bout["start_frame"]) + 1
        rows.append(
            {
                **meta,
                "class": cls,
                "gt_start_frame": int(gt_bout["start_frame"]),
                "gt_end_frame": int(gt_bout["end_frame"]),
                "gt_duration_frames": int(gt_bout["duration_frames"]),
                "overlapping_pred_bouts": overlapping,
                "covered_gt_frames": covered,
                "gt_coverage_fraction": covered / max(1, gt_len),
            }
        )
    return rows


def aggregate_seed_summaries(
    all_bouts: pd.DataFrame,
    transitions: pd.DataFrame,
    overlaps: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    bout_seed = (
        all_bouts.groupby(["track", "head", "capacity", "seed", "class"], as_index=False)
        .agg(
            n_bouts=("duration_frames", "size"),
            median_duration_frames=("duration_frames", "median"),
            mean_duration_frames=("duration_frames", "mean"),
            total_duration_frames=("duration_frames", "sum"),
        )
        .sort_values(["head", "capacity", "seed", "class"])
    )
    transition_seed = (
        transitions.groupby(["track", "head", "capacity", "seed", "from_behavior"], as_index=False)
        .agg(
            n_transitions=("same_class", "size"),
            same_class_transitions=("same_class", "sum"),
            same_class_transition_fraction=("same_class", "mean"),
            median_gap_frames=("gap_frames", "median"),
        )
        .sort_values(["head", "capacity", "seed", "from_behavior"])
    )
    same = transitions[transitions["same_class"]].copy()
    same_seed = (
        same.groupby(["track", "head", "capacity", "seed", "from_behavior"], as_index=False)
        .agg(
            same_class_transitions=("same_class", "size"),
            median_same_class_gap_frames=("gap_frames", "median"),
            p25_same_class_gap_frames=("gap_frames", lambda x: np.percentile(x, 25)),
            p75_same_class_gap_frames=("gap_frames", lambda x: np.percentile(x, 75)),
            fraction_same_class_gap_le_15_frames=("gap_frames", lambda x: float(np.mean(np.asarray(x) <= 15))),
        )
        .sort_values(["head", "capacity", "seed", "from_behavior"])
    )
    overlap_seed = (
        overlaps.groupby(["track", "head", "capacity", "seed", "class"], as_index=False)
        .agg(
            gt_bouts=("overlapping_pred_bouts", "size"),
            gt_bouts_with_any_overlap=("overlapping_pred_bouts", lambda x: int(np.sum(np.asarray(x) > 0))),
            gt_bouts_split_by_multiple_predictions=("overlapping_pred_bouts", lambda x: int(np.sum(np.asarray(x) > 1))),
            fraction_gt_bouts_split=("overlapping_pred_bouts", lambda x: float(np.mean(np.asarray(x) > 1))),
            mean_gt_coverage_fraction=("gt_coverage_fraction", "mean"),
        )
        .sort_values(["head", "capacity", "seed", "class"])
    )
    return {
        "bout_seed": bout_seed,
        "transition_seed": transition_seed,
        "same_seed": same_seed,
        "overlap_seed": overlap_seed,
    }


def mean_sd(seed_df: pd.DataFrame, group_cols: list[str], metrics: list[str]) -> pd.DataFrame:
    return (
        seed_df.groupby(group_cols, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            **{f"{m}_mean": (m, "mean") for m in metrics},
            **{f"{m}_sd": (m, "std") for m in metrics},
        )
    )


def add_ground_truth_reference(
    seed_df: pd.DataFrame,
    group_col: str,
    metric_cols: list[str],
    class_col: str = "class",
) -> pd.DataFrame:
    gt = seed_df[seed_df["track"].eq("Ground truth")].copy()
    gt["n_seeds"] = 1
    for metric in metric_cols:
        gt[f"{metric}_mean"] = gt[metric]
        gt[f"{metric}_sd"] = 0.0
    cols = ["track", "head", "capacity", group_col, "n_seeds"] + [f"{m}_mean" for m in metric_cols] + [
        f"{m}_sd" for m in metric_cols
    ]
    if group_col != class_col and class_col in gt.columns:
        pass
    return gt[cols]


def summarize_all(seed_summaries: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    bout_model = mean_sd(
        seed_summaries["bout_seed"][~seed_summaries["bout_seed"]["track"].eq("Ground truth")],
        ["track", "head", "capacity", "class"],
        ["n_bouts", "median_duration_frames", "mean_duration_frames", "total_duration_frames"],
    )
    transition_model = mean_sd(
        seed_summaries["transition_seed"][~seed_summaries["transition_seed"]["track"].eq("Ground truth")],
        ["track", "head", "capacity", "from_behavior"],
        ["n_transitions", "same_class_transitions", "same_class_transition_fraction", "median_gap_frames"],
    )
    same_model = mean_sd(
        seed_summaries["same_seed"][~seed_summaries["same_seed"]["track"].eq("Ground truth")],
        ["track", "head", "capacity", "from_behavior"],
        [
            "same_class_transitions",
            "median_same_class_gap_frames",
            "p25_same_class_gap_frames",
            "p75_same_class_gap_frames",
            "fraction_same_class_gap_le_15_frames",
        ],
    )
    overlap_model = mean_sd(
        seed_summaries["overlap_seed"],
        ["track", "head", "capacity", "class"],
        [
            "gt_bouts_with_any_overlap",
            "gt_bouts_split_by_multiple_predictions",
            "fraction_gt_bouts_split",
            "mean_gt_coverage_fraction",
        ],
    )
    gt_bout = add_ground_truth_reference(
        seed_summaries["bout_seed"],
        "class",
        ["n_bouts", "median_duration_frames", "mean_duration_frames", "total_duration_frames"],
    )
    gt_transition = add_ground_truth_reference(
        seed_summaries["transition_seed"],
        "from_behavior",
        ["n_transitions", "same_class_transitions", "same_class_transition_fraction", "median_gap_frames"],
        class_col="from_behavior",
    )
    gt_same = add_ground_truth_reference(
        seed_summaries["same_seed"],
        "from_behavior",
        [
            "same_class_transitions",
            "median_same_class_gap_frames",
            "p25_same_class_gap_frames",
            "p75_same_class_gap_frames",
            "fraction_same_class_gap_le_15_frames",
        ],
        class_col="from_behavior",
    )
    return {
        "bout_model": pd.concat([gt_bout, bout_model], ignore_index=True),
        "transition_model": pd.concat([gt_transition, transition_model], ignore_index=True),
        "same_model": pd.concat([gt_same, same_model], ignore_index=True),
        "overlap_model": overlap_model,
    }


def ordered_tracks(df: pd.DataFrame) -> list[str]:
    present = set(df["track"].dropna())
    return [t for t in TRACK_ORDER if t in present]


def plot_audit(seed_summaries: dict[str, pd.DataFrame], model_summaries: dict[str, pd.DataFrame], figure_dir: Path, output_name: str) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.6), constrained_layout=True)
    axes = axes.ravel()
    tracks = ordered_tracks(model_summaries["bout_model"])
    x = np.arange(len(tracks))

    # A. Bout counts by class.
    for behavior in BEHAVIORS:
        sub = model_summaries["bout_model"][
            model_summaries["bout_model"]["class"].eq(behavior)
        ].set_index("track").reindex(tracks)
        axes[0].plot(x, sub["n_bouts_mean"], marker="o", label=behavior.capitalize())
        axes[0].fill_between(
            x,
            sub["n_bouts_mean"] - sub["n_bouts_sd"].fillna(0),
            sub["n_bouts_mean"] + sub["n_bouts_sd"].fillna(0),
            alpha=0.12,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tracks, rotation=35, ha="right")
    axes[0].set_ylabel("Bout count")
    axes[0].set_title("A. Bout counts across capacities")
    axes[0].legend(frameon=False, fontsize=7)

    # B. Same-class transition rate heatmap averaged across behaviors.
    trans = model_summaries["transition_model"].copy()
    trans_avg = (
        trans.groupby(["track"], as_index=False)
        .agg(
            same_class_transition_fraction_mean=("same_class_transition_fraction_mean", "mean"),
            same_class_transition_fraction_sd=("same_class_transition_fraction_sd", "mean"),
        )
        .set_index("track")
        .reindex(tracks)
    )
    axes[1].bar(
        x,
        trans_avg["same_class_transition_fraction_mean"],
        yerr=trans_avg["same_class_transition_fraction_sd"].fillna(0),
        color=[TRACK_COLORS.get(t, "#9CA3AF") for t in tracks],
        alpha=0.88,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tracks, rotation=35, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Mean fraction")
    axes[1].set_title("B. Same-class consecutive bout rate")

    # C. Same-class gap distributions across all behaviors/seeds.
    same_raw = seed_summaries["same_seed"].copy()
    gap_data = []
    positions = []
    colors = []
    for idx, track in enumerate(tracks):
        vals = same_raw[same_raw["track"].eq(track)]["median_same_class_gap_frames"].to_numpy(dtype=float) / 30.0
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            vals = np.array([np.nan])
        gap_data.append(vals)
        positions.append(idx)
        colors.append(TRACK_COLORS.get(track, "#9CA3AF"))
    bp = axes[2].boxplot(gap_data, positions=positions, widths=0.55, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.45)
        patch.set_edgecolor(color)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tracks, rotation=35, ha="right")
    axes[2].set_ylabel("Median same-class gap (s)")
    axes[2].set_title("C. Same-class gap distribution")

    # D. Fraction of GT bouts split by multiple predicted bouts.
    split = model_summaries["overlap_model"].copy()
    for behavior in BEHAVIORS:
        sub = split[split["class"].eq(behavior)].set_index("track").reindex(tracks[1:])
        axes[3].plot(np.arange(len(tracks[1:])), sub["fraction_gt_bouts_split_mean"], marker="o", label=behavior.capitalize())
        axes[3].fill_between(
            np.arange(len(tracks[1:])),
            sub["fraction_gt_bouts_split_mean"] - sub["fraction_gt_bouts_split_sd"].fillna(0),
            sub["fraction_gt_bouts_split_mean"] + sub["fraction_gt_bouts_split_sd"].fillna(0),
            alpha=0.12,
        )
    axes[3].set_xticks(np.arange(len(tracks[1:])))
    axes[3].set_xticklabels(tracks[1:], rotation=35, ha="right")
    axes[3].set_ylim(0, 0.12)
    axes[3].set_ylabel("Fraction of GT bouts")
    axes[3].set_title("D. GT bouts split by multiple predictions")
    axes[3].legend(frameon=False, fontsize=7)

    # E. GT coverage by predicted bouts.
    for behavior in BEHAVIORS:
        sub = split[split["class"].eq(behavior)].set_index("track").reindex(tracks[1:])
        axes[4].plot(np.arange(len(tracks[1:])), sub["mean_gt_coverage_fraction_mean"], marker="o", label=behavior.capitalize())
    axes[4].set_xticks(np.arange(len(tracks[1:])))
    axes[4].set_xticklabels(tracks[1:], rotation=35, ha="right")
    axes[4].set_ylim(0, 1)
    axes[4].set_ylabel("Mean GT coverage")
    axes[4].set_title("E. Ground-truth bout coverage")
    axes[4].legend(frameon=False, fontsize=7)

    # F. Bout count error relative to GT.
    gt_counts = model_summaries["bout_model"][model_summaries["bout_model"]["track"].eq("Ground truth")][
        ["class", "n_bouts_mean"]
    ].rename(columns={"n_bouts_mean": "gt_n_bouts"})
    model_counts = model_summaries["bout_model"][~model_summaries["bout_model"]["track"].eq("Ground truth")].merge(
        gt_counts, on="class", validate="many_to_one"
    )
    model_counts["bout_count_delta_vs_gt"] = model_counts["n_bouts_mean"] - model_counts["gt_n_bouts"]
    for behavior in BEHAVIORS:
        sub = model_counts[model_counts["class"].eq(behavior)].set_index("track").reindex(tracks[1:])
        axes[5].plot(np.arange(len(tracks[1:])), sub["bout_count_delta_vs_gt"], marker="o", label=behavior.capitalize())
    axes[5].axhline(0, color="#6B7280", linewidth=1)
    axes[5].set_xticks(np.arange(len(tracks[1:])))
    axes[5].set_xticklabels(tracks[1:], rotation=35, ha="right")
    axes[5].set_ylabel("Predicted − GT bout count")
    axes[5].set_title("F. Bout count deviation")
    axes[5].legend(frameon=False, fontsize=7)

    fig.suptitle("Bout fragmentation audit across YOLO/SPPF full-stream capacities", fontsize=13)
    save_figure(fig, figure_dir / output_name)
    plt.close(fig)


def write_outputs(args: argparse.Namespace, seed_summaries: dict[str, pd.DataFrame], model_summaries: dict[str, pd.DataFrame]) -> None:
    for name, df in seed_summaries.items():
        df.to_csv(args.table_dir / f"{args.output_name}_{name}.csv", index=False)
    for name, df in model_summaries.items():
        df.to_csv(args.table_dir / f"{args.output_name}_{name}.csv", index=False)


def main() -> None:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    per_video, gt_table, pred_table = load_inputs(args.table_dir, args.prediction_type)
    videos = video_units(per_video, args.reference_model)
    model_specs = discover_full_models(per_video)

    all_bout_rows = []
    transition_rows = []
    overlap_rows_all = []

    gt_meta = {"track": "Ground truth", "head": "ground_truth", "capacity": 0, "seed": 0, "model_name": "ground_truth"}
    for _, video in videos.iterrows():
        split = str(video["split"])
        video_id = str(video["video_id"])
        gt_bouts = track_bouts(gt_table, split, video_id, None)
        gt_copy = gt_bouts.copy()
        for key, value in gt_meta.items():
            gt_copy[key] = value
        gt_copy["split"] = split
        gt_copy["video_id"] = video_id
        all_bout_rows.append(gt_copy)
        for row in consecutive_transition_rows(gt_meta, gt_bouts, args.fps):
            row.update({"split": split, "video_id": video_id})
            transition_rows.append(row)

        for spec in model_specs:
            pred_bouts = track_bouts(pred_table, split, video_id, spec["model_name"])
            pred_copy = pred_bouts.copy()
            for key, value in spec.items():
                pred_copy[key] = value
            pred_copy["split"] = split
            pred_copy["video_id"] = video_id
            all_bout_rows.append(pred_copy)
            for row in consecutive_transition_rows(spec, pred_bouts, args.fps):
                row.update({"split": split, "video_id": video_id})
                transition_rows.append(row)
            for row in gt_overlap_rows(gt_bouts, pred_bouts, spec):
                row.update({"split": split, "video_id": video_id})
                overlap_rows_all.append(row)

    all_bouts = pd.concat(all_bout_rows, ignore_index=True)
    transitions = pd.DataFrame(transition_rows)
    overlaps = pd.DataFrame(overlap_rows_all)
    all_bouts.to_csv(args.table_dir / f"{args.output_name}_all_bouts_long.csv", index=False)
    transitions.to_csv(args.table_dir / f"{args.output_name}_consecutive_bout_transitions_long.csv", index=False)
    overlaps.to_csv(args.table_dir / f"{args.output_name}_gt_overlap_detail_long.csv", index=False)

    seed_summaries = aggregate_seed_summaries(all_bouts, transitions, overlaps)
    model_summaries = summarize_all(seed_summaries)
    write_outputs(args, seed_summaries, model_summaries)
    plot_audit(seed_summaries, model_summaries, args.figure_dir, args.output_name)

    best = model_summaries["transition_model"]
    best = best[~best["track"].eq("Ground truth")].copy()
    best_rank = (
        best.groupby("track", as_index=False)["same_class_transition_fraction_mean"]
        .mean()
        .sort_values("same_class_transition_fraction_mean")
    )
    print("Mean same-class consecutive bout rate by model:")
    print(best_rank.to_string(index=False))
    print(f"\nWrote {args.figure_dir / (args.output_name + '.png')}")


if __name__ == "__main__":
    main()
