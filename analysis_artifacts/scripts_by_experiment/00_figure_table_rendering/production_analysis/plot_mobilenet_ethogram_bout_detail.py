"""Analyze MobileNetV3 bout and ethogram behavior relative to YOLO/SPPF."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
STYLE_DIR = THIS_DIR.parent / "current_analysis"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import (  # noqa: E402
    BACKBONE_COLORS,
    BEHAVIOR_FILL_COLORS,
    HEAD_COLORS,
    NEUTRAL,
    apply_style,
    panel_label,
    save_figure,
)


BEHAVIORS = ["attack", "investigation", "mount"]
ALL_CLASSES = ["attack", "investigation", "mount", "other"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(ALL_CLASSES)}
BEHAVIOR_TO_ID = {name: idx for idx, name in enumerate(BEHAVIORS)}
BACKBONE_LABELS = {"yolo_sppf": "YOLO/SPPF", "mobilenetv3_native": "MobileNetV3"}
HEAD_LABELS = {"attention": "Attention-256", "lstm": "LSTM-256"}
TRACK_ORDER = [
    "YOLO/SPPF Attention-256",
    "MobileNetV3 Attention-256",
    "YOLO/SPPF LSTM-256",
    "MobileNetV3 LSTM-256",
]
TRACK_COLORS = {
    "YOLO/SPPF Attention-256": BACKBONE_COLORS["yolo_sppf"],
    "YOLO/SPPF LSTM-256": "#7DB8D6",
    "MobileNetV3 Attention-256": BACKBONE_COLORS["mobilenetv3_native"],
    "MobileNetV3 LSTM-256": "#E8B07D",
}
MINUS = "\N{MINUS SIGN}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite_root",
        type=Path,
        default=Path("outputs/controlled_comparison_runs/run_20260528_205727"),
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=Path("Manuscript_penultimate/figures/production/mobilenet_portability"),
    )
    parser.add_argument(
        "--table_dir",
        type=Path,
        default=Path("Manuscript_penultimate/tables/production/mobilenet_portability"),
    )
    parser.add_argument("--prediction_type", default="smoothed_frames")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video_id", default="Mouse062_20160526_18-56-26")
    parser.add_argument("--split", default="test_1")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--output_prefix", default="fig_mobilenet_ethogram_bout_detail")
    return parser.parse_args()


def model_name(backbone: str, head: str, seed: int) -> str:
    if backbone == "yolo_sppf":
        return f"yolo_sppf_full_{head}256_seed{seed}"
    return f"mobilenetv3_native_full_{head}256_seed{seed}"


def track_label(backbone: str, head: str) -> str:
    return f"{BACKBONE_LABELS[backbone]} {HEAD_LABELS[head]}"


def model_specs() -> list[dict]:
    rows = []
    for backbone in ["yolo_sppf", "mobilenetv3_native"]:
        for head in ["attention", "lstm"]:
            for seed in [42, 43, 44]:
                rows.append(
                    {
                        "backbone": backbone,
                        "head": head,
                        "seed": seed,
                        "model_name": model_name(backbone, head, seed),
                        "track": track_label(backbone, head),
                    }
                )
    return rows


def read_eval_table(eval_root: Path, filename: str, spec: dict, prediction_type: str) -> pd.DataFrame:
    path = eval_root / filename
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    df = df[df["prediction_type"].eq(prediction_type)].copy()
    df["backbone"] = spec["backbone"]
    df["head"] = spec["head"]
    df["seed"] = spec["seed"]
    df["model_name"] = spec["model_name"]
    df["track"] = spec["track"]
    return df


def load_eval_outputs(suite_root: Path, prediction_type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_video_rows = []
    pred_rows = []
    gt_rows = []
    match_rows = []
    for spec in model_specs():
        eval_root = suite_root / "neural_eval" / spec["model_name"]
        per_video_rows.append(read_eval_table(eval_root, "per_video_summary.csv", spec, prediction_type))
        pred_rows.append(read_eval_table(eval_root, "predicted_bouts.csv", spec, prediction_type))
        gt_rows.append(read_eval_table(eval_root, "ground_truth_bouts.csv", spec, prediction_type))
        match_rows.append(read_eval_table(eval_root, "bout_matches.csv", spec, prediction_type))
    per_video = pd.concat(per_video_rows, ignore_index=True)
    pred = pd.concat(pred_rows, ignore_index=True)
    gt = pd.concat(gt_rows, ignore_index=True)
    matches = pd.concat(match_rows, ignore_index=True)
    for df in [per_video, pred, gt, matches]:
        for col in ["seed", "n_frames", "gt_bouts_total", "pred_bouts_total"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    for df in [pred, gt]:
        for col in ["start_frame", "end_frame", "duration_frames"]:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)
    for col in ["threshold", "tiou", "pred_start", "pred_end", "gt_start", "gt_end", "start_offset_frames", "end_offset_frames"]:
        if col in matches.columns:
            matches[col] = pd.to_numeric(matches[col], errors="coerce")
    return per_video, gt, pred, matches


def unique_gt(gt: pd.DataFrame) -> pd.DataFrame:
    return gt.drop_duplicates(["split", "video_id", "prediction_type", "class", "start_frame", "end_frame"]).copy()


def summarize_bouts(gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    gt_u = unique_gt(gt)
    gt_counts = (
        gt_u[gt_u["class"].isin(BEHAVIORS)]
        .groupby("class", as_index=False)
        .agg(gt_bouts=("duration_frames", "size"), gt_median_duration_frames=("duration_frames", "median"))
    )
    seed_summary = (
        pred[pred["class"].isin(BEHAVIORS)]
        .groupby(["backbone", "head", "track", "seed", "class"], as_index=False)
        .agg(
            pred_bouts=("duration_frames", "size"),
            pred_median_duration_frames=("duration_frames", "median"),
            pred_mean_duration_frames=("duration_frames", "mean"),
        )
        .merge(gt_counts, on="class", validate="many_to_one")
    )
    seed_summary["bout_count_ratio_pred_gt"] = seed_summary["pred_bouts"] / seed_summary["gt_bouts"]
    seed_summary["median_duration_ratio_pred_gt"] = (
        seed_summary["pred_median_duration_frames"] / seed_summary["gt_median_duration_frames"]
    )
    return (
        seed_summary.groupby(["backbone", "head", "track", "class"], as_index=False)
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
            gt_bouts=("gt_bouts", "first"),
            gt_median_duration_frames=("gt_median_duration_frames", "first"),
        )
    )


def merge_detail(gt: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    gt_u = unique_gt(gt)
    gt_by_key = {
        key: sub[["start_frame", "end_frame"]].to_numpy(dtype=np.int64)
        for key, sub in gt_u[gt_u["class"].isin(BEHAVIORS)].groupby(["split", "video_id", "class"], sort=False)
    }
    rows = []
    for key, sub in pred[pred["class"].isin(BEHAVIORS)].groupby(
        ["backbone", "head", "track", "seed", "model_name", "split", "video_id", "class"],
        sort=False,
    ):
        backbone, head, track, seed, model, split, video_id, cls = key
        gt_intervals = gt_by_key.get((split, video_id, cls))
        starts = sub["start_frame"].to_numpy(dtype=np.int64)
        ends = sub["end_frame"].to_numpy(dtype=np.int64)
        if gt_intervals is None or len(gt_intervals) == 0:
            n_gt = np.zeros(len(sub), dtype=np.int64)
            overlap_frames = np.zeros(len(sub), dtype=np.int64)
        else:
            pred_starts = starts[:, None]
            pred_ends = ends[:, None]
            gt_starts = gt_intervals[:, 0][None, :]
            gt_ends = gt_intervals[:, 1][None, :]
            overlap = np.maximum(0, np.minimum(pred_ends, gt_ends) - np.maximum(pred_starts, gt_starts) + 1)
            n_gt = (overlap > 0).sum(axis=1)
            overlap_frames = overlap.sum(axis=1)
        detail = sub.copy()
        detail["overlapping_gt_bouts"] = n_gt
        detail["overlap_gt_frames"] = overlap_frames
        detail["merges_multiple_gt_bouts"] = n_gt > 1
        rows.append(detail)
    return pd.concat(rows, ignore_index=True)


def summarize_merge(detail: pd.DataFrame) -> pd.DataFrame:
    seed = (
        detail.groupby(["backbone", "head", "track", "seed", "class"], as_index=False)
        .agg(
            fraction_pred_bouts_with_any_gt=("overlapping_gt_bouts", lambda x: float(np.mean(np.asarray(x) > 0))),
            fraction_pred_bouts_merging_gt=("merges_multiple_gt_bouts", "mean"),
            mean_overlapping_gt_bouts=("overlapping_gt_bouts", "mean"),
        )
    )
    return (
        seed.groupby(["backbone", "head", "track", "class"], as_index=False)
        .agg(
            fraction_pred_bouts_with_any_gt_mean=("fraction_pred_bouts_with_any_gt", "mean"),
            fraction_pred_bouts_with_any_gt_sd=("fraction_pred_bouts_with_any_gt", "std"),
            fraction_pred_bouts_merging_gt_mean=("fraction_pred_bouts_merging_gt", "mean"),
            fraction_pred_bouts_merging_gt_sd=("fraction_pred_bouts_merging_gt", "std"),
            mean_overlapping_gt_bouts_mean=("mean_overlapping_gt_bouts", "mean"),
            mean_overlapping_gt_bouts_sd=("mean_overlapping_gt_bouts", "std"),
        )
    )


def summarize_boundary(matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = matches[matches["class"].isin(BEHAVIORS)].copy()
    df["pred_duration_frames"] = df["pred_end"] - df["pred_start"] + 1
    df["gt_duration_frames"] = df["gt_end"] - df["gt_start"] + 1
    df["abs_start_offset_frames"] = df["start_offset_frames"].abs()
    df["abs_end_offset_frames"] = df["end_offset_frames"].abs()
    df["abs_mean_boundary_offset_frames"] = (df["abs_start_offset_frames"] + df["abs_end_offset_frames"]) / 2.0
    seed = (
        df.groupby(["backbone", "head", "track", "seed", "class", "threshold"], as_index=False)
        .agg(
            matched_bouts=("tiou", "size"),
            median_tiou=("tiou", "median"),
            median_abs_mean_boundary_offset_frames=("abs_mean_boundary_offset_frames", "median"),
            median_abs_start_offset_frames=("abs_start_offset_frames", "median"),
            median_abs_end_offset_frames=("abs_end_offset_frames", "median"),
        )
    )
    model = (
        seed.groupby(["backbone", "head", "track", "class", "threshold"], as_index=False)
        .agg(
            matched_bouts_mean=("matched_bouts", "mean"),
            matched_bouts_sd=("matched_bouts", "std"),
            median_tiou_mean=("median_tiou", "mean"),
            median_tiou_sd=("median_tiou", "std"),
            median_abs_mean_boundary_offset_frames_mean=("median_abs_mean_boundary_offset_frames", "mean"),
            median_abs_mean_boundary_offset_frames_sd=("median_abs_mean_boundary_offset_frames", "std"),
            median_abs_start_offset_frames_mean=("median_abs_start_offset_frames", "mean"),
            median_abs_end_offset_frames_mean=("median_abs_end_offset_frames", "mean"),
        )
    )
    return seed, model


def labels_from_bouts(bouts: pd.DataFrame, n_frames: int) -> np.ndarray:
    labels = np.full(n_frames, CLASS_TO_ID["other"], dtype=np.int8)
    for cls in BEHAVIORS:
        sub = bouts[bouts["class"].eq(cls)]
        for row in sub.itertuples(index=False):
            start = max(0, int(row.start_frame))
            end = min(n_frames - 1, int(row.end_frame))
            if end >= start:
                labels[start : end + 1] = CLASS_TO_ID[cls]
    return labels


def segments_from_labels(labels: np.ndarray) -> list[tuple[int, int, str]]:
    out = []
    start = 0
    current = int(labels[0])
    for idx in range(1, len(labels)):
        value = int(labels[idx])
        if value != current:
            out.append((start, idx - start, ALL_CLASSES[current]))
            start = idx
            current = value
    out.append((start, len(labels) - start, ALL_CLASSES[current]))
    return out


def draw_ethogram(ax, gt: pd.DataFrame, pred: pd.DataFrame, per_video: pd.DataFrame, split: str, video_id: str, seed: int, fps: float) -> None:
    ref = per_video[(per_video["split"].eq(split)) & (per_video["video_id"].eq(video_id))]
    if ref.empty:
        raise ValueError(f"Missing representative video {split}/{video_id}")
    n_frames = int(ref.iloc[0]["n_frames"])
    tracks = [("Ground truth", None, "gt")]
    for backbone, head in [
        ("yolo_sppf", "attention"),
        ("mobilenetv3_native", "attention"),
        ("yolo_sppf", "lstm"),
        ("mobilenetv3_native", "lstm"),
    ]:
        tracks.append((track_label(backbone, head), model_name(backbone, head, seed), "pred"))

    y_labels = []
    for y, (label, model, source) in enumerate(tracks[::-1]):
        if source == "gt":
            bouts = unique_gt(gt)
            bouts = bouts[(bouts["split"].eq(split)) & (bouts["video_id"].eq(video_id))]
        else:
            bouts = pred[
                pred["model_name"].eq(model)
                & pred["split"].eq(split)
                & pred["video_id"].eq(video_id)
                & pred["seed"].eq(seed)
            ]
        labels = labels_from_bouts(bouts[bouts["class"].isin(BEHAVIORS)], n_frames)
        for start, length, cls in segments_from_labels(labels):
            ax.broken_barh(
                [(start / fps, max(length / fps, 1.0 / fps))],
                (y - 0.34, 0.68),
                facecolors=BEHAVIOR_FILL_COLORS[cls],
                edgecolors="none",
            )
        ax.hlines(y - 0.34, 0, n_frames / fps, color="black", linewidth=0.45)
        y_labels.append(label)
    ax.set_yticks(np.arange(len(tracks)))
    ax.set_yticklabels(y_labels)
    ax.set_xlim(0, n_frames / fps)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Representative full-video ethogram: {video_id} (seed {seed})", loc="left")
    handles = [Patch(facecolor=BEHAVIOR_FILL_COLORS[c], edgecolor="none", label=c.capitalize()) for c in ALL_CLASSES]
    ax.legend(handles=handles, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.28))


def sequence_for(table: pd.DataFrame, split: str, video_id: str, model: str | None) -> list[str]:
    sub = table[(table["split"].eq(split)) & (table["video_id"].eq(video_id))].copy()
    if model is not None:
        sub = sub[sub["model_name"].eq(model)]
    else:
        sub = sub.drop_duplicates(["split", "video_id", "class", "start_frame", "end_frame"])
    sub = sub[sub["class"].isin(BEHAVIORS)].sort_values(["start_frame", "end_frame", "class"])
    return [str(x) for x in sub["class"]]


def ngrams(seq: list[str], n: int) -> Counter:
    return Counter(tuple(seq[i : i + n]) for i in range(max(0, len(seq) - n + 1)))


def multiset_f1(gt_seq: list[str], pred_seq: list[str], n: int) -> float:
    gt_c = ngrams(gt_seq, n)
    pred_c = ngrams(pred_seq, n)
    tp = sum(min(pred_c[k], gt_c.get(k, 0)) for k in pred_c)
    pred_total = sum(pred_c.values())
    gt_total = sum(gt_c.values())
    p = tp / pred_total if pred_total else 0.0
    r = tp / gt_total if gt_total else 0.0
    return 2 * p * r / (p + r) if p + r > 0 else 0.0


def ngram_summary(gt: pd.DataFrame, pred: pd.DataFrame, per_video: pd.DataFrame) -> pd.DataFrame:
    videos = per_video[["split", "video_id"]].drop_duplicates()
    rows = []
    for spec in model_specs():
        for n in [1, 2, 3, 4]:
            vals = []
            ratios = []
            for video in videos.itertuples(index=False):
                gt_seq = sequence_for(unique_gt(gt), video.split, video.video_id, None)
                pred_seq = sequence_for(pred[pred["model_name"].eq(spec["model_name"])], video.split, video.video_id, spec["model_name"])
                vals.append(multiset_f1(gt_seq, pred_seq, n))
                ratios.append((len(pred_seq) / len(gt_seq)) if gt_seq else np.nan)
            rows.append(
                {
                    **spec,
                    "n": n,
                    "ngram_f1_mean": float(np.nanmean(vals)),
                    "ngram_f1_sd": float(np.nanstd(vals, ddof=1)),
                    "sequence_length_ratio_pred_gt_mean": float(np.nanmean(ratios)),
                }
            )
    return pd.DataFrame(rows)


def transition_counts(table: pd.DataFrame, split: str, video_id: str, model: str | None) -> np.ndarray:
    seq = sequence_for(table, split, video_id, model)
    counts = np.zeros((len(BEHAVIORS), len(BEHAVIORS)), dtype=np.int64)
    for src, dst in zip(seq[:-1], seq[1:]):
        counts[BEHAVIOR_TO_ID[src], BEHAVIOR_TO_ID[dst]] += 1
    return counts


def transition_summary(gt: pd.DataFrame, pred: pd.DataFrame, per_video: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_u = unique_gt(gt)
    videos = per_video[["split", "video_id"]].drop_duplicates()
    rows = []
    gt_counts = np.zeros((len(BEHAVIORS), len(BEHAVIORS)), dtype=np.int64)
    for video in videos.itertuples(index=False):
        gt_counts += transition_counts(gt_u, video.split, video.video_id, None)
    rows.extend(transition_rows("Ground truth", "ground_truth", "", "", gt_counts))
    for spec in model_specs():
        counts = np.zeros((len(BEHAVIORS), len(BEHAVIORS)), dtype=np.int64)
        model_pred = pred[pred["model_name"].eq(spec["model_name"])]
        for video in videos.itertuples(index=False):
            counts += transition_counts(model_pred, video.split, video.video_id, spec["model_name"])
        rows.extend(transition_rows(spec["track"], spec["backbone"], spec["head"], spec["seed"], counts))
    counts_df = pd.DataFrame(rows)
    probs = counts_df.copy()
    probs["transition_probability"] = probs.groupby(["track", "backbone", "head", "seed", "from_behavior"])["transition_count"].transform(
        lambda x: x / x.sum() if x.sum() else 0.0
    )
    return counts_df, probs


def transition_rows(track: str, backbone: str, head: str, seed: str | int, counts: np.ndarray) -> list[dict]:
    rows = []
    for i, src in enumerate(BEHAVIORS):
        for j, dst in enumerate(BEHAVIORS):
            rows.append(
                {
                    "track": track,
                    "backbone": backbone,
                    "head": head,
                    "seed": seed,
                    "from_behavior": src,
                    "to_behavior": dst,
                    "transition_count": int(counts[i, j]),
                }
            )
    return rows


def draw_bout_bars(ax, summary: pd.DataFrame, value_col: str, err_col: str, title: str, ylabel: str, ref: float | None = None) -> None:
    width = 0.18
    x = np.arange(len(BEHAVIORS))
    offsets = {
        "YOLO/SPPF Attention-256": -0.27,
        "MobileNetV3 Attention-256": -0.09,
        "YOLO/SPPF LSTM-256": 0.09,
        "MobileNetV3 LSTM-256": 0.27,
    }
    for track in TRACK_ORDER:
        sub = summary[summary["track"].eq(track)].set_index("class").reindex(BEHAVIORS)
        ax.bar(
            x + offsets[track],
            sub[value_col],
            width=width,
            yerr=sub[err_col],
            color=TRACK_COLORS[track],
            alpha=0.86,
            edgecolor=NEUTRAL["dark"],
            linewidth=0.35,
            capsize=2,
            label=track,
        )
    if ref is not None:
        ax.axhline(ref, color=NEUTRAL["mid"], lw=1, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in BEHAVIORS])
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)


def draw_boundary(ax, boundary: pd.DataFrame) -> None:
    sub = boundary[boundary["threshold"].eq(0.25)].copy()
    draw_bout_bars(
        ax,
        sub,
        "median_abs_mean_boundary_offset_frames_mean",
        "median_abs_mean_boundary_offset_frames_sd",
        "Boundary offset for matched bouts (TIoU 0.25)",
        "Boundary offset (frames)",
    )
    ax.set_ylabel("Boundary offset (frames)", labelpad=10)


def draw_merge_fraction(ax, summary: pd.DataFrame) -> None:
    draw_bout_bars(
        ax,
        summary,
        "fraction_pred_bouts_merging_gt_mean",
        "fraction_pred_bouts_merging_gt_sd",
        "Predicted bouts spanning multiple GT bouts",
        "Fraction of predicted bouts",
        ref=None,
    )
    ax.set_ylim(0, 0.36)


def draw_ngram(ax, ngram: pd.DataFrame) -> None:
    for track in TRACK_ORDER:
        sub = ngram[ngram["track"].eq(track)].groupby("n", as_index=False).agg(
            mean=("ngram_f1_mean", "mean"),
            sd=("ngram_f1_mean", "std"),
        )
        ax.errorbar(
            sub["n"],
            sub["mean"],
            yerr=sub["sd"],
            marker="o" if "Attention" in track else "s",
            color=TRACK_COLORS[track],
            lw=1.8,
            capsize=2,
            label=track,
        )
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(0.45, 0.86)
    ax.set_xlabel("Bout n-gram order")
    ax.set_ylabel("N-gram F1")
    ax.set_title("Bout-sequence grammar recovery", loc="left")


def main() -> None:
    args = parse_args()
    apply_style()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    per_video, gt, pred, matches = load_eval_outputs(args.suite_root, args.prediction_type)

    bout_summary = summarize_bouts(gt, pred)
    detail = merge_detail(gt, pred)
    merge_summary = summarize_merge(detail)
    boundary_seed, boundary_model = summarize_boundary(matches)

    bout_detail_summary = bout_summary.merge(merge_summary, on=["backbone", "head", "track", "class"], validate="one_to_one")
    per_video.to_csv(args.table_dir / "mobilenet_cross_backbone_ethogram_per_video.csv", index=False)
    unique_gt(gt).to_csv(args.table_dir / "mobilenet_cross_backbone_ground_truth_bouts.csv", index=False)
    pred.to_csv(args.table_dir / "mobilenet_cross_backbone_predicted_bouts.csv", index=False)
    matches.to_csv(args.table_dir / "mobilenet_cross_backbone_bout_matches.csv", index=False)
    detail.to_csv(args.table_dir / "mobilenet_cross_backbone_predicted_bout_overlap_detail.csv", index=False)
    bout_detail_summary.to_csv(args.table_dir / "mobilenet_cross_backbone_bout_detail_summary.csv", index=False)
    boundary_seed.to_csv(args.table_dir / "mobilenet_cross_backbone_boundary_seed_summary.csv", index=False)
    boundary_model.to_csv(args.table_dir / "mobilenet_cross_backbone_boundary_model_summary.csv", index=False)

    fig = plt.figure(figsize=(14.5, 10.9), constrained_layout=False)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 0.88, 0.92], hspace=0.82, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[2, 0])
    ax_e = fig.add_subplot(gs[2, 1])
    draw_ethogram(ax_a, gt, pred, per_video, args.split, args.video_id, args.seed, args.fps)
    draw_bout_bars(
        ax_b,
        bout_detail_summary,
        "bout_count_ratio_pred_gt_mean",
        "bout_count_ratio_pred_gt_sd",
        "Bout-count recovery by class",
        "Predicted / GT bout count",
        ref=1.0,
    )
    draw_bout_bars(
        ax_c,
        bout_detail_summary,
        "median_duration_ratio_pred_gt_mean",
        "median_duration_ratio_pred_gt_sd",
        "Bout-duration inflation by class",
        "Predicted / GT median duration",
        ref=1.0,
    )
    draw_merge_fraction(ax_d, bout_detail_summary)
    draw_boundary(ax_e, boundary_model)
    handles = [Line2D([0], [0], color=TRACK_COLORS[t], lw=7, label=t) for t in TRACK_ORDER]
    fig.legend(handles=handles, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.025))
    panel_positions = {
        "A": (-0.06, 1.13),
        "B": (-0.10, 1.13),
        "C": (-0.10, 1.13),
        "D": (-0.08, 1.20),
        "E": (-0.08, 1.20),
    }
    for label, ax in zip(["A", "B", "C", "D", "E"], [ax_a, ax_b, ax_c, ax_d, ax_e]):
        x_pos, y_pos = panel_positions[label]
        panel_label(ax, label, x=x_pos, y=y_pos)
    fig.suptitle("Cross-backbone ethogram and bout-level behavior recovery", fontsize=14, y=0.99)
    fig.subplots_adjust(left=0.08, right=0.985, top=0.93, bottom=0.13)
    save_figure(fig, args.figure_dir / args.output_prefix)
    plt.close(fig)


if __name__ == "__main__":
    main()
