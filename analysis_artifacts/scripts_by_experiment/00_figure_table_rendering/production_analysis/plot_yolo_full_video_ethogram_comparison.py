"""Plot full-video ethogram and transition comparisons for YOLO/SPPF heads."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
STYLE_DIR = THIS_DIR.parent / "current_analysis"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import (  # noqa: E402
    BEHAVIOR_FILL_COLORS,
    apply_style,
    panel_label,
    save_figure,
)


BEHAVIOR_CLASSES = ["attack", "investigation", "mount"]
ALL_CLASSES = ["attack", "investigation", "mount", "other"]
CLASS_LABELS = {
    "attack": "Attack",
    "investigation": "Investigation",
    "mount": "Mount",
    "other": "Other",
}
CLASS_TO_ID = {name: i for i, name in enumerate(ALL_CLASSES)}
BOUT_TRANSITION_TO_ID = {name: i for i, name in enumerate(BEHAVIOR_CLASSES)}
MINUS = "\N{MINUS SIGN}"
DIFF_CMAP = LinearSegmentedColormap.from_list(
    "orange_white_green",
    ["#C2410C", "#F8FAFC", "#047857"],
)


def true_minus(text: str) -> str:
    """Render mathematical negatives with a true minus sign."""
    return text.replace("-", MINUS)


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
    parser.add_argument("--video_id", default="Mouse062_20160526_18-56-26")
    parser.add_argument("--split", default="test_1")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--output_name", default="fig8_yolo_full_video_ethogram_lstm_attention")
    return parser.parse_args()


def model_name(stream: str, head: str, capacity: int, seed: int) -> str:
    return f"yolo_sppf_{stream}_{head}{capacity}_seed{seed}"


def load_tables(table_dir: Path, prediction_type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_video = pd.read_csv(table_dir / "yolo_lstm_attention_per_video_long.csv")
    per_video = per_video[per_video["prediction_type"].eq(prediction_type)].copy()
    pred_cols = [
        "split",
        "video_id",
        "prediction_type",
        "class",
        "start_frame",
        "end_frame",
        "duration_frames",
        "model_name",
    ]
    pred = pd.read_csv(table_dir / "yolo_lstm_attention_predicted_bouts_long.csv.gz", usecols=pred_cols)
    pred = pred[pred["prediction_type"].eq(prediction_type)].copy()
    gt = pd.read_csv(table_dir / "yolo_lstm_attention_ground_truth_bouts_long.csv.gz", usecols=pred_cols)
    gt = gt[gt["prediction_type"].eq(prediction_type)].copy()
    return per_video, gt, pred


def get_n_frames(per_video: pd.DataFrame, split: str, video_id: str, model_name_value: str) -> int:
    row = per_video[
        per_video["split"].eq(split)
        & per_video["video_id"].eq(video_id)
        & per_video["model_name"].eq(model_name_value)
    ]
    if row.empty:
        raise ValueError(f"No per-video row for {split}/{video_id}/{model_name_value}")
    return int(pd.to_numeric(row.iloc[0]["n_frames"]))


def bouts_for_track(
    bouts: pd.DataFrame,
    split: str,
    video_id: str,
    model_name_value: str | None,
) -> pd.DataFrame:
    sub = bouts[bouts["split"].eq(split) & bouts["video_id"].eq(video_id)].copy()
    if model_name_value is not None:
        sub = sub[sub["model_name"].eq(model_name_value)].copy()
    else:
        sub = sub.drop_duplicates(["split", "video_id", "class", "start_frame", "end_frame"]).copy()
    sub = sub[sub["class"].isin(BEHAVIOR_CLASSES)].copy()
    sub["start_frame"] = pd.to_numeric(sub["start_frame"], errors="coerce").astype(int)
    sub["end_frame"] = pd.to_numeric(sub["end_frame"], errors="coerce").astype(int)
    return sub.sort_values(["start_frame", "end_frame", "class"])


def compress_bouts_to_segments(bouts: pd.DataFrame, n_frames: int) -> list[tuple[int, int, str]]:
    labels = labels_from_bouts(bouts, n_frames)
    segments: list[tuple[int, int, str]] = []
    start = 0
    current = int(labels[0])
    for idx in range(1, n_frames):
        value = int(labels[idx])
        if value != current:
            segments.append((start, idx - start, ALL_CLASSES[current]))
            start = idx
            current = value
    segments.append((start, n_frames - start, ALL_CLASSES[current]))
    return segments


def labels_from_bouts(bouts: pd.DataFrame, n_frames: int) -> np.ndarray:
    labels = np.full(n_frames, CLASS_TO_ID["other"], dtype=np.int8)
    # Later assignments intentionally overwrite earlier ones if rare class overlaps
    # occur, yielding a single-color ethogram row for manuscript readability.
    for cls in BEHAVIOR_CLASSES:
        sub = bouts[bouts["class"].eq(cls)]
        for _, row in sub.iterrows():
            start = max(0, int(row["start_frame"]))
            end = min(n_frames - 1, int(row["end_frame"]))
            if end >= start:
                labels[start : end + 1] = CLASS_TO_ID[cls]
    return labels


def draw_track(ax, segments: list[tuple[int, int, str]], y: float, fps: float) -> None:
    for start, length, cls in segments:
        ax.broken_barh(
            [(start / fps, max(length / fps, 1.0 / fps))],
            (y - 0.34, 0.68),
            facecolors=BEHAVIOR_FILL_COLORS[cls],
            edgecolors="none",
        )
    ax.hlines(y - 0.34, 0, segments[-1][0] / fps + segments[-1][1] / fps, color="black", linewidth=0.6)


def bout_transition_counts(bouts: pd.DataFrame) -> np.ndarray:
    """Count transitions from each behavior bout to the next behavior bout.

    This intentionally ignores the intervening `other` interval. The unit is a
    discrete annotated/predicted bout, not a framewise state change.
    """
    counts = np.zeros((len(BEHAVIOR_CLASSES), len(BEHAVIOR_CLASSES)), dtype=np.int64)
    if len(bouts) < 2:
        return counts
    ordered = bouts.sort_values(["start_frame", "end_frame", "class"]).reset_index(drop=True)
    sequence = [str(cls) for cls in ordered["class"].tolist() if str(cls) in BOUT_TRANSITION_TO_ID]
    for src, dst in zip(sequence[:-1], sequence[1:]):
        counts[BOUT_TRANSITION_TO_ID[src], BOUT_TRANSITION_TO_ID[dst]] += 1
    return counts


def collect_transition_tables(
    per_video: pd.DataFrame,
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    models: list[tuple[str, str | None, pd.DataFrame]],
    reference_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    video_rows = per_video[per_video["model_name"].eq(reference_model)].copy()
    video_rows["n_frames"] = pd.to_numeric(video_rows["n_frames"], errors="coerce").astype(int)
    counts_by_track: dict[str, np.ndarray] = {
        label: np.zeros((len(BEHAVIOR_CLASSES), len(BEHAVIOR_CLASSES)), dtype=np.int64)
        for label, _model_value, _table in models
    }
    for _, video in video_rows.iterrows():
        split = str(video["split"])
        video_id = str(video["video_id"])
        for label, model_value, table in models:
            bouts = bouts_for_track(table, split, video_id, model_value)
            counts_by_track[label] += bout_transition_counts(bouts)

    count_rows = []
    prob_rows = []
    for label, counts in counts_by_track.items():
        row_totals = counts.sum(axis=1, keepdims=True)
        probs = np.divide(counts, row_totals, out=np.zeros_like(counts, dtype=float), where=row_totals > 0)
        for src_idx, src in enumerate(BEHAVIOR_CLASSES):
            for dst_idx, dst in enumerate(BEHAVIOR_CLASSES):
                count_rows.append(
                    {
                        "track": label,
                        "from_behavior": src,
                        "to_behavior": dst,
                        "transition_count": int(counts[src_idx, dst_idx]),
                    }
                )
                prob_rows.append(
                    {
                        "track": label,
                        "from_behavior": src,
                        "to_behavior": dst,
                        "transition_probability": float(probs[src_idx, dst_idx]),
                    }
                )
    return pd.DataFrame(count_rows), pd.DataFrame(prob_rows)


def probability_matrix(probs: pd.DataFrame, track: str) -> np.ndarray:
    mat = np.zeros((len(BEHAVIOR_CLASSES), len(BEHAVIOR_CLASSES)), dtype=float)
    sub = probs[probs["track"].eq(track)]
    for _, row in sub.iterrows():
        src = BOUT_TRANSITION_TO_ID[str(row["from_behavior"])]
        dst = BOUT_TRANSITION_TO_ID[str(row["to_behavior"])]
        mat[src, dst] = float(row["transition_probability"])
    return mat


def draw_transition_heatmap(ax, mat: np.ndarray, title: str, vmax: float, *, show_ylabel: bool = True) -> None:
    image = ax.imshow(mat, cmap="Greens", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(BEHAVIOR_CLASSES)))
    ax.set_yticks(np.arange(len(BEHAVIOR_CLASSES)))
    ax.set_xticklabels([CLASS_LABELS[c] for c in BEHAVIOR_CLASSES], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([CLASS_LABELS[c] for c in BEHAVIOR_CLASSES], fontsize=7)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("Next bout")
    ax.set_ylabel("Current bout" if show_ylabel else "")
    ax.grid(False)
    for i in range(len(BEHAVIOR_CLASSES)):
        for j in range(len(BEHAVIOR_CLASSES)):
            value = mat[i, j]
            text = "" if value == 0 else f"{value * 100:.0f}%"
            color = "white" if value > vmax * 0.55 else "#374151"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
    return image


def draw_transition_difference_heatmap(
    ax,
    diff: np.ndarray,
    title: str,
    vmax: float,
    *,
    show_ylabel: bool = True,
) -> None:
    image = ax.imshow(diff, cmap=DIFF_CMAP, vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(BEHAVIOR_CLASSES)))
    ax.set_yticks(np.arange(len(BEHAVIOR_CLASSES)))
    ax.set_xticklabels([CLASS_LABELS[c] for c in BEHAVIOR_CLASSES], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([CLASS_LABELS[c] for c in BEHAVIOR_CLASSES], fontsize=7)
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("Next bout")
    ax.set_ylabel("Current bout" if show_ylabel else "")
    ax.grid(False)
    for i in range(len(BEHAVIOR_CLASSES)):
        for j in range(len(BEHAVIOR_CLASSES)):
            value = diff[i, j]
            text = "" if abs(value) < 0.005 else true_minus(f"{value * 100:+.0f}%")
            color = "white" if abs(value) > vmax * 0.55 else "#374151"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)
    return image


def main() -> None:
    args = parse_args()
    table_dir = args.table_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    per_video, gt, pred = load_tables(table_dir, args.prediction_type)

    full_attention = model_name("full", "attention", args.capacity, args.seed)
    full_lstm = model_name("full", "lstm", args.capacity, args.seed)
    pose_attention = model_name("pose_relations_only", "attention", args.capacity, args.seed)
    tracks = [
        ("Ground truth", None, gt),
        ("Full attention-256", full_attention, pred),
        ("Full LSTM-256", full_lstm, pred),
        ("Pose + relations\nattention-256", pose_attention, pred),
    ]

    n_frames = get_n_frames(per_video, args.split, args.video_id, full_attention)
    track_rows = []
    rendered_tracks = []
    for label, model_value, table in tracks:
        bouts = bouts_for_track(table, args.split, args.video_id, model_value)
        segments = compress_bouts_to_segments(bouts, n_frames)
        track_rows.append({"track": label, "model_name": model_value or "ground_truth", "n_bouts": len(bouts)})
        rendered_tracks.append((label, model_value, table, segments))

    out_csv = table_dir / f"{args.output_name}_track_summary.csv"
    pd.DataFrame(track_rows).to_csv(out_csv, index=False)
    transition_models = [
        ("Ground truth", None, gt),
        ("Full attention-256", full_attention, pred),
        ("Full LSTM-256", full_lstm, pred),
    ]
    count_df, prob_df = collect_transition_tables(
        per_video,
        gt,
        pred,
        transition_models,
        full_attention,
    )
    count_csv = table_dir / f"{args.output_name}_bout_transition_counts_all_videos.csv"
    prob_csv = table_dir / f"{args.output_name}_bout_transition_probabilities_all_videos.csv"
    count_df.to_csv(count_csv, index=False)
    prob_df.to_csv(prob_csv, index=False)
    gt_for_diff = prob_df[prob_df["track"].eq("Ground truth")][
        ["from_behavior", "to_behavior", "transition_probability"]
    ].rename(columns={"transition_probability": "ground_truth_probability"})
    diff_rows = []
    for track in ["Full attention-256", "Full LSTM-256"]:
        model_probs = prob_df[prob_df["track"].eq(track)][
            ["from_behavior", "to_behavior", "transition_probability"]
        ].rename(columns={"transition_probability": "model_probability"})
        merged = model_probs.merge(gt_for_diff, on=["from_behavior", "to_behavior"], validate="one_to_one")
        merged["difference_model_minus_ground_truth"] = (
            merged["model_probability"] - merged["ground_truth_probability"]
        )
        merged.insert(0, "track", track)
        diff_rows.append(merged)
    diff_df = pd.concat(diff_rows, ignore_index=True)
    diff_csv = table_dir / f"{args.output_name}_bout_transition_differences_all_videos.csv"
    diff_df.to_csv(diff_csv, index=False)

    apply_style()
    fig = plt.figure(figsize=(14.2, 4.8), constrained_layout=False)
    ax = fig.add_axes([0.11, 0.18, 0.86, 0.62])
    y_positions = np.arange(len(rendered_tracks))[::-1]
    for y, (label, _model_value, _table, segments) in zip(y_positions, rendered_tracks):
        draw_track(ax, segments, float(y), args.fps)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([t[0] for t in rendered_tracks])
    ax.set_ylim(-0.65, len(rendered_tracks) - 0.35)
    ax.set_xlim(0, n_frames / args.fps)
    ax.set_xlabel("Time (s)")
    ax.set_title(
        f"Representative held-out full-video ethogram: {args.video_id} "
        f"(YOLO/SPPF, capacity {args.capacity}, seed {args.seed})",
        fontsize=12,
        pad=14,
    )
    ax.grid(axis="x", alpha=0.20)
    ax.grid(axis="y", alpha=0.0)
    handles = [
        Patch(facecolor=BEHAVIOR_FILL_COLORS[cls], edgecolor="none", label=CLASS_LABELS[cls])
        for cls in ALL_CLASSES
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    panel_label(ax, "A", x=-0.12, y=1.25, suffix="", fontsize=14)
    outputs = save_figure(fig, figure_dir / args.output_name)
    plt.close(fig)
    print(f"Wrote {outputs[0]}")
    print(f"Wrote {outputs[1]}")
    print(f"Wrote {out_csv}")
    print(f"Wrote {count_csv}")
    print(f"Wrote {prob_csv}")
    print(f"Wrote {diff_csv}")


if __name__ == "__main__":
    main()
