"""Supplemental full-video ethogram comparison across model families."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
STYLE_DIR = THIS_DIR.parent / "current_analysis"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import BEHAVIOR_FILL_COLORS, apply_style, panel_label, save_figure  # noqa: E402


ALL_CLASSES = ["attack", "investigation", "mount", "other"]
BEHAVIOR_CLASSES = ["attack", "investigation", "mount"]
CLASS_LABELS = {"attack": "Attack", "investigation": "Investigation", "mount": "Mount", "other": "Other"}
CLASS_TO_ID = {name: i for i, name in enumerate(ALL_CLASSES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite_root", type=Path, default=Path("outputs/controlled_comparison_runs/run_20260528_205727"))
    parser.add_argument("--split", default="test_1")
    parser.add_argument("--video_id", default="Mouse062_20160526_18-56-26")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--prediction_suffix", default="behavior.smoothed_frames.csv")
    parser.add_argument("--figure_dir", type=Path, default=Path("Manuscript_penultimate/figures/production/supplement"))
    parser.add_argument("--table_dir", type=Path, default=Path("Manuscript_penultimate/tables/production/supplement"))
    parser.add_argument("--output_name", default="supp_mars_model_type_ethograms")
    return parser.parse_args()


def labels_from_bouts(bouts: pd.DataFrame, n_frames: int) -> np.ndarray:
    labels = np.full(n_frames, CLASS_TO_ID["other"], dtype=np.int8)
    for cls in BEHAVIOR_CLASSES:
        sub = bouts[bouts["class"].eq(cls)]
        for _, row in sub.iterrows():
            start = max(0, int(row["start_frame"]))
            end = min(n_frames - 1, int(row["end_frame"]))
            if end >= start:
                labels[start : end + 1] = CLASS_TO_ID[cls]
    return labels


def labels_from_prediction_csv(path: Path) -> np.ndarray:
    df = pd.read_csv(path, usecols=["frame_idx", "predicted_class"])
    df = df.sort_values("frame_idx")
    n_frames = int(df["frame_idx"].max()) + 1
    labels = np.full(n_frames, CLASS_TO_ID["other"], dtype=np.int8)
    for _, row in df.iterrows():
        cls = str(row["predicted_class"])
        labels[int(row["frame_idx"])] = CLASS_TO_ID.get(cls, CLASS_TO_ID["other"])
    return labels


def segments_from_labels(labels: np.ndarray) -> list[tuple[int, int, str]]:
    start = 0
    current = int(labels[0])
    segments: list[tuple[int, int, str]] = []
    for idx in range(1, len(labels)):
        value = int(labels[idx])
        if value != current:
            segments.append((start, idx - start, ALL_CLASSES[current]))
            start = idx
            current = value
    segments.append((start, len(labels) - start, ALL_CLASSES[current]))
    return segments


def draw_track(ax, segments: list[tuple[int, int, str]], y: float, fps: float) -> None:
    for start, length, cls in segments:
        ax.broken_barh(
            [(start / fps, max(length / fps, 1.0 / fps))],
            (y - 0.34, 0.68),
            facecolors=BEHAVIOR_FILL_COLORS[cls],
            edgecolors="none",
        )
    ax.hlines(y - 0.34, 0, segments[-1][0] / fps + segments[-1][1] / fps, color="black", linewidth=0.6)


def model_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    base = args.suite_root
    split = args.split
    video = args.video_id
    return [
        (
            "Full attention-256",
            base / "neural_eval" / "yolo_sppf_full_attention256_seed42" / split / video / f"{video}.{args.prediction_suffix}",
        ),
        (
            "Full LSTM-256",
            base / "neural_eval" / "yolo_sppf_full_lstm256_seed42" / split / video / f"{video}.{args.prediction_suffix}",
        ),
        (
            "RF pose window",
            base / "classical" / "eval" / "mars_yolo_sppf" / "rf_pose_window_seed42" / "pose_window" / split / video / f"{video}.{args.prediction_suffix}",
        ),
        (
            "XGBoost pose window",
            base / "classical" / "eval" / "mars_yolo_sppf" / "xgb_pose_window_seed42" / "pose_window" / split / video / f"{video}.{args.prediction_suffix}",
        ),
    ]


def main() -> None:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    gt_path = args.suite_root / "neural_eval" / "yolo_sppf_full_attention256_seed42" / "ground_truth_bouts.csv"
    gt = pd.read_csv(gt_path)
    gt = gt[
        gt["split"].eq(args.split)
        & gt["video_id"].eq(args.video_id)
        & gt["prediction_type"].eq("smoothed_frames")
    ].copy()
    predictions = []
    for label, path in model_paths(args):
        if not path.is_file():
            raise FileNotFoundError(path)
        labels = labels_from_prediction_csv(path)
        predictions.append((label, labels, str(path)))
    n_frames = len(predictions[0][1])
    gt_labels = labels_from_bouts(gt, n_frames)
    tracks = [("Ground truth", gt_labels, str(gt_path))] + predictions
    summary = [
        {
            "track": label,
            "source": source,
            "n_frames": len(labels),
            "attack_frames": int(np.sum(labels == CLASS_TO_ID["attack"])),
            "investigation_frames": int(np.sum(labels == CLASS_TO_ID["investigation"])),
            "mount_frames": int(np.sum(labels == CLASS_TO_ID["mount"])),
            "other_frames": int(np.sum(labels == CLASS_TO_ID["other"])),
        }
        for label, labels, source in tracks
    ]
    pd.DataFrame(summary).to_csv(args.table_dir / f"{args.output_name}_track_summary.csv", index=False)

    apply_style()
    fig, ax = plt.subplots(figsize=(13.5, 5.6), constrained_layout=True)
    y_positions = np.arange(len(tracks))[::-1]
    for y, (label, labels, _source) in zip(y_positions, tracks):
        draw_track(ax, segments_from_labels(labels), float(y), args.fps)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([track[0] for track in tracks])
    ax.set_ylim(-0.65, len(tracks) - 0.35)
    ax.set_xlim(0, n_frames / args.fps)
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Representative MARS ethogram by model family: {args.video_id}", fontsize=12, pad=14)
    ax.grid(axis="x", alpha=0.20)
    ax.grid(axis="y", alpha=0.0)
    handles = [Patch(facecolor=BEHAVIOR_FILL_COLORS[c], edgecolor="none", label=CLASS_LABELS[c]) for c in ALL_CLASSES]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.25), ncol=4, frameon=False)
    panel_label(ax, "A", x=-0.10, y=1.20, fontsize=14)
    save_figure(fig, args.figure_dir / args.output_name)
    plt.close(fig)
    print(f"Wrote {args.figure_dir / (args.output_name + '.png')}")


if __name__ == "__main__":
    main()
