"""Render MARS ethogram bout examples from current YOLO/SPPF outputs."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
STYLE_DIR = THIS_DIR.parent / "current_analysis"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import (  # noqa: E402
    BEHAVIOR_COLORS,
    apply_style,
    save_figure,
)


BEHAVIOR_CLASSES = ["attack", "investigation", "mount"]
CLASS_LABELS = {
    "attack": "Attack",
    "investigation": "Investigation",
    "mount": "Mount",
    "other": "Other",
}


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
    parser.add_argument(
        "--mp4_cache",
        type=Path,
        default=Path("outputs/source_mp4_cache/mars"),
    )
    parser.add_argument("--prediction_type", default="smoothed_frames")
    parser.add_argument("--full_model", default="yolo_sppf_full_attention256_seed42")
    parser.add_argument(
        "--reference_model",
        default="yolo_sppf_pose_relations_only_attention256_seed42",
    )
    parser.add_argument("--full_label", default="Full model")
    parser.add_argument("--reference_label", default="Pose + relations only")
    parser.add_argument("--output_name", default="fig5_yolo_ethogram_bout_examples")
    return parser.parse_args()


def load_bouts(table_dir: Path, prediction_type: str, full_model: str, reference_model: str):
    cols = [
        "split",
        "video_id",
        "prediction_type",
        "source",
        "class",
        "start_frame",
        "end_frame",
        "duration_frames",
        "model_name",
    ]
    pred = pd.read_csv(table_dir / "yolo_lstm_attention_predicted_bouts_long.csv.gz", usecols=cols)
    pred = pred[pred["prediction_type"].eq(prediction_type)].copy()
    full = pred[pred["model_name"].eq(full_model)].copy()
    ref = pred[pred["model_name"].eq(reference_model)].copy()
    if full.empty:
        raise ValueError(f"No predicted bouts for full_model={full_model}")
    if ref.empty:
        raise ValueError(f"No predicted bouts for reference_model={reference_model}")

    gt = pd.read_csv(table_dir / "yolo_lstm_attention_ground_truth_bouts_long.csv.gz", usecols=cols)
    gt = gt[gt["prediction_type"].eq(prediction_type)].copy()
    gt = gt[gt["model_name"].eq(full_model)].copy()
    if gt.empty:
        raise ValueError(f"No ground-truth bouts paired with full_model={full_model}")
    gt = gt.drop_duplicates(["split", "video_id", "class", "start_frame", "end_frame"]).copy()
    return gt, ref, full


def label_at(bouts: pd.DataFrame, split: str, video_id: str, frame: int) -> str:
    sub = bouts[
        bouts["split"].eq(split)
        & bouts["video_id"].eq(video_id)
        & bouts["start_frame"].le(frame)
        & bouts["end_frame"].ge(frame)
    ].copy()
    if sub.empty:
        return "other"
    sub["duration"] = sub["end_frame"] - sub["start_frame"] + 1
    return str(sub.sort_values("duration", ascending=False).iloc[0]["class"])


def interval_at(
    bouts: pd.DataFrame,
    split: str,
    video_id: str,
    frame: int,
    cls: str | None = None,
) -> tuple[int, int] | None:
    sub = bouts[
        bouts["split"].eq(split)
        & bouts["video_id"].eq(video_id)
        & bouts["start_frame"].le(frame)
        & bouts["end_frame"].ge(frame)
    ].copy()
    if cls is not None:
        sub = sub[sub["class"].eq(cls)]
    if sub.empty:
        return None
    sub["duration"] = sub["end_frame"] - sub["start_frame"] + 1
    row = sub.sort_values("duration", ascending=False).iloc[0]
    return int(row["start_frame"]), int(row["end_frame"])


def read_frame(mp4_cache: Path, split: str, video_id: str, frame_idx: int):
    path = mp4_cache / split / f"{video_id}.mp4"
    if not path.exists():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read {path} frame {frame_idx}")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def select_examples(
    gt: pd.DataFrame,
    ref: pd.DataFrame,
    full: pd.DataFrame,
    mp4_cache: Path,
) -> list[dict[str, object]]:
    examples: list[dict[str, object]] = []
    captions = {
        "investigation": "Pose + relations only confuses an investigation bout.",
        "attack": "Pose + relations only confuses an attack bout.",
        "mount": "Pose + relations only confuses a mount bout.",
    }
    # Prefer longer bouts, but sample multiple frames inside each bout because
    # predictions may align with only part of a human-labeled episode.
    gt_sorted = gt[gt["class"].isin(BEHAVIOR_CLASSES)].sort_values(
        ["class", "duration_frames"], ascending=[True, False]
    )
    for cls in ["investigation", "attack", "mount"]:
        found = None
        class_bouts = gt_sorted[gt_sorted["class"].eq(cls)].copy()
        for _, row in class_bouts.iterrows():
            split = str(row["split"])
            video_id = str(row["video_id"])
            if not (mp4_cache / split / f"{video_id}.mp4").exists():
                continue
            start = int(row["start_frame"])
            end = int(row["end_frame"])
            duration = max(1, end - start + 1)
            candidate_frames = sorted(
                set(
                    [
                        start + duration // 2,
                        start + max(1, duration // 3),
                        end - max(1, duration // 3),
                        start,
                        end,
                    ]
                )
            )
            # Add sampled frames for longer bouts.
            if duration > 30:
                step = max(4, duration // 8)
                candidate_frames.extend(range(start, end + 1, step))
            for frame in candidate_frames:
                gt_label = label_at(gt, split, video_id, frame)
                full_label = label_at(full, split, video_id, frame)
                ref_label = label_at(ref, split, video_id, frame)
                if gt_label == cls and full_label == cls and ref_label != cls:
                    found = {
                        "panel": chr(ord("A") + len(examples)),
                        "split": split,
                        "video_id": video_id,
                        "frame": int(frame),
                        "class": cls,
                        "gt_label": gt_label,
                        "reference_label": ref_label,
                        "full_label": full_label,
                        "caption": captions[cls],
                    }
                    break
            if found is not None:
                break
        if found is not None:
            examples.append(found)
    if not examples:
        raise RuntimeError("No qualitative examples found for the requested model pair.")
    return examples


def add_label_box(ax, model_name: str, gt_label: str, pred_label: str, frame: int) -> None:
    ax.text(
        0.015,
        0.985,
        f"{model_name}\nGround truth: {gt_label} | Pred: {pred_label}\nFrame {frame}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color="white",
        bbox=dict(facecolor="black", alpha=0.72, edgecolor="none", pad=4),
    )
    edge = BEHAVIOR_COLORS.get(pred_label, "#777777")
    ax.add_patch(
        plt.Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            fill=False,
            linewidth=3,
            edgecolor=edge,
        )
    )


def draw_timeline(
    ax,
    gt: pd.DataFrame,
    ref: pd.DataFrame,
    full: pd.DataFrame,
    example: dict[str, object],
    reference_label: str,
    full_label: str,
) -> None:
    split = str(example["split"])
    video_id = str(example["video_id"])
    frame = int(example["frame"])
    cls = str(example["class"])
    gt_interval = interval_at(gt, split, video_id, frame, cls)
    if gt_interval is None:
        raise RuntimeError(f"Selected frame not inside GT {cls}: {example}")
    start = max(0, min(gt_interval[0] - 45, frame - 90))
    end = max(gt_interval[1] + 45, frame + 90)
    tracks = [("Ground truth", gt), (reference_label, ref), (full_label, full)]
    y_pos = [2, 1, 0]
    for (label, bouts), y in zip(tracks, y_pos):
        sub = bouts[
            bouts["split"].eq(split)
            & bouts["video_id"].eq(video_id)
            & bouts["end_frame"].ge(start)
            & bouts["start_frame"].le(end)
            & bouts["class"].isin(BEHAVIOR_CLASSES)
        ].copy()
        for _, row in sub.iterrows():
            cls_i = str(row["class"])
            x0 = max(int(row["start_frame"]), start)
            x1 = min(int(row["end_frame"]), end)
            ax.broken_barh(
                [(x0, max(1, x1 - x0 + 1))],
                (y - 0.3, 0.6),
                facecolors=BEHAVIOR_COLORS[cls_i],
                edgecolors="none",
                alpha=0.94,
            )
        ax.text(start - (end - start) * 0.012, y, label, ha="right", va="center", fontsize=8)
    ax.axvline(frame, color="black", linewidth=1.2)
    ax.set_xlim(start, end)
    ax.set_ylim(-0.6, 2.6)
    ax.set_yticks([])
    ax.set_xlabel("Frame")
    ax.grid(axis="x", alpha=0.18)
    ax.grid(axis="y", alpha=0.0)


def render(
    examples: list[dict[str, object]],
    gt: pd.DataFrame,
    ref: pd.DataFrame,
    full: pd.DataFrame,
    mp4_cache: Path,
    reference_label: str,
    full_label: str,
    figure_dir: Path,
    output_name: str,
    model_context: str,
) -> None:
    apply_style()
    fig = plt.figure(figsize=(10.5, 3.8 * len(examples) + 0.65), constrained_layout=True)
    subfigs = fig.subfigures(
        nrows=len(examples) + 1,
        ncols=1,
        hspace=0.08,
        height_ratios=[1.0] * len(examples) + [0.20],
    )
    subfigs = np.atleast_1d(subfigs).ravel().tolist()
    panel_subfigs = subfigs[: len(examples)]
    legend_subfig = subfigs[-1]
    for subfig, ex in zip(panel_subfigs, examples):
        gs = subfig.add_gridspec(2, 2, height_ratios=[4.0, 1.05])
        ax_ref = subfig.add_subplot(gs[0, 0])
        ax_full = subfig.add_subplot(gs[0, 1])
        ax_timeline = subfig.add_subplot(gs[1, :])
        frame = read_frame(mp4_cache, str(ex["split"]), str(ex["video_id"]), int(ex["frame"]))
        for ax, label, pred in [
            (ax_ref, reference_label, str(ex["reference_label"])),
            (ax_full, full_label, str(ex["full_label"])),
        ]:
            ax.imshow(frame)
            ax.set_axis_off()
            add_label_box(ax, label, str(ex["gt_label"]), pred, int(ex["frame"]))
        draw_timeline(ax_timeline, gt, ref, full, ex, reference_label, full_label)
        subfig.suptitle(f"{ex['panel']}. {ex['caption']}", fontsize=10, fontweight="bold")
    handles = [Patch(color=BEHAVIOR_COLORS[c], label=CLASS_LABELS[c]) for c in BEHAVIOR_CLASSES]
    ax_legend = legend_subfig.add_subplot(111)
    ax_legend.set_axis_off()
    ax_legend.text(
        0.5,
        0.76,
        model_context,
        ha="center",
        va="center",
        fontsize=8,
        color="#374151",
        transform=ax_legend.transAxes,
    )
    ax_legend.legend(
        handles=handles,
        title="Behavior",
        loc="center",
        bbox_to_anchor=(0.5, 0.18),
        ncol=3,
        frameon=True,
        fontsize=7.5,
        title_fontsize=8,
    )
    save_figure(fig, figure_dir / output_name)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    table_dir = args.table_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    mp4_cache = args.mp4_cache.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    gt, ref, full = load_bouts(table_dir, args.prediction_type, args.full_model, args.reference_model)
    examples = select_examples(gt, ref, full, mp4_cache)
    model_context = (
        "Held-out MARS examples from attention head, hidden dimension 256, seed 42; "
        "smoothed frame outputs."
    )
    render(
        examples,
        gt,
        ref,
        full,
        mp4_cache,
        args.reference_label,
        args.full_label,
        figure_dir,
        args.output_name,
        model_context,
    )
    rows = []
    for ex in examples:
        rows.append(
            {
                **ex,
                "reference_model": args.reference_model,
                "full_model": args.full_model,
                "prediction_type": args.prediction_type,
            }
        )
    out_csv = table_dir / f"{args.output_name}_selected_examples.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {figure_dir / (args.output_name + '.png')}")
    print(f"Wrote {figure_dir / (args.output_name + '.pdf')}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
