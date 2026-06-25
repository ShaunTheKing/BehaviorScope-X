"""Explore IntegraPose-style Fly-v-Fly bout reconstruction from YOLO labels.

This diagnostic converts per-frame Ultralytics YOLO label files into a single
frame-wise behavior ethogram, applies a small smoothing/gap-fill sweep, and
evaluates the reconstructed bouts against the Fly-v-Fly primary annotation.

It is intentionally kept separate from manuscript figures/results. The default
input path supplied during development was a YOLO ``predict`` directory for
movie 2, so the output should be interpreted as an exploratory training-movie
diagnostic unless the same procedure is run on held-out movies with the same
evaluation protocol.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
MANUSCRIPT_ROOT = THIS_DIR.parents[1]
SHARED_DIR = REPO_ROOT / "shared_scripts"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from fly_run_all_eval import (  # noqa: E402
    BEHAVIOR_CLASSES,
    BEHAVIOR_PRIORITY,
    CLASS_TO_IDX,
    CLASSES,
    IDX_TO_CLASS,
    actions_to_bouts,
    bouts_to_frames,
    evaluate_frame_arrays,
    iou_suffix,
    load_actions_file,
    n_frames_for_job,
    VideoJob,
)


DEFAULT_LABEL_DIR = Path(r"F:\BehaviorScope-N\Fly_v_Fly_2021\Aggression\Aggression\movie2\predict-4\labels")
DEFAULT_MOVIE_ROOT = Path(r"F:\BehaviorScope-N\Fly_v_Fly_2021\Aggression\Aggression")
DEFAULT_MOVIE_DIR = DEFAULT_MOVIE_ROOT / "movie2"
DEFAULT_OUT_DIR = MANUSCRIPT_ROOT / "tables" / "exploratory" / "fly_v_fly_integrapose_style"

PRIORITY = [CLASS_TO_IDX[name] for name in BEHAVIOR_PRIORITY if name in CLASS_TO_IDX]
OTHER = CLASS_TO_IDX["other"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label_dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--movie_root", type=Path, default=DEFAULT_MOVIE_ROOT)
    parser.add_argument("--movie_dir", type=Path, default=DEFAULT_MOVIE_DIR)
    parser.add_argument("--movie_id", type=int, default=2)
    parser.add_argument("--movie_ids", type=int, nargs="+", default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--label_cache_dir", type=Path, default=None)
    parser.add_argument("--rebuild_label_cache", action="store_true")
    parser.add_argument("--smooth_window", type=int, default=3)
    parser.add_argument("--gap_values", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument(
        "--write_frame_inputs",
        action="store_true",
        help="Write one row per parsed label file. Slow for held-out runs with hundreds of thousands of files.",
    )
    return parser.parse_args()


def frame_index_from_label_name(path: Path, movie_id: int) -> int | None:
    match = re.search(rf"movie{int(movie_id)}_(\d+)$", path.stem)
    if not match:
        return None
    return int(match.group(1)) - 1


def priority_collapse(classes: list[int]) -> int:
    if not classes:
        return OTHER
    present = set(int(c) for c in classes)
    for cls_idx in PRIORITY:
        if cls_idx in present:
            return cls_idx
    return OTHER


def load_yolo_label_frames(label_dir: Path, movie_id: int, n_frames: int) -> tuple[np.ndarray, pd.DataFrame]:
    labels = np.full(n_frames, OTHER, dtype=np.int16)
    rows = []
    files = sorted(label_dir.glob(f"movie{int(movie_id)}_*.txt"))
    for path in files:
        frame = frame_index_from_label_name(path, movie_id)
        if frame is None or frame < 0 or frame >= n_frames:
            continue
        classes = []
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_idx = int(float(parts[0]))
            except ValueError:
                continue
            if 0 <= cls_idx < len(CLASSES):
                classes.append(cls_idx)
        labels[frame] = priority_collapse(classes)
        counts = Counter(classes)
        rows.append({
            "frame": frame,
            "label_file": str(path),
            "n_detections": len(classes),
            "collapsed_class": IDX_TO_CLASS[int(labels[frame])],
            "class_counts_json": json.dumps({IDX_TO_CLASS[k]: int(v) for k, v in sorted(counts.items())}),
        })
    return labels, pd.DataFrame(rows)


def load_yolo_label_frames_fast(
    label_dir: Path,
    movie_id: int,
    n_frames: int,
    *,
    write_frame_inputs: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fast parser for Ultralytics per-frame txt labels.

    Uses only the first token of each detection row as behavior class. Any
    trailing track ID written by YOLO track is intentionally ignored.
    """
    labels = np.full(n_frames, OTHER, dtype=np.int16)
    rows = [] if write_frame_inputs else None
    prefix = f"movie{int(movie_id)}_"
    prefix_len = len(prefix)
    suffix = ".txt"
    parsed = 0

    with os.scandir(label_dir) as it:
        for entry in it:
            if not entry.is_file():
                continue
            name = entry.name
            if not (name.startswith(prefix) and name.endswith(suffix)):
                continue
            frame_text = name[prefix_len:-len(suffix)]
            if not frame_text.isdigit():
                continue
            frame = int(frame_text) - 1
            if frame < 0 or frame >= n_frames:
                continue

            present: set[int] = set()
            n_detections = 0
            with open(entry.path, "rb") as fh:
                for line in fh:
                    stripped = line.lstrip()
                    if not stripped:
                        continue
                    first = stripped[:1]
                    if b"0" <= first <= b"9":
                        cls_idx = first[0] - 48
                        if 0 <= cls_idx < len(CLASSES):
                            present.add(cls_idx)
                            n_detections += 1

            label = OTHER
            if present:
                for cls_idx in PRIORITY:
                    if cls_idx in present:
                        label = cls_idx
                        break
            labels[frame] = label
            parsed += 1

            if rows is not None:
                rows.append({
                    "frame": frame,
                    "label_file": entry.path,
                    "n_detections": n_detections,
                    "collapsed_class": IDX_TO_CLASS[int(label)],
                    "class_counts_json": "",
                })

    if parsed == 0:
        raise RuntimeError(f"No label files parsed for movie{movie_id} in {label_dir}")
    frame_table = pd.DataFrame(rows) if rows is not None else pd.DataFrame()
    return labels, frame_table


def majority_smooth(labels: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1:
        return labels.copy()
    if window % 2 == 0:
        raise ValueError("smooth_window must be odd")
    radius = window // 2
    out = labels.copy()
    for i in range(len(labels)):
        start = max(0, i - radius)
        end = min(len(labels), i + radius + 1)
        vals = labels[start:end]
        counts = Counter(int(v) for v in vals)
        max_count = max(counts.values())
        tied = {k for k, v in counts.items() if v == max_count}
        if int(labels[i]) in tied:
            out[i] = labels[i]
        else:
            out[i] = priority_collapse(list(tied))
    return out.astype(np.int16)


def fill_same_class_gaps(labels: np.ndarray, max_gap: int) -> np.ndarray:
    max_gap = int(max_gap)
    if max_gap <= 0:
        return labels.copy()
    out = labels.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i] != OTHER:
            i += 1
            continue
        gap_start = i
        while i < n and out[i] == OTHER:
            i += 1
        gap_end = i - 1
        gap_len = gap_end - gap_start + 1
        if gap_len > max_gap or gap_start == 0 or i >= n:
            continue
        left = int(out[gap_start - 1])
        right = int(out[i])
        if left == right and left != OTHER:
            out[gap_start:i] = left
    return out.astype(np.int16)


def metrics_row(variant: str, pred: np.ndarray, gt: np.ndarray, *, movie_id: int) -> dict:
    metrics = evaluate_frame_arrays(gt, pred)
    row = {
        "movie_id": int(movie_id),
        "variant": variant,
        "frame_macro_precision": metrics["frame_macro_precision"],
        "frame_macro_recall": metrics["frame_macro_recall"],
        "frame_macro_f1": metrics["frame_macro_f1"],
        "gt_bouts_total": metrics["gt_bouts_total"],
        "pred_bouts_total": metrics["pred_bouts_total"],
        "per_class_precision_json": json.dumps(metrics["per_class_precision"], sort_keys=True),
        "per_class_recall_json": json.dumps(metrics["per_class_recall"], sort_keys=True),
        "per_class_f1_json": json.dumps(metrics["per_class_f1"], sort_keys=True),
        "confusion_matrix_json": json.dumps(metrics["confusion_matrix"]),
    }
    for thr in [0.10, 0.25, 0.50]:
        suffix = iou_suffix(thr)
        row[f"bout_macro_precision_{suffix}"] = metrics[f"bout_macro_precision_iou{suffix}"]
        row[f"bout_macro_recall_{suffix}"] = metrics[f"bout_macro_recall_iou{suffix}"]
        row[f"bout_macro_f1_{suffix}"] = metrics[f"bout_macro_f1_iou{suffix}"]
        row[f"per_class_bout_f1_{suffix}_json"] = json.dumps(
            metrics[f"per_class_bout_f1_{suffix}"], sort_keys=True
        )
    return row


def label_counts(labels: np.ndarray, variant: str, movie_id: int) -> list[dict]:
    rows = []
    for idx, cls in IDX_TO_CLASS.items():
        frames = int(np.sum(labels == idx))
        rows.append({
            "movie_id": int(movie_id),
            "variant": variant,
            "class": cls,
            "frames": frames,
            "seconds": frames / 30.0,
        })
    return rows


def run_movie(args: argparse.Namespace, movie_id: int, movie_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    primary_actions = movie_dir / f"movie{movie_id}_actions.mat"
    mp4 = movie_dir / f"movie{movie_id}.mp4"
    if not primary_actions.is_file():
        raise FileNotFoundError(primary_actions)
    if not args.label_dir.is_dir():
        raise FileNotFoundError(args.label_dir)

    job = VideoJob(
        movie_id=int(movie_id),
        mp4_path=mp4,
        actions_mat_dir=movie_dir,
        primary_actions_path=primary_actions,
        secondary_actions_paths=[],
    )
    actions = load_actions_file(primary_actions)
    n_frames = n_frames_for_job(job, actions)
    gt = bouts_to_frames(actions_to_bouts(actions), n_frames)
    cache_dir = args.label_cache_dir or (args.output_dir / "parsed_label_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"movie{int(movie_id)}_raw_yolo_labels.npy"
    if cache_path.is_file() and not args.rebuild_label_cache and not args.write_frame_inputs:
        raw = np.load(cache_path).astype(np.int16)
        if len(raw) != n_frames:
            raise ValueError(f"Cached labels for movie{movie_id} have length {len(raw)}, expected {n_frames}")
        frame_table = pd.DataFrame()
    else:
        raw, frame_table = load_yolo_label_frames_fast(
            args.label_dir,
            movie_id,
            n_frames,
            write_frame_inputs=bool(args.write_frame_inputs),
        )
        np.save(cache_path, raw.astype(np.int16))
    smooth = majority_smooth(raw, args.smooth_window)

    variants = {"raw_yolo_labels": raw, f"smooth{args.smooth_window}": smooth}
    for gap in args.gap_values:
        variants[f"smooth{args.smooth_window}_gap{gap}"] = fill_same_class_gaps(smooth, gap)

    metric_rows = []
    count_rows = []
    for variant, pred in variants.items():
        metric_rows.append(metrics_row(variant, pred, gt, movie_id=movie_id))
        count_rows.extend(label_counts(pred, variant, movie_id))
    count_rows.extend(label_counts(gt, "primary_annotation", movie_id))

    return pd.DataFrame(metric_rows), pd.DataFrame(count_rows), frame_table


def build_over_under(counts: pd.DataFrame) -> pd.DataFrame:
    primary = counts[counts["variant"].eq("primary_annotation")][["movie_id", "class", "frames", "seconds"]]
    primary = primary.rename(columns={"frames": "primary_frames", "seconds": "primary_seconds"})
    pred = counts[~counts["variant"].eq("primary_annotation")].copy()
    out = pred.merge(primary, on=["movie_id", "class"], how="left")
    out["delta_frames"] = out["frames"] - out["primary_frames"]
    out["delta_seconds"] = out["seconds"] - out["primary_seconds"]
    out["pred_to_primary_ratio"] = out["seconds"] / out["primary_seconds"].replace(0, np.nan)
    return out


def build_metric_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "frame_macro_precision",
        "frame_macro_recall",
        "frame_macro_f1",
        "bout_macro_precision_010",
        "bout_macro_recall_010",
        "bout_macro_f1_010",
        "bout_macro_precision_025",
        "bout_macro_recall_025",
        "bout_macro_f1_025",
        "bout_macro_precision_050",
        "bout_macro_recall_050",
        "bout_macro_f1_050",
        "gt_bouts_total",
        "pred_bouts_total",
    ]
    summary = (
        metrics.groupby("variant", as_index=False)[metric_cols]
        .mean(numeric_only=True)
        .rename(columns={c: f"mean_{c}" for c in metric_cols})
    )
    summary["n_movies"] = metrics.groupby("variant")["movie_id"].nunique().reindex(summary["variant"]).to_numpy()
    return summary


def main() -> None:
    args = parse_args()
    movie_ids = args.movie_ids or [int(args.movie_id)]
    all_metrics = []
    all_counts = []
    all_frames = []
    for movie_id in movie_ids:
        movie_dir = args.movie_root / f"movie{int(movie_id)}" if args.movie_ids else args.movie_dir
        metrics, counts, frames = run_movie(args, int(movie_id), movie_dir)
        all_metrics.append(metrics)
        all_counts.append(counts)
        all_frames.append(frames.assign(movie_id=int(movie_id)))

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    counts_all = pd.concat(all_counts, ignore_index=True)
    frames_all = pd.concat(all_frames, ignore_index=True) if any(len(f) for f in all_frames) else pd.DataFrame()
    over_under = build_over_under(counts_all)
    summary = build_metric_summary(metrics_all)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = "heldout" if len(movie_ids) > 1 else f"movie{movie_ids[0]}"
    metrics_all.to_csv(args.output_dir / f"{prefix}_integrapose_style_yolo_label_sweep_metrics.csv", index=False)
    counts_all.to_csv(args.output_dir / f"{prefix}_integrapose_style_yolo_label_frame_counts.csv", index=False)
    if args.write_frame_inputs:
        frames_all.to_csv(args.output_dir / f"{prefix}_integrapose_style_yolo_label_frame_inputs.csv", index=False)
    over_under.to_csv(args.output_dir / f"{prefix}_integrapose_style_yolo_label_over_under_calling.csv", index=False)
    summary.to_csv(args.output_dir / f"{prefix}_integrapose_style_yolo_label_sweep_summary.csv", index=False)
    print(f"[write] {args.output_dir}")
    print(summary[[
        "variant",
        "mean_frame_macro_precision",
        "mean_frame_macro_recall",
        "mean_frame_macro_f1",
        "mean_bout_macro_f1_010",
        "mean_bout_macro_f1_025",
        "mean_bout_macro_f1_050",
        "mean_pred_bouts_total",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
