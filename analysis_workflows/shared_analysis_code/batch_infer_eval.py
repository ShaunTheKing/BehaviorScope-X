#!/usr/bin/env python3
"""
Batch-run BehaviorScope-X on MARS test splits and evaluate against .annot bouts.

This script is intentionally a single experiment driver:

1. discover MARS test video folders under --mars-root/test_1 and/or test_2
2. convert each *_Top*.seq to an intermediate MP4 when requested
3. run BehaviorScope-X/infer_x.py with calibrated crop scaling
4. evaluate raw window predictions and smoothed per-frame predictions
   against the human BENTO .annot file
5. write per-video, per-class, bout, and manifest CSVs

The default source mode is "mp4" because OpenCV/Ultralytics handling of
NorPix .seq files is environment-dependent. This package does not include the
legacy MARS-to-YOLO conversion script; use pre-converted MP4 files for public
GUI/tutorial workflows.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import shutil

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

CLASSES = ["attack", "investigation", "mount", "other"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}
BEHAVIOR_CLASSES = ["attack", "investigation", "mount"]
IOU_THRESHOLDS = [0.10, 0.25, 0.50]


def probe_video(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    opened = bool(cap.isOpened())
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) if opened else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) if opened else 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) if opened else 0
    cap.release()
    return {
        "opened": opened,
        "frames": frame_count,
        "width": width,
        "height": height,
        "valid": bool(opened and frame_count > 0 and width > 0 and height > 0),
    }


@dataclass
class VideoJob:
    split: str
    video_id: str
    video_dir: Path
    seq_path: Path | None
    annot_path: Path
    seek_mat_path: Path | None


def _safe_name(text: str) -> str:
    keep = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "video"


def find_seq(video_dir: Path) -> Path | None:
    preferred = sorted(video_dir.glob("*_Top_J85.seq"))
    if preferred:
        return preferred[0]
    preferred = sorted(video_dir.glob("*_Top.seq"))
    if preferred:
        return preferred[0]
    any_top = sorted(video_dir.glob("*Top*.seq"))
    if any_top:
        return any_top[0]
    any_seq = sorted(video_dir.glob("*.seq"))
    return any_seq[0] if any_seq else None


def find_annot(video_dir: Path, *, allow_pred_named: bool = False) -> Path | None:
    annots = sorted(video_dir.glob("*.annot"))
    if not annots:
        return None
    filtered = annots
    if not allow_pred_named:
        filtered = [
            p for p in annots
            if "pred" not in p.name.lower()
            and "prediction" not in p.name.lower()
        ]
    if not filtered:
        return None
    # Prefer behavior/action annotation files over scorer names if present.
    priority_terms = ("bhvr", "behavior", "behav", "action", "anno")
    for term in priority_terms:
        matches = [p for p in filtered if term in p.name.lower()]
        if matches:
            return matches[0]
    return filtered[0]


def discover_jobs(
    mars_root: Path,
    splits: Sequence[str],
    *,
    allow_pred_named_annots: bool = False,
    allow_mp4_only_discovery: bool = False,
) -> list[VideoJob]:
    jobs: list[VideoJob] = []
    for split in splits:
        split_dir = mars_root / split
        if not split_dir.is_dir():
            print(f"[discover] WARNING: split folder missing: {split_dir}", flush=True)
            continue
        for video_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
            seq_path = find_seq(video_dir)
            annot_path = find_annot(video_dir, allow_pred_named=allow_pred_named_annots)
            if seq_path is None and not allow_mp4_only_discovery:
                print(f"[discover] skip {video_dir}: no .seq file", flush=True)
                continue
            if annot_path is None:
                print(f"[discover] skip {video_dir}: no accepted .annot file", flush=True)
                continue
            seek = seq_path.with_name(seq_path.stem + "-seek.mat") if seq_path is not None else None
            jobs.append(
                VideoJob(
                    split=split,
                    video_id=video_dir.name,
                    video_dir=video_dir,
                    seq_path=seq_path,
                    annot_path=annot_path,
                    seek_mat_path=seek if seek is not None and seek.is_file() else None,
                )
            )
    return jobs


def convert_seq_to_mp4(
    seq_path: Path,
    mp4_path: Path,
    *,
    seek_mat_path: Path | None,
    fps: float,
    overwrite: bool,
    tolerate_bad_frames: bool = False,
) -> dict:
    try:
        from mars_to_yolo_pose import SeqReader  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "SEQ-to-MP4 conversion requires the legacy mars_to_yolo_pose.py "
            "converter, which is not included in this package. Use pre-converted "
            "MP4 files or provide a source manifest that points to MP4 videos."
        ) from exc

    if mp4_path.is_file() and not overwrite:
        probe = probe_video(mp4_path)
        if not probe["valid"]:
            try:
                mp4_path.unlink()
            except OSError:
                pass
        else:
            frame_count = int(probe["frames"])
            width = int(probe["width"])
            height = int(probe["height"])
            return {
                "converted": False,
                "mp4_path": str(mp4_path),
                "frames": frame_count,
                "width": width,
                "height": height,
            }

    if mp4_path.is_file() and not overwrite:
        probe = probe_video(mp4_path)
        return {
            "converted": False,
            "mp4_path": str(mp4_path),
            "frames": int(probe["frames"]),
            "width": int(probe["width"]),
            "height": int(probe["height"]),
        }

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with SeqReader(seq_path) as reader:
        reader.build_seek_table(str(seek_mat_path) if seek_mat_path else None)
        n_frames = len(reader.seek_table or [])
        if n_frames <= 0:
            n_frames = int(reader.num_frames)
        width = int(reader.width)
        height = int(reader.height)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, float(fps), (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {mp4_path}")
        bad_frames = 0
        first_bad_frames: list[int] = []
        prev_bgr = None
        try:
            for idx in range(n_frames):
                try:
                    frame = reader.read_frame(idx)
                    if frame.ndim == 2:
                        bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        # PIL/OpenCV decoder returns RGB; OpenCV writer expects BGR.
                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    prev_bgr = bgr
                except Exception:
                    if not tolerate_bad_frames:
                        raise
                    bad_frames += 1
                    if len(first_bad_frames) < 10:
                        first_bad_frames.append(int(idx))
                    if prev_bgr is None:
                        bgr = np.zeros((height, width, 3), dtype=np.uint8)
                    else:
                        bgr = prev_bgr.copy()
                writer.write(bgr)
                if idx and idx % 5000 == 0:
                    print(f"[convert] {seq_path.name}: {idx}/{n_frames} frames", flush=True)
        finally:
            writer.release()

    return {
        "converted": True,
        "mp4_path": str(mp4_path),
        "frames": n_frames,
        "width": width,
        "height": height,
        "elapsed_s": round(time.time() - t0, 3),
        "bad_frames_replaced": bad_frames,
        "first_bad_frames": first_bad_frames,
    }




def convert_seq_to_mp4_with_retries(
    seq_path: Path,
    mp4_path: Path,
    *,
    seek_mat_path: Path | None,
    fps: float,
    overwrite: bool,
    retries: int,
    retry_delay_s: float,
    tolerate_bad_frames: bool = False,
) -> dict:
    attempts = max(1, int(retries) + 1)
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            info = convert_seq_to_mp4(
                seq_path,
                mp4_path,
                seek_mat_path=seek_mat_path,
                fps=fps,
                overwrite=overwrite,
                tolerate_bad_frames=tolerate_bad_frames,
            )
            info["conversion_attempts"] = attempt
            info["conversion_retries_requested"] = int(retries)
            return info
        except Exception as exc:
            last_exc = exc
            if mp4_path.exists():
                try:
                    mp4_path.unlink()
                except OSError:
                    pass
            if attempt >= attempts:
                break
            print(
                f"[convert] attempt {attempt}/{attempts} failed for {seq_path.name}: {exc}; retrying",
                flush=True,
            )
            time.sleep(max(0.0, float(retry_delay_s)))
    assert last_exc is not None
    raise last_exc

def run_inference(
    *,
    python_exe: str,
    infer_script: Path,
    model_path: Path,
    model_config: Path | None,
    yolo_weights: Path,
    source_path: Path,
    output_csv: Path,
    output_video: Path | None,
    calibration_json: Path | None,
    fps: float,
    window_stride: int,
    yolo_batch: int,
    pose_conf_threshold: float,
    animal_scale_factor: float,
    group_scale_factor: float,
    temporal_smoothing_window: int,
    bout_min_duration_frames: int,
    device: str,
    amp: bool,
    log_metrics: bool,
    print_each_window: bool,
    export_pose: bool = False,
    metrics_csv: Path | None = None,
) -> tuple[int, float]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_video is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
    if metrics_csv is not None:
        metrics_csv.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exe,
        str(infer_script),
        "--model_path", str(model_path),
        "--yolo_weights", str(yolo_weights),
        "--source", str(source_path),
        "--output", str(output_csv),
        "--fps", str(float(fps)),
        "--window_stride", str(int(window_stride)),
        "--yolo_batch", str(int(yolo_batch)),
        "--pose_conf_threshold", str(float(pose_conf_threshold)),
        "--animal_scale_factor", str(float(animal_scale_factor)),
        "--group_scale_factor", str(float(group_scale_factor)),
        "--temporal_smoothing_window", str(int(temporal_smoothing_window)),
        "--bout_min_duration_frames", str(int(bout_min_duration_frames)),
        "--device", str(device),
        "--smooth_bouts",
    ]
    if model_config is not None:
        cmd += ["--model_config", str(model_config)]
    if calibration_json is not None:
        cmd += ["--calibration_json", str(calibration_json)]
    if output_video is not None:
        cmd += ["--output_video", str(output_video)]
    if amp:
        cmd += ["--amp"]
    if log_metrics:
        cmd += ["--log_metrics"]
    if metrics_csv is not None:
        cmd += ["--metrics_csv", str(metrics_csv)]
    if export_pose:
        cmd += ["--export_pose"]
    if print_each_window:
        cmd += ["--print_each_window"]

    print("[infer-cmd] " + " ".join(f'"{c}"' if " " in c else c for c in cmd), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return int(proc.returncode), time.time() - t0


def parse_annot(path: Path) -> tuple[dict, list[dict]]:
    import re

    text = path.read_text(encoding="utf-8", errors="replace")
    header = {"fps": 30.0, "stop_time": None}
    m = re.search(r"Annotation framerate:\s*([\d.]+)", text)
    if m:
        header["fps"] = float(m.group(1))
    m = re.search(r"Annotation stop time:\s*([\d.eE+-]+)", text)
    if m:
        header["stop_time"] = float(m.group(1))

    bouts: list[dict] = []
    blocks = re.split(r"\n(Ch\w+)-+\n", text)
    for i in range(1, len(blocks), 2):
        channel = blocks[i].strip()
        body = blocks[i + 1] if i + 1 < len(blocks) else ""
        for match in re.finditer(
            r">(\S+)\s*\nStart\s+Stop\s+Duration\s*\n((?:[\d.\s]+\n?)+)",
            body,
        ):
            behavior = match.group(1).strip()
            if behavior not in CLASS_TO_IDX:
                continue
            for row in match.group(2).strip().splitlines():
                parts = row.split()
                if len(parts) < 2:
                    continue
                try:
                    start_s = float(parts[0])
                    stop_s = float(parts[1])
                except ValueError:
                    continue
                if stop_s <= start_s:
                    continue
                bouts.append(
                    {
                        "channel": channel,
                        "behavior": behavior,
                        "start_s": start_s,
                        "stop_s": stop_s,
                    }
                )
    return header, bouts


def expand_gt_to_frames(bouts: Sequence[dict], n_frames: int, fps: float) -> np.ndarray:
    out = np.full(n_frames, CLASS_TO_IDX["other"], dtype=np.int16)
    # Write lower-priority classes first, higher-priority classes last.
    for behavior in ["other", "investigation", "mount", "attack"]:
        if behavior == "other":
            continue
        idx = CLASS_TO_IDX[behavior]
        for bout in bouts:
            if bout["behavior"] != behavior:
                continue
            start = max(0, int(round(float(bout["start_s"]) * fps)))
            end = min(n_frames, int(round(float(bout["stop_s"]) * fps)))
            if end > start:
                out[start:end] = idx
    return out


def load_raw_window_predictions(csv_path: Path, n_frames: int) -> np.ndarray:
    prob_cols = [f"prob_{c}" for c in CLASSES]
    prob_sum = np.zeros((n_frames, len(CLASSES)), dtype=np.float64)
    count = np.zeros(n_frames, dtype=np.int32)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    have_probs = bool(rows) and all(c in rows[0] for c in prob_cols)
    for row in rows:
        start = max(0, int(row["frame_start"]))
        end = min(n_frames, int(row["frame_end"]) + 1)
        if end <= start:
            continue
        if have_probs:
            probs = np.array([float(row[c]) for c in prob_cols], dtype=np.float64)
        else:
            probs = np.zeros(len(CLASSES), dtype=np.float64)
            probs[CLASS_TO_IDX[str(row["predicted_class"])]] = 1.0
        prob_sum[start:end] += probs
        count[start:end] += 1
    pred = np.full(n_frames, CLASS_TO_IDX["other"], dtype=np.int16)
    covered = count > 0
    pred[covered] = prob_sum[covered].argmax(axis=1).astype(np.int16)
    return pred


def load_smoothed_predictions(csv_path: Path, n_frames: int) -> np.ndarray:
    pred = np.full(n_frames, CLASS_TO_IDX["other"], dtype=np.int16)
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    frame_col = "frame_idx" if rows and "frame_idx" in rows[0] else "frame_start"
    for row in rows:
        idx = int(row[frame_col])
        if 0 <= idx < n_frames:
            if row.get("predicted_class_id", "") != "":
                pred[idx] = int(row["predicted_class_id"])
            else:
                pred[idx] = CLASS_TO_IDX[str(row["predicted_class"])]
    return pred


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def prf_from_cm(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision = np.zeros(len(CLASSES), dtype=np.float64)
    recall = np.zeros(len(CLASSES), dtype=np.float64)
    f1 = np.zeros(len(CLASSES), dtype=np.float64)
    for i in range(len(CLASSES)):
        tp = float(cm[i, i])
        fp = float(cm[:, i].sum() - cm[i, i])
        fn = float(cm[i, :].sum() - cm[i, i])
        precision[i] = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall[i] = tp / (tp + fn) if tp + fn > 0 else 0.0
        f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if precision[i] + recall[i] > 0 else 0.0
    return precision, recall, f1


def frames_to_bouts(frames: np.ndarray, include_other: bool = False) -> list[dict]:
    bouts: list[dict] = []
    if len(frames) == 0:
        return bouts
    i = 0
    while i < len(frames):
        cls_idx = int(frames[i])
        j = i + 1
        while j < len(frames) and int(frames[j]) == cls_idx:
            j += 1
        cls = IDX_TO_CLASS[cls_idx]
        if include_other or cls != "other":
            bouts.append({"class": cls, "class_id": cls_idx, "start_frame": i, "end_frame": j - 1})
        i = j
    return bouts


def tiou(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    inter = max(0, min(a_end, b_end) - max(a_start, b_start) + 1)
    union = max(a_end, b_end) - min(a_start, b_start) + 1
    return float(inter / union) if union > 0 else 0.0


def bout_metrics(gt_bouts: Sequence[dict], pred_bouts: Sequence[dict], threshold: float) -> tuple[dict, list[dict]]:
    rows = {
        cls: {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
        for cls in CLASSES
    }
    matches: list[dict] = []
    matched_gt: set[int] = set()
    for pred_i, pred in enumerate(pred_bouts):
        best_iou = -1.0
        best_gt = -1
        for gt_i, gt in enumerate(gt_bouts):
            if gt_i in matched_gt or gt["class"] != pred["class"]:
                continue
            score = tiou(pred["start_frame"], pred["end_frame"], gt["start_frame"], gt["end_frame"])
            if score > best_iou:
                best_iou = score
                best_gt = gt_i
        if best_gt >= 0 and best_iou >= threshold:
            rows[pred["class"]]["tp"] += 1
            matched_gt.add(best_gt)
            gt = gt_bouts[best_gt]
            matches.append(
                {
                    "threshold": threshold,
                    "class": pred["class"],
                    "pred_bout_index": pred_i,
                    "gt_bout_index": best_gt,
                    "tiou": best_iou,
                    "pred_start": pred["start_frame"],
                    "pred_end": pred["end_frame"],
                    "gt_start": gt["start_frame"],
                    "gt_end": gt["end_frame"],
                    "start_offset_frames": pred["start_frame"] - gt["start_frame"],
                    "end_offset_frames": pred["end_frame"] - gt["end_frame"],
                }
            )
        else:
            rows[pred["class"]]["fp"] += 1
    for gt_i, gt in enumerate(gt_bouts):
        if gt_i not in matched_gt:
            rows[gt["class"]]["fn"] += 1
    for cls, row in rows.items():
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        row["precision"] = tp / (tp + fp) if tp + fp > 0 else 0.0
        row["recall"] = tp / (tp + fn) if tp + fn > 0 else 0.0
        row["f1"] = (
            2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"])
            if row["precision"] + row["recall"] > 0
            else 0.0
        )
    return rows, matches


def evaluate_prediction_set(
    *,
    split: str,
    video_id: str,
    prediction_type: str,
    predictions_csv: Path,
    annot_path: Path,
    n_frames: int,
    fps: float,
) -> tuple[dict, list[dict], list[dict], list[dict], list[dict]]:
    header, annot_bouts = parse_annot(annot_path)
    annot_fps = float(header.get("fps") or fps)
    gt_frames = expand_gt_to_frames(annot_bouts, n_frames, annot_fps)
    if prediction_type == "raw_windows":
        pred_frames = load_raw_window_predictions(predictions_csv, n_frames)
    elif prediction_type == "smoothed_frames":
        pred_frames = load_smoothed_predictions(predictions_csv, n_frames)
    else:
        raise ValueError(f"unknown prediction_type={prediction_type}")

    cm = confusion_matrix(gt_frames, pred_frames)
    precision, recall, f1 = prf_from_cm(cm)
    gt_bouts = frames_to_bouts(gt_frames, include_other=False)
    pred_bouts = frames_to_bouts(pred_frames, include_other=False)

    bout_results = {
        threshold: bout_metrics(gt_bouts, pred_bouts, threshold)
        for threshold in IOU_THRESHOLDS
    }
    present_classes = [c for c in BEHAVIOR_CLASSES if any(b["class"] == c for b in gt_bouts)]
    present_idx = [CLASS_TO_IDX[c] for c in present_classes]
    frame_macro = float(np.mean([f1[i] for i in present_idx])) if present_idx else 0.0
    bout_macro = {
        threshold: (
            float(np.mean([bout_results[threshold][0][c]["f1"] for c in present_classes]))
            if present_classes else 0.0
        )
        for threshold in IOU_THRESHOLDS
    }
    accuracy = float((gt_frames == pred_frames).mean()) if n_frames else 0.0

    summary = {
        "split": split,
        "video_id": video_id,
        "prediction_type": prediction_type,
        "predictions_csv": str(predictions_csv),
        "annot_path": str(annot_path),
        "n_frames": n_frames,
        "fps": fps,
        "annot_fps": annot_fps,
        "accuracy": accuracy,
        "frame_macro_f1_present_behaviors": frame_macro,
        "bout_macro_f1_iou10_present_behaviors": bout_macro[0.10],
        "bout_macro_f1_iou25_present_behaviors": bout_macro[0.25],
        "bout_macro_f1_iou50_present_behaviors": bout_macro[0.50],
        "gt_bouts_total": len(gt_bouts),
        "pred_bouts_total": len(pred_bouts),
    }

    per_class_rows: list[dict] = []
    for cls_idx, cls in enumerate(CLASSES):
        per_class_rows.append(
            {
                "split": split,
                "video_id": video_id,
                "prediction_type": prediction_type,
                "class": cls,
                "gt_frames": int((gt_frames == cls_idx).sum()),
                "pred_frames": int((pred_frames == cls_idx).sum()),
                "frame_precision": precision[cls_idx],
                "frame_recall": recall[cls_idx],
                "frame_f1": f1[cls_idx],
                "bout_tp_iou10": bout_results[0.10][0][cls]["tp"],
                "bout_fp_iou10": bout_results[0.10][0][cls]["fp"],
                "bout_fn_iou10": bout_results[0.10][0][cls]["fn"],
                "bout_precision_iou10": bout_results[0.10][0][cls]["precision"],
                "bout_recall_iou10": bout_results[0.10][0][cls]["recall"],
                "bout_f1_iou10": bout_results[0.10][0][cls]["f1"],
                "bout_tp_iou25": bout_results[0.25][0][cls]["tp"],
                "bout_fp_iou25": bout_results[0.25][0][cls]["fp"],
                "bout_fn_iou25": bout_results[0.25][0][cls]["fn"],
                "bout_precision_iou25": bout_results[0.25][0][cls]["precision"],
                "bout_recall_iou25": bout_results[0.25][0][cls]["recall"],
                "bout_f1_iou25": bout_results[0.25][0][cls]["f1"],
                "bout_tp_iou50": bout_results[0.50][0][cls]["tp"],
                "bout_fp_iou50": bout_results[0.50][0][cls]["fp"],
                "bout_fn_iou50": bout_results[0.50][0][cls]["fn"],
                "bout_precision_iou50": bout_results[0.50][0][cls]["precision"],
                "bout_recall_iou50": bout_results[0.50][0][cls]["recall"],
                "bout_f1_iou50": bout_results[0.50][0][cls]["f1"],
                "cm_attack": int(cm[cls_idx, CLASS_TO_IDX["attack"]]),
                "cm_investigation": int(cm[cls_idx, CLASS_TO_IDX["investigation"]]),
                "cm_mount": int(cm[cls_idx, CLASS_TO_IDX["mount"]]),
                "cm_other": int(cm[cls_idx, CLASS_TO_IDX["other"]]),
            }
        )

    def _bout_rows(source: str, bouts: Sequence[dict]) -> list[dict]:
        return [
            {
                "split": split,
                "video_id": video_id,
                "prediction_type": prediction_type,
                "source": source,
                "class": b["class"],
                "class_id": b["class_id"],
                "start_frame": b["start_frame"],
                "end_frame": b["end_frame"],
                "duration_frames": b["end_frame"] - b["start_frame"] + 1,
            }
            for b in bouts
        ]

    match_rows = []
    for _threshold, (_metrics, matches) in bout_results.items():
        for row in matches:
            row = dict(row)
            row.update({"split": split, "video_id": video_id, "prediction_type": prediction_type})
            match_rows.append(row)

    return summary, per_class_rows, _bout_rows("ground_truth", gt_bouts), _bout_rows("prediction", pred_bouts), match_rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def resolve_cli_path(path_value: str | os.PathLike | None, *, base: Path = REPO_ROOT) -> Path | None:
    """Resolve user-supplied paths consistently from the repository root.

    The batch script may be launched from BehaviorScope-X, the repo root, or a
    GUI process. Relative paths in the public examples are repo-root relative,
    so normalize them here before creating outputs or spawning infer_x.py.
    """
    if path_value is None:
        return None
    path_text = str(path_value).strip()
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return base / path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch BehaviorScope-X inference + bout-level MARS evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mars_root", default=str(REPO_ROOT / "MARS_data"))
    p.add_argument("--splits", nargs="+", default=["test_1", "test_2"])
    p.add_argument("--model_path", default=str(REPO_ROOT / "BehaviorScope" / "behavior_lstm_runs" / "R5_yolo_attn_smoke_ep10" / "best_model_macro_f1.pt"))
    p.add_argument("--model_config", default=str(REPO_ROOT / "BehaviorScope" / "behavior_lstm_runs" / "R5_yolo_attn_smoke_ep10" / "config.json"))
    p.add_argument("--yolo_weights", default=str(REPO_ROOT / "mars_yolo_pose" / "runs" / "pose" / "train" / "weights" / "best.pt"))
    p.add_argument("--calibration_json", default=str(REPO_ROOT / "mars_behaviorscope_npz_full" / "calibration.json"))
    p.add_argument("--output_root", default=str(REPO_ROOT / "BehaviorScope" / "behavior_lstm_runs" / "R5_yolo_attn_smoke_ep10_mars_test_eval"))
    p.add_argument("--source_mode", choices=["mp4", "seq"], default="mp4", help="'mp4' converts .seq before inference; 'seq' passes .seq directly to infer_x.py.")
    p.add_argument("--source_mp4_cache", default=None,
                   help="Directory of pre-converted MP4s (layout: <cache>/<split>/<video_id>.mp4). "
                        "When set, the script copies from this cache instead of re-converting .seq files. "
                        "Falls back to normal .seq conversion if the cached MP4 is not found.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_existing", action="store_true", help="Skip inference when behavior CSV already exists; still evaluate existing outputs.")
    p.add_argument("--keep_intermediate_mp4", action="store_true")
    p.add_argument("--conversion_retries", type=int, default=3,
                   help="Retry .seq to .mp4 conversion this many times after the first failed attempt.")
    p.add_argument("--conversion_retry_delay_s", type=float, default=2.0,
                   help="Seconds to wait between .seq conversion retry attempts.")
    p.add_argument("--tolerate_bad_seq_frames", action=argparse.BooleanOptionalAction, default=False,
                   help="During .seq -> .mp4 conversion, replace unreadable frames with the previous good frame.")
    p.add_argument("--allow_pred_named_annots", action=argparse.BooleanOptionalAction, default=False,
                   help="Accept .annot files whose names contain pred/prediction. Use only when these are approved labels.")
    p.add_argument("--allow_mp4_only_discovery", action="store_true",
                   help="Allow benchmark discovery without .seq files when source_mode=mp4 and source_mp4_cache supplies valid MP4s.")
    p.add_argument("--max_videos", type=int, default=0)
    p.add_argument("--video_filter", default=None,
                   help="Substring filter on video_id; only process matching videos. "
                        "Useful for targeted smoke tests (e.g., --video_filter Mouse077).")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--window_stride", type=int, default=16)
    p.add_argument("--yolo_batch", type=int, default=25)
    p.add_argument("--pose_conf_threshold", type=float, default=0.3)
    p.add_argument("--animal_scale_factor", type=float, default=4.0)
    p.add_argument("--group_scale_factor", type=float, default=8.0)
    p.add_argument("--temporal_smoothing_window", type=int, default=5)
    p.add_argument("--bout_min_duration_frames", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--log_metrics", action="store_true")
    p.add_argument("--export_pose", action="store_true",
                   help="Export per-frame YOLO keypoints + bbox CSV alongside the behavior CSV.")
    p.add_argument("--no_videos", action="store_true", help="Do not write annotated MP4 outputs.")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--print_each_window", action="store_true")
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    mars_root = resolve_cli_path(args.mars_root)
    out_root = resolve_cli_path(args.output_root)
    infer_script = SCRIPTS_DIR / "infer_x.py"
    if not infer_script.is_file():
        legacy_infer_script = REPO_ROOT / "BehaviorScope-X" / "infer_x.py"
        if legacy_infer_script.is_file():
            infer_script = legacy_infer_script
    model_path = resolve_cli_path(args.model_path)
    model_config = resolve_cli_path(args.model_config) if args.model_config else None
    yolo_weights = resolve_cli_path(args.yolo_weights)
    calibration_json = resolve_cli_path(args.calibration_json) if args.calibration_json else None

    assert mars_root is not None
    assert out_root is not None
    assert model_path is not None
    assert yolo_weights is not None

    jobs = discover_jobs(
        mars_root,
        args.splits,
        allow_pred_named_annots=bool(args.allow_pred_named_annots),
        allow_mp4_only_discovery=bool(args.allow_mp4_only_discovery),
    )
    if args.video_filter:
        jobs = [j for j in jobs if args.video_filter in j.video_id]
        print(f"[batch] video_filter={args.video_filter!r} -> {len(jobs)} matched", flush=True)
    if args.max_videos and args.max_videos > 0:
        jobs = jobs[: int(args.max_videos)]

    print(f"[batch] discovered {len(jobs)} video jobs", flush=True)
    print(f"[batch] output_root={out_root}", flush=True)
    print(f"[batch] model_path={model_path}", flush=True)
    print(f"[batch] yolo_weights={yolo_weights}", flush=True)

    manifest_rows: list[dict] = []
    failure_rows: list[dict] = []
    summary_rows: list[dict] = []
    per_class_rows_all: list[dict] = []
    gt_bout_rows_all: list[dict] = []
    pred_bout_rows_all: list[dict] = []
    bout_match_rows_all: list[dict] = []

    batch_t0 = time.time()
    n_jobs = len(jobs)

    for job_i, job in enumerate(jobs, start=1):
        video_t0 = time.time()
        safe_video = _safe_name(job.video_id)
        video_out = out_root / job.split / safe_video
        converted_mp4 = video_out / "source_mp4" / f"{safe_video}.mp4"
        pred_csv = video_out / f"{safe_video}.behavior.csv"
        smoothed_csv = pred_csv.with_suffix(".smoothed_frames.csv")
        annotated_mp4 = None if args.no_videos else video_out / f"{safe_video}.annotated.mp4"
        metrics_csv = pred_csv.with_suffix(".metrics.csv")

        print(f"\n[batch] {job_i}/{n_jobs} {job.split}/{job.video_id}", flush=True)
        print(f"[batch] seq={job.seq_path or '<mp4-only discovery>'}", flush=True)
        print(f"[batch] annot={job.annot_path}", flush=True)

        source_for_infer = job.seq_path
        conversion_info = {}
        outputs_exist = pred_csv.is_file() and smoothed_csv.is_file()
        resume_without_infer = bool(args.skip_existing and outputs_exist and not args.overwrite)
        if args.source_mode == "mp4":
            try:
                source_for_infer = converted_mp4
                if converted_mp4.is_file() and not args.dry_run and not resume_without_infer:
                    probe = probe_video(converted_mp4)
                    if not probe["valid"]:
                        print(f"[cache] removing invalid existing mp4 {converted_mp4}", flush=True)
                        try:
                            converted_mp4.unlink()
                        except OSError:
                            pass

                # Check source_mp4_cache before converting from .seq.
                if (args.source_mp4_cache and not converted_mp4.is_file()
                        and not args.dry_run and not resume_without_infer):
                    _cache_dir = Path(args.source_mp4_cache)
                    _cached = _cache_dir / job.split / f"{safe_video}.mp4"
                    if _cached.is_file():
                        cached_probe = probe_video(_cached)
                        if cached_probe["valid"]:
                            converted_mp4.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(str(_cached), str(converted_mp4))
                            print(f"[cache] copied {_cached} -> {converted_mp4}", flush=True)
                        else:
                            print(f"[cache] ignoring invalid cached mp4 {_cached}", flush=True)
                if not args.dry_run and not resume_without_infer:
                    if job.seq_path is None:
                        probe = probe_video(converted_mp4)
                        if not probe["valid"]:
                            raise RuntimeError(
                                f"mp4-only discovery requires a valid cached MP4: {converted_mp4}"
                            )
                        conversion_info = {
                            "converted": False,
                            "mp4_path": str(converted_mp4),
                            "frames": int(probe["frames"]),
                            "width": int(probe["width"]),
                            "height": int(probe["height"]),
                            "conversion_attempts": 1,
                            "conversion_retries_requested": int(args.conversion_retries),
                        }
                    else:
                        conversion_info = convert_seq_to_mp4_with_retries(
                            job.seq_path,
                            converted_mp4,
                            seek_mat_path=job.seek_mat_path,
                            fps=float(args.fps),
                            overwrite=bool(args.overwrite),
                            retries=int(args.conversion_retries),
                            retry_delay_s=float(args.conversion_retry_delay_s),
                            tolerate_bad_frames=bool(args.tolerate_bad_seq_frames),
                        )
            except Exception as exc:
                failure_rows.append(
                    {
                        "split": job.split,
                        "video_id": job.video_id,
                        "stage": "convert_seq_to_mp4",
                        "error": repr(exc),
                    }
                )
                print(f"[batch] ERROR converting {job.video_id}: {exc}", flush=True)
                continue

        manifest_rows.append(
            {
                "split": job.split,
                "video_id": job.video_id,
                "seq_path": str(job.seq_path) if job.seq_path else "",
                "annot_path": str(job.annot_path),
                "seek_mat_path": str(job.seek_mat_path) if job.seek_mat_path else "",
                "source_for_infer": str(source_for_infer),
                "predictions_csv": str(pred_csv),
                "smoothed_csv": str(smoothed_csv),
                "annotated_mp4": str(annotated_mp4) if annotated_mp4 else "",
                **{f"conversion_{k}": v for k, v in conversion_info.items()},
            }
        )

        if args.dry_run:
            continue

        need_infer = bool(args.overwrite) or not outputs_exist
        if resume_without_infer:
            need_infer = False

        if need_infer:
            rc, infer_elapsed = run_inference(
                python_exe=str(args.python),
                infer_script=infer_script,
                model_path=model_path,
                model_config=model_config if model_config and model_config.is_file() else None,
                yolo_weights=yolo_weights,
                source_path=source_for_infer,
                output_csv=pred_csv,
                output_video=annotated_mp4,
                calibration_json=calibration_json if calibration_json and calibration_json.is_file() else None,
                fps=float(args.fps),
                window_stride=int(args.window_stride),
                yolo_batch=int(args.yolo_batch),
                pose_conf_threshold=float(args.pose_conf_threshold),
                animal_scale_factor=float(args.animal_scale_factor),
                group_scale_factor=float(args.group_scale_factor),
                temporal_smoothing_window=int(args.temporal_smoothing_window),
                bout_min_duration_frames=int(args.bout_min_duration_frames),
                device=str(args.device),
                amp=bool(args.amp),
                log_metrics=bool(args.log_metrics),
                print_each_window=bool(args.print_each_window),
                export_pose=bool(args.export_pose),
                metrics_csv=(pred_csv.with_name(pred_csv.stem + "_metrics.csv")
                             if args.log_metrics else None),
            )
            if rc != 0:
                failure_rows.append(
                    {
                        "split": job.split,
                        "video_id": job.video_id,
                        "stage": "infer_x",
                        "returncode": rc,
                    }
                )
                print(f"[batch] ERROR inference failed for {job.video_id}: rc={rc}", flush=True)
                continue
            print(f"[batch] inference elapsed_s={infer_elapsed:.1f}", flush=True)
        else:
            print("[batch] using existing inference outputs", flush=True)

        # Determine evaluation frame count from the best available prediction file.
        try:
            if smoothed_csv.is_file():
                with smoothed_csv.open("r", newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                frame_col = "frame_idx" if rows and "frame_idx" in rows[0] else "frame_start"
                n_frames = max((int(r[frame_col]) for r in rows), default=-1) + 1
            else:
                with pred_csv.open("r", newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                n_frames = max((int(r["frame_end"]) for r in rows), default=-1) + 1
            if n_frames <= 0:
                raise ValueError("could not determine n_frames from prediction CSVs")

            for prediction_type, csv_path in [
                ("raw_windows", pred_csv),
                ("smoothed_frames", smoothed_csv),
            ]:
                if not csv_path.is_file():
                    continue
                summary, per_class, gt_bouts, pred_bouts, matches = evaluate_prediction_set(
                    split=job.split,
                    video_id=job.video_id,
                    prediction_type=prediction_type,
                    predictions_csv=csv_path,
                    annot_path=job.annot_path,
                    n_frames=n_frames,
                    fps=float(args.fps),
                )
                summary_rows.append(summary)
                per_class_rows_all.extend(per_class)
                if prediction_type == "smoothed_frames":
                    # GT bouts are identical for raw/smoothed; write once.
                    gt_bout_rows_all.extend(gt_bouts)
                pred_bout_rows_all.extend(pred_bouts)
                bout_match_rows_all.extend(matches)
                print(
                    f"[eval] {prediction_type}: frame_macro={summary['frame_macro_f1_present_behaviors']:.3f} "
                    f"bout25={summary['bout_macro_f1_iou25_present_behaviors']:.3f} "
                    f"bout50={summary['bout_macro_f1_iou50_present_behaviors']:.3f}",
                    flush=True,
                )
        except Exception as exc:
            failure_rows.append(
                {
                    "split": job.split,
                    "video_id": job.video_id,
                    "stage": "evaluate",
                    "error": repr(exc),
                }
            )
            print(f"[batch] ERROR evaluating {job.video_id}: {exc}", flush=True)
            continue

        if args.source_mode == "mp4" and not args.keep_intermediate_mp4 and converted_mp4.is_file():
            try:
                converted_mp4.unlink()
            except Exception:
                pass

        write_csv(out_root / "manifest.csv", manifest_rows)
        write_csv(out_root / "failures.csv", failure_rows)
        write_csv(out_root / "per_video_summary.csv", summary_rows)
        write_csv(out_root / "per_class_metrics.csv", per_class_rows_all)
        write_csv(out_root / "ground_truth_bouts.csv", gt_bout_rows_all)
        write_csv(out_root / "predicted_bouts.csv", pred_bout_rows_all)
        write_csv(out_root / "bout_matches.csv", bout_match_rows_all)

        # ---- Progress / ETA ----
        video_elapsed = time.time() - video_t0
        batch_elapsed = time.time() - batch_t0
        avg_per_video = batch_elapsed / job_i
        remaining = avg_per_video * (n_jobs - job_i)
        eta_min, eta_sec = divmod(int(remaining), 60)
        el_min, el_sec = divmod(int(batch_elapsed), 60)
        print(
            f"[progress] {job_i}/{n_jobs} done  "
            f"this={video_elapsed:.0f}s  "
            f"elapsed={el_min}m{el_sec:02d}s  "
            f"avg={avg_per_video:.0f}s/video  "
            f"remainingâ‰ˆ{eta_min}m{eta_sec:02d}s",
            flush=True,
        )

    write_csv(out_root / "manifest.csv", manifest_rows)
    write_csv(out_root / "failures.csv", failure_rows)
    write_csv(out_root / "per_video_summary.csv", summary_rows)
    write_csv(out_root / "per_class_metrics.csv", per_class_rows_all)
    write_csv(out_root / "ground_truth_bouts.csv", gt_bout_rows_all)
    write_csv(out_root / "predicted_bouts.csv", pred_bout_rows_all)
    write_csv(out_root / "bout_matches.csv", bout_match_rows_all)

    aggregate = {
        "n_jobs_discovered": len(jobs),
        "n_failures": len(failure_rows),
        "outputs": {
            "manifest_csv": str(out_root / "manifest.csv"),
            "failures_csv": str(out_root / "failures.csv"),
            "per_video_summary_csv": str(out_root / "per_video_summary.csv"),
            "per_class_metrics_csv": str(out_root / "per_class_metrics.csv"),
            "ground_truth_bouts_csv": str(out_root / "ground_truth_bouts.csv"),
            "predicted_bouts_csv": str(out_root / "predicted_bouts.csv"),
            "bout_matches_csv": str(out_root / "bout_matches.csv"),
        },
    }
    (out_root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(f"\n[batch] done. failures={len(failure_rows)} output_root={out_root}", flush=True)
    return 1 if failure_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())




