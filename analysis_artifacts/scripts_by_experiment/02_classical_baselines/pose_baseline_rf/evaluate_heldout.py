#!/usr/bin/env python3
"""Evaluate trained RF/XGBoost baselines on held-out MARS test videos.

Reconstructs per-video continuous frame predictions from overlapping NPZ
windows, applies the same post-processing as BehaviorScope-Y (median filter +
minimum bout duration), and computes identical frame-level and bout-level
metrics using the same evaluation functions.

Usage:
    cd pose_baseline_rf
    python evaluate_heldout.py
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
import time
import warnings
from collections import defaultdict

# Suppress sklearn parallel deprecation warnings (cosmetic, doesn't affect results)
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="sklearn")
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import median_filter

THIS_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((THIS_DIR / "config.yaml").read_text(encoding="utf-8"))

# Import BehaviorScope-Y evaluation functions
SHARED_SCRIPTS = Path(CONFIG["shared_scripts"])
if str(SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS))

from batch_infer_eval import (  # noqa: E402
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    BEHAVIOR_CLASSES,
    parse_annot,
    expand_gt_to_frames,
    confusion_matrix,
    prf_from_cm,
    frames_to_bouts,
    bout_metrics,
    find_annot,
    write_csv,
)
from utils.pose_features import extract_rich_pose_features  # noqa: E402

FEATURE_DIR = Path(CONFIG["output_root"]) / "features"
MODEL_DIR = Path(CONFIG["output_root"]) / "models"
PRED_DIR = Path(CONFIG["output_root"]) / "predictions"
RESULT_DIR = Path(CONFIG["output_root"]) / "results"
PRED_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

FPS = CONFIG["evaluation"]["fps"]
SMOOTH_WINDOW = CONFIG["evaluation"]["temporal_smoothing_window"]
MIN_BOUT_FRAMES = CONFIG["evaluation"]["bout_min_duration_frames"]
MARS_ROOT = Path(CONFIG["mars_data_root"])

# Map heldout manifest split names back to MARS split names
# The heldout NPZ builder used train_splits=['test_1'], val_splits=['test_2']
HELDOUT_SPLIT_MAP = {"train": "test_1", "val": "test_2"}


# ─── Temporal context (must match training) ─────────────────────────────────

def add_temporal_context_single_window(X: np.ndarray, half_window: int) -> np.ndarray:
    """Add temporal context to a single NPZ window (no cross-window boundaries)."""
    if half_window <= 0:
        return X
    T, D = X.shape
    K = int(half_window)
    width = 2 * K + 1
    X_aug = np.zeros((T, D * width), dtype=X.dtype)
    for t in range(T):
        for offset_idx, offset in enumerate(range(-K, K + 1)):
            neighbor = t + offset
            if 0 <= neighbor < T:
                X_aug[t, offset_idx * D:(offset_idx + 1) * D] = X[neighbor]
    return X_aug


# ─── Feature extraction (duplicated from extract_features.py for standalone use) ─

def extract_frame_features(npz_path: str | Path) -> np.ndarray:
    """Extract [T, 132] flat feature matrix from one NPZ window."""
    d = np.load(str(npz_path))
    T = int(d["animal_keypoints"].shape[0])
    N = int(d["n_animals"])

    # Pose-self per animal
    kpts = d["animal_keypoints"]
    pose_feats = []
    for a in range(N):
        pf = extract_rich_pose_features(kpts[:, a, :, :])
        pose_feats.append(pf)
    pose_self = np.concatenate(pose_feats, axis=1)

    # Relational features
    rel = d["relation_features"]
    relational = np.concatenate([rel[:, 0, 1, :], rel[:, 1, 0, :]], axis=1)

    # Reliability scalars
    reliability = np.stack([
        d["pose_conf"][:, 0], d["pose_conf"][:, 1],
        d["crop_conf"][:, 0], d["crop_conf"][:, 1],
        d["track_conf"][:, 0], d["track_conf"][:, 1],
        d["track_age"][:, 0], d["track_age"][:, 1],
    ], axis=1)

    # Masks
    masks = np.stack([
        d["animal_mask"][:, 0].astype(np.float32),
        d["animal_mask"][:, 1].astype(np.float32),
        d["pose_mask"][:, 0].astype(np.float32),
        d["pose_mask"][:, 1].astype(np.float32),
    ], axis=1)

    features = np.concatenate([pose_self, relational, reliability, masks], axis=1)
    return np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


# ─── Post-processing (matches BehaviorScope-Y infer_y.py) ──────────────────

def smooth_predictions(pred_frames: np.ndarray) -> np.ndarray:
    """Apply median filter + minimum bout duration, matching BehaviorScope-Y."""
    smoothed = pred_frames.copy()

    # Step 1: Median filter on class labels
    if SMOOTH_WINDOW > 0:
        smoothed = median_filter(smoothed, size=SMOOTH_WINDOW, mode="nearest")
        smoothed = smoothed.astype(np.int16)

    # Step 2: Remove short bouts by merging into neighbors
    if MIN_BOUT_FRAMES > 0:
        i = 0
        while i < len(smoothed):
            cls = int(smoothed[i])
            j = i + 1
            while j < len(smoothed) and int(smoothed[j]) == cls:
                j += 1
            bout_len = j - i
            if bout_len < MIN_BOUT_FRAMES:
                # Replace with the label of the longer adjacent run
                left_cls = int(smoothed[i - 1]) if i > 0 else cls
                right_cls = int(smoothed[j]) if j < len(smoothed) else cls
                # Find left run length
                li = i - 1
                while li >= 0 and int(smoothed[li]) == left_cls:
                    li -= 1
                left_len = i - li - 1
                # Find right run length
                rj = j
                while rj < len(smoothed) and int(smoothed[rj]) == right_cls:
                    rj += 1
                right_len = rj - j
                merge_cls = left_cls if left_len >= right_len else right_cls
                smoothed[i:j] = merge_cls
            i = j

    return smoothed


# ─── Per-video evaluation ───────────────────────────────────────────────────

def evaluate_video(
    video_id: str,
    mars_split: str,
    pred_frames_raw: np.ndarray,
    pred_frames_smoothed: np.ndarray,
    n_frames: int,
) -> tuple[list[dict], list[dict]]:
    """Evaluate a single video against its .annot ground truth.

    Returns (summary_rows, per_class_rows) for both raw and smoothed.
    """
    # Find the .annot file
    video_dir = MARS_ROOT / mars_split / video_id
    if not video_dir.is_dir():
        # Try to find by partial match (some video_ids have # replaced)
        candidates = list((MARS_ROOT / mars_split).glob(
            video_id.replace("_", "*").replace("#", "*") + "*"
        ))
        if candidates:
            video_dir = candidates[0]
        else:
            print(f"    WARNING: cannot find video dir for {mars_split}/{video_id}")
            return [], []

    # Try strict matching first (no pred-named files), then fallback to all.
    # Some MARS test_2 videos (Mouse457, Mouse518) have ground-truth annots
    # with "_pred_" in the filename (corrected prediction files used as GT).
    annot_path = find_annot(video_dir, allow_pred_named=False)
    if annot_path is None:
        annot_path = find_annot(video_dir, allow_pred_named=True)
    if annot_path is None:
        print(f"    WARNING: no .annot for {video_id}")
        return [], []

    header, annot_bouts = parse_annot(annot_path)
    annot_fps = float(header.get("fps") or FPS)
    gt_frames = expand_gt_to_frames(annot_bouts, n_frames, annot_fps)

    summary_rows = []
    per_class_rows = []

    for pred_type, pred in [("raw", pred_frames_raw), ("smoothed", pred_frames_smoothed)]:
        # Trim to same length
        min_len = min(len(gt_frames), len(pred))
        gt = gt_frames[:min_len]
        pr = pred[:min_len]

        cm = confusion_matrix(gt, pr)
        precision, recall, f1 = prf_from_cm(cm)
        gt_bouts = frames_to_bouts(gt, include_other=False)
        pred_bouts = frames_to_bouts(pr, include_other=False)

        metrics25, _ = bout_metrics(gt_bouts, pred_bouts, 0.25)
        metrics50, _ = bout_metrics(gt_bouts, pred_bouts, 0.50)

        present = [c for c in BEHAVIOR_CLASSES if any(b["class"] == c for b in gt_bouts)]
        present_idx = [CLASS_TO_IDX[c] for c in present]
        frame_macro = float(np.mean([f1[i] for i in present_idx])) if present_idx else 0.0
        bout_macro25 = float(np.mean([metrics25[c]["f1"] for c in present])) if present else 0.0
        bout_macro50 = float(np.mean([metrics50[c]["f1"] for c in present])) if present else 0.0
        accuracy = float(np.mean(gt == pr))

        summary_rows.append({
            "mars_split": mars_split,
            "video_id": video_id,
            "prediction_type": pred_type,
            "n_frames": int(min_len),
            "accuracy": round(accuracy, 6),
            "frame_macro_f1": round(frame_macro, 6),
            "bout_macro_f1_iou25": round(bout_macro25, 6),
            "bout_macro_f1_iou50": round(bout_macro50, 6),
            "gt_bouts": len(gt_bouts),
            "pred_bouts": len(pred_bouts),
        })

        for cls_idx, cls in enumerate(CLASSES):
            per_class_rows.append({
                "mars_split": mars_split,
                "video_id": video_id,
                "prediction_type": pred_type,
                "class": cls,
                "gt_frames": int(np.sum(gt == cls_idx)),
                "pred_frames": int(np.sum(pr == cls_idx)),
                "frame_precision": round(float(precision[cls_idx]), 6),
                "frame_recall": round(float(recall[cls_idx]), 6),
                "frame_f1": round(float(f1[cls_idx]), 6),
                "bout_f1_iou25": round(metrics25[cls]["f1"], 6),
                "bout_f1_iou50": round(metrics50[cls]["f1"], 6),
            })

    return summary_rows, per_class_rows


# ─── Reconstruct per-video predictions from overlapping windows ─────────────

def reconstruct_per_video_predictions(
    clf,
    manifest_path: Path,
    half_window: int,
) -> dict:
    """Run classifier on heldout NPZ windows and reconstruct per-video frames.

    For overlapping windows, average class probabilities across all windows
    that cover each frame, then take argmax — same as the LSTM pipeline.

    Returns: dict[video_id] -> {
        "mars_split": str,
        "n_frames": int,
        "pred_raw": np.ndarray,       # [n_frames] int16
        "pred_smoothed": np.ndarray,   # [n_frames] int16
    }
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_dir = manifest_path.parent

    # Group windows by source video
    video_windows: dict[str, list] = defaultdict(list)
    for split_name, samples in manifest["splits"].items():
        mars_split = HELDOUT_SPLIT_MAP.get(split_name, split_name)
        for sample in samples:
            vid = sample["source_video"]
            video_windows[vid].append({
                "sample": sample,
                "mars_split": mars_split,
                "npz_path": manifest_dir / sample["sequence_npz"],
            })

    # Process each video
    results = {}
    has_predict_proba = hasattr(clf, "predict_proba")

    for vid, windows in sorted(video_windows.items()):
        # Determine total frame span
        max_frame = max(w["sample"]["end_frame"] for w in windows) + 1
        mars_split = windows[0]["mars_split"]

        if has_predict_proba:
            prob_sum = np.zeros((max_frame, len(CLASSES)), dtype=np.float64)
            count = np.zeros(max_frame, dtype=np.int32)
        else:
            vote_count = np.zeros((max_frame, len(CLASSES)), dtype=np.int32)

        for w in windows:
            npz_path = w["npz_path"]
            if not npz_path.is_file():
                continue

            features = extract_frame_features(npz_path)  # [T, 132]
            if half_window > 0:
                features = add_temporal_context_single_window(features, half_window)
            T = features.shape[0]

            start = int(w["sample"]["start_frame"])

            if has_predict_proba:
                probs = clf.predict_proba(features)  # [T, n_classes]
                for t in range(T):
                    idx = start + t
                    if idx < max_frame:
                        prob_sum[idx] += probs[t]
                        count[idx] += 1
            else:
                preds = clf.predict(features)
                for t in range(T):
                    idx = start + t
                    if idx < max_frame:
                        vote_count[idx, int(preds[t])] += 1

        # Resolve predictions
        pred_raw = np.full(max_frame, CLASS_TO_IDX["other"], dtype=np.int16)
        if has_predict_proba:
            covered = count > 0
            pred_raw[covered] = prob_sum[covered].argmax(axis=1).astype(np.int16)
        else:
            covered = vote_count.sum(axis=1) > 0
            pred_raw[covered] = vote_count[covered].argmax(axis=1).astype(np.int16)

        pred_smoothed = smooth_predictions(pred_raw)

        results[vid] = {
            "mars_split": mars_split,
            "n_frames": max_frame,
            "pred_raw": pred_raw,
            "pred_smoothed": pred_smoothed,
        }

    return results


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("pose_baseline_rf — Held-out Evaluation")
    print("=" * 70)

    heldout_manifest = Path(CONFIG["npz_cache"]["heldout_manifest"])
    half_windows = CONFIG["temporal_context"]["half_windows"]

    # Discover all model files
    model_files = sorted(MODEL_DIR.glob("*.pkl"))
    if not model_files:
        print("ERROR: no trained models found in", MODEL_DIR)
        return

    print(f"\nFound {len(model_files)} trained models:")
    for mf in model_files:
        print(f"  {mf.name}")

    all_model_results = []

    for model_path in model_files:
        model_name = model_path.stem  # e.g., "rf_perframe", "xgb_context15"
        # Determine half_window from model name
        if "perframe" in model_name:
            K = 0
        else:
            # Parse from "context15" → K=7
            import re
            m = re.search(r"context(\d+)", model_name)
            K = (int(m.group(1)) - 1) // 2 if m else 0

        print(f"\n{'=' * 60}")
        print(f"Evaluating: {model_name}  (K={K})")
        print(f"{'=' * 60}")

        with open(model_path, "rb") as f:
            clf = pickle.load(f)

        t0 = time.time()
        video_results = reconstruct_per_video_predictions(clf, heldout_manifest, K)
        infer_time = time.time() - t0
        print(f"  Inference: {len(video_results)} videos in {infer_time:.1f}s")

        # Evaluate each video
        all_summary = []
        all_per_class = []

        for vid, vdata in sorted(video_results.items()):
            summaries, per_class = evaluate_video(
                vid, vdata["mars_split"],
                vdata["pred_raw"], vdata["pred_smoothed"],
                vdata["n_frames"],
            )
            all_summary.extend(summaries)
            all_per_class.extend(per_class)

        # Write per-video CSVs
        model_pred_dir = PRED_DIR / model_name
        model_pred_dir.mkdir(parents=True, exist_ok=True)
        write_csv(model_pred_dir / "per_video_summary.csv", all_summary)
        write_csv(model_pred_dir / "per_class_metrics.csv", all_per_class)

        # Compute aggregate metrics
        for pred_type in ["raw", "smoothed"]:
            rows = [r for r in all_summary if r["prediction_type"] == pred_type]
            if not rows:
                continue
            mean_frame_f1 = np.mean([r["frame_macro_f1"] for r in rows])
            mean_bout_f1_25 = np.mean([r["bout_macro_f1_iou25"] for r in rows])
            mean_bout_f1_50 = np.mean([r["bout_macro_f1_iou50"] for r in rows])
            mean_acc = np.mean([r["accuracy"] for r in rows])
            n_videos = len(rows)

            agg = {
                "model": model_name,
                "prediction_type": pred_type,
                "n_videos": n_videos,
                "mean_frame_macro_f1": round(float(mean_frame_f1), 4),
                "mean_bout_macro_f1_iou25": round(float(mean_bout_f1_25), 4),
                "mean_bout_macro_f1_iou50": round(float(mean_bout_f1_50), 4),
                "mean_accuracy": round(float(mean_acc), 4),
                "inference_time_s": round(infer_time, 1),
            }
            all_model_results.append(agg)

            prefix = "  [smoothed]" if pred_type == "smoothed" else "  [raw]     "
            print(f"{prefix} frame-F1={mean_frame_f1:.4f}  "
                  f"bout-F1@.25={mean_bout_f1_25:.4f}  "
                  f"bout-F1@.50={mean_bout_f1_50:.4f}  "
                  f"acc={mean_acc:.4f}  ({n_videos} videos)")

    # ── Final comparison table ──────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("HELD-OUT RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<22} {'Type':<10} {'Frame-F1':>9} {'Bout@.25':>9} {'Bout@.50':>9} {'Acc':>7}")
    print("-" * 70)
    for r in all_model_results:
        print(f"{r['model']:<22} {r['prediction_type']:<10} "
              f"{r['mean_frame_macro_f1']:>9.4f} "
              f"{r['mean_bout_macro_f1_iou25']:>9.4f} "
              f"{r['mean_bout_macro_f1_iou50']:>9.4f} "
              f"{r['mean_accuracy']:>7.4f}")

    # Save aggregate results
    agg_path = RESULT_DIR / "heldout_aggregate_results.json"
    agg_path.write_text(json.dumps(all_model_results, indent=2), encoding="utf-8")
    print(f"\n  -> {agg_path}")

    # Also write as CSV for easy comparison
    csv_path = RESULT_DIR / "heldout_aggregate_results.csv"
    write_csv(csv_path, all_model_results)
    print(f"  -> {csv_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
