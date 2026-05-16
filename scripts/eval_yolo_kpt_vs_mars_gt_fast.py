#!/usr/bin/env python3
"""
eval_yolo_kpt_vs_mars_gt_fast.py
=================================
FAST keypoint accuracy comparison: YOLO-predicted keypoints (already stored in
the manuscript NPZ build) vs MARS ground-truth keypoints (already stored in the
original NPZ build).

Both NPZ sets store ``animal_keypoints_raw`` in the SAME pixel coordinate space
(original frame), so we just need to match windows by (source_clip, start_frame)
and compare arrays directly. No YOLO inference, no video decoding -- pure numpy.

Metrics computed (same as eval_yolo_kpt_vs_mars_gt.py):
  - RMSE (pixels)
  - PCK @ body-fraction thresholds 0.05 / 0.10 / 0.20
  - OKS (Object Keypoint Similarity, MARS-tuned per-keypoint sigmas)

Usage (from project root):
    python scripts/eval_yolo_kpt_vs_mars_gt_fast.py ^
        --gt_manifest    mars_behaviorscope_npz_full\\sequence_manifest.json ^
        --pred_manifest  mars_behaviorscope_npz_yolo_kp\\sequence_manifest.json ^
        --output_csv     scripts\\kpt_eval_fast_results.csv ^
        --splits         train val

Output:
  - {output_csv}             per-keypoint and aggregate metrics table (CSV)
  - {output_csv}_raw.csv     one row per window (for debugging)
  - {output_csv}_summary.json  machine-readable summary
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Parse clip filename "..._f009712-010410.mp4" -> 9712 (the clip's source-video
# offset). Used to normalise start_frame between the two manifests:
#   GT manifest stores start_frame in source-video coords (e.g. 9712, 9728, ...)
#   Pred manifest stores start_frame in clip-relative coords (e.g. 0, 16, 32, ...)
_CLIP_FRAME_RANGE_RE = re.compile(r"_f(\d+)-(\d+)(?:\.[^.]+)?$")


def _clip_source_offset(clip_name: str) -> int:
    """Return the source-video frame index that corresponds to frame 0 of the clip,
    parsed from the clip filename pattern ``_f{start}-{end}``. Returns 0 if not found."""
    m = _CLIP_FRAME_RANGE_RE.search(clip_name)
    return int(m.group(1)) if m else 0

# ---------------------------------------------------------------------------
# MARS keypoint schema
# ---------------------------------------------------------------------------
KPT_NAMES = ["nose", "ear_1", "ear_2", "neck", "hip_1", "hip_2", "tail_base"]
K = len(KPT_NAMES)
NECK_IDX = 3
TAIL_IDX = 6
_OKS_SIGMAS = np.array([0.026, 0.025, 0.025, 0.035, 0.035, 0.035, 0.040], dtype=np.float32)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _body_length(kxy: np.ndarray) -> float:
    neck = kxy[NECK_IDX]
    tail = kxy[TAIL_IDX]
    if np.any(~np.isfinite(neck)) or np.any(~np.isfinite(tail)):
        return float("nan")
    return float(np.linalg.norm(neck - tail))


def _oks(gt_xy: np.ndarray, pred_xy: np.ndarray,
         gt_vis: np.ndarray, body_len: float) -> float:
    if body_len <= 0 or not np.isfinite(body_len):
        return float("nan")
    s2 = body_len ** 2
    sigs = _OKS_SIGMAS * 2.0
    d2 = np.sum((gt_xy - pred_xy) ** 2, axis=1)
    e = d2 / (2.0 * s2 * (sigs ** 2))
    visible = gt_vis & np.all(np.isfinite(pred_xy), axis=1)
    if visible.sum() == 0:
        return float("nan")
    return float(np.exp(-e)[visible].mean())


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _match_animals(gt_bbox_t: np.ndarray, pred_bbox_t: np.ndarray,
                   min_iou: float = 0.3) -> List[Tuple[int, int]]:
    """Greedy IoU matching of GT animal slots to predicted animal slots
    for a single frame. Returns list of (gt_idx, pred_idx)."""
    pairs: List[Tuple[int, int]] = []
    used: set[int] = set()
    n_gt = gt_bbox_t.shape[0]
    n_pred = pred_bbox_t.shape[0]
    for gi in range(n_gt):
        if not np.all(np.isfinite(gt_bbox_t[gi])):
            continue
        best_iou = min_iou
        best_pi = -1
        for pi in range(n_pred):
            if pi in used or not np.all(np.isfinite(pred_bbox_t[pi])):
                continue
            iou = _iou(gt_bbox_t[gi], pred_bbox_t[pi])
            if iou > best_iou:
                best_iou = iou
                best_pi = pi
        if best_pi >= 0:
            pairs.append((gi, best_pi))
            used.add(best_pi)
    return pairs


# ---------------------------------------------------------------------------
# manifest indexing
# ---------------------------------------------------------------------------
def _load_manifest(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _index_by_key(samples: List[dict], coord_mode: str) -> Dict[Tuple[str, int], dict]:
    """Index samples by (clip stem, clip-relative start_frame).

    coord_mode:
      "source_video": start_frame is in source-video coords; subtract the clip's
                      source-video offset (parsed from filename) to get clip-rel.
      "clip_relative": start_frame is already clip-relative; use as-is.

    Stems are normalised by stripping the file extension and replacing '#' -> '_'
    to bridge `_safe_stem` differences between the two builds.
    """
    out: Dict[Tuple[str, int], dict] = {}
    for s in samples:
        # Find a clip identifier across both manifest schemas.
        clip_ref = (
            s.get("source_clip")
            or s.get("clip_path")
            or s.get("clip_stem")
            or s.get("source_video", "")
        )
        if not clip_ref:
            continue
        clip_name = Path(clip_ref).name  # may include .mp4
        stem = Path(clip_ref).stem.replace("#", "_")
        sf = int(s.get("start_frame", 0) or 0)
        if coord_mode == "source_video":
            sf = sf - _clip_source_offset(clip_name)
        out[(stem, sf)] = s
    return out


# ---------------------------------------------------------------------------
# per-window comparison
# ---------------------------------------------------------------------------
def _compare_window(gt_npz: Path, pred_npz: Path) -> List[dict]:
    """Compare two NPZ windows that should cover the same frames of the same clip.
    Returns one row per (frame, gt_animal) with metric values."""
    gt = np.load(str(gt_npz), allow_pickle=False)
    pr = np.load(str(pred_npz), allow_pickle=False)

    if "animal_keypoints_raw" not in gt.files or "animal_keypoints_raw" not in pr.files:
        return []

    gt_kxy = gt["animal_keypoints_raw"].astype(np.float32)   # [T, N_gt, K, 2]
    gt_kc  = gt["animal_keypoints_conf"].astype(np.float32)  # [T, N_gt, K]
    pr_kxy = pr["animal_keypoints_raw"].astype(np.float32)   # [T, N_pr, K, 2]
    pr_kc  = pr["animal_keypoints_conf"].astype(np.float32)  # [T, N_pr, K]

    # bboxes for animal-slot matching
    gt_bb = gt["bbox_xyxy_animals"].astype(np.float32) if "bbox_xyxy_animals" in gt.files else None
    pr_bb = pr["bbox_xyxy_animals"].astype(np.float32) if "bbox_xyxy_animals" in pr.files else None

    T_gt = gt_kxy.shape[0]
    T_pr = pr_kxy.shape[0]
    T = min(T_gt, T_pr)
    N_gt = gt_kxy.shape[1]
    N_pr = pr_kxy.shape[1]
    K_gt = gt_kxy.shape[2]
    K_pr = pr_kxy.shape[2]
    K_use = min(K_gt, K_pr, K)

    rows: List[dict] = []
    for t in range(T):
        # Match animal slots by bbox IoU (handles tracker swaps between GT and YOLO)
        if gt_bb is not None and pr_bb is not None:
            pairs = _match_animals(gt_bb[t], pr_bb[t])
        else:
            # fall back to identity slot mapping
            pairs = [(i, i) for i in range(min(N_gt, N_pr))]

        matched_gt = {gi for gi, _ in pairs}

        # body length per frame from GT (more reliable than YOLO predictions)
        bls = [_body_length(gt_kxy[t, n]) for n in range(N_gt)]
        body_len = float(np.nanmedian(bls)) if any(np.isfinite(b) for b in bls) else float("nan")

        for gi, pi in pairs:
            row: dict = {
                "frame_in_window": t,
                "gt_animal": gi,
                "matched": True,
                "body_len_px": body_len,
            }
            gt_kxy_a = gt_kxy[t, gi, :K_use]
            gt_kc_a  = gt_kc[t, gi, :K_use]
            pr_kxy_a = pr_kxy[t, pi, :K_use]
            gt_vis = (gt_kc_a > 0.2) & np.all(np.isfinite(gt_kxy_a), axis=1)

            for k in range(K):
                if k >= K_use or not gt_vis[k] or not np.all(np.isfinite(pr_kxy_a[k])):
                    row[f"rmse_kpt{k}"]  = float("nan")
                    row[f"pck05_kpt{k}"] = float("nan")
                    row[f"pck10_kpt{k}"] = float("nan")
                    row[f"pck20_kpt{k}"] = float("nan")
                else:
                    dist = float(np.linalg.norm(gt_kxy_a[k] - pr_kxy_a[k]))
                    row[f"rmse_kpt{k}"] = dist
                    if np.isfinite(body_len) and body_len > 0:
                        d_norm = dist / body_len
                        row[f"pck05_kpt{k}"] = 1.0 if d_norm <= 0.05 else 0.0
                        row[f"pck10_kpt{k}"] = 1.0 if d_norm <= 0.10 else 0.0
                        row[f"pck20_kpt{k}"] = 1.0 if d_norm <= 0.20 else 0.0
                    else:
                        row[f"pck05_kpt{k}"] = float("nan")
                        row[f"pck10_kpt{k}"] = float("nan")
                        row[f"pck20_kpt{k}"] = float("nan")

            row["oks"] = _oks(gt_kxy_a, pr_kxy_a, gt_vis, body_len)
            rows.append(row)

        # Unmatched GT animals → miss
        for n in range(N_gt):
            if n in matched_gt:
                continue
            rows.append({
                "frame_in_window": t,
                "gt_animal": n,
                "matched": False,
                "body_len_px": body_len,
                **{f"rmse_kpt{k}": float("nan") for k in range(K)},
                **{f"pck05_kpt{k}": float("nan") for k in range(K)},
                **{f"pck10_kpt{k}": float("nan") for k in range(K)},
                **{f"pck20_kpt{k}": float("nan") for k in range(K)},
                "oks": float("nan"),
            })

    return rows


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
def _aggregate(raw_rows: List[dict]) -> dict:
    matched = [r for r in raw_rows if r["matched"]]
    total = len(raw_rows)
    det_rate = len(matched) / total if total > 0 else float("nan")
    mean_oks = float(np.nanmean([r["oks"] for r in matched])) if matched else float("nan")

    per_kpt: List[dict] = []
    for k, name in enumerate(KPT_NAMES):
        rmse_vals  = [r[f"rmse_kpt{k}"]  for r in matched]
        pck05_vals = [r[f"pck05_kpt{k}"] for r in matched]
        pck10_vals = [r[f"pck10_kpt{k}"] for r in matched]
        pck20_vals = [r[f"pck20_kpt{k}"] for r in matched]
        per_kpt.append({
            "keypoint": name,
            "rmse_px":  round(float(np.sqrt(np.nanmean(np.array(rmse_vals, dtype=float) ** 2))), 3),
            "mae_px":   round(float(np.nanmean(rmse_vals)), 3),
            "pck@0.05": round(float(np.nanmean(pck05_vals)), 4),
            "pck@0.10": round(float(np.nanmean(pck10_vals)), 4),
            "pck@0.20": round(float(np.nanmean(pck20_vals)), 4),
        })

    overall_rmse  = float(np.nanmean([p["rmse_px"]   for p in per_kpt]))
    overall_pck05 = float(np.nanmean([p["pck@0.05"]  for p in per_kpt]))
    overall_pck10 = float(np.nanmean([p["pck@0.10"]  for p in per_kpt]))
    overall_pck20 = float(np.nanmean([p["pck@0.20"]  for p in per_kpt]))

    return {
        "n_gt_instances":   total,
        "n_matched":        len(matched),
        "detection_rate":   round(det_rate, 4),
        "mean_oks":         round(mean_oks, 4),
        "overall_rmse_px":  round(overall_rmse, 3),
        "overall_pck@0.05": round(overall_pck05, 4),
        "overall_pck@0.10": round(overall_pck10, 4),
        "overall_pck@0.20": round(overall_pck20, 4),
        "per_keypoint":     per_kpt,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Fast YOLO-vs-MARS-GT keypoint comparison from two NPZ builds.")
    p.add_argument("--gt_manifest",   required=True,
                   help="Manifest from MARS GT NPZ build (e.g. mars_behaviorscope_npz_full).")
    p.add_argument("--pred_manifest", required=True,
                   help="Manifest from YOLO-pose NPZ build (e.g. mars_behaviorscope_npz_yolo_kp).")
    p.add_argument("--output_csv",    default="kpt_eval_fast_results.csv")
    p.add_argument("--splits",        nargs="+", default=["train", "val"],
                   help="Splits to evaluate from the GT manifest.")
    p.add_argument("--max_windows",   type=int, default=0,
                   help="0 = all matched windows. Sample N for a quick check.")
    args = p.parse_args()

    gt_manifest_path   = Path(args.gt_manifest)
    pred_manifest_path = Path(args.pred_manifest)
    gt_root   = gt_manifest_path.parent
    pred_root = pred_manifest_path.parent

    print(f"[fast_eval] GT manifest:   {gt_manifest_path}")
    print(f"[fast_eval] Pred manifest: {pred_manifest_path}")
    gt_man   = _load_manifest(gt_manifest_path)
    pred_man = _load_manifest(pred_manifest_path)

    gt_splits   = gt_man.get("splits", {})
    pred_splits = pred_man.get("splits", {})

    gt_samples: List[dict] = []
    for sp in args.splits:
        gt_samples.extend(gt_splits.get(sp, []))
    pred_samples: List[dict] = []
    for sp_rows in pred_splits.values():
        pred_samples.extend(sp_rows)

    print(f"[fast_eval] GT samples (splits {args.splits}): {len(gt_samples)}")
    print(f"[fast_eval] Pred samples (all splits):         {len(pred_samples)}")

    gt_idx   = _index_by_key(gt_samples,   coord_mode="source_video")
    pred_idx = _index_by_key(pred_samples, coord_mode="clip_relative")

    common_keys = sorted(set(gt_idx.keys()) & set(pred_idx.keys()))
    print(f"[fast_eval] Matched windows: {len(common_keys)}  "
          f"(GT-only={len(gt_idx) - len(common_keys)}  "
          f"pred-only={len(pred_idx) - len(common_keys)})")

    if args.max_windows > 0 and len(common_keys) > args.max_windows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(common_keys), size=args.max_windows, replace=False)
        common_keys = [common_keys[i] for i in sorted(idx)]
        print(f"[fast_eval] Subsampled to {len(common_keys)} windows.")

    all_rows: List[dict] = []
    skipped = 0
    for i, key in enumerate(common_keys):
        gt_s   = gt_idx[key]
        pred_s = pred_idx[key]
        gt_npz_path   = gt_root   / gt_s.get("sequence_npz", "")
        pred_npz_path = pred_root / pred_s.get("sequence_npz", "")
        if not gt_npz_path.exists() or not pred_npz_path.exists():
            skipped += 1
            continue
        if i % 500 == 0:
            print(f"[fast_eval] {i}/{len(common_keys)}  skipped={skipped}  rows={len(all_rows)}")
        try:
            rows = _compare_window(gt_npz_path, pred_npz_path)
            for r in rows:
                r["sample_id"]  = gt_s.get("id", "")
                r["class_name"] = gt_s.get("class_name", "")
                r["source_clip"] = gt_s.get("source_clip", "")
                r["start_frame"] = key[1]
            all_rows.extend(rows)
        except Exception as exc:
            print(f"[warn] {key}: {exc}")
            skipped += 1

    print(f"[fast_eval] Done. total_rows={len(all_rows)}  skipped={skipped}")
    if not all_rows:
        print("[fast_eval] No rows collected -- check manifest paths.")
        return

    output_csv  = Path(args.output_csv)
    raw_csv     = output_csv.with_name(output_csv.stem + "_raw.csv")
    summary_json = output_csv.with_name(output_csv.stem + "_summary.json")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    raw_fieldnames = (
        ["sample_id", "class_name", "source_clip", "start_frame",
         "frame_in_window", "gt_animal", "matched", "body_len_px", "oks"]
        + [f"rmse_kpt{k}"  for k in range(K)]
        + [f"pck05_kpt{k}" for k in range(K)]
        + [f"pck10_kpt{k}" for k in range(K)]
        + [f"pck20_kpt{k}" for k in range(K)]
    )
    with open(raw_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=raw_fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    print(f"[fast_eval] Raw rows -> {raw_csv}")

    agg = _aggregate(all_rows)
    per_kpt_rows = list(agg["per_keypoint"])
    per_kpt_rows.append({
        "keypoint":  "OVERALL",
        "rmse_px":   agg["overall_rmse_px"],
        "mae_px":    float("nan"),
        "pck@0.05":  agg["overall_pck@0.05"],
        "pck@0.10":  agg["overall_pck@0.10"],
        "pck@0.20":  agg["overall_pck@0.20"],
    })
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["keypoint","rmse_px","mae_px",
                                          "pck@0.05","pck@0.10","pck@0.20"])
        w.writeheader()
        w.writerows(per_kpt_rows)
    print(f"[fast_eval] Per-keypoint -> {output_csv}")

    # Pull YOLO build provenance from pred manifest meta (so the summary
    # records which weights / window settings produced these predictions).
    pred_meta = pred_man.get("meta", {}) or {}
    summary = {
        "gt_manifest":         str(gt_manifest_path),
        "pred_manifest":       str(pred_manifest_path),
        "yolo_weights":        pred_meta.get("yolo_weights"),
        "window_size":         pred_meta.get("window_size"),
        "window_stride":       pred_meta.get("window_stride"),
        "n_animals":           pred_meta.get("n_animals"),
        "keypoints_per_animal": pred_meta.get("keypoints_per_animal"),
        "splits_evaluated":    args.splits,
        "windows_matched":     len(common_keys),
        "windows_skipped":     skipped,
        **{k: v for k, v in agg.items() if k != "per_keypoint"},
        "per_keypoint":        agg["per_keypoint"],
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[fast_eval] Summary -> {summary_json}")

    print("\n" + "=" * 60)
    print(f"YOLO vs MARS GT keypoint evaluation (fast NPZ-vs-NPZ)")
    print(f"  Windows compared:    {len(common_keys) - skipped}")
    print(f"  Detection rate:      {agg['detection_rate']:.3f}")
    print(f"  Mean OKS:            {agg['mean_oks']:.4f}")
    print(f"  Overall RMSE (px):   {agg['overall_rmse_px']:.2f}")
    print(f"  Overall PCK@0.05:    {agg['overall_pck@0.05']:.3f}")
    print(f"  Overall PCK@0.10:    {agg['overall_pck@0.10']:.3f}")
    print(f"  Overall PCK@0.20:    {agg['overall_pck@0.20']:.3f}")
    print("\n  Per-keypoint:")
    for pk in agg["per_keypoint"]:
        print(f"    {pk['keypoint']:12s}  RMSE={pk['rmse_px']:6.2f}  "
              f"PCK@10={pk['pck@0.10']:.3f}  PCK@20={pk['pck@0.20']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
