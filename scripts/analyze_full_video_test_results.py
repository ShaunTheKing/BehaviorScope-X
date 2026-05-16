#!/usr/bin/env python3
"""
analyze_full_video_test_results.py
============================
Cross-arm analysis of the BehaviorScope-Y held-out test sweep.

Compares the full-video sliding-window training arm (this run) against the
clip-based training arm (the v4 reference) on the *same set of MARS test
videos* — 24-video apples-to-apples — and emits the slide-grade and
manuscript-grade artefacts:

  data/manuscript_v5/analysis/
    aggregated_metrics_<pred_type>.csv      one row per (arm x class)
    per_video_macroF1_<pred_type>.csv       per-video macro-F1 per arm
    head_to_head_full_vs_clip_<pred_type>.csv  per-video paired delta
    confusion_matrix_aggregate_full_<pt>.csv   pooled CM for the full-video arm
    confusion_matrix_aggregate_clip_<pt>.csv   pooled CM for the clip arm
    bout_duration_summary.csv               predicted vs GT bout durations
    raw_vs_smoothed_comparison.csv          smoothing lift per arm
    analysis_summary.json                   the headline JSON

Usage:
    python scripts/analyze_full_video_test_results.py

Default arms:
    full-video → data/manuscript_v5/test_eval/R5_yolo_attn_v5_full_mars_test_eval
    clip       → data/manuscript_v4/analysis/run_summaries/R5_yolo_attn_v4_full_mars_test_eval

(The clip-arm path points at the run-summary mirror because the live
test_eval directory was archived after the v4 cycle. The CSVs there are
identical to what the live eval produced.)
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_FULL_VIDEO_DIR = (
    REPO_ROOT / "data" / "manuscript_v5" / "test_eval"
    / "R5_yolo_attn_v5_full_mars_test_eval"
)
DEFAULT_CLIP_DIR = (
    REPO_ROOT / "data" / "manuscript_v4" / "analysis" / "run_summaries"
    / "R5_yolo_attn_v4_full_mars_test_eval"
)
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "manuscript_v5" / "analysis"

# Ablation matrix arms — order is the canonical reading order in the deck:
# the full model first, then ablations grouped by what they isolate.
# Each entry: (label, eval_dir). Missing dirs are tolerated and skipped.
TEST_EVAL_ROOT = REPO_ROOT / "data" / "manuscript_v5" / "test_eval"
ABLATION_ARMS: list[tuple[str, Path]] = [
    ("R5",  TEST_EVAL_ROOT / "R5_yolo_attn_v5_full_mars_test_eval"),
    ("R1",  TEST_EVAL_ROOT / "R1_yolo_attn_v5_crops_baseline_mars_test_eval"),
    ("R2",  TEST_EVAL_ROOT / "R2_yolo_attn_v5_pose_only_mars_test_eval"),
    ("R3",  TEST_EVAL_ROOT / "R3_yolo_attn_v5_visual_only_mars_test_eval"),
    ("R4",  TEST_EVAL_ROOT / "R4_yolo_attn_v5_no_group_rgb_mars_test_eval"),
    ("R4b", TEST_EVAL_ROOT / "R4b_yolo_attn_v5_no_relations_mars_test_eval"),
]
# Reference: the clip-arm v4 R5 result (kept separate from the v5 ablation
# matrix since it tests methodology rather than a stream toggle).
ABLATION_CLIP_REFERENCE: tuple[str, Path] = ("clip_R5", DEFAULT_CLIP_DIR)

CLASSES = ["attack", "investigation", "mount", "other"]
BEHAVIOR_CLASSES = ["attack", "investigation", "mount"]
PREDICTION_TYPES = ["raw_windows", "smoothed_frames"]

# QC exclusion: the archived clip-reference run only scored a short partial
# conversion for this video, while the full-video runs scored the full source.
# Without excluding it, the comparison has matching video IDs but mismatched
# frame support.
DEFAULT_QC_EXCLUDED_VIDEO_KEYS: set[tuple[str, str]] = {
    ("test_1", "Mouse060_20160526_18-16-27"),
}


def _safe(d: dict, k: str, cast=float, default=0):
    try:
        return cast(d.get(k, default) or default)
    except (ValueError, TypeError):
        return cast(default)


def _read_per_class(eval_dir: Path) -> list[dict]:
    p = eval_dir / "per_class_metrics.csv"
    if not p.exists():
        raise FileNotFoundError(f"per_class_metrics.csv not found under {eval_dir}")
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _video_keys(rows: list[dict]) -> set[tuple[str, str]]:
    return {(r["split"], r["video_id"]) for r in rows}


def _frame_support_by_video(
    rows: list[dict], pred_type: str, video_keys: set[tuple[str, str]]
) -> dict[tuple[str, str], int]:
    support: dict[tuple[str, str], int] = {}
    for r in rows:
        key = (r["split"], r["video_id"])
        if r.get("prediction_type") != pred_type or key not in video_keys:
            continue
        support[key] = support.get(key, 0) + _safe(r, "gt_frames", int)
    return support


def validate_matching_frame_support(
    left_rows: list[dict],
    right_rows: list[dict],
    video_keys: set[tuple[str, str]],
    pred_type: str,
    left_label: str,
    right_label: str,
) -> None:
    left = _frame_support_by_video(left_rows, pred_type, video_keys)
    right = _frame_support_by_video(right_rows, pred_type, video_keys)
    mismatches = [
        (key, left.get(key, 0), right.get(key, 0))
        for key in sorted(video_keys)
        if left.get(key, 0) != right.get(key, 0)
    ]
    if mismatches:
        details = "; ".join(
            f"{split}/{video}: {left_label}={lft}, {right_label}={rgt}"
            for (split, video), lft, rgt in mismatches
        )
        raise RuntimeError(
            f"Mismatched {pred_type} GT frame support after QC filtering: {details}"
        )


# --------------------------------------------------------------------------
# 1. Aggregated metrics per (arm x class) — frame + bout F1, restricted
#    to a shared video set so the two arms are apples-to-apples.
# --------------------------------------------------------------------------
def aggregate_metrics(
    rows: list[dict], pred_type: str, video_keys: set[tuple[str, str]] | None = None
) -> dict:
    rows = [r for r in rows if r.get("prediction_type") == pred_type]
    if video_keys is not None:
        rows = [r for r in rows if (r["split"], r["video_id"]) in video_keys]
    if not rows:
        return {}
    out: dict = {"pred_type": pred_type, "n_videos": len({(r["split"], r["video_id"]) for r in rows}), "per_class": {}}
    for c in CLASSES:
        # Frame counts — use cm_X columns to recover both correct and confused frames
        tp_f = sum(_safe(r, f"cm_{c}", int) for r in rows if r["class"] == c)
        fn_f = sum(
            _safe(r, f"cm_{other}", int)
            for r in rows if r["class"] == c
            for other in CLASSES if other != c
        )
        fp_f = sum(
            _safe(r, f"cm_{c}", int)
            for r in rows if r["class"] != c and r["class"] in CLASSES
        )
        prec_f = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0.0
        rec_f  = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0
        f1_f   = 2 * prec_f * rec_f / (prec_f + rec_f) if (prec_f + rec_f) > 0 else 0.0

        cls_rows = [r for r in rows if r["class"] == c]
        bouts = {}
        for thr in (25, 50):
            tp = sum(_safe(r, f"bout_tp_iou{thr}", int) for r in cls_rows)
            fp = sum(_safe(r, f"bout_fp_iou{thr}", int) for r in cls_rows)
            fn = sum(_safe(r, f"bout_fn_iou{thr}", int) for r in cls_rows)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            bouts[f"iou{thr}"] = {
                "tp": tp, "fp": fp, "fn": fn,
                "precision": round(prec, 4), "recall": round(rec, 4),
                "f1": round(f1, 4),
            }

        out["per_class"][c] = {
            "frame_tp": tp_f, "frame_fp": fp_f, "frame_fn": fn_f,
            "frame_precision": round(prec_f, 4),
            "frame_recall":    round(rec_f, 4),
            "frame_f1":        round(f1_f, 4),
            "bouts":           bouts,
        }

    out["frame_macro_f1"] = round(
        statistics.mean([out["per_class"][c]["frame_f1"] for c in CLASSES]), 4)
    out["frame_macro_f1_behavior"] = round(
        statistics.mean([out["per_class"][c]["frame_f1"] for c in BEHAVIOR_CLASSES]), 4)
    for thr in (25, 50):
        out[f"bout_macro_f1_iou{thr}"] = round(
            statistics.mean([out["per_class"][c]["bouts"][f"iou{thr}"]["f1"] for c in CLASSES]), 4)
        out[f"bout_macro_f1_behavior_iou{thr}"] = round(
            statistics.mean([out["per_class"][c]["bouts"][f"iou{thr}"]["f1"] for c in BEHAVIOR_CLASSES]), 4)
    return out


# --------------------------------------------------------------------------
# 2. Aggregate confusion matrix
# --------------------------------------------------------------------------
def aggregate_confusion(
    rows: list[dict], pred_type: str, video_keys: set[tuple[str, str]] | None = None
) -> list[list[int]]:
    rows = [r for r in rows if r.get("prediction_type") == pred_type]
    if video_keys is not None:
        rows = [r for r in rows if (r["split"], r["video_id"]) in video_keys]
    cm = [[0] * len(CLASSES) for _ in CLASSES]
    for r in rows:
        gt = r["class"]
        if gt not in CLASSES:
            continue
        gi = CLASSES.index(gt)
        for pj, pcls in enumerate(CLASSES):
            cm[gi][pj] += _safe(r, f"cm_{pcls}", int)
    return cm


def write_confusion(cm: list[list[int]], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt_class"] + [f"pred_{c}" for c in CLASSES] + ["row_total", "recall"])
        for i, gt in enumerate(CLASSES):
            row_total = sum(cm[i])
            recall = cm[i][i] / row_total if row_total > 0 else 0.0
            w.writerow([gt] + cm[i] + [row_total, round(recall, 4)])
    print(f"  -> {path}")


# --------------------------------------------------------------------------
# 3. Per-video macro-F1 (paired across arms)
# --------------------------------------------------------------------------
def per_video_macroF1(
    rows: list[dict], pred_type: str, video_keys: set[tuple[str, str]] | None = None
) -> list[dict]:
    rows = [r for r in rows if r.get("prediction_type") == pred_type]
    if video_keys is not None:
        rows = [r for r in rows if (r["split"], r["video_id"]) in video_keys]
    by_video: dict[tuple[str, str], dict[str, float]] = {}
    for r in rows:
        key = (r["split"], r["video_id"])
        by_video.setdefault(key, {})[r["class"]] = _safe(r, "frame_f1", float)
    out = []
    for (split, vid), perc in by_video.items():
        f1s = [perc.get(c, 0.0) for c in CLASSES]
        f1b = [perc.get(c, 0.0) for c in BEHAVIOR_CLASSES]
        out.append({
            "split": split, "video_id": vid,
            "macro_f1":          round(statistics.mean(f1s), 4),
            "macro_f1_behavior": round(statistics.mean(f1b), 4),
            **{f"f1_{c}": round(perc.get(c, 0.0), 4) for c in CLASSES},
        })
    out.sort(key=lambda r: (r["split"], r["video_id"]))
    return out


def write_head_to_head(
    full_pv: list[dict], clip_pv: list[dict], path: Path
) -> dict:
    a = {(r["split"], r["video_id"]): r for r in full_pv}
    b = {(r["split"], r["video_id"]): r for r in clip_pv}
    keys = sorted(set(a.keys()) & set(b.keys()))
    rows = []
    for k in keys:
        delta = a[k]["macro_f1"] - b[k]["macro_f1"]
        rows.append({
            "split": k[0], "video_id": k[1],
            "full_video_macro_f1": a[k]["macro_f1"],
            "clip_macro_f1":       b[k]["macro_f1"],
            "delta_full_minus_clip": round(delta, 4),
            "winner": "full_video" if delta > 0 else ("clip" if delta < 0 else "tie"),
            "full_video_macro_f1_behavior": a[k]["macro_f1_behavior"],
            "clip_macro_f1_behavior":       b[k]["macro_f1_behavior"],
            "delta_behavior_full_minus_clip": round(
                a[k]["macro_f1_behavior"] - b[k]["macro_f1_behavior"], 4),
        })
    if not rows:
        return {}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}")
    deltas = [r["delta_full_minus_clip"] for r in rows]
    deltas_b = [r["delta_behavior_full_minus_clip"] for r in rows]
    return {
        "n_videos": len(rows),
        "mean_delta_macro_f1":   round(statistics.mean(deltas), 4),
        "median_delta_macro_f1": round(statistics.median(deltas), 4),
        "stdev_delta_macro_f1":  round(statistics.stdev(deltas), 4) if len(deltas) > 1 else 0.0,
        "n_full_video_wins":     sum(1 for r in rows if r["winner"] == "full_video"),
        "n_clip_wins":           sum(1 for r in rows if r["winner"] == "clip"),
        "n_ties":                sum(1 for r in rows if r["winner"] == "tie"),
        "mean_delta_behavior":   round(statistics.mean(deltas_b), 4),
        "biggest_full_video_gain": max(rows, key=lambda r: r["delta_full_minus_clip"])["video_id"]
                                   if any(r["delta_full_minus_clip"] > 0 for r in rows) else None,
        "biggest_clip_gain":       min(rows, key=lambda r: r["delta_full_minus_clip"])["video_id"]
                                   if any(r["delta_full_minus_clip"] < 0 for r in rows) else None,
    }


# --------------------------------------------------------------------------
# 4. Bout-duration distribution (per arm, against shared GT)
# --------------------------------------------------------------------------
def bout_duration_stats(
    eval_dir: Path, video_keys: set[tuple[str, str]] | None = None
) -> dict:
    pred_path = eval_dir / "predicted_bouts.csv"
    gt_path   = eval_dir / "ground_truth_bouts.csv"
    if not pred_path.exists() or not gt_path.exists():
        return {}

    def _read_durations(p: Path):
        out: dict[str, list[float]] = {c: [] for c in CLASSES}
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if video_keys is not None and (r.get("split"), r.get("video_id")) not in video_keys:
                    continue
                cls = r.get("class") or r.get("behavior")
                if cls not in CLASSES:
                    continue
                start = _safe(r, "start_frame", float, default=0)
                end = _safe(r, "end_frame", float, default=0)
                pred_type = r.get("prediction_type", "")
                if "prediction_type" in r and pred_type and pred_type != "smoothed_frames":
                    continue
                if end > start:
                    out[cls].append(end - start)
        return out

    pred = _read_durations(pred_path)
    gt = _read_durations(gt_path)
    summary = {}
    for c in CLASSES:
        gd = gt[c]; pd_ = pred[c]
        summary[c] = {
            "gt_n":    len(gd),
            "pred_n":  len(pd_),
            "gt_mean_frames":   round(statistics.mean(gd), 1) if gd else 0.0,
            "gt_median_frames": round(statistics.median(gd), 1) if gd else 0.0,
            "pred_mean_frames":   round(statistics.mean(pd_), 1) if pd_ else 0.0,
            "pred_median_frames": round(statistics.median(pd_), 1) if pd_ else 0.0,
            "pred_to_gt_count_ratio": round(len(pd_) / len(gd), 3) if gd else None,
            "pred_to_gt_mean_ratio":  round(
                statistics.mean(pd_) / statistics.mean(gd), 3
            ) if (gd and pd_) else None,
        }
    return summary


def write_bout_duration(stats_by_arm: dict[str, dict], path: Path) -> None:
    rows = []
    for arm, bd in stats_by_arm.items():
        for c, stats in bd.items():
            rows.append({"arm": arm, "class": c, **stats})
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {path}")


# --------------------------------------------------------------------------
# 5. Raw vs smoothed comparison per arm
# --------------------------------------------------------------------------
def raw_vs_smoothed(rows: list[dict], video_keys: set[tuple[str, str]]) -> dict:
    raw = aggregate_metrics(rows, "raw_windows", video_keys)
    sm  = aggregate_metrics(rows, "smoothed_frames", video_keys)
    if not raw or not sm:
        return {}
    return {
        "frame_macro_f1_raw":      raw["frame_macro_f1"],
        "frame_macro_f1_smoothed": sm["frame_macro_f1"],
        "delta_smoothing_lift":    round(sm["frame_macro_f1"] - raw["frame_macro_f1"], 4),
        "bout_macro_f1_iou25_raw":      raw.get("bout_macro_f1_iou25"),
        "bout_macro_f1_iou25_smoothed": sm.get("bout_macro_f1_iou25"),
        "bout_macro_f1_iou50_raw":      raw.get("bout_macro_f1_iou50"),
        "bout_macro_f1_iou50_smoothed": sm.get("bout_macro_f1_iou50"),
    }


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--full_video_dir", type=Path, default=DEFAULT_FULL_VIDEO_DIR)
    p.add_argument("--clip_dir",       type=Path, default=DEFAULT_CLIP_DIR)
    p.add_argument("--out_dir",        type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--include_qc_failed_videos",
        action="store_true",
        help="Include videos excluded by the default cross-arm QC gate.",
    )
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[analyze] full-video arm: {args.full_video_dir}")
    print(f"[analyze] clip arm:       {args.clip_dir}")
    print(f"[analyze] output dir:     {args.out_dir}")
    print()

    full_rows = _read_per_class(args.full_video_dir)
    clip_rows = _read_per_class(args.clip_dir)

    excluded_video_keys = set() if args.include_qc_failed_videos else DEFAULT_QC_EXCLUDED_VIDEO_KEYS

    # Restrict to the shared video set so the two arms are apples-to-apples.
    full_keys = _video_keys(full_rows)
    clip_keys = _video_keys(clip_rows)
    shared = (full_keys & clip_keys) - excluded_video_keys
    print(f"[analyze] full-video videos: {len(full_keys)}")
    print(f"[analyze] clip videos:       {len(clip_keys)}")
    print(f"[analyze] shared:            {len(shared)}")
    if excluded_video_keys:
        print(f"[analyze] QC-excluded:       {sorted(excluded_video_keys)}")
    if full_keys - clip_keys:
        print(f"[analyze]   only full-video: {sorted(full_keys - clip_keys)}")
    if clip_keys - full_keys:
        print(f"[analyze]   only clip:       {sorted(clip_keys - full_keys)}")
    validate_matching_frame_support(
        full_rows, clip_rows, shared, "smoothed_frames", "full_video", "clip"
    )

    # --- 1. Aggregated metrics CSVs (full-video + clip side by side) -----
    print("\n[1/6] Aggregated metrics CSVs:")
    for pt in PREDICTION_TYPES:
        full_agg = aggregate_metrics(full_rows, pt, shared)
        clip_agg = aggregate_metrics(clip_rows, pt, shared)
        out_rows = []
        for c in CLASSES:
            for arm, agg in [("full_video", full_agg), ("clip", clip_agg)]:
                m = agg["per_class"][c]
                out_rows.append({
                    "arm": arm, "class": c,
                    "frame_precision": m["frame_precision"],
                    "frame_recall":    m["frame_recall"],
                    "frame_f1":        m["frame_f1"],
                    "bout_iou25_precision": m["bouts"]["iou25"]["precision"],
                    "bout_iou25_recall":    m["bouts"]["iou25"]["recall"],
                    "bout_iou25_f1":        m["bouts"]["iou25"]["f1"],
                    "bout_iou50_precision": m["bouts"]["iou50"]["precision"],
                    "bout_iou50_recall":    m["bouts"]["iou50"]["recall"],
                    "bout_iou50_f1":        m["bouts"]["iou50"]["f1"],
                })
        path = args.out_dir / f"aggregated_metrics_{pt}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"  -> {path}")

    # --- 2. Per-video macro-F1 ------------------------------------------
    print("\n[2/6] Per-video macro-F1 CSVs:")
    for pt in PREDICTION_TYPES:
        full_pv = per_video_macroF1(full_rows, pt, shared)
        clip_pv = per_video_macroF1(clip_rows, pt, shared)
        out_rows = []
        for arm, pv in [("full_video", full_pv), ("clip", clip_pv)]:
            for r in pv:
                out_rows.append({"arm": arm, **r})
        path = args.out_dir / f"per_video_macroF1_{pt}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"  -> {path}")

    # --- 3. Head-to-head (smoothed_frames is the headline) ---------------
    print("\n[3/6] Head-to-head full-video vs clip (paired per video):")
    full_pv_sm = per_video_macroF1(full_rows, "smoothed_frames", shared)
    clip_pv_sm = per_video_macroF1(clip_rows, "smoothed_frames", shared)
    h2h_path = args.out_dir / "head_to_head_full_vs_clip_smoothed_frames.csv"
    h2h_summary = write_head_to_head(full_pv_sm, clip_pv_sm, h2h_path)

    # --- 4. Aggregate confusion matrices ---------------------------------
    print("\n[4/6] Aggregate confusion matrices (smoothed_frames):")
    for arm, rows in [("full_video", full_rows), ("clip", clip_rows)]:
        cm = aggregate_confusion(rows, "smoothed_frames", shared)
        path = args.out_dir / f"confusion_matrix_aggregate_{arm}_smoothed_frames.csv"
        write_confusion(cm, path)

    # --- 5. Bout durations (per arm, against shared GT) ------------------
    print("\n[5/6] Bout duration summary:")
    full_bd = bout_duration_stats(args.full_video_dir, shared)
    clip_bd = bout_duration_stats(args.clip_dir, shared)
    write_bout_duration(
        {"full_video": full_bd, "clip": clip_bd},
        args.out_dir / "bout_duration_summary.csv",
    )

    # --- 6. Raw vs smoothed (smoothing lift per arm) ---------------------
    print("\n[6/6] Raw vs smoothed comparison:")
    rvs = {
        "full_video": raw_vs_smoothed(full_rows, shared),
        "clip":       raw_vs_smoothed(clip_rows, shared),
    }
    rvs_path = args.out_dir / "raw_vs_smoothed_comparison.csv"
    with open(rvs_path, "w", newline="", encoding="utf-8") as f:
        cols = ["arm"] + list(rvs["full_video"].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for arm, d in rvs.items():
            w.writerow({"arm": arm, **d})
    print(f"  -> {rvs_path}")

    # ---------------------------------------------------------------
    # Ablation matrix — 6 v5 arms (R5 + R1/R2/R3/R4/R4b) restricted to
    # the intersection of their video sets, plus the clip R5 reference
    # for direct cross-methodology readability.
    # ---------------------------------------------------------------
    print("\n[ablation] discovering arms...")
    ablation_data: dict[str, list[dict]] = {}
    ablation_paths: dict[str, str] = {}
    for label, eval_dir in ABLATION_ARMS:
        if (eval_dir / "per_class_metrics.csv").exists():
            ablation_data[label] = _read_per_class(eval_dir)
            ablation_paths[label] = str(eval_dir)
            print(f"  {label:<5} {len(_video_keys(ablation_data[label])):>3} videos  {eval_dir.name}")
        else:
            print(f"  {label:<5} (missing) {eval_dir}")
    # Add clip reference if available (already loaded into clip_rows above)
    clip_label = ABLATION_CLIP_REFERENCE[0]
    if clip_rows:
        ablation_data[clip_label] = clip_rows
        ablation_paths[clip_label] = str(args.clip_dir)
        print(f"  {clip_label:<5} {len(_video_keys(clip_rows)):>3} videos  {Path(args.clip_dir).name} (reference)")

    # Restrict every arm to the same shared video set
    shared_ablation: set[tuple[str, str]] | None = None
    for label, rows in ablation_data.items():
        keys = _video_keys(rows)
        shared_ablation = keys if shared_ablation is None else (shared_ablation & keys)
    shared_ablation = (shared_ablation or set()) - excluded_video_keys
    print(f"\n[ablation] shared video set across all {len(ablation_data)} arms: {len(shared_ablation)} videos")
    if "R5" in ablation_data and "clip_R5" in ablation_data:
        validate_matching_frame_support(
            ablation_data["R5"],
            ablation_data["clip_R5"],
            shared_ablation,
            "smoothed_frames",
            "R5",
            "clip_R5",
        )
    if "R5" in ablation_data:
        for label, rows in ablation_data.items():
            if label == "R5":
                continue
            validate_matching_frame_support(
                ablation_data["R5"],
                rows,
                shared_ablation,
                "smoothed_frames",
                "R5",
                label,
            )

    # Aggregated metrics per (arm × class) — long-format CSV
    print("\n[ablation] aggregated metrics CSVs (arm × class):")
    ablation_headlines: dict[str, dict] = {}
    for pt in PREDICTION_TYPES:
        out_rows = []
        for label, rows in ablation_data.items():
            agg = aggregate_metrics(rows, pt, shared_ablation)
            if not agg:
                continue
            for c in CLASSES:
                m = agg["per_class"][c]
                out_rows.append({
                    "arm": label, "class": c,
                    "frame_precision":      m["frame_precision"],
                    "frame_recall":         m["frame_recall"],
                    "frame_f1":             m["frame_f1"],
                    "bout_iou25_precision": m["bouts"]["iou25"]["precision"],
                    "bout_iou25_recall":    m["bouts"]["iou25"]["recall"],
                    "bout_iou25_f1":        m["bouts"]["iou25"]["f1"],
                    "bout_iou50_precision": m["bouts"]["iou50"]["precision"],
                    "bout_iou50_recall":    m["bouts"]["iou50"]["recall"],
                    "bout_iou50_f1":        m["bouts"]["iou50"]["f1"],
                })
            if pt == "smoothed_frames":
                ablation_headlines[label] = agg
        path = args.out_dir / f"ablation_aggregated_metrics_{pt}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"  -> {path}")

    # Per-stream attribution: R5 − R_i per class (smoothed_frames)
    print("\n[ablation] per-stream attribution CSV (R5 minus each ablation, per class):")
    if "R5" in ablation_headlines:
        ref = ablation_headlines["R5"]
        attr_rows = []
        for label, agg in ablation_headlines.items():
            if label == "R5":
                continue
            for c in CLASSES:
                ref_m = ref["per_class"][c]
                arm_m = agg["per_class"][c]
                attr_rows.append({
                    "ablation": label, "class": c,
                    "ref_R5_frame_f1":           ref_m["frame_f1"],
                    "arm_frame_f1":              arm_m["frame_f1"],
                    "delta_frame_f1":            round(ref_m["frame_f1"] - arm_m["frame_f1"], 4),
                    "ref_R5_bout_iou25_f1":      ref_m["bouts"]["iou25"]["f1"],
                    "arm_bout_iou25_f1":         arm_m["bouts"]["iou25"]["f1"],
                    "delta_bout_iou25_f1":       round(ref_m["bouts"]["iou25"]["f1"] - arm_m["bouts"]["iou25"]["f1"], 4),
                    "ref_R5_bout_iou50_f1":      ref_m["bouts"]["iou50"]["f1"],
                    "arm_bout_iou50_f1":         arm_m["bouts"]["iou50"]["f1"],
                    "delta_bout_iou50_f1":       round(ref_m["bouts"]["iou50"]["f1"] - arm_m["bouts"]["iou50"]["f1"], 4),
                })
        attr_path = args.out_dir / "ablation_per_class_attribution_smoothed_frames.csv"
        with open(attr_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(attr_rows[0].keys()))
            w.writeheader()
            w.writerows(attr_rows)
        print(f"  -> {attr_path}")

    # Per-video macro-F1 (long format: arm × video)
    print("\n[ablation] per-video macro-F1 CSVs (arm × video):")
    for pt in PREDICTION_TYPES:
        out_rows = []
        for label, rows in ablation_data.items():
            for r in per_video_macroF1(rows, pt, shared_ablation):
                out_rows.append({"arm": label, **r})
        path = args.out_dir / f"ablation_per_video_macroF1_{pt}.csv"
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"  -> {path}")

    # Confusion matrices per arm (smoothed_frames, on the shared set)
    print("\n[ablation] confusion matrices per arm (smoothed_frames):")
    ablation_confusions: dict[str, list[list[int]]] = {}
    for label, rows in ablation_data.items():
        cm = aggregate_confusion(rows, "smoothed_frames", shared_ablation)
        ablation_confusions[label] = cm
        path = args.out_dir / f"ablation_confusion_matrix_{label}_smoothed_frames.csv"
        write_confusion(cm, path)

    # --- Final JSON summary ----------------------------------------------
    summary = {
        "shared_video_count": len(shared),
        "shared_videos": [list(t) for t in sorted(shared)],
        "qc_excluded_videos": [list(t) for t in sorted(excluded_video_keys)],
        "qc_exclusion_reason": (
            "Excluded from all cross-arm comparisons because the archived "
            "clip-reference run scored only a short partial conversion for "
            "Mouse060, while the full-video runs scored the full source."
            if excluded_video_keys else ""
        ),
        "headlines_smoothed_frames": {
            "full_video": aggregate_metrics(full_rows, "smoothed_frames", shared),
            "clip":       aggregate_metrics(clip_rows, "smoothed_frames", shared),
        },
        "headlines_raw_windows": {
            "full_video": aggregate_metrics(full_rows, "raw_windows", shared),
            "clip":       aggregate_metrics(clip_rows, "raw_windows", shared),
        },
        "raw_vs_smoothed": rvs,
        "head_to_head_full_vs_clip": h2h_summary,
        "input_paths": {
            "full_video": str(args.full_video_dir),
            "clip":       str(args.clip_dir),
        },
        "ablation_matrix": {
            "shared_video_count": len(shared_ablation),
            "shared_videos": [list(t) for t in sorted(shared_ablation)],
            "qc_excluded_videos": [list(t) for t in sorted(excluded_video_keys)],
            "arm_paths": ablation_paths,
            "headlines_smoothed_frames": ablation_headlines,
        },
    }
    json_path = args.out_dir / "analysis_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  -> {json_path}")

    # --- Pretty-printed headline table -----------------------------------
    print()
    print("=" * 96)
    print(f"R5 — held-out test sweep on {len(shared)} shared MARS videos (smoothed_frames + temporal splitter)")
    print("=" * 96)
    print(f"{'Arm':<14} {'frame F1':>10} {'beh-only':>10} "
          f"{'bout@.25':>10} {'bout@.50':>10} {'beh@.25':>10} {'beh@.50':>10}")
    print("-" * 96)
    for arm in ("clip", "full_video"):
        a = summary["headlines_smoothed_frames"][arm]
        if not a:
            continue
        print(f"{arm:<14} "
              f"{a['frame_macro_f1']:>10.4f} "
              f"{a['frame_macro_f1_behavior']:>10.4f} "
              f"{a.get('bout_macro_f1_iou25', 0):>10.4f} "
              f"{a.get('bout_macro_f1_iou50', 0):>10.4f} "
              f"{a.get('bout_macro_f1_behavior_iou25', 0):>10.4f} "
              f"{a.get('bout_macro_f1_behavior_iou50', 0):>10.4f}")

    full_h = summary["headlines_smoothed_frames"]["full_video"]
    clip_h = summary["headlines_smoothed_frames"]["clip"]
    if full_h and clip_h:
        print("-" * 96)
        print(f"{'delta (full-clip)':<14} "
              f"{full_h['frame_macro_f1']-clip_h['frame_macro_f1']:>+10.4f} "
              f"{full_h['frame_macro_f1_behavior']-clip_h['frame_macro_f1_behavior']:>+10.4f} "
              f"{full_h['bout_macro_f1_iou25']-clip_h['bout_macro_f1_iou25']:>+10.4f} "
              f"{full_h['bout_macro_f1_iou50']-clip_h['bout_macro_f1_iou50']:>+10.4f} "
              f"{full_h['bout_macro_f1_behavior_iou25']-clip_h['bout_macro_f1_behavior_iou25']:>+10.4f} "
              f"{full_h['bout_macro_f1_behavior_iou50']-clip_h['bout_macro_f1_behavior_iou50']:>+10.4f}")
    print("=" * 96)
    if h2h_summary:
        print(f"Per-video paired (n={h2h_summary['n_videos']}): "
              f"full-video wins {h2h_summary['n_full_video_wins']}, "
              f"clip wins {h2h_summary['n_clip_wins']}, "
              f"ties {h2h_summary['n_ties']} | "
              f"mean delta macro-F1 = {h2h_summary['mean_delta_macro_f1']:+.4f} "
              f"(behavior-only delta = {h2h_summary['mean_delta_behavior']:+.4f})")
    print()

    # --- Ablation matrix headline table ----------------------------------
    if ablation_headlines:
        print("=" * 96)
        print(f"v5 ablation matrix on {len(shared_ablation)} shared MARS videos "
              f"(smoothed_frames + temporal splitter)")
        print("=" * 96)
        print(f"{'Arm':<8} {'frame F1':>10} {'beh-only':>10} "
              f"{'bout@.25':>10} {'bout@.50':>10} {'beh@.25':>10} {'beh@.50':>10}")
        print("-" * 96)
        # Sort by frame_macro_f1 descending so the manuscript-headline ranking is obvious
        ordered = sorted(
            ablation_headlines.items(),
            key=lambda kv: kv[1].get("frame_macro_f1", 0),
            reverse=True,
        )
        for label, a in ordered:
            print(f"{label:<8} "
                  f"{a['frame_macro_f1']:>10.4f} "
                  f"{a['frame_macro_f1_behavior']:>10.4f} "
                  f"{a.get('bout_macro_f1_iou25', 0):>10.4f} "
                  f"{a.get('bout_macro_f1_iou50', 0):>10.4f} "
                  f"{a.get('bout_macro_f1_behavior_iou25', 0):>10.4f} "
                  f"{a.get('bout_macro_f1_behavior_iou50', 0):>10.4f}")
        print("=" * 96)
        # Per-stream attribution one-liner
        if "R5" in ablation_headlines:
            r5 = ablation_headlines["R5"]
            print()
            print("Per-stream attribution (R5 frame F1 minus each ablation, on shared set):")
            for label, a in ordered:
                if label in ("R5", "clip_R5"):
                    continue
                d = r5["frame_macro_f1"] - a["frame_macro_f1"]
                print(f"  R5 - {label:<4} = {d:+.4f}  (frame F1)")
            if "clip_R5" in ablation_headlines:
                c = ablation_headlines["clip_R5"]
                print()
                print(f"Methodology baseline: R5 (full-video) - clip_R5 = "
                      f"{r5['frame_macro_f1']-c['frame_macro_f1']:+.4f} frame F1, "
                      f"{r5.get('bout_macro_f1_iou25',0)-c.get('bout_macro_f1_iou25',0):+.4f} bout F1 @ 0.25")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
