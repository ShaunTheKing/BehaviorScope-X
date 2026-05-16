#!/usr/bin/env python3
"""
analyze_clip_test_results.py
============================
Cross-run analysis of v4 BehaviorScope-Y test-eval outputs.

Reads each run's ``per_class_metrics.csv`` + ``predicted_bouts.csv`` +
``ground_truth_bouts.csv`` from ``data/manuscript_v4/test_eval/<run>/`` and
produces:

  - ``aggregated_metrics_<pred_type>.csv``  — one row per (run x class) with
    aggregated frame & bout counts (tp/fp/fn) and derived precision/recall/F1
    at IoU 0.25 and 0.50.
  - ``per_video_macroF1_<pred_type>.csv``    — per-video macro-F1 per run
  - ``head_to_head_<runA>_vs_<runB>.csv``    — per-video macro-F1 delta
  - ``confusion_matrix_aggregate_<run>.csv`` — test-set confusion matrix
    (rows = GT class, cols = predicted class)
  - ``bout_duration_<run>.csv``              — predicted vs GT bout duration
    distributions per class
  - ``raw_vs_smoothed_<run>.csv``            — does temporal smoothing help?
  - ``analysis_summary.json``                — headlines all in one file

Usage:
    python scripts/analyze_clip_test_results.py
    python scripts/analyze_clip_test_results.py --runs R5 R4
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_EVAL_DIR = REPO_ROOT / "data" / "manuscript_v4" / "test_eval"
ANALYSIS_DIR = REPO_ROOT / "data" / "manuscript_v4" / "analysis"

CLASSES = ["attack", "investigation", "mount", "other"]
BEHAVIOR_CLASSES = ["attack", "investigation", "mount"]
PREDICTION_TYPES = ["raw_windows", "smoothed_frames"]

# Map short -> dirname
RUN_MAP = {
    "R1":  "R1_yolo_attn_v4_crops_baseline",
    "R2":  "R2_yolo_attn_v4_pose_only",
    "R3":  "R3_yolo_attn_v4_visual_only",
    "R4":  "R4_yolo_attn_v4_no_group_rgb",
    "R4b": "R4b_yolo_attn_v4_no_relations",
    "R5":  "R5_yolo_attn_v4_full",
}


def _eval_dir(short: str) -> Path:
    return TEST_EVAL_DIR / f"{RUN_MAP[short]}_mars_test_eval"


def _read_per_class(short: str) -> list[dict]:
    p = _eval_dir(short) / "per_class_metrics.csv"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe(d: dict, k: str, cast=float, default=0):
    try:
        return cast(d.get(k, default) or default)
    except (ValueError, TypeError):
        return cast(default)


# --------------------------------------------------------------------------
# 1. Aggregated metrics per (run x class) — frame + bout F1 at both IoU
# --------------------------------------------------------------------------
def aggregate_metrics(short: str, pred_type: str) -> dict:
    rows = [r for r in _read_per_class(short) if r.get("prediction_type") == pred_type]
    if not rows:
        return {}
    out: dict = {"run": short, "pred_type": pred_type, "per_class": {}}
    for c in CLASSES:
        # Frame counts -> aggregate F1 per class via confusion matrix sums
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

        # Bout-level — sum tp/fp/fn across videos for this class
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

    # Macro-F1 across classes
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


def write_aggregated_metrics_csv(short_list: list[str], pred_type: str) -> None:
    rows = []
    for s in short_list:
        agg = aggregate_metrics(s, pred_type)
        if not agg:
            continue
        for c, m in agg["per_class"].items():
            rows.append({
                "run": s, "class": c,
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
    out_path = ANALYSIS_DIR / f"aggregated_metrics_{pred_type}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {out_path}")


# --------------------------------------------------------------------------
# 2. Per-video macro-F1
# --------------------------------------------------------------------------
def per_video_macroF1(short: str, pred_type: str) -> list[dict]:
    rows = [r for r in _read_per_class(short) if r.get("prediction_type") == pred_type]
    by_video: dict[str, dict[str, float]] = {}
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


def write_per_video_csv(short_list: list[str], pred_type: str) -> None:
    rows = []
    for s in short_list:
        for r in per_video_macroF1(s, pred_type):
            rows.append({"run": s, **r})
    if not rows:
        return
    out_path = ANALYSIS_DIR / f"per_video_macroF1_{pred_type}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {out_path}")


# --------------------------------------------------------------------------
# 3. Head-to-head per-video comparison
# --------------------------------------------------------------------------
def head_to_head(runA: str, runB: str, pred_type: str = "smoothed_frames") -> list[dict]:
    a = {(r["split"], r["video_id"]): r for r in per_video_macroF1(runA, pred_type)}
    b = {(r["split"], r["video_id"]): r for r in per_video_macroF1(runB, pred_type)}
    keys = sorted(set(a.keys()) & set(b.keys()))
    rows = []
    for k in keys:
        delta = a[k]["macro_f1"] - b[k]["macro_f1"]
        rows.append({
            "split": k[0], "video_id": k[1],
            f"{runA}_macro_f1": a[k]["macro_f1"],
            f"{runB}_macro_f1": b[k]["macro_f1"],
            "delta_A_minus_B":  round(delta, 4),
            "winner":           runA if delta > 0 else (runB if delta < 0 else "tie"),
        })
    return rows


def write_head_to_head(runA: str, runB: str, pred_type: str = "smoothed_frames") -> dict:
    rows = head_to_head(runA, runB, pred_type)
    if not rows:
        return {}
    out_path = ANALYSIS_DIR / f"head_to_head_{runA}_vs_{runB}_{pred_type}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {out_path}")
    deltas = [r["delta_A_minus_B"] for r in rows]
    summary = {
        "runA": runA, "runB": runB,
        "n_videos": len(rows),
        "mean_delta":   round(statistics.mean(deltas), 4),
        "median_delta": round(statistics.median(deltas), 4),
        "stdev_delta":  round(statistics.stdev(deltas), 4) if len(deltas) > 1 else 0.0,
        "n_A_wins":     sum(1 for r in rows if r["winner"] == runA),
        "n_B_wins":     sum(1 for r in rows if r["winner"] == runB),
        "n_ties":       sum(1 for r in rows if r["winner"] == "tie"),
        "biggest_A_gain":  max(rows, key=lambda r: r["delta_A_minus_B"])["video_id"]
                           if any(r["delta_A_minus_B"] > 0 for r in rows) else None,
        "biggest_B_gain":  min(rows, key=lambda r: r["delta_A_minus_B"])["video_id"]
                           if any(r["delta_A_minus_B"] < 0 for r in rows) else None,
    }
    return summary


# --------------------------------------------------------------------------
# 4. Aggregate confusion matrix (sums cm_X across all videos for smoothed)
# --------------------------------------------------------------------------
def aggregate_confusion(short: str, pred_type: str = "smoothed_frames") -> list[list[int]]:
    rows = [r for r in _read_per_class(short) if r.get("prediction_type") == pred_type]
    cm = [[0] * len(CLASSES) for _ in CLASSES]
    for r in rows:
        gt = r["class"]
        if gt not in CLASSES:
            continue
        gi = CLASSES.index(gt)
        for pj, pcls in enumerate(CLASSES):
            cm[gi][pj] += _safe(r, f"cm_{pcls}", int)
    return cm


def write_confusion(short: str, pred_type: str = "smoothed_frames") -> None:
    cm = aggregate_confusion(short, pred_type)
    out_path = ANALYSIS_DIR / f"confusion_matrix_aggregate_{short}_{pred_type}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gt_class"] + [f"pred_{c}" for c in CLASSES] + ["row_total", "recall"])
        for i, gt in enumerate(CLASSES):
            row_total = sum(cm[i])
            recall = cm[i][i] / row_total if row_total > 0 else 0.0
            w.writerow([gt] + cm[i] + [row_total, round(recall, 4)])
    print(f"  -> {out_path}")


# --------------------------------------------------------------------------
# 5. Bout duration analysis (predicted vs GT)
# --------------------------------------------------------------------------
def bout_duration_stats(short: str) -> dict:
    pred_path = _eval_dir(short) / "predicted_bouts.csv"
    gt_path   = _eval_dir(short) / "ground_truth_bouts.csv"
    if not pred_path.exists() or not gt_path.exists():
        return {}
    def _read_durations(p: Path):
        out: dict[str, list[float]] = {c: [] for c in CLASSES}
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                cls = r.get("class") or r.get("behavior")
                if cls not in CLASSES:
                    continue
                start = _safe(r, "start_frame", float, default=0)
                end = _safe(r, "end_frame", float, default=0)
                pred_type = r.get("prediction_type", "")
                # only count smoothed-frames bouts for predicted side
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


def write_bout_duration(short_list: list[str]) -> None:
    rows = []
    for s in short_list:
        bd = bout_duration_stats(s)
        for c, stats in bd.items():
            rows.append({"run": s, "class": c, **stats})
    if not rows:
        return
    out_path = ANALYSIS_DIR / "bout_duration_summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {out_path}")


# --------------------------------------------------------------------------
# 6. Raw vs smoothed comparison
# --------------------------------------------------------------------------
def raw_vs_smoothed(short: str) -> dict:
    raw = aggregate_metrics(short, "raw_windows")
    sm  = aggregate_metrics(short, "smoothed_frames")
    if not raw or not sm:
        return {}
    return {
        "run": short,
        "frame_macro_f1_raw":      raw["frame_macro_f1"],
        "frame_macro_f1_smoothed": sm["frame_macro_f1"],
        "delta_smoothing_lift":    round(sm["frame_macro_f1"] - raw["frame_macro_f1"], 4),
        "bout_macro_f1_iou25_raw":      raw.get("bout_macro_f1_iou25"),
        "bout_macro_f1_iou25_smoothed": sm.get("bout_macro_f1_iou25"),
        "bout_macro_f1_iou50_raw":      raw.get("bout_macro_f1_iou50"),
        "bout_macro_f1_iou50_smoothed": sm.get("bout_macro_f1_iou50"),
    }


def write_raw_vs_smoothed(short_list: list[str]) -> None:
    rows = [raw_vs_smoothed(s) for s in short_list]
    rows = [r for r in rows if r]
    if not rows:
        return
    out_path = ANALYSIS_DIR / "raw_vs_smoothed_comparison.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {out_path}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=None,
                   help="Subset (e.g. R5 R4). Default: all runs with completed eval.")
    args = p.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Discover runs that have per_class_metrics.csv
    available = []
    for short in RUN_MAP:
        if (_eval_dir(short) / "per_class_metrics.csv").exists():
            available.append(short)
    runs = args.runs or available
    print(f"[analyze] runs available: {available}")
    print(f"[analyze] processing:     {runs}")
    print(f"[analyze] output dir:     {ANALYSIS_DIR}")
    print()

    # 1. Aggregated metrics (frame + bout F1)
    print("[1/6] Aggregated metrics CSVs:")
    for pt in PREDICTION_TYPES:
        write_aggregated_metrics_csv(runs, pt)

    # 2. Per-video macro-F1
    print("\n[2/6] Per-video macro-F1 CSVs:")
    for pt in PREDICTION_TYPES:
        write_per_video_csv(runs, pt)

    # 3. Head-to-head (only if at least 2 runs)
    print("\n[3/6] Head-to-head comparisons:")
    h2h_summaries = []
    if len(runs) >= 2:
        # Compare every run to R5 (the headline)
        if "R5" in runs:
            for other in runs:
                if other != "R5":
                    s = write_head_to_head("R5", other, "smoothed_frames")
                    if s:
                        h2h_summaries.append(s)

    # 4. Aggregate confusion matrices
    print("\n[4/6] Aggregate confusion matrices (smoothed_frames):")
    for short in runs:
        write_confusion(short, "smoothed_frames")

    # 5. Bout durations
    print("\n[5/6] Bout duration summary:")
    write_bout_duration(runs)

    # 6. Raw vs smoothed
    print("\n[6/6] Raw vs smoothed comparison:")
    write_raw_vs_smoothed(runs)

    # Final JSON summary
    print()
    summary = {
        "runs_analyzed": runs,
        "headlines": {short: aggregate_metrics(short, "smoothed_frames") for short in runs},
        "raw_vs_smoothed": [raw_vs_smoothed(s) for s in runs],
        "head_to_head_vs_R5": h2h_summaries,
    }
    with open(ANALYSIS_DIR / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {ANALYSIS_DIR / 'analysis_summary.json'}")

    # Print the headlines
    print()
    print("=" * 90)
    print(f"{'Run':<5} {'frame F1':>10} {'beh-only':>10} "
          f"{'bout@.25':>10} {'bout@.50':>10} {'beh@.25':>10} {'beh@.50':>10}")
    print("-" * 90)
    for short in runs:
        a = aggregate_metrics(short, "smoothed_frames")
        if not a:
            continue
        print(f"{short:<5} "
              f"{a['frame_macro_f1']:>10.4f} "
              f"{a['frame_macro_f1_behavior']:>10.4f} "
              f"{a.get('bout_macro_f1_iou25', 0):>10.4f} "
              f"{a.get('bout_macro_f1_iou50', 0):>10.4f} "
              f"{a.get('bout_macro_f1_behavior_iou25', 0):>10.4f} "
              f"{a.get('bout_macro_f1_behavior_iou50', 0):>10.4f}")
    print("=" * 90)
    print("(All numbers are smoothed_frames + temporal_splitter post-processing.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
