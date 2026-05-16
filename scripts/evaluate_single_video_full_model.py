"""
evaluate_single_video_full_model.py

Evaluate BehaviorScope-Y inference (raw windows + smoothed per-frame) for
Mouse060_20160526_18-16-27 against MARS Bento ground truth, mirroring metric
definitions from scripts/evaluate_yolo_pipeline_smoke.py exactly.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOTAL_FRAMES = 24292
FPS = 30.0
CLASS_MAP = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
EVAL_CLASSES = [0, 1, 2]  # attack, investigation, mount
VIDEO = "Mouse060_20160526_18-16-27"

ANNOT_PATH = (
    r".\MARS_data\test_1\Mouse060_20160526_18-16-27"
    r"\Mouse060_20160526_18-16-27_anno.annot"
)
Y_RAW_CSV = (
    r".\BehaviorScope"
    r"\behavior_lstm_runs\R5_yolo_attn_ep10_mars_test_eval\test_1"
    r"\Mouse060_20160526_18-16-27\Mouse060_20160526_18-16-27.behavior.csv"
)
Y_SMOOTHED_CSV = (
    r".\BehaviorScope"
    r"\behavior_lstm_runs\R5_yolo_attn_ep10_mars_test_eval\test_1"
    r"\Mouse060_20160526_18-16-27\Mouse060_20160526_18-16-27.behavior.smoothed_frames.csv"
)
OUT_DIR = (
    r"."
    r"\manuscript_outputs\figures\data\R5_yolo_attn_full_test_eval"
)
OUT_CSV = os.path.join(OUT_DIR, f"{VIDEO}_eval.csv")
PIPELINE_NAME = "R5_yolo_attn_ep10"


# ---------------------------------------------------------------------------
# 1. Parse Bento .annot  (verbatim)
# ---------------------------------------------------------------------------
def parse_bento_annot(path, total_frames, fps):
    bouts_by_class = {}
    current_class = None
    in_data = False

    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            stripped = line.strip()

            if stripped.startswith(">"):
                current_class = stripped[1:].strip()
                bouts_by_class[current_class] = []
                in_data = False
                continue

            if "Start" in stripped and "Stop" in stripped:
                in_data = True
                continue

            if in_data and current_class is not None and stripped:
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        start_s = float(parts[0])
                        stop_s = float(parts[1])
                        start_f = int(round(start_s * fps))
                        end_f = int(round(stop_s * fps))
                        bouts_by_class[current_class].append((start_f, end_f))
                    except ValueError:
                        pass

    gt_frames = np.full(total_frames, CLASS_MAP["other"], dtype=int)
    for cname, bouts in bouts_by_class.items():
        cid = CLASS_MAP.get(cname, None)
        if cid is None:
            continue
        for sf, ef in bouts:
            sf = max(sf, 0)
            ef = min(ef, total_frames - 1)
            gt_frames[sf : ef + 1] = cid

    return bouts_by_class, gt_frames


# ---------------------------------------------------------------------------
# 2. Aggregation helpers (verbatim)
# ---------------------------------------------------------------------------
def aggregate_window_predictions(csv_path, total_frames):
    df = pd.read_csv(csv_path)
    prob_cols = ["prob_attack", "prob_investigation", "prob_mount", "prob_other"]

    prob_sum = np.zeros((total_frames, 4), dtype=np.float64)
    count = np.zeros(total_frames, dtype=np.int32)

    for _, row in df.iterrows():
        fs = int(row["frame_start"])
        fe = int(row["frame_end"])
        probs = np.array([row[c] for c in prob_cols], dtype=np.float64)
        fs = max(fs, 0)
        fe = min(fe, total_frames - 1)
        prob_sum[fs : fe + 1] += probs
        count[fs : fe + 1] += 1

    covered = count > 0
    per_frame_pred = np.full(total_frames, CLASS_MAP["other"], dtype=int)
    if covered.any():
        avg_probs = prob_sum[covered] / count[covered, np.newaxis]
        per_frame_pred[covered] = np.argmax(avg_probs, axis=1)

    stats = {
        "n_windows": len(df),
        "covered_frames": int(covered.sum()),
        "uncovered_frames": int((~covered).sum()),
        "min_windows": int(count[covered].min()) if covered.any() else 0,
        "max_windows": int(count[covered].max()) if covered.any() else 0,
        "mean_windows": float(count[covered].mean()) if covered.any() else 0.0,
    }
    return per_frame_pred, stats


def load_smoothed_predictions(csv_path, total_frames):
    df = pd.read_csv(csv_path)
    per_frame_pred = np.full(total_frames, CLASS_MAP["other"], dtype=int)
    covered = np.zeros(total_frames, dtype=bool)

    for _, row in df.iterrows():
        fi = int(row["frame_idx"])
        if 0 <= fi < total_frames:
            per_frame_pred[fi] = int(row["predicted_class_id"])
            covered[fi] = True

    stats = {
        "n_rows": len(df),
        "covered_frames": int(covered.sum()),
        "uncovered_frames": int((~covered).sum()),
    }
    return per_frame_pred, stats


# ---------------------------------------------------------------------------
# 3. Frame-level metrics
# ---------------------------------------------------------------------------
def compute_frame_metrics(gt, pred):
    labels = list(range(4))
    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
        gt, pred, labels=labels, average=None, zero_division=0
    )
    acc = accuracy_score(gt, pred)
    macro_f1 = float(np.mean(f1_arr))

    results = {}
    for i, lab in enumerate(labels):
        results[ID_TO_CLASS[lab]] = {
            "precision": float(prec_arr[i]),
            "recall": float(rec_arr[i]),
            "f1": float(f1_arr[i]),
        }
    results["__accuracy__"] = float(acc)
    results["__macro_f1__"] = macro_f1
    return results


# ---------------------------------------------------------------------------
# 4. Bout-level metrics
# ---------------------------------------------------------------------------
def extract_bouts(frame_array, class_id):
    bouts = []
    in_bout = False
    start = None
    for i, c in enumerate(frame_array):
        if c == class_id:
            if not in_bout:
                in_bout = True
                start = i
        else:
            if in_bout:
                bouts.append((start, i - 1))
                in_bout = False
    if in_bout:
        bouts.append((start, len(frame_array) - 1))
    return bouts


def tiou(pred_bout, gt_bout):
    ps, pe = pred_bout
    gs, ge = gt_bout
    inter = max(0, min(pe, ge) - max(ps, gs) + 1)
    union = (pe - ps + 1) + (ge - gs + 1) - inter
    return inter / union if union > 0 else 0.0


def bout_metrics(pred_bouts, gt_bouts, threshold):
    tp = 0
    fp = 0
    matched_gt = set()
    matched_pairs = []

    for pb in pred_bouts:
        best_iou = 0.0
        best_gi = -1
        for gi, gb in enumerate(gt_bouts):
            if gi in matched_gt:
                continue
            iou_val = tiou(pb, gb)
            if iou_val > best_iou:
                best_iou = iou_val
                best_gi = gi
        if best_iou >= threshold and best_gi >= 0:
            tp += 1
            matched_gt.add(best_gi)
            matched_pairs.append((pb, gt_bouts[best_gi]))
        else:
            fp += 1

    fn = len(gt_bouts) - len(matched_gt)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, matched_pairs


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------
def evaluate_pipeline(name, per_frame_pred, gt_frames, gt_bouts_by_id):
    print(f"\n{'=' * 70}")
    print(f"Pipeline: {name}")
    print('=' * 70)

    fm = compute_frame_metrics(gt_frames, per_frame_pred)
    print(f"\n  Frame-level metrics:")
    print(f"    {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"    {'-' * 47}")
    for cname in ["attack", "investigation", "mount", "other"]:
        m = fm[cname]
        print(
            f"    {cname:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}"
        )
    print(f"    {'-' * 47}")
    print(f"    {'Accuracy':<15} {fm['__accuracy__']:>10.4f}")
    print(f"    {'Macro F1 (4 cls)':<15} {fm['__macro_f1__']:>10.4f}")

    print(f"\n  Bout-level metrics (tIoU=0.25, 0.50):")
    bout_results = {}
    f1_25_vals, f1_50_vals = [], []

    for cid in EVAL_CLASSES:
        cname = ID_TO_CLASS[cid]
        pred_bouts = extract_bouts(per_frame_pred, cid)
        gt_bouts = gt_bouts_by_id.get(cid, [])

        p25, r25, f25, pairs25 = bout_metrics(pred_bouts, gt_bouts, 0.25)
        p50, r50, f50, pairs50 = bout_metrics(pred_bouts, gt_bouts, 0.50)

        bout_results[cid] = {
            "iou25": {"prec": p25, "rec": r25, "f1": f25},
            "iou50": {"prec": p50, "rec": r50, "f1": f50},
            "pairs50": pairs50,
            "n_pred_bouts": len(pred_bouts),
            "n_gt_bouts": len(gt_bouts),
        }
        # Macro: per spec, any class with 0 GT bouts contributes zero (i.e. F1=0)
        # bout_metrics already returns F1=0 when there are 0 GT bouts (rec=0).
        f1_25_vals.append(f25)
        f1_50_vals.append(f50)

        print(
            f"    {cname:<15}  GT_bouts={len(gt_bouts)}  Pred_bouts={len(pred_bouts)}"
        )
        print(f"      tIoU=0.25  P={p25:.4f}  R={r25:.4f}  F1={f25:.4f}")
        print(f"      tIoU=0.50  P={p50:.4f}  R={r50:.4f}  F1={f50:.4f}")

    macro25 = float(np.mean(f1_25_vals)) if f1_25_vals else 0.0
    macro50 = float(np.mean(f1_50_vals)) if f1_50_vals else 0.0
    print(f"    Macro Bout F1 (att+inv+mount) tIoU=0.25 : {macro25:.4f}")
    print(f"    Macro Bout F1 (att+inv+mount) tIoU=0.50 : {macro50:.4f}")

    n_mount_pred = bout_results[CLASS_MAP["mount"]]["n_pred_bouts"]
    print(f"\n  Mount predicted bouts: {n_mount_pred} (GT mount bouts: {bout_results[CLASS_MAP['mount']]['n_gt_bouts']})")

    print(f"\n  Mean boundary offsets (tIoU=0.50 matched pairs):")
    offsets = {}
    for cid in EVAL_CLASSES:
        cname = ID_TO_CLASS[cid]
        pairs = bout_results[cid]["pairs50"]
        if pairs:
            mean_start = float(np.mean([abs(pb[0] - gb[0]) for pb, gb in pairs]))
            mean_end = float(np.mean([abs(pb[1] - gb[1]) for pb, gb in pairs]))
        else:
            mean_start = float("nan")
            mean_end = float("nan")
        offsets[cid] = {"mean_start": mean_start, "mean_end": mean_end}
        print(
            f"    {cname:<15}  matched_pairs={len(pairs)}  "
            f"mean_start_offset={mean_start:.2f}  mean_end_offset={mean_end:.2f}"
        )

    return {
        "frame": fm,
        "bout": bout_results,
        "offsets": offsets,
        "macro_bout_f1_25": macro25,
        "macro_bout_f1_50": macro50,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print(f"STEP 1 - Parsing Bento annotation for {VIDEO}")
    print("=" * 70)
    bouts_by_class, gt_frames = parse_bento_annot(ANNOT_PATH, TOTAL_FRAMES, FPS)
    for cname, bouts in bouts_by_class.items():
        print(f"  {cname}: {len(bouts)} bouts")

    gt_bouts_by_id = {}
    for cname, bouts in bouts_by_class.items():
        cid = CLASS_MAP.get(cname)
        if cid is not None:
            gt_bouts_by_id[cid] = bouts
    for cid in EVAL_CLASSES:
        gt_bouts_by_id.setdefault(cid, [])

    print(f"\n  GT frame distribution:")
    for cid, cname in ID_TO_CLASS.items():
        n = int(np.sum(gt_frames == cid))
        print(f"    {cname:<15} (id={cid}): {n} frames ({100 * n / TOTAL_FRAMES:.1f}%)")

    print("\n" + "=" * 70)
    print("STEP 2a - Loading raw windows and aggregating to per-frame")
    print("=" * 70)
    y_raw_pred, y_raw_stats = aggregate_window_predictions(Y_RAW_CSV, TOTAL_FRAMES)
    print(f"  Windows           : {y_raw_stats['n_windows']}")
    print(f"  Covered frames    : {y_raw_stats['covered_frames']} / {TOTAL_FRAMES}")
    print(f"  Uncovered frames  : {y_raw_stats['uncovered_frames']}")
    print(f"  Windows/frame min : {y_raw_stats['min_windows']}")
    print(f"  Windows/frame max : {y_raw_stats['max_windows']}")
    print(f"  Windows/frame mean: {y_raw_stats['mean_windows']:.3f}")

    print("\n" + "=" * 70)
    print("STEP 2b - Loading smoothed per-frame predictions")
    print("=" * 70)
    y_sm_pred, y_sm_stats = load_smoothed_predictions(Y_SMOOTHED_CSV, TOTAL_FRAMES)
    print(f"  Rows              : {y_sm_stats['n_rows']}")
    print(f"  Covered frames    : {y_sm_stats['covered_frames']} / {TOTAL_FRAMES}")
    print(f"  Uncovered frames  : {y_sm_stats['uncovered_frames']}")

    y_raw_results = evaluate_pipeline(
        "raw windows (prob-avg argmax)", y_raw_pred, gt_frames, gt_bouts_by_id
    )
    y_sm_results = evaluate_pipeline(
        "smoothed per-frame", y_sm_pred, gt_frames, gt_bouts_by_id
    )

    # ---- Summary table ----
    print("\n" + "=" * 70)
    print("STEP 3 - Side-by-side: raw windows vs smoothed frames")
    print("=" * 70)

    header = f"  {'Metric':<28}{'attack':>10}  {'investigation':>14}  {'mount':>10}"

    def metric_row(metric_key, results):
        return {
            "attack": f"{results['frame']['attack'][metric_key]:.4f}",
            "investigation": f"{results['frame']['investigation'][metric_key]:.4f}",
            "mount": f"{results['frame']['mount'][metric_key]:.4f}",
        }

    def bout_row(iou_key, results):
        return {
            "attack": f"{results['bout'][0][iou_key]['f1']:.4f}",
            "investigation": f"{results['bout'][1][iou_key]['f1']:.4f}",
            "mount": f"{results['bout'][2][iou_key]['f1']:.4f}",
        }

    def fmt_row(label, vals):
        return (
            f"  {label:<28}"
            f"{vals['attack']:>10}  "
            f"{vals['investigation']:>14}  "
            f"{vals['mount']:>10}"
        )

    for label, results in [
        ("raw windows", y_raw_results),
        ("smoothed frames", y_sm_results),
    ]:
        print(f"\n  --- {label} ---")
        print(header)
        print(fmt_row("Frame Precision", metric_row("precision", results)))
        print(fmt_row("Frame Recall", metric_row("recall", results)))
        print(fmt_row("Frame F1", metric_row("f1", results)))
        print(fmt_row("Bout F1 tIoU=0.25", bout_row("iou25", results)))
        print(fmt_row("Bout F1 tIoU=0.50", bout_row("iou50", results)))
        print(f"    Mount pred bout count    : {results['bout'][2]['n_pred_bouts']}  (GT={results['bout'][2]['n_gt_bouts']})")
        print(f"    Macro bout F1 (3 cls) @0.25 : {results['macro_bout_f1_25']:.4f}")
        print(f"    Macro bout F1 (3 cls) @0.50 : {results['macro_bout_f1_50']:.4f}")

    # ---- Head-to-head macro (att+inv only, comparable to smoke result) ----
    print("\n  Head-to-head (macro F1 over attack + investigation, smoke-comparable):")
    print(f"    {'Pipeline':<28}{'frame_macroF1':>16}{'bout_F1@0.25':>16}{'bout_F1@0.50':>16}")
    def frame_macro_attinv(res):
        return (res["frame"]["attack"]["f1"] + res["frame"]["investigation"]["f1"]) / 2
    def bout_macro_attinv(res, key):
        return (res["bout"][0][key]["f1"] + res["bout"][1][key]["f1"]) / 2
    for label, res in [("raw windows", y_raw_results), ("smoothed frames", y_sm_results)]:
        print(
            f"    {label:<28}"
            f"{frame_macro_attinv(res):>16.4f}"
            f"{bout_macro_attinv(res, 'iou25'):>16.4f}"
            f"{bout_macro_attinv(res, 'iou50'):>16.4f}"
        )

    # ---- Save CSV ----
    print("\n" + "=" * 70)
    print("STEP 4 - Saving results CSV")
    print("=" * 70)
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    for pipeline_label, source_label, results in [
        ("R5_yolo_attn_ep10 raw", "raw_windows", y_raw_results),
        ("R5_yolo_attn_ep10 smoothed", "smoothed_frames", y_sm_results),
    ]:
        for cid in EVAL_CLASSES:
            cname = ID_TO_CLASS[cid]
            fm = results["frame"][cname]
            br = results["bout"][cid]
            of = results["offsets"][cid]
            rows.append({
                "pipeline": pipeline_label,
                "source": source_label,
                "video": VIDEO,
                "class": cname,
                "frame_precision": round(fm["precision"], 6),
                "frame_recall": round(fm["recall"], 6),
                "frame_f1": round(fm["f1"], 6),
                "bout_f1_iou25": round(br["iou25"]["f1"], 6),
                "bout_f1_iou50": round(br["iou50"]["f1"], 6),
                "mean_start_offset_frames": (
                    round(of["mean_start"], 4) if not np.isnan(of["mean_start"]) else float("nan")
                ),
                "mean_end_offset_frames": (
                    round(of["mean_end"], 4) if not np.isnan(of["mean_end"]) else float("nan")
                ),
                "predicted_bouts": br["n_pred_bouts"],
                "gt_bouts": br["n_gt_bouts"],
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"  Saved to: {OUT_CSV}")
    print()
    print(out_df.to_string(index=False))

    print("\nDONE")


if __name__ == "__main__":
    main()
