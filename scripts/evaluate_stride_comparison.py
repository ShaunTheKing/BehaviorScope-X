"""
evaluate_stride_comparison.py

Compare three stride inference CSVs against MARS ground-truth annotations
for Mouse155_20151124_20-56-05.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TOTAL_FRAMES = 10390
FPS = 30.0
CLASS_MAP = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
EVAL_CLASSES = [0, 1, 2]  # attack, investigation, mount  (not "other")

ANNOT_PATH = (
    r".\MARS_data\Mouse155_20151124_20-56-05"
    r"\Mouse155_20151124_20-56-05_bhvr.annot"
)

STRIDE_FILES = {
    "stride-16": (
        r".\BehaviorScope"
        r"\behavior_lstm_runs\R5_Inference_Stage2_frozen"
        r"\Mouse155_20151124_20-56-05_stage2.csv"
    ),
    "stride-8": (
        r".\BehaviorScope"
        r"\behavior_lstm_runs\R5_Inference_Stage2_frozen_st8"
        r"\Mouse155_20151124_20-56-05_stage2_st8.csv"
    ),
    "stride-4": (
        r".\BehaviorScope"
        r"\behavior_lstm_runs\R5_Inference_Stage2_frozen_st4"
        r"\Mouse155_20151124_20-56-05_stage2_st4.csv"
    ),
}

RESULTS_CSV = (
    r".\scripts"
    r"\stride_comparison_results.csv"
)

# ---------------------------------------------------------------------------
# 1. Parse Bento .annot
# ---------------------------------------------------------------------------
def parse_bento_annot(path, total_frames, fps):
    """
    Returns:
        bouts_by_class : dict  class_name -> list of (start_frame, end_frame)
        gt_frames      : np.ndarray shape (total_frames,) dtype int, default 3
    """
    bouts_by_class = {}
    current_class = None
    in_data = False

    with open(path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            stripped = line.strip()

            # Section header
            if stripped.startswith(">"):
                current_class = stripped[1:].strip()
                bouts_by_class[current_class] = []
                in_data = False
                continue

            # Column header row
            if "Start" in stripped and "Stop" in stripped:
                in_data = True
                continue

            # Data row (two or three tab-separated numbers)
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

    # Build per-frame GT array
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


print("=" * 70)
print("STEP 1 — Parsing Bento annotation")
print("=" * 70)
bouts_by_class, gt_frames = parse_bento_annot(ANNOT_PATH, TOTAL_FRAMES, FPS)
for cname, bouts in bouts_by_class.items():
    print(f"  {cname}: {len(bouts)} bouts")

# Also build GT bouts dict keyed by class_id (for bout evaluation)
gt_bouts_by_id = {}
for cname, bouts in bouts_by_class.items():
    cid = CLASS_MAP.get(cname)
    if cid is not None:
        gt_bouts_by_id[cid] = bouts

print(f"\nGT frame distribution:")
for cid, cname in ID_TO_CLASS.items():
    n = int(np.sum(gt_frames == cid))
    print(f"  {cname} (id={cid}): {n} frames ({100*n/TOTAL_FRAMES:.1f}%)")


# ---------------------------------------------------------------------------
# 2. Aggregate window predictions to per-frame
# ---------------------------------------------------------------------------
def aggregate_predictions(csv_path, total_frames):
    """
    Returns per_frame_pred (int array), coverage stats.
    """
    df = pd.read_csv(csv_path)
    prob_cols = ["prob_attack", "prob_investigation", "prob_mount", "prob_other"]

    # Accumulate sum of prob vectors and count per frame
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

    # Frames covered by at least one window
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


print("\n" + "=" * 70)
print("STEP 2 — Aggregating window predictions to per-frame")
print("=" * 70)

stride_preds = {}
for stride_name, csv_path in STRIDE_FILES.items():
    pred, stats = aggregate_predictions(csv_path, TOTAL_FRAMES)
    stride_preds[stride_name] = pred
    print(f"\n  {stride_name}:")
    print(f"    Windows           : {stats['n_windows']}")
    print(f"    Covered frames    : {stats['covered_frames']} / {TOTAL_FRAMES}")
    print(f"    Uncovered frames  : {stats['uncovered_frames']}")
    print(f"    Windows/frame min : {stats['min_windows']}")
    print(f"    Windows/frame max : {stats['max_windows']}")
    print(f"    Windows/frame mean: {stats['mean_windows']:.3f}")


# ---------------------------------------------------------------------------
# 3. Frame-level metrics
# ---------------------------------------------------------------------------
def compute_frame_metrics(gt, pred, class_ids, id_to_class):
    """Returns dict: class_name -> {precision, recall, f1}"""
    labels = list(range(4))  # all 4 classes for overall accuracy
    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
        gt, pred, labels=labels, average=None, zero_division=0
    )
    acc = accuracy_score(gt, pred)
    macro_f1 = float(np.mean(f1_arr))

    results = {}
    for i, lab in enumerate(labels):
        results[id_to_class[lab]] = {
            "precision": prec_arr[i],
            "recall": rec_arr[i],
            "f1": f1_arr[i],
        }
    results["__accuracy__"] = acc
    results["__macro_f1__"] = macro_f1
    return results


print("\n" + "=" * 70)
print("STEP 3 — Frame-level metrics")
print("=" * 70)

frame_metrics_all = {}
for stride_name, pred in stride_preds.items():
    fm = compute_frame_metrics(gt_frames, pred, EVAL_CLASSES, ID_TO_CLASS)
    frame_metrics_all[stride_name] = fm

    print(f"\n  {stride_name}:")
    print(f"    {'Class':<15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"    {'-'*47}")
    for cname in ["attack", "investigation", "mount", "other"]:
        m = fm[cname]
        print(
            f"    {cname:<15} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}"
        )
    print(f"    {'-'*47}")
    print(f"    {'Accuracy':<15} {fm['__accuracy__']:>10.4f}")
    print(f"    {'Macro F1':<15} {fm['__macro_f1__']:>10.4f}")


# ---------------------------------------------------------------------------
# 4. Bout-level metrics
# ---------------------------------------------------------------------------
def extract_bouts(frame_array, class_id):
    """Extract consecutive runs of class_id; return list of (start, end)."""
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
    """
    Greedy matching. Returns (precision, recall, f1, matched_pairs).
    matched_pairs: list of (pred_bout, gt_bout) for TP at this threshold.
    """
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


print("\n" + "=" * 70)
print("STEP 4 — Bout-level metrics (tIoU = 0.25 and 0.50)")
print("=" * 70)

bout_results_all = {}  # stride -> class_id -> {iou25: {...}, iou50: {...}}

for stride_name, pred in stride_preds.items():
    print(f"\n  {stride_name}:")
    bout_results_all[stride_name] = {}

    macro_f1_25_vals = []
    macro_f1_50_vals = []

    for cid in EVAL_CLASSES:
        cname = ID_TO_CLASS[cid]
        pred_bouts = extract_bouts(pred, cid)
        gt_bouts = gt_bouts_by_id.get(cid, [])

        p25, r25, f25, pairs25 = bout_metrics(pred_bouts, gt_bouts, 0.25)
        p50, r50, f50, pairs50 = bout_metrics(pred_bouts, gt_bouts, 0.50)

        bout_results_all[stride_name][cid] = {
            "iou25": {"prec": p25, "rec": r25, "f1": f25},
            "iou50": {"prec": p50, "rec": r50, "f1": f50},
            "pairs50": pairs50,
            "n_pred_bouts": len(pred_bouts),
            "n_gt_bouts": len(gt_bouts),
        }
        macro_f1_25_vals.append(f25)
        macro_f1_50_vals.append(f50)

        print(
            f"    {cname:<15}  GT_bouts={len(gt_bouts)}  Pred_bouts={len(pred_bouts)}"
        )
        print(
            f"      tIoU=0.25  P={p25:.4f}  R={r25:.4f}  F1={f25:.4f}"
        )
        print(
            f"      tIoU=0.50  P={p50:.4f}  R={r50:.4f}  F1={f50:.4f}"
        )

    macro25 = float(np.mean(macro_f1_25_vals))
    macro50 = float(np.mean(macro_f1_50_vals))
    bout_results_all[stride_name]["__macro25__"] = macro25
    bout_results_all[stride_name]["__macro50__"] = macro50
    print(f"    {'Macro Bout F1 tIoU=0.25':<25}: {macro25:.4f}")
    print(f"    {'Macro Bout F1 tIoU=0.50':<25}: {macro50:.4f}")


# ---------------------------------------------------------------------------
# 5. Mean boundary offset (using tIoU 0.50 matched pairs)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 5 — Mean boundary offset (tIoU=0.50 matched pairs)")
print("=" * 70)

offset_results_all = {}  # stride -> class_id -> {mean_start, mean_end}

for stride_name in STRIDE_FILES:
    print(f"\n  {stride_name}:")
    offset_results_all[stride_name] = {}

    for cid in EVAL_CLASSES:
        cname = ID_TO_CLASS[cid]
        pairs = bout_results_all[stride_name][cid]["pairs50"]

        if pairs:
            start_offsets = [abs(pb[0] - gb[0]) for pb, gb in pairs]
            end_offsets = [abs(pb[1] - gb[1]) for pb, gb in pairs]
            mean_start = float(np.mean(start_offsets))
            mean_end = float(np.mean(end_offsets))
        else:
            mean_start = float("nan")
            mean_end = float("nan")

        offset_results_all[stride_name][cid] = {
            "mean_start": mean_start,
            "mean_end": mean_end,
        }
        print(
            f"    {cname:<15}  matched_pairs={len(pairs)}  "
            f"mean_start_offset={mean_start:.2f} frames  "
            f"mean_end_offset={mean_end:.2f} frames"
        )


# ---------------------------------------------------------------------------
# 6. Save results CSV
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6 — Saving results CSV")
print("=" * 70)

rows = []
for stride_name in STRIDE_FILES:
    for cid in EVAL_CLASSES:
        cname = ID_TO_CLASS[cid]
        fm = frame_metrics_all[stride_name][cname]
        br = bout_results_all[stride_name][cid]
        of = offset_results_all[stride_name][cid]

        rows.append(
            {
                "stride": stride_name,
                "class": cname,
                "frame_precision": round(fm["precision"], 6),
                "frame_recall": round(fm["recall"], 6),
                "frame_f1": round(fm["f1"], 6),
                "bout_f1_iou25": round(br["iou25"]["f1"], 6),
                "bout_f1_iou50": round(br["iou50"]["f1"], 6),
                "mean_start_offset_frames": round(of["mean_start"], 4)
                if not np.isnan(of["mean_start"])
                else float("nan"),
                "mean_end_offset_frames": round(of["mean_end"], 4)
                if not np.isnan(of["mean_end"])
                else float("nan"),
            }
        )

results_df = pd.DataFrame(rows)
results_df.to_csv(RESULTS_CSV, index=False)
print(f"  Saved to: {RESULTS_CSV}")
print()
print(results_df.to_string(index=False))

print("\nDONE")
