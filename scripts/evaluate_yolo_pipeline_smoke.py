"""
evaluate_yolo_pipeline_smoke.py

Evaluate BehaviorScope-Y inference (raw windows + smoothed per-frame) against
MARS ground truth for Mouse155_20151124_20-56-05, mirroring the metric
definitions in scripts/evaluate_stride_comparison.py. Compare side-by-side
against the BehaviorScope-Y stride-16 raw baseline already on disk.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# ---------------------------------------------------------------------------
# Constants (mirrored from evaluate_stride_comparison.py)
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

Y_RAW_CSV = (
    r".\BehaviorScope"
    r"\behavior_lstm_runs\R5_yolo_attn_smoke_ep10_inference"
    r"\Mouse155_20151124_20-56-05_yolo_smoke_ep10.csv"
)
Y_SMOOTHED_CSV = (
    r".\BehaviorScope"
    r"\behavior_lstm_runs\R5_yolo_attn_smoke_ep10_inference"
    r"\Mouse155_20151124_20-56-05_yolo_smoke_ep10.smoothed_frames.csv"
)
N_RESULTS_CSV = (
    r"."
    r"\research_notebook\figures\data\stride_comparison_results.csv"
)

OUT_DIR = (
    r"."
    r"\manuscript_outputs\figures\data"
)
OUT_CSV = os.path.join(OUT_DIR, "R5_yolo_attn_smoke_ep10_inference_eval.csv")


# ---------------------------------------------------------------------------
# 1. Parse Bento .annot  (verbatim from evaluate_stride_comparison.py)
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
# 2. Aggregate window predictions to per-frame  (verbatim methodology)
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
    """The smoothed CSV is already per-frame (frame_start == frame_end == frame_idx)."""
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

    # Frame-level
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
    print(f"    {'Macro F1':<15} {fm['__macro_f1__']:>10.4f}")

    # Bout-level
    print(f"\n  Bout-level metrics (tIoU=0.25, 0.50):")
    bout_results = {}
    f1_25_vals, f1_50_vals = [], []
    # Eval over attack + investigation only for macro (mount has 0 GT bouts)
    macro_classes = [0, 1]

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
        if cid in macro_classes:
            f1_25_vals.append(f25)
            f1_50_vals.append(f50)

        print(
            f"    {cname:<15}  GT_bouts={len(gt_bouts)}  Pred_bouts={len(pred_bouts)}"
        )
        print(f"      tIoU=0.25  P={p25:.4f}  R={r25:.4f}  F1={f25:.4f}")
        print(f"      tIoU=0.50  P={p50:.4f}  R={r50:.4f}  F1={f50:.4f}")

    macro25 = float(np.mean(f1_25_vals)) if f1_25_vals else 0.0
    macro50 = float(np.mean(f1_50_vals)) if f1_50_vals else 0.0
    print(f"    {'Macro Bout F1 (att+inv) tIoU=0.25':<35}: {macro25:.4f}")
    print(f"    {'Macro Bout F1 (att+inv) tIoU=0.50':<35}: {macro50:.4f}")

    # Mount FP count
    n_mount_pred = bout_results[CLASS_MAP["mount"]]["n_pred_bouts"]
    print(f"\n  Mount predicted bouts (false positives, GT mount = 0): {n_mount_pred}")

    # Boundary offsets at tIoU 0.50
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
    print("STEP 1 - Parsing Bento annotation")
    print("=" * 70)
    bouts_by_class, gt_frames = parse_bento_annot(ANNOT_PATH, TOTAL_FRAMES, FPS)
    for cname, bouts in bouts_by_class.items():
        print(f"  {cname}: {len(bouts)} bouts")

    gt_bouts_by_id = {}
    for cname, bouts in bouts_by_class.items():
        cid = CLASS_MAP.get(cname)
        if cid is not None:
            gt_bouts_by_id[cid] = bouts
    # Ensure mount key exists with empty list (no GT mount)
    for cid in EVAL_CLASSES:
        gt_bouts_by_id.setdefault(cid, [])

    print(f"\n  GT frame distribution:")
    for cid, cname in ID_TO_CLASS.items():
        n = int(np.sum(gt_frames == cid))
        print(f"    {cname:<15} (id={cid}): {n} frames ({100 * n / TOTAL_FRAMES:.1f}%)")

    # ---- Y raw windows ----
    print("\n" + "=" * 70)
    print("STEP 2a - Loading Y raw windows and aggregating to per-frame")
    print("=" * 70)
    y_raw_pred, y_raw_stats = aggregate_window_predictions(Y_RAW_CSV, TOTAL_FRAMES)
    print(f"  Windows           : {y_raw_stats['n_windows']}")
    print(f"  Covered frames    : {y_raw_stats['covered_frames']} / {TOTAL_FRAMES}")
    print(f"  Uncovered frames  : {y_raw_stats['uncovered_frames']}")
    print(f"  Windows/frame min : {y_raw_stats['min_windows']}")
    print(f"  Windows/frame max : {y_raw_stats['max_windows']}")
    print(f"  Windows/frame mean: {y_raw_stats['mean_windows']:.3f}")

    # ---- Y smoothed per-frame ----
    print("\n" + "=" * 70)
    print("STEP 2b - Loading Y smoothed per-frame predictions")
    print("=" * 70)
    y_sm_pred, y_sm_stats = load_smoothed_predictions(Y_SMOOTHED_CSV, TOTAL_FRAMES)
    print(f"  Rows              : {y_sm_stats['n_rows']}")
    print(f"  Covered frames    : {y_sm_stats['covered_frames']} / {TOTAL_FRAMES}")
    print(f"  Uncovered frames  : {y_sm_stats['uncovered_frames']}")

    # ---- Evaluate ----
    y_raw_results = evaluate_pipeline(
        "Y raw windows (prob-avg argmax)", y_raw_pred, gt_frames, gt_bouts_by_id
    )
    y_sm_results = evaluate_pipeline(
        "Y smoothed per-frame (smooth=5, bout_min=15)",
        y_sm_pred,
        gt_frames,
        gt_bouts_by_id,
    )

    # ---- Side-by-side comparison vs N stride-16 ----
    print("\n" + "=" * 70)
    print("STEP 3 - Side-by-side: Y smoothed | Y raw | N stride-16 raw")
    print("=" * 70)

    n_df = pd.read_csv(N_RESULTS_CSV)
    n_st16 = n_df[n_df["stride"] == "stride-16"].set_index("class")

    def fmt_row(label, vals):
        return (
            f"  {label:<24}"
            f"{vals['attack']:>10}  "
            f"{vals['investigation']:>14}  "
            f"{vals['mount']:>10}"
        )

    header = f"  {'Metric':<24}{'attack':>10}  {'investigation':>14}  {'mount':>10}"

    def metric_row(metric_key, results):
        return {
            "attack": f"{results['frame'][ 'attack' ][metric_key]:.4f}",
            "investigation": f"{results['frame'][ 'investigation' ][metric_key]:.4f}",
            "mount": f"{results['frame'][ 'mount' ][metric_key]:.4f}",
        }

    def bout_row(iou_key, results):
        return {
            "attack": f"{results['bout'][0][iou_key]['f1']:.4f}",
            "investigation": f"{results['bout'][1][iou_key]['f1']:.4f}",
            "mount": f"{results['bout'][2][iou_key]['f1']:.4f}",
        }

    def n_metric_row(col):
        return {
            "attack": f"{n_st16.loc['attack', col]:.4f}",
            "investigation": f"{n_st16.loc['investigation', col]:.4f}",
            "mount": f"{n_st16.loc['mount', col]:.4f}",
        }

    for label, results in [
        ("Y smoothed", y_sm_results),
        ("Y raw windows", y_raw_results),
    ]:
        print(f"\n  --- {label} ---")
        print(header)
        print(fmt_row("Frame Precision", metric_row("precision", results)))
        print(fmt_row("Frame Recall", metric_row("recall", results)))
        print(fmt_row("Frame F1", metric_row("f1", results)))
        print(fmt_row("Bout F1 tIoU=0.25", bout_row("iou25", results)))
        print(fmt_row("Bout F1 tIoU=0.50", bout_row("iou50", results)))
        n_pred_mount = results["bout"][2]["n_pred_bouts"]
        print(f"    Mount FP bout count   : {n_pred_mount}")
        print(f"    Macro bout F1 (att+inv) @0.25 : {results['macro_bout_f1_25']:.4f}")
        print(f"    Macro bout F1 (att+inv) @0.50 : {results['macro_bout_f1_50']:.4f}")

    print(f"\n  --- N stride-16 raw (from existing CSV) ---")
    print(header)
    print(fmt_row("Frame Precision", n_metric_row("frame_precision")))
    print(fmt_row("Frame Recall", n_metric_row("frame_recall")))
    print(fmt_row("Frame F1", n_metric_row("frame_f1")))
    print(fmt_row("Bout F1 tIoU=0.25", n_metric_row("bout_f1_iou25")))
    print(fmt_row("Bout F1 tIoU=0.50", n_metric_row("bout_f1_iou50")))
    n_macro25 = (n_st16.loc["attack", "bout_f1_iou25"] + n_st16.loc["investigation", "bout_f1_iou25"]) / 2
    n_macro50 = (n_st16.loc["attack", "bout_f1_iou50"] + n_st16.loc["investigation", "bout_f1_iou50"]) / 2
    print(f"    Macro bout F1 (att+inv) @0.25 : {n_macro25:.4f}")
    print(f"    Macro bout F1 (att+inv) @0.50 : {n_macro50:.4f}")

    # Compact head-to-head macro table
    print("\n  Head-to-head (macro F1 over attack + investigation):")
    print(f"    {'Pipeline':<28}{'frame_macroF1':>16}{'bout_F1@0.25':>16}{'bout_F1@0.50':>16}")
    def frame_macro_attinv(res):
        return (res["frame"]["attack"]["f1"] + res["frame"]["investigation"]["f1"]) / 2
    print(
        f"    {'Y smoothed':<28}"
        f"{frame_macro_attinv(y_sm_results):>16.4f}"
        f"{y_sm_results['macro_bout_f1_25']:>16.4f}"
        f"{y_sm_results['macro_bout_f1_50']:>16.4f}"
    )
    print(
        f"    {'Y raw windows':<28}"
        f"{frame_macro_attinv(y_raw_results):>16.4f}"
        f"{y_raw_results['macro_bout_f1_25']:>16.4f}"
        f"{y_raw_results['macro_bout_f1_50']:>16.4f}"
    )
    n_frame_macro = (n_st16.loc["attack", "frame_f1"] + n_st16.loc["investigation", "frame_f1"]) / 2
    print(
        f"    {'N stride-16 raw':<28}"
        f"{n_frame_macro:>16.4f}"
        f"{n_macro25:>16.4f}"
        f"{n_macro50:>16.4f}"
    )

    # ---- Save CSV ----
    print("\n" + "=" * 70)
    print("STEP 4 - Saving results CSV")
    print("=" * 70)
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    for pipeline_label, source_label, results in [
        ("Y smoothed", "smoothed_frames", y_sm_results),
        ("Y raw windows", "raw_windows", y_raw_results),
    ]:
        for cid in EVAL_CLASSES:
            cname = ID_TO_CLASS[cid]
            fm = results["frame"][cname]
            br = results["bout"][cid]
            of = results["offsets"][cid]
            rows.append({
                "pipeline": pipeline_label,
                "source": source_label,
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
            })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"  Saved to: {OUT_CSV}")
    print()
    print(out_df.to_string(index=False))

    print("\nDONE")


if __name__ == "__main__":
    main()
