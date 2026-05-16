"""
threshold_sweep_eval.py

Post-hoc per-class confidence thresholding on existing BehaviorScope-Y test
predictions, re-evaluated against MARS ground truth. No retraining --- just a
quick test of whether thresholding can suppress false-positive bouts on videos
with zero ground-truth instances of a given class.

Pipeline:
    1. Load each video's RAW window predictions (.behavior.csv)
    2. Aggregate window probs to per-frame averaged prob vector
    3. Apply Rule A (MARS-like one-vs-rest):
           candidates = {c : avg_prob[c] > threshold[c]} for c in attack/inv/mount
           predicted = argmax(candidates) if non-empty else "other"
       (Note: the "Baseline" config uses argmax over all 4 classes, no threshold)
    4. Extract bouts and score against Bento .annot ground truth
       (greedy match, tIoU 0.25 and 0.50)
    5. Report pooled per-class P/R/F1 (frame), bout right/wrong/missed
       counts, and the hallucination metric (predicted bouts on
       zero-GT videos for that class).

NOTE: the existing per_video_eval.csv was computed from SMOOTHED predictions
(median filter + bout_min=15). This script's "Baseline" config operates on
unsmoothed window-prob argmax, so bout counts will differ (typically more,
shorter bouts). Frame-level metrics should be close. The comparison across
threshold configs is internally apples-to-apples.
"""

import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FPS = 30.0
CLASS_MAP = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
EVAL_CLASSES = [0, 1, 2]  # attack, investigation, mount

INFERENCE_ROOT = (
    r".\BehaviorScope"
    r"\behavior_lstm_runs\R5_yolo_attn_ep10_mars_test_eval"
)
GT_ROOT = r".\MARS_data"
SPLITS = ["test_1", "test_2"]

OUT_DIR = (
    r"."
    r"\manuscript_outputs\figures\data\R5_yolo_attn_full_test_eval"
)
OUT_CSV = os.path.join(OUT_DIR, "threshold_sweep_quick.csv")
EXISTING_PER_VIDEO_CSV = os.path.join(OUT_DIR, "per_video_eval.csv")

# ---------------------------------------------------------------------------
# Threshold configurations
#   None for a class means "no threshold" (used only for the Baseline config,
#   which falls back to argmax over all 4 classes including "other")
# ---------------------------------------------------------------------------
CONFIGS = [
    {
        "name": "Baseline",
        "rule": "argmax_all",          # argmax over all 4 classes incl. other
        "thresholds": {0: None, 1: None, 2: None},
    },
    {
        "name": "All-0.5",
        "rule": "rule_a",
        "thresholds": {0: 0.5, 1: 0.5, 2: 0.5},
    },
    {
        "name": "Attack-0.7",
        "rule": "rule_a",
        "thresholds": {0: 0.7, 1: 0.5, 2: 0.5},
    },
    {
        "name": "All-0.7",
        "rule": "rule_a",
        "thresholds": {0: 0.7, 1: 0.7, 2: 0.7},
    },
    {
        "name": "All-0.6",
        "rule": "rule_a",
        "thresholds": {0: 0.6, 1: 0.6, 2: 0.6},
    },
]


# ---------------------------------------------------------------------------
# Bento .annot parser  (verbatim from evaluate_full_model_test_set.py)
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
# Window-prob aggregation
# ---------------------------------------------------------------------------
def aggregate_window_probs(csv_path, total_frames):
    """Returns averaged-prob array (T, 4) and a `covered` boolean (T,)."""
    df = pd.read_csv(csv_path)
    prob_cols = ["prob_attack", "prob_investigation", "prob_mount", "prob_other"]

    prob_sum = np.zeros((total_frames, 4), dtype=np.float64)
    count = np.zeros(total_frames, dtype=np.int32)

    fs = df["frame_start"].astype(int).clip(lower=0, upper=total_frames - 1).to_numpy()
    fe = df["frame_end"].astype(int).clip(lower=0, upper=total_frames - 1).to_numpy()
    P = df[prob_cols].to_numpy(dtype=np.float64)

    for i in range(len(df)):
        a, b = fs[i], fe[i]
        prob_sum[a : b + 1] += P[i]
        count[a : b + 1] += 1

    covered = count > 0
    avg_probs = np.zeros((total_frames, 4), dtype=np.float64)
    if covered.any():
        avg_probs[covered] = prob_sum[covered] / count[covered, np.newaxis]
    return avg_probs, covered


def apply_rule(avg_probs, covered, rule, thresholds):
    """
    Returns per-frame predicted class id (T,).
    rule == 'argmax_all' : argmax over all 4 classes for covered frames
    rule == 'rule_a'     : among {attack,inv,mount} where p_c > thresh[c],
                            pick argmax; else "other"
    Uncovered frames default to "other".
    """
    T = avg_probs.shape[0]
    pred = np.full(T, CLASS_MAP["other"], dtype=int)
    if not covered.any():
        return pred

    if rule == "argmax_all":
        pred[covered] = np.argmax(avg_probs[covered], axis=1)
    elif rule == "rule_a":
        # Build threshold vector for the 3 evaluated classes
        thr = np.array([thresholds[0], thresholds[1], thresholds[2]], dtype=np.float64)
        sub = avg_probs[covered][:, :3]                     # (Nc, 3)
        passes = sub > thr[np.newaxis, :]                   # (Nc, 3)
        # Mask out classes that didn't pass by setting prob to -inf for argmax
        masked = np.where(passes, sub, -np.inf)             # (Nc, 3)
        any_pass = passes.any(axis=1)                       # (Nc,)
        chosen = np.argmax(masked, axis=1)                  # (Nc,) in {0,1,2}
        out = np.where(any_pass, chosen, CLASS_MAP["other"])
        pred[covered] = out
    else:
        raise ValueError(f"Unknown rule {rule!r}")
    return pred


# ---------------------------------------------------------------------------
# Bout helpers
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


def bout_counts(pred_bouts, gt_bouts, threshold):
    """Greedy match. Returns (right, wrong, missed) i.e. tp/fp/fn."""
    tp = 0
    fp = 0
    matched_gt = set()
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
        else:
            fp += 1
    fn = len(gt_bouts) - len(matched_gt)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Discovery / IO
# ---------------------------------------------------------------------------
def find_annot(split, video):
    pattern = os.path.join(GT_ROOT, split, video, f"{video}_*.annot")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def get_total_frames(behavior_csv):
    try:
        df = pd.read_csv(behavior_csv, usecols=["frame_end"])
        if len(df):
            return int(df["frame_end"].max()) + 1
    except Exception:
        pass
    return None


def discover_videos():
    out = []
    for split in SPLITS:
        split_dir = os.path.join(INFERENCE_ROOT, split)
        if not os.path.isdir(split_dir):
            continue
        for v in sorted(os.listdir(split_dir)):
            vd = os.path.join(split_dir, v)
            if not os.path.isdir(vd):
                continue
            csv = os.path.join(vd, f"{v}.behavior.csv")
            if os.path.exists(csv):
                out.append((split, v, csv))
            # silently skip missing
    return out


# ---------------------------------------------------------------------------
# Per-video eval for a single config
# ---------------------------------------------------------------------------
def evaluate_video_config(avg_probs, covered, gt_frames, gt_bouts_by_id,
                          rule, thresholds):
    pred_frames = apply_rule(avg_probs, covered, rule, thresholds)

    # Frame-level per-class TP/FP/FN
    frame_tp = np.zeros(4, dtype=np.int64)
    frame_fp = np.zeros(4, dtype=np.int64)
    frame_fn = np.zeros(4, dtype=np.int64)
    for cid in range(4):
        gt_pos = gt_frames == cid
        pr_pos = pred_frames == cid
        frame_tp[cid] = int(np.sum(gt_pos & pr_pos))
        frame_fp[cid] = int(np.sum(~gt_pos & pr_pos))
        frame_fn[cid] = int(np.sum(gt_pos & ~pr_pos))

    per_class = {}
    for cid in EVAL_CLASSES:
        pred_bouts = extract_bouts(pred_frames, cid)
        gt_bouts = gt_bouts_by_id.get(cid, [])
        tp25, fp25, fn25 = bout_counts(pred_bouts, gt_bouts, 0.25)
        tp50, fp50, fn50 = bout_counts(pred_bouts, gt_bouts, 0.50)
        per_class[cid] = {
            "frame_tp": int(frame_tp[cid]),
            "frame_fp": int(frame_fp[cid]),
            "frame_fn": int(frame_fn[cid]),
            "n_pred_bouts": len(pred_bouts),
            "n_gt_bouts": len(gt_bouts),
            "right_25": tp25, "wrong_25": fp25, "missed_25": fn25,
            "right_50": tp50, "wrong_50": fp50, "missed_50": fn50,
        }
    return per_class


# ---------------------------------------------------------------------------
# Aggregation across videos for one config
# ---------------------------------------------------------------------------
def pool_config(video_results):
    """video_results: list of per_class dicts (keyed by cid)"""
    pooled = {}
    for cid in EVAL_CLASSES:
        agg = {
            "frame_tp": 0, "frame_fp": 0, "frame_fn": 0,
            "n_pred_bouts": 0, "n_gt_bouts": 0,
            "right_25": 0, "wrong_25": 0, "missed_25": 0,
            "right_50": 0, "wrong_50": 0, "missed_50": 0,
            "halluc_pred_bouts_zero_gt": 0,
        }
        for vr in video_results:
            pc = vr[cid]
            for k in ["frame_tp", "frame_fp", "frame_fn",
                      "n_pred_bouts", "n_gt_bouts",
                      "right_25", "wrong_25", "missed_25",
                      "right_50", "wrong_50", "missed_50"]:
                agg[k] += pc[k]
            if pc["n_gt_bouts"] == 0:
                agg["halluc_pred_bouts_zero_gt"] += pc["n_pred_bouts"]

        tp, fp, fn = agg["frame_tp"], agg["frame_fp"], agg["frame_fn"]
        agg["frame_p"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        agg["frame_r"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        agg["frame_f1"] = (
            2 * agg["frame_p"] * agg["frame_r"] / (agg["frame_p"] + agg["frame_r"])
            if (agg["frame_p"] + agg["frame_r"]) > 0 else 0.0
        )
        pooled[cid] = agg
    return pooled


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 78)
    print("THRESHOLD SWEEP (post-hoc, no retraining)")
    print("=" * 78)
    print(f"  Inference root : {INFERENCE_ROOT}")
    print(f"  GT root        : {GT_ROOT}")
    print(f"  Splits         : {SPLITS}")
    print(f"  Output CSV     : {OUT_CSV}")

    videos = discover_videos()
    print(f"\n  Discovered video CSVs: {len(videos)}")

    # Pre-load per-video data once (probs aggregated to per-frame, GT)
    cached = []   # list of dicts: video, split, avg_probs, covered, gt_frames, gt_bouts_by_id
    skipped = []
    for split, video, csv_path in videos:
        annot = find_annot(split, video)
        if annot is None:
            skipped.append((split, video, "no_annot"))
            continue
        total_frames = get_total_frames(csv_path)
        if total_frames is None or total_frames <= 0:
            skipped.append((split, video, "no_total_frames"))
            continue
        bouts_by_class, gt_frames = parse_bento_annot(annot, total_frames, FPS)
        gt_bouts_by_id = {cid: [] for cid in EVAL_CLASSES + [3]}
        for cname, bouts in bouts_by_class.items():
            cid = CLASS_MAP.get(cname)
            if cid is not None:
                gt_bouts_by_id[cid] = bouts
        avg_probs, covered = aggregate_window_probs(csv_path, total_frames)
        cached.append({
            "split": split,
            "video": video,
            "avg_probs": avg_probs,
            "covered": covered,
            "gt_frames": gt_frames,
            "gt_bouts_by_id": gt_bouts_by_id,
        })
    print(f"  Loaded         : {len(cached)} videos")
    if skipped:
        print(f"  Skipped        : {len(skipped)}")
        for s, v, why in skipped:
            print(f"    [{s}] {v}: {why}")

    # Run all configs
    all_results = {}   # config_name -> pooled dict
    for cfg in CONFIGS:
        per_video = []
        for c in cached:
            per_video.append(evaluate_video_config(
                c["avg_probs"], c["covered"], c["gt_frames"],
                c["gt_bouts_by_id"], cfg["rule"], cfg["thresholds"],
            ))
        all_results[cfg["name"]] = pool_config(per_video)

    # ---- Sanity check vs existing per_video_eval.csv ----
    print("\n" + "=" * 78)
    print("SANITY CHECK: Baseline (window-prob argmax) vs existing smoothed eval")
    print("=" * 78)
    if os.path.exists(EXISTING_PER_VIDEO_CSV):
        ex = pd.read_csv(EXISTING_PER_VIDEO_CSV)
        # Pool existing across rows (already per-video per-class)
        for cid in EVAL_CLASSES:
            cname = ID_TO_CLASS[cid]
            sub = ex[ex["class_id"] == cid]
            tp = sub["frame_tp"].sum(); fp = sub["frame_fp"].sum(); fn = sub["frame_fn"].sum()
            pp = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            ff = 2 * pp * rr / (pp + rr) if (pp + rr) > 0 else 0.0
            n_pred_b_ex = int(sub["n_pred_bouts"].sum())
            n_gt_b_ex = int(sub["n_gt_bouts"].sum())
            tp25_ex = int(sub["bout_tp_iou25"].sum())
            fp25_ex = int(sub["bout_fp_iou25"].sum())
            fn25_ex = int(sub["bout_fn_iou25"].sum())

            base = all_results["Baseline"][cid]
            print(f"\n  {cname}:")
            print(f"    existing(smoothed)    frame  P={pp:.4f} R={rr:.4f} F1={ff:.4f}  "
                  f"pred_bouts={n_pred_b_ex} gt_bouts={n_gt_b_ex}  "
                  f"right25={tp25_ex} wrong25={fp25_ex} missed25={fn25_ex}")
            print(f"    Baseline(unsmoothed)  frame  P={base['frame_p']:.4f} R={base['frame_r']:.4f} F1={base['frame_f1']:.4f}  "
                  f"pred_bouts={base['n_pred_bouts']} gt_bouts={base['n_gt_bouts']}  "
                  f"right25={base['right_25']} wrong25={base['wrong_25']} missed25={base['missed_25']}")
        print("\n  NOTE: existing eval used smoothed predictions (median + bout_min=15);")
        print("        Baseline here is unsmoothed window-prob argmax. Frame metrics")
        print("        should be close; bout counts will differ (more / shorter bouts).")
    else:
        print("  (existing per_video_eval.csv not found, skipping)")

    # ---- Side-by-side comparison tables ----
    cfg_names = [c["name"] for c in CONFIGS]

    def print_table(iou_label, iou_key_right, iou_key_wrong):
        print("\n" + "=" * 78)
        print(f"COMPARISON  ({iou_label})")
        print("=" * 78)
        col_w = 12
        header = f"  {'Metric':<46}" + "".join(f"{n:>{col_w}}" for n in cfg_names)
        print(header)
        print(f"  {'-' * 46}" + "".join(f"{'-' * col_w}" for _ in cfg_names))

        def row(label, getter):
            vals = [getter(all_results[n]) for n in cfg_names]
            cells = "".join(
                f"{v:>{col_w}.4f}" if isinstance(v, float) else f"{v:>{col_w}}" for v in vals
            )
            print(f"  {label:<46}{cells}")

        for cid in EVAL_CLASSES:
            cname = ID_TO_CLASS[cid]
            row(f"{cname} right ({iou_label})",
                lambda r, cid=cid: r[cid][iou_key_right])
            row(f"{cname} wrong ({iou_label})",
                lambda r, cid=cid: r[cid][iou_key_wrong])
            row(f"{cname} missed ({iou_label})",
                lambda r, cid=cid: r[cid]["missed_25" if iou_label == "loose" else "missed_50"])
            row(f"{cname} hallucinated FP bouts (0-GT vids)",
                lambda r, cid=cid: r[cid]["halluc_pred_bouts_zero_gt"])

        # Frame-level + macro
        for cid in EVAL_CLASSES:
            cname = ID_TO_CLASS[cid]
            row(f"{cname} frame P", lambda r, cid=cid: r[cid]["frame_p"])
            row(f"{cname} frame R", lambda r, cid=cid: r[cid]["frame_r"])
            row(f"{cname} frame F1", lambda r, cid=cid: r[cid]["frame_f1"])

        def macro(r):
            return float(np.mean([r[c]["frame_f1"] for c in EVAL_CLASSES]))
        row("Frame macro-F1 (att+inv+mount, pooled)", macro)

    print_table("loose", "right_25", "wrong_25")
    print_table("strict", "right_50", "wrong_50")

    # ---- Save CSV ----
    rows = []
    for cfg in CONFIGS:
        name = cfg["name"]
        thr = cfg["thresholds"]
        for cid in EVAL_CLASSES:
            cname = ID_TO_CLASS[cid]
            d = all_results[name][cid]
            rows.append({
                "config": name,
                "threshold_attack": thr[0] if thr[0] is not None else "",
                "threshold_inv": thr[1] if thr[1] is not None else "",
                "threshold_mount": thr[2] if thr[2] is not None else "",
                "class": cname,
                "frame_p": round(d["frame_p"], 6),
                "frame_r": round(d["frame_r"], 6),
                "frame_f1": round(d["frame_f1"], 6),
                "pred_bouts": d["n_pred_bouts"],
                "gt_bouts": d["n_gt_bouts"],
                "right_bouts_iou25": d["right_25"],
                "wrong_bouts_iou25": d["wrong_25"],
                "missed_bouts_iou25": d["missed_25"],
                "right_bouts_iou50": d["right_50"],
                "wrong_bouts_iou50": d["wrong_50"],
                "missed_bouts_iou50": d["missed_50"],
                "hallucinated_fp_bouts_on_zero_gt_videos": d["halluc_pred_bouts_zero_gt"],
            })
    pd.DataFrame(rows).to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV}  ({len(rows)} rows)")

    print("\nDONE")


if __name__ == "__main__":
    main()
