"""
evaluate_full_model_test_set.py

Held-out test-set evaluation of BehaviorScope-Y R5_yolo_attn_full across all
videos in test_1 and test_2 splits. Mirrors metric definitions from
scripts/evaluate_yolo_pipeline_smoke.py (Bento .annot parser, smoothed-frame
prediction loader, frame-level P/R/F1, greedy bout matching at tIoU 0.25/0.50).
"""

import os
import glob
import json
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
PER_VIDEO_CSV = os.path.join(OUT_DIR, "per_video_eval.csv")
AGGREGATE_CSV = os.path.join(OUT_DIR, "aggregate_eval.csv")


# ---------------------------------------------------------------------------
# Bento .annot parser (verbatim from evaluate_yolo_pipeline_smoke.py)
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


def load_smoothed_predictions(csv_path, total_frames):
    df = pd.read_csv(csv_path)
    per_frame_pred = np.full(total_frames, CLASS_MAP["other"], dtype=int)
    for _, row in df.iterrows():
        fi = int(row["frame_idx"])
        if 0 <= fi < total_frames:
            per_frame_pred[fi] = int(row["predicted_class_id"])
    return per_frame_pred


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
    """Greedy match: tp/fp/fn + P/R/F1."""
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
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, prec, rec, f1


def find_annot(split, video):
    pattern = os.path.join(GT_ROOT, split, video, f"{video}_*.annot")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def get_total_frames(video_dir, video):
    smoothed_csv = os.path.join(video_dir, f"{video}.behavior.smoothed_frames.csv")
    raw_csv = os.path.join(video_dir, f"{video}.behavior.csv")
    if os.path.exists(smoothed_csv):
        try:
            df = pd.read_csv(smoothed_csv)
            if len(df):
                return int(df["frame_idx"].max()) + 1
        except Exception:
            pass
    if os.path.exists(raw_csv):
        try:
            df = pd.read_csv(raw_csv)
            if len(df):
                return int(df["frame_end"].max()) + 1
        except Exception:
            pass
    return None


# ---------------------------------------------------------------------------
# Per-video evaluation
# ---------------------------------------------------------------------------
def evaluate_video(split, video):
    video_dir = os.path.join(INFERENCE_ROOT, split, video)
    smoothed_csv = os.path.join(video_dir, f"{video}.behavior.smoothed_frames.csv")
    if not os.path.exists(smoothed_csv):
        return None, "no_smoothed_csv"

    annot_path = find_annot(split, video)
    if annot_path is None:
        return None, "no_annot"

    total_frames = get_total_frames(video_dir, video)
    if total_frames is None or total_frames <= 0:
        return None, "no_total_frames"

    bouts_by_class, gt_frames = parse_bento_annot(annot_path, total_frames, FPS)
    pred_frames = load_smoothed_predictions(smoothed_csv, total_frames)

    gt_bouts_by_id = {cid: [] for cid in EVAL_CLASSES + [3]}
    for cname, bouts in bouts_by_class.items():
        cid = CLASS_MAP.get(cname)
        if cid is not None:
            gt_bouts_by_id[cid] = bouts

    # Frame-level metrics using sklearn (per-class TP/FP/FN computed manually
    # so we can pool across videos)
    labels = list(range(4))
    prec_arr, rec_arr, f1_arr, _ = precision_recall_fscore_support(
        gt_frames, pred_frames, labels=labels, average=None, zero_division=0
    )

    # Manual per-class confusion at frame level for pooling
    frame_tp = np.zeros(4, dtype=np.int64)
    frame_fp = np.zeros(4, dtype=np.int64)
    frame_fn = np.zeros(4, dtype=np.int64)
    for cid in labels:
        gt_pos = gt_frames == cid
        pr_pos = pred_frames == cid
        frame_tp[cid] = int(np.sum(gt_pos & pr_pos))
        frame_fp[cid] = int(np.sum(~gt_pos & pr_pos))
        frame_fn[cid] = int(np.sum(gt_pos & ~pr_pos))

    per_class = {}
    for cid in EVAL_CLASSES:
        cname = ID_TO_CLASS[cid]
        pred_bouts = extract_bouts(pred_frames, cid)
        gt_bouts = gt_bouts_by_id.get(cid, [])

        tp25, fp25, fn25, p25, r25, f25 = bout_metrics(pred_bouts, gt_bouts, 0.25)
        tp50, fp50, fn50, p50, r50, f50 = bout_metrics(pred_bouts, gt_bouts, 0.50)

        per_class[cid] = {
            "class": cname,
            "frame_precision": float(prec_arr[cid]),
            "frame_recall": float(rec_arr[cid]),
            "frame_f1": float(f1_arr[cid]),
            "frame_tp": int(frame_tp[cid]),
            "frame_fp": int(frame_fp[cid]),
            "frame_fn": int(frame_fn[cid]),
            "n_pred_bouts": len(pred_bouts),
            "n_gt_bouts": len(gt_bouts),
            "bout_tp_iou25": tp25, "bout_fp_iou25": fp25, "bout_fn_iou25": fn25,
            "bout_p_iou25": p25, "bout_r_iou25": r25, "bout_f1_iou25": f25,
            "bout_tp_iou50": tp50, "bout_fp_iou50": fp50, "bout_fn_iou50": fn50,
            "bout_p_iou50": p50, "bout_r_iou50": r50, "bout_f1_iou50": f50,
        }

    # Macro frame F1 across all 4 classes (matches sklearn's macro)
    macro_f1 = float(np.mean(f1_arr))

    return {
        "video": video,
        "split": split,
        "total_frames": total_frames,
        "annot_path": annot_path,
        "macro_frame_f1": macro_f1,
        "per_class": per_class,
    }, "ok"


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def aggregate_per_video_macro(video_results, cid):
    vals = {
        "frame_precision": [],
        "frame_recall": [],
        "frame_f1": [],
        "bout_f1_iou25": [],
        "bout_f1_iou50": [],
        "n_pred_bouts": 0,
        "n_gt_bouts": 0,
    }
    for v in video_results:
        pc = v["per_class"][cid]
        vals["frame_precision"].append(pc["frame_precision"])
        vals["frame_recall"].append(pc["frame_recall"])
        vals["frame_f1"].append(pc["frame_f1"])
        vals["bout_f1_iou25"].append(pc["bout_f1_iou25"])
        vals["bout_f1_iou50"].append(pc["bout_f1_iou50"])
        vals["n_pred_bouts"] += pc["n_pred_bouts"]
        vals["n_gt_bouts"] += pc["n_gt_bouts"]
    return {
        "frame_precision": float(np.mean(vals["frame_precision"])) if vals["frame_precision"] else 0.0,
        "frame_recall": float(np.mean(vals["frame_recall"])) if vals["frame_recall"] else 0.0,
        "frame_f1": float(np.mean(vals["frame_f1"])) if vals["frame_f1"] else 0.0,
        "bout_f1_iou25": float(np.mean(vals["bout_f1_iou25"])) if vals["bout_f1_iou25"] else 0.0,
        "bout_f1_iou50": float(np.mean(vals["bout_f1_iou50"])) if vals["bout_f1_iou50"] else 0.0,
        "total_pred_bouts": vals["n_pred_bouts"],
        "total_gt_bouts": vals["n_gt_bouts"],
    }


def aggregate_pooled(video_results, cid):
    tp = fp = fn = 0
    btp25 = bfp25 = bfn25 = 0
    btp50 = bfp50 = bfn50 = 0
    n_pred_bouts = n_gt_bouts = 0
    for v in video_results:
        pc = v["per_class"][cid]
        tp += pc["frame_tp"]; fp += pc["frame_fp"]; fn += pc["frame_fn"]
        btp25 += pc["bout_tp_iou25"]; bfp25 += pc["bout_fp_iou25"]; bfn25 += pc["bout_fn_iou25"]
        btp50 += pc["bout_tp_iou50"]; bfp50 += pc["bout_fp_iou50"]; bfn50 += pc["bout_fn_iou50"]
        n_pred_bouts += pc["n_pred_bouts"]
        n_gt_bouts += pc["n_gt_bouts"]
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    p25 = btp25 / (btp25 + bfp25) if (btp25 + bfp25) > 0 else 0.0
    r25 = btp25 / (btp25 + bfn25) if (btp25 + bfn25) > 0 else 0.0
    f25 = 2 * p25 * r25 / (p25 + r25) if (p25 + r25) > 0 else 0.0
    p50 = btp50 / (btp50 + bfp50) if (btp50 + bfp50) > 0 else 0.0
    r50 = btp50 / (btp50 + bfn50) if (btp50 + bfn50) > 0 else 0.0
    f50 = 2 * p50 * r50 / (p50 + r50) if (p50 + r50) > 0 else 0.0
    return {
        "frame_precision": p, "frame_recall": r, "frame_f1": f1,
        "bout_p_iou25": p25, "bout_r_iou25": r25, "bout_f1_iou25": f25,
        "bout_p_iou50": p50, "bout_r_iou50": r50, "bout_f1_iou50": f50,
        "total_pred_bouts": n_pred_bouts,
        "total_gt_bouts": n_gt_bouts,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Discover videos
    discovered = []
    for split in SPLITS:
        split_dir = os.path.join(INFERENCE_ROOT, split)
        if not os.path.isdir(split_dir):
            continue
        for v in sorted(os.listdir(split_dir)):
            vd = os.path.join(split_dir, v)
            if os.path.isdir(vd):
                discovered.append((split, v))

    print("=" * 78)
    print("R5_yolo_attn_full TEST-SET EVALUATION (test_1 + test_2)")
    print("=" * 78)
    print(f"\nSetup")
    print(f"  Inference root : {INFERENCE_ROOT}")
    print(f"  GT root        : {GT_ROOT}")
    print(f"  Splits         : {SPLITS}")
    print(f"  Discovered     : {len(discovered)} video directories")
    for sp in SPLITS:
        n = sum(1 for s, _ in discovered if s == sp)
        print(f"    {sp:<10}: {n}")

    # Run per-video eval
    successes = []
    failures = []
    for split, video in discovered:
        result, status = evaluate_video(split, video)
        if result is None:
            failures.append((split, video, status))
        else:
            successes.append(result)

    print(f"\n  Successful evaluations: {len(successes)}")
    print(f"  Failures              : {len(failures)}")
    for sp, v, st in failures:
        print(f"    [{sp}] {v}: {st}")

    # ---- Per-video CSV ----
    per_video_rows = []
    for v in successes:
        for cid in EVAL_CLASSES:
            pc = v["per_class"][cid]
            per_video_rows.append({
                "split": v["split"],
                "video": v["video"],
                "total_frames": v["total_frames"],
                "class_id": cid,
                "class": pc["class"],
                "frame_tp": pc["frame_tp"], "frame_fp": pc["frame_fp"], "frame_fn": pc["frame_fn"],
                "frame_precision": round(pc["frame_precision"], 6),
                "frame_recall": round(pc["frame_recall"], 6),
                "frame_f1": round(pc["frame_f1"], 6),
                "n_pred_bouts": pc["n_pred_bouts"],
                "n_gt_bouts": pc["n_gt_bouts"],
                "bout_tp_iou25": pc["bout_tp_iou25"],
                "bout_fp_iou25": pc["bout_fp_iou25"],
                "bout_fn_iou25": pc["bout_fn_iou25"],
                "bout_p_iou25": round(pc["bout_p_iou25"], 6),
                "bout_r_iou25": round(pc["bout_r_iou25"], 6),
                "bout_f1_iou25": round(pc["bout_f1_iou25"], 6),
                "bout_tp_iou50": pc["bout_tp_iou50"],
                "bout_fp_iou50": pc["bout_fp_iou50"],
                "bout_fn_iou50": pc["bout_fn_iou50"],
                "bout_p_iou50": round(pc["bout_p_iou50"], 6),
                "bout_r_iou50": round(pc["bout_r_iou50"], 6),
                "bout_f1_iou50": round(pc["bout_f1_iou50"], 6),
            })
    per_video_df = pd.DataFrame(per_video_rows)
    per_video_df.to_csv(PER_VIDEO_CSV, index=False)
    print(f"\nSaved: {PER_VIDEO_CSV}  ({len(per_video_df)} rows)")

    # ---- Aggregates ----
    splits_for_agg = {
        "test_1": [v for v in successes if v["split"] == "test_1"],
        "test_2": [v for v in successes if v["split"] == "test_2"],
        "combined": successes,
    }

    agg_rows = []
    for split_label, vids in splits_for_agg.items():
        if not vids:
            continue
        for cid in EVAL_CLASSES:
            cname = ID_TO_CLASS[cid]
            pvm = aggregate_per_video_macro(vids, cid)
            pld = aggregate_pooled(vids, cid)
            agg_rows.append({
                "split": split_label,
                "n_videos": len(vids),
                "class_id": cid,
                "class": cname,
                # per-video macro
                "pvm_frame_precision": round(pvm["frame_precision"], 6),
                "pvm_frame_recall": round(pvm["frame_recall"], 6),
                "pvm_frame_f1": round(pvm["frame_f1"], 6),
                "pvm_bout_f1_iou25": round(pvm["bout_f1_iou25"], 6),
                "pvm_bout_f1_iou50": round(pvm["bout_f1_iou50"], 6),
                # pooled
                "pooled_frame_precision": round(pld["frame_precision"], 6),
                "pooled_frame_recall": round(pld["frame_recall"], 6),
                "pooled_frame_f1": round(pld["frame_f1"], 6),
                "pooled_bout_p_iou25": round(pld["bout_p_iou25"], 6),
                "pooled_bout_r_iou25": round(pld["bout_r_iou25"], 6),
                "pooled_bout_f1_iou25": round(pld["bout_f1_iou25"], 6),
                "pooled_bout_p_iou50": round(pld["bout_p_iou50"], 6),
                "pooled_bout_r_iou50": round(pld["bout_r_iou50"], 6),
                "pooled_bout_f1_iou50": round(pld["bout_f1_iou50"], 6),
                # totals
                "total_pred_bouts": pld["total_pred_bouts"],
                "total_gt_bouts": pld["total_gt_bouts"],
            })
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(AGGREGATE_CSV, index=False)
    print(f"Saved: {AGGREGATE_CSV}  ({len(agg_df)} rows)")

    # ---- Print aggregate summaries ----
    for split_label, vids in splits_for_agg.items():
        if not vids:
            continue
        print("\n" + "=" * 78)
        print(f"AGGREGATE  split={split_label}  n_videos={len(vids)}")
        print("=" * 78)

        print("\n  --- Per-video macro (each video weighted equally) ---")
        print(f"  {'class':<14}{'frame_P':>10}{'frame_R':>10}{'frame_F1':>10}"
              f"{'bout_F1@25':>12}{'bout_F1@50':>12}{'pred_bouts':>12}{'gt_bouts':>10}")
        macro_frame_f1s = []
        for cid in EVAL_CLASSES:
            cname = ID_TO_CLASS[cid]
            pvm = aggregate_per_video_macro(vids, cid)
            macro_frame_f1s.append(pvm["frame_f1"])
            print(f"  {cname:<14}{pvm['frame_precision']:>10.4f}{pvm['frame_recall']:>10.4f}"
                  f"{pvm['frame_f1']:>10.4f}{pvm['bout_f1_iou25']:>12.4f}"
                  f"{pvm['bout_f1_iou50']:>12.4f}{pvm['total_pred_bouts']:>12d}"
                  f"{pvm['total_gt_bouts']:>10d}")
        print(f"  {'macro(att+inv+mount) frame F1':<40}: {np.mean(macro_frame_f1s):.4f}")
        macro_videowise = float(np.mean([v["macro_frame_f1"] for v in vids]))
        print(f"  {'mean(per-video sklearn macro frame F1, 4cls)':<40}: {macro_videowise:.4f}")

        print("\n  --- Pooled (sum TP/FP/FN across videos) ---")
        print(f"  {'class':<14}{'frame_P':>10}{'frame_R':>10}{'frame_F1':>10}"
              f"{'bout_F1@25':>12}{'bout_F1@50':>12}{'pred_bouts':>12}{'gt_bouts':>10}")
        pooled_f1s = []
        for cid in EVAL_CLASSES:
            cname = ID_TO_CLASS[cid]
            pld = aggregate_pooled(vids, cid)
            pooled_f1s.append(pld["frame_f1"])
            print(f"  {cname:<14}{pld['frame_precision']:>10.4f}{pld['frame_recall']:>10.4f}"
                  f"{pld['frame_f1']:>10.4f}{pld['bout_f1_iou25']:>12.4f}"
                  f"{pld['bout_f1_iou50']:>12.4f}{pld['total_pred_bouts']:>12d}"
                  f"{pld['total_gt_bouts']:>10d}")
        print(f"  {'macro(att+inv+mount) frame F1 pooled':<40}: {np.mean(pooled_f1s):.4f}")

    # ---- Hallucination analysis ----
    print("\n" + "=" * 78)
    print("PER-CLASS HALLUCINATION ANALYSIS (combined splits)")
    print("=" * 78)
    print(f"  {'class':<14}{'N_videos':>10}{'N_present':>12}{'N_zero_GT':>12}"
          f"{'N_hallucinate':>16}{'tot_FP_bouts_on_hall':>22}")
    for cid in EVAL_CLASSES:
        cname = ID_TO_CLASS[cid]
        n_present = 0
        n_zero_gt = 0
        n_hallucinate = 0
        tot_fp_on_hall = 0
        for v in successes:
            pc = v["per_class"][cid]
            if pc["n_gt_bouts"] > 0:
                n_present += 1
            else:
                n_zero_gt += 1
                if pc["n_pred_bouts"] > 0:
                    n_hallucinate += 1
                    tot_fp_on_hall += pc["n_pred_bouts"]
        print(f"  {cname:<14}{len(successes):>10d}{n_present:>12d}{n_zero_gt:>12d}"
              f"{n_hallucinate:>16d}{tot_fp_on_hall:>22d}")

    # ---- Per-video table ----
    print("\n" + "=" * 78)
    print("PER-VIDEO TABLE")
    print("=" * 78)
    header = (
        f"  {'video':<48}{'split':<8}"
        f"{'macroF1':>9}{'att_F1':>9}{'inv_F1':>9}{'mnt_F1':>9}"
        f"{'att_pb':>8}{'att_gb':>8}{'inv_pb':>8}{'inv_gb':>8}{'mnt_pb':>8}{'mnt_gb':>8}"
    )
    print(header)
    for v in successes:
        pc_a = v["per_class"][0]
        pc_i = v["per_class"][1]
        pc_m = v["per_class"][2]
        # frame macro across attack + inv + mount (3 evaluated classes)
        mac3 = (pc_a["frame_f1"] + pc_i["frame_f1"] + pc_m["frame_f1"]) / 3
        print(
            f"  {v['video']:<48}{v['split']:<8}"
            f"{mac3:>9.4f}{pc_a['frame_f1']:>9.4f}{pc_i['frame_f1']:>9.4f}{pc_m['frame_f1']:>9.4f}"
            f"{pc_a['n_pred_bouts']:>8d}{pc_a['n_gt_bouts']:>8d}"
            f"{pc_i['n_pred_bouts']:>8d}{pc_i['n_gt_bouts']:>8d}"
            f"{pc_m['n_pred_bouts']:>8d}{pc_m['n_gt_bouts']:>8d}"
        )

    # ---- Sanity check vs Mouse060 anchor ----
    print("\n" + "=" * 78)
    print("SANITY CHECK vs Mouse060 anchor")
    print("=" * 78)
    anchor = next((v for v in successes if v["video"] == "Mouse060_20160526_18-16-27"), None)
    if anchor is None:
        print("  Mouse060_20160526_18-16-27 not in successes")
    else:
        pc_a = anchor["per_class"][0]
        pc_i = anchor["per_class"][1]
        pc_m = anchor["per_class"][2]
        print(f"  attack:        F1={pc_a['frame_f1']:.4f} (expected 0.000 frame; bout_pred={pc_a['n_pred_bouts']}, bout_gt={pc_a['n_gt_bouts']})")
        print(f"  investigation: F1={pc_i['frame_f1']:.4f} (anchor bout F1~0.744)")
        print(f"  mount:         F1={pc_m['frame_f1']:.4f} (anchor bout F1~0.542)")
        print(f"  attack bout F1@0.50={pc_a['bout_f1_iou50']:.4f}  inv bout F1@0.50={pc_i['bout_f1_iou50']:.4f}  mnt bout F1@0.50={pc_m['bout_f1_iou50']:.4f}")
        print(f"  attack bout F1@0.25={pc_a['bout_f1_iou25']:.4f}  inv bout F1@0.25={pc_i['bout_f1_iou25']:.4f}  mnt bout F1@0.25={pc_m['bout_f1_iou25']:.4f}")

    print("\nDONE")


if __name__ == "__main__":
    main()
