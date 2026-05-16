"""sweep_posthoc_smoothing.py

Re-derive bout extraction from R5 v2 per-frame averaged-window probabilities
across a grid of (smoothing_window, min_duration) settings - entirely
post-hoc, no GPU re-run needed.

Reads:
    R5_yolo_attn_v2_pipeline_mars_test_eval/<split>/<video>/
        <video>.behavior.smoothed_frames.csv  (uses prob_* columns;
                                               ignores predicted_class_id)

For each setting, computes per video / per class:
    - bout count (pred vs GT)
    - bout F1 @ tIoU 0.25 + 0.50
    - frame F1 (invariant to smoothing/duration; sanity check)
    - attack hallucinations (FP bouts on zero-GT videos)
    - investigation bout count match ratio

Writes:
    figures/data/R5_yolo_attn_v2_pipeline_test_eval/post_hoc_sweep.csv
    figures/bout_posthoc_analysis/sweep_pareto_invest_vs_attack.png
    figures/bout_posthoc_analysis/sweep_metrics_grid.png

Used to characterize the smoothing/min-duration trade-off without
re-running inference. Choose the operating point that gives the best
manuscript story; the existing v2 inference outputs are reused.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(r".")
INFERENCE_ROOT = REPO / "BehaviorScope" / "behavior_lstm_runs" / "R5_yolo_attn_v2_pipeline_mars_test_eval"
GT_ROOT = Path(r".\MARS_data")
DECODER_THRESHOLDS = [0.6, 0.4, 0.35, 0.0]   # from R5 v2 config.json
DECODER_BG_INDEX = 3                          # "other"

OUT_CSV  = REPO / "manuscript_outputs" / "figures" / "data" / "R5_yolo_attn_v2_pipeline_test_eval"
OUT_FIGS = REPO / "manuscript_outputs" / "figures" / "bout_posthoc_analysis"
OUT_CSV.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

FPS = 30.0
CLASS_MAP = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
EVAL_CLASSES = [0, 1, 2]
SPLITS = ["test_1", "test_2"]

CLASS_COLORS = {"attack": "#d62728", "investigation": "#1a73e8", "mount": "#2ca02c"}

# Sweep grid: (smoothing_window, min_duration_frames)
# 0/1 = no filter. Current default is (5, 15). We sweep around it.
SWEEP_GRID = [
    (1, 1),    # raw (no smoothing, no min duration)
    (1, 5),    # no smoothing, weak min duration (~167 ms)
    (3, 5),    # weak smoothing + weak min duration
    (3, 10),
    (5, 10),
    (5, 15),   # current default
    (5, 20),
    (7, 15),
    (7, 25),
]


# ---------------------------------------------------------------------------
# Decoder + filters (replicas of infer_y.py logic)
# ---------------------------------------------------------------------------
def apply_decoder(probs: np.ndarray) -> np.ndarray:
    """probs: [T, n_classes] ? preds: [T]"""
    n_classes = probs.shape[1]
    thr = np.asarray(DECODER_THRESHOLDS, dtype=np.float64)
    behavior_indices = [i for i in range(n_classes) if i != DECODER_BG_INDEX]
    preds = np.full(probs.shape[0], DECODER_BG_INDEX, dtype=np.int64)
    for ti, row in enumerate(probs):
        passing = [i for i in behavior_indices if float(row[i]) >= float(thr[i])]
        if passing:
            preds[ti] = int(passing[int(np.argmax(row[passing]))])
    return preds


def median_filter_1d(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.copy()
    pad = window // 2
    padded = np.pad(x, pad, mode="edge")
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = int(np.median(padded[i:i + window]))
    return out


def apply_min_duration(pred: np.ndarray, min_frames: int) -> np.ndarray:
    if min_frames <= 1:
        return pred.copy()
    out = pred.copy()
    n = len(out)
    i = 0
    while i < n:
        c = out[i]
        j = i
        while j < n and out[j] == c:
            j += 1
        # bout [i, j) of class c
        if c != DECODER_BG_INDEX and (j - i) < min_frames:
            out[i:j] = DECODER_BG_INDEX
        i = j
    return out


# ---------------------------------------------------------------------------
# Bout helpers (verbatim from deep analysis)
# ---------------------------------------------------------------------------
def parse_bento_annot(path, total_frames, fps):
    bouts_by_class = {}
    current_class = None
    in_data = False
    with open(path, "r") as fh:
        for line in fh:
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
                        bouts_by_class[current_class].append(
                            (int(round(start_s * fps)), int(round(stop_s * fps)))
                        )
                    except ValueError:
                        pass
    return bouts_by_class


def extract_bouts(frame_array, class_id):
    bouts, in_bout, start = [], False, None
    for i, c in enumerate(frame_array):
        if c == class_id:
            if not in_bout:
                in_bout, start = True, i
        else:
            if in_bout:
                bouts.append((start, i - 1)); in_bout = False
    if in_bout:
        bouts.append((start, len(frame_array) - 1))
    return bouts


def tiou(pb, gb):
    ps, pe = pb; gs, ge = gb
    inter = max(0, min(pe, ge) - max(ps, gs) + 1)
    union = (pe - ps + 1) + (ge - gs + 1) - inter
    return inter / union if union > 0 else 0.0


def bout_f1(pred_bouts, gt_bouts, threshold):
    tp = fp = 0
    matched = set()
    for pb in pred_bouts:
        best, best_gi = 0.0, -1
        for gi, gb in enumerate(gt_bouts):
            if gi in matched:
                continue
            iv = tiou(pb, gb)
            if iv > best:
                best, best_gi = iv, gi
        if best >= threshold and best_gi >= 0:
            tp += 1; matched.add(best_gi)
        else:
            fp += 1
    fn = len(gt_bouts) - len(matched)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return tp, fp, fn, p, r, f


def find_annot(split, video):
    pattern = str(GT_ROOT / split / video / f"{video}_*.annot")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


# ---------------------------------------------------------------------------
# Per-video sweep
# ---------------------------------------------------------------------------
def sweep_video(split, video):
    vd = INFERENCE_ROOT / split / video
    sm = vd / f"{video}.behavior.smoothed_frames.csv"
    if not sm.is_file():
        return None
    annot = find_annot(split, video)
    if annot is None:
        return None
    df = pd.read_csv(sm)
    if not len(df):
        return None
    probs = df[["prob_attack", "prob_investigation", "prob_mount", "prob_other"]].values.astype(np.float64)
    n_frames = len(probs)
    gt_by_name = parse_bento_annot(annot, n_frames, FPS)
    # GT per-frame array
    gt_frames = np.full(n_frames, CLASS_MAP["other"], dtype=int)
    gt_bouts_by_id = {cid: [] for cid in EVAL_CLASSES}
    for cn, bs in gt_by_name.items():
        cid = CLASS_MAP.get(cn)
        if cid is None or cid == CLASS_MAP["other"]:
            continue
        gt_bouts_by_id[cid] = bs
        for sf, ef in bs:
            sf = max(sf, 0); ef = min(ef, n_frames - 1)
            gt_frames[sf:ef + 1] = cid

    # Apply decoder once (identical to v2 inference) - preds [T]
    raw_preds = apply_decoder(probs)
    rows = []
    for sw, md in SWEEP_GRID:
        sm_preds = median_filter_1d(raw_preds, sw)
        f_preds = apply_min_duration(sm_preds, md)
        for cid in EVAL_CLASSES:
            cname = ID_TO_CLASS[cid]
            pred_bouts = extract_bouts(f_preds, cid)
            gt_bouts = gt_bouts_by_id[cid]
            tp25, fp25, fn25, p25, r25, f25 = bout_f1(pred_bouts, gt_bouts, 0.25)
            tp50, fp50, fn50, p50, r50, f50 = bout_f1(pred_bouts, gt_bouts, 0.50)
            # frame counts for frame F1 pooling
            frame_tp = int(np.sum((f_preds == cid) & (gt_frames == cid)))
            frame_fp = int(np.sum((f_preds == cid) & (gt_frames != cid)))
            frame_fn = int(np.sum((f_preds != cid) & (gt_frames == cid)))
            rows.append({
                "split": split, "video": video,
                "smoothing_window": sw, "min_duration": md,
                "class": cname,
                "n_pred_bouts": len(pred_bouts),
                "n_gt_bouts": len(gt_bouts),
                "tp25": tp25, "fp25": fp25, "fn25": fn25, "f1_25": f25,
                "tp50": tp50, "fp50": fp50, "fn50": fn50, "f1_50": f50,
                "frame_tp": frame_tp, "frame_fp": frame_fp, "frame_fn": frame_fn,
            })
    return rows


def main():
    discovered = []
    for split in SPLITS:
        sd = INFERENCE_ROOT / split
        if not sd.is_dir():
            continue
        for v in sorted(os.listdir(sd)):
            if (sd / v).is_dir():
                discovered.append((split, v))

    all_rows = []
    for split, video in discovered:
        res = sweep_video(split, video)
        if res:
            all_rows.extend(res)
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV / "post_hoc_sweep.csv", index=False)

    # Pool per (smoothing_window, min_duration, class)
    agg = df.groupby(["smoothing_window", "min_duration", "class"]).agg({
        "n_pred_bouts": "sum",
        "n_gt_bouts": "sum",
        "tp25": "sum", "fp25": "sum", "fn25": "sum",
        "tp50": "sum", "fp50": "sum", "fn50": "sum",
        "frame_tp": "sum", "frame_fp": "sum", "frame_fn": "sum",
    }).reset_index()

    def _f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    agg["pooled_bout_f1_25"]  = agg.apply(lambda r: _f1(r["tp25"], r["fp25"], r["fn25"]), axis=1)
    agg["pooled_bout_f1_50"]  = agg.apply(lambda r: _f1(r["tp50"], r["fp50"], r["fn50"]), axis=1)
    agg["pooled_frame_f1"]    = agg.apply(lambda r: _f1(r["frame_tp"], r["frame_fp"], r["frame_fn"]), axis=1)
    agg["bout_count_match"]   = agg["n_pred_bouts"] / agg["n_gt_bouts"].replace(0, np.nan)
    agg.to_csv(OUT_CSV / "post_hoc_sweep_pooled.csv", index=False)

    # Attack hallucination count per (sw, md): zero-GT videos only
    attack = df[df["class"] == "attack"].copy()
    halluc_rows = []
    for (sw, md), sub in attack.groupby(["smoothing_window", "min_duration"]):
        zero_gt = sub[sub["n_gt_bouts"] == 0]
        n_halluc_bouts = int(zero_gt["n_pred_bouts"].sum())
        n_halluc_videos = int((zero_gt["n_pred_bouts"] > 0).sum())
        halluc_rows.append({
            "smoothing_window": sw, "min_duration": md,
            "attack_halluc_bouts": n_halluc_bouts,
            "attack_halluc_videos": n_halluc_videos,
            "n_zero_gt_videos": len(zero_gt),
        })
    halluc_df = pd.DataFrame(halluc_rows)
    halluc_df.to_csv(OUT_CSV / "post_hoc_sweep_halluc.csv", index=False)

    # ----- Pareto plot: investigation bout match vs attack hallucinations -----
    inv = agg[agg["class"] == "investigation"].copy()
    inv = inv.merge(halluc_df, on=["smoothing_window", "min_duration"])
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, r in inv.iterrows():
        is_default = (r["smoothing_window"] == 5 and r["min_duration"] == 15)
        marker = "*" if is_default else "o"
        size = 350 if is_default else 130
        edge = "black" if is_default else "white"
        ax.scatter(r["bout_count_match"], r["attack_halluc_bouts"],
                   s=size, marker=marker, color="#1a73e8",
                   edgecolors=edge, linewidths=1.5, zorder=3)
        label = f"sw={int(r['smoothing_window'])}\nmd={int(r['min_duration'])}"
        if is_default:
            label += " (current)"
        ax.annotate(label,
                    (r["bout_count_match"], r["attack_halluc_bouts"]),
                    xytext=(8, 8), textcoords="offset points",
                    fontsize=8.5,
                    fontweight="bold" if is_default else "normal",
                    color="#1a3a8a" if is_default else "#374151")
    ax.axvline(1.0, color="#16a34a", linestyle="--", alpha=0.5,
               label="GT bout count parity (1.0)")
    ax.set_xlabel("Investigation bout count ratio (pred / GT) - closer to 1 = better")
    ax.set_ylabel("Attack hallucination bouts (zero-GT videos)")
    ax.set_title("Smoothing x min-duration trade-off - R5 v2 post-hoc sweep\n"
                 "(top-left = bad; bottom-right of x=1.0 = ideal Pareto point)",
                 fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "sweep_pareto_invest_vs_attack.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    # ----- 4-panel grid of metrics across the sweep -----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sweep_labels = [f"sw={s},md={m}" for (s, m) in SWEEP_GRID]
    panel_specs = [
        (axes[0, 0], "pooled_frame_f1",  "Pooled frame F1"),
        (axes[0, 1], "pooled_bout_f1_25","Pooled bout F1 @ tIoU 0.25"),
        (axes[1, 0], "pooled_bout_f1_50","Pooled bout F1 @ tIoU 0.50"),
        (axes[1, 1], "bout_count_match", "Bout count ratio (pred/GT)"),
    ]
    x = np.arange(len(SWEEP_GRID))
    for ax, key, title in panel_specs:
        for cname, color in CLASS_COLORS.items():
            sub = agg[agg["class"] == cname].copy()
            sub["sw_md"] = list(zip(sub["smoothing_window"], sub["min_duration"]))
            ordered = [sub[sub["sw_md"] == sm].iloc[0][key] for sm in SWEEP_GRID]
            ax.plot(x, ordered, marker="o", linewidth=2, color=color, label=cname)
        if title.startswith("Bout count"):
            ax.axhline(1.0, color="black", linestyle=":", alpha=0.6, linewidth=0.8)
            ax.set_ylim(0, max(2.5, ax.get_ylim()[1]))
        else:
            ax.set_ylim(0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(sweep_labels, rotation=35, ha="right", fontsize=8.5)
        # mark current default
        try:
            di = SWEEP_GRID.index((5, 15))
            ax.axvline(di, color="gold", linestyle="--", alpha=0.7,
                       linewidth=1.2, label="current default")
        except ValueError:
            pass
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5, loc="lower right")
    plt.suptitle("R5 v2 post-hoc smoothing x min-duration sweep - same model, varied post-processing",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "sweep_metrics_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ----- Console summary -----
    print("=" * 90)
    print("Pooled metrics across smoothing x min-duration sweep (current default = sw=5, md=15)")
    print("=" * 90)
    pivot_invest = agg[agg["class"] == "investigation"].pivot(
        index="smoothing_window", columns="min_duration", values="bout_count_match"
    ).round(2)
    print("\n--- Investigation bout count ratio (pred/GT, target=1.00) ---")
    print(pivot_invest.to_string())

    pivot_inv_f1 = agg[agg["class"] == "investigation"].pivot(
        index="smoothing_window", columns="min_duration", values="pooled_bout_f1_50"
    ).round(3)
    print("\n--- Investigation pooled bout F1 @ tIoU 0.50 ---")
    print(pivot_inv_f1.to_string())

    pivot_attack_halluc = halluc_df.pivot(
        index="smoothing_window", columns="min_duration", values="attack_halluc_bouts"
    )
    print("\n--- Attack hallucination bouts (zero-GT videos) ---")
    print(pivot_attack_halluc.to_string())

    pivot_invest_frame = agg[agg["class"] == "investigation"].pivot(
        index="smoothing_window", columns="min_duration", values="pooled_frame_f1"
    ).round(3)
    print("\n--- Investigation pooled FRAME F1 (sanity: should barely change) ---")
    print(pivot_invest_frame.to_string())

    print(f"\nFigures: {OUT_FIGS}")
    print(f"CSVs:    {OUT_CSV}")


if __name__ == "__main__":
    main()
