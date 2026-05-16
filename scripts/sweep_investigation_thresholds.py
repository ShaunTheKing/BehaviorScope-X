"""sweep_investigation_thresholds.py

Post-hoc sweep over (investigation decoder threshold) x (smoothing window) x
(min duration) on the existing R5 v2 inference outputs. Goal: characterize
the bout-count-vs-frame-F1 trade-off when we tune the investigation
threshold to encourage bout *splitting* (current decoder uses inv_thr = 0.4
which leaves the model in "merge adjacent investigations" mode).

Inputs (no GPU re-run needed):
    R5_yolo_attn_v2_pipeline_mars_test_eval/<split>/<video>/
        <video>.behavior.smoothed_frames.csv  (uses prob_* columns only)

Sweep grid:
    investigation_threshold ? {0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70}
    smoothing_window         ? {1, 3, 5}
    min_duration             ? {1, 5, 15}

Attack and mount thresholds are held at the val-fit decoder defaults
(0.6 and 0.35) - only investigation varies.

Outputs:
    figures/data/R5_yolo_attn_v2_pipeline_test_eval/
        invest_threshold_sweep.csv          per-video, per-config rows
        invest_threshold_sweep_pooled.csv   pooled-across-videos summary
    figures/bout_posthoc_analysis/
        invest_pareto_count_vs_frame.png    bout-count vs frame-F1 Pareto
        invest_bout_f1_vs_threshold.png     bout F1 @ 0.5 vs threshold
        invest_threshold_grid.png           4-panel summary across grid
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

# val-fit decoder for R5 v2 - only investigation will be swept
DECODER_THRESHOLDS_DEFAULT = [0.6, 0.4, 0.35, 0.0]
DECODER_BG_INDEX = 3   # other

OUT_CSV  = REPO / "manuscript_outputs" / "figures" / "data" / "R5_yolo_attn_v2_pipeline_test_eval"
OUT_FIGS = REPO / "manuscript_outputs" / "figures" / "bout_posthoc_analysis"
OUT_CSV.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

FPS = 30.0
CLASS_MAP = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
INV_IDX = CLASS_MAP["investigation"]
EVAL_CLASSES = [0, 1, 2]
SPLITS = ["test_1", "test_2"]

INV_THR_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
SW_GRID      = [1, 3, 5]
MD_GRID      = [1, 5, 15]


# ---------------------------------------------------------------------------
# Decoder + filters (same as post-hoc sweep)
# ---------------------------------------------------------------------------
def apply_decoder(probs: np.ndarray, thresholds) -> np.ndarray:
    n_classes = probs.shape[1]
    thr = np.asarray(thresholds, dtype=np.float64)
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
        if c != DECODER_BG_INDEX and (j - i) < min_frames:
            out[i:j] = DECODER_BG_INDEX
        i = j
    return out


# ---------------------------------------------------------------------------
# Bout helpers
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
                        s = float(parts[0]); e = float(parts[1])
                        bouts_by_class[current_class].append(
                            (int(round(s * fps)), int(round(e * fps)))
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


def bout_match_counts(pred_bouts, gt_bouts, threshold):
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
    return tp, fp, fn


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
        return []
    annot = find_annot(split, video)
    if annot is None:
        return []
    df = pd.read_csv(sm)
    if not len(df):
        return []
    probs = df[["prob_attack", "prob_investigation", "prob_mount", "prob_other"]].values.astype(np.float64)
    n_frames = len(probs)
    gt_by_name = parse_bento_annot(annot, n_frames, FPS)
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

    rows = []
    for inv_thr in INV_THR_GRID:
        thresholds = list(DECODER_THRESHOLDS_DEFAULT)
        thresholds[INV_IDX] = inv_thr
        raw_preds = apply_decoder(probs, thresholds)
        for sw in SW_GRID:
            sm_preds = median_filter_1d(raw_preds, sw)
            for md in MD_GRID:
                f_preds = apply_min_duration(sm_preds, md)
                for cid in EVAL_CLASSES:
                    cname = ID_TO_CLASS[cid]
                    pred_bouts = extract_bouts(f_preds, cid)
                    gt_bouts = gt_bouts_by_id[cid]
                    tp25, fp25, fn25 = bout_match_counts(pred_bouts, gt_bouts, 0.25)
                    tp50, fp50, fn50 = bout_match_counts(pred_bouts, gt_bouts, 0.50)
                    frame_tp = int(np.sum((f_preds == cid) & (gt_frames == cid)))
                    frame_fp = int(np.sum((f_preds == cid) & (gt_frames != cid)))
                    frame_fn = int(np.sum((f_preds != cid) & (gt_frames == cid)))
                    rows.append({
                        "split": split, "video": video,
                        "inv_thr": inv_thr, "smoothing_window": sw, "min_duration": md,
                        "class": cname,
                        "n_pred_bouts": len(pred_bouts),
                        "n_gt_bouts": len(gt_bouts),
                        "tp25": tp25, "fp25": fp25, "fn25": fn25,
                        "tp50": tp50, "fp50": fp50, "fn50": fn50,
                        "frame_tp": frame_tp, "frame_fp": frame_fp, "frame_fn": frame_fn,
                    })
    return rows


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def main():
    discovered = []
    for split in SPLITS:
        sd = INFERENCE_ROOT / split
        if not sd.is_dir():
            continue
        for v in sorted(os.listdir(sd)):
            if (sd / v).is_dir():
                discovered.append((split, v))
    print(f"Sweeping {len(discovered)} videos x "
          f"{len(INV_THR_GRID)} thresholds x {len(SW_GRID)} smoothing x "
          f"{len(MD_GRID)} min-duration = "
          f"{len(discovered) * len(INV_THR_GRID) * len(SW_GRID) * len(MD_GRID)} configs")

    all_rows = []
    for split, video in discovered:
        all_rows.extend(sweep_video(split, video))

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV / "invest_threshold_sweep.csv", index=False)

    agg = df.groupby(["inv_thr", "smoothing_window", "min_duration", "class"]).agg({
        "n_pred_bouts": "sum", "n_gt_bouts": "sum",
        "tp25": "sum", "fp25": "sum", "fn25": "sum",
        "tp50": "sum", "fp50": "sum", "fn50": "sum",
        "frame_tp": "sum", "frame_fp": "sum", "frame_fn": "sum",
    }).reset_index()
    agg["pooled_bout_f1_25"] = agg.apply(lambda r: _f1(r["tp25"], r["fp25"], r["fn25"]), axis=1)
    agg["pooled_bout_f1_50"] = agg.apply(lambda r: _f1(r["tp50"], r["fp50"], r["fn50"]), axis=1)
    agg["pooled_frame_f1"]   = agg.apply(lambda r: _f1(r["frame_tp"], r["frame_fp"], r["frame_fn"]), axis=1)
    agg["bout_count_match"]  = agg["n_pred_bouts"] / agg["n_gt_bouts"].replace(0, np.nan)
    agg.to_csv(OUT_CSV / "invest_threshold_sweep_pooled.csv", index=False)

    inv_agg = agg[agg["class"] == "investigation"].copy()

    # ----- Pareto plot: bout count match vs frame F1 -----
    fig, ax = plt.subplots(figsize=(11, 7))
    sw_markers = {1: "o", 3: "s", 5: "^"}
    md_alphas  = {1: 0.45, 5: 0.7, 15: 1.0}
    cmap = plt.get_cmap("viridis")
    for _, r in inv_agg.iterrows():
        sw = int(r["smoothing_window"]); md = int(r["min_duration"])
        thr = float(r["inv_thr"])
        is_default = abs(thr - 0.40) < 1e-6 and sw == 5 and md == 15
        color = cmap((thr - 0.40) / 0.30)
        size = 220 if is_default else 90
        edge = "red" if is_default else "white"
        lw = 2.0 if is_default else 0.6
        ax.scatter(r["bout_count_match"], r["pooled_frame_f1"],
                   s=size, marker=sw_markers[sw], color=color,
                   alpha=md_alphas[md], edgecolors=edge, linewidths=lw, zorder=3)
        if is_default:
            ax.annotate("CURRENT DEFAULT\n(thr=0.40, sw=5, md=15)",
                        (r["bout_count_match"], r["pooled_frame_f1"]),
                        xytext=(10, -12), textcoords="offset points",
                        fontsize=9, fontweight="bold", color="red")
    ax.axvline(1.0, color="#16a34a", linestyle="--", alpha=0.6,
               linewidth=1.2, label="GT bout count parity")
    # Color legend (threshold)
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0.40, vmax=0.70))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Investigation decoder threshold")
    # Marker legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker=m, color="w", markerfacecolor="gray",
                      markersize=9, label=f"smoothing_window = {sw}")
               for sw, m in sw_markers.items()]
    handles.append(Line2D([0],[0], color="#16a34a", linestyle="--",
                          label="bout count parity"))
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    ax.set_xlabel("Investigation bout count ratio (pred / GT) - closer to 1.0 = better")
    ax.set_ylabel("Pooled investigation FRAME F1 (24 videos)")
    ax.set_title("Investigation Pareto trade-off - bout count match vs frame F1\n"
                 "(7 thresholds x 3 smoothing x 3 min-duration = 63 operating points)",
                 fontsize=10)
    ax.set_ylim(0.45, 0.80)
    ax.set_xlim(0.4, 1.6)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "invest_pareto_count_vs_frame.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    # ----- Bout F1 @ 0.5 vs threshold (multiple smoothing curves) -----
    fig, ax = plt.subplots(figsize=(10, 5.5))
    md_use = 15  # fixed at default min-duration
    for sw in SW_GRID:
        sub = inv_agg[(inv_agg["smoothing_window"] == sw) &
                      (inv_agg["min_duration"] == md_use)].sort_values("inv_thr")
        ax.plot(sub["inv_thr"], sub["pooled_bout_f1_50"], marker="o",
                linewidth=2, label=f"smoothing window = {sw}")
        for _, r in sub.iterrows():
            ax.text(r["inv_thr"], r["pooled_bout_f1_50"] + 0.008,
                    f"{r['pooled_bout_f1_50']:.3f}",
                    ha="center", fontsize=7.5, color="#374151")
    ax.axvline(0.40, color="red", linestyle=":", alpha=0.6,
               label="current default (0.40)")
    ax.set_xlabel("Investigation decoder threshold")
    ax.set_ylabel("Pooled investigation bout F1 @ tIoU 0.50")
    ax.set_title(f"Investigation bout F1 vs threshold (min_duration = {md_use})\n"
                 "Higher threshold = more bout splitting",
                 fontsize=10)
    ax.set_xticks(INV_THR_GRID)
    ax.grid(alpha=0.3); ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "invest_bout_f1_vs_threshold.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    # ----- 4-panel summary (vs threshold, sw=5 fixed) -----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    sw_use = 5
    for ax, key, title, ylim in [
        (axes[0, 0], "bout_count_match",  "Bout count ratio (pred/GT)", (0.4, 1.6)),
        (axes[0, 1], "pooled_frame_f1",   "Pooled frame F1",            (0.45, 0.80)),
        (axes[1, 0], "pooled_bout_f1_25", "Pooled bout F1 @ tIoU 0.25", (0.30, 0.65)),
        (axes[1, 1], "pooled_bout_f1_50", "Pooled bout F1 @ tIoU 0.50", (0.25, 0.50)),
    ]:
        for md in MD_GRID:
            sub = inv_agg[(inv_agg["smoothing_window"] == sw_use) &
                          (inv_agg["min_duration"] == md)].sort_values("inv_thr")
            ax.plot(sub["inv_thr"], sub[key], marker="o", linewidth=2,
                    label=f"min_duration = {md}")
        if key == "bout_count_match":
            ax.axhline(1.0, color="#16a34a", linestyle="--", alpha=0.6,
                       label="GT parity")
        ax.axvline(0.40, color="red", linestyle=":", alpha=0.5)
        ax.set_xticks(INV_THR_GRID)
        ax.set_xlabel("Investigation decoder threshold")
        ax.set_title(title)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.3); ax.legend(fontsize=8.5)
    plt.suptitle(f"Investigation threshold sweep - smoothing_window = {sw_use} fixed",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "invest_threshold_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ----- Console summary -----
    print("\n" + "=" * 90)
    print("Investigation pooled metrics across the threshold x smoothing x min-duration grid")
    print("=" * 90)

    for metric, label in [("bout_count_match", "Bout count ratio (pred/GT, target=1.00)"),
                           ("pooled_frame_f1",   "Pooled frame F1"),
                           ("pooled_bout_f1_50", "Pooled bout F1 @ tIoU 0.50")]:
        # Pivot at md=15 (current default) for readability
        sub = inv_agg[inv_agg["min_duration"] == 15]
        pv = sub.pivot(index="smoothing_window", columns="inv_thr", values=metric).round(3)
        print(f"\n--- {label}  (min_duration = 15) ---")
        print(pv.to_string())

    # Find best operating points by different criteria
    print("\n" + "=" * 90)
    print("Best operating points by criterion (across full 63-point grid)")
    print("=" * 90)

    def fmt(r, label):
        return (f"  {label:42s}  "
                f"thr={r['inv_thr']:.2f} sw={int(r['smoothing_window'])} md={int(r['min_duration'])}  "
                f"|  bout_count={r['bout_count_match']:.2f}  "
                f"frame_F1={r['pooled_frame_f1']:.3f}  "
                f"bout_F1@0.5={r['pooled_bout_f1_50']:.3f}")

    # Closest to bout count parity
    inv_agg["dist_to_parity"] = (inv_agg["bout_count_match"] - 1.0).abs()
    best_parity = inv_agg.nsmallest(1, "dist_to_parity").iloc[0]
    print(fmt(best_parity, "closest to bout count parity"))

    # Best bout F1 @ 0.50
    best_bout = inv_agg.nlargest(1, "pooled_bout_f1_50").iloc[0]
    print(fmt(best_bout, "max bout F1 @ tIoU 0.50"))

    # Best frame F1
    best_frame = inv_agg.nlargest(1, "pooled_frame_f1").iloc[0]
    print(fmt(best_frame, "max frame F1"))

    # Composite: bout count near 1.0 AND frame F1 above 0.65
    feasible = inv_agg[(inv_agg["bout_count_match"] >= 0.85) &
                       (inv_agg["bout_count_match"] <= 1.15) &
                       (inv_agg["pooled_frame_f1"] >= 0.65)]
    if len(feasible):
        best_compromise = feasible.nlargest(1, "pooled_bout_f1_50").iloc[0]
        print(fmt(best_compromise, "best compromise (bout_count 0.85-1.15, frame_F1 >= 0.65)"))
    else:
        print("  (no operating points satisfy bout_count in [0.85, 1.15] AND frame_F1 >= 0.65)")

    print(f"\nFigures: {OUT_FIGS}")
    print(f"CSVs:    {OUT_CSV}")


if __name__ == "__main__":
    main()
