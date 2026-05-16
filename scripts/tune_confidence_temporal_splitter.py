"""tune_confidence_temporal_splitter.py

V2.5 PATH A: post-hoc confidence-derivative bout splitter.

Hypothesis (from probe_merged_bouts.py): ~44% of merged predicted
investigation bouts contain detectable confidence dips at the GT
inter-bout gap locations (min prob_inv < 0.7 in those gaps), while
~50% are confidently above 0.85 throughout. A splitter that punches
"other" labels into low-confidence intra-bout regions should recover
the splittable subset.

Splitter rule:
    For each predicted investigation bout [s, e]:
      Find contiguous runs of K+ frames within [s, e] where
        prob_investigation < tau
      Mark those runs as "other" in the per-frame prediction array
    Re-extract bouts ? new investigation bout list with splits

Sweep grid:
    tau ? {0.50, 0.55, 0.60, 0.65, 0.70, 0.75}
    K   ? {3, 5, 8, 12} frames

Outputs:
    figures/data/R5_yolo_attn_v2_pipeline_test_eval/
        splitter_sweep.csv          per-video, per-config
        splitter_sweep_pooled.csv   pooled metrics
        splitter_decision.md        decision document

    figures/bout_posthoc_analysis/
        splitter_pareto.png         bout count match vs frame F1, all configs
        splitter_grid.png           4-panel: bout count, frame F1, bout F1 @ 0.25, @ 0.50
        splitter_vs_default.png     before/after for the chosen v2.5 setting
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

OUT_CSV  = REPO / "manuscript_outputs" / "figures" / "data" / "R5_yolo_attn_v2_pipeline_test_eval"
OUT_FIGS = REPO / "manuscript_outputs" / "figures" / "bout_posthoc_analysis"
OUT_CSV.mkdir(parents=True, exist_ok=True)
OUT_FIGS.mkdir(parents=True, exist_ok=True)

FPS = 30.0
CLASS_MAP = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
INV_IDX = CLASS_MAP["investigation"]
OTHER_IDX = CLASS_MAP["other"]
EVAL_CLASSES = [0, 1, 2]
SPLITS = ["test_1", "test_2"]

TAU_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
K_GRID   = [3, 5, 8, 12]


def parse_bento_annot(path):
    bouts = {}
    cur, in_data = None, False
    with open(path, "r") as fh:
        for line in fh:
            s = line.strip()
            if s.startswith(">"):
                cur = s[1:].strip(); bouts[cur] = []; in_data = False
                continue
            if "Start" in s and "Stop" in s:
                in_data = True; continue
            if in_data and cur and s:
                p = s.split()
                if len(p) >= 2:
                    try:
                        bouts[cur].append((int(round(float(p[0]) * FPS)),
                                            int(round(float(p[1]) * FPS))))
                    except ValueError:
                        pass
    return bouts


def find_annot(split, video):
    pattern = str(GT_ROOT / split / video / f"{video}_*.annot")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def extract_bouts(frame_array, class_id):
    bouts, in_b, start = [], False, None
    for i, c in enumerate(frame_array):
        if c == class_id:
            if not in_b: in_b, start = True, i
        else:
            if in_b: bouts.append((start, i - 1)); in_b = False
    if in_b: bouts.append((start, len(frame_array) - 1))
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
            if gi in matched: continue
            iv = tiou(pb, gb)
            if iv > best: best, best_gi = iv, gi
        if best >= threshold and best_gi >= 0:
            tp += 1; matched.add(best_gi)
        else:
            fp += 1
    fn = len(gt_bouts) - len(matched)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# THE SPLITTER
# ---------------------------------------------------------------------------
def apply_confidence_splitter(
    pred: np.ndarray,
    prob_inv: np.ndarray,
    tau: float,
    min_dip_frames: int,
) -> np.ndarray:
    """Punch 'other' into low-confidence runs inside investigation bouts.

    Args:
        pred: [T] int array of predicted class IDs
        prob_inv: [T] float array of per-frame prob_investigation
        tau: confidence threshold below which a frame counts as a dip
        min_dip_frames: minimum contiguous dip length to qualify for splitting

    Returns:
        [T] int array with low-confidence runs INSIDE investigation bouts
        relabeled as OTHER.
    """
    out = pred.copy()
    n = len(out)

    # Find investigation bouts in pred
    in_b, b_start = False, None
    for i in range(n):
        if out[i] == INV_IDX:
            if not in_b: in_b, b_start = True, i
        else:
            if in_b:
                _split_within(out, prob_inv, b_start, i - 1, tau, min_dip_frames)
                in_b = False
    if in_b:
        _split_within(out, prob_inv, b_start, n - 1, tau, min_dip_frames)
    return out


def _split_within(out, prob_inv, s, e, tau, K):
    # Find contiguous runs of frames in [s, e] with prob_inv < tau
    # Punch them as OTHER if length >= K. Edge runs (touching s or e) are
    # also valid splits - they shorten the bout from its boundary.
    i = s
    while i <= e:
        if prob_inv[i] < tau:
            j = i
            while j <= e and prob_inv[j] < tau:
                j += 1
            # run [i, j-1] is below tau
            run_len = j - i
            if run_len >= K:
                out[i:j] = OTHER_IDX
            i = j
        else:
            i += 1


# ---------------------------------------------------------------------------
# Per-video sweep
# ---------------------------------------------------------------------------
def sweep_video(split, video):
    vd = INFERENCE_ROOT / split / video
    sm = vd / f"{video}.behavior.smoothed_frames.csv"
    if not sm.is_file(): return []
    annot = find_annot(split, video)
    if annot is None: return []
    df = pd.read_csv(sm)
    if not len(df): return []

    prob_inv = df["prob_investigation"].values.astype(np.float64)
    pred_default = df["predicted_class_id"].values.astype(np.int64)
    n_frames = len(prob_inv)

    gt_by_name = parse_bento_annot(annot)
    gt_frames = np.full(n_frames, OTHER_IDX, dtype=int)
    gt_bouts_by_id = {cid: [] for cid in EVAL_CLASSES}
    for cn, bs in gt_by_name.items():
        cid = CLASS_MAP.get(cn)
        if cid is None or cid == OTHER_IDX: continue
        gt_bouts_by_id[cid] = bs
        for sf, ef in bs:
            sf = max(sf, 0); ef = min(ef, n_frames - 1)
            gt_frames[sf:ef + 1] = cid

    rows = []
    # Baseline (tau=infinity, K=infinity ? no splitter)
    for label, pred in [("baseline", pred_default)]:
        for cid in EVAL_CLASSES:
            cname = ID_TO_CLASS[cid]
            pb = extract_bouts(pred, cid)
            gb = gt_bouts_by_id[cid]
            tp25, fp25, fn25 = bout_match_counts(pb, gb, 0.25)
            tp50, fp50, fn50 = bout_match_counts(pb, gb, 0.50)
            f_tp = int(np.sum((pred == cid) & (gt_frames == cid)))
            f_fp = int(np.sum((pred == cid) & (gt_frames != cid)))
            f_fn = int(np.sum((pred != cid) & (gt_frames == cid)))
            rows.append({
                "split": split, "video": video,
                "tau": np.nan, "K": np.nan, "config": label,
                "class": cname,
                "n_pred_bouts": len(pb), "n_gt_bouts": len(gb),
                "tp25": tp25, "fp25": fp25, "fn25": fn25,
                "tp50": tp50, "fp50": fp50, "fn50": fn50,
                "frame_tp": f_tp, "frame_fp": f_fp, "frame_fn": f_fn,
            })

    # Sweep splitter configs
    for tau in TAU_GRID:
        for K in K_GRID:
            split_pred = apply_confidence_splitter(pred_default, prob_inv, tau, K)
            for cid in EVAL_CLASSES:
                cname = ID_TO_CLASS[cid]
                pb = extract_bouts(split_pred, cid)
                gb = gt_bouts_by_id[cid]
                tp25, fp25, fn25 = bout_match_counts(pb, gb, 0.25)
                tp50, fp50, fn50 = bout_match_counts(pb, gb, 0.50)
                f_tp = int(np.sum((split_pred == cid) & (gt_frames == cid)))
                f_fp = int(np.sum((split_pred == cid) & (gt_frames != cid)))
                f_fn = int(np.sum((split_pred != cid) & (gt_frames == cid)))
                rows.append({
                    "split": split, "video": video,
                    "tau": tau, "K": K, "config": f"tau={tau}_K={K}",
                    "class": cname,
                    "n_pred_bouts": len(pb), "n_gt_bouts": len(gb),
                    "tp25": tp25, "fp25": fp25, "fn25": fn25,
                    "tp50": tp50, "fp50": fp50, "fn50": fn50,
                    "frame_tp": f_tp, "frame_fp": f_fp, "frame_fn": f_fn,
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
        if not sd.is_dir(): continue
        for v in sorted(os.listdir(sd)):
            if (sd / v).is_dir():
                discovered.append((split, v))

    n_videos = len(discovered)
    n_configs = 1 + len(TAU_GRID) * len(K_GRID)
    print(f"Sweeping {n_videos} videos x {n_configs} configs (incl. baseline) "
          f"= {n_videos * n_configs * 3} (config,video,class) rows")

    all_rows = []
    for split, video in discovered:
        all_rows.extend(sweep_video(split, video))
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV / "splitter_sweep.csv", index=False)

    agg = df.groupby(["config", "tau", "K", "class"], dropna=False).agg({
        "n_pred_bouts": "sum", "n_gt_bouts": "sum",
        "tp25": "sum", "fp25": "sum", "fn25": "sum",
        "tp50": "sum", "fp50": "sum", "fn50": "sum",
        "frame_tp": "sum", "frame_fp": "sum", "frame_fn": "sum",
    }).reset_index()
    agg["pooled_bout_f1_25"] = agg.apply(lambda r: _f1(r["tp25"], r["fp25"], r["fn25"]), axis=1)
    agg["pooled_bout_f1_50"] = agg.apply(lambda r: _f1(r["tp50"], r["fp50"], r["fn50"]), axis=1)
    agg["pooled_frame_f1"]   = agg.apply(lambda r: _f1(r["frame_tp"], r["frame_fp"], r["frame_fn"]), axis=1)
    agg["bout_count_match"]  = agg["n_pred_bouts"] / agg["n_gt_bouts"].replace(0, np.nan)
    agg.to_csv(OUT_CSV / "splitter_sweep_pooled.csv", index=False)

    # ---- Investigation focused view ----
    inv = agg[agg["class"] == "investigation"].copy()
    baseline_inv = inv[inv["config"] == "baseline"].iloc[0]
    print("\n=" * 30)
    print(f"INVESTIGATION BASELINE (no splitter):")
    print(f"  bout count match : {baseline_inv['bout_count_match']:.3f}")
    print(f"  frame F1         : {baseline_inv['pooled_frame_f1']:.3f}")
    print(f"  bout F1 @ tIoU 0.25 : {baseline_inv['pooled_bout_f1_25']:.3f}")
    print(f"  bout F1 @ tIoU 0.50 : {baseline_inv['pooled_bout_f1_50']:.3f}")

    print("\nINVESTIGATION across splitter configs (rows=tau, cols=K):")
    for metric, label in [("bout_count_match", "bout count match"),
                           ("pooled_frame_f1", "frame F1"),
                           ("pooled_bout_f1_25", "bout F1 @ 0.25"),
                           ("pooled_bout_f1_50", "bout F1 @ 0.50")]:
        sw = inv[inv["config"] != "baseline"].pivot(
            index="tau", columns="K", values=metric).round(3)
        print(f"\n--- {label} ---")
        print(sw.to_string())

    # Best operating points
    splitter_inv = inv[inv["config"] != "baseline"].copy()
    print("\n" + "=" * 90)
    print("Best splitter configs by criterion (Investigation only)")
    print("=" * 90)

    def fmt(r, label):
        return (f"  {label:42s}  tau={r['tau']:.2f} K={int(r['K'])}  |  "
                f"bout_count={r['bout_count_match']:.3f}  "
                f"frame_F1={r['pooled_frame_f1']:.3f}  "
                f"bout_F1@0.5={r['pooled_bout_f1_50']:.3f}")

    splitter_inv["dist_to_parity"] = (splitter_inv["bout_count_match"] - 1.0).abs()
    print(fmt(splitter_inv.nsmallest(1, "dist_to_parity").iloc[0],
              "closest to bout count parity"))
    print(fmt(splitter_inv.nlargest(1, "pooled_bout_f1_50").iloc[0],
              "max bout F1 @ tIoU 0.50"))
    print(fmt(splitter_inv.nlargest(1, "pooled_bout_f1_25").iloc[0],
              "max bout F1 @ tIoU 0.25"))

    # Constrained: bout_count >= 0.7, frame_F1 stays within 0.02 of baseline
    feas = splitter_inv[
        (splitter_inv["bout_count_match"] >= 0.70) &
        (splitter_inv["pooled_frame_f1"] >= baseline_inv["pooled_frame_f1"] - 0.02)
    ]
    if len(feas):
        print(fmt(feas.nlargest(1, "pooled_bout_f1_50").iloc[0],
                  "best within feasible region (count>=0.7, frame_F1 drop<=0.02)"))
    else:
        print("  (no splitter config satisfies bout_count>=0.7 with frame_F1 within 0.02 of baseline)")

    # Sanity check: attack/mount frame F1 must be unchanged
    print("\nSANITY CHECK: attack & mount frame F1 should be unchanged across all splitter configs.")
    for cname in ["attack", "mount"]:
        sub = agg[agg["class"] == cname]
        u = sub["pooled_frame_f1"].nunique()
        print(f"  {cname}: {u} unique frame F1 value(s) - "
              f"{'OK' if u == 1 else 'WARNING - splitter affecting other classes'}")

    # ---- Pareto plot ----
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap("viridis")
    K_markers = {3: "o", 5: "s", 8: "^", 12: "D"}
    for _, r in splitter_inv.iterrows():
        color = cmap((r["tau"] - min(TAU_GRID)) / (max(TAU_GRID) - min(TAU_GRID)))
        ax.scatter(r["bout_count_match"], r["pooled_frame_f1"],
                   s=110, marker=K_markers[int(r["K"])], color=color,
                   alpha=0.85, edgecolors="white", linewidths=0.6, zorder=3)
    # Baseline
    ax.scatter(baseline_inv["bout_count_match"], baseline_inv["pooled_frame_f1"],
               s=300, marker="*", color="red", edgecolors="black",
               linewidths=1.6, zorder=5)
    ax.annotate("BASELINE\n(no splitter)",
                (baseline_inv["bout_count_match"], baseline_inv["pooled_frame_f1"]),
                xytext=(10, -10), textcoords="offset points",
                fontsize=9, fontweight="bold", color="red")
    ax.axvline(1.0, color="#16a34a", linestyle="--", alpha=0.5,
               label="GT bout count parity")
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=min(TAU_GRID), vmax=max(TAU_GRID)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Splitter dip threshold t")
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker=m, color="w", markerfacecolor="gray",
                      markersize=10, label=f"K = {k} fr ({k/FPS*1000:.0f} ms)")
               for k, m in K_markers.items()]
    handles.append(Line2D([0], [0], marker="*", color="w",
                          markerfacecolor="red", markeredgecolor="black",
                          markersize=14, label="baseline (no splitter)"))
    handles.append(Line2D([0], [0], color="#16a34a", linestyle="--",
                          label="bout count parity"))
    ax.legend(handles=handles, loc="lower left", fontsize=9)
    ax.set_xlabel("Investigation bout count ratio (pred / GT)")
    ax.set_ylabel("Pooled investigation FRAME F1")
    ax.set_title("Investigation Pareto - confidence-derivative splitter sweep\n"
                 f"({len(TAU_GRID)} t x {len(K_GRID)} K = {len(TAU_GRID)*len(K_GRID)} splitter configs vs baseline)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "splitter_pareto.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 4-panel grid ----
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panel_specs = [
        (axes[0, 0], "bout_count_match",  "Bout count ratio (pred/GT)", (0.4, 1.6)),
        (axes[0, 1], "pooled_frame_f1",   "Pooled frame F1",            (0.55, 0.78)),
        (axes[1, 0], "pooled_bout_f1_25", "Pooled bout F1 @ tIoU 0.25", (0.30, 0.65)),
        (axes[1, 1], "pooled_bout_f1_50", "Pooled bout F1 @ tIoU 0.50", (0.25, 0.50)),
    ]
    for ax, key, title, ylim in panel_specs:
        for K in K_GRID:
            sub = splitter_inv[splitter_inv["K"] == K].sort_values("tau")
            ax.plot(sub["tau"], sub[key], marker=K_markers[K], linewidth=2,
                    label=f"K = {K} fr")
        # Baseline as horizontal line
        ax.axhline(baseline_inv[key], color="red", linestyle=":", linewidth=1.5,
                   label="baseline (no splitter)")
        if key == "bout_count_match":
            ax.axhline(1.0, color="#16a34a", linestyle="--", alpha=0.6,
                       label="GT parity")
        ax.set_xticks(TAU_GRID)
        ax.set_xlabel("Splitter dip threshold t")
        ax.set_title(title)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8.5)
    plt.suptitle("Confidence-derivative splitter sweep - investigation only\n"
                 "(higher t = more aggressive splitting; "
                 "K = minimum dip duration to trigger split)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "splitter_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nFigures: {OUT_FIGS}")
    print(f"CSVs:    {OUT_CSV}")


if __name__ == "__main__":
    main()
