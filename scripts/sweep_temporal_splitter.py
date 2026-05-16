"""sweep_temporal_splitter.py

V2.5 PATH A - second attempt: U-shape confidence splitter.

The naive splitter (tune_confidence_temporal_splitter.py) failed because it
destroyed borderline-confidence whole bouts rather than splitting genuine
merged regions. This version requires a *dip-and-recover* pattern:
    [confidence = t_high]  ?  [confidence < t_low for = K frames]  ?  [confidence = t_high]
Only splits at locations that match this U-shape - not at bout edges,
not at sustained low-confidence regions.

Sweep grid:
    t_high ? {0.75, 0.80, 0.85}  (must reach this on both sides of dip)
    t_low  ? {0.45, 0.50, 0.55, 0.60}  (dip must drop below this)
    K      ? {3, 5, 8}  (dip must be at least K frames long)

t_high > t_low always.

Outputs:
    figures/data/R5_yolo_attn_v2_pipeline_test_eval/ushape_splitter_sweep.csv
    figures/data/R5_yolo_attn_v2_pipeline_test_eval/ushape_splitter_pooled.csv
    figures/bout_posthoc_analysis/ushape_splitter_pareto.png
    figures/bout_posthoc_analysis/ushape_splitter_grid.png
"""

from __future__ import annotations

import glob
import itertools
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

FPS = 30.0
CLASS_MAP = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
ID_TO_CLASS = {v: k for k, v in CLASS_MAP.items()}
INV_IDX = CLASS_MAP["investigation"]
OTHER_IDX = CLASS_MAP["other"]
EVAL_CLASSES = [0, 1, 2]
SPLITS = ["test_1", "test_2"]

TAU_HIGH_GRID = [0.75, 0.80, 0.85]
TAU_LOW_GRID  = [0.45, 0.50, 0.55, 0.60]
K_GRID        = [3, 5, 8]


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
# U-SHAPE SPLITTER
# ---------------------------------------------------------------------------
def apply_ushape_splitter(
    pred: np.ndarray,
    prob_inv: np.ndarray,
    tau_high: float,
    tau_low: float,
    K: int,
) -> np.ndarray:
    """Split investigation bouts only at U-shape dip-and-recover patterns.

    For each investigation bout [s, e] in pred:
      Walk through with state machine:
        - look for first frame where prob_inv >= tau_high (call this h1)
        - while in high-confidence run, advance
        - on exit from high run (prob_inv drops below tau_high),
          search forward for either:
            (a) prob_inv >= tau_high again (high-low-high U-shape)
            (b) end of bout (then it's just edge erosion - DON'T split)
        - if we found case (a), examine the inter-high region:
            - find longest contiguous run where prob_inv < tau_low
            - if that run is >= K frames, mark it as OTHER
    """
    out = pred.copy()
    n = len(out)

    # Find investigation bouts in pred
    bouts = []
    in_b, b_start = False, None
    for i in range(n):
        if pred[i] == INV_IDX:
            if not in_b: in_b, b_start = True, i
        else:
            if in_b:
                bouts.append((b_start, i - 1)); in_b = False
    if in_b:
        bouts.append((b_start, n - 1))

    for s, e in bouts:
        # Find frames within [s, e] where prob_inv >= tau_high
        high_mask = prob_inv[s:e + 1] >= tau_high
        # Find contiguous high-confidence runs
        high_runs = []
        in_h, hs = False, None
        for j in range(len(high_mask)):
            if high_mask[j]:
                if not in_h: in_h, hs = True, j
            else:
                if in_h:
                    high_runs.append((hs + s, j - 1 + s)); in_h = False
        if in_h:
            high_runs.append((hs + s, len(high_mask) - 1 + s))

        if len(high_runs) < 2:
            continue   # need at least two high runs to have a U-shape

        # For each adjacent pair of high runs, examine the dip between them
        for (h1s, h1e), (h2s, h2e) in zip(high_runs[:-1], high_runs[1:]):
            dip_s = h1e + 1
            dip_e = h2s - 1
            if dip_e < dip_s:
                continue
            # Find longest contiguous run where prob_inv < tau_low within [dip_s, dip_e]
            best_run_s, best_run_e, best_len = None, None, 0
            cur_s, cur_len = None, 0
            for j in range(dip_s, dip_e + 1):
                if prob_inv[j] < tau_low:
                    if cur_s is None:
                        cur_s = j; cur_len = 1
                    else:
                        cur_len += 1
                else:
                    if cur_s is not None and cur_len > best_len:
                        best_run_s, best_run_e, best_len = cur_s, j - 1, cur_len
                    cur_s, cur_len = None, 0
            if cur_s is not None and cur_len > best_len:
                best_run_s, best_run_e, best_len = cur_s, dip_e, cur_len
            if best_run_s is not None and best_len >= K:
                out[best_run_s:best_run_e + 1] = OTHER_IDX
    return out


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def sweep_video(split, video, configs):
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

    def _measure(pred, label, tau_h, tau_l, K):
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
                "tau_high": tau_h, "tau_low": tau_l, "K": K,
                "config": label, "class": cname,
                "n_pred_bouts": len(pb), "n_gt_bouts": len(gb),
                "tp25": tp25, "fp25": fp25, "fn25": fn25,
                "tp50": tp50, "fp50": fp50, "fn50": fn50,
                "frame_tp": f_tp, "frame_fp": f_fp, "frame_fn": f_fn,
            })

    _measure(pred_default, "baseline", np.nan, np.nan, np.nan)
    for tau_h, tau_l, K in configs:
        sp = apply_ushape_splitter(pred_default, prob_inv, tau_h, tau_l, K)
        _measure(sp, f"th={tau_h}_tl={tau_l}_K={K}", tau_h, tau_l, K)
    return rows


def main():
    configs = [(th, tl, K) for th, tl, K
               in itertools.product(TAU_HIGH_GRID, TAU_LOW_GRID, K_GRID)
               if th > tl]
    discovered = []
    for split in SPLITS:
        sd = INFERENCE_ROOT / split
        if not sd.is_dir(): continue
        for v in sorted(os.listdir(sd)):
            if (sd / v).is_dir():
                discovered.append((split, v))

    print(f"Sweeping {len(discovered)} videos x {len(configs)+1} configs (incl. baseline)")
    all_rows = []
    for split, video in discovered:
        all_rows.extend(sweep_video(split, video, configs))
    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV / "ushape_splitter_sweep.csv", index=False)

    agg = df.groupby(["config", "tau_high", "tau_low", "K", "class"], dropna=False).agg({
        "n_pred_bouts": "sum", "n_gt_bouts": "sum",
        "tp25": "sum", "fp25": "sum", "fn25": "sum",
        "tp50": "sum", "fp50": "sum", "fn50": "sum",
        "frame_tp": "sum", "frame_fp": "sum", "frame_fn": "sum",
    }).reset_index()
    agg["pooled_bout_f1_25"] = agg.apply(lambda r: _f1(r["tp25"], r["fp25"], r["fn25"]), axis=1)
    agg["pooled_bout_f1_50"] = agg.apply(lambda r: _f1(r["tp50"], r["fp50"], r["fn50"]), axis=1)
    agg["pooled_frame_f1"]   = agg.apply(lambda r: _f1(r["frame_tp"], r["frame_fp"], r["frame_fn"]), axis=1)
    agg["bout_count_match"]  = agg["n_pred_bouts"] / agg["n_gt_bouts"].replace(0, np.nan)
    agg.to_csv(OUT_CSV / "ushape_splitter_pooled.csv", index=False)

    inv = agg[agg["class"] == "investigation"].copy()
    base = inv[inv["config"] == "baseline"].iloc[0]
    sp_inv = inv[inv["config"] != "baseline"].copy()

    # Show change vs baseline
    sp_inv["delta_bout_count"]    = sp_inv["bout_count_match"] - base["bout_count_match"]
    sp_inv["delta_frame_f1"]      = sp_inv["pooled_frame_f1"]   - base["pooled_frame_f1"]
    sp_inv["delta_bout_f1_25"]    = sp_inv["pooled_bout_f1_25"] - base["pooled_bout_f1_25"]
    sp_inv["delta_bout_f1_50"]    = sp_inv["pooled_bout_f1_50"] - base["pooled_bout_f1_50"]

    print("\n" + "=" * 90)
    print(f"BASELINE INVESTIGATION (no splitter):")
    print(f"  bout_count = {base['bout_count_match']:.3f}   "
          f"frame_F1 = {base['pooled_frame_f1']:.3f}   "
          f"bout_F1@0.25 = {base['pooled_bout_f1_25']:.3f}   "
          f"bout_F1@0.50 = {base['pooled_bout_f1_50']:.3f}")
    print("=" * 90)

    # Top 10 configs by various criteria
    print("\nTop 5 configs by bout_count gain (closest to GT parity):")
    top_count = sp_inv.nlargest(5, "delta_bout_count")
    for _, r in top_count.iterrows():
        print(f"  th={r['tau_high']:.2f} tl={r['tau_low']:.2f} K={int(r['K'])}  "
              f"bout_count={r['bout_count_match']:.3f} ({r['delta_bout_count']:+.3f})  "
              f"frame_F1={r['pooled_frame_f1']:.3f} ({r['delta_frame_f1']:+.3f})  "
              f"bout_F1@0.50={r['pooled_bout_f1_50']:.3f} ({r['delta_bout_f1_50']:+.3f})")

    print("\nTop 5 configs by bout_F1@0.50 gain:")
    top_f1 = sp_inv.nlargest(5, "delta_bout_f1_50")
    for _, r in top_f1.iterrows():
        print(f"  th={r['tau_high']:.2f} tl={r['tau_low']:.2f} K={int(r['K'])}  "
              f"bout_count={r['bout_count_match']:.3f} ({r['delta_bout_count']:+.3f})  "
              f"frame_F1={r['pooled_frame_f1']:.3f} ({r['delta_frame_f1']:+.3f})  "
              f"bout_F1@0.50={r['pooled_bout_f1_50']:.3f} ({r['delta_bout_f1_50']:+.3f})")

    # Configs that are STRICTLY better than baseline on at least one metric without hurting others
    print("\nConfigs strictly improving SOMETHING without hurting OTHERS (delta within 0.005 = no harm):")
    strict_improve = sp_inv[
        ((sp_inv["delta_bout_f1_50"] > 0.001) | (sp_inv["delta_bout_f1_25"] > 0.001)) &
        (sp_inv["delta_frame_f1"] >= -0.005)
    ]
    if len(strict_improve):
        for _, r in strict_improve.iterrows():
            print(f"  th={r['tau_high']:.2f} tl={r['tau_low']:.2f} K={int(r['K'])}  "
                  f"bout_count={r['bout_count_match']:.3f} ({r['delta_bout_count']:+.3f})  "
                  f"frame_F1={r['pooled_frame_f1']:.3f} ({r['delta_frame_f1']:+.3f})  "
                  f"bout_F1@0.25={r['pooled_bout_f1_25']:.3f} ({r['delta_bout_f1_25']:+.3f})  "
                  f"bout_F1@0.50={r['pooled_bout_f1_50']:.3f} ({r['delta_bout_f1_50']:+.3f})")
    else:
        print("  (NONE - no U-shape splitter config strictly improves on baseline)")

    # ---- Pareto plot ----
    fig, ax = plt.subplots(figsize=(11, 7))
    cmap = plt.get_cmap("viridis")
    for _, r in sp_inv.iterrows():
        color = cmap((r["tau_high"] - min(TAU_HIGH_GRID)) /
                     (max(TAU_HIGH_GRID) - min(TAU_HIGH_GRID) + 1e-9))
        ax.scatter(r["bout_count_match"], r["pooled_bout_f1_50"],
                   s=80, marker="o", color=color, alpha=0.7,
                   edgecolors="white", linewidths=0.4, zorder=3)
    ax.scatter(base["bout_count_match"], base["pooled_bout_f1_50"],
               s=300, marker="*", color="red", edgecolors="black",
               linewidths=1.6, zorder=5)
    ax.annotate("BASELINE",
                (base["bout_count_match"], base["pooled_bout_f1_50"]),
                xytext=(10, -10), textcoords="offset points",
                fontsize=10, fontweight="bold", color="red")
    ax.axvline(1.0, color="#16a34a", linestyle="--", alpha=0.5,
               label="GT bout count parity")
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=min(TAU_HIGH_GRID),
                                                   vmax=max(TAU_HIGH_GRID)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("t_high (high-confidence flank threshold)")
    ax.set_xlabel("Investigation bout count ratio (pred / GT)")
    ax.set_ylabel("Investigation pooled bout F1 @ tIoU 0.50")
    ax.set_title("U-shape splitter sweep - investigation\n"
                 "(only configs above and to the right of red star are improvements)",
                 fontsize=10)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "ushape_splitter_pareto.png", dpi=150, bbox_inches="tight")
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
            for th in TAU_HIGH_GRID:
                sub = sp_inv[(sp_inv["K"] == K) &
                             (sp_inv["tau_high"] == th)].sort_values("tau_low")
                ax.plot(sub["tau_low"], sub[key], marker="o", linewidth=1.5,
                        label=f"K={K}, tH={th}", alpha=0.8)
        ax.axhline(base[key], color="red", linestyle=":", linewidth=2,
                   label="baseline (no splitter)")
        if key == "bout_count_match":
            ax.axhline(1.0, color="#16a34a", linestyle="--", alpha=0.6,
                       label="GT parity")
        ax.set_xlabel("t_low (dip threshold)")
        ax.set_title(title)
        ax.set_ylim(ylim)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best", ncol=2)
    plt.suptitle("U-shape confidence splitter - investigation only\n"
                 "(t_low = dip must drop below this; t_high = must recover above this; "
                 "K = min dip duration in frames)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "ushape_splitter_grid.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nFigures: {OUT_FIGS}")
    print(f"CSVs:    {OUT_CSV}")


if __name__ == "__main__":
    main()
