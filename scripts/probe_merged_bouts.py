"""probe_merged_bouts.py

Diagnostic: when the model emits ONE predicted investigation bout that
overlaps TWO OR MORE human-annotated investigation bouts, what does
prob_investigation look like across that interval?

If there are visible dips at the GT bout boundaries ? a confidence-
derivative bout splitter at inference time will work without retraining.
If the prob is uniformly high ? only train-time supervision (auxiliary
boundary head) can split these bouts; v2.5 is a real architecture
project, not a quick win.

Reads:
    R5_yolo_attn_v2_pipeline_mars_test_eval/<split>/<video>/
        <video>.behavior.smoothed_frames.csv

Writes:
    figures/data/R5_yolo_attn_v2_pipeline_test_eval/merged_bout_stats.csv
    figures/bout_posthoc_analysis/merged_bout_prob_traces.png
    figures/bout_posthoc_analysis/merged_bout_dip_distribution.png
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
OUT_CSV = REPO / "manuscript_outputs" / "figures" / "data" / "R5_yolo_attn_v2_pipeline_test_eval"
OUT_FIGS = REPO / "manuscript_outputs" / "figures" / "bout_posthoc_analysis"

FPS = 30.0
CLASS_MAP = {"attack": 0, "investigation": 1, "mount": 2, "other": 3}
SPLITS = ["test_1", "test_2"]
INV_IDX = CLASS_MAP["investigation"]


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


def extract_predicted_bouts(pred_class_id, target_cid):
    bouts = []
    in_b, start = False, None
    for i, c in enumerate(pred_class_id):
        if c == target_cid:
            if not in_b:
                in_b, start = True, i
        else:
            if in_b:
                bouts.append((start, i - 1)); in_b = False
    if in_b:
        bouts.append((start, len(pred_class_id) - 1))
    return bouts


def find_annot(split, video):
    pattern = str(GT_ROOT / split / video / f"{video}_*.annot")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def main():
    discovered = []
    for split in SPLITS:
        sd = INFERENCE_ROOT / split
        if not sd.is_dir(): continue
        for v in sorted(os.listdir(sd)):
            if (sd / v).is_dir():
                discovered.append((split, v))

    merged_records = []  # one per merged bout: (video, pred_bout, contained_gt_bouts, prob_trace)
    for split, video in discovered:
        sm = INFERENCE_ROOT / split / video / f"{video}.behavior.smoothed_frames.csv"
        if not sm.is_file(): continue
        annot = find_annot(split, video)
        if annot is None: continue
        df = pd.read_csv(sm)
        if not len(df): continue

        prob_inv = df["prob_investigation"].values.astype(np.float64)
        pred_cls = df["predicted_class_id"].values.astype(int)
        gt_by_name = parse_bento_annot(annot)
        gt_inv_bouts = gt_by_name.get("investigation", [])

        pred_bouts = extract_predicted_bouts(pred_cls, INV_IDX)

        for ps, pe in pred_bouts:
            # Find GT investigation bouts whose CENTRE falls inside [ps, pe]
            contained = []
            for gs, ge in gt_inv_bouts:
                gc = (gs + ge) // 2
                if ps <= gc <= pe:
                    contained.append((max(gs, ps), min(ge, pe)))
            if len(contained) >= 2:
                # Merged predicted bout - extract prob trace
                trace = prob_inv[ps:pe + 1]
                merged_records.append({
                    "split": split,
                    "video": video,
                    "pred_start": ps,
                    "pred_end": pe,
                    "pred_dur_frames": pe - ps + 1,
                    "n_contained_gt_bouts": len(contained),
                    "trace": trace.tolist(),
                    "gt_starts_local": [gs - ps for gs, _ in contained],
                    "gt_ends_local":   [ge - ps for _, ge in contained],
                })

    print(f"Found {len(merged_records)} merged predicted investigation bouts "
          f"(containing 2+ GT bouts each)")

    if not merged_records:
        print("No merged bouts found - diagnostic inconclusive.")
        return

    # ---- Per-merged-bout dip statistic ----
    # For each merged bout with K contained GT bouts, look at prob_inv at
    # the K-1 inter-bout gap regions (frames between consecutive GT bouts).
    # Compute: min prob_inv in the gap region.
    dip_rows = []
    for r in merged_records:
        trace = np.array(r["trace"])
        starts = sorted(r["gt_starts_local"])
        ends   = sorted(r["gt_ends_local"])
        # iterate gap regions
        for i in range(len(starts) - 1):
            gap_start = ends[i] + 1
            gap_end   = starts[i + 1] - 1
            if gap_end < gap_start:
                continue  # GT bouts touch or overlap
            gap = trace[gap_start:gap_end + 1]
            if len(gap) == 0:
                continue
            dip_rows.append({
                "split": r["split"],
                "video": r["video"],
                "pred_start": r["pred_start"],
                "gap_local_start": gap_start,
                "gap_local_end": gap_end,
                "gap_dur_frames": gap_end - gap_start + 1,
                "min_prob_inv_in_gap": float(gap.min()),
                "mean_prob_inv_in_gap": float(gap.mean()),
                "min_prob_inv_in_bout": float(trace.min()),
                "mean_prob_inv_in_bout": float(trace.mean()),
                # Bout boundaries: prob_inv just before the gap (last frame of GT_i)
                # and just after (first frame of GT_{i+1})
                "prob_inv_at_gt_end":   float(trace[max(ends[i],   0)]),
                "prob_inv_at_gt_start": float(trace[min(starts[i + 1], len(trace) - 1)]),
            })
    dip_df = pd.DataFrame(dip_rows)
    dip_df.to_csv(OUT_CSV / "merged_bout_dip_stats.csv", index=False)

    # ---- Trace plots: 6 representative merged bouts ----
    # Sort by descending number of contained GTs to get the most interesting cases
    sorted_records = sorted(merged_records, key=lambda r: r["n_contained_gt_bouts"], reverse=True)
    show = sorted_records[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    for ax, r in zip(axes.flatten(), show):
        trace = np.array(r["trace"])
        x = np.arange(len(trace)) / FPS  # seconds within the merged bout
        ax.plot(x, trace, color="#1a73e8", linewidth=1.5, label="prob_inv")
        ax.axhline(0.4, color="red", linestyle=":", alpha=0.6,
                   label="decoder thr (0.4)")
        ax.axhline(0.7, color="orange", linestyle=":", alpha=0.4,
                   label="thr=0.7")
        # Shade GT investigation bouts in green
        for gs_l, ge_l in zip(r["gt_starts_local"], r["gt_ends_local"]):
            ax.axvspan(gs_l / FPS, ge_l / FPS, color="#2ca02c", alpha=0.18)
        ax.set_ylim(0, 1.0)
        ax.set_xlabel("seconds within predicted bout")
        ax.set_ylabel("prob_investigation")
        ax.set_title(f"{r['video'][:28]}\n"
                     f"pred bout {r['pred_start']}-{r['pred_end']} "
                     f"({r['pred_dur_frames']} fr, {r['pred_dur_frames']/FPS:.1f} s) "
                     f"contains {r['n_contained_gt_bouts']} GT bouts",
                     fontsize=9)
        ax.grid(alpha=0.3)
        if ax is axes[0, 0]:
            ax.legend(loc="lower right", fontsize=7.5)
    plt.suptitle("prob_investigation traces inside MERGED predicted bouts\n"
                 "(green shade = human-annotated GT investigation bouts; "
                 "white between green = inter-bout gaps)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "merged_bout_prob_traces.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    # ---- Distribution of min-prob-in-gap ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    if len(dip_df):
        ax.hist(dip_df["min_prob_inv_in_gap"], bins=40,
                color="#1a73e8", edgecolor="white", alpha=0.85)
        ax.axvline(0.4, color="red", linestyle="--",
                   label=f"decoder thr 0.4 - would split {(dip_df['min_prob_inv_in_gap'] < 0.4).sum()}/{len(dip_df)} gaps")
        ax.axvline(0.7, color="orange", linestyle="--",
                   label=f"thr 0.7 - would split {(dip_df['min_prob_inv_in_gap'] < 0.7).sum()}/{len(dip_df)} gaps")
        ax.axvline(dip_df["min_prob_inv_in_gap"].median(),
                   color="black", linestyle=":",
                   label=f"median = {dip_df['min_prob_inv_in_gap'].median():.3f}")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("min prob_investigation in inter-bout gap")
    ax.set_ylabel("count of inter-bout gaps")
    ax.set_title("Confidence dip distribution at GT inter-bout gaps\n"
                 "(if mass is at high prob ? no detectable dips ? "
                 "confidence-splitter cannot work)",
                 fontsize=10)
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    if len(dip_df):
        # Mean prob in gap vs gap duration
        ax.scatter(dip_df["gap_dur_frames"] / FPS,
                   dip_df["mean_prob_inv_in_gap"],
                   s=22, color="#1a73e8", alpha=0.6, edgecolors="white")
    ax.set_xlabel("inter-bout gap duration (s)")
    ax.set_ylabel("mean prob_investigation in gap")
    ax.set_ylim(0, 1.0)
    ax.set_title("Gap-duration vs mean-confidence-in-gap\n"
                 "(short gaps may be unrecoverable; "
                 "long gaps with high confidence = model genuinely sees activity)",
                 fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "merged_bout_dip_distribution.png", dpi=150,
                bbox_inches="tight")
    plt.close()

    # ---- Console summary ----
    print(f"\nFound {len(dip_df)} inter-bout gaps inside {len(merged_records)} merged predicted bouts")
    if len(dip_df):
        print("\nDip statistics across all inter-bout gaps:")
        print(f"  median min-prob-in-gap: {dip_df['min_prob_inv_in_gap'].median():.3f}")
        print(f"  mean   min-prob-in-gap: {dip_df['min_prob_inv_in_gap'].mean():.3f}")
        print(f"  std    min-prob-in-gap: {dip_df['min_prob_inv_in_gap'].std():.3f}")
        print(f"  fraction of gaps with min_prob < 0.4: "
              f"{(dip_df['min_prob_inv_in_gap'] < 0.4).mean():.1%}")
        print(f"  fraction of gaps with min_prob < 0.6: "
              f"{(dip_df['min_prob_inv_in_gap'] < 0.6).mean():.1%}")
        print(f"  fraction of gaps with min_prob < 0.7: "
              f"{(dip_df['min_prob_inv_in_gap'] < 0.7).mean():.1%}")
        print(f"  fraction of gaps with min_prob < 0.8: "
              f"{(dip_df['min_prob_inv_in_gap'] < 0.8).mean():.1%}")
        print("\nGap duration statistics (frames):")
        print(f"  median: {int(dip_df['gap_dur_frames'].median())}  "
              f"({dip_df['gap_dur_frames'].median()/FPS:.2f} s)")
        print(f"  mean:   {dip_df['gap_dur_frames'].mean():.1f}  "
              f"({dip_df['gap_dur_frames'].mean()/FPS:.2f} s)")
    print(f"\nFigures: {OUT_FIGS}")
    print(f"CSV:     {OUT_CSV / 'merged_bout_dip_stats.csv'}")


if __name__ == "__main__":
    main()
