"""analyze_deep_bout_metrics.py

Deep bout-level characterization of full-model predictions vs human GT
(MARS .annot files), beyond the pooled F1 numbers in
evaluate_attention_model_test_set.py.

Reads:
    full_model_mars_test_eval/<split>/<video>/
        <video>.behavior.smoothed_frames.csv     (per-frame predictions)
    ./MARS_data/<split>/<video>/<video>_*.annot   (GT)

Computes for each (video, class) pair:
    - GT bout list with onset/offset/duration
    - Pred bout list with onset/offset/duration
    - Greedy 1-to-1 match at multiple tIoU thresholds (0.1, 0.25, 0.5, 0.75)
    - Onset error (pred_start - gt_start, frames)
    - Offset error (pred_end - gt_end, frames)
    - Duration error (pred_dur - gt_dur, frames + ratio)
    - Cross-class confusion: when we predict class X, what fraction of
      frames in that bout were GT-labeled as something else?

Writes:
    figures/data/R5_yolo_attn_v2_pipeline_test_eval/
        bout_pairs.csv          one row per matched/unmatched bout
        tIoU_sweep.csv          per-class F1 across tIoU thresholds
        per_video_bout_f1.csv   per-video, per-class bout F1 @ tIoU 0.5
        bout_summary.csv        per-class deep-stats summary
        deep_bout_analysis.md   markdown report

    figures/bout_posthoc_analysis/
        tiou_sweep.png             F1 vs tIoU threshold per class
        onset_offset_dist.png      histogram per class
        duration_scatter.png       pred vs GT duration, matched bouts
        per_video_heatmap.png      per-video bout F1 heatmap
        bout_count_scatter.png     pred bout count vs GT bout count per video
        cross_class_confusion.png  bout-level cross-class overlap

Run from repo root:
    python scripts/analyze_deep_bout_metrics.py
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
EVAL_CLASSES = [0, 1, 2]
SPLITS = ["test_1", "test_2"]
TIOU_GRID = [0.1, 0.25, 0.5, 0.75]

CLASS_COLORS = {"attack": "#d62728", "investigation": "#1a73e8", "mount": "#2ca02c"}


# ---------------------------------------------------------------------------
# Parsers (verbatim from evaluate_attention_model_test_set.py)
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
                        start_f = int(round(start_s * fps))
                        end_f = int(round(stop_s * fps))
                        bouts_by_class[current_class].append((start_f, end_f))
                    except ValueError:
                        pass
    gt_frames = np.full(total_frames, CLASS_MAP["other"], dtype=int)
    for cname, bouts in bouts_by_class.items():
        cid = CLASS_MAP.get(cname)
        if cid is None:
            continue
        for sf, ef in bouts:
            sf = max(sf, 0); ef = min(ef, total_frames - 1)
            gt_frames[sf:ef + 1] = cid
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


def find_annot(split, video):
    pattern = str(GT_ROOT / split / video / f"{video}_*.annot")
    matches = sorted(glob.glob(pattern))
    return matches[0] if matches else None


def get_total_frames(video_dir, video):
    sm = video_dir / f"{video}.behavior.smoothed_frames.csv"
    if sm.is_file():
        df = pd.read_csv(sm)
        if len(df):
            return int(df["frame_idx"].max()) + 1
    return None


# ---------------------------------------------------------------------------
# Greedy 1-to-1 match returning per-bout pairs
# ---------------------------------------------------------------------------
def match_bouts(pred_bouts, gt_bouts):
    """Return list of dicts: each predicted bout paired with its best GT bout
    (or unmatched), and unmatched GT bouts as separate rows."""
    rows = []
    matched_gt = set()
    for pi, pb in enumerate(pred_bouts):
        best_iou, best_gi = 0.0, -1
        for gi, gb in enumerate(gt_bouts):
            if gi in matched_gt:
                continue
            iv = tiou(pb, gb)
            if iv > best_iou:
                best_iou, best_gi = iv, gi
        if best_gi >= 0 and best_iou > 0:
            gb = gt_bouts[best_gi]
            matched_gt.add(best_gi)
            rows.append(dict(
                kind="matched",
                pred_idx=pi, gt_idx=best_gi,
                pred_start=pb[0], pred_end=pb[1], pred_dur=pb[1] - pb[0] + 1,
                gt_start=gb[0],   gt_end=gb[1],   gt_dur=gb[1] - gb[0] + 1,
                tiou=best_iou,
                onset_err=pb[0] - gb[0],
                offset_err=pb[1] - gb[1],
                dur_err=(pb[1] - pb[0] + 1) - (gb[1] - gb[0] + 1),
                dur_ratio=(pb[1] - pb[0] + 1) / max(1, gb[1] - gb[0] + 1),
            ))
        else:
            rows.append(dict(
                kind="pred_unmatched",
                pred_idx=pi, gt_idx=-1,
                pred_start=pb[0], pred_end=pb[1], pred_dur=pb[1] - pb[0] + 1,
                gt_start=-1, gt_end=-1, gt_dur=-1,
                tiou=0.0, onset_err=np.nan, offset_err=np.nan,
                dur_err=np.nan, dur_ratio=np.nan,
            ))
    for gi, gb in enumerate(gt_bouts):
        if gi in matched_gt:
            continue
        rows.append(dict(
            kind="gt_unmatched",
            pred_idx=-1, gt_idx=gi,
            pred_start=-1, pred_end=-1, pred_dur=-1,
            gt_start=gb[0], gt_end=gb[1], gt_dur=gb[1] - gb[0] + 1,
            tiou=0.0, onset_err=np.nan, offset_err=np.nan,
            dur_err=np.nan, dur_ratio=np.nan,
        ))
    return rows


def cross_class_overlap(pred_bouts, gt_frames, target_cid):
    """For predicted bouts of target_cid, count frames inside that bout
    that were GT-labeled as each class. Returns array of shape [n_classes].
    Pooled across all predicted bouts of this class on this video."""
    counts = np.zeros(4, dtype=np.int64)
    for ps, pe in pred_bouts:
        seg = gt_frames[ps:pe + 1]
        for cid in range(4):
            counts[cid] += int(np.sum(seg == cid))
    return counts


# ---------------------------------------------------------------------------
# Build per-video bout pair table
# ---------------------------------------------------------------------------
def build_pair_table():
    rows = []
    cross_class = {c: np.zeros(4, dtype=np.int64) for c in EVAL_CLASSES}
    per_video_bout_f1 = []

    discovered = []
    for split in SPLITS:
        sd = INFERENCE_ROOT / split
        if not sd.is_dir():
            continue
        for v in sorted(os.listdir(sd)):
            if (sd / v).is_dir():
                discovered.append((split, v))
    print(f"discovered {len(discovered)} video dirs")

    for split, video in discovered:
        vd = INFERENCE_ROOT / split / video
        sm = vd / f"{video}.behavior.smoothed_frames.csv"
        if not sm.is_file():
            print(f"  skip {video}: no smoothed CSV")
            continue
        annot = find_annot(split, video)
        if annot is None:
            print(f"  skip {video}: no annot")
            continue
        total = get_total_frames(vd, video)
        if total is None:
            continue
        gt_by_name, gt_frames = parse_bento_annot(annot, total, FPS)
        pred_frames = load_smoothed_predictions(str(sm), total)

        per_video_row = {"split": split, "video": video, "total_frames": total}
        for cid in EVAL_CLASSES:
            cname = ID_TO_CLASS[cid]
            gt_bouts = []
            for cn, bs in gt_by_name.items():
                if CLASS_MAP.get(cn) == cid:
                    gt_bouts = bs
                    break
            pred_bouts = extract_bouts(pred_frames, cid)
            pairs = match_bouts(pred_bouts, gt_bouts)
            for r in pairs:
                r.update(dict(split=split, video=video,
                              class_id=cid, class_name=cname))
                rows.append(r)
            # Per-video bout F1 @ tIoU 0.5
            tp50 = sum(1 for r in pairs if r["kind"] == "matched" and r["tiou"] >= 0.5)
            fp50 = sum(1 for r in pairs if r["kind"] == "pred_unmatched") + \
                   sum(1 for r in pairs if r["kind"] == "matched" and r["tiou"] < 0.5)
            fn50 = sum(1 for r in pairs if r["kind"] == "gt_unmatched") + \
                   sum(1 for r in pairs if r["kind"] == "matched" and r["tiou"] < 0.5)
            p = tp50 / (tp50 + fp50) if (tp50 + fp50) > 0 else 0.0
            r_ = tp50 / (tp50 + fn50) if (tp50 + fn50) > 0 else 0.0
            f1 = 2 * p * r_ / (p + r_) if (p + r_) > 0 else 0.0
            per_video_row[f"bout_f1_50_{cname}"] = f1
            per_video_row[f"n_pred_{cname}"] = len(pred_bouts)
            per_video_row[f"n_gt_{cname}"] = len(gt_bouts)
            cross_class[cid] += cross_class_overlap(pred_bouts, gt_frames, cid)
        per_video_bout_f1.append(per_video_row)

    return pd.DataFrame(rows), pd.DataFrame(per_video_bout_f1), cross_class


# ---------------------------------------------------------------------------
# tIoU sweep summary
# ---------------------------------------------------------------------------
def tiou_sweep(pairs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cid in EVAL_CLASSES:
        cname = ID_TO_CLASS[cid]
        sub = pairs[pairs["class_id"] == cid]
        n_pred = int(((sub["kind"] == "matched") | (sub["kind"] == "pred_unmatched")).sum())
        n_gt   = int(((sub["kind"] == "matched") | (sub["kind"] == "gt_unmatched")).sum())
        for thr in TIOU_GRID:
            tp = int(((sub["kind"] == "matched") & (sub["tiou"] >= thr)).sum())
            fp = n_pred - tp
            fn = n_gt - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            out.append({
                "class": cname, "tiou": thr,
                "tp": tp, "fp": fp, "fn": fn,
                "precision": p, "recall": r, "f1": f1,
                "n_pred_total": n_pred, "n_gt_total": n_gt,
            })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Per-class summary
# ---------------------------------------------------------------------------
def class_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cid in EVAL_CLASSES:
        cname = ID_TO_CLASS[cid]
        sub = pairs[pairs["class_id"] == cid]
        matched = sub[sub["kind"] == "matched"]
        n_pred = int(((sub["kind"] == "matched") | (sub["kind"] == "pred_unmatched")).sum())
        n_gt   = int(((sub["kind"] == "matched") | (sub["kind"] == "gt_unmatched")).sum())
        n_matched = len(matched)
        med_tiou = float(matched["tiou"].median()) if n_matched else 0.0
        med_onset = float(matched["onset_err"].median()) if n_matched else np.nan
        med_offset = float(matched["offset_err"].median()) if n_matched else np.nan
        med_dur_ratio = float(matched["dur_ratio"].median()) if n_matched else np.nan
        mae_onset = float(matched["onset_err"].abs().median()) if n_matched else np.nan
        mae_offset = float(matched["offset_err"].abs().median()) if n_matched else np.nan
        mean_gt_dur = float((sub[sub["gt_dur"] > 0]["gt_dur"]).mean()) if (sub["gt_dur"] > 0).any() else np.nan
        mean_pred_dur = float((sub[sub["pred_dur"] > 0]["pred_dur"]).mean()) if (sub["pred_dur"] > 0).any() else np.nan
        out.append({
            "class": cname,
            "n_pred_bouts": n_pred,
            "n_gt_bouts": n_gt,
            "n_matched_any_overlap": n_matched,
            "median_tiou_matched": med_tiou,
            "median_onset_err_frames": med_onset,
            "median_offset_err_frames": med_offset,
            "median_abs_onset_err_frames": mae_onset,
            "median_abs_offset_err_frames": mae_offset,
            "median_duration_ratio": med_dur_ratio,
            "mean_pred_dur_frames": mean_pred_dur,
            "mean_gt_dur_frames": mean_gt_dur,
        })
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def fig_tiou_sweep(sweep: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 5))
    for cname, color in CLASS_COLORS.items():
        sub = sweep[sweep["class"] == cname].sort_values("tiou")
        ax.plot(sub["tiou"], sub["f1"], marker="o", linewidth=2.2,
                label=cname, color=color)
        for _, row in sub.iterrows():
            ax.text(row["tiou"], row["f1"] + 0.015, f"{row['f1']:.2f}",
                    ha="center", fontsize=8, color=color)
    ax.set_xlabel("Bout matching tIoU threshold")
    ax.set_ylabel("Bout F1 (pooled across 24 videos)")
    ax.set_xticks(TIOU_GRID); ax.set_xticklabels([f"{t:.2f}" for t in TIOU_GRID])
    ax.set_ylim(0, 1.0); ax.grid(alpha=0.3); ax.legend()
    ax.set_title("Bout F1 vs tIoU threshold - R5 v2 on test_1+test_2\n"
                 "(higher tIoU = stricter onset/offset alignment)")
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "tiou_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_onset_offset(pairs: pd.DataFrame):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey="row")
    for col, cid in enumerate(EVAL_CLASSES):
        cname = ID_TO_CLASS[cid]
        color = CLASS_COLORS[cname]
        m = pairs[(pairs["class_id"] == cid) & (pairs["kind"] == "matched")]
        for row, key, label in [(0, "onset_err", "onset error"),
                                (1, "offset_err", "offset error")]:
            ax = axes[row, col]
            data = m[key].dropna()
            if len(data) == 0:
                ax.text(0.5, 0.5, "no matched bouts", ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_title(f"{cname}: {label}")
                continue
            data_clip = data.clip(-300, 300)
            ax.hist(data_clip, bins=40, color=color, edgecolor="white", alpha=0.85)
            med = float(data.median())
            mae = float(data.abs().median())
            ax.axvline(med, color="black", linestyle="--", linewidth=1,
                       label=f"median = {med:+.0f} fr")
            ax.axvline(0, color="gray", linewidth=0.7)
            ax.set_title(f"{cname}: {label}\n"
                         f"median {med:+.0f} fr * MAE-median {mae:.0f} fr * "
                         f"n={len(data)}",
                         fontsize=9)
            ax.set_xlabel("frames (pred - GT)")
            ax.set_xlim(-300, 300)
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "onset_offset_dist.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_duration_scatter(pairs: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, cid in zip(axes, EVAL_CLASSES):
        cname = ID_TO_CLASS[cid]
        color = CLASS_COLORS[cname]
        m = pairs[(pairs["class_id"] == cid) & (pairs["kind"] == "matched")]
        if not len(m):
            ax.text(0.5, 0.5, "no matched bouts", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(cname); continue
        gt = m["gt_dur"].values / FPS  # convert frames to seconds
        pr = m["pred_dur"].values / FPS
        ax.scatter(gt, pr, s=22, color=color, alpha=0.55, edgecolors="white", linewidths=0.4)
        lim = max(gt.max(), pr.max()) * 1.1
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.6, label="y = x")
        med_ratio = float(m["dur_ratio"].median())
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("GT duration (s)"); ax.set_ylabel("Pred duration (s)")
        ax.set_title(f"{cname}: matched-bout durations\n"
                     f"median pred/GT ratio = {med_ratio:.2f} * n={len(m)}",
                     fontsize=10)
        ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "duration_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_per_video_heatmap(per_video_df: pd.DataFrame):
    df = per_video_df.copy()
    cols = [f"bout_f1_50_{c}" for c in ["attack", "investigation", "mount"]]
    mat = df[cols].values
    fig, ax = plt.subplots(figsize=(6.5, max(6, 0.27 * len(df))))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(["attack", "investigation", "mount"], fontsize=10)
    ax.set_yticks(range(len(df)))
    labels = [f"{s[:6]}/{v[:24]}{'...' if len(v) > 24 else ''}"
              for s, v in zip(df["split"], df["video"])]
    ax.set_yticklabels(labels, fontsize=7.5)
    for i in range(len(df)):
        for j, c in enumerate(["attack", "investigation", "mount"]):
            v = float(df.iloc[i][f"bout_f1_50_{c}"])
            n_gt = int(df.iloc[i][f"n_gt_{c}"])
            ax.text(j, i, f"{v:.2f}\n(n={n_gt})", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if v < 0.4 else "black")
    plt.colorbar(im, ax=ax, label="Bout F1 @ tIoU 0.5", shrink=0.6)
    ax.set_title("Per-video bout F1 @ tIoU 0.5 - R5 v2\n"
                 "(numbers below F1 = GT bout count for that class)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "per_video_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_bout_count_scatter(per_video_df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, cname in zip(axes, ["attack", "investigation", "mount"]):
        color = CLASS_COLORS[cname]
        gt = per_video_df[f"n_gt_{cname}"].values
        pr = per_video_df[f"n_pred_{cname}"].values
        ax.scatter(gt, pr, s=42, color=color, alpha=0.7, edgecolors="white", linewidths=0.5)
        lim = max(int(gt.max()), int(pr.max())) + 5
        ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.6, label="y = x")
        ax.set_xlim(-1, lim); ax.set_ylim(-1, lim)
        ax.set_xlabel("# GT bouts (per video)")
        ax.set_ylabel("# predicted bouts")
        if cname == "attack":
            zero_gt = per_video_df[gt == 0]
            n_zero_overpred = int((zero_gt[f"n_pred_{cname}"] > 0).sum())
            sub_msg = f"\n{n_zero_overpred}/{len(zero_gt)} zero-GT videos still predict =1 attack"
        else:
            sub_msg = ""
        ax.set_title(f"{cname}: bout count per video{sub_msg}", fontsize=10)
        ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "bout_count_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()


def fig_cross_class_confusion(cross_class: dict):
    """Stacked bars: when we predict class X, what fraction of frames in
    those predicted bouts are GT-labeled as each class?"""
    classes = ["attack", "investigation", "mount"]
    cids = [CLASS_MAP[c] for c in classes]
    n = len(classes)
    bottoms = np.zeros(n)
    fig, ax = plt.subplots(figsize=(8.5, 5))
    palette = {0: "#d62728", 1: "#1a73e8", 2: "#2ca02c", 3: "#9ca3af"}
    for gid in [0, 1, 2, 3]:
        vals = []
        for pcid in cids:
            counts = cross_class[pcid]
            tot = counts.sum()
            vals.append(counts[gid] / tot if tot else 0)
        ax.bar(classes, vals, bottom=bottoms,
               label=ID_TO_CLASS[gid], color=palette[gid],
               edgecolor="white")
        for i, v in enumerate(vals):
            if v > 0.05:
                ax.text(i, bottoms[i] + v / 2, f"{v:.1%}",
                        ha="center", va="center", fontsize=9,
                        color="white", fontweight="bold")
        bottoms += np.array(vals)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Fraction of frames in predicted bouts (GT labels)")
    ax.set_xlabel("Predicted class")
    ax.set_title("Bout-level cross-class confusion - R5 v2\n"
                 "When the model predicts class X, what is the underlying GT?",
                 fontsize=10)
    ax.legend(title="GT label", loc="upper right", bbox_to_anchor=(1.18, 1.0))
    plt.tight_layout()
    plt.savefig(OUT_FIGS / "cross_class_confusion.png", dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def write_markdown(summary: pd.DataFrame, sweep: pd.DataFrame,
                   pairs: pd.DataFrame, per_video_df: pd.DataFrame):
    lines = [
        "## R5 v2 deep bout-level analysis - vs MARS human GT",
        "",
        "### Per-class summary (pooled across 24 videos)",
        "",
        "| Class | n_pred | n_GT | n_matched | median tIoU | median onset (fr) | median offset (fr) | median |onset| (fr) | median dur ratio (pred/GT) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r['class']} | {int(r['n_pred_bouts'])} | {int(r['n_gt_bouts'])} | "
            f"{int(r['n_matched_any_overlap'])} | {r['median_tiou_matched']:.3f} | "
            f"{r['median_onset_err_frames']:+.0f} | {r['median_offset_err_frames']:+.0f} | "
            f"{r['median_abs_onset_err_frames']:.0f} | {r['median_duration_ratio']:.2f} |"
        )
    lines += ["", "### Bout F1 sweep across tIoU thresholds", "",
              "| Class | tIoU 0.10 | tIoU 0.25 | tIoU 0.50 | tIoU 0.75 |",
              "|---|---:|---:|---:|---:|"]
    for cname in ["attack", "investigation", "mount"]:
        s = sweep[sweep["class"] == cname].set_index("tiou")
        lines.append(
            f"| {cname} | {s.loc[0.10, 'f1']:.3f} | {s.loc[0.25, 'f1']:.3f} | "
            f"{s.loc[0.50, 'f1']:.3f} | {s.loc[0.75, 'f1']:.3f} |"
        )

    # Per-video easy/hard ranking by mean bout F1
    pv = per_video_df.copy()
    pv["mean_bout_f1"] = pv[[f"bout_f1_50_{c}" for c in ["attack","investigation","mount"]]].mean(axis=1)
    easy = pv.nlargest(3, "mean_bout_f1")[["video", "mean_bout_f1"]]
    hard = pv.nsmallest(3, "mean_bout_f1")[["video", "mean_bout_f1"]]
    lines += ["", "### Easiest 3 videos (mean bout F1 @ 0.5)", ""]
    for _, r in easy.iterrows():
        lines.append(f"- {r['video']} ? {r['mean_bout_f1']:.3f}")
    lines += ["", "### Hardest 3 videos (mean bout F1 @ 0.5)", ""]
    for _, r in hard.iterrows():
        lines.append(f"- {r['video']} ? {r['mean_bout_f1']:.3f}")

    (OUT_CSV / "deep_bout_analysis.md").write_text("\n".join(lines) + "\n",
                                                   encoding="utf-8")


def main():
    pairs, per_video_df, cross_class = build_pair_table()
    pairs.to_csv(OUT_CSV / "bout_pairs.csv", index=False)
    per_video_df.to_csv(OUT_CSV / "per_video_bout_f1.csv", index=False)

    summary = class_summary(pairs)
    summary.to_csv(OUT_CSV / "bout_summary.csv", index=False)

    sweep = tiou_sweep(pairs)
    sweep.to_csv(OUT_CSV / "tIoU_sweep.csv", index=False)

    fig_tiou_sweep(sweep)
    fig_onset_offset(pairs)
    fig_duration_scatter(pairs)
    fig_per_video_heatmap(per_video_df)
    fig_bout_count_scatter(per_video_df)
    fig_cross_class_confusion(cross_class)
    write_markdown(summary, sweep, pairs, per_video_df)

    print("\n=== Per-class summary ===")
    print(summary.to_string(index=False))
    print("\n=== tIoU sweep ===")
    print(sweep.pivot(index="class", columns="tiou", values="f1").round(3).to_string())
    print(f"\nFigures written to {OUT_FIGS}")
    print(f"CSVs + report written to {OUT_CSV}")


if __name__ == "__main__":
    main()
