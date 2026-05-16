"""Evaluate BehaviorScope-Y predictions against a BENTO .annot ground truth.

Computes frame-level + bout-level metrics for ONE video, plus optional figures.

Usage:
    python evaluate_against_annot.py \
        --predictions  path/to/Test_R5.csv \
        --annot        path/to/Mouse155_..._bhvr.annot \
        --label        "Hybrid (Pose + RGB)" \
        --output-dir   path/to/eval_out/

Or for a paired R2 vs R5 comparison with combined plots:
    python evaluate_against_annot.py \
        --predictions  R2_Inference/Test_R2.csv R5_Inference/Test_R5.csv \
        --annot        Mouse155_..._bhvr.annot \
        --label        "RGB-only" "Hybrid (Pose + RGB)" \
        --output-dir   eval_out/

Inputs
------
- predictions CSV (from infer_y.py): columns include `frame_start`, `frame_end`,
  `predicted_class`. Window-level. Stride < window length is OK; overlaps are
  resolved by majority vote (with class-probability tiebreak).
- BENTO .annot file: human-annotated bouts with channels and behavior names.

Outputs
-------
- metrics.json   per-model frame + bout metrics + cross-model summary
- summary.txt    human-readable report
- timeline.png   stacked timeline (GT vs each model)
- per_class.png  per-class F1 bar chart (frame + bout)
- confusion.png  confusion matrices (one panel per model)
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

CLASSES_DEFAULT = ["attack", "investigation", "mount", "other"]


# -----------------------------------------------------------------------------
# Parsers
# -----------------------------------------------------------------------------

def parse_annot(path: Path, default_other: str = "other") -> Tuple[Dict, List[dict]]:
    """Parse a BENTO .annot file.

    Returns
    -------
    header : dict with keys 'fps', 'start_time', 'stop_time', 'channels',
             'annotations' (list of behavior names declared in header).
    bouts  : list of {'channel': str, 'behavior': str, 'start_s': float,
             'stop_s': float}.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    header = {
        "fps": 30.0,
        "start_time": 0.0,
        "stop_time": None,
        "channels": [],
        "annotations": [],
        "default_other": default_other,
    }

    # Header fields
    m = re.search(r"Annotation framerate:\s*([\d.]+)", text)
    if m:
        header["fps"] = float(m.group(1))
    m = re.search(r"Annotation start time:\s*([\d.eE+-]+)", text)
    if m:
        header["start_time"] = float(m.group(1))
    m = re.search(r"Annotation stop time:\s*([\d.eE+-]+)", text)
    if m:
        header["stop_time"] = float(m.group(1))

    # Channel + annotation lists (between the headings and the first "Ch...---" block)
    m = re.search(r"List of channels:\s*\n((?:\w[^\n]*\n?)+)\n", text)
    if m:
        header["channels"] = [c.strip() for c in m.group(1).strip().splitlines() if c.strip()]
    m = re.search(r"List of annotations:\s*\n((?:[^\n]+\n?)+?)\n\n", text)
    if m:
        header["annotations"] = [a.strip() for a in m.group(1).strip().splitlines() if a.strip()]

    # Bouts: split on channel markers like "Ch1----------"
    bouts: List[dict] = []
    channel_blocks = re.split(r"\n(Ch\w+)-+\n", text)
    # channel_blocks[0] is the preamble; [1] is first channel name, [2] is its body, etc.
    for i in range(1, len(channel_blocks), 2):
        channel = channel_blocks[i].strip()
        body = channel_blocks[i + 1] if i + 1 < len(channel_blocks) else ""
        # Each behavior section starts with ">name", then a header line, then
        # rows of "start stop duration"
        for behavior_match in re.finditer(
            r">(\S+)\s*\nStart\s+Stop\s+Duration\s*\n((?:[\d.\s]+\n?)+)",
            body,
        ):
            behavior = behavior_match.group(1).strip()
            rows_block = behavior_match.group(2)
            for row in rows_block.strip().splitlines():
                tokens = row.split()
                if len(tokens) < 2:
                    continue
                try:
                    start_s = float(tokens[0])
                    stop_s = float(tokens[1])
                except ValueError:
                    continue
                if stop_s < start_s:
                    continue
                bouts.append(
                    {
                        "channel": channel,
                        "behavior": behavior,
                        "start_s": start_s,
                        "stop_s": stop_s,
                    }
                )
    return header, bouts


def expand_bouts_to_frames(
    bouts: Sequence[dict],
    n_frames: int,
    fps: float,
    classes: Sequence[str],
    default: str = "other",
    priority: Sequence[str] | None = None,
) -> np.ndarray:
    """Build a length-`n_frames` array of class indices from bout list.

    On overlap (e.g., GT only — predictions never overlap themselves),
    `priority` ordering decides which class wins. Default priority: classes
    in the order they appear in `classes`, MINUS `default`, then `default`
    last.
    """
    if priority is None:
        priority = [c for c in classes if c != default] + [default]
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    default_idx = cls_to_idx[default]
    arr = np.full(n_frames, default_idx, dtype=np.int8)

    # Apply bouts in REVERSE priority so the highest-priority class is written
    # last (and survives overlap).
    rev_priority = list(reversed(priority))
    for cls_name in rev_priority:
        if cls_name == default:
            continue
        for b in bouts:
            if b["behavior"] != cls_name:
                continue
            s = int(round(b["start_s"] * fps))
            e = int(round(b["stop_s"] * fps))
            s = max(0, s)
            e = min(n_frames, e)
            if e > s:
                arr[s:e] = cls_to_idx[cls_name]
    return arr


def predictions_to_per_frame(
    df: pd.DataFrame,
    n_frames: int,
    classes: Sequence[str],
    default: str = "other",
) -> np.ndarray:
    """Expand window-level predictions to per-frame predicted class indices.

    Each frame is covered by zero or more windows (typically ~window/stride
    of them). When multiple windows overlap a frame, we average their class
    probabilities (from `prob_<class>` columns) and take argmax. This is more
    robust than majority-vote of hard labels because boundary frames often
    have nearly tied probabilities and a single window's hard call can be
    arbitrary.
    """
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)
    prob_sum = np.zeros((n_frames, n_classes), dtype=np.float64)
    coverage = np.zeros(n_frames, dtype=np.int32)

    prob_cols = [f"prob_{c}" for c in classes]
    have_probs = all(c in df.columns for c in prob_cols)

    for _, row in df.iterrows():
        s = int(row["frame_start"])
        # frame_end in our CSV is INCLUSIVE; turn into half-open [s, e)
        e = int(row["frame_end"]) + 1
        s = max(0, s)
        e = min(n_frames, e)
        if e <= s:
            continue
        if have_probs:
            probs = np.array([float(row[c]) for c in prob_cols], dtype=np.float64)
            prob_sum[s:e] += probs
        else:
            cls_idx = cls_to_idx[row["predicted_class"]]
            prob_sum[s:e, cls_idx] += 1.0
        coverage[s:e] += 1

    pred = np.full(n_frames, cls_to_idx[default], dtype=np.int8)
    covered = coverage > 0
    pred[covered] = prob_sum[covered].argmax(axis=1).astype(np.int8)
    return pred


# -----------------------------------------------------------------------------
# Frame-level metrics
# -----------------------------------------------------------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (precision, recall, f1) arrays, one entry per class."""
    n = cm.shape[0]
    p = np.zeros(n)
    r = np.zeros(n)
    f = np.zeros(n)
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p[i] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r[i] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f[i] = 2 * p[i] * r[i] / (p[i] + r[i]) if (p[i] + r[i]) > 0 else 0.0
    return p, r, f


def macro_f1_present(cm: np.ndarray, present_idx: Sequence[int]) -> float:
    _, _, f = per_class_prf(cm)
    if not present_idx:
        return 0.0
    return float(np.mean([f[i] for i in present_idx]))


# -----------------------------------------------------------------------------
# Bout-level metrics
# -----------------------------------------------------------------------------

def frames_to_bouts(per_frame: np.ndarray, default_idx: int) -> List[Tuple[int, int, int]]:
    """Extract contiguous same-class runs as bouts. Skip the default class.
    Returns list of (cls_idx, start_frame, end_frame) with end EXCLUSIVE."""
    bouts = []
    n = len(per_frame)
    if n == 0:
        return bouts
    i = 0
    while i < n:
        c = int(per_frame[i])
        if c == default_idx:
            i += 1
            continue
        j = i + 1
        while j < n and per_frame[j] == c:
            j += 1
        bouts.append((c, i, j))
        i = j
    return bouts


def temporal_iou(a: Tuple[int, int], b: Tuple[int, int]) -> float:
    inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def bout_metrics(
    gt_bouts: List[Tuple[int, int, int]],
    pred_bouts: List[Tuple[int, int, int]],
    n_classes: int,
    iou_threshold: float,
) -> Dict[str, np.ndarray]:
    """Per-class precision/recall/F1 at one tIoU threshold.

    A predicted bout is a TP for class c if (a) its class is c and (b) it has
    tIoU >= threshold with some unmatched GT bout of class c. Each GT bout
    can match at most one prediction.
    """
    tp = np.zeros(n_classes, dtype=int)
    fp = np.zeros(n_classes, dtype=int)
    fn = np.zeros(n_classes, dtype=int)
    matched_gt = [False] * len(gt_bouts)

    # Greedy match: sort predictions by best-available IoU per class
    for pc, ps, pe in pred_bouts:
        best_iou = -1.0
        best_j = -1
        for j, (gc, gs, ge) in enumerate(gt_bouts):
            if gc != pc or matched_gt[j]:
                continue
            iou = temporal_iou((ps, pe), (gs, ge))
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            tp[pc] += 1
            matched_gt[best_j] = True
        else:
            fp[pc] += 1

    for j, (gc, _, _) in enumerate(gt_bouts):
        if not matched_gt[j]:
            fn[gc] += 1

    p = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    r = np.where(tp + fn > 0, tp / np.maximum(tp + fn, 1), 0.0)
    f = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f}


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def class_palette(classes: Sequence[str]) -> Dict[str, str]:
    palette = {
        "attack": "#dc2626",
        "investigation": "#1f77b4",
        "mount": "#10b981",
        "other": "#94a3b8",
    }
    return {c: palette.get(c, "#64748b") for c in classes}


def plot_timeline(
    gt_frames: np.ndarray,
    model_frames: Dict[str, np.ndarray],
    classes: Sequence[str],
    fps: float,
    output_path: Path,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    palette = class_palette(classes)
    n_frames = len(gt_frames)
    n_models = len(model_frames)
    rows = 1 + n_models
    fig, axes = plt.subplots(rows, 1, figsize=(13, 0.9 * rows + 1.0), sharex=True)
    if rows == 1:
        axes = [axes]
    time_s = np.arange(n_frames) / fps

    def _draw(ax, frames, label):
        # Draw colored spans for each class run (skip the bg by drawing all)
        i = 0
        while i < n_frames:
            c = int(frames[i])
            j = i + 1
            while j < n_frames and frames[j] == c:
                j += 1
            ax.axvspan(time_s[i], time_s[min(j, n_frames - 1)], color=palette[classes[c]],
                       alpha=0.85 if classes[c] != "other" else 0.18)
            i = j
        ax.set_yticks([])
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=11)
        ax.set_xlim(0, time_s[-1])

    _draw(axes[0], gt_frames, "Human\n(GT)")
    for ax, (name, frames) in zip(axes[1:], model_frames.items()):
        _draw(ax, frames, name)

    axes[-1].set_xlabel("Time (s)")
    legend_handles = [Patch(facecolor=palette[c], alpha=0.85, label=c)
                      for c in classes if c != "other"]
    legend_handles.append(Patch(facecolor=palette["other"], alpha=0.18, label="other"))
    axes[0].legend(handles=legend_handles, loc="upper right", ncol=4,
                   bbox_to_anchor=(1.0, 1.7), frameon=False, fontsize=10)
    fig.suptitle("Frame-level predictions vs human annotation", y=1.02, fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(
    metrics_by_model: Dict[str, dict],
    classes_present: Sequence[str],
    output_path: Path,
):
    import matplotlib.pyplot as plt

    palette = {"RGB-only": "#d62728", "Hybrid (Pose + RGB)": "#1f77b4"}
    model_names = list(metrics_by_model.keys())
    n_classes = len(classes_present)
    x = np.arange(n_classes)
    width = 0.8 / max(1, len(model_names))

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))
    titles = ["Frame-level F1", "Bout F1 @ tIoU 0.25", "Bout F1 @ tIoU 0.50"]
    keys = ["frame_f1_per_class", "bout_f1_iou25", "bout_f1_iou50"]

    for ax, title, key in zip(axes, titles, keys):
        for k, model in enumerate(model_names):
            color = palette.get(model, f"C{k}")
            vals = [metrics_by_model[model][key].get(c, 0.0) for c in classes_present]
            ax.bar(x + k * width - 0.4 + width / 2, vals, width=width,
                   label=model, color=color, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(classes_present, rotation=20, ha="right")
        ax.set_ylim(0, 1.0)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("F1")
    axes[0].legend(loc="upper left", fontsize=9, framealpha=0.95)

    fig.suptitle("Per-class F1 — preliminary held-out evaluation (single test_2 video)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(
    cms_by_model: Dict[str, np.ndarray],
    classes: Sequence[str],
    output_path: Path,
):
    import matplotlib.pyplot as plt

    n_models = len(cms_by_model)
    fig, axes = plt.subplots(1, n_models, figsize=(5.0 * n_models, 4.4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cms_by_model.items()):
        cm_norm = cm.astype(float)
        row_sum = cm_norm.sum(axis=1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            cm_norm = np.where(row_sum > 0, cm_norm / row_sum, 0.0)
        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        for i in range(len(classes)):
            for j in range(len(classes)):
                txt = f"{cm_norm[i, j]:.2f}\n({cm[i, j]:,})"
                color = "white" if cm_norm[i, j] > 0.55 else "black"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=20, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(name, fontsize=11)
    fig.suptitle("Frame-level confusion matrices (row-normalized recall)",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Driver
# -----------------------------------------------------------------------------

def evaluate_one(
    pred_path: Path,
    annot_path: Path,
    classes: Sequence[str],
    label: str,
) -> dict:
    df = pd.read_csv(pred_path)
    n_frames = int(df["frame_end"].max()) + 1

    header, bouts = parse_annot(annot_path)
    fps = float(header["fps"])

    cls_to_idx = {c: i for i, c in enumerate(classes)}
    default_idx = cls_to_idx["other"]

    # Ground truth + predictions, both expanded to per-frame
    gt = expand_bouts_to_frames(bouts, n_frames, fps, classes, default="other")
    pred = predictions_to_per_frame(df, n_frames, classes, default="other")

    # Frame-level metrics
    cm = confusion_matrix(gt, pred, len(classes))
    p_arr, r_arr, f_arr = per_class_prf(cm)
    classes_with_gt = [c for c, idx in cls_to_idx.items()
                       if int((gt == idx).sum()) > 0]

    macro_f1_all = macro_f1_present(cm, list(range(len(classes))))
    macro_f1_present_only = macro_f1_present(
        cm, [cls_to_idx[c] for c in classes_with_gt]
    )

    # Bout-level metrics
    gt_bouts = frames_to_bouts(gt, default_idx)
    pred_bouts = frames_to_bouts(pred, default_idx)
    bout_25 = bout_metrics(gt_bouts, pred_bouts, len(classes), 0.25)
    bout_50 = bout_metrics(gt_bouts, pred_bouts, len(classes), 0.50)

    return {
        "label": label,
        "predictions_csv": str(pred_path),
        "n_frames": int(n_frames),
        "fps": fps,
        "classes": list(classes),
        "classes_with_gt": classes_with_gt,
        "n_gt_bouts_total": len(gt_bouts),
        "n_pred_bouts_total": len(pred_bouts),
        "frame_level": {
            "confusion_matrix": cm.tolist(),
            "precision_per_class": {c: float(p_arr[i]) for i, c in enumerate(classes)},
            "recall_per_class": {c: float(r_arr[i]) for i, c in enumerate(classes)},
            "f1_per_class": {c: float(f_arr[i]) for i, c in enumerate(classes)},
            "macro_f1_all_classes": macro_f1_all,
            "macro_f1_classes_with_gt": macro_f1_present_only,
            "accuracy": float((gt == pred).mean()),
        },
        "bout_level": {
            "iou_0.25": {
                "tp": bout_25["tp"].tolist(),
                "fp": bout_25["fp"].tolist(),
                "fn": bout_25["fn"].tolist(),
                "precision_per_class": {c: float(bout_25["precision"][i]) for i, c in enumerate(classes)},
                "recall_per_class": {c: float(bout_25["recall"][i]) for i, c in enumerate(classes)},
                "f1_per_class": {c: float(bout_25["f1"][i]) for i, c in enumerate(classes)},
                "macro_f1_classes_with_gt": float(np.mean([bout_25["f1"][cls_to_idx[c]] for c in classes_with_gt])) if classes_with_gt else 0.0,
            },
            "iou_0.50": {
                "tp": bout_50["tp"].tolist(),
                "fp": bout_50["fp"].tolist(),
                "fn": bout_50["fn"].tolist(),
                "precision_per_class": {c: float(bout_50["precision"][i]) for i, c in enumerate(classes)},
                "recall_per_class": {c: float(bout_50["recall"][i]) for i, c in enumerate(classes)},
                "f1_per_class": {c: float(bout_50["f1"][i]) for i, c in enumerate(classes)},
                "macro_f1_classes_with_gt": float(np.mean([bout_50["f1"][cls_to_idx[c]] for c in classes_with_gt])) if classes_with_gt else 0.0,
            },
        },
        "_arrays": {  # for plotting only — not serialized
            "gt": gt,
            "pred": pred,
            "cm": cm,
        },
    }


def write_summary(metrics_by_model: List[dict], out_path: Path):
    lines = []
    lines.append("BehaviorScope-Y — preliminary held-out evaluation")
    lines.append("=" * 60)
    for m in metrics_by_model:
        lines.append(f"\nModel: {m['label']}")
        lines.append(f"  predictions: {m['predictions_csv']}")
        lines.append(f"  n_frames: {m['n_frames']}    fps: {m['fps']:.2f}")
        lines.append(f"  classes with ground-truth bouts: {m['classes_with_gt']}")
        lines.append(f"  GT bouts: {m['n_gt_bouts_total']}    predicted bouts: {m['n_pred_bouts_total']}")
        fl = m["frame_level"]
        lines.append("  frame-level:")
        lines.append(f"    accuracy: {fl['accuracy']:.4f}")
        lines.append(f"    macro-F1 (classes with GT only): {fl['macro_f1_classes_with_gt']:.4f}")
        lines.append(f"    macro-F1 (all 4 classes):        {fl['macro_f1_all_classes']:.4f}")
        for c in m["classes"]:
            lines.append(
                f"    {c:14s}  P={fl['precision_per_class'][c]:.3f}"
                f"  R={fl['recall_per_class'][c]:.3f}"
                f"  F1={fl['f1_per_class'][c]:.3f}"
            )
        for tag, key in [("0.25", "iou_0.25"), ("0.50", "iou_0.50")]:
            bl = m["bout_level"][key]
            lines.append(f"  bout-level @ tIoU {tag}:")
            lines.append(f"    macro-F1 (classes with GT only): {bl['macro_f1_classes_with_gt']:.4f}")
            for c in m["classes_with_gt"]:
                lines.append(
                    f"    {c:14s}  P={bl['precision_per_class'][c]:.3f}"
                    f"  R={bl['recall_per_class'][c]:.3f}"
                    f"  F1={bl['f1_per_class'][c]:.3f}"
                )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", nargs="+", required=True,
                        help="One or more predictions CSV(s).")
    parser.add_argument("--annot", required=True,
                        help="BENTO .annot ground-truth file.")
    parser.add_argument("--label", nargs="+", default=None,
                        help="Label for each predictions CSV. Same count as --predictions.")
    parser.add_argument("--output-dir", required=True,
                        help="Where to write metrics.json, summary.txt, and figures.")
    parser.add_argument("--classes", nargs="+", default=CLASSES_DEFAULT,
                        help="Class labels in canonical order.")
    args = parser.parse_args()

    annot_path = Path(args.annot)
    pred_paths = [Path(p) for p in args.predictions]
    if args.label is None:
        labels = [p.stem for p in pred_paths]
    else:
        if len(args.label) != len(pred_paths):
            raise SystemExit("--label count must match --predictions count")
        labels = args.label

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for p, lbl in zip(pred_paths, labels):
        results.append(evaluate_one(p, annot_path, args.classes, lbl))

    # Save JSON (strip _arrays before serializing)
    serializable = []
    for r in results:
        rr = {k: v for k, v in r.items() if k != "_arrays"}
        serializable.append(rr)
    (out_dir / "metrics.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )
    write_summary(serializable, out_dir / "summary.txt")

    # Plots
    classes = list(args.classes)
    classes_with_gt = results[0]["classes_with_gt"]  # same GT for all models

    plot_timeline(
        gt_frames=results[0]["_arrays"]["gt"],
        model_frames={r["label"]: r["_arrays"]["pred"] for r in results},
        classes=classes,
        fps=results[0]["fps"],
        output_path=out_dir / "timeline.png",
    )

    metrics_by_label = {
        r["label"]: {
            "frame_f1_per_class": r["frame_level"]["f1_per_class"],
            "bout_f1_iou25": r["bout_level"]["iou_0.25"]["f1_per_class"],
            "bout_f1_iou50": r["bout_level"]["iou_0.50"]["f1_per_class"],
        }
        for r in results
    }
    plot_per_class_f1(metrics_by_label, classes_with_gt, out_dir / "per_class.png")
    plot_confusion(
        {r["label"]: r["_arrays"]["cm"] for r in results},
        classes,
        out_dir / "confusion.png",
    )

    print(f"Wrote {out_dir / 'metrics.json'}")
    print(f"Wrote {out_dir / 'summary.txt'}")
    print(f"Wrote {out_dir / 'timeline.png'}")
    print(f"Wrote {out_dir / 'per_class.png'}")
    print(f"Wrote {out_dir / 'confusion.png'}")
    print()
    print("=== Headline numbers ===")
    for r in results:
        print(f"\n{r['label']}:")
        print(f"  frame-level macro-F1 (classes with GT): {r['frame_level']['macro_f1_classes_with_gt']:.4f}")
        print(f"  bout-level @ IoU 0.25 macro-F1:         {r['bout_level']['iou_0.25']['macro_f1_classes_with_gt']:.4f}")
        print(f"  bout-level @ IoU 0.50 macro-F1:         {r['bout_level']['iou_0.50']['macro_f1_classes_with_gt']:.4f}")


if __name__ == "__main__":
    main()
