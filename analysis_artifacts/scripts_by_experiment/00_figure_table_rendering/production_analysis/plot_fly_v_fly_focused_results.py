"""Production Fly-v-Fly figures for the focused adaptation evaluation.

This script intentionally targets the focused attention-256, seed-42 adaptation
run. It also writes reproducibility-facing CSVs used to audit strict primary-label
evaluation, median-bout collar sensitivity, and the limited movie 6
secondary-annotation reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
REPO_ROOT = THIS_DIR.parents[2]
STYLE_DIR = THIS_DIR.parent / "current_analysis"
SHARED_DIR = REPO_ROOT / "shared_scripts"
for path in (STYLE_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from behaviorscope_figure_style import (  # noqa: E402
    HEAD_COLORS,
    NEUTRAL,
    apply_style,
    panel_label,
    save_figure,
)
from fly_run_all_eval import (  # noqa: E402
    BEHAVIOR_CLASSES,
    CLASS_TO_IDX,
    CLASSES,
    actions_to_bouts,
    bouts_to_frames,
    frames_to_bouts,
    load_actions_file,
    load_raw_window_predictions,
)
from fly_to_behaviorscope_npz_full import FLY_FPS  # noqa: E402
from plot_fly_v_fly_pose_validation_profile import plot_pose_profile as plot_fly_pose_profile  # noqa: E402


TABLE_DIR = ROOT / "tables" / "production" / "fly_v_fly"
FIG_DIR = ROOT / "figures" / "production" / "fly_v_fly"
SUPP_FIG_DIR = ROOT / "figures" / "production" / "supplement"
DEFAULT_EVAL_DIR = (
    REPO_ROOT
    / "outputs"
    / "controlled_comparison_runs"
    / "fly_adaptation_attention256_cw12_seed42"
    / "fly"
    / "eval"
)
POSE_VAL_TXT = REPO_ROOT / "Fly_YOLO_Pose_Model" / "weights" / "Fly_model_val.txt"

BEHAVIOR_LABELS = {
    "lunge": "Lunge",
    "wing_threat": "Wing threat",
    "charge": "Charge",
    "hold": "Hold",
    "tussle": "Tussle",
    "other": "Other",
}
BEHAVIOR_COLORS = {
    "lunge": "#0072B2",
    "wing_threat": "#009E73",
    "charge": "#D55E00",
    "hold": "#8E6C88",
    "tussle": "#C9A227",
    "other": "#777777",
}
WINDOW_LABELS = {"w16s8": "16-frame window", "w8s4": "8-frame window"}
WINDOW_COLORS = {"w16s8": NEUTRAL["dark"], "w8s4": HEAD_COLORS["attention"]}
COLLARS = list(range(0, 31))
COLLAR_TICKS = [0, 4, 8, 12, 16, 21, 30]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--table_dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--fig_dir", type=Path, default=FIG_DIR)
    parser.add_argument("--supp_fig_dir", type=Path, default=SUPP_FIG_DIR)
    parser.add_argument("--pose_val_txt", type=Path, default=POSE_VAL_TXT)
    return parser.parse_args()


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=NEUTRAL["grid"], linewidth=0.8, alpha=0.28)
    ax.set_axisbelow(True)


def savefig(fig: plt.Figure, out_stem: Path) -> None:
    save_figure(fig, out_stem)
    plt.close(fig)


def parse_run_fields(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    fields = out["run"].str.extract(
        r"full_(?P<head>lstm|attention)(?P<hidden>\d+)_(?P<window>w\d+s\d+)_seed(?P<seed>\d+)"
    )
    out = pd.concat([out, fields], axis=1)
    out["hidden"] = pd.to_numeric(out["hidden"], errors="coerce").astype("Int64")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    out["window_label"] = out["window"].map(WINDOW_LABELS)
    return out


def copy_eval_csvs(eval_dir: Path, table_dir: Path) -> None:
    if not (eval_dir / "eval_summary.json").is_file():
        raise FileNotFoundError(f"Missing completed eval summary: {eval_dir / 'eval_summary.json'}")
    copies = {
        "cross_run_summary.csv": "fly_focused_eval_cross_run_summary.csv",
        "per_video_metrics.csv": "fly_focused_eval_per_video_metrics.csv",
        "failures.csv": "fly_focused_eval_failures.csv",
        "human_human_agreement.csv": "fly_focused_eval_human_human_agreement.csv",
        "eval_summary.json": "fly_focused_eval_summary.json",
    }
    table_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in copies.items():
        src = eval_dir / src_name
        if src.is_file():
            shutil.copy2(src, table_dir / dst_name)


def load_pose_validation(path: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(
        r"^\s*(?P<class>all|lunge|wing_threat|charge|hold|tussle|other)\s+"
        r"(?P<images>\d+)\s+(?P<instances>\d+)\s+"
        r"(?P<box_p>[0-9.]+)\s+(?P<box_r>[0-9.]+)\s+(?P<box_map50>[0-9.]+)\s+(?P<box_map5095>[0-9.]+)\s+"
        r"(?P<pose_p>[0-9.]+)\s+(?P<pose_r>[0-9.]+)\s+(?P<pose_map50>[0-9.]+)\s+(?P<pose_map5095>[0-9.]+)"
    )
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        row = match.groupdict()
        for key, value in row.items():
            if key != "class":
                row[key] = float(value)
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No pose-validation rows parsed from {path}")
    return pd.DataFrame(rows)


def primary_bout_medians(heldout: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in heldout.iterrows():
        for behavior in BEHAVIOR_CLASSES:
            value = row.get(f"{behavior}_median_frames", np.nan)
            if pd.notna(value) and value != "":
                rows.append({
                    "movie_id": int(row["movie_id"]),
                    "behavior": behavior,
                    "median_frames": float(value),
                })
    return pd.DataFrame(rows)


def boundary_match_metrics(gt_bouts: list[dict], pred_bouts: list[dict], collar: int) -> dict:
    present = [c for c in BEHAVIOR_CLASSES if any(b["class"] == c for b in gt_bouts)]
    rows = {c: {"tp": 0, "fp": 0, "fn": 0} for c in BEHAVIOR_CLASSES}
    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    candidate_pairs = []
    for pi, pred in enumerate(pred_bouts):
        for gi, gt in enumerate(gt_bouts):
            if gt["class"] != pred["class"]:
                continue
            start_error = abs(int(pred["start_frame"]) - int(gt["start_frame"]))
            end_error = abs(int(pred["end_frame"]) - int(gt["end_frame"]))
            if start_error <= collar and end_error <= collar:
                candidate_pairs.append((
                    int(max(start_error, end_error)),
                    int(start_error + end_error),
                    pi,
                    gi,
                ))
    for _, _, pi, gi in sorted(candidate_pairs):
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        rows[pred_bouts[pi]["class"]]["tp"] += 1
    for pi, pred in enumerate(pred_bouts):
        if pi not in matched_pred and pred["class"] in rows:
            rows[pred["class"]]["fp"] += 1
    for gi, gt in enumerate(gt_bouts):
        if gi not in matched_gt and gt["class"] in rows:
            rows[gt["class"]]["fn"] += 1

    for behavior, row in rows.items():
        tp, fp, fn = row["tp"], row["fp"], row["fn"]
        row["precision"] = tp / (tp + fp) if tp + fp else 0.0
        row["recall"] = tp / (tp + fn) if tp + fn else 0.0
        row["f1"] = (
            2 * row["precision"] * row["recall"] / (row["precision"] + row["recall"])
            if row["precision"] + row["recall"]
            else 0.0
        )
    macro = {
        metric: float(np.mean([rows[c][metric] for c in present])) if present else 0.0
        for metric in ("precision", "recall", "f1")
    }
    return {"present": present, "per_class": rows, "macro": macro}


def compute_collar_metrics(eval_dir: Path, heldout: pd.DataFrame, table_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(p for p in eval_dir.iterdir() if p.is_dir() and p.name.startswith("fly_yolo")):
        match = re.search(r"_(w\d+s\d+)_seed", run_dir.name)
        if not match:
            continue
        window = match.group(1)
        for _, video in heldout.iterrows():
            movie_id = int(video["movie_id"])
            n_frames = int(video["n_frames"])
            action_path = Path(str(video["primary_actions_path"]))
            pred_csv = run_dir / f"movie{movie_id}" / f"movie{movie_id}.behavior.csv"
            if not pred_csv.is_file():
                continue
            gt_frames = bouts_to_frames(actions_to_bouts(load_actions_file(action_path)), n_frames)
            pred_frames = load_raw_window_predictions(pred_csv, n_frames)
            gt_bouts = frames_to_bouts(gt_frames)
            pred_bouts = frames_to_bouts(pred_frames)
            present = [c for c in BEHAVIOR_CLASSES if any(b["class"] == c for b in gt_bouts)]
            for collar in COLLARS:
                result = boundary_match_metrics(gt_bouts, pred_bouts, collar)
                rows.append({
                    "run": run_dir.name,
                    "window": window,
                    "movie_id": movie_id,
                    "collar_frames": collar,
                    "collar_seconds": round(float(collar) / FLY_FPS, 6),
                    "heldout_median_bout_frames": float(video["primary_priority_median_bout_frames"]),
                    "gt_bouts_total": len(gt_bouts),
                    "pred_bouts_total": len(pred_bouts),
                    "present_classes": ";".join(present),
                    "macro_precision": result["macro"]["precision"],
                    "macro_recall": result["macro"]["recall"],
                    "macro_f1": result["macro"]["f1"],
                    "per_class_json": json.dumps(result["per_class"], sort_keys=True),
                })
    out = pd.DataFrame(rows)
    out.to_csv(table_dir / "fly_collar_event_metrics_by_video.csv", index=False)
    summary = (
        out.groupby(["run", "window", "collar_frames"], as_index=False)
        .agg(
            mean_macro_precision=("macro_precision", "mean"),
            mean_macro_recall=("macro_recall", "mean"),
            mean_macro_f1=("macro_f1", "mean"),
            sd_macro_f1=("macro_f1", "std"),
            n_videos=("movie_id", "nunique"),
            median_heldout_bout_frames=("heldout_median_bout_frames", "median"),
        )
    )
    summary.to_csv(table_dir / "fly_collar_event_metrics_summary.csv", index=False)
    sweep_rows = []
    for run, sub in summary.groupby("run"):
        sub = sub.sort_values("collar_frames")
        max_f1 = float(sub["mean_macro_f1"].max()) if len(sub) else 0.0
        for fraction in [0.50, 0.75, 0.90, 0.95]:
            eligible = sub[sub["mean_macro_f1"] >= max_f1 * fraction]
            sweep_rows.append({
                "run": run,
                "window": sub["window"].iloc[0] if len(sub) else "",
                "max_mean_macro_f1_in_sweep": max_f1,
                "fraction_of_sweep_max": fraction,
                "first_collar_frames": int(eligible["collar_frames"].iloc[0]) if len(eligible) else "",
            })
    pd.DataFrame(sweep_rows).to_csv(table_dir / "fly_collar_event_sweep_landmarks.csv", index=False)
    return out


def write_primary_tables(cross: pd.DataFrame, per_video: pd.DataFrame, table_dir: Path) -> None:
    primary = cross[cross["gt_mode"].eq("primary")].copy()
    keep = [
        "run",
        "window",
        "n_video_mode_rows",
        "mean_raw_frame_macro_precision",
        "mean_raw_frame_macro_recall",
        "mean_raw_frame_macro_f1",
        "mean_raw_bout_macro_precision_25",
        "mean_raw_bout_macro_recall_25",
        "mean_raw_bout_macro_f1_25",
        "mean_smoothed_bout_macro_f1_10",
        "mean_smoothed_bout_macro_f1_25",
        "mean_smoothed_bout_macro_f1_50",
    ]
    primary[keep].to_csv(table_dir / "fly_focused_eval_primary_summary_by_config.csv", index=False)
    cross[
        [
            "run",
            "window",
            "gt_mode",
            "n_video_mode_rows",
            "mean_raw_frame_macro_precision",
            "mean_raw_frame_macro_recall",
            "mean_raw_frame_macro_f1",
            "mean_raw_bout_macro_precision_25",
            "mean_raw_bout_macro_recall_25",
            "mean_raw_bout_macro_f1_25",
            "mean_smoothed_bout_macro_f1_10",
            "mean_smoothed_bout_macro_f1_25",
            "mean_smoothed_bout_macro_f1_50",
        ]
    ].to_csv(table_dir / "fly_focused_eval_annotation_mode_summary.csv", index=False)

    best_rows = []
    for metric in [
        "mean_raw_frame_macro_f1",
        "mean_raw_bout_macro_f1_25",
        "mean_smoothed_bout_macro_f1_10",
        "mean_smoothed_bout_macro_f1_50",
    ]:
        idx = primary[metric].idxmax()
        row = primary.loc[idx]
        best_rows.append({
            "selection_metric": metric,
            "run": row["run"],
            "window": row["window"],
            "value": row[metric],
        })
    pd.DataFrame(best_rows).to_csv(table_dir / "fly_focused_eval_best_primary_runs_by_metric.csv", index=False)

    selected = per_video[per_video["gt_mode"].eq("primary") & per_video["run"].str.contains("w8s4")].copy()
    class_rows = []
    for behavior in BEHAVIOR_CLASSES:
        class_rows.append({
            "behavior": behavior,
            "label": BEHAVIOR_LABELS[behavior],
            "mean_precision": selected[f"raw_precision_{behavior}"].mean(),
            "mean_recall": selected[f"raw_recall_{behavior}"].mean(),
            "mean_frame_f1": selected[f"raw_f1_{behavior}"].mean(),
        })
    pd.DataFrame(class_rows).to_csv(
        table_dir / "fly_focused_eval_primary_class_precision_recall.csv",
        index=False,
    )


def plot_main_figure(
    heldout: pd.DataFrame,
    cross: pd.DataFrame,
    per_video: pd.DataFrame,
    collar: pd.DataFrame,
    pose: pd.DataFrame,
    fig_dir: Path,
) -> None:
    med = primary_bout_medians(heldout)
    primary = cross[cross["gt_mode"].eq("primary")].sort_values("window")
    w8_per_class = per_video[per_video["gt_mode"].eq("primary") & per_video["run"].str.contains("w8s4")]
    collar_summary = (
        collar.groupby(["window", "collar_frames"], as_index=False)
        .agg(mean_macro_f1=("macro_f1", "mean"), sd_macro_f1=("macro_f1", "std"))
    )

    fig = plt.figure(figsize=(13.2, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], wspace=0.35, hspace=0.45)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    ax = axes[0]
    x = np.arange(len(heldout))
    bottom = np.zeros(len(heldout))
    for behavior in BEHAVIOR_CLASSES:
        vals = heldout[f"{behavior}_bouts"].astype(float).to_numpy()
        ax.bar(x, vals, bottom=bottom, width=0.72, color=BEHAVIOR_COLORS[behavior], label=BEHAVIOR_LABELS[behavior])
        bottom += vals
    ax.set_xticks(x, [f"Movie {int(m)}" for m in heldout["movie_id"]], rotation=25, ha="right")
    ax.set_ylim(0, max(bottom) * 1.18)
    ax.set_ylabel("Primary bouts")
    panel_label(ax, "A")
    ax.set_title("Held-out Fly-v-Fly aggression bouts", loc="left")
    ax.legend(frameon=False, fontsize=8, ncols=2)
    style_axis(ax)

    ax = axes[1]
    order = BEHAVIOR_CLASSES
    pos = np.arange(len(order))
    median_by_class = med.groupby("behavior")["median_frames"].median().reindex(order)
    ax.bar(pos, median_by_class.to_numpy(), color=[BEHAVIOR_COLORS[b] for b in order], width=0.66)
    for i, behavior in enumerate(order):
        vals = med[med["behavior"].eq(behavior)]["median_frames"].to_numpy()
        if len(vals):
            jitter = np.linspace(-0.16, 0.16, len(vals)) if len(vals) > 1 else np.array([0.0])
            ax.scatter(np.full(len(vals), i) + jitter, vals, s=18, color=NEUTRAL["black"], alpha=0.75, zorder=3)
    ax.axhline(4, color=NEUTRAL["dark"], linestyle="--", linewidth=1.0)
    ax.text(
        len(order) - 0.2,
        4.25,
        "Median-collar scale",
        ha="right",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color=NEUTRAL["black"],
    )
    ax.set_xticks(pos, [BEHAVIOR_LABELS[b] for b in order], rotation=25, ha="right")
    ax.set_ylabel("Median duration (frames)")
    panel_label(ax, "B")
    ax.set_title("Short-bout duration scale", loc="left")
    style_axis(ax)

    ax = axes[2]
    metric_rows = []
    for _, row in primary.iterrows():
        metric_rows.extend([
            {"window": row["window"], "metric": "Precision", "value": row["mean_raw_frame_macro_precision"]},
            {"window": row["window"], "metric": "Recall", "value": row["mean_raw_frame_macro_recall"]},
            {"window": row["window"], "metric": "F1", "value": row["mean_raw_frame_macro_f1"]},
        ])
    frame_plot = pd.DataFrame(metric_rows)
    metric_order = ["Precision", "Recall", "F1"]
    metric_colors = {"Precision": "#0072B2", "Recall": "#D55E00", "F1": "#009E73"}
    x = np.arange(len(metric_order))
    width = 0.34
    for j, window in enumerate(["w16s8", "w8s4"]):
        vals = [frame_plot[(frame_plot["window"].eq(window)) & (frame_plot["metric"].eq(m))]["value"].iloc[0] for m in metric_order]
        ax.bar(x + (j - 0.5) * width, vals, width=width, color=WINDOW_COLORS[window], label=WINDOW_LABELS[window])
    ax.set_xticks(x, metric_order)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Macro metric")
    panel_label(ax, "C")
    ax.set_title("Strict primary-label frame metrics", loc="left")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    ax = axes[3]
    thresholds = ["0.10", "0.25", "0.50"]
    cols = ["mean_smoothed_bout_macro_f1_10", "mean_smoothed_bout_macro_f1_25", "mean_smoothed_bout_macro_f1_50"]
    x = np.arange(len(thresholds))
    for j, window in enumerate(["w16s8", "w8s4"]):
        row = primary[primary["window"].eq(window)].iloc[0]
        vals = [row[c] for c in cols]
        ax.plot(x, vals, marker="o", linewidth=2.0, color=WINDOW_COLORS[window], label=WINDOW_LABELS[window])
    ax.set_xticks(x, thresholds)
    ax.set_ylim(0, 0.55)
    ax.set_xlabel("tIoU threshold")
    ax.set_ylabel("Bout macro-F1")
    panel_label(ax, "D")
    ax.set_title("Strict bout recovery", loc="left")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    ax = axes[4]
    for window in ["w16s8", "w8s4"]:
        sub = collar_summary[collar_summary["window"].eq(window)].sort_values("collar_frames")
        ax.plot(
            sub["collar_frames"],
            sub["mean_macro_f1"],
            marker="o",
            linewidth=2.0,
            color=WINDOW_COLORS[window],
            label=WINDOW_LABELS[window],
        )
        ax.fill_between(
            sub["collar_frames"].to_numpy(dtype=float),
            (sub["mean_macro_f1"] - sub["sd_macro_f1"].fillna(0)).to_numpy(dtype=float),
            (sub["mean_macro_f1"] + sub["sd_macro_f1"].fillna(0)).to_numpy(dtype=float),
            color=WINDOW_COLORS[window],
            alpha=0.12,
            linewidth=0,
        )
    ax.axvline(4, color=NEUTRAL["dark"], linestyle="--", linewidth=1.0)
    ax.axvline(21, color=NEUTRAL["mid"], linestyle=":", linewidth=1.0)
    ax.text(4.3, 0.50, "median", rotation=90, va="center", ha="left", fontsize=8, color=NEUTRAL["dark"])
    ax.text(21.3, 0.02, "movie 6\nhuman SD", rotation=90, va="bottom", ha="left", fontsize=8, color=NEUTRAL["mid"])
    ax.set_xticks(COLLAR_TICKS)
    ax.set_ylim(0, 0.75)
    ax.set_xlabel("Boundary collar (frames)")
    ax.set_ylabel("Event macro-F1")
    panel_label(ax, "E")
    ax.set_title("Median-bout collar sensitivity", loc="left")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    ax = axes[5]
    class_rows = []
    for behavior in BEHAVIOR_CLASSES:
        class_rows.append({
            "behavior": behavior,
            "frame_f1": w8_per_class[f"raw_f1_{behavior}"].mean(),
            "precision": w8_per_class[f"raw_precision_{behavior}"].mean(),
            "recall": w8_per_class[f"raw_recall_{behavior}"].mean(),
        })
    class_plot = pd.DataFrame(class_rows)
    x = np.arange(len(class_plot))
    ax.bar(x, class_plot["frame_f1"], color=[BEHAVIOR_COLORS[b] for b in class_plot["behavior"]], width=0.64)
    ax.scatter(
        x - 0.12,
        class_plot["precision"],
        color=NEUTRAL["black"],
        marker="o",
        s=24,
        label="Precision",
        zorder=3,
    )
    ax.scatter(
        x + 0.12,
        class_plot["recall"],
        color=NEUTRAL["mid"],
        marker="^",
        s=28,
        label="Recall",
        zorder=3,
    )
    ax.set_xticks(x, [BEHAVIOR_LABELS[b] for b in class_plot["behavior"]], rotation=25, ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Frame metric")
    panel_label(ax, "F")
    ax.set_title("Attention-256 w8/s4 class profile", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    style_axis(ax)

    fig.tight_layout()
    savefig(fig, fig_dir / "fig_fly_v_fly_short_bout_adaptation")

    # Pose profile is kept as a separate figure because it uses a different
    # validation target than behavior decoding. Prefer the held-out track-based
    # PCK/OKS profile when the raw table has been generated; retain the mAP
    # summary fallback for incomplete workspaces.
    pose_raw = TABLE_DIR / "fly_pose_track_keypoint_eval_raw.csv"
    if pose_raw.exists():
        plot_fly_pose_profile(pd.read_csv(pose_raw), fig_dir, "fig_fly_v_fly_pose_profile")
    else:
        pose_plot = pose[pose["class"].isin(["all", *BEHAVIOR_CLASSES])].copy()
        pose_plot["label"] = pose_plot["class"].map(lambda c: "All" if c == "all" else BEHAVIOR_LABELS[c])
        order_labels = ["All", *[BEHAVIOR_LABELS[b] for b in BEHAVIOR_CLASSES]]
        pose_plot["label"] = pd.Categorical(pose_plot["label"], order_labels, ordered=True)
        pose_plot = pose_plot.sort_values("label")
        fig, ax = plt.subplots(figsize=(7.2, 3.8))
        y = np.arange(len(pose_plot))
        ax.barh(y - 0.18, pose_plot["box_map5095"], height=0.32, color=NEUTRAL["dark"], label="Box mAP50-95")
        ax.barh(y + 0.18, pose_plot["pose_map5095"], height=0.32, color="#009E73", label="Pose mAP50-95")
        ax.set_yticks(y, pose_plot["label"])
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("Validation mAP50-95")
        panel_label(ax, "A", x=-0.16)
        ax.set_title("Fly YOLO-pose validation profile", loc="left")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        style_axis(ax, grid_axis="x")
        fig.tight_layout()
        savefig(fig, fig_dir / "fig_fly_v_fly_pose_profile")


def plot_annotation_supplement(
    heldout: pd.DataFrame,
    class_agree: pd.DataFrame,
    frame_agree: pd.DataFrame,
    human_human: pd.DataFrame,
    cross: pd.DataFrame,
    per_video: pd.DataFrame,
    fig_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))
    axes = axes.ravel()

    ax = axes[0]
    x = np.arange(len(heldout))
    available = heldout["secondary_annotation_available"].astype(bool).to_numpy()
    ax.bar(x, np.ones(len(heldout)), color=np.where(available, "#009E73", NEUTRAL["light"]), width=0.65)
    ax.set_xticks(x, [f"Movie {int(m)}" for m in heldout["movie_id"]])
    ax.set_yticks([0, 1], ["", "Secondary\nannotation"])
    ax.set_ylim(0, 1.25)
    panel_label(ax, "A")
    ax.set_title("Held-out secondary-annotation availability", loc="left")
    style_axis(ax)

    ax = axes[1]
    movie6 = class_agree[class_agree["movie_id"].eq(6) & class_agree["behavior"].isin(BEHAVIOR_CLASSES)].copy()
    movie6 = movie6.set_index("behavior").reindex(BEHAVIOR_CLASSES).reset_index()
    x = np.arange(len(movie6))
    same = movie6["same_label_pct"].astype(float).fillna(0)
    other = movie6["secondary_other_pct"].astype(float).fillna(0)
    diff = movie6["secondary_different_behavior_pct"].astype(float).fillna(0)
    ax.bar(x, same, color="#0072B2", label="Same label")
    ax.bar(x, other, bottom=same, color="#C9A227", label="Secondary other")
    ax.bar(x, diff, bottom=same + other, color="#D55E00", label="Different behavior")
    ax.set_xticks(x, [BEHAVIOR_LABELS[b] for b in BEHAVIOR_CLASSES], rotation=25, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Primary behavior frames (%)")
    panel_label(ax, "B")
    ax.set_title("Movie 6 class-conditioned agreement", loc="left")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    ax = axes[2]
    modes = ["primary", "secondary", "union", "intersection", "padded_primary"]
    mode_labels = ["Primary", "Secondary", "Union", "Intersection", "Padded\nprimary"]
    w8 = per_video[
        per_video["window"].eq("w8s4")
        & per_video["movie_id"].eq(6)
        & per_video["gt_mode"].isin(modes)
    ].copy()
    w8["gt_mode"] = pd.Categorical(w8["gt_mode"], modes, ordered=True)
    w8 = w8.sort_values("gt_mode")
    x = np.arange(len(w8))
    ax.bar(x - 0.18, w8["raw_frame_macro_f1"], width=0.34, color="#0072B2", label="Frame macro-F1")
    ax.bar(x + 0.18, w8["raw_bout_macro_f1_025"], width=0.34, color="#009E73", label="Bout F1@0.25")
    ax.set_xticks(x, mode_labels)
    ax.set_ylim(0, 0.62)
    ax.set_ylabel("Macro-F1")
    panel_label(ax, "C")
    ax.set_title("Movie 6 annotation-mode sensitivity", loc="left")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    ax = axes[3]
    hh = human_human.iloc[0]
    movie6_model = w8[w8["gt_mode"].eq("primary")].iloc[0]
    labels = ["Human-human", "Model-primary"]
    frame_vals = [hh["frame_macro_f1"], movie6_model["raw_frame_macro_f1"]]
    bout_vals = [hh["bout_macro_f1_025"], movie6_model["raw_bout_macro_f1_025"]]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, frame_vals, width=0.34, color="#0072B2", label="Frame macro-F1")
    ax.bar(x + 0.18, bout_vals, width=0.34, color="#009E73", label="Bout F1@0.25")
    note = frame_agree[frame_agree["movie_id"].eq(6)].iloc[0]
    ax.text(
        0.0,
        0.665,
        f"Movie 6 only: overall agreement {note['overall_frame_agreement_pct']:.1f}%",
        ha="left",
        va="top",
        fontsize=8,
        color=NEUTRAL["dark"],
    )
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 0.72)
    ax.set_ylabel("Macro-F1")
    panel_label(ax, "D")
    ax.set_title("Local human-reference comparison", loc="left")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    fig.tight_layout()
    savefig(fig, fig_dir / "supp_fly_v_fly_annotation_reference")


def main() -> int:
    args = parse_args()
    apply_style()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    args.supp_fig_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    copy_eval_csvs(args.eval_dir, args.table_dir)
    cross = parse_run_fields(pd.read_csv(args.eval_dir / "cross_run_summary.csv"))
    per_video = parse_run_fields(pd.read_csv(args.eval_dir / "per_video_metrics.csv"))
    heldout = pd.read_csv(args.table_dir / "fly_video_annotation_surface_heldout.csv")
    class_agree = pd.read_csv(args.table_dir / "fly_interannotator_class_agreement_by_video.csv")
    frame_agree = pd.read_csv(args.table_dir / "fly_interannotator_frame_agreement_by_video.csv")
    human_human = pd.read_csv(args.eval_dir / "human_human_agreement.csv")
    pose = load_pose_validation(args.pose_val_txt)

    write_primary_tables(cross, per_video, args.table_dir)
    collar = compute_collar_metrics(args.eval_dir, heldout, args.table_dir)

    plot_main_figure(heldout, cross, per_video, collar, pose, args.fig_dir)
    plot_annotation_supplement(heldout, class_agree, frame_agree, human_human, cross, per_video, args.supp_fig_dir)

    print(f"Wrote figures to {args.fig_dir} and {args.supp_fig_dir}")
    print(f"Wrote tables to {args.table_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
