from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_DIR = (
    Path("outputs")
    / "dlc_topdown_heldout_eval"
    / "mars_dlc_topdown_hrnet_attention256_seed42"
)
DEFAULT_ANALYSIS_DIR = (
    Path("outputs")
    / "dlc_topdown_heldout_eval"
    / "mars_dlc_topdown_hrnet_attention256_seed42"
    / "analysis_reproduced"
)
DEFAULT_MANUSCRIPT_ROOT = PACKAGE_ROOT
DEFAULT_STYLE_PATH = (
    PACKAGE_ROOT
    / "scripts_by_experiment"
    / "00_figure_table_rendering"
    / "current_analysis"
    / "behaviorscope_figure_style.py"
)
CLASS_ORDER = ["attack", "investigation", "mount", "other"]
PREDICTION_ORDER = ["raw_windows", "smoothed_frames"]
COMPARISON_METRICS = [
    ("frame_macro_f1_present_behaviors", "Frame macro F1"),
    ("bout_macro_f1_iou25_present_behaviors", "Bout F1 IoU@0.25"),
    ("bout_macro_f1_iou50_present_behaviors", "Bout F1 IoU@0.50"),
]


def load_style():
    style_path = DEFAULT_STYLE_PATH
    if style_path.is_file():
        import importlib.util

        spec = importlib.util.spec_from_file_location("behaviorscope_figure_style", style_path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            module.apply_style()
            return module

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.alpha": 0.25,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    class FallbackStyle:
        BEHAVIOR_COLORS = {
            "attack": "#D55E00",
            "investigation": "#0072B2",
            "mount": "#009E73",
            "other": "#777777",
        }
        NEUTRAL = {"dark": "#374151", "mid": "#6B7280", "light": "#E5E7EB"}

        @staticmethod
        def save_figure(fig, out_base, dpi=600):
            out_base = Path(out_base).with_suffix("")
            out_base.parent.mkdir(parents=True, exist_ok=True)
            outputs = [out_base.with_suffix(".png"), out_base.with_suffix(".pdf")]
            fig.savefig(outputs[0], dpi=dpi, bbox_inches="tight")
            fig.savefig(outputs[1], bbox_inches="tight")
            return outputs

        @staticmethod
        def title_with_panel(ax, label, title, pad=8.0):
            ax.set_title(f"({label}) {title}", loc="left", fontsize=11, pad=pad)

    return FallbackStyle


STYLE = load_style()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze DLC-HRNet held-out BehaviorScope-X evaluation outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--eval_dir", type=Path, default=DEFAULT_EVAL_DIR)
    p.add_argument("--analysis_dir", type=Path, default=DEFAULT_ANALYSIS_DIR)
    p.add_argument("--manuscript_root", type=Path, default=DEFAULT_MANUSCRIPT_ROOT)
    p.add_argument("--no_manuscript_copy", action="store_true")
    return p.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def sem(series: pd.Series) -> float:
    series = pd.to_numeric(series, errors="coerce").dropna()
    if len(series) <= 1:
        return float("nan")
    return float(series.std(ddof=1) / math.sqrt(len(series)))


def summarize_per_video(per_video: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "accuracy",
        "frame_macro_f1_present_behaviors",
        "bout_macro_f1_iou10_present_behaviors",
        "bout_macro_f1_iou25_present_behaviors",
        "bout_macro_f1_iou50_present_behaviors",
        "gt_bouts_total",
        "pred_bouts_total",
    ]
    rows = []
    for (prediction_type, split), group in per_video.groupby(["prediction_type", "split"], sort=False):
        row = {
            "prediction_type": prediction_type,
            "split": split,
            "n_videos": int(group["video_id"].nunique()),
        }
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_median"] = float(vals.median())
            row[f"{metric}_sem"] = sem(vals)
        rows.append(row)
    for prediction_type, group in per_video.groupby("prediction_type", sort=False):
        row = {
            "prediction_type": prediction_type,
            "split": "pooled_test_1_test_2",
            "n_videos": int(group["video_id"].nunique()),
        }
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_median"] = float(vals.median())
            row[f"{metric}_sem"] = sem(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_per_class(per_class: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "frame_precision",
        "frame_recall",
        "frame_f1",
        "bout_precision_iou25",
        "bout_recall_iou25",
        "bout_f1_iou25",
        "bout_precision_iou50",
        "bout_recall_iou50",
        "bout_f1_iou50",
    ]
    rows = []
    for (prediction_type, cls), group in per_class.groupby(["prediction_type", "class"], sort=False):
        row = {
            "prediction_type": prediction_type,
            "class": cls,
            "n_videos": int(group["video_id"].nunique()),
            "gt_frames_total": int(pd.to_numeric(group["gt_frames"], errors="coerce").sum()),
            "pred_frames_total": int(pd.to_numeric(group["pred_frames"], errors="coerce").sum()),
        }
        for metric in metrics:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_median"] = float(vals.median())
            row[f"{metric}_sem"] = sem(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def build_confusion(per_class: pd.DataFrame, prediction_type: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = per_class[per_class["prediction_type"] == prediction_type].copy()
    rows = []
    for cls in CLASS_ORDER:
        cls_rows = sub[sub["class"] == cls]
        row = {}
        for pred_cls in CLASS_ORDER:
            col = f"cm_{pred_cls}"
            row[pred_cls] = int(pd.to_numeric(cls_rows[col], errors="coerce").sum())
        rows.append(row)
    counts = pd.DataFrame(rows, index=CLASS_ORDER)
    denom = counts.sum(axis=1).replace(0, np.nan)
    recall = counts.div(denom, axis=0).fillna(0.0)
    return counts, recall


def plot_metric_summary(per_video: pd.DataFrame, out_dir: Path) -> list[Path]:
    metric_specs = [
        ("accuracy", "Accuracy"),
        ("frame_macro_f1_present_behaviors", "Frame macro F1"),
        ("bout_macro_f1_iou25_present_behaviors", "Bout F1 IoU@0.25"),
        ("bout_macro_f1_iou50_present_behaviors", "Bout F1 IoU@0.50"),
    ]
    pooled = per_video.copy()
    pooled["prediction_type"] = pd.Categorical(
        pooled["prediction_type"], PREDICTION_ORDER, ordered=True
    )
    fig, axes = plt.subplots(1, len(metric_specs), figsize=(10.8, 3.0), sharey=False)
    colors = {"raw_windows": "#2563EB", "smoothed_frames": "#10B981"}
    for panel, ax, (metric, label) in zip(["A", "B", "C", "D"], axes, metric_specs):
        groups = []
        means = []
        errors = []
        for pred_type in PREDICTION_ORDER:
            vals = pd.to_numeric(
                pooled.loc[pooled["prediction_type"] == pred_type, metric], errors="coerce"
            ).dropna()
            groups.append("Raw\nwindows" if pred_type == "raw_windows" else "Smoothed\nframes")
            means.append(vals.mean())
            errors.append(sem(vals))
        x = np.arange(len(groups))
        ax.bar(x, means, yerr=errors, capsize=3, color=[colors[p] for p in PREDICTION_ORDER], edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylim(0, 1.02 if "bouts_total" not in metric else None)
        ax.set_ylabel(label if ax is axes[0] else "")
        ax.set_title(label, fontsize=10)
        ax.text(
            -0.18,
            1.08,
            f"({panel})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )
    fig.suptitle("DLC-HRNet held-out performance", x=0.01, y=1.04, ha="left", fontsize=11)
    fig.tight_layout(w_pad=1.0)
    outputs = STYLE.save_figure(fig, out_dir / "dlc_heldout_metric_summary")
    plt.close(fig)
    return outputs


def plot_per_class(per_class_summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    metrics = [
        ("frame_f1_mean", "Frame F1"),
        ("bout_f1_iou25_mean", "Bout F1 IoU@0.25"),
        ("bout_f1_iou50_mean", "Bout F1 IoU@0.50"),
    ]
    raw = per_class_summary[per_class_summary["prediction_type"] == "raw_windows"].copy()
    raw["class"] = pd.Categorical(raw["class"], CLASS_ORDER, ordered=True)
    raw = raw.sort_values("class")
    x = np.arange(len(CLASS_ORDER))
    width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    colors = ["#2563EB", "#94A3B8", "#10B981"]
    for i, (metric, label) in enumerate(metrics):
        vals = [float(raw.loc[raw["class"] == cls, metric].iloc[0]) if cls in set(raw["class"].astype(str)) else np.nan for cls in CLASS_ORDER]
        ax.bar(x + (i - 1) * width, vals, width=width, color=colors[i], label=label, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([c.capitalize() for c in CLASS_ORDER])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Mean per-video score")
    ax.set_title("DLC-HRNet held-out class-wise scores", loc="left", fontsize=11, pad=8)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    outputs = STYLE.save_figure(fig, out_dir / "dlc_heldout_per_class_scores")
    plt.close(fig)
    return outputs


def plot_confusion(recall: pd.DataFrame, out_dir: Path) -> list[Path]:
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(recall.loc[CLASS_ORDER, CLASS_ORDER].values, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(np.arange(len(CLASS_ORDER)))
    ax.set_yticks(np.arange(len(CLASS_ORDER)))
    ax.set_xticklabels([c.capitalize() for c in CLASS_ORDER], rotation=35, ha="right")
    ax.set_yticklabels([c.capitalize() for c in CLASS_ORDER])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i, true_cls in enumerate(CLASS_ORDER):
        for j, pred_cls in enumerate(CLASS_ORDER):
            val = recall.loc[true_cls, pred_cls]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="#111827" if val < 0.55 else "white")
    ax.set_title("DLC-HRNet held-out row-normalized confusion", loc="left", fontsize=11, pad=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall")
    fig.tight_layout()
    outputs = STYLE.save_figure(fig, out_dir / "dlc_heldout_confusion_recall_raw")
    plt.close(fig)
    return outputs


def load_backbone_comparison(per_video: pd.DataFrame, manuscript_root: Path) -> pd.DataFrame:
    tables_root = manuscript_root / "tables" / "production"
    yolo_path = tables_root / "yolo_sppf_main" / "yolo_lstm_attention_per_video_long.csv"
    mobilenet_path = tables_root / "mobilenet_portability" / "mobilenet_portability_per_video_long.csv"
    if not yolo_path.is_file() or not mobilenet_path.is_file():
        return pd.DataFrame()

    metric_cols = [m[0] for m in COMPARISON_METRICS]
    keep = ["split", "video_id", "prediction_type", *metric_cols]
    rows = []

    dlc = per_video[keep].copy()
    dlc["backbone_label"] = "DLC-HRNet"
    dlc["comparison_note"] = "DLC top-down detector+pose cache; single seed 42"
    rows.append(dlc)

    yolo = pd.read_csv(yolo_path)
    for col in ["stream", "head", "prediction_type"]:
        yolo[col] = yolo[col].astype(str).str.strip()
    yolo_sub = yolo[
        (yolo["stream"] == "full")
        & (yolo["head"] == "attention")
        & (pd.to_numeric(yolo["capacity"], errors="coerce") == 256)
        & (pd.to_numeric(yolo["seed"], errors="coerce") == 42)
    ][keep].copy()
    yolo_sub["backbone_label"] = "YOLO-SPPF"
    yolo_sub["comparison_note"] = "YOLO-pose SPPF features; Attention-256 seed 42"
    rows.append(yolo_sub)

    mobile = pd.read_csv(mobilenet_path)
    for col in ["backbone", "head", "prediction_type"]:
        mobile[col] = mobile[col].astype(str).str.strip()
    mobile_sub = mobile[
        (mobile["backbone"] == "mobilenetv3_native")
        & (mobile["head"] == "attention")
        & (pd.to_numeric(mobile["seed"], errors="coerce") == 42)
    ][keep].copy()
    mobile_sub["backbone_label"] = "MobileNetV3"
    mobile_sub["comparison_note"] = "MobileNetV3-native features; Attention-256 seed 42; smoothed output available in production table"
    rows.append(mobile_sub)

    comp = pd.concat(rows, ignore_index=True)
    comp["backbone_label"] = pd.Categorical(
        comp["backbone_label"], ["YOLO-SPPF", "MobileNetV3", "DLC-HRNet"], ordered=True
    )
    comp["prediction_type"] = pd.Categorical(comp["prediction_type"], PREDICTION_ORDER, ordered=True)
    return comp.sort_values(["backbone_label", "prediction_type", "split", "video_id"])


def summarize_backbone_comparison(comp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (backbone, prediction_type), group in comp.groupby(["backbone_label", "prediction_type"], observed=True):
        row = {
            "backbone_label": str(backbone),
            "prediction_type": str(prediction_type),
            "n_videos": int(group["video_id"].nunique()),
        }
        for metric, _label in COMPARISON_METRICS:
            vals = pd.to_numeric(group[metric], errors="coerce")
            row[f"{metric}_mean"] = float(vals.mean())
            row[f"{metric}_median"] = float(vals.median())
            row[f"{metric}_sem"] = sem(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_backbone_comparison(summary: pd.DataFrame, out_dir: Path) -> list[Path]:
    if summary.empty:
        return []
    plot_rows = [
        ("YOLO-SPPF", "raw_windows", "YOLO-SPPF\nraw"),
        ("YOLO-SPPF", "smoothed_frames", "YOLO-SPPF\nsmoothed"),
        ("MobileNetV3", "smoothed_frames", "MobileNetV3\nsmoothed"),
        ("DLC-HRNet", "raw_windows", "DLC-HRNet\nraw"),
        ("DLC-HRNet", "smoothed_frames", "DLC-HRNet\nsmoothed"),
    ]
    colors = {
        "YOLO-SPPF": "#0072B2",
        "MobileNetV3": "#D55E00",
        "DLC-HRNet": "#10B981",
    }
    fig, axes = plt.subplots(1, len(COMPARISON_METRICS), figsize=(11.8, 3.6), sharey=True)
    for panel, ax, (metric, label) in zip(["A", "B", "C"], axes, COMPARISON_METRICS):
        means = []
        errors = []
        labels = []
        bar_colors = []
        for backbone, pred_type, display in plot_rows:
            row = summary[
                (summary["backbone_label"] == backbone)
                & (summary["prediction_type"] == pred_type)
            ]
            labels.append(display)
            bar_colors.append(colors[backbone])
            if row.empty:
                means.append(np.nan)
                errors.append(np.nan)
            else:
                means.append(float(row[f"{metric}_mean"].iloc[0]))
                errors.append(float(row[f"{metric}_sem"].iloc[0]))
        x = np.arange(len(labels))
        ax.bar(x, means, yerr=errors, capsize=3, color=bar_colors, edgecolor="none")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylim(0, 1.02)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("Mean per-video score" if ax is axes[0] else "")
        ax.text(
            -0.16,
            1.08,
            f"({panel})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )
    fig.suptitle("Held-out pose-backbone feature pipeline comparison", x=0.01, y=1.04, ha="left", fontsize=11)
    fig.tight_layout(w_pad=1.0)
    outputs = STYLE.save_figure(fig, out_dir / "dlc_vs_yolo_mobilenet_metric_comparison")
    plt.close(fig)
    return outputs


def plot_dlc_yolo_raw_delta(comp: pd.DataFrame, out_dir: Path) -> list[Path]:
    if comp.empty:
        return []
    dlc = comp[
        (comp["backbone_label"].astype(str) == "DLC-HRNet")
        & (comp["prediction_type"].astype(str) == "raw_windows")
    ]
    yolo = comp[
        (comp["backbone_label"].astype(str) == "YOLO-SPPF")
        & (comp["prediction_type"].astype(str) == "raw_windows")
    ]
    if dlc.empty or yolo.empty:
        return []
    metric_cols = [m[0] for m in COMPARISON_METRICS]
    merged = dlc[["split", "video_id", *metric_cols]].merge(
        yolo[["video_id", *metric_cols]], on="video_id", suffixes=("_dlc", "_yolo")
    )
    if merged.empty:
        return []
    fig, axes = plt.subplots(1, len(COMPARISON_METRICS), figsize=(10.8, 3.2), sharey=False)
    rng = np.random.default_rng(42)
    for panel, ax, (metric, label) in zip(["A", "B", "C"], axes, COMPARISON_METRICS):
        delta = pd.to_numeric(merged[f"{metric}_dlc"], errors="coerce") - pd.to_numeric(
            merged[f"{metric}_yolo"], errors="coerce"
        )
        delta = delta.dropna()
        x = rng.normal(loc=0.0, scale=0.025, size=len(delta))
        ax.axhline(0, color="#374151", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.scatter(x, delta, s=22, color="#10B981", alpha=0.78, edgecolor="white", linewidth=0.4)
        mean_delta = float(delta.mean())
        ax.errorbar(
            [0.18],
            [mean_delta],
            yerr=[sem(delta)],
            fmt="o",
            color="#111827",
            capsize=4,
            markersize=5,
            label="Mean +/- SEM" if panel == "A" else None,
        )
        ax.set_xlim(-0.16, 0.34)
        ax.set_xticks([0.0, 0.18])
        ax.set_xticklabels(["Videos", "Mean"], rotation=0)
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("DLC-HRNet raw - YOLO-SPPF raw" if ax is axes[0] else "")
        ax.text(
            -0.18,
            1.08,
            f"({panel})",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
            clip_on=False,
        )
    fig.suptitle("Per-video DLC-HRNet raw-window gain over YOLO-SPPF", x=0.01, y=1.04, ha="left", fontsize=11)
    fig.tight_layout(w_pad=1.1)
    outputs = STYLE.save_figure(fig, out_dir / "dlc_vs_yolo_raw_per_video_delta")
    plt.close(fig)
    return outputs


def write_report(
    out_path: Path,
    aggregate: dict,
    per_video_summary: pd.DataFrame,
    per_class_summary: pd.DataFrame,
    confusion_counts: pd.DataFrame,
    backbone_summary: pd.DataFrame,
    generated_files: list[Path],
) -> None:
    raw_row = per_video_summary[
        (per_video_summary["prediction_type"] == "raw_windows")
        & (per_video_summary["split"] == "pooled_test_1_test_2")
    ].iloc[0]
    smooth_row = per_video_summary[
        (per_video_summary["prediction_type"] == "smoothed_frames")
        & (per_video_summary["split"] == "pooled_test_1_test_2")
    ].iloc[0]
    cm_lines = [
        "| True \\ Pred | Attack | Investigation | Mount | Other |",
        "|---|---:|---:|---:|---:|",
    ]
    for true_cls in CLASS_ORDER:
        cm_lines.append(
            "| "
            + true_cls.capitalize()
            + " | "
            + " | ".join(str(int(confusion_counts.loc[true_cls, pred_cls])) for pred_cls in CLASS_ORDER)
            + " |"
        )
    lines = [
        "# DLC-HRNet Held-Out Evaluation Analysis",
        "",
        "Date generated: 2026-06-08",
        "",
        "## Input",
        "",
        f"- Evaluated videos: {aggregate.get('n_evaluated_videos')}",
        f"- Manifest videos: {aggregate.get('n_manifest_videos')}",
        f"- Windows: {aggregate.get('n_windows')}",
        f"- Blocking failures: {aggregate.get('n_blocking_failures')}",
        f"- Threshold decoder disabled: {aggregate.get('threshold_decoder_disabled')}",
        "",
        "## Pooled Per-Video Summary",
        "",
        "| Prediction | Accuracy | Frame macro F1 | Bout F1 IoU@0.25 | Bout F1 IoU@0.50 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Raw windows | {raw_row['accuracy_mean']:.3f} | "
            f"{raw_row['frame_macro_f1_present_behaviors_mean']:.3f} | "
            f"{raw_row['bout_macro_f1_iou25_present_behaviors_mean']:.3f} | "
            f"{raw_row['bout_macro_f1_iou50_present_behaviors_mean']:.3f} |"
        ),
        (
            f"| Smoothed frames | {smooth_row['accuracy_mean']:.3f} | "
            f"{smooth_row['frame_macro_f1_present_behaviors_mean']:.3f} | "
            f"{smooth_row['bout_macro_f1_iou25_present_behaviors_mean']:.3f} | "
            f"{smooth_row['bout_macro_f1_iou50_present_behaviors_mean']:.3f} |"
        ),
        "",
        "Scores are means over held-out videos, not pooled frame-level aggregates.",
        "",
        "## Confusion Counts",
        "",
        *cm_lines,
        "",
    ]
    if not backbone_summary.empty:
        lines.extend(
            [
                "## Pose-Backbone Feature Pipeline Comparison",
                "",
                "| Backbone | Prediction | Frame macro F1 | Bout F1 IoU@0.25 | Bout F1 IoU@0.50 |",
                "|---|---|---:|---:|---:|",
            ]
        )
        for _, row in backbone_summary.iterrows():
            lines.append(
                f"| {row['backbone_label']} | {row['prediction_type']} | "
                f"{float(row['frame_macro_f1_present_behaviors_mean']):.3f} | "
                f"{float(row['bout_macro_f1_iou25_present_behaviors_mean']):.3f} | "
                f"{float(row['bout_macro_f1_iou50_present_behaviors_mean']):.3f} |"
            )
        lines.extend(
            [
                "",
                "The raw-window DLC-HRNet comparison is directly paired with YOLO-SPPF on the same 28 held-out videos. "
                "MobileNetV3 production rows are available as smoothed-frame outputs, so the three-backbone comparison should be interpreted as a pipeline-level comparison rather than an isolated visual-backbone-only ablation.",
                "",
            ]
        )
    lines.extend(
        [
        "## Generated Files",
        "",
        ]
    )
    lines.extend(f"- `{path.name}`" for path in generated_files)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_tree_files(source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, target_dir / path.name)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    args = parse_args()
    eval_dir = args.eval_dir.resolve()
    analysis_dir = args.analysis_dir.resolve()
    table_dir = analysis_dir / "tables"
    figure_dir = analysis_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    aggregate = json.loads((eval_dir / "aggregate_summary.json").read_text(encoding="utf-8"))
    per_video = safe_read_csv(eval_dir / "per_video_summary.csv")
    per_class = safe_read_csv(eval_dir / "per_class_metrics.csv")

    per_video_summary = summarize_per_video(per_video)
    per_class_summary = summarize_per_class(per_class)
    confusion_counts, confusion_recall = build_confusion(per_class, "raw_windows")
    backbone_comparison = load_backbone_comparison(per_video, args.manuscript_root)
    backbone_summary = summarize_backbone_comparison(backbone_comparison) if not backbone_comparison.empty else pd.DataFrame()

    outputs: list[Path] = []
    per_video_summary_path = table_dir / "dlc_heldout_per_video_summary.csv"
    per_class_summary_path = table_dir / "dlc_heldout_per_class_summary.csv"
    confusion_counts_path = table_dir / "dlc_heldout_confusion_counts_raw.csv"
    confusion_recall_path = table_dir / "dlc_heldout_confusion_recall_raw.csv"
    backbone_comparison_path = table_dir / "dlc_yolo_mobilenet_per_video_comparison_long.csv"
    backbone_summary_path = table_dir / "dlc_yolo_mobilenet_metric_comparison_summary.csv"
    per_video_summary.to_csv(per_video_summary_path, index=False)
    per_class_summary.to_csv(per_class_summary_path, index=False)
    confusion_counts.to_csv(confusion_counts_path)
    confusion_recall.to_csv(confusion_recall_path)
    if not backbone_comparison.empty:
        backbone_comparison.to_csv(backbone_comparison_path, index=False)
        backbone_summary.to_csv(backbone_summary_path, index=False)
        outputs.extend([backbone_comparison_path, backbone_summary_path])
    outputs.extend([per_video_summary_path, per_class_summary_path, confusion_counts_path, confusion_recall_path])

    outputs.extend(plot_metric_summary(per_video, figure_dir))
    outputs.extend(plot_per_class(per_class_summary, figure_dir))
    outputs.extend(plot_confusion(confusion_recall, figure_dir))
    if not backbone_summary.empty:
        outputs.extend(plot_backbone_comparison(backbone_summary, figure_dir))
        outputs.extend(plot_dlc_yolo_raw_delta(backbone_comparison, figure_dir))

    report_path = analysis_dir / "DLC_HELDOUT_ANALYSIS_SUMMARY.md"
    write_report(report_path, aggregate, per_video_summary, per_class_summary, confusion_counts, backbone_summary, outputs)
    outputs.append(report_path)

    provenance = {
        "eval_dir": str(eval_dir),
        "analysis_dir": str(analysis_dir),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256(Path(__file__).resolve()),
        "n_generated_files": len(outputs),
        "generated_files": [str(p) for p in outputs],
    }
    provenance_path = analysis_dir / "analysis_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    outputs.append(provenance_path)

    if not args.no_manuscript_copy and args.manuscript_root.exists():
        f_tables = args.manuscript_root / "tables" / "production" / "dlc_superanimal_topdown"
        f_figures = args.manuscript_root / "figures" / "production" / "dlc_superanimal_topdown"
        f_scripts = args.manuscript_root / "penultimate_scripts" / "current_analysis" / "dlc_superanimal_topdown"
        copy_tree_files(table_dir, f_tables)
        copy_tree_files(figure_dir, f_figures)
        f_scripts.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve(), f_scripts / Path(__file__).name)
        shutil.copy2(report_path, f_scripts / report_path.name)
        shutil.copy2(provenance_path, f_scripts / provenance_path.name)
        print(f"[copy] tables -> {f_tables}")
        print(f"[copy] figures -> {f_figures}")
        print(f"[copy] scripts -> {f_scripts}")

    print(f"[done] analysis -> {analysis_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
