import argparse
from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLES = PACKAGE_ROOT / "tables" / "production" / "dlc_superanimal_topdown"
DEFAULT_OUT = PACKAGE_ROOT / "figures_reproduced" / "production" / "dlc_superanimal_topdown"
DEFAULT_STYLE_PATH = (
    PACKAGE_ROOT
    / "scripts_by_experiment"
    / "00_figure_table_rendering"
    / "current_analysis"
    / "behaviorscope_figure_style.py"
)

CLASS_ORDER = ["attack", "investigation", "mount", "other"]
CLASS_LABELS = ["Attack", "Investigation", "Mount", "Other"]
PREDICTION_ORDER = ["raw_windows", "smoothed_frames"]
PREDICTION_LABELS = ["Raw\nwindows", "Smoothed\nframes"]
PREDICTION_COLORS = {
    "raw_windows": "#2563EB",
    "smoothed_frames": "#10B981",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table_dir", type=Path, default=DEFAULT_TABLES)
    parser.add_argument("--figure_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--style_path", type=Path, default=DEFAULT_STYLE_PATH)
    return parser.parse_args()


def load_style(style_path: Path):
    spec = importlib.util.spec_from_file_location("behaviorscope_figure_style", style_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def pooled_row(per_video: pd.DataFrame, prediction_type: str) -> pd.Series:
    row = per_video[
        (per_video["prediction_type"] == prediction_type)
        & (per_video["split"] == "pooled_test_1_test_2")
    ]
    if row.empty:
        raise ValueError(f"Missing pooled row for {prediction_type}")
    return row.iloc[0]


def main():
    args = parse_args()
    style = load_style(args.style_path)
    style.apply_style()
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    per_video = pd.read_csv(args.table_dir / "dlc_heldout_per_video_summary.csv")
    per_class = pd.read_csv(args.table_dir / "dlc_heldout_per_class_summary.csv")
    confusion = pd.read_csv(args.table_dir / "dlc_heldout_confusion_recall_raw.csv", index_col=0)

    fig = plt.figure(figsize=(11.4, 7.1))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.05],
        width_ratios=[1.48, 1.0],
        hspace=0.48,
        wspace=0.34,
    )
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    metric_specs = [
        ("accuracy", "Accuracy"),
        ("frame_macro_f1_present_behaviors", "Frame macro F1"),
        ("bout_macro_f1_iou25_present_behaviors", "Bout F1\nTIoU@0.25"),
        ("bout_macro_f1_iou50_present_behaviors", "Bout F1\nTIoU@0.50"),
    ]
    x = np.arange(len(metric_specs))
    width = 0.34
    for i, prediction_type in enumerate(PREDICTION_ORDER):
        row = pooled_row(per_video, prediction_type)
        means = [row[f"{metric}_mean"] for metric, _ in metric_specs]
        sems = [row[f"{metric}_sem"] for metric, _ in metric_specs]
        offset = (i - 0.5) * width
        ax_a.bar(
            x + offset,
            means,
            yerr=sems,
            capsize=3,
            width=width,
            color=PREDICTION_COLORS[prediction_type],
            alpha=0.92,
            ecolor="#374151",
            linewidth=0.8,
            label=PREDICTION_LABELS[i].replace("\n", " "),
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([label for _, label in metric_specs])
    ax_a.set_ylim(0, 1.02)
    ax_a.set_ylabel("Mean per-video score")
    style.title_with_panel(ax_a, "A", "Held-out performance and smoothing sensitivity")
    ax_a.legend(frameon=False, loc="upper right", ncol=2)

    class_metrics = [
        ("frame_f1_mean", "frame_f1_sem", "Frame F1", "#2563EB"),
        ("bout_f1_iou25_mean", "bout_f1_iou25_sem", "Bout F1\nTIoU@0.25", "#94A3B8"),
        ("bout_f1_iou50_mean", "bout_f1_iou50_sem", "Bout F1\nTIoU@0.50", "#10B981"),
    ]
    raw_classes = (
        per_class[per_class["prediction_type"] == "raw_windows"]
        .set_index("class")
        .loc[CLASS_ORDER]
        .reset_index()
    )
    x = np.arange(len(raw_classes))
    width = 0.24
    for i, (metric, err_metric, label, color) in enumerate(class_metrics):
        ax_b.bar(
            x + (i - 1) * width,
            raw_classes[metric],
            yerr=raw_classes[err_metric],
            capsize=2.5,
            width=width,
            color=color,
            alpha=0.92,
            ecolor="#374151",
            linewidth=0.8,
            label=label.replace("\n", " "),
        )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels(CLASS_LABELS, rotation=18, ha="right")
    ax_b.set_ylim(0, 1.02)
    ax_b.set_ylabel("Mean per-video score")
    style.title_with_panel(ax_b, "B", "Class-wise raw-window performance")
    ax_b.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)

    confusion = confusion.loc[CLASS_ORDER, CLASS_ORDER]
    mat = confusion.to_numpy(dtype=float)
    im = ax_c.imshow(mat, vmin=0, vmax=1, cmap="Blues")
    ax_c.grid(False)
    ax_c.set_xticks(np.arange(len(CLASS_ORDER)))
    ax_c.set_yticks(np.arange(len(CLASS_ORDER)))
    ax_c.set_xticklabels(CLASS_LABELS, rotation=30, ha="right")
    ax_c.set_yticklabels(CLASS_LABELS)
    ax_c.set_xlabel("Predicted behavior")
    ax_c.set_ylabel("Annotated behavior")
    for row_idx in range(mat.shape[0]):
        for col_idx in range(mat.shape[1]):
            val = mat[row_idx, col_idx]
            color = "white" if val > 0.55 else "#111827"
            ax_c.text(col_idx, row_idx, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)
    style.title_with_panel(ax_c, "C", "Raw-window confusion profile")
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized recall", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    outputs = style.save_figure(fig, args.figure_dir / "dlc_heldout_summary_multiplot")
    plt.close(fig)
    for output in outputs:
        print(output)


if __name__ == "__main__":
    main()
