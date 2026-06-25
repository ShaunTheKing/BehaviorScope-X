"""Plot per-video paired effects for YOLO/SPPF held-out MARS models."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
STYLE_DIR = THIS_DIR.parent / "current_analysis"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import (  # noqa: E402
    HEAD_COLORS,
    NEUTRAL,
    apply_style,
    panel_label,
    save_figure,
)


METRICS = {
    "frame_macro_f1_present_behaviors": "Frame macro-F1",
    "bout_macro_f1_iou25_present_behaviors": "Bout macro-F1 @0.25",
    "bout_macro_f1_iou50_present_behaviors": "Bout macro-F1 @0.50",
}
MINUS = "\N{MINUS SIGN}"


def true_minus(text: str) -> str:
    """Render mathematical negatives with a true minus sign."""
    return text.replace("-", MINUS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table_dir",
        type=Path,
        default=Path("Manuscript_penultimate/tables/production/yolo_sppf_main"),
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=Path("Manuscript_penultimate/figures/production/yolo_sppf_main"),
    )
    parser.add_argument("--prediction_type", default="smoothed_frames")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity", type=int, default=256)
    parser.add_argument("--output_name", default="fig7_yolo_per_video_paired_effects")
    return parser.parse_args()


def model_name(stream: str, head: str, capacity: int, seed: int) -> str:
    return f"yolo_sppf_{stream}_{head}{capacity}_seed{seed}"


def short_video_label(video_id: str) -> str:
    match = re.search(r"(Mouse\d+)", video_id)
    if match:
        return match.group(1)
    return video_id.split("_")[0]


def load_per_video(table_dir: Path, prediction_type: str) -> pd.DataFrame:
    path = table_dir / "yolo_lstm_attention_per_video_long.csv"
    df = pd.read_csv(path)
    df = df[df["prediction_type"].eq(prediction_type)].copy()
    numeric_cols = [
        "n_frames",
        "gt_bouts_total",
        "pred_bouts_total",
        *METRICS.keys(),
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def paired_table(df: pd.DataFrame, first_model: str, second_model: str) -> pd.DataFrame:
    first = df[df["model_name"].eq(first_model)].copy()
    second = df[df["model_name"].eq(second_model)].copy()
    if first.empty:
        raise ValueError(f"Missing first model: {first_model}")
    if second.empty:
        raise ValueError(f"Missing second model: {second_model}")
    merged = first.merge(
        second,
        on=["split", "video_id"],
        suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError(f"No paired videos for {first_model} and {second_model}")
    for metric in METRICS:
        merged[f"delta_{metric}"] = merged[f"{metric}_first"] - merged[f"{metric}_second"]
    first_err = (merged["pred_bouts_total_first"] - merged["gt_bouts_total_first"]).abs()
    second_err = (merged["pred_bouts_total_second"] - merged["gt_bouts_total_second"]).abs()
    merged["bout_count_error_reduction"] = second_err - first_err
    merged["video_label"] = merged["video_id"].map(short_video_label)
    return merged


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return mean, float(lo), float(hi)


def draw_heatmap(ax, paired: pd.DataFrame, first_label: str, second_label: str) -> None:
    rows = [
        ("delta_frame_macro_f1_present_behaviors", "Frame macro-F1", "{:+.2f}"),
        ("delta_bout_macro_f1_iou25_present_behaviors", "Bout macro-F1 @0.25", "{:+.2f}"),
        ("delta_bout_macro_f1_iou50_present_behaviors", "Bout macro-F1 @0.50", "{:+.2f}"),
        ("bout_count_error_reduction", "Bout-count error reduction", "{:+.0f}"),
    ]
    order = paired.sort_values("delta_frame_macro_f1_present_behaviors")["video_id"].tolist()
    data = paired.set_index("video_id").loc[order]
    color_grid: list[list[str]] = []
    text_grid: list[list[str]] = []
    for col, _, fmt in rows:
        vals = data[col].to_numpy(dtype=float)
        colors = []
        texts = []
        for val in vals:
            is_tie = abs(val) < (0.005 if col.startswith("delta_") else 0.5)
            if is_tie:
                colors.append("#E5E7EB")
            elif val > 0:
                colors.append("#047857")
            else:
                colors.append("#C2410C")
            texts.append(true_minus(fmt.format(val)))
        color_grid.append(colors)
        text_grid.append(texts)

    for y, row_colors in enumerate(color_grid):
        for x, color in enumerate(row_colors):
            ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=color, edgecolor="white", linewidth=0.5))
            txt_color = "white" if color != "#E5E7EB" else NEUTRAL["dark"]
            ax.text(x + 0.5, y + 0.5, text_grid[y][x], ha="center", va="center", fontsize=6.7, color=txt_color)

    ax.set_xlim(0, len(order))
    ax.set_ylim(0, len(rows))
    ax.invert_yaxis()
    ax.set_yticks(np.arange(len(rows)) + 0.5)
    ax.set_yticklabels([r[1] for r in rows])
    ax.set_xticks(np.arange(len(order)) + 0.5)
    ax.set_xticklabels(data["video_label"], rotation=90, fontsize=6.5)
    ax.tick_params(length=0)
    ax.set_title(
        f"Direction of paired effects across held-out videos (n={len(order)}): {first_label} {MINUS} {second_label}",
        fontsize=11,
        pad=8,
    )
    ax.grid(False)
    handles = [
        Patch(facecolor="#047857", label=f"{first_label} better"),
        Patch(facecolor="#C2410C", label=f"{second_label} better"),
        Patch(facecolor="#E5E7EB", label="tie"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.005, 0.5), frameon=False, fontsize=8)


def contrast_summary(df: pd.DataFrame, contrasts: list[dict[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260601)
    f1_rows = []
    count_rows = []
    for contrast in contrasts:
        paired = paired_table(df, contrast["first_model"], contrast["second_model"])
        for metric, metric_label in METRICS.items():
            mean, lo, hi = bootstrap_mean_ci(paired[f"delta_{metric}"].to_numpy(), rng)
            f1_rows.append(
                {
                    "comparison": contrast["label"],
                    "metric": metric_label,
                    "mean": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "color": contrast["color"],
                    "n_videos": len(paired),
                    "first_model": contrast["first_model"],
                    "second_model": contrast["second_model"],
                }
            )
        mean, lo, hi = bootstrap_mean_ci(paired["bout_count_error_reduction"].to_numpy(), rng)
        count_rows.append(
            {
                "comparison": contrast["label"],
                "mean": mean,
                "ci_low": lo,
                "ci_high": hi,
                "color": contrast["color"],
                "n_videos": len(paired),
                "first_model": contrast["first_model"],
                "second_model": contrast["second_model"],
            }
        )
    return pd.DataFrame(f1_rows), pd.DataFrame(count_rows)


def draw_forest(ax, summary: pd.DataFrame) -> None:
    metric_order = ["Bout macro-F1 @0.50", "Bout macro-F1 @0.25", "Frame macro-F1"]
    rows = []
    for metric in metric_order:
        sub = summary[summary["metric"].eq(metric)]
        for _, row in sub.iterrows():
            rows.append(row)
    y = np.arange(len(rows))
    for yi, row in zip(y, rows):
        mean = float(row["mean"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])
        color = str(row["color"])
        ax.errorbar(
            mean,
            yi,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            markersize=4.5,
        )
        ax.text(mean, yi - 0.22, true_minus(f"{mean:.2f}"), ha="center", va="bottom", fontsize=7, color=NEUTRAL["dark"])
    labels = [f"{r['metric']}: {r['comparison']}" for r in rows]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.axvline(0, color=NEUTRAL["mid"], linewidth=0.8)
    ax.set_xlabel(f"Mean paired F1 gain (first {MINUS} second; 95% bootstrap CI)")
    ax.set_title("Per-video F1 effects", loc="left", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.0)
    ax.invert_yaxis()


def draw_count_forest(ax, summary: pd.DataFrame) -> None:
    y = np.arange(len(summary))
    for yi, (_, row) in zip(y, summary.iterrows()):
        mean = float(row["mean"])
        lo = float(row["ci_low"])
        hi = float(row["ci_high"])
        color = str(row["color"])
        ax.errorbar(
            mean,
            yi,
            xerr=[[mean - lo], [hi - mean]],
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3,
            markersize=4.5,
        )
        ax.text(mean + 0.18, yi, true_minus(f"{mean:.1f}"), ha="left", va="center", fontsize=7, color=NEUTRAL["dark"])
    ax.set_yticks(y)
    ax.set_yticklabels(summary["comparison"], fontsize=8)
    ax.axvline(0, color=NEUTRAL["mid"], linewidth=0.8)
    ax.set_xlabel("Bout-count error reduction\n(first model closer to ground truth; 95% bootstrap CI)")
    ax.set_title("Fragmentation/count burden", loc="left", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", alpha=0.0)
    ax.invert_yaxis()


def main() -> None:
    args = parse_args()
    table_dir = args.table_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)
    df = load_per_video(table_dir, args.prediction_type)

    full_attention = model_name("full", "attention", args.capacity, args.seed)
    pose_attention = model_name("pose_relations_only", "attention", args.capacity, args.seed)
    visual_attention = model_name("visual_only", "attention", args.capacity, args.seed)
    full_lstm = model_name("full", "lstm", args.capacity, args.seed)

    heatmap_paired = paired_table(df, full_attention, pose_attention)
    heatmap_paired = heatmap_paired.sort_values("delta_frame_macro_f1_present_behaviors")
    heatmap_csv = table_dir / f"{args.output_name}_full_vs_pose_relations_per_video.csv"
    heatmap_paired.to_csv(heatmap_csv, index=False)

    contrasts = [
        {
            "label": f"Full attention {MINUS} pose+relations",
            "first_model": full_attention,
            "second_model": pose_attention,
            "color": HEAD_COLORS["attention"],
        },
        {
            "label": f"Full attention {MINUS} visual only",
            "first_model": full_attention,
            "second_model": visual_attention,
            "color": "#059669",
        },
        {
            "label": f"Full attention {MINUS} full LSTM",
            "first_model": full_attention,
            "second_model": full_lstm,
            "color": HEAD_COLORS["lstm"],
        },
    ]
    f1_summary, count_summary = contrast_summary(df, contrasts)
    f1_csv = table_dir / f"{args.output_name}_f1_bootstrap_summary.csv"
    count_csv = table_dir / f"{args.output_name}_bout_count_bootstrap_summary.csv"
    f1_summary.to_csv(f1_csv, index=False)
    count_summary.to_csv(count_csv, index=False)

    apply_style()
    fig = plt.figure(figsize=(13.5, 7.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], width_ratios=[1.8, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    draw_heatmap(ax_a, heatmap_paired, "full attention-256", "pose+relations attention-256")
    draw_forest(ax_b, f1_summary)
    draw_count_forest(ax_c, count_summary)
    panel_label(ax_a, "A", x=-0.03, y=1.15)
    panel_label(ax_b, "B", x=-0.12, y=1.10)
    panel_label(ax_c, "C", x=-0.28, y=1.10)
    outputs = save_figure(fig, figure_dir / args.output_name)
    plt.close(fig)
    print(f"Wrote {outputs[0]}")
    print(f"Wrote {outputs[1]}")
    print(f"Wrote {heatmap_csv}")
    print(f"Wrote {f1_csv}")
    print(f"Wrote {count_csv}")


if __name__ == "__main__":
    main()
