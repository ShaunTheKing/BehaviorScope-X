"""Plot the full YOLO/SPPF LSTM and attention ablation matrix.

This supplemental figure is intentionally denser than the main manuscript
panels. It records every completed stream, head, capacity, and seed so the
headline stream-ablation and temporal-head figures can remain readable.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_DIR = SCRIPT_DIR.parent / "current_analysis"
sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import (  # noqa: E402
    HEAD_COLORS,
    STREAM_SHORT_LABELS,
    apply_style,
    save_figure,
)


STREAM_ORDER = [
    "full",
    "no_group_visual",
    "pose_relations_only",
    "no_relations",
    "animal_visual_only",
    "visual_only",
]
CAPACITY_ORDER = [256, 512, 896]
HEAD_ORDER = ["lstm", "attention"]
HEAD_LABELS = {"lstm": "LSTM", "attention": "Attention"}
MINUS = "\N{MINUS SIGN}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table_root",
        type=Path,
        default=Path("Manuscript_penultimate/tables/production/yolo_sppf_main"),
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=Path("Manuscript_penultimate/figures/production/supplement"),
    )
    parser.add_argument(
        "--table_dir",
        type=Path,
        default=Path("Manuscript_penultimate/tables/production/supplement"),
    )
    parser.add_argument("--prediction_type", default="smoothed_frames")
    parser.add_argument("--output_name", default="supp_yolo_full_ablation_matrix")
    return parser.parse_args()


def clean_summary(df: pd.DataFrame, prediction_type: str) -> pd.DataFrame:
    out = df[df["prediction_type"].eq(prediction_type)].copy()
    out["stream"] = pd.Categorical(out["stream"], STREAM_ORDER, ordered=True)
    out["head"] = pd.Categorical(out["head"], HEAD_ORDER, ordered=True)
    out["capacity"] = pd.Categorical(out["capacity"], CAPACITY_ORDER, ordered=True)
    return out.sort_values(["head", "stream", "capacity"])


def aggregate_validation(model_summary: pd.DataFrame, prediction_type: str) -> pd.DataFrame:
    df = model_summary[model_summary["prediction_type"].eq(prediction_type)].copy()
    grouped = (
        df.groupby(["stream", "head", "capacity"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            validation_macro_f1_mean=("best_validation_macroF1", "mean"),
            validation_macro_f1_sd=("best_validation_macroF1", "std"),
        )
        .sort_values(["head", "stream", "capacity"])
    )
    grouped["stream"] = pd.Categorical(grouped["stream"], STREAM_ORDER, ordered=True)
    grouped["head"] = pd.Categorical(grouped["head"], HEAD_ORDER, ordered=True)
    grouped["capacity"] = pd.Categorical(grouped["capacity"], CAPACITY_ORDER, ordered=True)
    return grouped.sort_values(["head", "stream", "capacity"])


def draw_heatmap(
    ax,
    data: pd.DataFrame,
    *,
    head: str,
    value_col: str,
    sd_col: str,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> None:
    sub = data[data["head"].eq(head)].copy()
    mat = (
        sub.pivot_table(index="stream", columns="capacity", values=value_col, observed=False)
        .reindex(index=STREAM_ORDER, columns=CAPACITY_ORDER)
        .to_numpy(dtype=float)
    )
    sd = (
        sub.pivot_table(index="stream", columns="capacity", values=sd_col, observed=False)
        .reindex(index=STREAM_ORDER, columns=CAPACITY_ORDER)
        .to_numpy(dtype=float)
    )
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xticks(np.arange(len(CAPACITY_ORDER)), labels=[str(c) for c in CAPACITY_ORDER])
    ax.set_yticks(
        np.arange(len(STREAM_ORDER)),
        labels=[STREAM_SHORT_LABELS[s].replace("\n", " ") for s in STREAM_ORDER],
    )
    ax.tick_params(axis="x", labelrotation=0)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                text = "NA"
            else:
                text = f"{val:.3f}\n±{sd[i, j]:.3f}"
            color = "white" if not np.isnan(val) and val > (vmin + vmax) / 2 else "#111827"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.3, color=color)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color("#D1D5DB")
    return im


def seed_scatter(ax, model_summary: pd.DataFrame, prediction_type: str) -> None:
    df = model_summary[model_summary["prediction_type"].eq(prediction_type)].copy()
    markers = {256: "o", 512: "s", 896: "^"}
    for head in HEAD_ORDER:
        sub = df[df["head"].eq(head)]
        for capacity, cap_sub in sub.groupby("capacity"):
            ax.scatter(
                cap_sub["best_validation_macroF1"],
                cap_sub["mean_frame_macro_f1_present_behaviors"],
                s=28,
                marker=markers[int(capacity)],
                color=HEAD_COLORS[head],
                alpha=0.55,
                edgecolor="white",
                linewidth=0.35,
                label=f"{HEAD_LABELS[head]}-{capacity}",
            )
    ax.set_xlabel("Validation macro-F1")
    ax.set_ylabel("Held-out frame macro-F1")
    ax.set_title("Seed-level validation-to-held-out relationship", fontsize=10, pad=6)
    ax.axline((0.65, 0.65), slope=1, color="#6B7280", linestyle="--", linewidth=1)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(
        unique.values(),
        unique.keys(),
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.56),
        frameon=False,
        fontsize=8,
    )


def write_seed_table(model_summary: pd.DataFrame, table_dir: Path, output_name: str, prediction_type: str) -> None:
    cols = [
        "model_name",
        "stream",
        "head",
        "capacity",
        "seed",
        "prediction_type",
        "best_validation_macroF1",
        "mean_frame_macro_f1_present_behaviors",
        "mean_bout_macro_f1_iou25_present_behaviors",
        "mean_bout_macro_f1_iou50_present_behaviors",
        "pred_bouts_total",
        "cpu_pct",
        "ram_gb_used",
        "gpu_util_pct",
        "vram_gb_used",
        "gpu_power_w",
    ]
    available = [c for c in cols if c in model_summary.columns]
    seed_rows = model_summary[model_summary["prediction_type"].eq(prediction_type)][available].copy()
    seed_rows.to_csv(table_dir / f"{output_name}_seed_level.csv", index=False)


def main() -> None:
    args = parse_args()
    apply_style()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    seed_summary = pd.read_csv(args.table_root / "yolo_lstm_attention_seed_summary.csv")
    model_summary = pd.read_csv(args.table_root / "yolo_lstm_attention_model_summary.csv")

    held = clean_summary(seed_summary, args.prediction_type)
    val = aggregate_validation(model_summary, args.prediction_type)

    held.to_csv(args.table_dir / f"{args.output_name}_mean_sd.csv", index=False)
    val.to_csv(args.table_dir / f"{args.output_name}_validation_mean_sd.csv", index=False)
    write_seed_table(model_summary, args.table_dir, args.output_name, args.prediction_type)

    frame_col = "mean_frame_macro_f1_present_behaviors_seed_mean"
    frame_sd = "mean_frame_macro_f1_present_behaviors_seed_sd"
    bout_col = "mean_bout_macro_f1_iou25_present_behaviors_seed_mean"
    bout_sd = "mean_bout_macro_f1_iou25_present_behaviors_seed_sd"

    fig = plt.figure(figsize=(13.2, 11.2), constrained_layout=False)
    gs = fig.add_gridspec(4, 2, height_ratios=[1.08, 1.08, 1.08, 1.0], hspace=0.42, wspace=0.34)

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[2, 0]),
        fig.add_subplot(gs[2, 1]),
    ]

    draw_heatmap(
        axes[0],
        held,
        head="lstm",
        value_col=frame_col,
        sd_col=frame_sd,
        title="A. Held-out frame macro-F1: LSTM",
        cmap="Blues",
        vmin=0.62,
        vmax=0.77,
    )
    draw_heatmap(
        axes[1],
        held,
        head="attention",
        value_col=frame_col,
        sd_col=frame_sd,
        title="B. Held-out frame macro-F1: attention",
        cmap="Greens",
        vmin=0.62,
        vmax=0.77,
    )
    draw_heatmap(
        axes[2],
        held,
        head="lstm",
        value_col=bout_col,
        sd_col=bout_sd,
        title="C. Held-out bout macro-F1 at TIoU 0.25: LSTM",
        cmap="Blues",
        vmin=0.54,
        vmax=0.70,
    )
    draw_heatmap(
        axes[3],
        held,
        head="attention",
        value_col=bout_col,
        sd_col=bout_sd,
        title="D. Held-out bout macro-F1 at TIoU 0.25: attention",
        cmap="Greens",
        vmin=0.54,
        vmax=0.70,
    )
    draw_heatmap(
        axes[4],
        val,
        head="lstm",
        value_col="validation_macro_f1_mean",
        sd_col="validation_macro_f1_sd",
        title="E. Validation macro-F1: LSTM",
        cmap="Blues",
        vmin=0.62,
        vmax=0.82,
    )
    draw_heatmap(
        axes[5],
        val,
        head="attention",
        value_col="validation_macro_f1_mean",
        sd_col="validation_macro_f1_sd",
        title="F. Validation macro-F1: attention",
        cmap="Greens",
        vmin=0.62,
        vmax=0.82,
    )

    ax_seed = fig.add_subplot(gs[3, :])
    seed_scatter(ax_seed, model_summary, args.prediction_type)
    ax_seed.set_title(
        f"G. Individual seeds for all stream/head/capacity configurations "
        f"(each point = one model seed; diagonal is validation {MINUS} held-out parity)",
        fontsize=10,
        pad=8,
    )

    fig.suptitle(
        "Complete YOLO/SPPF neural ablation matrix across streams, heads, capacities, and seeds",
        fontsize=14,
        y=0.985,
    )
    fig.subplots_adjust(top=0.945, bottom=0.14)
    outputs = save_figure(fig, args.figure_dir / args.output_name)
    plt.close(fig)
    print("Saved:")
    for out in outputs:
        print(out)
    print(f"Tables written to {args.table_dir}")


if __name__ == "__main__":
    main()
