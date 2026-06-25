"""Supplemental YOLO/SPPF neural training throughput/resource figure."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
STYLE_DIR = THIS_DIR.parent / "current_analysis"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import HEAD_COLORS, apply_style, panel_label, save_figure  # noqa: E402

HEAD_LABELS = {"lstm": "LSTM", "attention": "Attention"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("outputs/controlled_comparison_runs/run_20260528_205727/manuscript_figures/interim_qc/throughput"),
    )
    parser.add_argument("--figure_dir", type=Path, default=Path("Manuscript_penultimate/figures/production/supplement"))
    parser.add_argument("--table_dir", type=Path, default=Path("Manuscript_penultimate/tables/production/supplement"))
    parser.add_argument("--output_name", default="supp_yolo_training_throughput")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.source_dir / "neural_training_throughput_by_head_backend_capacity.csv")
    per_run = pd.read_csv(args.source_dir / "neural_training_throughput_per_run.csv")
    summary = summary[summary["backend"].eq("yolo_sppf")].copy()
    per_run = per_run[per_run["backend"].eq("yolo_sppf")].copy()
    summary.to_csv(args.table_dir / f"{args.output_name}_summary.csv", index=False)
    per_run.to_csv(args.table_dir / f"{args.output_name}_per_run.csv", index=False)

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), constrained_layout=True)
    ax_a, ax_b, ax_c = axes
    capacities = [256, 512, 896]
    x = np.arange(len(capacities))
    for offset, head in [(-0.12, "lstm"), (0.12, "attention")]:
        sub = summary[summary["head"].eq(head)].set_index("hidden_dim").loc[capacities].reset_index()
        ax_a.errorbar(
            x + offset,
            sub["mean_train_val_windows_per_s_command"],
            yerr=sub["sd_train_val_windows_per_s_command"].fillna(0),
            fmt="o",
            capsize=3,
            color=HEAD_COLORS[head],
            label=HEAD_LABELS[head],
        )
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([str(c) for c in capacities])
    ax_a.set_xlabel("Hidden dimension")
    ax_a.set_ylabel("Train+validation windows/s")
    ax_a.set_title("Training throughput by head/capacity")
    ax_a.legend(frameon=False)

    for head in ["lstm", "attention"]:
        sub = per_run[per_run["head"].eq(head)]
        ax_b.scatter(
            sub["train_val_windows_per_s_command"],
            sub["max_gpu_util_pct"],
            s=28,
            color=HEAD_COLORS[head],
            alpha=0.75,
            label=HEAD_LABELS[head],
        )
    ax_b.set_xlabel("Train+validation windows/s")
    ax_b.set_ylabel("Max GPU utilization (%)")
    ax_b.set_title("GPU utilization during training")
    ax_b.legend(frameon=False)

    for head in ["lstm", "attention"]:
        sub = per_run[per_run["head"].eq(head)]
        ax_c.scatter(
            sub["max_gpu_mem_used_mb"] / 1024.0,
            sub["max_ram_used_gb"],
            s=28,
            color=HEAD_COLORS[head],
            alpha=0.75,
            label=HEAD_LABELS[head],
        )
    ax_c.set_xlabel("Max VRAM used (GB)")
    ax_c.set_ylabel("Max system RAM used (GB)")
    ax_c.set_title("Memory footprint during training")
    ax_c.legend(frameon=False)

    for label, ax in zip(["A", "B", "C"], axes):
        panel_label(ax, label, x=-0.18, y=1.12)
    fig.suptitle("YOLO/SPPF neural training throughput and resource telemetry (training only)", fontsize=12)
    save_figure(fig, args.figure_dir / args.output_name)
    plt.close(fig)
    print(f"Wrote {args.figure_dir / (args.output_name + '.png')}")


if __name__ == "__main__":
    main()
