import argparse
from pathlib import Path
import importlib.util

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLES = PACKAGE_ROOT / "tables" / "production" / "descriptor_provenance"
DEFAULT_OUT = PACKAGE_ROOT / "figures_reproduced" / "production" / "descriptor_provenance"
DEFAULT_STYLE_PATH = (
    PACKAGE_ROOT
    / "scripts_by_experiment"
    / "00_figure_table_rendering"
    / "current_analysis"
    / "behaviorscope_figure_style.py"
)

ROUTES = ["YOLO-SPPF", "MobileNetV3", "DLC-HRNet"]
COLORS = {
    "YOLO-SPPF": "#1f77b4",
    "MobileNetV3": "#d95f02",
    "DLC-HRNet": "#009E73",
}
LIGHT = {
    "YOLO-SPPF": "#9ecae1",
    "MobileNetV3": "#f2bf8f",
    "DLC-HRNet": "#8fd4b8",
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


def main():
    args = parse_args()
    style = load_style(args.style_path)
    style.apply_style()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(args.table_dir / "visual_descriptor_summary.csv")
    comp = pd.read_csv(args.table_dir / "visual_component_probe_summary.csv")
    cka = pd.read_csv(args.table_dir / "visual_descriptor_linear_cka.csv")

    fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.2))

    # A. Behavior separability and visual-only probe.
    ax = axs[0, 0]
    sub = summary.set_index("route").loc[ROUTES].reset_index()
    x = np.arange(len(sub))
    ax.bar(
        x - 0.18,
        sub["fisher_ratio_balanced"],
        width=0.36,
        color=[COLORS[r] for r in sub["route"]],
        alpha=0.9,
        label="Fisher ratio",
    )
    ax2 = ax.twinx()
    ax2.bar(
        x + 0.18,
        sub["probe_macro_f1_mean"],
        yerr=sub["probe_macro_f1_sd"],
        capsize=3,
        width=0.36,
        color=[LIGHT[r] for r in sub["route"]],
        edgecolor=[COLORS[r] for r in sub["route"]],
        ecolor="#374151",
        linewidth=1.0,
        label="Linear probe macro-F1",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(sub["route"], rotation=20, ha="right")
    ax.set_ylabel("Behavior Fisher ratio")
    ax2.set_ylabel("Visual-only macro-F1")
    style.title_with_panel(ax, "A", "Behavior structure in matched visual descriptors")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)
    ax2.tick_params(labelsize=9)

    # B. Group versus animal descriptor behavior probe.
    ax = axs[0, 1]
    components = ["group", "animal_pair", "combined"]
    labels = ["Group", "Animal pair", "Combined"]
    width = 0.23
    for i, route in enumerate(ROUTES):
        vals = [
            comp[(comp["route"] == route) & (comp["component"] == c)]["behavior_probe_macro_f1"].iloc[0]
            for c in components
        ]
        errs = [
            comp[(comp["route"] == route) & (comp["component"] == c)]["behavior_probe_fold_macro_f1_sd"].iloc[0]
            for c in components
        ]
        ax.bar(
            np.arange(len(components)) + (i - 1) * width,
            vals,
            yerr=errs,
            capsize=2.5,
            width=width,
            color=COLORS[route],
            alpha=0.9,
            ecolor="#374151",
            linewidth=0.8,
            label=route,
        )
    ax.set_xticks(np.arange(len(components)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Visual-only macro-F1")
    ax.set_ylim(0.45, 0.95)
    style.title_with_panel(ax, "B", "Component-level behavior accessibility")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))

    # C. Behavior versus video identity for combined descriptors.
    ax = axs[1, 0]
    combined = comp[comp["component"] == "combined"].set_index("route").loc[ROUTES].reset_index()
    x = np.arange(len(combined))
    ax.bar(
        x - 0.18,
        combined["behavior_probe_macro_f1"],
        yerr=combined["behavior_probe_fold_macro_f1_sd"],
        capsize=3,
        width=0.36,
        color=[COLORS[r] for r in combined["route"]],
        ecolor="#374151",
        linewidth=0.8,
        label="Behavior macro-F1",
    )
    ax.bar(
        x + 0.18,
        combined["video_probe_balanced_accuracy"],
        width=0.36,
        color=[LIGHT[r] for r in combined["route"]],
        edgecolor=[COLORS[r] for r in combined["route"]],
        label="Video identity balanced accuracy",
    )
    ax.axhline(combined["video_probe_chance"].iloc[0], color="#666666", linestyle="--", linewidth=1.0, label="Video chance")
    ax.set_xticks(x)
    ax.set_xticklabels(combined["route"], rotation=20, ha="right")
    ax.set_ylabel("Probe score")
    ax.set_ylim(0, 0.92)
    style.title_with_panel(ax, "C", "Behavior and video-identity readout")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)

    # D. Linear CKA heatmap.
    ax = axs[1, 1]
    mat = cka.pivot(index="route_1", columns="route_2", values="linear_cka_balanced").loc[ROUTES, ROUTES]
    plot_mat = mat.values.astype(float).copy()
    np.fill_diagonal(plot_mat, np.nan)
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#f4f4f4")
    im = ax.imshow(plot_mat, vmin=0, vmax=0.8, cmap=cmap)
    ax.set_xticks(np.arange(len(ROUTES)))
    ax.set_yticks(np.arange(len(ROUTES)))
    ax.set_xticklabels(ROUTES, rotation=25, ha="right")
    ax.set_yticklabels(ROUTES)
    for i in range(len(ROUTES)):
        for j in range(len(ROUTES)):
            val = plot_mat[i, j]
            label = "-" if np.isnan(val) else f"{val:.2f}"
            ax.text(j, i, label, ha="center", va="center", color="white" if np.isfinite(val) and val > 0.55 else "#333333", fontsize=10)
    ax.grid(False)
    style.title_with_panel(ax, "D", "Cross-route descriptor geometry")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(args.figure_dir / "fig_descriptor_provenance_multiplot.png", dpi=300)
    fig.savefig(args.figure_dir / "fig_descriptor_provenance_multiplot.pdf")
    print(args.figure_dir / "fig_descriptor_provenance_multiplot.png")
    print(args.figure_dir / "fig_descriptor_provenance_multiplot.pdf")


if __name__ == "__main__":
    main()
