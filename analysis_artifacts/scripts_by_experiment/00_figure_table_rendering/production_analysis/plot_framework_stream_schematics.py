"""Create production schematics for pose reuse and stream ablation design."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_DIR = SCRIPT_DIR.parent / "current_analysis"
sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import (  # noqa: E402
    HEAD_COLORS,
    NEUTRAL,
    STREAM_LABELS,
    apply_style,
    panel_label,
    save_figure,
)


STREAMS = [
    ("pose_self", "Self-pose\ngeometry", "#EFF6FF", "#2563EB"),
    ("relations", "Pairwise\nrelations", "#F5F3FF", "#7C3AED"),
    ("group_visual", "Group-scene\nvisual", "#FEF3C7", "#D97706"),
    ("animal_visual", "Per-animal\nvisual", "#FCE7F3", "#DB2777"),
]

ABLATIONS = [
    {
        "condition": "Full multimodal reuse",
        "stream_key": "full",
        "pose_self": True,
        "relations": True,
        "group_visual": True,
        "animal_visual": True,
        "rationale": "All cached coordinate, relational, and visual evidence.",
    },
    {
        "condition": "Animal visual + pose geometry, no group",
        "stream_key": "no_group_visual",
        "pose_self": True,
        "relations": True,
        "group_visual": False,
        "animal_visual": True,
        "rationale": "Tests whether scene-level context is necessary.",
    },
    {
        "condition": "Pose geometry only",
        "stream_key": "pose_relations_only",
        "pose_self": True,
        "relations": True,
        "group_visual": False,
        "animal_visual": False,
        "rationale": "Tests keypoints plus pairwise social geometry without visual descriptors.",
    },
    {
        "condition": "Visual + self-pose, no pairwise geometry",
        "stream_key": "no_relations",
        "pose_self": True,
        "relations": False,
        "group_visual": True,
        "animal_visual": True,
        "rationale": "Tests whether pairwise geometry adds signal beyond visual and self-pose streams.",
    },
    {
        "condition": "Per-animal visual only",
        "stream_key": "animal_visual_only",
        "pose_self": False,
        "relations": False,
        "group_visual": False,
        "animal_visual": True,
        "rationale": "Tests animal-centered visual descriptors without explicit geometry.",
    },
    {
        "condition": "Visual descriptors only",
        "stream_key": "visual_only",
        "pose_self": False,
        "relations": False,
        "group_visual": True,
        "animal_visual": True,
        "rationale": "Tests reused visual descriptors without explicit pose geometry.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=Path("Manuscript_penultimate/figures/production/yolo_sppf_main"),
    )
    parser.add_argument(
        "--table_dir",
        type=Path,
        default=Path("Manuscript_penultimate/tables/production/yolo_sppf_main"),
    )
    return parser.parse_args()


def setup_axis(ax, xlim=(0, 1), ylim=(0, 1)) -> None:
    ax.set_axis_off()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)


def box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    face: str,
    edge: str,
    *,
    fontsize: float = 9.0,
    weight: str = "normal",
    radius: float = 0.035,
    lw: float = 1.35,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color="#111827",
        linespacing=1.08,
    )
    return patch


def arrow(ax, start, end, *, color="#374151", rad=0.0, lw=1.3, scale=12) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=3,
            shrinkB=3,
        )
    )


def render_framework(args: argparse.Namespace) -> None:
    apply_style()
    fig = plt.figure(figsize=(13.6, 7.3), constrained_layout=False)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.08, 0.78], hspace=0.22, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    # Panel A: full pipeline.
    setup_axis(ax_a, xlim=(0, 1), ylim=(0, 1))
    panel_label(ax_a, "A", x=0.01, y=0.98, fontsize=16)
    box(ax_a, 0.035, 0.42, 0.12, 0.18, "Full-video\nframes", "#F9FAFB", "#9CA3AF", fontsize=9.8, weight="bold")
    box(ax_a, 0.19, 0.42, 0.13, 0.18, "Sliding\nwindows", "#EEF2FF", "#4F46E5", fontsize=9.8, weight="bold")
    box(ax_a, 0.37, 0.22, 0.22, 0.58, "", "#ECFDF5", "#059669", radius=0.025)
    ax_a.text(
        0.48,
        0.73,
        "Frozen pose-trained\ncheckpoint",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        linespacing=1.05,
        color="#111827",
    )
    box(ax_a, 0.405, 0.53, 0.15, 0.13, "Pose head\noutputs", "#DBEAFE", "#2563EB", fontsize=9.0, weight="bold")
    box(ax_a, 0.405, 0.34, 0.15, 0.13, "Backbone\nfeature maps", "#FEF3C7", "#D97706", fontsize=8.8, weight="bold")

    stream_boxes = {
        "pose_self": (0.65, 0.74, 0.18, 0.105, "Pose features\nkeypoints + confidence", "#EFF6FF", "#2563EB"),
        "relations": (0.65, 0.57, 0.18, 0.105, "Relationship features\ndistance + overlap", "#F5F3FF", "#7C3AED"),
        "group_visual": (0.65, 0.40, 0.18, 0.105, "Group visual features\nscene embedding", "#FEF3C7", "#D97706"),
        "animal_visual": (0.65, 0.23, 0.18, 0.105, "Animal visual features\ncrop embeddings", "#FCE7F3", "#DB2777"),
    }
    for _, (x, y, w, h, label, fc, ec) in stream_boxes.items():
        box(ax_a, x, y, w, h, label, fc, ec, fontsize=8.3)

    box(ax_a, 0.88, 0.54, 0.10, 0.13, "Stream\nfusion", "#F3F4F6", "#6B7280", fontsize=9.0, weight="bold")
    box(
        ax_a,
        0.88,
        0.34,
        0.10,
        0.13,
        "Temporal\nhead",
        "#DBEAFE",
        HEAD_COLORS["lstm"],
        fontsize=8.9,
        weight="bold",
    )
    box(ax_a, 0.88, 0.14, 0.10, 0.12, "Frame scores\nand bouts", "#F9FAFB", "#9CA3AF", fontsize=8.5, weight="bold")

    arrow(ax_a, (0.155, 0.51), (0.19, 0.51))
    arrow(ax_a, (0.32, 0.51), (0.37, 0.51))
    arrow(ax_a, (0.555, 0.595), (0.65, 0.795), color="#2563EB", rad=0.16)
    arrow(ax_a, (0.555, 0.585), (0.65, 0.625), color="#7C3AED", rad=0.04)
    arrow(ax_a, (0.555, 0.405), (0.65, 0.455), color="#D97706", rad=-0.04)
    arrow(ax_a, (0.555, 0.392), (0.65, 0.285), color="#DB2777", rad=-0.16)
    for y, rad in [(0.795, 0.14), (0.625, 0.04), (0.455, -0.04), (0.285, -0.14)]:
        arrow(ax_a, (0.83, y), (0.88, 0.605), color="#4B5563", rad=rad, lw=1.2)
    arrow(ax_a, (0.93, 0.54), (0.93, 0.47), color="#4B5563")
    arrow(ax_a, (0.93, 0.34), (0.93, 0.26), color="#4B5563")

    # Panel B: amortized cache.
    setup_axis(ax_b)
    panel_label(ax_b, "B", x=0.00, y=0.99, fontsize=15)
    ax_b.text(0.16, 0.95, "Pose checkpoint\nrun once", fontsize=10.2, ha="left", va="top", linespacing=1.05)
    box(ax_b, 0.05, 0.57, 0.26, 0.16, "Videos\nfull recordings", "#E0F2FE", "#334155", fontsize=7.6)
    box(ax_b, 0.39, 0.57, 0.30, 0.16, "Assay-trained\npose checkpoint", "#DCFCE7", "#334155", fontsize=7.3)
    box(ax_b, 0.76, 0.69, 0.22, 0.13, "Pose tracks\n+ confidence", "#FEF9C3", "#334155", fontsize=6.9)
    box(ax_b, 0.76, 0.43, 0.22, 0.13, "Frozen visual\ndescriptors", "#EDE9FE", "#334155", fontsize=6.9)
    arrow(ax_b, (0.31, 0.65), (0.39, 0.65))
    arrow(ax_b, (0.69, 0.67), (0.76, 0.75))
    arrow(ax_b, (0.69, 0.61), (0.76, 0.50))

    # Panel C: reusable evidence.
    setup_axis(ax_c)
    panel_label(ax_c, "C", x=0.00, y=0.99, fontsize=15)
    ax_c.text(0.16, 0.95, "Coordinate + visual\nevidence reused", fontsize=10.2, ha="left", va="top", linespacing=1.05)
    arrow(ax_c, (0.33, 0.72), (0.43, 0.64))
    arrow(ax_c, (0.33, 0.50), (0.43, 0.58))
    arrow(ax_c, (0.73, 0.62), (0.81, 0.62))
    box(ax_c, 0.05, 0.65, 0.28, 0.14, "Pose stream\ngeometry", "#FEF9C3", "#334155", fontsize=7.3)
    box(ax_c, 0.05, 0.43, 0.28, 0.14, "Visual stream\nfeatures", "#EDE9FE", "#334155", fontsize=7.3)
    box(ax_c, 0.43, 0.54, 0.30, 0.16, "Fused frame\nevidence", "#E0F2FE", "#334155", fontsize=7.0)
    box(ax_c, 0.81, 0.54, 0.17, 0.16, "Temporal\nhead", "#FEE2E2", "#334155", fontsize=7.3)

    # Panel D: ethological evaluation.
    setup_axis(ax_d)
    panel_label(ax_d, "D", x=0.00, y=0.99, fontsize=15)
    ax_d.text(0.16, 0.95, "Evaluation preserves\nethological units", fontsize=10.2, ha="left", va="top", linespacing=1.05)
    box(ax_d, 0.06, 0.60, 0.38, 0.14, "Frame metrics\nmacro-F1", "#E0F2FE", "#334155", fontsize=7.2)
    box(ax_d, 0.56, 0.60, 0.38, 0.14, "Bout metrics\nTIoU 0.25 / 0.50", "#DCFCE7", "#334155", fontsize=7.1)
    box(ax_d, 0.06, 0.34, 0.38, 0.14, "Boundary error\nstart/end offsets", "#FEF9C3", "#334155", fontsize=7.0)
    box(ax_d, 0.56, 0.34, 0.38, 0.14, "Class-specific\nfailure modes", "#FEE2E2", "#334155", fontsize=7.0)

    save_figure(fig, args.figure_dir / "schematic_pose_reuse_framework")
    plt.close(fig)


def render_ablation_design(args: argparse.Namespace) -> None:
    apply_style()
    fig, ax = plt.subplots(figsize=(10.8, 6.5), constrained_layout=False)
    setup_axis(ax, xlim=(0, 1), ylim=(0, 1))

    panel_label(ax, "A", x=0.02, y=0.97, fontsize=16)
    ax.text(
        0.06,
        0.935,
        "Stream-ablation design",
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="top",
        color="#111827",
    )

    # Compact legend in the open title band.
    legend_x = 0.735
    legend_y = 0.925
    ax.add_patch(Circle((legend_x, legend_y), 0.011, facecolor="#DBEAFE", edgecolor="#2563EB", linewidth=1.0))
    ax.text(legend_x, legend_y - 0.001, "✓", fontsize=8.5, fontweight="bold", ha="center", va="center", color="#2563EB")
    ax.text(legend_x + 0.018, legend_y, "included", fontsize=7.8, va="center", ha="left", color="#111827")
    ax.add_patch(Circle((legend_x + 0.095, legend_y), 0.011, facecolor="#F3F4F6", edgecolor="#CBD5E1", linewidth=1.0))
    ax.text(legend_x + 0.113, legend_y, "excluded", fontsize=7.8, va="center", ha="left", color="#111827")

    left = 0.07
    top = 0.735
    row_h = 0.080
    name_w = 0.37
    col_w = 0.105
    gap = 0.022

    ax.text(left, top + 0.064, "Ablation condition", fontsize=9.3, fontweight="bold", ha="left", va="bottom")
    for i, (_, label, face, edge) in enumerate(STREAMS):
        x = left + name_w + i * (col_w + gap)
        box(ax, x, top + 0.025, col_w, 0.052, label, face, edge, fontsize=6.2, weight="bold", radius=0.014, lw=1.1)

    rows = []
    for r, item in enumerate(ABLATIONS):
        y = top - (r + 1) * row_h
        bg = "#F9FAFB" if r % 2 == 0 else "#FFFFFF"
        ax.add_patch(
            FancyBboxPatch(
                (left - 0.01, y - 0.006),
                0.88,
                row_h * 0.86,
                boxstyle="round,pad=0.004,rounding_size=0.008",
                facecolor=bg,
                edgecolor="#E5E7EB",
                linewidth=0.7,
            )
        )
        ax.text(left, y + 0.032, item["condition"], fontsize=7.8, ha="left", va="center", color="#111827")
        for i, (key, _label, face, edge) in enumerate(STREAMS):
            x = left + name_w + i * (col_w + gap)
            included = bool(item[key])
            cx = x + col_w / 2
            cy = y + 0.032
            if included:
                ax.add_patch(Circle((cx, cy), 0.020, facecolor=face, edgecolor=edge, linewidth=1.2))
                ax.text(cx, cy - 0.001, "✓", fontsize=13, fontweight="bold", ha="center", va="center", color=edge)
            else:
                ax.add_patch(Circle((cx, cy), 0.020, facecolor="#F3F4F6", edgecolor="#CBD5E1", linewidth=1.0))
        row = {"condition": item["condition"], "stream_key": item["stream_key"], "manuscript_label": STREAM_LABELS[item["stream_key"]]}
        for key, *_rest in STREAMS:
            row[key] = "included" if item[key] else "excluded"
        row["rationale"] = item["rationale"]
        rows.append(row)

    # Compact model-control strip.
    strip_y = 0.055
    ax.text(left, strip_y + 0.105, "Controls held constant across rows", fontsize=10.0, fontweight="bold", ha="left")
    controls = [
        ("Frozen cache", "#ECFDF5", "#059669"),
        ("32-frame windows", "#EEF2FF", "#4F46E5"),
        ("Train/val/test split", "#F9FAFB", "#6B7280"),
        ("Matched seeds", "#E0F2FE", "#2563EB"),
        ("Same decoder family\nwithin comparison", "#FEE2E2", "#DC2626"),
    ]
    x = left
    for label, face, edge in controls:
        box(ax, x, strip_y, 0.150, 0.068, label, face, edge, fontsize=6.7, weight="bold", radius=0.018, lw=1.1)
        x += 0.172

    pd.DataFrame(rows).to_csv(args.table_dir / "schematic_stream_ablation_design.csv", index=False)
    save_figure(fig, args.figure_dir / "schematic_stream_ablation_design")
    plt.close(fig)


def write_framework_metadata(args: argparse.Namespace) -> None:
    rows = [
        {
            "figure": "schematic_pose_reuse_framework",
            "purpose": "Conceptual pipeline for amortized pose-model reuse.",
            "notes": "Schematic only; all quantitative stream/capacity results are reported in separate result figures.",
        },
        {
            "figure": "schematic_stream_ablation_design",
            "purpose": "Visual definition of the six stream-ablation conditions.",
            "notes": "All rows keep the frozen cache, splits, temporal windows, and seeds fixed while changing stream access.",
        },
    ]
    pd.DataFrame(rows).to_csv(args.table_dir / "schematic_framework_stream_metadata.csv", index=False)


def main() -> None:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    render_framework(args)
    render_ablation_design(args)
    write_framework_metadata(args)
    print(f"Wrote {args.figure_dir / 'schematic_pose_reuse_framework.png'}")
    print(f"Wrote {args.figure_dir / 'schematic_stream_ablation_design.png'}")


if __name__ == "__main__":
    main()
