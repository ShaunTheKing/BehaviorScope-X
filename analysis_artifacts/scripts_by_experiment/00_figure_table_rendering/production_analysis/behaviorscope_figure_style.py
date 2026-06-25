"""Shared figure style for the BehaviorScope-Y penultimate manuscript.

This module codifies the visual conventions used by the legacy manuscript
figures so newly staged analysis scripts keep the same visual language.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


# Core manuscript plotting style observed in the legacy scripts.
BASE_RCPARAMS = {
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


# Behavior colors use the same Okabe-Ito-like mapping used in the qualitative
# bout examples and older learning-curve figures.
BEHAVIOR_COLORS = {
    "attack": "#D55E00",
    "investigation": "#0072B2",
    "mount": "#009E73",
    "other": "#777777",
}

BEHAVIOR_FILL_COLORS = {
    "attack": "#D55E00",
    "investigation": "#0072B2",
    "mount": "#009E73",
    "other": "#E5E7EB",
}


# Stream colors are mapped from the legacy stream-ablation figure.
STREAM_COLORS = {
    "full": "#2563EB",
    "no_group_visual": "#38BDF8",
    "pose_relations_only": "#059669",
    "no_relations": "#F97316",
    "animal_visual_only": "#A855F7",
    "visual_only": "#64748B",
}

STREAM_LABELS = {
    "full": "Full multimodal reuse",
    "no_group_visual": "Animal visual + pose geometry, no group",
    "pose_relations_only": "Pose geometry only",
    "no_relations": "Visual + self-pose, no pairwise geometry",
    "animal_visual_only": "Per-animal visual only",
    "visual_only": "Visual descriptors only",
}

STREAM_SHORT_LABELS = {
    "full": "Full",
    "no_group_visual": "No group\nvisual",
    "pose_relations_only": "Pose\ngeometry",
    "no_relations": "No pairwise\ngeometry",
    "animal_visual_only": "Animal\nvisual",
    "visual_only": "Visual\nonly",
}


HEAD_COLORS = {
    "lstm": "#2271B2",
    "attention": "#10B981",
    "tcn": "#64748B",
}

HEAD_LIGHT_COLORS = {
    "lstm": "#A9CCE3",
    "attention": "#A7F3D0",
    "tcn": "#CBD5E1",
}

CLASSICAL_MODEL_COLORS = {
    "rf": "#3B82F6",
    "xgb": "#F97316",
}

BACKBONE_COLORS = {
    "yolo_sppf": "#0072B2",
    "mobilenetv3_native": "#D55E00",
}

ANIMAL_COLORS = {
    "animal_1": "#D55E00",
    "animal_2": "#0072B2",
}

NEUTRAL = {
    "black": "#111827",
    "dark": "#374151",
    "mid": "#6B7280",
    "light": "#E5E7EB",
    "grid": "#D1D5DB",
}


def apply_style() -> None:
    """Apply the manuscript style globally to matplotlib."""
    plt.rcParams.update(BASE_RCPARAMS)


def panel_label(
    ax,
    label: str,
    *,
    x: float = -0.10,
    y: float = 1.08,
    suffix: str = "",
    fontsize: int = 13,
) -> None:
    """Place a bold A/B/C panel label in the legacy manuscript style."""
    ax.text(
        x,
        y,
        f"{label}{suffix}",
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False,
    )


def title_with_panel(ax, label: str, title: str, *, pad: float = 8.0) -> None:
    """Use the legacy `(A) Title` style for compact plot panels."""
    ax.set_title(f"({label}) {title}", loc="left", fontsize=11, pad=pad)


def despine(ax) -> None:
    """Remove top and right spines from a single axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def despine_all(axes: Iterable) -> None:
    """Remove top and right spines from an iterable of axes."""
    for ax in axes:
        despine(ax)


def save_figure(fig, out_base: str | Path, *, dpi: int = 600) -> list[Path]:
    """Save a figure as high-resolution PNG and vector PDF.

    `out_base` may include or omit a suffix. If a suffix is present, it is
    ignored and both `.png` and `.pdf` are written.
    """
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    stem_path = out_base.with_suffix("")
    outputs = [stem_path.with_suffix(".png"), stem_path.with_suffix(".pdf")]
    fig.savefig(outputs[0], dpi=dpi, bbox_inches="tight")
    fig.savefig(outputs[1], bbox_inches="tight")
    return outputs


def behavior_handles(include_other: bool = False):
    """Return legend handles for behavior-class ethogram plots."""
    from matplotlib.patches import Patch

    keys = ["attack", "investigation", "mount"]
    if include_other:
        keys.append("other")
    return [
        Patch(facecolor=BEHAVIOR_FILL_COLORS[k], edgecolor="none", label=k.capitalize())
        for k in keys
    ]
