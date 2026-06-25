"""Render Fly-v-Fly ethogram examples from the focused adaptation run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
    IDX_TO_CLASS,
    actions_to_bouts,
    bouts_to_frames,
    load_actions_file,
    load_raw_window_predictions,
)
from fly_to_behaviorscope_npz_full import FLY_FPS  # noqa: E402


DEFAULT_EVAL_DIR = (
    REPO_ROOT
    / "outputs"
    / "controlled_comparison_runs"
    / "fly_adaptation_attention256_cw12_seed42"
    / "fly"
    / "eval"
)
TABLE_DIR = ROOT / "tables" / "production" / "fly_v_fly"
FIG_DIR = ROOT / "figures" / "production" / "fly_v_fly"

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
    "other": "#E5E7EB",
}
RUNS = {
    "16-frame": "fly_yolo_sppf_full_attention256_w16s8_seed42",
    "8-frame": "fly_yolo_sppf_full_attention256_w8s4_seed42",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--table_dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--figure_dir", type=Path, default=FIG_DIR)
    parser.add_argument("--movies", type=int, nargs="+", default=[6, 8])
    parser.add_argument("--zoom_frames", type=int, default=1800)
    parser.add_argument("--output_name", default="fig_fly_v_fly_ethogram_examples")
    return parser.parse_args()


def style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", color=NEUTRAL["grid"], linewidth=0.7, alpha=0.25)
    ax.set_axisbelow(True)


def labels_from_actions(path: Path, n_frames: int) -> np.ndarray:
    return bouts_to_frames(actions_to_bouts(load_actions_file(path)), n_frames)


def segments_from_labels(labels: np.ndarray, start: int, end: int) -> list[tuple[int, int, str]]:
    view = labels[start:end]
    if len(view) == 0:
        return []
    segments = []
    seg_start = 0
    current = int(view[0])
    for idx in range(1, len(view)):
        value = int(view[idx])
        if value != current:
            segments.append((start + seg_start, idx - seg_start, IDX_TO_CLASS[current]))
            seg_start = idx
            current = value
    segments.append((start + seg_start, len(view) - seg_start, IDX_TO_CLASS[current]))
    return segments


def draw_track(ax: plt.Axes, labels: np.ndarray, y: float, start: int, end: int, *, height: float = 0.62) -> None:
    for seg_start, length, cls in segments_from_labels(labels, start, end):
        ax.broken_barh(
            [((seg_start - start) / FLY_FPS, max(length / FLY_FPS, 1.0 / FLY_FPS))],
            (y - height / 2.0, height),
            facecolors=BEHAVIOR_COLORS.get(cls, NEUTRAL["light"]),
            edgecolors="none",
        )
    ax.hlines(y - height / 2.0, 0, (end - start) / FLY_FPS, color=NEUTRAL["dark"], linewidth=0.35)


def choose_zoom(gt_labels: np.ndarray, window_frames: int) -> tuple[int, int, dict]:
    n = len(gt_labels)
    if n <= window_frames:
        return 0, n, {"n_bouts": 0, "n_classes": 0}
    step = max(30, window_frames // 20)
    best = (0, -1, {})
    other = CLASS_TO_IDX["other"]
    for start in range(0, n - window_frames + 1, step):
        end = start + window_frames
        view = gt_labels[start:end]
        bouts = []
        idx = 0
        while idx < len(view):
            cls = int(view[idx])
            j = idx + 1
            while j < len(view) and int(view[j]) == cls:
                j += 1
            if cls != other:
                bouts.append((cls, idx, j - 1))
            idx = j
        classes = {cls for cls, _, _ in bouts}
        score = len(bouts) + 25 * len(classes)
        # Prefer windows with at least two behavior classes when possible.
        if len(classes) < 2:
            score -= 50
        if score > best[1]:
            best = (start, score, {"n_bouts": len(bouts), "n_classes": len(classes)})
    return best[0], best[0] + window_frames, best[2]


def load_movie_labels(eval_dir: Path, row: pd.Series) -> dict[str, np.ndarray]:
    movie_id = int(row["movie_id"])
    n_frames = int(row["n_frames"])
    labels = {"Primary labels": labels_from_actions(Path(str(row["primary_actions_path"])), n_frames)}
    for label, run_name in RUNS.items():
        pred_csv = eval_dir / run_name / f"movie{movie_id}" / f"movie{movie_id}.behavior.csv"
        labels[label] = load_raw_window_predictions(pred_csv, n_frames)
    return labels


def draw_ethogram_panel(
    ax: plt.Axes,
    labels: dict[str, np.ndarray],
    start: int,
    end: int,
    title: str,
    panel: str,
    *,
    show_ylabels: bool = True,
) -> None:
    tracks = ["Primary labels", "16-frame", "8-frame"]
    y_positions = np.arange(len(tracks))[::-1]
    for y, track in zip(y_positions, tracks):
        draw_track(ax, labels[track], float(y), start, end)
    ax.set_xlim(0, (end - start) / FLY_FPS)
    ax.set_ylim(-0.7, len(tracks) - 0.25)
    ax.set_xlabel("Time within panel (s)")
    if show_ylabels:
        ax.set_yticks(y_positions, tracks)
    else:
        ax.set_yticks([])
    panel_label(ax, panel)
    ax.set_title(title, loc="left")
    style_axis(ax)


def write_zoom_table(rows: list[dict], table_dir: Path, output_name: str) -> None:
    pd.DataFrame(rows).to_csv(table_dir / f"{output_name}_zoom_windows.csv", index=False)


def main() -> int:
    args = parse_args()
    apply_style()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    heldout = pd.read_csv(args.table_dir / "fly_video_annotation_surface_heldout.csv")
    selected = heldout[heldout["movie_id"].isin(args.movies)].copy()
    if selected.empty:
        raise ValueError(f"No held-out movie rows found for {args.movies}")

    fig = plt.figure(figsize=(13.6, 6.8))
    gs = fig.add_gridspec(len(selected), 2, width_ratios=[1.15, 1.0], hspace=0.52, wspace=0.28)
    zoom_rows = []
    panel_ord = ord("A")
    for row_idx, (_, row) in enumerate(selected.iterrows()):
        movie_id = int(row["movie_id"])
        labels = load_movie_labels(args.eval_dir, row)
        n_frames = int(row["n_frames"])
        zoom_start, zoom_end, zoom_meta = choose_zoom(labels["Primary labels"], int(args.zoom_frames))
        zoom_rows.append({
            "movie_id": movie_id,
            "zoom_start_frame": zoom_start,
            "zoom_end_frame": zoom_end,
            "zoom_start_seconds": round(zoom_start / FLY_FPS, 3),
            "zoom_end_seconds": round(zoom_end / FLY_FPS, 3),
            **zoom_meta,
        })

        ax = fig.add_subplot(gs[row_idx, 0])
        draw_ethogram_panel(
            ax,
            labels,
            0,
            n_frames,
            f"Movie {movie_id}: full 30-min ethogram",
            chr(panel_ord),
            show_ylabels=True,
        )
        ax.set_xlabel("Time (s)")
        panel_ord += 1

        ax = fig.add_subplot(gs[row_idx, 1])
        draw_ethogram_panel(
            ax,
            labels,
            zoom_start,
            zoom_end,
            f"Movie {movie_id}: behavior-rich 60-s zoom",
            chr(panel_ord),
            show_ylabels=True,
        )
        panel_ord += 1

    handles = [
        Patch(facecolor=BEHAVIOR_COLORS[b], edgecolor="none", label=BEHAVIOR_LABELS[b])
        for b in ["lunge", "wing_threat", "charge", "hold", "tussle", "other"]
    ]
    fig.legend(handles=handles, frameon=False, ncol=6, loc="upper center", bbox_to_anchor=(0.52, 1.01))
    save_figure(fig, args.figure_dir / args.output_name)
    plt.close(fig)
    write_zoom_table(zoom_rows, args.table_dir, args.output_name)
    print(f"Wrote {args.figure_dir / (args.output_name + '.png')}")
    print(f"Wrote {args.table_dir / (args.output_name + '_zoom_windows.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
