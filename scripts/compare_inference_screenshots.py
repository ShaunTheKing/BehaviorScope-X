#!/usr/bin/env python3
"""
compare_inference_screenshots.py
================================
Build a side-by-side keyframe-comparison figure from two inference videos
of the *same* source footage produced by different BehaviorScope-Y models.

Both videos must share the same source — same frame count, same fps, same
arena. The script samples N random frame indices (reproducible via --seed),
grabs that frame from each video, and writes a single PNG with two rows:

    [RGB-only ]  [t1] [t2] [t3] [t4] [t5]
    [Pose+RGB ]  [t1] [t2] [t3] [t4] [t5]

Each cell is annotated with its frame index + timestamp at the bottom.
Each row has a colored left-margin label naming the model variant.

Naming convention used in the figure (kept architectural, not run-name):
  * top row:    "RGB-only"   — the per-animal-RGB-only ablation (R2-style)
  * bottom row: "Pose + RGB" — the full four-stream hybrid    (R5-style)

Usage (Windows, BehaviorScope conda env):
    python compare_inference_screenshots.py ^
        --rgb-only-video  path\\to\\rgb_only_annotated.mp4 ^
        --pose-rgb-video  path\\to\\hybrid_annotated.mp4 ^
        --output          figures\\qualitative\\compare.png ^
        --num-frames 5 ^
        --seed 42

The output PNG is self-contained — open it in any image viewer or drop it
straight into a Quarto figure. The selected frame indices and timestamps
are also printed to stdout so they can be recorded in the figure caption.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np


# ---- defaults --------------------------------------------------------------

DEFAULT_NUM_FRAMES = 5
DEFAULT_MIN_SPACING = 30      # frames between consecutive samples (≈1s @ 30fps)
DEFAULT_CELL_WIDTH = 384      # px per cell after resize; 5 cells -> ~1920 wide
DEFAULT_LEFT_LABEL_W = 200
DEFAULT_LABEL_STRIP_H = 32
DEFAULT_SEED = 42

ROW_COLOR_RGB_ONLY = (60, 60, 160)   # BGR — muted red, signals "baseline / limited"
ROW_COLOR_POSE_RGB = (140, 110, 60)  # BGR — muted teal, signals "full / upgraded"
LABEL_STRIP_BG = (40, 40, 40)


def parse_args():
    p = argparse.ArgumentParser(
        description="Side-by-side keyframe figure: RGB-only vs Pose+RGB inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rgb-only-video", required=True, type=Path,
                   help="Annotated MP4 from the per-animal-RGB-only model "
                        "(R2-style). This becomes the top row.")
    p.add_argument("--pose-rgb-video", required=True, type=Path,
                   help="Annotated MP4 from the full-hybrid model (R5-style). "
                        "This becomes the bottom row.")
    p.add_argument("--output", required=True, type=Path,
                   help="Path for the output PNG.")
    p.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES,
                   help="Number of frame samples per row.")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED,
                   help="Random seed; same seed + same video lengths gives "
                        "the same frame indices.")
    p.add_argument("--min-spacing", type=int, default=DEFAULT_MIN_SPACING,
                   help="Minimum frame-index spacing between samples, so "
                        "samples don't bunch together temporally.")
    p.add_argument("--cell-width", type=int, default=DEFAULT_CELL_WIDTH,
                   help="Width in pixels each frame is resized to before "
                        "tiling into the grid.")
    p.add_argument("--title", type=str, default="",
                   help="Optional one-line title rendered as a header strip "
                        "above the grid (e.g., 'Mouse155_20151124_..._Top'). "
                        "Empty string = no title strip.")
    return p.parse_args()


# ---- video helpers ---------------------------------------------------------

def get_video_info(path: Path) -> tuple[float, int]:
    """Return (fps, frame_count). Raises if the video can't be opened."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        sys.exit(f"[error] cannot open video: {path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if n_frames <= 0:
        sys.exit(f"[error] video reports 0 frames: {path}")
    return fps, n_frames


def grab_frames(path: Path, indices: list[int]) -> dict[int, np.ndarray]:
    """Read the given frame indices from `path`. Returns {idx: BGR frame}."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        sys.exit(f"[error] cannot open video: {path}")
    out: dict[int, np.ndarray] = {}
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            sys.exit(f"[error] failed to read frame {idx} from {path}")
        out[idx] = frame
    cap.release()
    return out


# ---- sampling --------------------------------------------------------------

def sample_frame_indices(
    n_frames: int, num_frames: int, min_spacing: int, seed: int
) -> list[int]:
    """Random sample of `num_frames` integers in [0, n_frames-1] with at
    least `min_spacing` between consecutive samples (after sorting).

    Tries up to 1000 attempts to find a satisfying random sample; if no
    luck, falls back to evenly-spaced sampling so the script never errors
    on this account.
    """
    if num_frames <= 0:
        sys.exit("[error] --num-frames must be positive.")
    if n_frames < num_frames:
        sys.exit(f"[error] video has {n_frames} frames; cannot draw "
                 f"{num_frames} samples.")

    rng = random.Random(seed)
    pool = list(range(n_frames))
    for _ in range(1000):
        sample = sorted(rng.sample(pool, num_frames))
        if all(b - a >= min_spacing for a, b in zip(sample, sample[1:])):
            return sample

    # Fallback: deterministic evenly-spaced sampling. Loud about it so the
    # user notices.
    print(f"[warn] could not find a {num_frames}-sample set with "
          f"min_spacing={min_spacing} after 1000 tries; falling back to "
          "evenly-spaced indices.", file=sys.stderr)
    if num_frames == 1:
        return [n_frames // 2]
    step = (n_frames - 1) / (num_frames - 1)
    return [int(round(i * step)) for i in range(num_frames)]


# ---- compositing -----------------------------------------------------------

def resize_to_width(frame: np.ndarray, target_w: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = target_w / w
    new_h = int(round(h * scale))
    return cv2.resize(frame, (target_w, new_h), interpolation=cv2.INTER_AREA)


def make_label_strip(width: int, height: int, text: str) -> np.ndarray:
    strip = np.full((height, width, 3), LABEL_STRIP_BG, dtype=np.uint8)
    cv2.putText(
        strip, text, (10, height - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA,
    )
    return strip


def make_row(
    frames: dict[int, np.ndarray],
    indices: list[int],
    fps: float,
    cell_width: int,
    label_height: int,
    left_label_text: str,
    left_label_color: tuple[int, int, int],
    left_label_width: int,
) -> np.ndarray:
    """Build a single row: left model-label + N (frame, timestamp) cells."""
    cells = []
    for idx in indices:
        frame = frames[idx]
        resized = resize_to_width(frame, cell_width)
        ts = idx / max(fps, 1e-6)
        label_text = f"frame {idx}    t = {ts:.1f} s"
        cap = make_label_strip(cell_width, label_height, label_text)
        cells.append(np.vstack([resized, cap]))

    body = np.hstack(cells)
    row_height = body.shape[0]

    # Left margin model label, vertically centered.
    margin = np.full((row_height, left_label_width, 3), left_label_color, dtype=np.uint8)
    text_size, _ = cv2.getTextSize(
        left_label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.85, 2,
    )
    text_w, text_h = text_size
    text_x = max(10, (left_label_width - text_w) // 2)
    text_y = (row_height + text_h) // 2
    cv2.putText(
        margin, left_label_text, (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA,
    )
    return np.hstack([margin, body])


def make_title_strip(width: int, text: str, height: int = 36) -> np.ndarray:
    strip = np.full((height, width, 3), (20, 20, 20), dtype=np.uint8)
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    tx = max(10, (width - text_size[0]) // 2)
    ty = (height + text_size[1]) // 2
    cv2.putText(
        strip, text, (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2, cv2.LINE_AA,
    )
    return strip


def build_comparison_image(
    rgb_frames: dict[int, np.ndarray],
    pose_frames: dict[int, np.ndarray],
    indices: list[int],
    fps: float,
    cell_width: int,
    title: str,
) -> np.ndarray:
    row_rgb = make_row(
        rgb_frames, indices, fps, cell_width,
        DEFAULT_LABEL_STRIP_H, "RGB-only", ROW_COLOR_RGB_ONLY,
        DEFAULT_LEFT_LABEL_W,
    )
    row_pose = make_row(
        pose_frames, indices, fps, cell_width,
        DEFAULT_LABEL_STRIP_H, "Pose + RGB", ROW_COLOR_POSE_RGB,
        DEFAULT_LEFT_LABEL_W,
    )
    grid = np.vstack([row_rgb, row_pose])

    if title.strip():
        title_strip = make_title_strip(grid.shape[1], title.strip())
        grid = np.vstack([title_strip, grid])

    return grid


# ---- main ------------------------------------------------------------------

def main():
    args = parse_args()

    if not args.rgb_only_video.is_file():
        sys.exit(f"[error] file not found: {args.rgb_only_video}")
    if not args.pose_rgb_video.is_file():
        sys.exit(f"[error] file not found: {args.pose_rgb_video}")

    rgb_fps, rgb_n = get_video_info(args.rgb_only_video)
    pose_fps, pose_n = get_video_info(args.pose_rgb_video)

    # The two videos must come from the same source. Warn loudly if not —
    # the user might be comparing apples to oranges.
    if abs(rgb_fps - pose_fps) > 0.5:
        print(f"[warn] FPS mismatch: RGB-only={rgb_fps:.2f}, "
              f"Pose+RGB={pose_fps:.2f}. Using {rgb_fps:.2f} for timestamps.",
              file=sys.stderr)
    if rgb_n != pose_n:
        print(f"[warn] frame-count mismatch: RGB-only={rgb_n}, "
              f"Pose+RGB={pose_n}. Sampling within the shared range "
              f"[0, {min(rgb_n, pose_n) - 1}].", file=sys.stderr)
    fps = rgb_fps
    n_min = min(rgb_n, pose_n)

    print(f"[info] RGB-only  : {rgb_n:>6d} frames @ {rgb_fps:.2f} fps "
          f"({args.rgb_only_video.name})")
    print(f"[info] Pose+RGB  : {pose_n:>6d} frames @ {pose_fps:.2f} fps "
          f"({args.pose_rgb_video.name})")

    indices = sample_frame_indices(
        n_min, args.num_frames, args.min_spacing, args.seed,
    )
    print(f"[info] sampled indices : {indices}")
    print(f"[info] timestamps (s)  : "
          f"{[round(i / max(fps, 1e-6), 2) for i in indices]}")
    print(f"[info] seed            : {args.seed}")

    rgb_frames = grab_frames(args.rgb_only_video, indices)
    pose_frames = grab_frames(args.pose_rgb_video, indices)

    img = build_comparison_image(
        rgb_frames, pose_frames, indices, fps,
        cell_width=args.cell_width, title=args.title,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), img):
        sys.exit(f"[error] cv2.imwrite failed for {args.output}")
    print(f"[done] wrote {args.output}  "
          f"({img.shape[1]} x {img.shape[0]} px, {img.nbytes // 1024} KB)")


if __name__ == "__main__":
    main()
