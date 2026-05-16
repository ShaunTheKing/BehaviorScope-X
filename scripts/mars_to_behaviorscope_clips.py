#!/usr/bin/env python3
"""
mars_to_behaviorscope_clips.py
==============================
Convert the Caltech MARS .seq videos + .annot bouts into per-class video
clip folders that BehaviorScope's `prepare_sequences_from_yolo.py` expects:

    out_root/
      investigation/   one .mp4 per annotated investigation bout
      mount/           one .mp4 per annotated mount bout
      attack/          one .mp4 per annotated attack bout
      other/           one .mp4 per non-bout sample
                       (mix of distance-stratified, single-mouse, plain random)

Each behavior bout becomes ONE clip whose length matches the bout duration.
Each `other` clip is a fixed length (default 60 frames = 2 s at 30 fps),
centered on an anchor frame chosen by the same three-strategy sampler used in
mars_to_yolo_pose.py (distance-stratified / single-mouse / plain random
non-bout).

Usage (Windows, IntegraPose conda env):
    python mars_to_behaviorscope_clips.py ^
        --mars-root      path\\to\\MARS_data ^
        --out-root       path\\to\\mars_behaviorscope_clips ^
        --source-splits  train validation ^
        --other-clips    1500 ^
        --seed 0

Run a separate pass for the held-out test set:
    python mars_to_behaviorscope_clips.py ^
        --mars-root      path\\to\\MARS_data ^
        --out-root       path\\to\\mars_behaviorscope_clips_test2 ^
        --source-splits  test_2 ^
        --other-clips    300 ^
        --seed 0

After this finishes, hand the folder to BehaviorScope:

    python prepare_sequences_from_yolo.py ^
        --dataset_root      ...\\mars_behaviorscope_clips ^
        --yolo_weights      ...\\runs\\mars_pose\\v1\\weights\\best.pt ^
        --output_root       ...\\mars_behaviorscope_processed ^
        --save_sequence_npz --save_frame_crops --keep_last_box ^
        --split_strategy    video
"""

import argparse
import csv
import glob
import json
import os
import random
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# OpenCV is needed for cv2.VideoWriter (mp4 emission).
try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required. Install with:\n  pip install opencv-python", file=sys.stderr)
    raise

# Reuse helpers from the sibling script.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mars_to_yolo_pose import (  # noqa: E402
    BEHAVIOR_NAMES,
    OTHER_NAME,
    OTHER_DISTANCE_BODYLENGTHS_DEFAULT,
    SeqReader,
    build_video_index,
    collect_other_candidates,
    sample_other_combined,
)


CLIP_CLASSES = list(BEHAVIOR_NAMES) + [OTHER_NAME]   # ["investigation","mount","attack","other"]


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def safe_basename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._\-#]", "_", s)


def to_bgr(arr: np.ndarray) -> np.ndarray:
    """Convert a frame array (grayscale 2D or RGB 3D) to BGR for cv2.VideoWriter."""
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported frame shape: {arr.shape}")


def write_clip(seq: SeqReader, out_path: Path, frame_indices, fps: float, codec: str = "mp4v") -> int:
    """Decode the requested frames from `seq` and write them as a video file.

    Returns the number of frames actually written. Skips frames that fail to
    decode but keeps writing the rest so a single bad frame doesn't kill a clip.
    """
    fourcc = cv2.VideoWriter_fourcc(*codec)
    W = int(seq.width); H = int(seq.height)
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (W, H), isColor=True)
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open {out_path} (codec={codec})")
    n_written = 0
    try:
        for fi in frame_indices:
            if fi < 0 or fi >= len(seq.seek_table):
                continue
            try:
                arr = seq.read_frame(fi)
            except Exception:
                continue
            try:
                writer.write(to_bgr(arr))
                n_written += 1
            except Exception:
                continue
    finally:
        writer.release()
    return n_written


def collect_bout_clips(index, min_bout_frames):
    """Return list of (vidx, class_name, start_frame, end_frame, bout_idx)."""
    clips = []
    for vidx, vid in enumerate(index):
        per_class_seen = defaultdict(int)
        for cname in BEHAVIOR_NAMES:
            for (fs, fe) in vid["behaviors"].get(cname, []):
                fs = max(0, int(fs)); fe = max(0, int(fe))
                if fe < fs:
                    continue
                length = fe - fs + 1
                if length < min_bout_frames:
                    continue
                bout_idx = per_class_seen[cname]
                per_class_seen[cname] += 1
                clips.append((vidx, cname, fs, fe, bout_idx))
    return clips


def expand_anchor_to_window(f_anchor: int, n_frames_clip: int, n_total: int):
    """Expand an anchor frame to a [start, end_inclusive] range of length
    `n_frames_clip`, clipped to the video bounds. If the video is too short
    to fit a full window centered on the anchor, the window is shifted but
    its length is preserved when possible."""
    half = n_frames_clip // 2
    start = f_anchor - half
    end = start + n_frames_clip - 1
    if start < 0:
        end -= start
        start = 0
    if end >= n_total:
        delta = end - (n_total - 1)
        start = max(0, start - delta)
        end = n_total - 1
    return start, end


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Build per-class video clip folders for BehaviorScope from MARS data.")
    p.add_argument("--mars-root", required=True, help="MARS_data folder.")
    p.add_argument("--out-root", required=True, help="Output root containing investigation/, mount/, attack/, other/.")
    p.add_argument("--source-splits", nargs="+", default=["train", "validation"],
                   help="Which MARS splits to draw from (e.g. train validation, or test_2).")
    p.add_argument("--fps", type=float, default=30.0, help="Output video framerate (matches MARS source).")
    p.add_argument("--codec", default="mp4v", choices=["mp4v", "avc1", "MJPG", "XVID"],
                   help="cv2 fourcc. mp4v works without libx264.")
    p.add_argument("--min-bout-frames", type=int, default=30,
                   help="Drop bouts shorter than this many frames (default = BehaviorScope window_size).")
    p.add_argument("--other-clip-frames", type=int, default=60,
                   help="Length (in frames) of each 'other' clip (default 60 = 2 s at 30 fps).")
    p.add_argument("--other-clips", type=int, default=1500,
                   help="Total 'other' clips to write across all source videos.")
    p.add_argument("--intra-bout-stride", type=int, default=6,
                   help="Stride for non-bout anchor sampling (matches mars_to_yolo_pose default).")
    p.add_argument("--other-distance-bodylengths", type=float, default=OTHER_DISTANCE_BODYLENGTHS_DEFAULT,
                   help="Distance threshold (in body lengths) for 'clearly apart'.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-other", action="store_true",
                   help="Skip the 'other' class entirely (only emit behavior bout clips).")
    p.add_argument("--max-videos", type=int, default=0, help="Testing only: limit to first N indexed videos.")
    p.add_argument("--max-bout-clips-per-class", type=int, default=0,
                   help="If >0, cap bout-clip emission at this count per class (test/preview).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written without decoding or writing any video files.")
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    mars_root = os.path.abspath(args.mars_root)
    out_root = Path(args.out_root)
    for cname in CLIP_CLASSES:
        if cname == OTHER_NAME and args.no_other:
            continue
        (out_root / cname).mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Indexing videos in: {args.source_splits}", flush=True)
    index = build_video_index(mars_root, args.source_splits)
    if args.max_videos and args.max_videos > 0:
        index = index[: args.max_videos]
        print(f"      [debug] limiting to first {len(index)} videos", flush=True)
    print(f"      indexed {len(index)} videos", flush=True)

    # ---- behavior bout clips ----
    print(f"[2/4] Cataloguing bouts (min_bout_frames={args.min_bout_frames}) ...", flush=True)
    bout_clips = collect_bout_clips(index, args.min_bout_frames)
    counts = defaultdict(int)
    for _, cname, _, _, _ in bout_clips:
        counts[cname] += 1
    for cname in BEHAVIOR_NAMES:
        print(f"        {cname:14s} bouts: {counts.get(cname, 0)}", flush=True)
    if args.max_bout_clips_per_class and args.max_bout_clips_per_class > 0:
        kept = []
        per_class = defaultdict(int)
        for c in bout_clips:
            cname = c[1]
            if per_class[cname] < args.max_bout_clips_per_class:
                kept.append(c); per_class[cname] += 1
        print(f"      [debug] capping at {args.max_bout_clips_per_class} per class -> total bouts kept = {len(kept)}", flush=True)
        bout_clips = kept

    # ---- 'other' anchors ----
    other_anchors = []  # list of (vidx, frame_anchor)
    if not args.no_other:
        print(f"[3/4] Sampling 'other' anchors (target = {args.other_clips}) ...", flush=True)
        other_by_strategy = collect_other_candidates(
            index, args.intra_bout_stride, args.other_distance_bodylengths,
        )
        for k, v in other_by_strategy.items():
            print(f"        pool '{k}': {len(v)} frames", flush=True)
        anchors = sample_other_combined(other_by_strategy, args.other_clips, rng)
        other_anchors = [(c[0], c[1]) for c in anchors]
        print(f"        'other' anchors picked: {len(other_anchors)}", flush=True)
    else:
        print(f"[3/4] --no-other set; skipping 'other' clips.", flush=True)

    # ---- write clips ----
    print(f"[4/4] Writing clips ...", flush=True)
    if args.dry_run:
        print(f"      [dry-run] would write {len(bout_clips)} bout clips and "
              f"{len(other_anchors)} 'other' clips. Exiting without I/O.", flush=True)
        return

    # Group by video so each .seq is opened/decoded once.
    by_video_bouts = defaultdict(list)
    for (vidx, cname, fs, fe, bidx) in bout_clips:
        by_video_bouts[vidx].append(("bout", cname, fs, fe, bidx))
    by_video_other = defaultdict(list)
    for oi, (vidx, fa) in enumerate(other_anchors):
        by_video_other[vidx].append(("other", OTHER_NAME, fa, oi))

    manifest_rows = []
    n_written = 0
    n_skipped = 0
    n_skipped_short = 0

    all_vidx = sorted(set(list(by_video_bouts.keys()) + list(by_video_other.keys())))
    for vi_count, vidx in enumerate(all_vidx, 1):
        vid = index[vidx]
        try:
            seq = SeqReader(vid["seq_path"])
            seek_mat = vid["seq_path"].replace(".seq", "-seek.mat")
            seq.build_seek_table(seek_mat_path=seek_mat if os.path.isfile(seek_mat) else None)
        except Exception as e:
            print(f"      [seq error] {vid['video_id']}: {e}", flush=True)
            n_skipped += 1
            continue
        n_total = len(seq.seek_table)
        safe_vid = safe_basename(vid["video_id"])
        print(f"      [{vi_count}/{len(all_vidx)}] {vid['video_id']} (frames={n_total})", flush=True)

        # bout clips
        for entry in by_video_bouts.get(vidx, []):
            _, cname, fs, fe, bidx = entry
            fs = max(0, fs); fe = min(n_total - 1, fe)
            length = fe - fs + 1
            if length < args.min_bout_frames:
                n_skipped_short += 1
                continue
            out_name = f"{safe_vid}__{cname}_b{bidx:04d}_f{fs:06d}-{fe:06d}.mp4"
            out_path = out_root / cname / out_name
            if out_path.is_file() and out_path.stat().st_size > 0:
                # Skip if already produced (idempotent re-run)
                manifest_rows.append({
                    "split": vid["split"], "video_id": vid["video_id"],
                    "class": cname, "kind": "bout",
                    "start_frame": fs, "end_frame": fe, "n_frames": length,
                    "clip_path": str(out_path), "status": "exists",
                })
                continue
            written = write_clip(seq, out_path, range(fs, fe + 1), fps=args.fps, codec=args.codec)
            manifest_rows.append({
                "split": vid["split"], "video_id": vid["video_id"],
                "class": cname, "kind": "bout",
                "start_frame": fs, "end_frame": fe, "n_frames": written,
                "clip_path": str(out_path), "status": "written" if written > 0 else "empty",
            })
            if written > 0:
                n_written += 1
            else:
                n_skipped += 1
                try:
                    out_path.unlink()
                except Exception:
                    pass

        # 'other' clips
        for entry in by_video_other.get(vidx, []):
            _, cname, fa, oi = entry
            fs, fe = expand_anchor_to_window(fa, args.other_clip_frames, n_total)
            length = fe - fs + 1
            if length < args.min_bout_frames:
                n_skipped_short += 1
                continue
            out_name = f"{safe_vid}__{cname}_o{oi:06d}_f{fs:06d}-{fe:06d}.mp4"
            out_path = out_root / cname / out_name
            if out_path.is_file() and out_path.stat().st_size > 0:
                manifest_rows.append({
                    "split": vid["split"], "video_id": vid["video_id"],
                    "class": cname, "kind": "other",
                    "start_frame": fs, "end_frame": fe, "n_frames": length,
                    "clip_path": str(out_path), "status": "exists",
                })
                continue
            written = write_clip(seq, out_path, range(fs, fe + 1), fps=args.fps, codec=args.codec)
            manifest_rows.append({
                "split": vid["split"], "video_id": vid["video_id"],
                "class": cname, "kind": "other",
                "start_frame": fs, "end_frame": fe, "n_frames": written,
                "clip_path": str(out_path), "status": "written" if written > 0 else "empty",
            })
            if written > 0:
                n_written += 1
            else:
                n_skipped += 1
                try:
                    out_path.unlink()
                except Exception:
                    pass

        seq.close()

    manifest_path = out_root / "clips_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "split", "video_id", "class", "kind",
            "start_frame", "end_frame", "n_frames",
            "clip_path", "status",
        ])
        w.writeheader()
        w.writerows(manifest_rows)

    # Summary
    by_class = defaultdict(int)
    for r in manifest_rows:
        if r["status"] in ("written", "exists"):
            by_class[r["class"]] += 1
    print(f"\nDone.")
    print(f"  written: {n_written}   skipped: {n_skipped}   skipped (too short): {n_skipped_short}")
    for cname in CLIP_CLASSES:
        if cname == OTHER_NAME and args.no_other:
            continue
        print(f"  total clips for {cname}: {by_class.get(cname, 0)}")
    print(f"  manifest: {manifest_path}")
    print()
    print("Next: hand the folder to BehaviorScope's prepare_sequences_from_yolo.py:")
    print(f"  python prepare_sequences_from_yolo.py \\")
    print(f"      --dataset_root {out_root} \\")
    print(f"      --yolo_weights path\\to\\runs\\pose\\weights\\best.pt \\")
    print(f"      --output_root  path\\to\\mars_behaviorscope_processed \\")
    print(f"      --save_sequence_npz --save_frame_crops --keep_last_box \\")
    print(f"      --split_strategy video --yolo_task pose")


if __name__ == "__main__":
    main()
