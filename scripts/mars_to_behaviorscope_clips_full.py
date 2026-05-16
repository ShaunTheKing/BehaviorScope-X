#!/usr/bin/env python3
"""
mars_to_behaviorscope_clips_full.py
====================================
Build per-class video clip folders from the **complete** Caltech MARS dataset
(train + validation + test_1 + test_2) in a single pass.

This script is the "full-dataset" companion to mars_to_behaviorscope_clips.py.
The key differences:

  - Defaults to all four native MARS splits so no split is accidentally omitted.
  - Sizes the 'other' pool proportionally to the larger multi-split corpus.
  - The "Next step" at the end points to mars_to_behaviorscope_npz_full.py
    (the MARS-direct NPZ builder), NOT to prepare_sequences_from_yolo.py.
    No YOLO inference is involved in this pipeline.

Pipeline
--------
  1. mars_to_behaviorscope_clips_full.py   <- this script
       Reads .seq videos + MARS .annot files.
       Writes one .mp4 per annotated behavior bout and one .mp4 per sampled
       'other' anchor, tagged with the source video's MARS split.

  2. mars_to_behaviorscope_npz_full.py
       Reads those .mp4 clips + the MARS pose JSON (ground-truth keypoints).
       Writes windowed NPZs in behaviorscope-n-v1 schema plus a
       sequence_manifest.json that preserves native MARS split identities
       (train -> train, validation -> val, test_1 -> test_1, test_2 -> test_2).
       No YOLO inference at any stage.

Output layout
-------------
    out_root/
      investigation/   one .mp4 per annotated investigation bout
      mount/           one .mp4 per annotated mount bout
      attack/          one .mp4 per annotated attack bout
      other/           one .mp4 per sampled non-bout anchor
      clips_manifest.csv   split, video_id, class, kind, frames, path, status

Usage (Windows, IntegraPose conda env):
    python scripts\\mars_to_behaviorscope_clips_full.py ^
        --mars-root .\\MARS_data ^
        --out-root  .\\mars_behaviorscope_clips_full ^
        --seed 42

All four splits (train, validation, test_1, test_2) are processed by default.
Pass --source-splits to restrict to a subset (e.g. for debugging).
"""

import argparse
import csv
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required. Install with:\n  pip install opencv-python",
          file=sys.stderr)
    raise

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


CLIP_CLASSES = list(BEHAVIOR_NAMES) + [OTHER_NAME]


# ----------------------------------------------------------------------------
# Helpers (identical to mars_to_behaviorscope_clips.py)
# ----------------------------------------------------------------------------

def safe_basename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._\-#]", "_", s)


def to_bgr(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported frame shape: {arr.shape}")


def write_clip(seq: SeqReader, out_path: Path, frame_indices, fps: float,
               codec: str = "mp4v") -> int:
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
                if fe - fs + 1 < min_bout_frames:
                    continue
                bout_idx = per_class_seen[cname]
                per_class_seen[cname] += 1
                clips.append((vidx, cname, fs, fe, bout_idx))
    return clips


def expand_anchor_to_window(f_anchor: int, n_frames_clip: int, n_total: int):
    half = n_frames_clip // 2
    start = f_anchor - half
    end = start + n_frames_clip - 1
    if start < 0:
        end -= start; start = 0
    if end >= n_total:
        delta = end - (n_total - 1)
        start = max(0, start - delta)
        end = n_total - 1
    return start, end


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "Build per-class video clip folders from ALL MARS splits for the "
            "MARS-direct BehaviorScope-Y pipeline (no YOLO)."
        )
    )
    p.add_argument("--mars-root", required=True,
                   help="MARS_data folder (must contain train/, validation/, test_1/, test_2/).")
    p.add_argument("--out-root", required=True,
                   help="Output root. Will contain investigation/, mount/, attack/, other/ "
                        "subdirectories and clips_manifest.csv.")
    p.add_argument("--source-splits", nargs="+",
                   default=["train", "validation", "test_1", "test_2"],
                   help="MARS splits to process. Default: all four. "
                        "Restrict for debugging only (e.g. --source-splits test_1 test_2).")
    p.add_argument("--fps", type=float, default=30.0,
                   help="Output video framerate (matches MARS source, default 30).")
    p.add_argument("--codec", default="mp4v",
                   choices=["mp4v", "avc1", "MJPG", "XVID"],
                   help="cv2 fourcc codec. mp4v works without libx264.")
    p.add_argument("--min-bout-frames", type=int, default=32,
                   help="Drop bouts shorter than this many frames. "
                        "Default 32 matches the Stage-2 NPZ window_size.")
    p.add_argument("--other-clip-frames", type=int, default=64,
                   help="Length (frames) of each 'other' clip. Default 64 gives a "
                        "small margin over the 32-frame window so windowing never "
                        "under-fills.")
    p.add_argument("--other-clips", type=int, default=2000,
                   help="Total 'other' clips to write across all source splits. "
                        "Default 2000 is scaled up from the 1500 used for train+val "
                        "only, proportional to the larger four-split corpus.")
    p.add_argument("--intra-bout-stride", type=int, default=6,
                   help="Stride for non-bout anchor candidate sampling.")
    p.add_argument("--other-distance-bodylengths", type=float,
                   default=OTHER_DISTANCE_BODYLENGTHS_DEFAULT,
                   help="Distance threshold (body lengths) for 'clearly apart' anchors.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible 'other' sampling.")
    p.add_argument("--no-other", action="store_true",
                   help="Skip 'other' class entirely (behavior bouts only).")
    p.add_argument("--max-videos", type=int, default=0,
                   help="Debugging: limit to first N indexed videos.")
    p.add_argument("--max-bout-clips-per-class", type=int, default=0,
                   help="Debugging: cap bout clips per class.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be written without decoding or writing.")
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    mars_root = os.path.abspath(args.mars_root)
    out_root = Path(args.out_root)
    for cname in CLIP_CLASSES:
        if cname == OTHER_NAME and args.no_other:
            continue
        (out_root / cname).mkdir(parents=True, exist_ok=True)

    # ---- verify requested splits exist ----
    missing_dirs = [s for s in args.source_splits
                    if not os.path.isdir(os.path.join(mars_root, s))]
    if missing_dirs:
        raise SystemExit(
            f"Requested MARS split directories not found under {mars_root}: "
            f"{missing_dirs}"
        )

    print(f"[1/4] Indexing videos in splits: {args.source_splits}", flush=True)
    index = build_video_index(mars_root, args.source_splits)
    if args.max_videos and args.max_videos > 0:
        index = index[: args.max_videos]
        print(f"      [debug] limiting to first {len(index)} videos", flush=True)
    per_split = defaultdict(int)
    for vid in index:
        per_split[vid["split"]] += 1
    print(f"      indexed {len(index)} videos across splits: {dict(per_split)}", flush=True)

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
        print(f"      [debug] capping at {args.max_bout_clips_per_class} per class "
              f"-> {len(kept)} bouts kept", flush=True)
        bout_clips = kept

    # ---- 'other' anchors ----
    other_anchors = []
    if not args.no_other:
        print(f"[3/4] Sampling 'other' anchors (target={args.other_clips}) ...", flush=True)
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
            seq.build_seek_table(
                seek_mat_path=seek_mat if os.path.isfile(seek_mat) else None
            )
        except Exception as e:
            print(f"      [seq error] {vid['video_id']}: {e}", flush=True)
            n_skipped += 1
            continue
        n_total = len(seq.seek_table)
        safe_vid = safe_basename(vid["video_id"])
        print(
            f"      [{vi_count}/{len(all_vidx)}] {vid['video_id']} "
            f"(split={vid['split']}  frames={n_total})",
            flush=True,
        )

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
                manifest_rows.append({
                    "split": vid["split"], "video_id": vid["video_id"],
                    "class": cname, "kind": "bout",
                    "start_frame": fs, "end_frame": fe, "n_frames": length,
                    "clip_path": str(out_path), "status": "exists",
                })
                continue
            written = write_clip(seq, out_path, range(fs, fe + 1),
                                 fps=args.fps, codec=args.codec)
            manifest_rows.append({
                "split": vid["split"], "video_id": vid["video_id"],
                "class": cname, "kind": "bout",
                "start_frame": fs, "end_frame": fe, "n_frames": written,
                "clip_path": str(out_path),
                "status": "written" if written > 0 else "empty",
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
            written = write_clip(seq, out_path, range(fs, fe + 1),
                                 fps=args.fps, codec=args.codec)
            manifest_rows.append({
                "split": vid["split"], "video_id": vid["video_id"],
                "class": cname, "kind": "other",
                "start_frame": fs, "end_frame": fe, "n_frames": written,
                "clip_path": str(out_path),
                "status": "written" if written > 0 else "empty",
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

    # ---- write manifest ----
    manifest_path = out_root / "clips_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "split", "video_id", "class", "kind",
            "start_frame", "end_frame", "n_frames",
            "clip_path", "status",
        ])
        w.writeheader()
        w.writerows(manifest_rows)

    # ---- summary ----
    by_class = defaultdict(int)
    by_split_class = defaultdict(lambda: defaultdict(int))
    for r in manifest_rows:
        if r["status"] in ("written", "exists"):
            by_class[r["class"]] += 1
            by_split_class[r["split"]][r["class"]] += 1

    print(f"\nDone.")
    print(f"  written: {n_written}   skipped: {n_skipped}   "
          f"skipped (too short): {n_skipped_short}")
    print(f"  clips per class (all splits):")
    for cname in CLIP_CLASSES:
        if cname == OTHER_NAME and args.no_other:
            continue
        print(f"    {cname:14s}: {by_class.get(cname, 0)}")
    print(f"  clips per split:")
    for split in args.source_splits:
        total = sum(by_split_class[split].values())
        breakdown = "  ".join(
            f"{c}={by_split_class[split].get(c, 0)}"
            for c in CLIP_CLASSES if not (c == OTHER_NAME and args.no_other)
        )
        print(f"    {split:12s}: {total} ({breakdown})")
    print(f"  manifest: {manifest_path}")
    print()
    print("Next — build windowed NPZs directly from MARS pose JSON (no YOLO):")
    print(f"  python scripts\\mars_to_behaviorscope_npz_full.py ^")
    print(f"      --mars-root   {os.path.abspath(args.mars_root)} ^")
    print(f"      --clips-root  {out_root} ^")
    print(f"      --out-root    {out_root.parent / 'mars_behaviorscope_npz_full'} ^")
    print(f"      --source-splits {' '.join(args.source_splits)} ^")
    print(f"      --emit-mode   n_animal ^")
    print(f"      --window-size 32 --window-stride 16 ^")
    print(f"      --seed 42")


if __name__ == "__main__":
    main()
