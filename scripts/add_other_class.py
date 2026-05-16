#!/usr/bin/env python3
"""
add_other_class.py
==================
Top up an EXISTING YOLO-pose dataset (built by mars_to_yolo_pose.py with
classes 0=investigation, 1=mount, 2=attack) with the 4th class
3=other -- without re-decoding any of the behavior frames.

Use this when you've already paid the cost of extracting the behavior
images and don't want to redo it. The script:

  1. Validates the existing dataset layout (images/{train,val}, labels/...,
     data.yaml, manifest.csv).
  2. Re-uses mars_to_yolo_pose.py's helpers to walk MARS_data, sample
     'other' frames using the three combined strategies (distance-stratified
     / single-mouse / plain random non-bout), and decode only those frames
     from .seq.
  3. Skips any frame whose deterministic basename already exists in the
     dataset (so this is safe to re-run).
  4. Writes new images + labels under the existing
     images/{train,val} and labels/{train,val} folders, stratified 80/20.
  5. Updates data.yaml: appends "3: other" to names.
  6. Appends rows to manifest.csv.

Usage (Windows, IntegraPose conda env):
    python add_other_class.py ^
        --dataset-root path\\to\\mars_yolo_pose ^
        --mars-root    path\\to\\MARS_data ^
        --other-frames 8000 ^
        --val-frac 0.2 ^
        --seed 0
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Re-use the helpers from the sibling script
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mars_to_yolo_pose import (
    BEHAVIOR_NAMES,
    OTHER_NAME,
    NUM_KPTS,
    OTHER_DISTANCE_BODYLENGTHS_DEFAULT,
    SeqReader,
    build_video_index,
    collect_other_candidates,
    sample_other_combined,
    build_yolo_rows,
    basename_for,
    save_frame,
)


# ----------------------------------------------------------------------------
# Dataset inspection
# ----------------------------------------------------------------------------

def parse_existing_yaml(yaml_path):
    """Lightweight read of names: from data.yaml without pyyaml dep."""
    if not yaml_path.is_file():
        return None
    text = yaml_path.read_text()
    names = {}
    in_names = False
    for line in text.splitlines():
        s = line.rstrip()
        if s.startswith("names:"):
            in_names = True
            continue
        if in_names:
            m = re.match(r"\s+(\d+):\s*(.+)$", s)
            if m:
                names[int(m.group(1))] = m.group(2).strip()
            elif s and not s.startswith(" "):
                in_names = False
    return names


def existing_basenames(dataset_root):
    """Return set of basenames already in images/{train,val}/ for collision dedup."""
    out = set()
    for split in ("train", "val"):
        d = Path(dataset_root) / "images" / split
        if d.is_dir():
            for p in d.iterdir():
                if p.suffix.lower() == ".jpg":
                    out.add(p.stem)
    return out


def append_yaml_class(yaml_path, new_id, new_name):
    """Append a class line to data.yaml's names: block. Idempotent."""
    text = yaml_path.read_text()
    # If the class line already exists, do nothing
    pat = re.compile(rf"^\s+{new_id}:\s*{re.escape(new_name)}\s*$", re.M)
    if pat.search(text):
        print(f"  [yaml] class {new_id}: {new_name} already present, leaving as-is")
        return
    # Insert at end of names: block (or end of file)
    if "names:" not in text:
        # Append a fresh names: block
        text = text.rstrip() + f"\n\nnames:\n  {new_id}: {new_name}\n"
    else:
        # Find last `\d+:` line under names: and insert after it
        lines = text.splitlines()
        out = []
        in_names = False
        last_class_idx = None
        for i, line in enumerate(lines):
            out.append(line)
            if line.startswith("names:"):
                in_names = True
                last_class_idx = i
                continue
            if in_names:
                if re.match(r"\s+\d+:\s*", line):
                    last_class_idx = i
                elif line.strip() == "":
                    continue
                elif not line.startswith(" "):
                    in_names = False
        if last_class_idx is not None:
            out.insert(last_class_idx + 1, f"  {new_id}: {new_name}")
        else:
            out.append(f"  {new_id}: {new_name}")
        text = "\n".join(out)
        if not text.endswith("\n"):
            text += "\n"
    yaml_path.write_text(text)


def append_manifest(manifest_path, new_rows):
    """Append rows to an existing manifest.csv (or create one if missing)."""
    fieldnames = [
        "split", "video_id", "frame_index", "class", "class_id",
        "n_mouse_rows", "image", "label",
    ]
    write_header = not manifest_path.is_file()
    with open(manifest_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        for r in new_rows:
            # only keep recognized fields
            w.writerow({k: r.get(k, "") for k in fieldnames})


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Add the 'other' class to an existing YOLO-pose dataset.")
    p.add_argument("--dataset-root", required=True,
                   help="Existing dataset root (contains images/, labels/, data.yaml).")
    p.add_argument("--mars-root", required=True,
                   help="MARS_data folder (with train/, validation/, optionally test_*/).")
    p.add_argument("--other-frames", type=int, default=8000,
                   help="Total 'other' frames to add (default 8000).")
    p.add_argument("--val-frac", type=float, default=0.2,
                   help="Fraction of new frames assigned to val (default 0.2).")
    p.add_argument("--source-splits", nargs="+", default=["train", "validation"],
                   help="Which MARS split folders to draw 'other' frames from.")
    p.add_argument("--intra-bout-stride", type=int, default=6,
                   help="Sampling stride within non-bout regions (default 6 = ~5 Hz at 30 fps).")
    p.add_argument("--other-distance-bodylengths", type=float,
                   default=OTHER_DISTANCE_BODYLENGTHS_DEFAULT,
                   help="Neck-to-neck distance threshold for 'clearly apart' (in body lengths).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--max-videos", type=int, default=0,
                   help="Testing only: limit to first N indexed videos.")
    p.add_argument("--dry-run", action="store_true",
                   help="Preview counts without writing any files.")
    args = p.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    dataset_root = Path(args.dataset_root)
    yaml_path = dataset_root / "data.yaml"
    manifest_path = dataset_root / "manifest.csv"

    if not dataset_root.is_dir():
        print(f"ERROR: dataset root not found: {dataset_root}", file=sys.stderr); sys.exit(1)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        d = dataset_root / sub
        if not d.is_dir():
            print(f"ERROR: missing {d}", file=sys.stderr); sys.exit(1)
    if not yaml_path.is_file():
        print(f"WARNING: no data.yaml at {yaml_path}; will create one when adding the class.")

    # Inspect existing class layout
    existing_names = parse_existing_yaml(yaml_path) or {}
    print(f"[1/6] Existing classes in data.yaml: {existing_names}")
    if 3 in existing_names:
        if existing_names[3].strip() == OTHER_NAME:
            print(f"  [info] class 3 is already '{OTHER_NAME}'. We'll add new frames into it.")
        else:
            print(f"  [warn] class 3 is already '{existing_names[3]}', not '{OTHER_NAME}'. "
                  f"Continuing will mix '{OTHER_NAME}' frames into id=3 — abort if unintended.")
    other_class_id = 3

    # Pre-existing basenames so we never overwrite
    existing = existing_basenames(dataset_root)
    print(f"      existing image count under images/: {len(existing)}")

    # Build video index from MARS_data
    print(f"[2/6] Indexing videos in: {args.source_splits}")
    index = build_video_index(os.path.abspath(args.mars_root), args.source_splits)
    if args.max_videos and args.max_videos > 0:
        index = index[: args.max_videos]
        print(f"      [debug] limiting to first {len(index)} videos")
    print(f"      indexed {len(index)} videos")

    # Collect 'other' candidates with three strategies
    print(f"[3/6] Collecting 'other' candidates ...")
    by_strategy = collect_other_candidates(
        index, args.intra_bout_stride, args.other_distance_bodylengths,
    )
    for k, v in by_strategy.items():
        print(f"      pool '{k}': {len(v)} frames")

    print(f"[4/6] Stratified-combined sampling toward {args.other_frames} 'other' frames ...")
    other_picked = sample_other_combined(by_strategy, args.other_frames, rng)
    print(f"      picked {len(other_picked)} 'other' frames")

    # 80/20 split among the new frames
    rng.shuffle(other_picked)
    n_val = int(round(len(other_picked) * args.val_frac))
    val_set = other_picked[:n_val]
    train_set = other_picked[n_val:]
    print(f"      adding to train: {len(train_set)}   adding to val: {len(val_set)}")

    if args.dry_run:
        print("\n[dry-run] not writing any files.")
        return

    # Group by video for sequential reads
    by_video = defaultdict(list)  # vidx -> [(f_idx, split_name)]
    for vidx, f_idx, _ in train_set:
        by_video[vidx].append((f_idx, "train"))
    for vidx, f_idx, _ in val_set:
        by_video[vidx].append((f_idx, "val"))

    n_written = 0
    n_skipped_existing = 0
    n_skipped_decode = 0
    n_skipped_label = 0
    new_manifest_rows = []
    pose_cache = {}

    print(f"[5/6] Writing 'other' frames ...")
    for vidx, items in by_video.items():
        vid = index[vidx]
        if vidx not in pose_cache:
            with open(vid["pose_path"], "r") as f:
                pose_cache[vidx] = json.load(f)
        jpose = pose_cache[vidx]
        try:
            seq = SeqReader(vid["seq_path"])
            seek_mat = vid["seq_path"].replace(".seq", "-seek.mat")
            seq.build_seek_table(seek_mat_path=seek_mat if os.path.isfile(seek_mat) else None)
        except Exception as e:
            print(f"      [seq error] {vid['video_id']}: {e}")
            continue
        W, H = seq.width, seq.height
        items.sort(key=lambda x: x[0])

        for f_idx, split_name in items:
            base = basename_for(vid["video_id"], f_idx)
            if base in existing:
                n_skipped_existing += 1
                continue
            rows = build_yolo_rows(jpose, f_idx, other_class_id, W, H)
            if not rows:
                n_skipped_label += 1
                continue
            try:
                arr = seq.read_frame(f_idx)
            except Exception as e:
                print(f"      [decode error] {vid['video_id']} f={f_idx}: {e}")
                n_skipped_decode += 1
                continue
            img_path = dataset_root / "images" / split_name / f"{base}.jpg"
            lbl_path = dataset_root / "labels" / split_name / f"{base}.txt"
            save_frame(arr, img_path, jpeg_quality=args.jpeg_quality)
            lbl_path.write_text("\n".join(rows) + "\n")
            existing.add(base)
            new_manifest_rows.append({
                "split": split_name,
                "video_id": vid["video_id"],
                "frame_index": f_idx,
                "class": OTHER_NAME,
                "class_id": other_class_id,
                "n_mouse_rows": len(rows),
                "image": str(img_path),
                "label": str(lbl_path),
            })
            n_written += 1
        seq.close()

    print(f"      wrote: {n_written}")
    print(f"      skipped (already exists):   {n_skipped_existing}")
    print(f"      skipped (no valid label):   {n_skipped_label}")
    print(f"      skipped (decode error):     {n_skipped_decode}")

    print(f"[6/6] Updating data.yaml + manifest.csv ...")
    if not yaml_path.is_file():
        # Bootstrap a new yaml if there wasn't one
        path_str = str(dataset_root.resolve())
        yaml_path.write_text(
            f"path: {path_str}\n"
            f"train: images/train\n"
            f"val: images/val\n\n"
            f"kpt_shape: [{NUM_KPTS}, 3]\n\n"
            f"names:\n"
        )
        for i, n in enumerate(BEHAVIOR_NAMES):
            with open(yaml_path, "a") as f:
                f.write(f"  {i}: {n}\n")
    append_yaml_class(yaml_path, other_class_id, OTHER_NAME)
    print(f"      data.yaml updated with class {other_class_id}: {OTHER_NAME}")

    if new_manifest_rows:
        append_manifest(manifest_path, new_manifest_rows)
        print(f"      appended {len(new_manifest_rows)} rows to {manifest_path.name}")

    print(f"\nDone. Train this dataset with (in IntegraPose conda env):")
    print(f"  yolo pose train model=yolo11n-pose.pt \\")
    print(f"      data={yaml_path} \\")
    print(f"      imgsz=640 epochs=100 batch=16 fliplr=0.0 flipud=0.0 \\")
    print(f"      project=runs/mars_pose name=v2_4class")


if __name__ == "__main__":
    main()
