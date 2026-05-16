#!/usr/bin/env python3
"""
mars_to_behaviorscope_npz_full.py
=================================
Build a full, publication-grade BehaviorScope sequence_manifest.json and
NPZ set directly from MARS data, **bypassing YOLO inference entirely**. Uses
the MARS ground-truth pose JSON for keypoints and bboxes, and decodes frames
from the already-chopped per-class .mp4 clips produced by
mars_to_behaviorscope_clips.py.

Two emit modes:

  --emit-mode single   Legacy single-animal NPZ (one NPZ per mouse per
                       window). Each mouse contributes its own training
                       sample with the bout's behavior label. Trains
                       directly with the existing single-animal
                       BehaviorScope `train_behavior_lstm.py`.

  --emit-mode n_animal BehaviorScope-Y two-animal NPZ (one NPZ per window)
                       matching the schema produced by cropping_n.py.
                       For when the new N-animal data loader is wired up.

Usage (Windows, IntegraPose conda env):
    python mars_to_behaviorscope_npz_full.py ^
        --mars-root  .\\MARS_data ^
        --clips-root .\\mars_behaviorscope_clips ^
        --out-root   .\\mars_behaviorscope_full ^
        --source-splits train validation test_1 test_2 ^
        --emit-mode  n_animal ^
        --window-size 32 --window-stride 16 ^
        --crop-size 224 --pad-ratio 0.2 ^
        --seed 42

The script:
  - parses the deterministic clip filenames written by
    mars_to_behaviorscope_clips.py (`<video_id>__<class>_b<idx>_f<start>-<end>.mp4`)
    to recover (source video, source frame range) for each clip,
  - loads each source video's MARS pose JSON once and uses it for keypoints
    and bboxes (no YOLO inference),
  - decodes frames from the .mp4 (no .seq decode either - already done by
    the clip extractor),
  - slides a window of `window_size` frames at stride `window_stride`,
  - emits NPZs in the requested format,
  - writes sequence_manifest.json without re-splitting clips by ratio. The
    top-level split keys stay BehaviorScope-compatible (`train`, `val`,
    `test_1`, `test_2`), while each sample keeps its original `mars_split`
    provenance (`validation` remains `validation` on the sample record).
"""

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# Reuse helpers from mars_to_yolo_pose.py for parsing + body length
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from mars_to_yolo_pose import (   # noqa: E402
    BEHAVIOR_NAMES,
    OTHER_NAME,
    NUM_KPTS,
    BSCORE_MIN,
    NECK_KPT_INDEX,
    TAIL_KPT_INDEX,
    build_video_index,
)

_BEHAVIORSCOPE_DIR = os.path.join(os.path.dirname(_HERE), "BehaviorScope")
if _BEHAVIORSCOPE_DIR not in sys.path:
    sys.path.insert(0, _BEHAVIORSCOPE_DIR)
try:
    from utils.pose_features_n import extract_relational_features, REL_FEATURE_DIM
except Exception as _e:
    print(
        f"[warn] could not import BehaviorScope relation extractor: {_e}. "
        "Relation features will be zero.",
        flush=True,
    )
    extract_relational_features = None
    REL_FEATURE_DIM = 11


# ----------------------------------------------------------------------------
# Constants & helpers
# ----------------------------------------------------------------------------

CLASS_NAMES_DEFAULT = list(BEHAVIOR_NAMES) + [OTHER_NAME]   # ["investigation","mount","attack","other"]
MARS_TO_MANIFEST_SPLIT = {
    "train": "train",
    "validation": "val",
    "test_1": "test_1",
    "test_2": "test_2",
}

# Filename: <video_id>__<class>_b<idx>_f<fs>-<fe>.mp4
#       OR  <video_id>__<class>_o<idx>_f<fs>-<fe>.mp4   (other-class anchors)
CLIP_FILENAME_RE = re.compile(
    r"^(?P<video_id>.+?)__(?P<class>[A-Za-z_]+)_(?P<kind>b|o)(?P<idx>\d+)"
    r"_f(?P<fs>\d+)-(?P<fe>\d+)\.mp4$"
)


def parse_clip_filename(path: Path):
    m = CLIP_FILENAME_RE.match(path.name)
    if not m:
        return None
    return {
        "video_id": m.group("video_id"),
        "class_name": m.group("class"),
        "kind": m.group("kind"),
        "idx": int(m.group("idx")),
        "frame_start": int(m.group("fs")),
        "frame_end": int(m.group("fe")),
    }


def load_video_frames(path: Path):
    """Decode all frames from an .mp4. Returns list of HxWx3 uint8 RGB arrays."""
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok or bgr is None:
            break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(rgb)
    cap.release()
    return frames


def expand_bbox_xyxy(bbox, pad_ratio, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    width = max(x2 - x1, 1.0)
    height = max(y2 - y1, 1.0)
    cx = x1 + width / 2.0
    cy = y1 + height / 2.0
    scale = 1.0 + float(pad_ratio)
    half_w = width * scale / 2.0
    half_h = height * scale / 2.0
    nx1 = max(0.0, cx - half_w)
    ny1 = max(0.0, cy - half_h)
    nx2 = min(float(w), cx + half_w)
    ny2 = min(float(h), cy + half_h)
    if nx2 <= nx1 or ny2 <= ny1:
        return (0, 0, int(w), int(h))
    return (int(nx1), int(ny1), int(nx2), int(ny2))


def crop_resize_pad_rgb(frame_rgb, crop_xyxy, crop_size):
    h, w = frame_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in crop_xyxy]
    out_w = max(x2 - x1, 1)
    out_h = max(y2 - y1, 1)
    canvas = np.zeros((out_h, out_w, 3), dtype=frame_rgb.dtype)
    sx1 = max(0, x1); sy1 = max(0, y1)
    sx2 = min(w, x2); sy2 = min(h, y2)
    if sx2 > sx1 and sy2 > sy1:
        dx1 = sx1 - x1
        dy1 = sy1 - y1
        canvas[dy1: dy1 + (sy2 - sy1), dx1: dx1 + (sx2 - sx1)] = frame_rgb[sy1:sy2, sx1:sx2]
    return cv2.resize(canvas, (int(crop_size), int(crop_size)), interpolation=cv2.INTER_LINEAR)


def fixed_square_xyxy(image_shape, center_xy, side_px):
    """Fixed-size square crop centered on a point, clipped to frame."""
    h, w = image_shape[:2]
    side = max(float(side_px), 2.0)
    cx, cy = float(center_xy[0]), float(center_xy[1])
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))
    if x2 <= x1: x2 = x1 + 1
    if y2 <= y1: y2 = y1 + 1
    return (x1, y1, x2, y2)


def project_keypoints_to_crop(kp_xy_orig, kp_conf, crop_xyxy, crop_size):
    """Project [K, 2] keypoints from original-frame coords into crop pixel coords (0..crop_size).

    Returns a [K, 3] array of (x_crop_px, y_crop_px, conf).
    """
    x1, y1, x2, y2 = [float(v) for v in crop_xyxy]
    cw = max(x2 - x1, 1.0)
    ch = max(y2 - y1, 1.0)
    K = kp_xy_orig.shape[0]
    out = np.zeros((K, 3), dtype=np.float32)
    out[:, 0] = (kp_xy_orig[:, 0] - x1) / cw * float(crop_size)
    out[:, 1] = (kp_xy_orig[:, 1] - y1) / ch * float(crop_size)
    if kp_conf is not None:
        out[:, 2] = kp_conf
    return out


def crop_coverage(crop_xyxy, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [float(v) for v in crop_xyxy]
    area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    if area <= 1e-9:
        return 0.0
    ix1 = max(0.0, x1); iy1 = max(0.0, y1)
    ix2 = min(float(w), x2); iy2 = min(float(h), y2)
    inside = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    return float(max(0.0, min(1.0, inside / area)))


def estimate_body_length_calibration(
    kp,
    bscores,
    *,
    bscore_threshold: float,
    min_valid_frames: int,
    fallback_px: float,
) -> dict:
    """Robust per-video neck-to-tail calibration across all visible animals."""
    distances = []
    n_animals = int(bscores.shape[1]) if bscores.ndim >= 2 else 0
    for mouse_id in range(n_animals):
        good = np.asarray(bscores[:, mouse_id] >= float(bscore_threshold))
        if not np.any(good):
            continue
        neck = kp[good, mouse_id, :, NECK_KPT_INDEX]
        tail = kp[good, mouse_id, :, TAIL_KPT_INDEX]
        d = np.linalg.norm(neck - tail, axis=-1)
        d = d[np.isfinite(d) & (d > 0)]
        if d.size:
            distances.append(d)

    if distances:
        all_distances = np.concatenate(distances).astype(np.float32, copy=False)
    else:
        all_distances = np.asarray([], dtype=np.float32)

    valid_frames = int(all_distances.size)
    if valid_frames >= int(min_valid_frames):
        return {
            "body_length_px": float(np.median(all_distances)),
            "body_length_source": "per_video_median",
            "body_length_valid_frames": valid_frames,
        }

    return {
        "body_length_px": float(fallback_px),
        "body_length_source": "train_setup_fallback",
        "body_length_valid_frames": valid_frames,
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=(
            "Build full BehaviorScope NPZs directly from MARS pose JSON while "
            "preserving native MARS train/validation/test split identities."
        )
    )
    p.add_argument("--mars-root", required=True, help="MARS_data folder.")
    p.add_argument("--clips-root", required=True, help="Output of mars_to_behaviorscope_clips.py (contains <class>/*.mp4).")
    p.add_argument("--out-root", required=True, help="Where to write sequence_npz/, sequence_manifest.json, calibration.json.")
    p.add_argument("--source-splits", nargs="+", default=["train", "validation", "test_1", "test_2"],
                   help=("Native MARS splits to index. Top-level manifest keys "
                         "map validation -> val for BehaviorScope compatibility."))
    p.add_argument("--allow-missing-source-splits", action="store_true",
                   help="Allow requested source split directories to be missing or empty.")
    p.add_argument("--allow-zero-relation-features", action="store_true",
                   help="Continue even if the BehaviorScope relation extractor could not be "
                        "imported (relation features will be all zeros). Safe for single-mode "
                        "or quick triage builds; do NOT use for paper-grade n_animal R5 builds.")
    p.add_argument("--window-size", type=int, default=32)
    p.add_argument("--window-stride", type=int, default=16)
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--pad-ratio", type=float, default=0.2)
    p.add_argument("--pose-conf-threshold", type=float, default=0.3)
    p.add_argument("--bscore-threshold", type=float, default=BSCORE_MIN)
    p.add_argument("--emit-mode", choices=["single", "n_animal"], default="n_animal")
    p.add_argument("--n-animals", type=int, default=2,
                   help="Used when --emit-mode n_animal (default 2 for MARS dyads).")
    p.add_argument("--animal-scale-factor", type=float, default=4.0)
    p.add_argument("--group-scale-factor", type=float, default=8.0)
    p.add_argument("--body-length-min-valid-frames", type=int, default=100,
                   help="Minimum finite neck-to-tail distances needed for per-video calibration.")
    p.add_argument("--body-length-fallback-px", type=float, default=100.0,
                   help="Fallback body length when pose calibration has too few valid frames.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-clips", type=int, default=0,
                   help="If >0, limit total clips processed (testing only). Not balanced across classes.")
    p.add_argument("--max-per-class", type=int, default=0,
                   help="If >0, randomly sample up to N clips per class. Use this for a quick "
                        "balanced subset (e.g., --max-per-class 75 yields up to 300 clips for "
                        "4 classes). Random sampling is seeded by --seed.")
    p.add_argument("--no-compress", action="store_true", default=False,
                   help="Write uncompressed NPZs (np.savez instead of np.savez_compressed). "
                        "Roughly 4x faster — eliminates the single-threaded zlib compression "
                        "bottleneck — at the cost of ~4x larger files on disk. "
                        "Use when rebuild speed matters more than disk space.")
    args = p.parse_args()

    # Fail fast if the relation extractor is unavailable and this is a
    # paper-grade n_animal build. Zero relation features would silently
    # corrupt the R5 dataset; the flag must be explicit to override.
    if args.emit_mode == "n_animal" and extract_relational_features is None:
        if not args.allow_zero_relation_features:
            raise SystemExit(
                "BehaviorScope relation extractor could not be imported and "
                "--allow-zero-relation-features was not set. "
                "For a paper-grade n_animal build, relation features must be "
                "non-zero (R5 depends on them). "
                "Fix the import (check that BehaviorScope/utils/pose_features_n.py "
                "is importable from this environment), or pass "
                "--allow-zero-relation-features for triage builds only."
            )

    savez_fn = np.savez if args.no_compress else np.savez_compressed
    if args.no_compress:
        print("[npz] --no-compress: writing uncompressed NPZs (faster build, larger files)",
              flush=True)

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    mars_root = os.path.abspath(args.mars_root)
    clips_root = Path(args.clips_root)
    out_root = Path(args.out_root)
    (out_root / "sequence_npz").mkdir(parents=True, exist_ok=True)

    missing_split_dirs = [
        split for split in args.source_splits
        if not os.path.isdir(os.path.join(mars_root, split))
    ]
    if missing_split_dirs and not args.allow_missing_source_splits:
        raise SystemExit(
            f"Requested MARS split directories not found under {mars_root}: "
            f"{missing_split_dirs}. Pass --allow-missing-source-splits for a partial run."
        )

    # ---- Discover all .mp4 clips and parse filenames ----
    print(f"[1/5] Scanning clips under {clips_root} ...", flush=True)
    clip_records = []
    for class_dir in sorted(clips_root.iterdir()):
        if not class_dir.is_dir():
            continue
        cname = class_dir.name
        for mp4 in sorted(class_dir.glob("*.mp4")):
            meta = parse_clip_filename(mp4)
            if meta is None:
                print(f"  [skip] unparseable filename: {mp4.name}", flush=True)
                continue
            # The class encoded in the filename should match the parent folder
            if meta["class_name"] != cname:
                print(f"  [warn] class mismatch: file says {meta['class_name']}, folder is {cname} - using folder")
                meta["class_name"] = cname
            meta["mp4_path"] = mp4
            clip_records.append(meta)
    print(f"      found {len(clip_records)} parseable clips", flush=True)
    if args.max_per_class and args.max_per_class > 0:
        # Random per-class subsample (seeded by --seed for reproducibility)
        by_class = defaultdict(list)
        for c in clip_records:
            by_class[c["class_name"]].append(c)
        balanced = []
        for cname in sorted(by_class.keys()):
            pool = by_class[cname]
            rng.shuffle(pool)
            taken = pool[: int(args.max_per_class)]
            balanced.extend(taken)
            print(f"      [balanced subset] class '{cname}': pool={len(pool)} taken={len(taken)}")
        clip_records = balanced
        print(f"      [balanced subset] total clips: {len(clip_records)}")
    if args.max_clips and args.max_clips > 0:
        clip_records = clip_records[: args.max_clips]
        print(f"      [debug] limiting to first {len(clip_records)} clips")
    if not clip_records:
        raise SystemExit("No usable clips found. Check --clips-root.")

    classes_present = sorted({c["class_name"] for c in clip_records})
    class_to_idx = {n: i for i, n in enumerate(classes_present)}
    print(f"      class_to_idx: {class_to_idx}")

    # ---- Index MARS source videos to find pose JSON for each clip ----
    print(f"[2/5] Indexing MARS source videos in: {args.source_splits}", flush=True)
    index = build_video_index(mars_root, args.source_splits)
    video_id_to_pose = {vid["video_id"]: vid["pose_path"] for vid in index}
    video_id_to_split = {vid["video_id"]: vid["split"] for vid in index}
    indexed_counts = {split: 0 for split in args.source_splits}
    for vid in index:
        indexed_counts[vid["split"]] = indexed_counts.get(vid["split"], 0) + 1
    empty_indexed_splits = [s for s in args.source_splits if indexed_counts.get(s, 0) == 0]
    if empty_indexed_splits and not args.allow_missing_source_splits:
        raise SystemExit(
            f"No usable videos were indexed for requested MARS split(s): "
            f"{empty_indexed_splits}. Pass --allow-missing-source-splits for a partial run."
        )
    print(f"      {len(video_id_to_pose)} source videos available")
    print(f"      videos per MARS split: {indexed_counts}")

    missing = [c for c in clip_records if c["video_id"] not in video_id_to_pose]
    if missing:
        print(f"      [warn] {len(missing)} clips reference a video_id not found in MARS index. Skipping.")
        clip_records = [c for c in clip_records if c["video_id"] in video_id_to_pose]
    print(f"      {len(clip_records)} clips will be processed")

    # ---- Compute per-video body length & per-video pose cache ----
    print(f"[3/5] Loading pose JSON + body-length calibration ...", flush=True)
    pose_cache: dict[str, dict] = {}
    calibration_cache: dict[str, dict] = {}
    needed_video_ids = sorted({c["video_id"] for c in clip_records})
    for vid_id in needed_video_ids:
        with open(video_id_to_pose[vid_id], "r") as f:
            jpose = json.load(f)
        pose_cache[vid_id] = {
            "keypoints": np.asarray(jpose["keypoints"], dtype=np.float32),  # [F, M, coord, K]
            "bbox": np.asarray(jpose["bbox"], dtype=np.float32),            # [F, M, 4] (x1,y1,x2,y2 normalized)
            "scores": np.asarray(jpose["scores"], dtype=np.float32),        # [F, M, K]
            "bscores": np.asarray(jpose["bscores"], dtype=np.float32),      # [F, M]
        }
        calibration_cache[vid_id] = estimate_body_length_calibration(
            pose_cache[vid_id]["keypoints"],
            pose_cache[vid_id]["bscores"],
            bscore_threshold=float(args.bscore_threshold),
            min_valid_frames=int(args.body_length_min_valid_frames),
            fallback_px=float(args.body_length_fallback_px),
        )
    print(f"      cached pose for {len(pose_cache)} videos")

    # ---- Sample windows from each clip, emit NPZs ----
    print(f"[4/5] Emitting NPZs (mode={args.emit_mode}) ...", flush=True)
    samples = []   # list of dicts that will become manifest entries
    npz_root = out_root / "sequence_npz"
    n_animals = int(args.n_animals)

    for cidx, clip in enumerate(clip_records, 1):
        vid_id = clip["video_id"]
        cname = clip["class_name"]
        fs = clip["frame_start"]
        fe = clip["frame_end"]
        mp4_path = clip["mp4_path"]
        mars_split = video_id_to_split.get(vid_id)
        if mars_split is None:
            raise SystemExit(f"Internal error: no MARS split recorded for video_id={vid_id!r}")
        calibration = calibration_cache.get(vid_id, {
            "body_length_px": float(args.body_length_fallback_px),
            "body_length_source": "train_setup_fallback",
            "body_length_valid_frames": 0,
        })
        body_len = float(calibration["body_length_px"])

        frames_rgb = load_video_frames(mp4_path)
        T_clip = len(frames_rgb)
        if T_clip == 0:
            print(f"  [skip empty mp4] {mp4_path.name}")
            continue

        # The MARS pose arrays have shape [F_total, M, ...]; we index by (fs + t)
        pose = pose_cache[vid_id]
        pose_F = pose["keypoints"].shape[0]

        # Slide windows
        win = int(args.window_size)
        stride = int(args.window_stride)
        n_windows = max(0, (T_clip - win) // stride + 1) if T_clip >= win else 0
        if n_windows == 0:
            continue

        clip_dir = npz_root / cname
        clip_dir.mkdir(parents=True, exist_ok=True)

        for w_idx in range(n_windows):
            t_start = w_idx * stride
            t_end = t_start + win
            window_frames = frames_rgb[t_start:t_end]   # list of [H, W, 3] uint8
            if len(window_frames) < win:
                continue

            # Source frame indices in the MARS video for this window
            src_indices = [fs + t_start + i for i in range(win)]
            if any(s < 0 or s >= pose_F for s in src_indices):
                continue   # window runs off MARS source

            # Stack frame arrays: [T, H, W, 3]
            window_arr = np.stack(window_frames, axis=0)
            H_full, W_full = int(window_arr.shape[1]), int(window_arr.shape[2])

            if args.emit_mode == "single":
                # Emit one NPZ per mouse per window (legacy schema)
                for m in range(2):  # MARS has 2 mice; iterate both
                    bs_window = pose["bscores"][src_indices, m]   # [T]
                    if float(np.mean(bs_window)) < float(args.bscore_threshold):
                        continue   # skip windows where this mouse is mostly missing

                    # bbox per frame in original-pixel xyxy
                    bb_norm = pose["bbox"][src_indices, m, :]   # [T, 4] normalized to (W,H,W,H)
                    bbox_px = np.zeros_like(bb_norm)
                    bbox_px[:, 0] = bb_norm[:, 0] * W_full
                    bbox_px[:, 1] = bb_norm[:, 1] * H_full
                    bbox_px[:, 2] = bb_norm[:, 2] * W_full
                    bbox_px[:, 3] = bb_norm[:, 3] * H_full

                    # Build per-frame crop (expand bbox by pad_ratio, resize)
                    frames_out = np.zeros((win, args.crop_size, args.crop_size, 3), dtype=np.uint8)
                    kp_crop = np.zeros((win, NUM_KPTS, 3), dtype=np.float32)
                    kp_xy_raw = np.zeros((win, NUM_KPTS, 2), dtype=np.float32)
                    kp_conf_raw = np.zeros((win, NUM_KPTS), dtype=np.float32)
                    bbox_xyxy = np.zeros((win, 4), dtype=np.int32)
                    crop_xyxy = np.zeros((win, 4), dtype=np.int32)

                    for ti, src_t in enumerate(src_indices):
                        bb = bbox_px[ti]
                        crop = expand_bbox_xyxy(bb, args.pad_ratio, window_arr[ti].shape)
                        frames_out[ti] = crop_resize_pad_rgb(window_arr[ti], crop, args.crop_size)

                        # MARS pose layout: keypoints[F, mouse_id, coord, kpt]
                        # x = keypoints[F, m, 0, k], y = keypoints[F, m, 1, k]
                        kx = pose["keypoints"][src_t, m, 0, :]   # [K] x in original-pixel
                        ky = pose["keypoints"][src_t, m, 1, :]   # [K] y in original-pixel
                        kc = pose["scores"][src_t, m, :]         # [K] confidence
                        kxy_orig = np.stack([kx, ky], axis=1).astype(np.float32)
                        kp_xy_raw[ti] = kxy_orig
                        kp_conf_raw[ti] = kc.astype(np.float32)
                        kp_crop[ti] = project_keypoints_to_crop(kxy_orig, kc, crop, args.crop_size)
                        bbox_xyxy[ti] = bb.astype(np.int32)
                        crop_xyxy[ti] = np.array(crop, dtype=np.int32)

                    # Write NPZ in legacy single-animal format
                    sample_id = f"{cname}_{mp4_path.stem}_m{m}_w{w_idx:04d}"
                    npz_name = f"{mp4_path.stem}_m{m}_w{w_idx:04d}.npz"
                    npz_path = clip_dir / npz_name
                    savez_fn(
                        npz_path,
                        frames=frames_out,
                        keypoints=kp_crop,
                        keypoints_xy_raw=kp_xy_raw,
                        keypoints_conf_raw=kp_conf_raw,
                        bbox_xyxy=bbox_xyxy,
                        crop_xyxy=crop_xyxy,
                    )

                    samples.append({
                        "id": sample_id,
                        "label": int(class_to_idx[cname]),
                        "class_name": cname,
                        "mars_split": mars_split,
                        "sequence_npz": str(npz_path.relative_to(out_root)).replace("\\", "/"),
                        "frame_paths": [],
                        "start_frame": int(fs + t_start),
                        "end_frame": int(fs + t_start + win - 1),
                        "fps": 30.0,
                        "source_video": vid_id,
                        "source_clip": str(mp4_path.relative_to(clips_root)).replace("\\", "/"),
                        "clip_kind": clip["kind"],
                        "body_length_px": body_len,
                        "body_length_source": calibration["body_length_source"],
                        "body_length_valid_frames": int(calibration["body_length_valid_frames"]),
                    })

            else:
                # n_animal mode: one NPZ per window with [T, N, ...] arrays
                # Compute group + per-animal crops at constant pixel size.
                group_frames_out = np.zeros((win, args.crop_size, args.crop_size, 3), dtype=np.uint8)
                animal_frames_out = np.zeros((win, n_animals, args.crop_size, args.crop_size, 3), dtype=np.uint8)
                animal_kp = np.zeros((win, n_animals, NUM_KPTS, 3), dtype=np.float32)
                animal_kp_raw = np.zeros((win, n_animals, NUM_KPTS, 2), dtype=np.float32)
                animal_kp_conf = np.zeros((win, n_animals, NUM_KPTS), dtype=np.float32)
                animal_mask = np.zeros((win, n_animals), dtype=bool)
                pose_mask = np.zeros((win, n_animals), dtype=bool)
                pose_conf = np.zeros((win, n_animals), dtype=np.float32)
                crop_conf_arr = np.zeros((win, n_animals), dtype=np.float32)
                track_conf_arr = np.zeros((win, n_animals), dtype=np.float32)
                track_age_arr = np.zeros((win, n_animals), dtype=np.float32)
                bbox_xyxy_animals = np.zeros((win, n_animals, 4), dtype=np.int32)
                crop_xyxy_animals = np.zeros((win, n_animals, 4), dtype=np.int32)
                bbox_xyxy_group = np.zeros((win, 4), dtype=np.int32)
                crop_xyxy_group = np.zeros((win, 4), dtype=np.int32)
                track_ids_arr = np.tile(np.arange(n_animals, dtype=np.int32), (win, 1))

                # MARS only has 2 mice; pad with zeros if user requested more
                M_mars = pose["bbox"].shape[1]

                for ti, src_t in enumerate(src_indices):
                    centers = []
                    for m in range(min(n_animals, M_mars)):
                        bs = float(pose["bscores"][src_t, m])
                        bb_norm = pose["bbox"][src_t, m, :]
                        bb = np.array([
                            bb_norm[0] * W_full, bb_norm[1] * H_full,
                            bb_norm[2] * W_full, bb_norm[3] * H_full,
                        ], dtype=np.float32)
                        animal_mask[ti, m] = bs >= float(args.bscore_threshold)
                        bbox_xyxy_animals[ti, m] = bb.astype(np.int32)
                        track_conf_arr[ti, m] = bs
                        kx = pose["keypoints"][src_t, m, 0, :]
                        ky = pose["keypoints"][src_t, m, 1, :]
                        kc = pose["scores"][src_t, m, :]
                        kxy_orig = np.stack([kx, ky], axis=1).astype(np.float32)
                        animal_kp_raw[ti, m] = kxy_orig
                        animal_kp_conf[ti, m] = kc.astype(np.float32)
                        pose_conf[ti, m] = float(kc.mean())
                        pose_mask[ti, m] = bool(animal_mask[ti, m] and pose_conf[ti, m] >= float(args.pose_conf_threshold))
                        cx = (bb[0] + bb[2]) / 2.0
                        cy = (bb[1] + bb[3]) / 2.0
                        centers.append((cx, cy))

                    # group center = mean of present-animal centers
                    valid_centers = [c for c, m in zip(centers, animal_mask[ti]) if m]
                    if valid_centers:
                        gx = float(np.mean([c[0] for c in valid_centers]))
                        gy = float(np.mean([c[1] for c in valid_centers]))
                    else:
                        gx, gy = W_full / 2.0, H_full / 2.0

                    # group crop = fixed pixel size from body length
                    group_box = fixed_square_xyxy(window_arr[ti].shape, (gx, gy), body_len * args.group_scale_factor)
                    group_frames_out[ti] = crop_resize_pad_rgb(window_arr[ti], group_box, args.crop_size)
                    bbox_xyxy_group[ti] = np.array(group_box, dtype=np.int32)
                    crop_xyxy_group[ti] = bbox_xyxy_group[ti]

                    # per-animal crops
                    for m in range(min(n_animals, M_mars)):
                        cx, cy = centers[m]
                        a_box = fixed_square_xyxy(window_arr[ti].shape, (cx, cy), body_len * args.animal_scale_factor)
                        animal_frames_out[ti, m] = crop_resize_pad_rgb(window_arr[ti], a_box, args.crop_size)
                        crop_xyxy_animals[ti, m] = np.array(a_box, dtype=np.int32)
                        crop_conf_arr[ti, m] = crop_coverage(a_box, window_arr[ti].shape)
                        animal_kp[ti, m] = project_keypoints_to_crop(
                            animal_kp_raw[ti, m], animal_kp_conf[ti, m], a_box, args.crop_size
                        )

                # track_age = since first present in window, capped
                for m in range(n_animals):
                    first_seen = None
                    for ti in range(win):
                        if animal_mask[ti, m]:
                            if first_seen is None:
                                first_seen = ti
                            track_age_arr[ti, m] = min(float(ti - first_seen + 1) / 30.0, 1.0)

                # Relations (bbox-only fallback computed inline; pose half from MARS keypoints)
                rel_features = np.zeros((win, n_animals, n_animals, REL_FEATURE_DIM), dtype=np.float32)
                rel_pose_mask = np.zeros((win, n_animals, n_animals), dtype=bool)
                rel_present = np.zeros((win, n_animals, n_animals), dtype=bool)

                if extract_relational_features is not None:
                    prev_kxy = None
                    prev_bb = None
                    for ti in range(win):
                        rel, mp, pres = extract_relational_features(
                            animal_kp_raw[ti],
                            bbox_xyxy_animals[ti].astype(np.float32),
                            animal_mask[ti],
                            pose_mask[ti],
                            body_len,
                            prev_keypoints_raw=prev_kxy,
                            prev_bboxes=prev_bb,
                            fps=30.0,
                            pose_conf_threshold=float(args.pose_conf_threshold),
                            keypoints_conf_raw=animal_kp_conf[ti],
                        )
                        rel_features[ti] = rel
                        rel_pose_mask[ti] = mp
                        rel_present[ti] = pres
                        prev_kxy = animal_kp_raw[ti].copy()
                        prev_bb = bbox_xyxy_animals[ti].astype(np.float32).copy()

                relation_pose_conf = (pose_conf[:, :, None] * pose_conf[:, None, :]).astype(np.float32)

                sample_id = f"{cname}_{mp4_path.stem}_w{w_idx:04d}"
                npz_name = f"{mp4_path.stem}_w{w_idx:04d}.npz"
                npz_path = clip_dir / npz_name
                np.savez_compressed(
                    npz_path,
                    group_frames=group_frames_out,
                    animal_frames=animal_frames_out,
                    animal_keypoints=animal_kp,
                    animal_keypoints_raw=animal_kp_raw,
                    animal_keypoints_conf=animal_kp_conf,
                    animal_mask=animal_mask,
                    pose_mask=pose_mask,
                    pose_conf=pose_conf,
                    crop_conf=crop_conf_arr,
                    track_conf=track_conf_arr,
                    track_age=track_age_arr,
                    relation_features=rel_features,
                    relation_pose_mask=rel_pose_mask,
                    relation_pose_conf=relation_pose_conf,
                    relation_present=rel_present,
                    bbox_xyxy_animals=bbox_xyxy_animals,
                    bbox_xyxy_group=bbox_xyxy_group,
                    crop_xyxy_animals=crop_xyxy_animals,
                    crop_xyxy_group=crop_xyxy_group,
                    track_ids=track_ids_arr,
                    body_length_px=np.float32(body_len),
                    n_animals=np.int32(n_animals),
                    schema_version=np.asarray("behaviorscope-n-v1"),
                )

                samples.append({
                    "id": sample_id,
                    "label": int(class_to_idx[cname]),
                    "class_name": cname,
                    "mars_split": mars_split,
                    "sequence_npz": str(npz_path.relative_to(out_root)).replace("\\", "/"),
                    "frame_paths": [],
                    "start_frame": int(fs + t_start),
                    "end_frame": int(fs + t_start + win - 1),
                    "fps": 30.0,
                    "source_video": vid_id,
                    "source_clip": str(mp4_path.relative_to(clips_root)).replace("\\", "/"),
                    "clip_kind": clip["kind"],
                    "body_length_px": body_len,
                    "body_length_source": calibration["body_length_source"],
                    "body_length_valid_frames": int(calibration["body_length_valid_frames"]),
                })

        if cidx % 50 == 0:
            print(f"      [{cidx}/{len(clip_records)}] processed", flush=True)

    print(f"      total samples written: {len(samples)}")

    # ---- BehaviorScope-compatible splits, with native MARS provenance on samples ----
    print(f"[5/5] Writing manifest + calibration ...", flush=True)
    splits = {}
    for split_name in args.source_splits:
        manifest_split = MARS_TO_MANIFEST_SPLIT.get(split_name, split_name)
        splits.setdefault(manifest_split, [])
    for s in samples:
        mars_split = s.get("mars_split")
        manifest_split = MARS_TO_MANIFEST_SPLIT.get(mars_split, mars_split)
        if manifest_split not in splits:
            raise SystemExit(
                f"Sample {s.get('id', '<unknown>')} has mars_split={mars_split!r}, "
                f"which maps to manifest split {manifest_split!r}, not present in "
                f"--source-splits={args.source_splits!r}."
            )
        splits[manifest_split].append(s)
    for required_split in ("train", "val"):
        if required_split in splits and not splits[required_split]:
            raise SystemExit(
                f"Manifest split {required_split!r} produced zero samples; "
                "check --clips-root, class folders, and clip filenames."
            )

    manifest = {
        "class_to_idx": class_to_idx,
        "idx_to_class": {v: k for k, v in class_to_idx.items()},
        "splits": splits,
        "meta": {
            "window_size": int(args.window_size),
            "window_stride": int(args.window_stride),
            "crop_size": int(args.crop_size),
            "pad_ratio": float(args.pad_ratio),
            "emit_mode": args.emit_mode,
            "n_animals": int(args.n_animals) if args.emit_mode == "n_animal" else 1,
            "animal_scale_factor": float(args.animal_scale_factor),
            "group_scale_factor": float(args.group_scale_factor),
            "keypoints_per_animal": NUM_KPTS,
            "calibration_strategy": "per-video-mars-pose-median",
            "body_length_min_valid_frames": int(args.body_length_min_valid_frames),
            "body_length_fallback_px": float(args.body_length_fallback_px),
            "split_strategy": "native_mars",
            "native_mars_splits": list(args.source_splits),
            "manifest_split_mapping": {
                split_name: MARS_TO_MANIFEST_SPLIT.get(split_name, split_name)
                for split_name in args.source_splits
            },
            "train_split": "train",
            "validation_split": "val",
            "heldout_splits": [s for s in args.source_splits if s.startswith("test")],
            "has_keypoints": True,
            "schema_version": (
                "behaviorscope-n-v1" if args.emit_mode == "n_animal" else "behaviorscope-legacy"
            ),
            "source_dataset": "MARS (Caltech behavior annotation dataset)",
            "yolo_skipped": True,
        },
    }
    manifest_path = out_root / "sequence_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"      manifest -> {manifest_path}")

    calibration = {
        "version": "behaviorscope-n v1 (MARS-direct)",
        "calibration_inputs": "MARS pose JSON only - no behavior labels used",
        "calibration_strategy": "per-video-mars-pose-median",
        "body_length_min_valid_frames": int(args.body_length_min_valid_frames),
        "body_length_fallback_px": float(args.body_length_fallback_px),
        "per_video_calibration": [
            {"video_id": vid, "split": video_id_to_split.get(vid, "unknown"),
             "body_length_px": float(calibration_cache.get(vid, {}).get(
                 "body_length_px", float(args.body_length_fallback_px)
             )),
             "body_length_source": calibration_cache.get(vid, {}).get(
                 "body_length_source", "train_setup_fallback"
             ),
             "body_length_valid_frames": int(calibration_cache.get(vid, {}).get(
                 "body_length_valid_frames", 0
             ))}
            for vid in needed_video_ids
        ],
    }
    cal_path = out_root / "calibration.json"
    with open(cal_path, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"      calibration -> {cal_path}")

    # Summary
    counts_per_split = {k: len(v) for k, v in splits.items()}
    print(f"\nDone. samples per split: {counts_per_split}")
    print(f"      mode: {args.emit_mode}, schema_version: {manifest['meta']['schema_version']}")
    print()
    print("Train BehaviorScope-Y using native MARS train/validation splits:")
    print(f"  python BehaviorScope/train_y.py \\")
    print(f"      --manifest_path {manifest_path} \\")
    print(f"      --backbone slowfast_full")


if __name__ == "__main__":
    main()
