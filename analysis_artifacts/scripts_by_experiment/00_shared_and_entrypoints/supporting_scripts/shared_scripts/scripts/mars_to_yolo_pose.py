#!/usr/bin/env python3
"""
mars_to_yolo_pose.py
====================
Build a YOLO-pose dataset from the Caltech MARS behavior dataset.

Default classes (4):
    0 = investigation
    1 = mount
    2 = attack
    3 = other          (no-interaction / single-mouse / mice-apart)

The "other" class is drawn from frames OUTSIDE annotated bouts using three
combined strategies (each contributing roughly one third of the budget):

  (a) distance-stratified: both mice present, neck-to-neck distance is
      either > k * body_length (clearly apart) -- sub-pool A1 -- or random
      anywhere -- sub-pool A2.  We split this strategy 50/50 between A1 and A2.
  (b) single-mouse: exactly one mouse passes the bscore filter, so the
      label has only one row.  Teaches the model "one mouse alone = other".
  (c) plain random non-bout: pure random non-bout frames at the same
      intra-bout-stride (no distance filter).

Pass --no-other-class to revert to the legacy 3-class behavior.

Pipeline:
  * walks <mars_root>/{train,validation}/*  (or any --source-splits you pass)
  * parses BENTO .annot for each video to get bout intervals (.txt fallback)
  * stratified-samples ~total_frames/3 frames per behavior class
  * additionally samples --other-frames for the "other" class (default ~total/3)
  * decodes ONLY those frames from .seq (targeted seek; no full extraction)
  * uses the JSON bbox (already normalized) and 7 top-view keypoints with
    per-keypoint confidence as the visibility flag
  * writes 26-field YOLO-pose labels (one row per mouse, normally 2 rows
    per image; 1 row for "single-mouse other" frames)
  * splits the sampled frames 80/20 train/val, stratified by class
  * emits data.yaml ready for `yolo pose train`

Usage (Windows, IntegraPose conda env):
    python mars_to_yolo_pose.py ^
        --mars-root C:\\path\\to\\MARS_data ^
        --out-root  C:\\path\\to\\mars_yolo_pose ^
        --total-frames 24000 ^
        --val-frac 0.2 ^
        --seed 0 ^
        --abs-paths-in-yaml

Notes:
  * Pose JSON axis order is [F, mouse_id, coord, kpt] with coord 0=x, 1=y.
  * BBox in JSON is (x_min, y_min, x_max, y_max) normalized to [W, H, W, H].
  * Visibility (3-state) from per-keypoint score:
        v=2 if score >= 0.50 and point in image
        v=1 if 0.20 <= score < 0.50 and point in image
        v=0 otherwise   (forces x=y=0 for that keypoint)
  * A mouse row is dropped if bscore < 0.50 OR fewer than 3 visible keypoints.
  * Image is dropped if both mouse rows are dropped.
  * fliplr/flipud disabled in YAML — ear_1/ear_2 may be ID-specific.
"""

import argparse
import csv
import glob
import io
import json
import os
import random
import re
import struct
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ----------------------------------------------------------------------------
# Class configuration
# ----------------------------------------------------------------------------

BEHAVIOR_NAMES = ["investigation", "mount", "attack"]
OTHER_NAME = "other"

# Default = 4 classes with "other" appended at the END so existing class IDs
# (0=investigation, 1=mount, 2=attack) stay stable. main() rebuilds these
# based on --no-other-class.
CLASS_NAMES = BEHAVIOR_NAMES + [OTHER_NAME]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

KPT_NAMES = ["nose", "ear_1", "ear_2", "neck", "hip_1", "hip_2", "tail_base"]
NUM_KPTS = len(KPT_NAMES)
NECK_KPT_INDEX = 3
TAIL_KPT_INDEX = 6

VIS_HIGH = 0.50  # >= -> v=2
VIS_LOW = 0.20   # >= -> v=1, else 0
BSCORE_MIN = 0.50
MIN_VIS_KPTS = 3

OTHER_DISTANCE_BODYLENGTHS_DEFAULT = 2.0
DISTANCE_STRATEGY_FAR_FRAC = 0.5

RAW_TO_BEHAVIOR = {
    "investigation": "investigation",
    "closeinvestigation": "investigation",
    "closeinvestigate": "investigation",
    "close_investigation": "investigation",
    "social_investigation": "investigation",
    "sniff": "investigation",
    "sniffing": "investigation",
    "mount": "mount",
    "mounting": "mount",
    "aggressivemount": "mount",
    "attack": "attack",
    "aggression": "attack",
    "aggressive": "attack",
}


# ----------------------------------------------------------------------------
# .seq reader (NorPix / StreamPix; raw 8-bit gray or JPEG-compressed)
# ----------------------------------------------------------------------------

FRAME_FORMAT_RAW_GRAY = 100
FRAME_FORMAT_JPEG_GRAY = 102
FRAME_FORMAT_RAW_COLOR = 200
FRAME_FORMAT_JPEG_COLOR = 201


class SeqReader:
    def __init__(self, filename):
        self.filename = str(filename)
        self.file = open(self.filename, "rb")
        self.header = {}
        self.timestamp_length = 10
        self.seek_table = None
        self._walk_retried = False
        self._parse_header()

    def _parse_header(self):
        data = self.file.read(1024)
        (
            self.header["magic_number"],
            self.header["name"],
            self.header["version"],
            self.header["header_size"],
            self.header["description"],
            self.header["image_width"],
            self.header["image_height"],
            self.header["bit_depth"],
            self.header["bit_depth_real"],
            self.header["image_size"],
            self.header["image_format"],
            self.header["allocated_frames"],
            self.header["origin"],
            self.header["true_image_size"],
            self.header["frame_rate"],
            self.header["description_format"],
            self.header["padding"],
        ) = struct.unpack("i24sii512sIIIIIiIIIdi428s", data)
        self.image_format = self.header["image_format"]
        if self.header["image_format"] == 101:
            self.header["image_format"] = 102
        self.bit_depth = self.header["bit_depth"]
        self.compressed = self.image_format in (FRAME_FORMAT_JPEG_GRAY, FRAME_FORMAT_JPEG_COLOR)
        if not self.compressed:
            expected = self.bit_depth / 8 * (self.header["image_height"] * self.header["image_width"]) + self.timestamp_length
            if expected != self.header["true_image_size"]:
                self.timestamp_length = int(
                    self.header["true_image_size"]
                    - (self.bit_depth / 8 * (self.header["image_height"] * self.header["image_width"]))
                )
                if self.timestamp_length < 0 or self.timestamp_length > 4096:
                    self.timestamp_length = 10
        else:
            detected = self._detect_timestamp_length_compressed()
            if detected is not None:
                self.timestamp_length = detected

    @property
    def num_frames(self): return self.header["allocated_frames"]
    @property
    def width(self): return self.header["image_width"]
    @property
    def height(self): return self.header["image_height"]

    def _detect_timestamp_length_compressed(self):
        if not self.compressed:
            return None
        try:
            self.file.seek(1024)
            sb = self.file.read(4)
            if len(sb) < 4:
                return None
            size = struct.unpack("i", sb)[0]
            if size <= 4:
                return None
            self.file.seek(1024 + 4)
            window = self.file.read(min(size + 4096, 16 * 1024 * 1024))
            jpeg_region = window[: size - 4 + 256]
            eoi_idx = jpeg_region.rfind(b"\xff\xd9")
            if eoi_idx < 0:
                return None
            soi_idx = window.find(b"\xff\xd8", eoi_idx + 2)
            if soi_idx < 0:
                return None
            ts = (soi_idx - (eoi_idx + 2)) - 4
            if 0 <= ts <= 4096:
                return ts
            return None
        except Exception:
            return None

    def _file_size(self):
        try:
            cur = self.file.tell()
            self.file.seek(0, 2)
            sz = self.file.tell()
            self.file.seek(cur, 0)
            return sz
        except Exception:
            return None

    def _validate_seek_table(self, table):
        if not table:
            return False
        offset, size = table[0]
        try:
            file_size = self._file_size()
            if offset < 0 or size <= 0:
                return False
            if file_size is not None and offset >= file_size:
                return False
            self.file.seek(offset)
            head = self.file.read(min(4, size))
            if self.compressed:
                return len(head) >= 2 and head[0] == 0xFF and head[1] == 0xD8
            return True
        except Exception:
            return False

    def _build_table_from_mat(self, seek_mat_path):
        try:
            import scipy.io as sio
            seek = sio.loadmat(seek_mat_path)["seek"].ravel().astype(np.int64)
        except Exception as e:
            print(f"  [seek mat load failed for {os.path.basename(seek_mat_path)}: {e}]")
            return None
        table = []
        try:
            for i, off in enumerate(seek):
                off = int(off)
                if i + 1 < len(seek):
                    size = int(seek[i + 1] - off - 4 - self.timestamp_length)
                else:
                    self.file.seek(off)
                    sb = self.file.read(4)
                    size = struct.unpack("i", sb)[0] - 4 if len(sb) == 4 else 0
                table.append((off + 4, size))
        except Exception as e:
            print(f"  [seek mat build failed: {e}]")
            return None
        return table

    def _build_table_from_walk(self):
        table = []
        file_size = self._file_size()
        if self.compressed:
            try:
                self.file.seek(1024, 0)
            except Exception:
                return table
            while True:
                size_bytes = self.file.read(4)
                if not size_bytes or len(size_bytes) < 4:
                    break
                try:
                    size = struct.unpack("i", size_bytes)[0]
                except Exception:
                    break
                if size <= 4:
                    break
                if file_size is not None and size > file_size:
                    break
                offset = self.file.tell()
                advance = size - 4 + self.timestamp_length
                if advance <= 0:
                    break
                if file_size is not None and offset + advance > file_size:
                    break
                try:
                    self.file.seek(advance, 1)
                except OSError:
                    break
                table.append((offset, size))
        else:
            for f in range(self.num_frames):
                offset = f * self.header["true_image_size"] + 1024
                table.append((offset, self.header["image_size"]))
        return table

    def build_seek_table(self, seek_mat_path=None):
        if self.seek_table is not None:
            return
        mat_table = None
        if seek_mat_path is not None and os.path.isfile(seek_mat_path):
            mat_table = self._build_table_from_mat(seek_mat_path)
            if mat_table is not None and self._validate_seek_table(mat_table):
                self.seek_table = mat_table
                return
            elif mat_table is not None:
                print(f"  [seek mat for {os.path.basename(seek_mat_path)} failed validation; walking file instead]")
        walk_table = self._build_table_from_walk()
        if walk_table and self._validate_seek_table(walk_table):
            self.seek_table = walk_table
            return
        self.seek_table = walk_table or mat_table or []

    def _decode_at(self, offset, size):
        self.file.seek(offset)
        data = self.file.read(size + self.timestamp_length)
        if self.compressed:
            jpg_bytes = data[:-self.timestamp_length] if self.timestamp_length > 0 else data
            try:
                im = Image.open(io.BytesIO(jpg_bytes))
                im.load()
                return np.array(im)
            except Exception as pil_exc:
                arr = np.frombuffer(jpg_bytes, dtype=np.uint8)
                bgr = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if bgr is None:
                    raise pil_exc
                if bgr.ndim == 3:
                    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                return bgr
        if self.image_format == FRAME_FORMAT_RAW_GRAY and self.bit_depth == 8:
            arr = np.frombuffer(data[:-self.timestamp_length], dtype=np.uint8)
            return arr.reshape(self.header["image_height"], self.header["image_width"])
        raise NotImplementedError(f"Unsupported .seq format: image_format={self.image_format} bit_depth={self.bit_depth}")

    def read_frame(self, index):
        if self.seek_table is None:
            self.build_seek_table()
        if index < 0 or index >= len(self.seek_table):
            raise IndexError(f"frame {index} out of range (n={len(self.seek_table)})")
        try:
            offset, size = self.seek_table[index]
            return self._decode_at(offset, size)
        except Exception:
            if not self._walk_retried:
                self._walk_retried = True
                walk = self._build_table_from_walk()
                if walk and self._validate_seek_table(walk):
                    self.seek_table = walk
                    if index < len(walk):
                        offset, size = walk[index]
                        return self._decode_at(offset, size)
            raise

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


# ----------------------------------------------------------------------------
# Annotation parsers
# ----------------------------------------------------------------------------

def parse_bento_annot(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    framerate = 30.0
    for ln in lines:
        m = re.match(r"\s*Annotation framerate:\s*([0-9.eE+\-]+)", ln)
        if m:
            try:
                framerate = float(m.group(1))
            except Exception:
                pass
            break
    behaviors = defaultdict(list)
    cur = None
    in_data = False
    for ln in lines:
        s = ln.strip()
        if s.startswith(">"):
            cur = s[1:].strip().lower()
            in_data = False
            continue
        if cur is None:
            continue
        if s.startswith("Start"):
            in_data = True
            continue
        if not in_data:
            continue
        if s == "":
            in_data = False
            cur = None
            continue
        parts = s.split()
        if len(parts) < 2:
            continue
        try:
            start = float(parts[0]); stop = float(parts[1])
        except ValueError:
            continue
        f_start = int(round(start * framerate))
        f_stop = int(round(stop * framerate))
        if f_stop < f_start:
            f_start, f_stop = f_stop, f_start
        behaviors[cur].append((f_start, f_stop))
    return dict(behaviors)


def parse_caltech_txt(path):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    behaviors = defaultdict(list)
    in_data = False
    for ln in lines:
        s = ln.strip()
        if s.startswith("-----"):
            in_data = True
            continue
        if not in_data:
            continue
        if s == "":
            in_data = False
            continue
        parts = s.split()
        if len(parts) < 3:
            continue
        try:
            a = int(parts[0]); b = int(parts[1])
        except ValueError:
            continue
        name = parts[2].lower()
        behaviors[name].append((a - 1, b - 1))
    return dict(behaviors)


def normalize_behavior_dict(raw_beh):
    out = defaultdict(list)
    for raw_name, bouts in raw_beh.items():
        canonical = RAW_TO_BEHAVIOR.get(raw_name)
        if canonical is None:
            continue
        out[canonical].extend(bouts)
    return dict(out)


# ----------------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------------

def find_top_seq(folder):
    cands = sorted(glob.glob(os.path.join(folder, "*_Top.seq")))
    cands += sorted(glob.glob(os.path.join(folder, "*_Top_J85.seq")))
    cands += [p for p in sorted(glob.glob(os.path.join(folder, "*Top*.seq"))) if p not in cands]
    cands = [p for p in cands if "Front" not in os.path.basename(p)]
    return cands[0] if cands else None


def find_top_pose_json(folder):
    v18 = sorted(glob.glob(os.path.join(folder, "*_pose_top_v1_8.json")))
    v17 = sorted(glob.glob(os.path.join(folder, "*_pose_top_v1_7.json")))
    other = [p for p in sorted(glob.glob(os.path.join(folder, "*pose*top*.json")))
             if p not in v18 and p not in v17]
    cands = v18 + v17 + other
    cands = [p for p in cands if "front" not in os.path.basename(p).lower()]
    return cands[0] if cands else None


def find_annotation_files(folder):
    cands = sorted(glob.glob(os.path.join(folder, "*.annot"))) + sorted(glob.glob(os.path.join(folder, "*.txt")))
    cands = [p for p in cands
             if "pred" not in os.path.basename(p).lower()
             and "front" not in os.path.basename(p).lower()]
    return cands


def select_primary_annotation(folder):
    cands = find_annotation_files(folder)
    if not cands:
        return None, None
    best = None
    best_score = (-1, -1)
    for p in cands:
        try:
            raw = parse_bento_annot(p) if p.endswith(".annot") else parse_caltech_txt(p)
        except Exception:
            continue
        canonical = normalize_behavior_dict(raw)
        n_classes = sum(1 for k in BEHAVIOR_NAMES if canonical.get(k))
        n_bouts = sum(len(v) for v in canonical.values())
        score = (n_classes, n_bouts)
        if score > best_score:
            best_score = score
            best = (p, canonical)
    return best if best else (None, None)


def build_video_index(mars_root, source_splits):
    out = []
    for split in source_splits:
        split_dir = os.path.join(mars_root, split)
        if not os.path.isdir(split_dir):
            continue
        for d in sorted(os.listdir(split_dir)):
            folder = os.path.join(split_dir, d)
            if not os.path.isdir(folder):
                continue
            seq_p = find_top_seq(folder)
            pose_p = find_top_pose_json(folder)
            annot_p, beh = select_primary_annotation(folder)
            if seq_p is None or pose_p is None or annot_p is None:
                print(f"  [skip] {split}/{d}  seq={bool(seq_p)} pose={bool(pose_p)} annot={bool(annot_p)}")
                continue
            out.append({
                "video_id": d, "split": split, "folder": folder,
                "seq_path": seq_p, "pose_path": pose_p,
                "annot_path": annot_p, "behaviors": beh,
            })
    return out


# ----------------------------------------------------------------------------
# Sampling: behavior classes
# ----------------------------------------------------------------------------

def collect_candidate_frames(index, intra_bout_stride):
    candidates = []
    for vidx, vid in enumerate(index):
        try:
            with open(vid["pose_path"], "r") as f:
                jpose = json.load(f)
            bscores = np.asarray(jpose["bscores"], dtype=np.float32)
            del jpose
        except Exception as e:
            print(f"  [skip {vid['video_id']}] pose load error: {e}")
            continue
        n_pose = bscores.shape[0]
        good = (bscores[:, 0] >= BSCORE_MIN) & (bscores[:, 1] >= BSCORE_MIN)
        for cname in BEHAVIOR_NAMES:
            for (fs, fe) in vid["behaviors"].get(cname, []):
                fs = max(0, fs); fe = min(n_pose - 1, fe)
                if fe < fs:
                    continue
                for f in range(fs, fe + 1, intra_bout_stride):
                    if 0 <= f < n_pose and good[f]:
                        candidates.append((vidx, f, cname))
    return candidates


def stratified_sample(candidates, total_frames, rng, class_list=None):
    if class_list is None:
        class_list = BEHAVIOR_NAMES
    by_class = defaultdict(list)
    for c in candidates:
        by_class[c[2]].append(c)
    target = max(1, total_frames // len(class_list))
    picked = []
    for cname in class_list:
        pool = by_class.get(cname, [])
        if not pool:
            print(f"  [warn] no candidates for class '{cname}'")
            continue
        if len(pool) <= target:
            picked.extend(pool)
            print(f"  class '{cname}': pool={len(pool)} <= target={target}, taking all")
        else:
            picked.extend(rng.sample(pool, target))
            print(f"  class '{cname}': pool={len(pool)} target={target} sampled={target}")
    rng.shuffle(picked)
    return picked


def split_train_val(picked, val_frac, rng):
    by_class = defaultdict(list)
    for c in picked:
        by_class[c[2]].append(c)
    train, val = [], []
    for cname, items in by_class.items():
        rng.shuffle(items)
        n_val = int(round(len(items) * val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train); rng.shuffle(val)
    return train, val


# ----------------------------------------------------------------------------
# Sampling: 'other' class
# ----------------------------------------------------------------------------

def estimate_body_length_pixels(kp, bscores, mouse_id=0):
    """Median neck-to-tail distance for one mouse over good-pose frames."""
    good = bscores[:, mouse_id] >= BSCORE_MIN
    if good.sum() < 100:
        return None
    neck = kp[good, mouse_id, :, NECK_KPT_INDEX]
    tail = kp[good, mouse_id, :, TAIL_KPT_INDEX]
    d = np.linalg.norm(neck - tail, axis=-1)
    d = d[np.isfinite(d)]
    if len(d) < 100:
        return None
    return float(np.median(d))


def build_in_bout_mask(behaviors_dict, n_frames):
    """Boolean mask of frames inside ANY annotated behavior bout."""
    in_bout = np.zeros(n_frames, dtype=bool)
    for cname in BEHAVIOR_NAMES:
        for fs, fe in behaviors_dict.get(cname, []):
            in_bout[max(0, fs):min(n_frames, fe + 1)] = True
    return in_bout


def collect_other_candidates(index, intra_bout_stride, distance_bodylengths):
    """Return dict of four sublists by strategy.

    The 'plain_random' bucket contains every two-mouse non-bout frame
    irrespective of distance, and is used as the source for strategy (c).
    The 'distance_apart' and 'near_random' buckets are used for the 50/50
    distance-stratified strategy (a). The 'single_mouse' bucket is strategy
    (b). Sampling deduplicates across strategies in `sample_other_combined`.
    """
    out = {"distance_apart": [], "near_random": [], "single_mouse": [], "plain_random": []}
    for vidx, vid in enumerate(index):
        try:
            with open(vid["pose_path"], "r") as f:
                jpose = json.load(f)
            kp = np.asarray(jpose["keypoints"], dtype=np.float32)
            bscores = np.asarray(jpose["bscores"], dtype=np.float32)
            del jpose
        except Exception as e:
            print(f"  [skip {vid['video_id']}] pose load error: {e}")
            continue
        n = bscores.shape[0]
        if kp.shape[0] != n or kp.ndim != 4:
            continue
        in_bout = build_in_bout_mask(vid["behaviors"], n)
        non_bout = ~in_bout
        good0 = bscores[:, 0] >= BSCORE_MIN
        good1 = bscores[:, 1] >= BSCORE_MIN
        both_good = good0 & good1
        one_good = good0 ^ good1
        body_len = estimate_body_length_pixels(kp, bscores)
        if body_len is None or body_len < 10:
            far_mask = np.zeros(n, dtype=bool)
        else:
            neck0 = kp[:, 0, :, NECK_KPT_INDEX]
            neck1 = kp[:, 1, :, NECK_KPT_INDEX]
            d = np.linalg.norm(neck0 - neck1, axis=-1)
            far_mask = d > (distance_bodylengths * body_len)
        for f in range(0, n, intra_bout_stride):
            if not non_bout[f]:
                continue
            if both_good[f]:
                if far_mask[f]:
                    out["distance_apart"].append((vidx, f, OTHER_NAME))
                else:
                    out["near_random"].append((vidx, f, OTHER_NAME))
                out["plain_random"].append((vidx, f, OTHER_NAME))
            elif one_good[f]:
                out["single_mouse"].append((vidx, f, OTHER_NAME))
    return out


def sample_other_combined(by_strategy, total_other, rng):
    """Take total_other/3 each from three top-level strategies:
        1) distance-stratified  (50% from distance_apart, 50% from near_random)
        2) single-mouse
        3) plain random non-bout
    Avoids duplicates across strategies."""
    per = max(1, total_other // 3)
    picked_keys = set()
    out = []

    # Strategy 1: distance-stratified (50/50 split)
    n_far = max(1, int(round(per * DISTANCE_STRATEGY_FAR_FRAC)))
    n_near = per - n_far
    far_pool = list(by_strategy.get("distance_apart", []))
    near_pool = list(by_strategy.get("near_random", []))
    rng.shuffle(far_pool); rng.shuffle(near_pool)
    s1_far = 0
    for c in far_pool:
        if s1_far >= n_far: break
        key = (c[0], c[1])
        if key in picked_keys: continue
        picked_keys.add(key); out.append(c); s1_far += 1
    s1_near = 0
    for c in near_pool:
        if s1_near >= n_near: break
        key = (c[0], c[1])
        if key in picked_keys: continue
        picked_keys.add(key); out.append(c); s1_near += 1

    # Strategy 2: single-mouse
    sm_pool = list(by_strategy.get("single_mouse", []))
    rng.shuffle(sm_pool)
    s2_taken = 0
    for c in sm_pool:
        if s2_taken >= per: break
        key = (c[0], c[1])
        if key in picked_keys: continue
        picked_keys.add(key); out.append(c); s2_taken += 1

    # Strategy 3: plain random non-bout
    pr_pool = list(by_strategy.get("plain_random", []))
    rng.shuffle(pr_pool)
    s3_taken = 0
    for c in pr_pool:
        if s3_taken >= per: break
        key = (c[0], c[1])
        if key in picked_keys: continue
        picked_keys.add(key); out.append(c); s3_taken += 1

    print(f"  other strategy taken: distance_apart={s1_far}/{n_far}  "
          f"near_random(in distance-strategy)={s1_near}/{n_near}  "
          f"single_mouse={s2_taken}/{per}  plain_random={s3_taken}/{per}")
    rng.shuffle(out)
    return out


# ----------------------------------------------------------------------------
# Label building
# ----------------------------------------------------------------------------

def visibility_from_score(score, x, y, W, H):
    if not (np.isfinite(x) and np.isfinite(y)):
        return 0
    if not (0 <= x < W and 0 <= y < H):
        return 0
    if score >= VIS_HIGH:
        return 2
    if score >= VIS_LOW:
        return 1
    return 0


def build_yolo_rows(jpose, frame_idx, class_id, W, H):
    kp = np.asarray(jpose["keypoints"])
    bb = np.asarray(jpose["bbox"])
    sc = np.asarray(jpose["scores"])
    bs = np.asarray(jpose["bscores"])
    rows = []
    for m in (0, 1):
        if bs[frame_idx, m] < BSCORE_MIN:
            continue
        x1, y1, x2, y2 = bb[frame_idx, m]
        x1 = float(np.clip(x1, 0.0, 1.0)); y1 = float(np.clip(y1, 0.0, 1.0))
        x2 = float(np.clip(x2, 0.0, 1.0)); y2 = float(np.clip(y2, 0.0, 1.0))
        if x2 <= x1 or y2 <= y1:
            continue
        bxc = (x1 + x2) / 2.0; byc = (y1 + y2) / 2.0
        bw = x2 - x1; bh = y2 - y1
        n_vis = 0
        kp_strs = []
        for k in range(NUM_KPTS):
            x_px = float(kp[frame_idx, m, 0, k])
            y_px = float(kp[frame_idx, m, 1, k])
            score = float(sc[frame_idx, m, k])
            v = visibility_from_score(score, x_px, y_px, W, H)
            if v > 0:
                xn = float(np.clip(x_px / W, 0.0, 1.0))
                yn = float(np.clip(y_px / H, 0.0, 1.0))
                n_vis += 1
            else:
                xn, yn = 0.0, 0.0
            kp_strs.append(f"{xn:.6f} {yn:.6f} {v}")
        if n_vis < MIN_VIS_KPTS:
            continue
        rows.append(f"{class_id} {bxc:.6f} {byc:.6f} {bw:.6f} {bh:.6f} " + " ".join(kp_strs))
    return rows


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------

def basename_for(video_id, frame_idx):
    safe_vid = re.sub(r"[^A-Za-z0-9._\-#]", "_", video_id)
    return f"{safe_vid}__f{frame_idx:06d}"


def save_frame(arr, out_path, jpeg_quality=95):
    if arr.ndim == 2:
        im = Image.fromarray(arr, mode="L")
    else:
        im = Image.fromarray(arr)
    im.save(out_path, format="JPEG", quality=jpeg_quality)


def write_yaml(out_root, abs_paths, class_names):
    yaml_path = Path(out_root) / "data.yaml"
    path_str = str(Path(out_root).resolve()) if abs_paths else str(Path(out_root))
    lines = [
        f"path: {path_str}",
        f"train: images/train",
        f"val: images/val",
        f"",
        f"kpt_shape: [{NUM_KPTS}, 3]",
        f"# Initial training: do NOT flip; ear_1/ear_2 may be ID-specific.",
        f"# fliplr: 0.0",
        f"# flipud: 0.0",
        f"",
        f"names:",
    ]
    for i, n in enumerate(class_names):
        lines.append(f"  {i}: {n}")
    yaml_path.write_text("\n".join(lines) + "\n")
    return yaml_path


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Build a YOLO-pose dataset from MARS behavior data.")
    p.add_argument("--mars-root", required=True)
    p.add_argument("--out-root", required=True)
    p.add_argument("--total-frames", type=int, default=5000,
                   help="Target frames for the BEHAVIOR classes (split equally across investigation/mount/attack).")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--source-splits", nargs="+", default=["train", "validation"])
    p.add_argument("--intra-bout-stride", type=int, default=6,
                   help="Within a bout (and within non-bout regions), sample every Nth frame.")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--abs-paths-in-yaml", action="store_true")
    p.add_argument("--max-videos", type=int, default=0)
    p.add_argument("--no-other-class", action="store_true",
                   help="Disable the 'other' (no-interaction) class -- legacy 3-class build.")
    p.add_argument("--other-frames", type=int, default=None,
                   help="Total 'other' frames. Default: total_frames // 3.")
    p.add_argument("--other-distance-bodylengths", type=float, default=OTHER_DISTANCE_BODYLENGTHS_DEFAULT,
                   help="Distance threshold (in body lengths) for 'clearly apart'.")
    args = p.parse_args()

    global CLASS_NAMES, CLASS_TO_ID
    if args.no_other_class:
        CLASS_NAMES = list(BEHAVIOR_NAMES)
    else:
        CLASS_NAMES = list(BEHAVIOR_NAMES) + [OTHER_NAME]
    CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    mars_root = os.path.abspath(args.mars_root)
    out_root = Path(args.out_root)
    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Indexing videos in: {args.source_splits}", flush=True)
    index = build_video_index(mars_root, args.source_splits)
    if args.max_videos and args.max_videos > 0:
        index = index[: args.max_videos]
        print(f"      [debug] limiting to first {len(index)} videos", flush=True)
    print(f"      indexed {len(index)} videos", flush=True)

    print(f"[2/5] Collecting BEHAVIOR candidate frames (intra-bout stride = {args.intra_bout_stride}) ...", flush=True)
    candidates = collect_candidate_frames(index, args.intra_bout_stride)
    by_class_count = defaultdict(int)
    for c in candidates:
        by_class_count[c[2]] += 1
    print("      candidate counts per behavior class:", flush=True)
    for cname in BEHAVIOR_NAMES:
        print(f"        {cname:14s} {by_class_count.get(cname, 0)}", flush=True)

    other_candidates = []
    if not args.no_other_class:
        print(f"[2.5/5] Collecting 'other' candidates ...", flush=True)
        other_by_strategy = collect_other_candidates(
            index, args.intra_bout_stride, args.other_distance_bodylengths,
        )
        for k, v in other_by_strategy.items():
            print(f"        pool '{k}': {len(v)} frames", flush=True)
        if args.other_frames is not None:
            other_target = args.other_frames
        else:
            other_target = max(1, args.total_frames // len(BEHAVIOR_NAMES))
        print(f"        'other' target: {other_target} frames", flush=True)
        other_candidates = sample_other_combined(other_by_strategy, other_target, rng)
        print(f"        'other' picked: {len(other_candidates)} frames", flush=True)

    print(f"[3/5] Stratified sampling toward total_frames={args.total_frames} (behaviors only) ...", flush=True)
    picked = stratified_sample(candidates, args.total_frames, rng, class_list=BEHAVIOR_NAMES)
    print(f"      behavior frames picked: {len(picked)}", flush=True)
    if other_candidates:
        picked = picked + other_candidates
        rng.shuffle(picked)
        print(f"      grand total picked (behaviors + other): {len(picked)} frames", flush=True)

    print(f"[4/5] Splitting 80/20 (val_frac={args.val_frac}) ...", flush=True)
    train_set, val_set = split_train_val(picked, args.val_frac, rng)
    print(f"      train={len(train_set)}  val={len(val_set)}", flush=True)

    by_video_train = defaultdict(list)
    for vidx, f, c in train_set:
        by_video_train[vidx].append((f, c))
    by_video_val = defaultdict(list)
    for vidx, f, c in val_set:
        by_video_val[vidx].append((f, c))

    manifest_rows = []
    n_written = 0
    n_skipped = 0
    pose_cache = {}

    for split_name, by_video in (("train", by_video_train), ("val", by_video_val)):
        for vidx, items in by_video.items():
            vid = index[vidx]
            if vidx in pose_cache:
                jpose = pose_cache[vidx]
            else:
                with open(vid["pose_path"], "r") as f:
                    jpose = json.load(f)
                pose_cache[vidx] = jpose
            try:
                seq = SeqReader(vid["seq_path"])
                seek_mat = vid["seq_path"].replace(".seq", "-seek.mat")
                seq.build_seek_table(seek_mat_path=seek_mat if os.path.isfile(seek_mat) else None)
            except Exception as e:
                print(f"      [seq error] {vid['video_id']}: {e}", flush=True)
                continue
            W, H = seq.width, seq.height
            items.sort(key=lambda x: x[0])
            for (f_idx, cname) in items:
                cls_id = CLASS_TO_ID[cname]
                rows = build_yolo_rows(jpose, f_idx, cls_id, W, H)
                if not rows:
                    n_skipped += 1
                    continue
                try:
                    arr = seq.read_frame(f_idx)
                except Exception as e:
                    print(f"      [decode error] {vid['video_id']} f={f_idx}: {e}", flush=True)
                    n_skipped += 1
                    continue
                base = basename_for(vid["video_id"], f_idx)
                img_path = out_root / "images" / split_name / f"{base}.jpg"
                lbl_path = out_root / "labels" / split_name / f"{base}.txt"
                save_frame(arr, img_path, jpeg_quality=args.jpeg_quality)
                lbl_path.write_text("\n".join(rows) + "\n")
                manifest_rows.append({
                    "split": split_name,
                    "video_id": vid["video_id"],
                    "frame_index": f_idx,
                    "class": cname,
                    "class_id": cls_id,
                    "n_mouse_rows": len(rows),
                    "image": str(img_path),
                    "label": str(lbl_path),
                })
                n_written += 1
            seq.close()

    manifest_path = out_root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "split", "video_id", "frame_index", "class", "class_id",
            "n_mouse_rows", "image", "label",
        ])
        w.writeheader()
        w.writerows(manifest_rows)

    yaml_path = write_yaml(out_root, abs_paths=args.abs_paths_in_yaml, class_names=CLASS_NAMES)

    print(f"[5/5] Done.", flush=True)
    print(f"      wrote {n_written} frame/label pairs (skipped {n_skipped})", flush=True)
    print(f"      class IDs: " + ", ".join(f"{i}={n}" for i, n in enumerate(CLASS_NAMES)), flush=True)
    print(f"      manifest: {manifest_path}", flush=True)
    print(f"      data.yaml: {yaml_path}", flush=True)
    print()
    print("Train (in IntegraPose conda env):")
    print(f"  yolo pose train model=yolo11n-pose.pt data={yaml_path} imgsz=640 epochs=100 \\")
    print(f"      batch=16 fliplr=0.0 flipud=0.0 project=runs/mars_pose name=v1")


if __name__ == "__main__":
    main()
