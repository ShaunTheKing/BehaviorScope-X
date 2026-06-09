"""
fly_to_behaviorscope_npz_full.py
================================
Build BehaviorScope-N NPZ training data from Caltech Fly-v-Fly Aggression
videos using a domain-adapted YOLO-pose model for detection and pose
estimation â€” the same pipeline the model will see at deployment.

Architecture follows ``prepare_full_video_npz.py`` (the MARS full-video
builder): streams YOLO-pose detections via ``iterate_yolo_n_crops()``, emits
sliding windows with O(window_size) memory, writes NPZs asynchronously.

Three Fly-v-Fly-specific replacements vs the MARS reference:
  1. **Discovery** â€” movies found via traintest.mat, not MARS split dirs.
  2. **Per-frame labels** â€” parsed from *_actions.mat bout tables, not BENTO
     .annot files.
  3. **Split assignment** â€” official train IDs [1-5] split into train/val via
     ``--val_movie_ids``; test IDs [6-10] excluded entirely (leakage guard).

Window labeling uses **priority-OR**: if *any* frame in the window carries a
non-other behavior, the window is labeled as the highest-priority behavior
present. This is necessary because Fly-v-Fly bouts are very short (3-5
frames) while the default window is 30 frames; dominance-based labeling would
drop virtually all behavior windows.

Classes (6):
    0 = lunge, 1 = wing_threat, 2 = charge, 3 = hold, 4 = tussle, 5 = other

Emits behaviorscope-n-v1 NPZs compatible with data_x.py / train_x.py.

Usage (Windows, IntegraPose conda env):
    python fly_to_behaviorscope_npz_full.py ^
        --aggression_root path\\to\\Fly_v_Fly_2021\\Aggression\\Aggression ^
        --yolo_weights path\\to\\fly_yolo_pose.pt ^
        --output_root outputs\\fly_behaviorscope_npz ^
        --window_size 30 --window_stride 15 ^
        --seed 42
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import queue
import random
import re
import sys
import threading
import time
import warnings
import zipfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from numpy.lib import format as np_format
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

warnings.filterwarnings(
    "ignore", message=".*pynvml.*deprecated.*", category=FutureWarning,
)

import cv2
import numpy as np
import scipy.io as sio

# ---------------------------------------------------------------------------
# Local imports from scripts/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_SCRIPTS_DIR = _HERE / "scripts"
sys.path.insert(0, str(_SCRIPTS_DIR))

from cropping import load_yolo_model
from cropping_x import NCropDetection, iterate_yolo_n_crops
from utils.logging_utils import setup_logger
from utils.pose_features_x import REL_FEATURE_DIM


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "behaviorscope-n-v1"
BUILD_KIND = "full_video_sliding"

BEHAVIOR_NAMES = ["lunge", "wing_threat", "charge", "hold", "tussle"]
OTHER_NAME = "other"
CLASS_NAMES = BEHAVIOR_NAMES + [OTHER_NAME]
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# Higher-priority behaviors win when multiple overlap in the same frame/window.
BEHAVIOR_PRIORITY = ["lunge", "tussle", "hold", "charge", "wing_threat"]

N_ANIMALS = 2
FLY_FPS = 30.0

DEFAULT_VAL_MOVIE_IDS = [3, 5]


# ---------------------------------------------------------------------------
# .mat file loaders (shared with fly_to_yolo_pose.py)
# ---------------------------------------------------------------------------
def load_traintest(aggression_root: str | Path):
    mat = sio.loadmat(os.path.join(str(aggression_root), "traintest.mat"))
    train_ids = mat["trainids"].ravel().astype(int).tolist()
    test_ids = mat["testids"].ravel().astype(int).tolist()
    return train_ids, test_ids


def load_actions(movie_dir: str | Path):
    movie_path = Path(movie_dir)
    canonical = movie_path / f"{movie_path.name}_actions.mat"
    if canonical.exists():
        action_file = canonical
    else:
        candidates = sorted(
            p for p in movie_path.glob("*_actions.mat")
            if "second" not in p.name.lower()
        )
        if not candidates:
            raise FileNotFoundError(f"No canonical *_actions.mat in {movie_path}")
        action_file = candidates[0]

    mat = sio.loadmat(str(action_file))
    behs_raw = mat["behs"]
    bouts_raw = mat["bouts"]
    n_flies, n_behs = bouts_raw.shape
    beh_names = [
        str(behs_raw[0, i].ravel()[0]) if behs_raw[0, i].size else f"beh_{i}"
        for i in range(n_behs)
    ]
    actions = {}
    for bi, bname in enumerate(beh_names):
        per_fly = []
        for fly in range(n_flies):
            b = bouts_raw[fly, bi]
            if b.size == 0 or b.ndim < 2:
                per_fly.append([])
                continue
            per_fly.append([
                (int(b[r, 0]) - 1, int(b[r, 1]) - 1)
                for r in range(b.shape[0])
                if int(b[r, 1]) >= int(b[r, 0])
            ])
        actions[bname] = per_fly
    return actions


# ---------------------------------------------------------------------------
# Per-frame label expansion from bout tables
# ---------------------------------------------------------------------------
def expand_actions_to_frame_labels(
    actions: dict, n_frames: int, class_to_idx: dict[str, int],
) -> np.ndarray:
    """Build per-frame label vector from Fly-v-Fly actions dict.

    Non-other behaviors overwrite in BEHAVIOR_PRIORITY order (highest-priority
    last so it wins on overlap).
    """
    out = np.full(n_frames, class_to_idx["other"], dtype=np.int16)
    for bname in reversed(BEHAVIOR_PRIORITY):
        if bname not in actions or bname not in class_to_idx:
            continue
        idx = class_to_idx[bname]
        for fly_bouts in actions[bname]:
            for (start, end) in fly_bouts:
                s = max(0, start)
                e = min(n_frames, end + 1)
                if e > s:
                    out[s:e] = idx
    return out


# ---------------------------------------------------------------------------
# Window labeling â€” priority-OR for short bouts
# ---------------------------------------------------------------------------
def window_label_priority_or(
    label_vec: np.ndarray,
    t: int,
    window_size: int,
    class_names: Sequence[str],
    class_to_idx: dict[str, int],
    priority_order: Sequence[str],
) -> str:
    """Label a window by the highest-priority non-other class present in any
    frame. Returns 'other' only if every frame is 'other'."""
    seg = label_vec[t:t + window_size]
    other_idx = class_to_idx["other"]
    present = set(seg[seg != other_idx].tolist())
    if not present:
        return "other"
    for bname in priority_order:
        if class_to_idx[bname] in present:
            return bname
    return "other"


# ---------------------------------------------------------------------------
# Async NPZ writer (from prepare_full_video_npz.py)
# ---------------------------------------------------------------------------
def save_npz_payload(
    path: Path,
    payload: dict,
    *,
    compression: str = "deflated",
    compresslevel: int = 1,
) -> None:
    path = Path(path)
    if compression == "stored":
        np.savez(path, **payload)
        return
    zip_compression = zipfile.ZIP_DEFLATED
    level = max(0, min(9, int(compresslevel)))
    with zipfile.ZipFile(
        path, mode="w", compression=zip_compression,
        compresslevel=level, allowZip64=True,
    ) as zf:
        for key, value in payload.items():
            if not re.match(r"^[A-Za-z0-9_]+$", str(key)):
                raise ValueError(f"Invalid NPZ key: {key!r}")
            with zf.open(f"{key}.npy", mode="w", force_zip64=True) as fh:
                np_format.write_array(fh, np.asanyarray(value), allow_pickle=False)


class _AsyncNPZWriter:
    def __init__(self, num_workers: int = 3, *, compression: str = "deflated",
                 compresslevel: int = 1) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._error: Optional[Exception] = None
        self._compression = str(compression)
        self._compresslevel = int(compresslevel)
        self._workers = [
            threading.Thread(target=self._run, daemon=True, name=f"npz-writer-{i}")
            for i in range(max(1, num_workers))
        ]
        for w in self._workers:
            w.start()

    def submit(self, path: Path, payload: dict) -> None:
        if self._error is not None:
            raise self._error
        self._q.put((path, payload))

    def flush(self) -> None:
        self._q.join()
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        self.flush()
        for _ in self._workers:
            self._q.put(None)
        for w in self._workers:
            w.join()

    def _run(self) -> None:
        while True:
            item = self._q.get()
            try:
                if item is None:
                    return
                path, payload = item
                path.parent.mkdir(parents=True, exist_ok=True)
                save_npz_payload(
                    path, payload,
                    compression=self._compression,
                    compresslevel=self._compresslevel,
                )
            except Exception as exc:
                self._error = exc
            finally:
                self._q.task_done()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------
@dataclass
class FlyVideoSource:
    split: str
    movie_id: int
    movie_name: str
    movie_dir: Path
    video_path: Path
    n_frames: int
    actions: dict
    label_vec: np.ndarray


def discover_fly_sources(
    aggression_root: Path,
    train_ids: list[int],
    val_ids: list[int],
    test_ids: list[int],
    class_to_idx: dict[str, int],
    logger,
    *,
    include_test: bool = False,
    only_test: bool = False,
    test_split_name: str = "test",
) -> list[FlyVideoSource]:
    sources = []
    process_ids = sorted(set(test_ids if only_test else (train_ids + val_ids + (test_ids if include_test else []))))

    for mid in process_ids:
        mname = f"movie{mid}"
        mdir = aggression_root / mname
        if not mdir.is_dir():
            logger.warning(f"[discover] {mname}: directory not found, skipping")
            continue

        video_path = mdir / f"{mname}.mp4"
        if not video_path.is_file():
            logger.warning(f"[discover] {mname}: no .mp4 found, skipping")
            continue

        cap = cv2.VideoCapture(str(video_path))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if n_frames <= 0:
            logger.warning(f"[discover] {mname}: no readable frames, skipping")
            continue

        actions = load_actions(str(mdir))
        label_vec = expand_actions_to_frame_labels(actions, n_frames, class_to_idx)

        if mid in test_ids:
            split = test_split_name
        else:
            split = "val" if mid in val_ids else "train"
        n_beh_frames = int(np.sum(label_vec != class_to_idx["other"]))
        logger.info(
            f"[discover] {mname}: {n_frames} frames, {n_beh_frames} behavior frames, "
            f"split={split}"
        )

        sources.append(FlyVideoSource(
            split=split,
            movie_id=mid,
            movie_name=mname,
            movie_dir=mdir,
            video_path=video_path,
            n_frames=n_frames,
            actions=actions,
            label_vec=label_vec,
        ))

    if not include_test and not only_test:
        for mid in test_ids:
            logger.info(f"[discover] movie{mid}: EXCLUDED (held-out test set)")

    if not sources:
        raise SystemExit("[discover] no valid source videos found")

    n_train = sum(1 for s in sources if s.split == "train")
    n_val = sum(1 for s in sources if s.split == "val")
    n_test = sum(1 for s in sources if s.split == test_split_name)
    logger.info(
        f"[discover] total: {len(sources)} sources "
        f"(train={n_train}, val={n_val}, {test_split_name}={n_test})"
    )
    return sources


# ---------------------------------------------------------------------------
# Payload builder (from NCropDetection list)
# ---------------------------------------------------------------------------
def _build_payload(frames: list[NCropDetection], n_animals: int) -> dict:
    group_frames = np.stack([e.group_rgb for e in frames], axis=0)
    animal_frames = np.stack([e.animal_rgbs for e in frames], axis=0)
    animal_keypoints = np.stack([e.keypoints_crop_norm for e in frames], axis=0)
    animal_keypoints_raw = np.stack([e.keypoints_xy_raw for e in frames], axis=0)
    animal_keypoints_conf = np.stack([e.keypoints_conf_raw for e in frames], axis=0)
    animal_mask = np.stack([e.animal_mask for e in frames], axis=0)
    pose_mask = np.stack([e.pose_mask for e in frames], axis=0)
    pose_conf = np.stack([e.pose_conf for e in frames], axis=0)
    crop_conf = np.stack([e.crop_conf for e in frames], axis=0)
    track_conf = np.stack([e.track_conf for e in frames], axis=0)
    track_age = np.stack([e.track_age for e in frames], axis=0)
    relation_features = np.stack([e.relation_features for e in frames], axis=0)
    relation_pose_mask = np.stack([e.relation_pose_mask for e in frames], axis=0)
    relation_present = np.stack([e.relation_present for e in frames], axis=0)
    bbox_xyxy_animals = np.stack([e.bbox_xyxy_animals for e in frames], axis=0)
    bbox_xyxy_group = np.asarray([e.bbox_xyxy_group for e in frames], dtype=np.int32)
    crop_xyxy_animals = np.stack([e.crop_xyxy_animals for e in frames], axis=0)
    crop_xyxy_group = np.asarray([e.crop_xyxy_group for e in frames], dtype=np.int32)
    track_ids = np.stack([e.track_ids for e in frames], axis=0)

    return {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "group_frames": group_frames.astype(np.uint8),
        "animal_frames": animal_frames.astype(np.uint8),
        "animal_keypoints": animal_keypoints.astype(np.float32),
        "animal_keypoints_raw": animal_keypoints_raw.astype(np.float32),
        "animal_keypoints_conf": animal_keypoints_conf.astype(np.float32),
        "animal_mask": animal_mask.astype(bool),
        "pose_mask": pose_mask.astype(bool),
        "pose_conf": pose_conf.astype(np.float32),
        "crop_conf": crop_conf.astype(np.float32),
        "track_conf": track_conf.astype(np.float32),
        "track_age": track_age.astype(np.float32),
        "relation_features": relation_features.astype(np.float32),
        "relation_pose_mask": relation_pose_mask.astype(bool),
        "relation_pose_conf": (pose_conf[:, :, None] * pose_conf[:, None, :]).astype(np.float32),
        "relation_present": relation_present.astype(bool),
        "bbox_xyxy_animals": bbox_xyxy_animals.astype(np.int32),
        "bbox_xyxy_group": bbox_xyxy_group.astype(np.int32),
        "crop_xyxy_animals": crop_xyxy_animals.astype(np.int32),
        "crop_xyxy_group": crop_xyxy_group.astype(np.int32),
        "track_ids": track_ids.astype(np.int32),
        "body_length_px": np.float32(0.0),
        "n_animals": np.int32(n_animals),
    }


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def _safe_stem(text: str, max_len: int = 200) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return (clean[:max_len] or "video")


def _stable_video_seed(seed: int, video_id: str) -> int:
    digest = hashlib.blake2b(str(video_id).encode("utf-8"), digest_size=8).hexdigest()
    return (int(seed) + int(digest, 16)) % (2**32)


def _done_marker_path(output_root: Path, safe_stem: str) -> Path:
    return output_root / "build_state" / "completed_videos" / f"{safe_stem}.json"


def _video_build_signature(src: FlyVideoSource, args) -> dict:
    return {
        "video_id": src.movie_name,
        "video_path": str(src.video_path.resolve()),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "other_subsample": float(args.other_subsample),
        "seed": int(args.seed),
        "n_animals": N_ANIMALS,
        "crop_size": int(args.crop_size),
        "group_crop_size": int(args.group_crop_size or args.crop_size),
        "animal_scale_factor": float(args.animal_scale_factor),
        "group_scale_factor": float(args.group_scale_factor),
        "yolo_weights": str(Path(args.yolo_weights).resolve()),
        "yolo_conf": float(args.yolo_conf),
        "yolo_iou": float(args.yolo_iou),
        "yolo_imgsz": int(args.yolo_imgsz),
        "pose_conf_threshold": float(args.pose_conf_threshold),
        "keep_last_box": bool(args.keep_last_box),
    }


def _load_done_marker(
    output_root: Path, safe_stem: str, src: FlyVideoSource, args, logger,
) -> tuple[list[dict], Counter] | None:
    marker_path = _done_marker_path(output_root, safe_stem)
    if not marker_path.is_file():
        return None
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"[{src.movie_name}] unreadable done marker: {exc}")
        return None
    expected = _video_build_signature(src, args)
    if marker.get("signature") != expected:
        logger.info(f"[{src.movie_name}] done marker exists but build params changed; rebuilding")
        return None
    samples = list(marker.get("samples") or [])
    base = output_root
    missing = [
        s.get("sequence_npz", "")
        for s in samples
        if not (base / str(s.get("sequence_npz", ""))).is_file()
    ]
    if missing:
        logger.warning(f"[{src.movie_name}] done marker has {len(missing)} missing NPZs; rebuilding")
        return None
    cls_counts = Counter(s.get("class_name", "") for s in samples)
    cls_counts.pop("", None)
    return samples, cls_counts


def _write_done_marker(
    output_root: Path, safe_stem: str, src: FlyVideoSource, args,
    samples: list[dict], cls_counts: Counter,
) -> None:
    marker_path = _done_marker_path(output_root, safe_stem)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "fly-v-fly-full-video-build-v1",
        "completed_at_unix": time.time(),
        "signature": _video_build_signature(src, args),
        "n_samples": len(samples),
        "class_counts": dict(cls_counts),
        "samples": samples,
    }
    marker_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _delete_existing_video_npzs(seq_dir: Path, safe_stem: str) -> int:
    if not seq_dir.exists():
        return 0
    deleted = 0
    for cls_dir in seq_dir.iterdir():
        if not cls_dir.is_dir():
            continue
        for npz in cls_dir.glob(f"{safe_stem}_*.npz"):
            npz.unlink()
            deleted += 1
    return deleted


# ---------------------------------------------------------------------------
# Per-video processing â€” streaming YOLO + sliding windows
# ---------------------------------------------------------------------------
def process_video(
    src: FlyVideoSource, model, args, output_root: Path,
    writer: _AsyncNPZWriter, logger,
) -> tuple[list[dict], Counter]:
    """Stream YOLO detections for one fly movie, emit sliding windows with
    priority-OR labeling, write NPZs asynchronously.

    Memory: O(window_size) frames buffered, not O(n_frames).
    """
    rng = random.Random(_stable_video_seed(args.seed, src.movie_name))
    safe_stem = _safe_stem(src.movie_name)
    seq_dir = output_root / "sequence_npz"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Resume check
    if args.skip_existing:
        existing = _load_done_marker(output_root, safe_stem, src, args, logger)
        if existing is not None:
            samples, cls_counts = existing
            logger.info(
                f"[{src.movie_name}] skip_existing: reusing {len(samples)} manifest entries"
            )
            return samples, cls_counts

    n_existing = _video_already_done(seq_dir, safe_stem)
    if n_existing and args.clean_incomplete_existing:
        deleted = _delete_existing_video_npzs(seq_dir, safe_stem)
        logger.warning(
            f"[{src.movie_name}] removed {deleted} stale NPZ files before rebuilding"
        )

    # Stream YOLO detections
    detections_iter = iterate_yolo_n_crops(
        model=model,
        video_path=src.video_path,
        n_animals=N_ANIMALS,
        crop_size=int(args.crop_size),
        group_crop_size=int(args.group_crop_size or args.crop_size),
        animal_scale_factor=float(args.animal_scale_factor),
        group_scale_factor=float(args.group_scale_factor),
        body_length_px=args.body_length_px,
        crop_strategy="fixed_scale",
        conf_threshold=float(args.yolo_conf),
        iou_threshold=float(args.yolo_iou),
        imgsz=int(args.yolo_imgsz),
        device=str(args.device),
        keep_last_box=bool(args.keep_last_box),
        pose_conf_threshold=float(args.pose_conf_threshold),
        fps=FLY_FPS,
        batch_size=int(args.yolo_batch),
    )

    buffer: deque = deque()
    next_window_t = 0
    samples: list[dict] = []
    cls_counts: Counter = Counter()
    n_dropped_subsample = 0
    n_emitted = 0
    n_skipped_missing = 0

    def _try_emit_windows() -> None:
        nonlocal next_window_t, n_emitted, n_dropped_subsample, n_skipped_missing
        while buffer:
            t = next_window_t
            t_end = t + args.window_size - 1
            if buffer[-1][0] < t_end:
                return
            while buffer and buffer[0][0] < t:
                buffer.popleft()
            window_frames: list[NCropDetection] = []
            ok = True
            expected = t
            for fi, det in buffer:
                if fi > t_end:
                    break
                if fi != expected:
                    ok = False
                    break
                window_frames.append(det)
                expected += 1
            if not ok or len(window_frames) != args.window_size:
                n_skipped_missing += 1
                next_window_t += args.window_stride
                continue

            cls = window_label_priority_or(
                src.label_vec, t, args.window_size,
                CLASS_NAMES, CLASS_TO_ID, BEHAVIOR_PRIORITY,
            )

            if cls == "other" and rng.random() > args.other_subsample:
                n_dropped_subsample += 1
                next_window_t += args.window_stride
                continue

            seq_name = f"{safe_stem}_f{t:06d}"
            npz_path = seq_dir / cls / f"{seq_name}.npz"
            payload = _build_payload(window_frames, N_ANIMALS)
            writer.submit(npz_path, payload)

            samples.append({
                "id": f"{cls}_{seq_name}",
                "label": int(CLASS_TO_ID[cls]),
                "class_name": cls,
                "sequence_npz": str(npz_path.relative_to(output_root)).replace("\\", "/"),
                "start_frame": int(t),
                "end_frame": int(t + args.window_size - 1),
                "fps": FLY_FPS,
                "source_video": src.movie_name,
                "dataset_split": src.split,
            })
            cls_counts[cls] += 1
            n_emitted += 1
            next_window_t += args.window_stride

    for det in detections_iter:
        buffer.append((int(det.frame_idx), det))
        while buffer and buffer[0][0] < next_window_t:
            buffer.popleft()
        _try_emit_windows()

    _try_emit_windows()

    if not samples:
        logger.warning(
            f"[{src.movie_name}] no windows emitted "
            f"(dropped_other_sub={n_dropped_subsample}, missing={n_skipped_missing})"
        )
        return [], Counter()

    logger.info(
        f"[{src.movie_name}] frames={src.n_frames}  emitted={n_emitted}  "
        f"dropped_other_sub={n_dropped_subsample}  missing={n_skipped_missing}  "
        f"per_cls={dict(cls_counts)}"
    )
    writer.flush()
    _write_done_marker(output_root, safe_stem, src, args, samples, cls_counts)
    return samples, cls_counts


def _video_already_done(seq_dir: Path, safe_stem: str) -> int:
    n = 0
    if not seq_dir.exists():
        return 0
    for cls_dir in seq_dir.iterdir():
        if cls_dir.is_dir():
            n += sum(1 for _ in cls_dir.glob(f"{safe_stem}_*.npz"))
    return n


# ---------------------------------------------------------------------------
# Manifest writing
# ---------------------------------------------------------------------------
def write_manifest(
    manifest_path: Path,
    splits: dict[str, list[dict]],
    meta: dict,
    class_to_idx: dict[str, int],
) -> None:
    payload = {
        "class_to_idx": dict(class_to_idx),
        "idx_to_class": {idx: name for name, idx in class_to_idx.items()},
        "meta": meta,
        "splits": splits,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BehaviorScope-Y NPZ builder for Fly-v-Fly full-video sliding windows.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--aggression_root", type=Path, required=True,
                   help="Aggression/Aggression folder with movie* dirs + traintest.mat.")
    p.add_argument("--yolo_weights", type=Path, required=True,
                   help="Path to domain-adapted fly YOLO-pose checkpoint.")
    p.add_argument("--output_root", type=Path,
                   default=Path("fly_behaviorscope_npz_full_video"))
    p.add_argument("--manifest_path", type=Path, default=None,
                   help="Default: <output_root>/sequence_manifest.json")

    p.add_argument("--val_movie_ids", nargs="+", type=int, default=DEFAULT_VAL_MOVIE_IDS,
                   help="Movie IDs carved from official train set as validation.")
    p.add_argument("--include_test", action="store_true",
                   help="Also build official held-out test movies into --test_split_name.")
    p.add_argument("--only_test", action="store_true",
                   help="Build only official held-out test movies into --test_split_name.")
    p.add_argument("--test_split_name", default="test",
                   help="Manifest split name for official held-out test movies.")
    p.add_argument("--window_size", type=int, default=16)
    p.add_argument("--window_stride", type=int, default=8)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--group_crop_size", type=int, default=224)
    p.add_argument("--animal_scale_factor", type=float, default=4.0)
    p.add_argument("--group_scale_factor", type=float, default=8.0)
    p.add_argument("--body_length_px", type=float, default=None,
                   help="Body length in pixels. None = auto-estimate from YOLO box sizes.")
    p.add_argument("--pose_conf_threshold", type=float, default=0.3)
    p.add_argument("--yolo_conf", type=float, default=0.25)
    p.add_argument("--yolo_iou", type=float, default=0.45)
    p.add_argument("--yolo_imgsz", type=int, default=640)
    p.add_argument("--yolo_batch", type=int, default=64)
    p.add_argument("--yolo_task", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--keep_last_box", action="store_true", default=True)

    p.add_argument("--other_subsample", type=float, default=0.30,
                   help="Keep this fraction of 'other' windows (1.0 = all, 0 = none).")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--npz_writers", type=int, default=3)
    p.add_argument("--npz_compression", choices=["deflated", "stored"], default="deflated")
    p.add_argument("--npz_compresslevel", type=int, default=1)
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip videos with a matching completed-video marker.")
    p.add_argument("--clean_incomplete_existing", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--dry_run", action="store_true",
                   help="Discover videos and report label counts; skip YOLO and NPZ writing.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    args = parse_args()
    if int(args.window_size) <= 0:
        raise SystemExit("--window_size must be > 0")
    if int(args.window_stride) <= 0:
        raise SystemExit("--window_stride must be > 0")
    if not (0.0 <= float(args.other_subsample) <= 1.0):
        raise SystemExit("--other_subsample must be between 0 and 1")

    output_root = args.output_root.resolve()
    manifest_path = (args.manifest_path or (output_root / "sequence_manifest.json")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_path = output_root / "fly_npz_build.log"
    logger = setup_logger("FlyNPZBuilder", log_path)

    # --- 1. Discover movies ---
    aggression_root = args.aggression_root.resolve()
    train_ids_official, test_ids = load_traintest(aggression_root)
    val_ids = [mid for mid in args.val_movie_ids if mid in train_ids_official]
    train_ids = [mid for mid in train_ids_official if mid not in val_ids]

    logger.info(f"[fly-npz] aggression_root:  {aggression_root}")
    logger.info(f"[fly-npz] train_ids:        {train_ids}")
    logger.info(f"[fly-npz] val_ids:          {val_ids}")
    if bool(args.only_test):
        test_status = f"ONLY -> {args.test_split_name}"
    elif bool(args.include_test):
        test_status = f"INCLUDED -> {args.test_split_name}"
    else:
        test_status = "EXCLUDED"
    logger.info(f"[fly-npz] test_ids:         {test_ids} ({test_status})")
    logger.info(f"[fly-npz] yolo_weights:     {args.yolo_weights}")
    logger.info(f"[fly-npz] window_size:      {args.window_size}  stride={args.window_stride}")
    logger.info(f"[fly-npz] labeling_rule:    priority-OR (short-bout safe)")
    logger.info(f"[fly-npz] other_subsample:  {args.other_subsample}")
    logger.info(f"[fly-npz] output_root:      {output_root}")

    sources = discover_fly_sources(
        aggression_root, train_ids, val_ids, test_ids, CLASS_TO_ID, logger,
        include_test=bool(args.include_test),
        only_test=bool(args.only_test),
        test_split_name=str(args.test_split_name),
    )

    if args.dry_run:
        logger.info("[fly-npz] DRY RUN â€” reporting label counts only.")
        for src in sources:
            counts = Counter()
            for idx in src.label_vec:
                counts[CLASS_NAMES[idx]] += 1
            logger.info(
                f"[dry] {src.movie_name} ({src.split}): {src.n_frames} frames, "
                f"per-class: {dict(counts)}"
            )
        logger.info("[fly-npz] dry run complete.")
        return 0

    # --- 2. Load YOLO model ---
    logger.info(f"[fly-npz] loading YOLO weights: {args.yolo_weights}")
    model = load_yolo_model(args.yolo_weights, device=args.device, task=args.yolo_task)

    npz_writer = _AsyncNPZWriter(
        num_workers=int(args.npz_writers),
        compression=str(args.npz_compression),
        compresslevel=int(args.npz_compresslevel),
    )

    # --- 3. Process each video ---
    splits_out: dict[str, list[dict]] = {split: [] for split in sorted({s.split for s in sources})}
    class_counts_total: Counter = Counter()
    per_video_summary: list[dict] = []

    t0 = time.time()
    for i, src in enumerate(sources):
        logger.info(f"[fly-npz] [{i+1}/{len(sources)}] {src.split}/{src.movie_name}")
        try:
            samples, cls_counts = process_video(src, model, args, output_root, npz_writer, logger)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            logger.error(f"[fly-npz] {src.movie_name}: failed: {exc}", exc_info=True)
            per_video_summary.append({
                "split": src.split, "video_id": src.movie_name,
                "status": "error", "error": str(exc),
            })
            continue

        splits_out[src.split].extend(samples)
        class_counts_total.update(cls_counts)
        per_video_summary.append({
            "split": src.split, "video_id": src.movie_name,
            "status": "ok", "n_windows": len(samples),
            **{f"n_{c}": int(cls_counts.get(c, 0)) for c in CLASS_NAMES},
        })

    npz_writer.close()
    elapsed = time.time() - t0

    # --- 4. Write manifest ---
    split_window_counts = {split: len(samples) for split, samples in sorted(splits_out.items())}
    if sum(split_window_counts.values()) == 0:
        raise SystemExit("[fly-npz] no windows emitted for any video.")

    # Determine num_keypoints from first NPZ (YOLO model decides keypoint count)
    num_keypoints = 0
    first_npz = None
    for split_samples in splits_out.values():
        if split_samples:
            first_npz = output_root / split_samples[0]["sequence_npz"]
            break
    if first_npz is not None and first_npz.is_file():
        with np.load(first_npz, allow_pickle=False) as data:
            if "animal_keypoints" in data:
                num_keypoints = data["animal_keypoints"].shape[2]

    meta = {
        "schema_version": SCHEMA_VERSION,
        "build_kind": BUILD_KIND,
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "n_animals": N_ANIMALS,
        "keypoints_per_animal": num_keypoints,
        "class_names": list(CLASS_NAMES),
        "labeling_rule": "priority_or",
        "other_subsample": float(args.other_subsample),
        "npz_compression": str(args.npz_compression),
        "npz_compresslevel": int(args.npz_compresslevel),
        "yolo_weights": str(args.yolo_weights),
        "yolo_imgsz": int(args.yolo_imgsz),
        "source_dataset": "Fly-v-Fly Aggression (Caltech, Eyjolfsdottir et al. 2021)",
        "train_movie_ids": train_ids,
        "val_movie_ids": val_ids,
        "test_movie_ids": test_ids,
        "test_movie_ids_included": test_ids if (bool(args.include_test) or bool(args.only_test)) else [],
        "test_movie_ids_excluded": [] if (bool(args.include_test) or bool(args.only_test)) else test_ids,
        "test_split_name": str(args.test_split_name),
        "build_elapsed_s": round(elapsed, 1),
        "class_counts": dict(class_counts_total),
        "split_window_counts": split_window_counts,
        "source_video_counts_by_split": {
            split: sum(1 for s in sources if s.split == split)
            for split in sorted({s.split for s in sources})
        },
    }

    write_manifest(manifest_path, splits_out, meta, CLASS_TO_ID)
    logger.info(f"[fly-npz] manifest -> {manifest_path}")

    # Per-video summary CSV
    pv_csv = output_root / "per_video_build_summary.csv"
    if per_video_summary:
        fieldnames = sorted({key for row in per_video_summary for key in row.keys()})
        with open(pv_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(per_video_summary)
        logger.info(f"[fly-npz] per-video summary -> {pv_csv}")

    summary = {
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "elapsed_s": round(elapsed, 1),
        "windows_total": int(sum(split_window_counts.values())),
        "split_window_counts": split_window_counts,
        "class_counts": dict(class_counts_total),
        "num_keypoints": num_keypoints,
        "labeling_rule": "priority_or",
    }
    summary_path = output_root / "fly_npz_build_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"[fly-npz] summary -> {summary_path}")

    print(f"\n{'='*60}")
    print(f"Fly-v-Fly NPZ build complete in {elapsed:.1f}s")
    for split, count in split_window_counts.items():
        print(f"  {split} windows: {count}")
    print(f"  Per class:     {dict(class_counts_total)}")
    print(f"  Keypoints:     {num_keypoints}")
    print(f"  Manifest:      {manifest_path}")
    print(f"{'='*60}")
    print(f"\nTrain BehaviorScope-Y on Fly-v-Fly:")
    print(f"  cd {_SCRIPTS_DIR}")
    print(f"  python train_x.py \\")
    print(f"      --manifest_path {manifest_path} \\")
    print(f"      --num_keypoints {num_keypoints} \\")
    print(f"      --num_classes {len(CLASS_NAMES)}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"[fly-npz] ERROR: {exc}", file=sys.stderr)
        raise



