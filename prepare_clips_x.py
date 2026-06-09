"""Prepare BehaviorScope-Y training windows from class-folder behavior clips.

This is the release-facing upstream path for users who do not have dense
full-video bout annotations. Users place short behavior clips into one folder
per behavior class; this script runs the YOLO pose model, writes strict
``behaviorscope-n-v1`` NPZ windows, and emits a ``sequence_manifest.json`` that
``train_x.py --auto_feature_cache`` can consume directly.

The manifest deliberately records clip provenance. Class-folder clips are a
practical training format, but validation is only as strict as the grouping
metadata provided by the user.
"""
from __future__ import annotations

import argparse
import csv
import json
import queue
import re
import sys
import threading
import time
import warnings
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

warnings.filterwarnings(
    "ignore",
    message=".*pynvml.*deprecated.*",
    category=FutureWarning,
)

import cv2
import numpy as np
from PIL import Image

if __package__ is None or __package__ == "":  # pragma: no cover - CLI entry
    from config import ALLOWED_EXTENSIONS
    from cropping import load_yolo_model
    from cropping_x import NCropDetection, iterate_yolo_n_crops
    from data_x import load_n_manifest, validate_n_npz_payload
    from utils.logging_utils import setup_logger
    from utils.pose_features_x import REL_FEATURE_DIM
else:  # pragma: no cover
    from .config import ALLOWED_EXTENSIONS
    from .cropping import load_yolo_model
    from .cropping_x import NCropDetection, iterate_yolo_n_crops
    from .data_x import load_n_manifest, validate_n_npz_payload
    from .utils.logging_utils import setup_logger
    from .utils.pose_features_x import REL_FEATURE_DIM


KNOWN_OUTPUT_NAMES = {
    "processed_sequences",
    "behaviorscope_x_processed",
    "behaviorscope_x_processed",
    "yolo_feature_cache",
    "qa_crops",
    "sequence_npz",
    "logs",
    "__pycache__",
}

MANIFEST_EXTRA_KEYS = {
    "clip_path",
    "clip_stem",
    "clip_start_frame",
    "clip_end_frame",
    "source_group",
    "source_grouping",
    "split",
}


@dataclass(frozen=True)
class SequenceSample:
    id: str
    label: int
    class_name: str
    sequence_npz: Path
    frame_paths: Tuple[Path, ...]
    start_frame: int
    end_frame: int
    fps: float
    source_video: str


@dataclass(frozen=True)
class ClipInfo:
    class_name: str
    label: int
    path: Path
    rel_path: str
    stem: str
    source_group: str
    source_video: str
    clip_start_frame: Optional[int]
    clip_end_frame: Optional[int]
    explicit_split: Optional[str]
    fps_override: Optional[float]


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    truthy = {"1", "true", "t", "yes", "y", "on"}
    falsy = {"0", "false", "f", "no", "n", "off"}
    normalized = str(value).strip().lower()
    if normalized in truthy:
        return True
    if normalized in falsy:
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build BehaviorScope-Y NPZ windows + sequence_manifest.json from "
            "class-folder behavior clips."
        )
    )
    p.add_argument("--dataset_root", type=Path, required=True,
                   help="Folder containing one immediate subfolder per behavior class.")
    p.add_argument("--yolo_weights", type=Path, required=True)
    p.add_argument("--output_root", type=Path, default=None,
                   help="Default: <dataset_root>/behaviorscope_x_processed.")
    p.add_argument("--manifest_path", type=Path, default=None,
                   help="Default: <output_root>/sequence_manifest.json.")

    p.add_argument("--window_size", type=int, default=32)
    p.add_argument("--window_stride", type=int, default=16)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--group_crop_size", type=int, default=None)
    p.add_argument("--n_animals", type=int, default=2)
    p.add_argument("--animal_scale_factor", type=float, default=4.0)
    p.add_argument("--group_scale_factor", type=float, default=8.0)
    p.add_argument("--body_length_px", type=float, default=None)
    p.add_argument("--pose_conf_threshold", type=float, default=0.3)
    p.add_argument("--yolo_conf", type=float, default=0.25)
    p.add_argument("--yolo_iou", type=float, default=0.45)
    p.add_argument("--yolo_imgsz", type=int, default=640)
    p.add_argument("--yolo_task", default=None)
    p.add_argument("--yolo_batch", type=int, default=64,
                   help="Frames per YOLO inference batch (default 64). Higher values improve GPU "
                        "utilisation at the cost of VRAM. Try 128 on >=12 GB cards.")
    p.add_argument("--npz_writers", type=int, default=2,
                   help="Background threads used to compress and write NPZ files (default 2). "
                        "Overlaps disk I/O with GPU inference so the GPU is not stalled.")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--keep_last_box", action="store_true")
    p.add_argument("--default_fps", type=float, default=30.0)

    p.add_argument("--save_frame_crops", nargs="?", const=True, default=False,
                   type=_parse_bool, help="Optional QA crop images.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Reuse existing NPZ windows for matching clips.")
    p.add_argument("--min_sequence_count", type=int, default=1)

    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--test_ratio", type=float, default=0.0)
    p.add_argument(
        "--split_strategy",
        choices=["clip", "source", "window", "explicit"],
        default="clip",
        help=(
            "clip keeps each clipped file in one split; source groups by "
            "metadata/filename source id; window splits individual windows; "
            "explicit uses split values from metadata or split-map CSV."
        ),
    )
    p.add_argument(
        "--source_grouping",
        choices=["clip", "metadata", "filename_prefix"],
        default="clip",
        help=(
            "How to infer the source group used by --split_strategy source. "
            "clip is easiest; metadata is strongest when clips share recordings."
        ),
    )
    p.add_argument("--clip_metadata_csv", type=Path, default=None,
                   help="Optional CSV with clip_path or clip_stem plus provenance fields.")
    p.add_argument("--split_map_csv", type=Path, default=None,
                   help="Optional CSV with clip_path or clip_stem and split.")
    p.add_argument("--filename_prefix_delimiter", default="__",
                   help="Delimiter for --source_grouping filename_prefix.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dense_bout_annotations", action="store_true",
                   help="Set manifest meta dense_bout_annotations=true.")

    p.add_argument("--validate_manifest", nargs="?", const=True, default=True,
                   type=_parse_bool, help="Validate the produced manifest and NPZ schema.")
    p.add_argument("--validation_sample_limit", type=int, default=0,
                   help="0 validates all samples; otherwise validates first N samples.")

    args = p.parse_args()
    if args.window_size <= 0 or args.window_stride <= 0:
        raise SystemExit("window_size and window_stride must be positive.")
    if args.n_animals <= 0:
        raise SystemExit("--n_animals must be >= 1.")
    if min(args.train_ratio, args.val_ratio, args.test_ratio) < 0:
        raise SystemExit("Split ratios must be non-negative.")
    ratio_sum = float(args.train_ratio + args.val_ratio + args.test_ratio)
    if ratio_sum <= 0:
        raise SystemExit("At least one split ratio must be > 0.")
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(
            "--train_ratio + --val_ratio + --test_ratio must sum to 1.0 "
            f"(got {ratio_sum:.6g})."
        )
    if args.split_strategy == "explicit" and not (args.clip_metadata_csv or args.split_map_csv):
        raise SystemExit("--split_strategy explicit requires --clip_metadata_csv or --split_map_csv.")
    if args.source_grouping == "metadata" and not args.clip_metadata_csv:
        raise SystemExit("--source_grouping metadata requires --clip_metadata_csv.")
    return args


def _norm_rel(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def _safe_stem(text: str, max_len: int = 96) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return (clean[:max_len] or "clip")


def _optional_int(value) -> Optional[int]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return int(float(text))


def _optional_float(value) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def _metadata_key_variants(path: Path, dataset_root: Path) -> List[str]:
    rel = _norm_rel(path, dataset_root)
    return [
        rel.lower().replace("\\", "/"),
        path.name.lower(),
        path.stem.lower(),
    ]


def load_metadata_csv(path: Optional[Path], dataset_root: Path) -> Dict[str, dict]:
    if not path:
        return {}
    if not path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {path}")
    out: Dict[str, dict] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"{path}: no CSV header")
        for row_i, row in enumerate(reader, start=2):
            raw = (
                row.get("clip_path")
                or row.get("path")
                or row.get("file")
                or row.get("filename")
                or row.get("clip_stem")
            )
            if not raw:
                raise ValueError(
                    f"{path}:{row_i}: expected clip_path, path, file, filename, or clip_stem"
                )
            keys = {str(raw).strip().lower().replace("\\", "/")}
            if not Path(str(raw)).suffix and "/" not in str(raw).replace("\\", "/"):
                keys.add(Path(str(raw)).stem.lower())
            else:
                raw_path = Path(str(raw))
                if not raw_path.is_absolute():
                    raw_path = dataset_root / raw_path
                keys.update(_metadata_key_variants(raw_path, dataset_root))
            for key in keys:
                if key:
                    out[key] = dict(row)
    return out


def merged_metadata_for_clip(
    clip_path: Path,
    dataset_root: Path,
    metadata: Dict[str, dict],
    split_map: Dict[str, dict],
) -> dict:
    merged = {}
    for key in _metadata_key_variants(clip_path, dataset_root):
        if key in metadata:
            merged.update(metadata[key])
        if key in split_map:
            merged.update(split_map[key])
    return merged


def find_class_dirs(dataset_root: Path, output_root: Path) -> Tuple[List[Path], Dict[str, int]]:
    excluded = set()
    try:
        excluded.add(output_root.resolve())
    except Exception:
        pass
    class_dirs: List[Path] = []
    skipped: List[Tuple[Path, str]] = []
    for path in sorted(dataset_root.iterdir()):
        if not path.is_dir():
            continue
        reason = None
        if path.name.startswith(".") or path.name.startswith("_"):
            reason = "hidden/dunder name"
        elif path.name in KNOWN_OUTPUT_NAMES:
            reason = f"known output folder name {path.name!r}"
        else:
            try:
                if path.resolve() in excluded:
                    reason = "matches output_root"
            except Exception:
                pass
        if reason:
            skipped.append((path, reason))
            continue
        class_dirs.append(path)
    if skipped:
        for path, reason in skipped:
            print(f"[prepare-y] skipping non-class dir: {path.name} ({reason})")
    if not class_dirs:
        raise RuntimeError(f"No behavior class folders found under {dataset_root}")
    return class_dirs, {path.name: i for i, path in enumerate(class_dirs)}


def iter_video_files(class_dir: Path) -> Iterable[Path]:
    for path in sorted(class_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS:
            yield path


def resolve_source_group(
    clip_path: Path,
    dataset_root: Path,
    grouping: str,
    row: dict,
    delimiter: str,
) -> Tuple[str, str]:
    if grouping == "metadata":
        src = (
            row.get("source_group")
            or row.get("recording_id")
            or row.get("source_video")
            or row.get("video_id")
        )
        if not src:
            raise ValueError(
                f"{clip_path}: --source_grouping metadata requires source_group, "
                "recording_id, source_video, or video_id in metadata CSV."
            )
        return str(src), str(row.get("source_video") or src)
    if grouping == "filename_prefix":
        stem = clip_path.stem
        if delimiter and delimiter in stem:
            src = stem.split(delimiter, 1)[0]
        elif "_" in stem:
            src = stem.split("_", 1)[0]
        else:
            src = stem
        return src, src
    rel = _norm_rel(clip_path, dataset_root)
    return rel, rel


def discover_clips(args, class_to_idx: Dict[str, int]) -> List[ClipInfo]:
    metadata = load_metadata_csv(args.clip_metadata_csv, args.dataset_root)
    split_map = load_metadata_csv(args.split_map_csv, args.dataset_root)
    clips: List[ClipInfo] = []
    for class_dir_name, label in class_to_idx.items():
        class_dir = args.dataset_root / class_dir_name
        for path in iter_video_files(class_dir):
            row = merged_metadata_for_clip(path, args.dataset_root, metadata, split_map)
            source_group, source_video = resolve_source_group(
                path, args.dataset_root, args.source_grouping, row,
                args.filename_prefix_delimiter,
            )
            explicit_split = row.get("split") or row.get("set")
            if explicit_split is not None:
                explicit_split = str(explicit_split).strip().lower() or None
            clips.append(ClipInfo(
                class_name=class_dir_name,
                label=label,
                path=path,
                rel_path=_norm_rel(path, args.dataset_root),
                stem=path.stem,
                source_group=str(source_group),
                source_video=str(source_video),
                clip_start_frame=_optional_int(
                    row.get("clip_start_frame") or row.get("start_frame")
                ),
                clip_end_frame=_optional_int(
                    row.get("clip_end_frame") or row.get("end_frame")
                ),
                explicit_split=explicit_split,
                fps_override=_optional_float(row.get("fps")),
            ))
    if not clips:
        raise RuntimeError("No behavior clips found.")
    return clips


def ensure_output_dirs(output_root: Path, class_name: str, clip_stem: str, save_qa: bool):
    seq_dir = output_root / "sequence_npz" / class_name
    seq_dir.mkdir(parents=True, exist_ok=True)
    qa_dir = None
    if save_qa:
        qa_dir = output_root / "qa_crops" / class_name / clip_stem
        qa_dir.mkdir(parents=True, exist_ok=True)
    return seq_dir, qa_dir


def save_n_frame_crops(qa_dir: Path, detection: NCropDetection) -> None:
    frame_dir = qa_dir / f"{detection.frame_idx:06d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(detection.group_rgb).save(frame_dir / "group.jpg", quality=95)
    for idx, crop in enumerate(detection.animal_rgbs):
        Image.fromarray(crop).save(frame_dir / f"animal_{idx}.jpg", quality=95)


def probe_fps(video_path: Path, fallback: float) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps is None or fps <= 1e-3:
        return float(fallback)
    return float(fps)


def save_y_npz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


class _AsyncNPZWriter:
    """Compress and write NPZ files on background threads so GPU inference
    is not stalled waiting for disk I/O or gzip compression."""

    def __init__(self, num_workers: int = 2) -> None:
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._error: Optional[Exception] = None
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
        """Block until all pending writes finish (call after each clip)."""
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
                np.savez_compressed(path, **payload)
            except Exception as exc:  # noqa: BLE001
                self._error = exc
            finally:
                self._q.task_done()


def sample_extra(info: ClipInfo, sample_split: Optional[str] = None) -> dict:
    return {
        "clip_path": info.rel_path,
        "clip_stem": info.stem,
        "clip_start_frame": info.clip_start_frame,
        "clip_end_frame": info.clip_end_frame,
        "source_group": info.source_group,
        "source_grouping": None,
        "split": sample_split,
    }


def load_existing_sequences(
    info: ClipInfo,
    seq_paths: Sequence[Path],
    args,
    fps: float,
) -> Tuple[List[SequenceSample], set[int]]:
    samples: List[SequenceSample] = []
    keypoint_counts: set[int] = set()
    for seq_path in sorted(seq_paths):
        with np.load(seq_path, allow_pickle=False) as data:
            schema = str(np.asarray(data["schema_version"]).item()) if "schema_version" in data.files else ""
            if schema != "behaviorscope-n-v1":
                raise ValueError(f"{seq_path}: schema_version={schema!r}; expected behaviorscope-n-v1")
            keypoint_counts.add(int(data["animal_keypoints"].shape[2]))
            window_size = int(data["animal_keypoints"].shape[0])
            if window_size != int(args.window_size):
                raise ValueError(
                    f"{seq_path}: window_size={window_size}, expected {args.window_size}"
                )
        start_frame = 0
        marker = "_f"
        if marker in seq_path.stem:
            try:
                start_frame = int(seq_path.stem.rsplit(marker, 1)[1].split("_", 1)[0])
            except Exception:
                start_frame = 0
        end_frame = start_frame + int(args.window_size) - 1
        samples.append(SequenceSample(
            id=f"{info.class_name}_{seq_path.stem}",
            label=info.label,
            class_name=info.class_name,
            sequence_npz=seq_path,
            frame_paths=tuple(),
            start_frame=start_frame,
            end_frame=end_frame,
            fps=fps,
            source_video=info.source_video,
        ))
    return samples, keypoint_counts


def build_sequences_for_clip(model, info: ClipInfo, args, output_root: Path, logger,
                             writer: Optional["_AsyncNPZWriter"] = None):
    seq_dir, qa_dir = ensure_output_dirs(
        output_root, info.class_name, _safe_stem(info.stem), args.save_frame_crops
    )
    fps = float(info.fps_override or probe_fps(info.path, args.default_fps))
    if args.skip_existing:
        existing = sorted(seq_dir.glob(f"{_safe_stem(info.stem)}_*.npz"))
        if existing:
            logger.info(f"Reusing {len(existing)} existing windows for {info.path}")
            samples, k_counts = load_existing_sequences(info, existing, args, fps)
            return samples, k_counts

    detections = iterate_yolo_n_crops(
        model=model,
        video_path=info.path,
        n_animals=int(args.n_animals),
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
        fps=fps,
        batch_size=int(getattr(args, "yolo_batch", 64)),
    )

    buffer: deque[NCropDetection] = deque()
    samples: List[SequenceSample] = []
    keypoint_counts: set[int] = set()
    seq_index = 0
    for det in detections:
        if qa_dir:
            save_n_frame_crops(qa_dir, det)
        buffer.append(det)

        while len(buffer) >= int(args.window_size):
            entries = list(buffer)[: int(args.window_size)]
            start_frame = int(entries[0].frame_idx)
            end_frame = int(entries[-1].frame_idx)

            group_frames = np.stack([e.group_rgb for e in entries], axis=0)
            animal_frames = np.stack([e.animal_rgbs for e in entries], axis=0)
            animal_keypoints = np.stack([e.keypoints_crop_norm for e in entries], axis=0)
            animal_keypoints_raw = np.stack([e.keypoints_xy_raw for e in entries], axis=0)
            animal_keypoints_conf = np.stack([e.keypoints_conf_raw for e in entries], axis=0)
            animal_mask = np.stack([e.animal_mask for e in entries], axis=0)
            pose_mask = np.stack([e.pose_mask for e in entries], axis=0)
            pose_conf = np.stack([e.pose_conf for e in entries], axis=0)
            crop_conf = np.stack([e.crop_conf for e in entries], axis=0)
            track_conf = np.stack([e.track_conf for e in entries], axis=0)
            track_age = np.stack([e.track_age for e in entries], axis=0)
            relation_features = np.stack([e.relation_features for e in entries], axis=0)
            relation_pose_mask = np.stack([e.relation_pose_mask for e in entries], axis=0)
            relation_present = np.stack([e.relation_present for e in entries], axis=0)
            bbox_xyxy_animals = np.stack([e.bbox_xyxy_animals for e in entries], axis=0)
            bbox_xyxy_group = np.asarray([e.bbox_xyxy_group for e in entries], dtype=np.int32)
            crop_xyxy_animals = np.stack([e.crop_xyxy_animals for e in entries], axis=0)
            crop_xyxy_group = np.asarray([e.crop_xyxy_group for e in entries], dtype=np.int32)
            track_ids = np.stack([e.track_ids for e in entries], axis=0)

            num_keypoints = int(animal_keypoints.shape[2])
            keypoint_counts.add(num_keypoints)
            seq_name = f"{_safe_stem(info.stem)}_f{start_frame:06d}_s{seq_index:04d}"
            npz_path = seq_dir / f"{seq_name}.npz"
            _payload = {
                "schema_version": np.asarray("behaviorscope-n-v1"),
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
                "body_length_px": np.float32(args.body_length_px or 0.0),
                "n_animals": np.int32(args.n_animals),
            }
            if writer is not None:
                writer.submit(npz_path, _payload)
            else:
                save_y_npz(npz_path, _payload)
            samples.append(SequenceSample(
                id=f"{info.class_name}_{seq_name}",
                label=int(info.label),
                class_name=info.class_name,
                sequence_npz=npz_path,
                frame_paths=tuple(),
                start_frame=start_frame,
                end_frame=end_frame,
                fps=fps,
                source_video=info.source_video,
            ))
            seq_index += 1
            for _ in range(int(args.window_stride)):
                if buffer:
                    buffer.popleft()

    if len(samples) < int(args.min_sequence_count):
        logger.info(f"Skipping {info.path}: only {len(samples)} windows.")
        return [], keypoint_counts
    return samples, keypoint_counts


def _split_counts(n_items: int, ratios: Sequence[float]) -> List[int]:
    if n_items == 0:
        return [0 for _ in ratios]
    total = float(sum(ratios))
    if total <= 0:
        raise ValueError("At least one split ratio must be > 0")
    scaled = [r / total * n_items for r in ratios]
    counts = [int(np.floor(x)) for x in scaled]
    remainder = n_items - sum(counts)
    order = sorted(((scaled[i] - counts[i], i) for i in range(len(ratios))), reverse=True)
    for _, idx in order[:remainder]:
        counts[idx] += 1
    return counts


def stratified_split(
    samples: Sequence[SequenceSample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[SequenceSample]]:
    rng = np.random.default_rng(int(seed))
    by_class: Dict[int, List[SequenceSample]] = defaultdict(list)
    for sample in samples:
        by_class[int(sample.label)].append(sample)

    out = {"train": [], "val": [], "test": []}
    for class_samples in by_class.values():
        rows = list(class_samples)
        rng.shuffle(rows)
        train_n, val_n, test_n = _split_counts(
            len(rows), (train_ratio, val_ratio, test_ratio)
        )
        out["train"].extend(rows[:train_n])
        out["val"].extend(rows[train_n:train_n + val_n])
        out["test"].extend(rows[train_n + val_n:train_n + val_n + test_n])
    return out


def stratified_split_by_group(
    samples: Sequence[SequenceSample],
    group_by_sample_id: Dict[str, str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[SequenceSample]]:
    rng = np.random.default_rng(int(seed))
    by_class_group: Dict[int, Dict[str, List[SequenceSample]]] = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        group = group_by_sample_id.get(sample.id) or sample.source_video or sample.id
        by_class_group[int(sample.label)][str(group)].append(sample)

    out = {"train": [], "val": [], "test": []}
    split_names = ("train", "val", "test")
    for groups in by_class_group.values():
        items = list(groups.items())
        rng.shuffle(items)
        items.sort(key=lambda item: -len(item[1]))
        total = sum(len(v) for _, v in items)
        targets = dict(zip(split_names, _split_counts(total, (train_ratio, val_ratio, test_ratio))))
        counts = {name: 0 for name in split_names}
        for _, group_samples in items:
            size = len(group_samples)

            def score(split_name: str) -> Tuple[float, float]:
                if targets[split_name] == 0 and size > 0:
                    return (1e12, 0.0)
                nxt = dict(counts)
                nxt[split_name] += size
                err = sum(abs(nxt[name] - targets[name]) for name in split_names)
                remaining = targets[split_name] - counts[split_name]
                return (float(err), -float(remaining))

            chosen = min(split_names, key=score)
            out[chosen].extend(group_samples)
            counts[chosen] += size
    return out


def split_explicit(samples: Sequence[SequenceSample], split_by_sample_id: Dict[str, Optional[str]]):
    out = {"train": [], "val": [], "test": []}
    aliases = {"validation": "val", "valid": "val", "dev": "val"}
    for sample in samples:
        split = split_by_sample_id.get(sample.id)
        if not split:
            raise ValueError(
                f"{sample.id}: no explicit split found. Provide split in metadata/split-map CSV."
            )
        split = aliases.get(str(split).lower(), str(split).lower())
        if split not in out:
            raise ValueError(f"{sample.id}: unsupported split {split!r}; expected train/val/test")
        out[split].append(sample)
    return out


def _relative_path(path: Optional[Path], base: Path) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        return path.as_posix()


def write_manifest(
    manifest_path: Path,
    splits: Dict[str, Sequence[SequenceSample]],
    class_to_idx: Dict[str, int],
    metadata: dict,
    extra_by_sample_id: Dict[str, dict],
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    base = manifest_path.parent
    payload = {
        "class_to_idx": class_to_idx,
        "idx_to_class": {idx: name for name, idx in class_to_idx.items()},
        "meta": metadata,
        "splits": {},
    }
    for split_name, split_samples in splits.items():
        rows = []
        for sample in split_samples:
            extra = dict(extra_by_sample_id.get(sample.id, {}))
            extra["split"] = split_name
            row = {
                "id": sample.id,
                "label": int(sample.label),
                "class_name": sample.class_name,
                "sequence_npz": _relative_path(sample.sequence_npz, base),
                "frame_paths": [_relative_path(p, base) for p in sample.frame_paths],
                "start_frame": int(sample.start_frame),
                "end_frame": int(sample.end_frame),
                "fps": float(sample.fps),
                "source_video": sample.source_video,
            }
            for key, value in extra.items():
                if key in MANIFEST_EXTRA_KEYS:
                    row[key] = value
                elif value is not None and value != "":
                    row[key] = value
            rows.append(row)
        payload["splits"][split_name] = rows
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_manifest(manifest_path: Path, sample_limit: int = 0) -> dict:
    splits, class_to_idx, _idx_to_class, meta = load_n_manifest(manifest_path)
    schema = meta.get("schema_version")
    if schema != "behaviorscope-n-v1":
        raise ValueError(f"Manifest schema_version={schema!r}; expected behaviorscope-n-v1")
    num_frames = int(meta["window_size"])
    n_animals = int(meta["n_animals"])
    num_keypoints = int(meta["keypoints_per_animal"])
    all_samples = [s for rows in splits.values() for s in rows]
    if sample_limit and sample_limit > 0:
        samples_to_check = all_samples[: int(sample_limit)]
    else:
        samples_to_check = all_samples
    for sample in samples_to_check:
        with np.load(sample.sequence_npz, allow_pickle=False) as data:
            validate_n_npz_payload(
                data,
                sample_id=sample.id,
                num_frames=num_frames,
                n_animals=n_animals,
                num_keypoints=num_keypoints,
                rel_feature_dim=REL_FEATURE_DIM,
                require_schema_version=True,
            )
    by_split = {name: len(rows) for name, rows in splits.items()}
    by_class = Counter(s.class_name for s in all_samples)
    return {
        "samples_total": len(all_samples),
        "samples_validated": len(samples_to_check),
        "splits": by_split,
        "classes": dict(sorted(by_class.items())),
        "class_to_idx": class_to_idx,
    }


def main() -> None:
    args = parse_args()
    args.dataset_root = args.dataset_root.resolve()
    output_root = (args.output_root or (args.dataset_root / "behaviorscope_x_processed")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (args.manifest_path or (output_root / "sequence_manifest.json")).resolve()

    logger = setup_logger("PrepareBehaviorScopeX", output_root / "prepare_clips_x.log")
    t0 = time.time()

    class_dirs, class_to_idx = find_class_dirs(args.dataset_root, output_root)
    logger.info(f"Found {len(class_dirs)} behavior folders: {list(class_to_idx)}")
    clips = discover_clips(args, class_to_idx)
    logger.info(f"Found {len(clips)} clips.")

    model = load_yolo_model(args.yolo_weights, device=args.device, task=args.yolo_task)
    logger.info(f"Loaded YOLO weights from {args.yolo_weights}")
    logger.info(f"YOLO batch size: {args.yolo_batch}  |  NPZ writer threads: {args.npz_writers}")

    npz_writer = _AsyncNPZWriter(num_workers=int(args.npz_writers))

    all_samples: List[SequenceSample] = []
    extra_by_sample_id: Dict[str, dict] = {}
    clip_group_by_sample_id: Dict[str, str] = {}
    source_group_by_sample_id: Dict[str, str] = {}
    explicit_split_by_sample_id: Dict[str, Optional[str]] = {}
    keypoint_counts: set[int] = set()
    clip_window_counts: Counter[str] = Counter()

    for info in clips:
        logger.info(f"Processing {info.path}")
        samples, k_counts = build_sequences_for_clip(model, info, args, output_root, logger,
                                                      writer=npz_writer)
        keypoint_counts.update(k_counts)
        clip_window_counts[info.rel_path] += len(samples)
        for sample in samples:
            all_samples.append(sample)
            extra = sample_extra(info)
            extra["source_grouping"] = args.source_grouping
            extra_by_sample_id[sample.id] = extra
            clip_group_by_sample_id[sample.id] = info.rel_path
            source_group_by_sample_id[sample.id] = info.source_group
            explicit_split_by_sample_id[sample.id] = info.explicit_split

    npz_writer.close()  # flush + join all background writer threads before manifest

    if not all_samples:
        raise RuntimeError("No windows were generated. Check YOLO detections and clip lengths.")
    if not keypoint_counts:
        raise RuntimeError("No keypoint count could be inferred from generated windows.")
    if len(keypoint_counts) != 1:
        raise RuntimeError(
            "Inconsistent keypoint counts across windows: "
            f"{sorted(keypoint_counts)}. Use one YOLO pose schema per manifest."
        )
    keypoints_per_animal = int(next(iter(keypoint_counts)))

    if args.split_strategy == "window":
        splits = stratified_split(
            all_samples,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        validation_strictness = "window-level; overlapping clip windows may cross splits"
    elif args.split_strategy == "source":
        splits = stratified_split_by_group(
            all_samples, source_group_by_sample_id,
            args.train_ratio, args.val_ratio, args.test_ratio, args.seed,
        )
        validation_strictness = (
            "source-level when source_group metadata is accurate; recommended for manuscripts"
        )
    elif args.split_strategy == "explicit":
        splits = split_explicit(all_samples, explicit_split_by_sample_id)
        validation_strictness = "explicit user-provided split"
    else:
        splits = stratified_split_by_group(
            all_samples, clip_group_by_sample_id,
            args.train_ratio, args.val_ratio, args.test_ratio, args.seed,
        )
        validation_strictness = (
            "clip-level; possible source-recording leakage if clips share original videos"
        )

    metadata = {
        "schema_version": "behaviorscope-n-v1",
        "annotation_mode": "clip_folder",
        "dense_bout_annotations": bool(args.dense_bout_annotations),
        "source_grouping": str(args.source_grouping),
        "split_strategy": str(args.split_strategy),
        "validation_strictness": validation_strictness,
        "source_dataset_root": str(args.dataset_root),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "crop_size": int(args.crop_size),
        "group_crop_size": int(args.group_crop_size or args.crop_size),
        "n_animals": int(args.n_animals),
        "crop_strategy": "fixed_scale",
        "animal_scale_factor": float(args.animal_scale_factor),
        "group_scale_factor": float(args.group_scale_factor),
        "body_length_px": float(args.body_length_px or 0.0),
        "any_keypoints": keypoints_per_animal > 0,
        "keypoints_per_animal": keypoints_per_animal,
        "yolo_weights": str(args.yolo_weights),
        "clips_total": int(len(clips)),
        "clips_with_windows": int(sum(1 for count in clip_window_counts.values() if count > 0)),
        "generated_by": "BehaviorScope-Y/prepare_clips_x.py",
    }

    write_manifest(manifest_path, splits, class_to_idx, metadata, extra_by_sample_id)
    logger.info(f"Wrote manifest -> {manifest_path}")

    summary = {
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "elapsed_s": round(time.time() - t0, 3),
        "windows_total": int(len(all_samples)),
        "clips_total": int(len(clips)),
        "clips_with_windows": int(sum(1 for count in clip_window_counts.values() if count > 0)),
        "keypoints_per_animal": int(keypoints_per_animal),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "class_counts": dict(sorted(Counter(s.class_name for s in all_samples).items())),
        "validation_strictness": validation_strictness,
    }
    if args.validate_manifest:
        validation = validate_manifest(manifest_path, int(args.validation_sample_limit))
        summary["manifest_validation"] = validation
        logger.info(
            "[validate] ok "
            f"samples={validation['samples_total']} checked={validation['samples_validated']}"
        )

    summary_path = output_root / "prepare_clips_x_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        print(f"[prepare-x] ERROR: {exc}", file=sys.stderr)
        raise



