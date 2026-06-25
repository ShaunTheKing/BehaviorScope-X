from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter, deque
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

THIS = Path(__file__).resolve().parent
APP_ROOT = THIS.parents[2]
sys.path.insert(0, str(APP_ROOT))

from cropping_x import (  # noqa: E402
    NCropDetection,
    compute_fixed_square_xyxy,
    crop_coverage_conf,
    crop_resize_pad_rgb,
)
from data_x import load_n_manifest, validate_n_npz_payload  # noqa: E402
from prepare_full_video_npz import (  # noqa: E402
    DEFAULT_FPS,
    MARS_CLASSES,
    REL_FEATURE_DIM,
    SCHEMA_VERSION,
    _AsyncNPZWriter,
    _build_payload,
    _delete_existing_video_npzs,
    _safe_stem,
    _stable_video_seed,
    build_class_maps,
    discover_sources,
    discover_sources_from_manifest,
    expand_gt_to_frames_for_classes,
    load_class_names,
    parse_annot_for_classes,
    window_label,
    write_manifest,
)
from utils.logging_utils import setup_logger  # noqa: E402
from utils.pose_features_x import extract_relational_features  # noqa: E402

from deeplabcut.core.config import read_config_as_dict  # noqa: E402
from deeplabcut.pose_estimation_pytorch.apis.utils import get_inference_runners  # noqa: E402


BUILD_KIND = "full_video_sliding_dlc_topdown"
MARKER_VERSION = "behaviorscope-x-dlc-topdown-build-v1"


def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32)
    out = np.zeros_like(boxes, dtype=np.float32)
    if boxes.size:
        out[:, 0] = boxes[:, 0]
        out[:, 1] = boxes[:, 1]
        out[:, 2] = boxes[:, 0] + np.maximum(boxes[:, 2], 0.0)
        out[:, 3] = boxes[:, 1] + np.maximum(boxes[:, 3], 0.0)
    return out


def _clip_xyxy(boxes: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    h, w = shape[:2]
    out = np.asarray(boxes, dtype=np.float32).copy()
    if out.size:
        out[:, [0, 2]] = np.clip(out[:, [0, 2]], 0, w)
        out[:, [1, 3]] = np.clip(out[:, [1, 3]], 0, h)
    return out


def _bbox_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    inter = np.maximum(ix2 - ix1, 0.0) * np.maximum(iy2 - iy1, 0.0)
    area_a = np.maximum(a[:, 2:3] - a[:, 0:1], 0.0) * np.maximum(a[:, 3:4] - a[:, 1:2], 0.0)
    area_b = np.maximum(b[:, 2] - b[:, 0], 0.0) * np.maximum(b[:, 3] - b[:, 1], 0.0)
    denom = area_a + area_b - inter
    return np.where(denom > 1e-9, inter / denom, 0.0).astype(np.float32)


def _select_detections(det: dict, n_animals: int, conf_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    boxes = np.asarray(det.get("bboxes", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
    scores = np.asarray(det.get("bbox_scores", np.zeros((0,), dtype=np.float32)), dtype=np.float32)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        boxes = np.zeros((0, 4), dtype=np.float32)
    if scores.ndim != 1 or scores.shape[0] != boxes.shape[0]:
        scores = np.ones((boxes.shape[0],), dtype=np.float32)
    keep = scores >= float(conf_threshold)
    boxes, scores = boxes[keep], scores[keep]
    order = np.argsort(-scores)[: int(n_animals)]
    return boxes[order].astype(np.float32), scores[order].astype(np.float32)


def _assign_slots(
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    keypoints: np.ndarray,
    n_animals: int,
    prev_boxes: np.ndarray | None,
    prev_present: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = int(keypoints.shape[1]) if keypoints.ndim == 3 else 0
    out_boxes = np.zeros((n_animals, 4), dtype=np.float32)
    out_scores = np.zeros((n_animals,), dtype=np.float32)
    out_kxy = np.zeros((n_animals, k, 2), dtype=np.float32)
    out_kconf = np.zeros((n_animals, k), dtype=np.float32)
    present = np.zeros((n_animals,), dtype=bool)

    if boxes_xyxy.shape[0] == 0:
        return out_boxes, out_scores, out_kxy, out_kconf, present

    used_det: set[int] = set()
    if prev_boxes is not None and prev_present is not None and bool(np.any(prev_present)):
        iou = _bbox_iou_matrix(prev_boxes.astype(np.float32), boxes_xyxy.astype(np.float32))
        for slot in np.argsort(-prev_present.astype(np.int32)):
            slot = int(slot)
            if not prev_present[slot]:
                continue
            candidates = [(float(iou[slot, j]), j) for j in range(boxes_xyxy.shape[0]) if j not in used_det]
            if not candidates:
                continue
            best_iou, best_j = max(candidates, key=lambda item: item[0])
            if best_iou <= 0.0:
                continue
            out_boxes[slot] = boxes_xyxy[best_j]
            out_scores[slot] = scores[best_j]
            if k:
                out_kxy[slot] = keypoints[best_j, :, :2]
                out_kconf[slot] = keypoints[best_j, :, 2]
            present[slot] = True
            used_det.add(best_j)

    remaining = [j for j in range(boxes_xyxy.shape[0]) if j not in used_det]
    if prev_boxes is None:
        centers_x = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) * 0.5
        remaining = sorted(remaining, key=lambda j: float(centers_x[j]))

    for j in remaining:
        empty = np.where(~present)[0]
        if empty.size == 0:
            break
        slot = int(empty[0])
        out_boxes[slot] = boxes_xyxy[j]
        out_scores[slot] = scores[j]
        if k:
            out_kxy[slot] = keypoints[j, :, :2]
            out_kconf[slot] = keypoints[j, :, 2]
        present[slot] = True

    return out_boxes, out_scores, out_kxy, out_kconf, present


def _make_detection(
    frame_idx: int,
    frame_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    kxy: np.ndarray,
    kconf: np.ndarray,
    present: np.ndarray,
    args: argparse.Namespace,
    prev_kxy: np.ndarray | None,
    prev_boxes: np.ndarray | None,
    first_seen: dict[int, int],
) -> NCropDetection:
    n_animals = int(args.n_animals)
    crop_present = present.copy()
    centers = np.array([(bb[0] + bb[2], bb[1] + bb[3]) for bb in boxes_xyxy], dtype=np.float32) * 0.5
    valid_centers = centers[crop_present]
    if valid_centers.size:
        group_center = valid_centers.mean(axis=0)
    else:
        h, w = frame_bgr.shape[:2]
        group_center = np.array([w / 2.0, h / 2.0], dtype=np.float32)

    body_len = float(args.body_length_px or 0.0)
    if body_len <= 0:
        valid = boxes_xyxy[crop_present]
        if valid.size:
            widths = valid[:, 2] - valid[:, 0]
            heights = valid[:, 3] - valid[:, 1]
            body_len = float(np.nanmedian(np.maximum(widths, heights)))
        else:
            body_len = float(args.crop_size)
        body_len = max(body_len, 1.0)

    group_xyxy = compute_fixed_square_xyxy(
        frame_bgr.shape,
        (float(group_center[0]), float(group_center[1])),
        body_len * float(args.group_scale_factor),
    )
    animal_xyxys = np.array(
        [
            compute_fixed_square_xyxy(
                frame_bgr.shape,
                (float(centers[i, 0]), float(centers[i, 1])),
                body_len * float(args.animal_scale_factor),
            )
            for i in range(n_animals)
        ],
        dtype=np.int32,
    )

    group_rgb = crop_resize_pad_rgb(frame_bgr, group_xyxy, int(args.group_crop_size or args.crop_size))
    animal_rgbs = np.stack(
        [crop_resize_pad_rgb(frame_bgr, tuple(xyxy), int(args.crop_size)) for xyxy in animal_xyxys],
        axis=0,
    )

    pose_conf = np.zeros((n_animals,), dtype=np.float32)
    if kconf.size:
        pose_conf = np.nan_to_num(kconf.mean(axis=1), nan=0.0).astype(np.float32)
    pose_conf = np.where(present, pose_conf, 0.0).astype(np.float32)
    pose_mask = (pose_conf >= float(args.pose_conf_threshold)) & present
    crop_conf = np.array([crop_coverage_conf(tuple(xyxy), frame_bgr.shape) for xyxy in animal_xyxys], dtype=np.float32)
    track_conf = np.where(present, scores, 0.0).astype(np.float32)
    track_ids = np.arange(n_animals, dtype=np.int32)
    track_age = np.zeros((n_animals,), dtype=np.float32)
    for i in range(n_animals):
        if present[i]:
            first_seen.setdefault(i, int(frame_idx))
            track_age[i] = min(float(frame_idx - first_seen[i] + 1) / 30.0, 1.0)

    keypoints_crop_norm = np.zeros((n_animals, kxy.shape[1], 3), dtype=np.float32)
    if kxy.ndim == 3 and kxy.shape[1] > 0:
        for i in range(n_animals):
            x1, y1, x2, y2 = [float(v) for v in animal_xyxys[i]]
            cw = max(x2 - x1, 1.0)
            ch = max(y2 - y1, 1.0)
            keypoints_crop_norm[i, :, 0] = (kxy[i, :, 0] - x1) / cw
            keypoints_crop_norm[i, :, 1] = (kxy[i, :, 1] - y1) / ch
            keypoints_crop_norm[i, :, 2] = kconf[i] if kconf.ndim == 2 else 0.0

    rel, rel_pose_mask, rel_present = extract_relational_features(
        kxy,
        boxes_xyxy,
        present,
        pose_mask,
        body_len,
        prev_keypoints_raw=prev_kxy,
        prev_bboxes=prev_boxes,
        fps=float(args.fps),
        pose_conf_threshold=float(args.pose_conf_threshold),
        keypoints_conf_raw=kconf if kconf.size else None,
    )

    return NCropDetection(
        frame_idx=int(frame_idx),
        group_rgb=group_rgb,
        animal_rgbs=animal_rgbs,
        animal_mask=present.astype(bool),
        pose_mask=pose_mask.astype(bool),
        pose_conf=pose_conf,
        crop_conf=crop_conf,
        track_conf=track_conf,
        track_age=track_age,
        relation_features=rel,
        relation_pose_mask=rel_pose_mask,
        relation_present=rel_present,
        bbox_xyxy_animals=boxes_xyxy.astype(np.int32),
        bbox_xyxy_group=tuple(int(v) for v in group_xyxy),
        crop_xyxy_animals=animal_xyxys.astype(np.int32),
        crop_xyxy_group=tuple(int(v) for v in group_xyxy),
        track_ids=track_ids,
        keypoints_xy_raw=kxy.astype(np.float32),
        keypoints_conf_raw=kconf.astype(np.float32),
        keypoints_crop_norm=keypoints_crop_norm.astype(np.float32),
        yolo_status="dlc_topdown_detected" if bool(present.any()) else "dlc_topdown_missed",
    )


def iter_dlc_topdown_detections(src_video: Path, pose_runner, detector_runner, args: argparse.Namespace):
    cap = cv2.VideoCapture(str(src_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src_video}")

    frame_idx = 0
    prev_boxes: np.ndarray | None = None
    prev_kxy: np.ndarray | None = None
    prev_present: np.ndarray | None = None
    last_boxes: np.ndarray | None = None
    last_kxy: np.ndarray | None = None
    last_kconf: np.ndarray | None = None
    last_scores: np.ndarray | None = None
    last_present: np.ndarray | None = None
    first_seen: dict[int, int] = {}
    det_batch = max(1, int(args.detector_batch_size))

    try:
        while True:
            frames_bgr: list[np.ndarray] = []
            frames_rgb: list[np.ndarray] = []
            for _ in range(det_batch):
                ok, frame_bgr = cap.read()
                if not ok:
                    break
                frames_bgr.append(frame_bgr)
                frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if not frames_bgr:
                break

            det_outputs = detector_runner.inference(frames_rgb)
            pose_inputs = []
            pose_map = []
            selected: list[tuple[np.ndarray, np.ndarray]] = []
            for i, det in enumerate(det_outputs):
                boxes_xywh, scores = _select_detections(det, int(args.n_animals), float(args.detector_conf))
                selected.append((boxes_xywh, scores))
                if boxes_xywh.shape[0] > 0:
                    pose_map.append(i)
                    pose_inputs.append((frames_rgb[i], {"bboxes": boxes_xywh}))

            pose_outputs = pose_runner.inference(pose_inputs) if pose_inputs else []
            pose_by_frame = {frame_i: pose_outputs[j] for j, frame_i in enumerate(pose_map)}

            for i, frame_bgr in enumerate(frames_bgr):
                boxes_xywh, scores = selected[i]
                bodyparts = np.zeros((boxes_xywh.shape[0], int(args.num_keypoints_resolved), 3), dtype=np.float32)
                if i in pose_by_frame:
                    bodyparts = np.asarray(pose_by_frame[i].get("bodyparts", bodyparts), dtype=np.float32)
                boxes_xyxy = _clip_xyxy(_xywh_to_xyxy(boxes_xywh), frame_bgr.shape)
                boxes, slot_scores, kxy, kconf, present = _assign_slots(
                    boxes_xyxy,
                    scores,
                    bodyparts,
                    int(args.n_animals),
                    prev_boxes,
                    prev_present,
                )

                if not present.any() and bool(args.keep_last_box) and last_boxes is not None:
                    boxes = last_boxes.copy()
                    kxy = last_kxy.copy() if last_kxy is not None else kxy
                    kconf = last_kconf.copy() if last_kconf is not None else kconf
                    slot_scores = last_scores.copy() if last_scores is not None else slot_scores
                    present = np.zeros_like(last_present, dtype=bool) if last_present is not None else present
                elif present.any():
                    last_boxes = boxes.copy()
                    last_kxy = kxy.copy()
                    last_kconf = kconf.copy()
                    last_scores = slot_scores.copy()
                    last_present = present.copy()

                det = _make_detection(
                    frame_idx,
                    frame_bgr,
                    boxes,
                    slot_scores,
                    kxy,
                    kconf,
                    present,
                    args,
                    prev_kxy,
                    prev_boxes,
                    first_seen,
                )
                prev_boxes = boxes.copy()
                prev_kxy = kxy.copy()
                prev_present = present.copy()
                yield det
                frame_idx += 1
    finally:
        cap.release()


def _marker_path(output_root: Path, safe_stem: str) -> Path:
    return Path(output_root) / "build_state" / "completed_videos" / f"{safe_stem}.json"


def _signature(src, args: argparse.Namespace, class_names: list[str]) -> dict:
    return {
        "version": MARKER_VERSION,
        "video_id": src.video_id,
        "mars_root": str(Path(args.mars_root).resolve()),
        "pose_config": str(Path(args.pose_config).resolve()),
        "pose_snapshot": str(Path(args.pose_snapshot).resolve()),
        "detector_snapshot": str(Path(args.detector_snapshot).resolve()),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "class_names": list(class_names),
        "n_animals": int(args.n_animals),
        "num_keypoints": int(args.num_keypoints_resolved),
        "crop_size": int(args.crop_size),
        "group_crop_size": int(args.group_crop_size or args.crop_size),
        "animal_scale_factor": float(args.animal_scale_factor),
        "group_scale_factor": float(args.group_scale_factor),
        "body_length_px": None if args.body_length_px is None else float(args.body_length_px),
        "detector_conf": float(args.detector_conf),
        "pose_conf_threshold": float(args.pose_conf_threshold),
        "label_min_dominance": float(args.label_min_dominance),
        "other_subsample": float(args.other_subsample),
        "seed": int(args.seed),
    }


def _load_marker(output_root: Path, safe_stem: str, src, args: argparse.Namespace, class_names: list[str], logger):
    marker = _marker_path(output_root, safe_stem)
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning(f"[{src.video_id}] ignoring unreadable marker {marker}: {exc}")
        return None
    if payload.get("signature") != _signature(src, args, class_names):
        logger.info(f"[{src.video_id}] marker exists but build signature changed; rebuilding")
        return None
    samples = list(payload.get("samples") or [])
    missing = [s.get("sequence_npz", "") for s in samples if not (Path(output_root) / s.get("sequence_npz", "")).is_file()]
    if missing:
        logger.warning(f"[{src.video_id}] marker has missing NPZ files; rebuilding. First: {missing[0]}")
        return None
    cls_counts = Counter(s.get("class_name", "") for s in samples)
    cls_counts.pop("", None)
    return samples, cls_counts


def _write_marker(output_root: Path, safe_stem: str, src, args: argparse.Namespace, class_names: list[str], samples, cls_counts):
    marker = _marker_path(output_root, safe_stem)
    marker.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": MARKER_VERSION,
        "completed_at_unix": time.time(),
        "signature": _signature(src, args, class_names),
        "n_samples": len(samples),
        "class_counts": dict(cls_counts),
        "samples": samples,
    }
    marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_video(src, pose_runner, detector_runner, args: argparse.Namespace, output_root: Path, writer, logger):
    class_names = list(args.class_names_resolved)
    class_to_idx = dict(args.class_to_idx_resolved)
    rng = random.Random(_stable_video_seed(int(args.seed), src.video_id))
    fps = float(src.fps_override or args.fps)
    safe_stem = _safe_stem(src.video_id)
    seq_dir = output_root / "sequence_npz"
    seq_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.skip_existing):
        existing = _load_marker(output_root, safe_stem, src, args, class_names, logger)
        if existing is not None:
            samples, cls_counts = existing
            logger.info(f"[{src.video_id}] skip_existing: reusing {len(samples)} manifest entries")
            return samples, cls_counts

    existing_count = sum(1 for cls_dir in seq_dir.glob("*") if cls_dir.is_dir() for _ in cls_dir.glob(f"{safe_stem}_*.npz"))
    if existing_count and bool(args.clean_incomplete_existing):
        deleted = _delete_existing_video_npzs(seq_dir, safe_stem)
        logger.warning(f"[{src.video_id}] removed {deleted} stale NPZs before rebuilding")

    if src.source_path is not None:
        video_path = Path(src.source_path)
    elif src.seq_path is not None:
        video_path = src.video_dir / f"{src.seq_path.stem}.mp4"
    else:
        raise RuntimeError(f"{src.video_id}: no source video path available")
    if not video_path.is_file():
        raise RuntimeError(f"{src.video_id}: expected MP4 not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if n_frames <= 0:
        logger.warning(f"[{src.video_id}] no readable frames; skipping")
        return [], Counter()

    header, bouts, unknown = parse_annot_for_classes(Path(src.annot_path), class_to_idx)
    if unknown:
        logger.warning(f"[{src.video_id}] ignored annotation labels not in class_names: {sorted(unknown)}")
    annot_fps = float(header.get("fps") or fps)
    label_vec = expand_gt_to_frames_for_classes(bouts, n_frames, annot_fps, class_to_idx, class_names)

    buffer: deque = deque()
    next_window_t = 0
    samples: list[dict] = []
    cls_counts: Counter = Counter()
    n_emitted = 0
    n_dropped_dominance = 0
    n_dropped_subsample = 0
    n_skipped_missing = 0
    t_video_start = time.time()
    t_last_progress = t_video_start

    def emit_ready() -> bool:
        nonlocal next_window_t, n_emitted, n_dropped_dominance, n_dropped_subsample, n_skipped_missing
        while buffer:
            t = next_window_t
            t_end = t + int(args.window_size) - 1
            if buffer[-1][0] < t_end:
                return True
            while buffer and buffer[0][0] < t:
                buffer.popleft()
            window_frames = []
            expected = t
            ok = True
            for fi, det in buffer:
                if fi > t_end:
                    break
                if fi != expected:
                    ok = False
                    break
                window_frames.append(det)
                expected += 1
            if not ok or len(window_frames) != int(args.window_size):
                n_skipped_missing += 1
                next_window_t += int(args.window_stride)
                continue
            cls = window_label(label_vec, t, int(args.window_size), float(args.label_min_dominance), class_names)
            if cls is None:
                n_dropped_dominance += 1
                next_window_t += int(args.window_stride)
                continue
            if cls == "other" and rng.random() > float(args.other_subsample):
                n_dropped_subsample += 1
                next_window_t += int(args.window_stride)
                continue

            seq_name = f"{safe_stem}_f{t:06d}"
            npz_path = seq_dir / cls / f"{seq_name}.npz"
            writer.submit(npz_path, _build_payload(window_frames, args))
            samples.append(
                {
                    "id": f"{cls}_{seq_name}",
                    "label": int(class_to_idx[cls]),
                    "class_name": cls,
                    "sequence_npz": str(npz_path.relative_to(output_root)).replace("\\", "/"),
                    "start_frame": int(t),
                    "end_frame": int(t + int(args.window_size) - 1),
                    "fps": float(fps),
                    "source_video": src.video_id,
                    "mars_split": src.mars_split,
                    "dataset_split": src.split,
                }
            )
            cls_counts[cls] += 1
            n_emitted += 1
            next_window_t += int(args.window_stride)
            if int(args.max_windows_per_video) > 0 and n_emitted >= int(args.max_windows_per_video):
                return False
        return True

    for det in iter_dlc_topdown_detections(video_path, pose_runner, detector_runner, args):
        buffer.append((int(det.frame_idx), det))
        while buffer and buffer[0][0] < next_window_t:
            buffer.popleft()
        if not emit_ready():
            break
        now = time.time()
        if float(args.progress_interval_s) > 0 and (now - t_last_progress) >= float(args.progress_interval_s):
            frames_done = int(det.frame_idx) + 1
            elapsed = max(now - t_video_start, 1e-6)
            fps_now = frames_done / elapsed
            pct = 100.0 * frames_done / max(int(n_frames), 1)
            frames_left = max(int(n_frames) - frames_done, 0)
            eta_s = frames_left / max(fps_now, 1e-6)
            logger.info(
                f"[{src.video_id}] progress frame={frames_done}/{n_frames} "
                f"({pct:.1f}%) fps={fps_now:.2f} eta={eta_s/3600:.2f}h "
                f"windows={n_emitted} per_cls={dict(cls_counts)}"
            )
            t_last_progress = now
    emit_ready()

    if not samples:
        logger.warning(
            f"[{src.video_id}] no windows emitted "
            f"(dropped_dom={n_dropped_dominance}, dropped_other_sub={n_dropped_subsample}, missing={n_skipped_missing})"
        )
        return [], Counter()

    writer.flush()
    _write_marker(output_root, safe_stem, src, args, class_names, samples, cls_counts)
    logger.info(
        f"[{src.video_id}] frames={n_frames} emitted={n_emitted} "
        f"dropped_dom={n_dropped_dominance} dropped_other_sub={n_dropped_subsample} "
        f"missing={n_skipped_missing} per_cls={dict(cls_counts)}"
    )
    return samples, cls_counts


def _default_pose_config() -> Path:
    return APP_ROOT / "dlc-models-pytorch" / "iteration-0" / "MARS_DLC_SuperAnimal-trainset95shuffle2" / "train" / "pytorch_config.yaml"


def _default_snapshot(name: str) -> Path:
    return APP_ROOT / "dlc-models-pytorch" / "iteration-0" / "MARS_DLC_SuperAnimal-trainset95shuffle2" / "train" / name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mars_root", type=Path, default=APP_ROOT / "MARS_data")
    p.add_argument("--source_manifest_csv", type=Path, default=None)
    p.add_argument("--train_splits", nargs="+", default=["train"])
    p.add_argument("--val_splits", nargs="+", default=["validation"])
    p.add_argument("--exclude_splits", nargs="+", default=["test_1", "test_2"])
    p.add_argument("--class_names", nargs="+", default=list(MARS_CLASSES))
    p.add_argument("--class_names_file", type=Path, default=None)
    p.add_argument("--output_root", type=Path, default=APP_ROOT / "outputs" / "npz_cache" / "mars_full_video_dlc_topdown")
    p.add_argument("--manifest_path", type=Path, default=None)
    p.add_argument("--pose_config", type=Path, default=_default_pose_config())
    p.add_argument("--pose_snapshot", type=Path, default=_default_snapshot("snapshot-best.pt"))
    p.add_argument("--detector_snapshot", type=Path, default=_default_snapshot("snapshot-detector-best.pt"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--pose_batch_size", type=int, default=16)
    p.add_argument("--detector_batch_size", type=int, default=2)
    p.add_argument("--detector_conf", type=float, default=0.25)
    p.add_argument("--window_size", type=int, default=32)
    p.add_argument("--window_stride", type=int, default=16)
    p.add_argument("--n_animals", type=int, default=2)
    p.add_argument("--num_keypoints", type=int, default=7)
    p.add_argument("--crop_size", type=int, default=224)
    p.add_argument("--group_crop_size", type=int, default=224)
    p.add_argument("--animal_scale_factor", type=float, default=4.0)
    p.add_argument("--group_scale_factor", type=float, default=8.0)
    p.add_argument("--body_length_px", type=float, default=None)
    p.add_argument("--pose_conf_threshold", type=float, default=0.3)
    p.add_argument("--fps", type=float, default=DEFAULT_FPS)
    p.add_argument("--label_min_dominance", type=float, default=0.5)
    p.add_argument("--other_subsample", type=float, default=0.30)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--npz_writers", type=int, default=3)
    p.add_argument("--npz_compression", choices=["deflated", "stored"], default="deflated")
    p.add_argument("--npz_compresslevel", type=int, default=1)
    p.add_argument("--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--clean_incomplete_existing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--keep_last_box", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--validate_manifest", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--validation_sample_limit", type=int, default=64)
    p.add_argument("--progress_interval_s", type=float, default=30.0)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--max_videos", type=int, default=0, help="Smoke-test limit across all sources. 0 processes all discovered videos.")
    p.add_argument("--max_videos_per_split", type=int, default=0, help="Smoke-test limit per train/val split. 0 disables this limiter.")
    p.add_argument("--max_windows_per_video", type=int, default=0, help="Smoke-test limit. 0 emits all windows.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    class_names, class_to_idx = build_class_maps(load_class_names(args))
    args.class_names_resolved = class_names
    args.class_to_idx_resolved = class_to_idx
    args.num_keypoints_resolved = int(args.num_keypoints)

    output_root = Path(args.output_root).resolve()
    manifest_path = (args.manifest_path or (output_root / "sequence_manifest.json")).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("DlcTopdownFullVideoNPZ", output_root / "dlc_topdown_full_video_npz.log")

    for path_attr in ("pose_config", "pose_snapshot", "detector_snapshot"):
        path = Path(getattr(args, path_attr))
        if not path.is_file():
            raise SystemExit(f"[dlc-topdown] {path_attr} not found: {path}")

    if args.source_manifest_csv is not None:
        sources = discover_sources_from_manifest(args.source_manifest_csv, args.exclude_splits, logger)
    else:
        sources = discover_sources(args.mars_root, args.train_splits, args.val_splits, args.exclude_splits, logger)
    if int(args.max_videos_per_split) > 0:
        limit = int(args.max_videos_per_split)
        limited = []
        counts = Counter()
        for src in sources:
            if counts[src.split] >= limit:
                continue
            limited.append(src)
            counts[src.split] += 1
        sources = limited
    if int(args.max_videos) > 0:
        sources = sources[: int(args.max_videos)]
    if not sources:
        raise SystemExit("[dlc-topdown] no sources discovered")

    logger.info(f"[dlc-topdown] sources: {len(sources)}")
    logger.info(f"[dlc-topdown] output_root: {output_root}")
    logger.info(f"[dlc-topdown] pose_snapshot: {args.pose_snapshot}")
    logger.info(f"[dlc-topdown] detector_snapshot: {args.detector_snapshot}")
    logger.info(f"[dlc-topdown] npz_compression: {args.npz_compression} level={args.npz_compresslevel}")

    if args.dry_run:
        for src in sources:
            logger.info(f"[dry] {src.split}/{src.video_id} annot={src.annot_path}")
        return 0

    model_config = read_config_as_dict(str(args.pose_config))
    pose_runner, detector_runner = get_inference_runners(
        model_config,
        snapshot_path=str(args.pose_snapshot),
        detector_path=str(args.detector_snapshot),
        batch_size=int(args.pose_batch_size),
        detector_batch_size=int(args.detector_batch_size),
        device=str(args.device),
        max_individuals=int(args.n_animals),
    )
    if detector_runner is None:
        raise SystemExit("[dlc-topdown] detector runner was not created")

    writer = _AsyncNPZWriter(
        num_workers=int(args.npz_writers),
        compression=str(args.npz_compression),
        compresslevel=int(args.npz_compresslevel),
    )
    splits_out = {"train": [], "val": []}
    class_counts_total: Counter = Counter()
    per_video_rows: list[dict] = []
    t0 = time.time()
    try:
        for i, src in enumerate(sources):
            logger.info(f"[dlc-topdown] [{i + 1}/{len(sources)}] {src.split}/{src.video_id}")
            try:
                samples, cls_counts = process_video(src, pose_runner, detector_runner, args, output_root, writer, logger)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                logger.error(f"[dlc-topdown] {src.video_id}: process failed: {exc}", exc_info=True)
                per_video_rows.append({"split": src.split, "video_id": src.video_id, "status": "error", "error": str(exc)})
                continue
            splits_out[src.split].extend(samples)
            class_counts_total.update(cls_counts)
            per_video_rows.append(
                {
                    "split": src.split,
                    "video_id": src.video_id,
                    "status": "ok",
                    "n_windows": len(samples),
                    **{f"n_{c}": int(cls_counts.get(c, 0)) for c in class_names},
                }
            )
    finally:
        writer.close()

    n_train, n_val = len(splits_out["train"]), len(splits_out["val"])
    if n_train == 0 or n_val == 0:
        raise SystemExit(f"[dlc-topdown] empty split after build: train={n_train} val={n_val}")

    elapsed = time.time() - t0
    meta = {
        "schema_version": SCHEMA_VERSION,
        "build_kind": BUILD_KIND,
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "n_animals": int(args.n_animals),
        "keypoints_per_animal": int(args.num_keypoints_resolved),
        "class_names": list(class_names),
        "label_min_dominance": float(args.label_min_dominance),
        "other_subsample": float(args.other_subsample),
        "npz_compression": str(args.npz_compression),
        "npz_compresslevel": int(args.npz_compresslevel),
        "dlc_pose_config": str(Path(args.pose_config).resolve()),
        "dlc_pose_snapshot": str(Path(args.pose_snapshot).resolve()),
        "dlc_detector_snapshot": str(Path(args.detector_snapshot).resolve()),
        "detector_conf": float(args.detector_conf),
        "pose_batch_size": int(args.pose_batch_size),
        "detector_batch_size": int(args.detector_batch_size),
        "mars_root": str(Path(args.mars_root).resolve()),
        "source_manifest_csv": str(args.source_manifest_csv) if args.source_manifest_csv else None,
        "train_splits": list(args.train_splits),
        "val_splits": list(args.val_splits),
        "exclude_splits": list(args.exclude_splits),
        "build_elapsed_s": round(elapsed, 1),
        "class_counts": dict(class_counts_total),
        "n_source_videos_train": len({s["source_video"] for s in splits_out["train"]}),
        "n_source_videos_val": len({s["source_video"] for s in splits_out["val"]}),
    }
    write_manifest(manifest_path, splits_out, meta, class_to_idx)
    logger.info(f"[dlc-topdown] manifest -> {manifest_path}")

    if per_video_rows:
        csv_path = output_root / "per_video_build_summary.csv"
        fieldnames = sorted({key for row in per_video_rows for key in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer_csv = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer_csv.writeheader()
            writer_csv.writerows(per_video_rows)
        logger.info(f"[dlc-topdown] per-video summary -> {csv_path}")

    summary = {
        "manifest_path": str(manifest_path),
        "output_root": str(output_root),
        "elapsed_s": round(elapsed, 1),
        "windows_total": n_train + n_val,
        "windows_train": n_train,
        "windows_val": n_val,
        "class_counts": dict(class_counts_total),
        "n_source_videos_train": meta["n_source_videos_train"],
        "n_source_videos_val": meta["n_source_videos_val"],
    }
    if bool(args.validate_manifest):
        try:
            splits_loaded, _ci, _ic, _meta = load_n_manifest(manifest_path)
            all_samples = splits_loaded["train"] + splits_loaded["val"]
            n_check = min(int(args.validation_sample_limit), len(all_samples))
            random.seed(int(args.seed))
            for sample in random.sample(all_samples, n_check):
                with np.load(sample.sequence_npz, allow_pickle=False) as data:
                    validate_n_npz_payload(
                        data,
                        sample_id=sample.id,
                        num_frames=int(args.window_size),
                        n_animals=int(args.n_animals),
                        num_keypoints=int(args.num_keypoints_resolved),
                        rel_feature_dim=REL_FEATURE_DIM,
                        require_schema_version=True,
                    )
            summary["manifest_validation"] = {"ok": True, "samples_validated": n_check}
        except Exception as exc:
            summary["manifest_validation"] = {"ok": False, "error": str(exc)}
            (output_root / "dlc_topdown_full_video_npz_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            raise

    summary_path = output_root / "dlc_topdown_full_video_npz_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(f"[dlc-topdown] summary -> {summary_path}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


