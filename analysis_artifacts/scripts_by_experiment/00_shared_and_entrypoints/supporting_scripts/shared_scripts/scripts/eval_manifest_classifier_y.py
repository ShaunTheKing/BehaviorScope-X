#!/usr/bin/env python3
"""Classifier-only held-out evaluation from cached BehaviorScope-Y windows.

This is the normal final-run evaluator for MARS held-out videos after the
held-out NPZ window cache and frozen YOLO visual-feature cache have been built
once. It does not run YOLO detection or the YOLO visual trunk per model.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from torch.amp import autocast
except ImportError:  # torch < 2.0
    from torch.cuda.amp import autocast  # type: ignore[no-redef]

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

from data_y import (  # noqa: E402
    MultiAnimalSequenceDataset,
    collate_multi_animal,
    load_n_manifest,
)
from infer_y import (  # noqa: E402
    _write_smoothed_frames_csv,
    _resolve_amp_dtype,
    apply_threshold_decoder_probs,
    load_model_from_checkpoint,
    load_threshold_decoder_config,
)
from batch_infer_eval import (  # noqa: E402
    CLASSES,
    discover_jobs,
    evaluate_prediction_set,
    write_csv,
)
from utils.pose_features_y import REL_FEATURE_DIM  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained BehaviorScope-Y head from held-out NPZ/feature caches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--feature_cache_dir", required=True)
    p.add_argument("--mars_root", required=True)
    p.add_argument("--splits", nargs="+", default=["test_1", "test_2"])
    p.add_argument("--model_path", required=True)
    p.add_argument("--model_config", default=None)
    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", choices=["auto", "fp16", "bf16"], default="auto")
    p.add_argument("--temporal_smoothing_window", type=int, default=5)
    p.add_argument("--bout_min_duration_frames", type=int, default=15)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--allow_pred_named_annots", action=argparse.BooleanOptionalAction, default=False,
                   help="Accept .annot files whose names contain pred/prediction. Use only when these are approved labels.")
    p.add_argument("--disable_threshold_decoder", action="store_true",
                   help="Use argmax probabilities for controlled comparisons instead of the validation-fitted threshold decoder.")
    return p.parse_args()


def _load_raw_manifest(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for split_name, rows in payload.get("splits", {}).items():
        for row in rows:
            item = dict(row)
            item.setdefault("dataset_split", split_name)
            out[str(item["id"])] = item
    return out


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text)).strip("_") or "video"


def _write_window_csv(path: Path, rows: list[dict], class_names: list[str]) -> None:
    fieldnames = (
        ["window_index", "frame_start", "frame_end", "time_s", "predicted_class",
         "predicted_class_id", "class_conf"]
        + [f"prob_{c}" for c in class_names]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    t0 = time.time()
    manifest_path = Path(args.manifest_path).resolve()
    feature_cache_dir = Path(args.feature_cache_dir).resolve()
    mars_root = Path(args.mars_root).resolve()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, cfg, idx_to_class, num_classes, n_animals, num_keypoints, checkpoint = load_model_from_checkpoint(
        args.model_path,
        args.model_config,
        device=str(device),
        yolo_weights_fallback=str(args.yolo_weights),
    )
    class_names = [str(idx_to_class[i]) for i in range(num_classes)]
    if class_names != CLASSES:
        raise SystemExit(f"class order mismatch: model={class_names}, evaluator={CLASSES}")

    threshold_decoder = load_threshold_decoder_config(
        args.model_path,
        args.model_config,
        cfg,
        decoder_config_path=None,
        disable_decoder=bool(args.disable_threshold_decoder),
        checkpoint=checkpoint,
    )

    splits, _class_to_idx, _idx_to_class, meta = load_n_manifest(manifest_path)
    raw_lookup = _load_raw_manifest(manifest_path)
    all_samples = []
    for split_samples in splits.values():
        all_samples.extend(split_samples)
    selected = [
        sample for sample in all_samples
        if str(raw_lookup.get(sample.id, {}).get("mars_split", "")) in set(args.splits)
    ]
    if not selected:
        raise SystemExit(f"No manifest samples matched held-out splits {args.splits}")

    jobs_by_key = {
        (j.split, j.video_id): j
        for j in discover_jobs(
            mars_root,
            args.splits,
            allow_pred_named_annots=bool(args.allow_pred_named_annots),
        )
    }
    selected_by_video: dict[tuple[str, str], list] = defaultdict(list)
    for sample in selected:
        meta_row = raw_lookup.get(sample.id, {})
        split = str(meta_row.get("mars_split", ""))
        selected_by_video[(split, sample.source_video)].append(sample)

    visual_alive = bool(getattr(model, "_visual_alive", False))
    dataset = MultiAnimalSequenceDataset(
        samples=selected,
        num_frames=int(meta.get("window_size", 32)),
        n_animals=int(n_animals),
        num_keypoints=int(num_keypoints),
        rel_feature_dim=int(cfg.get("rel_feature_dim", REL_FEATURE_DIM)),
        strict_schema=True,
        feature_cache_dir=feature_cache_dir if visual_alive else None,
        require_feature_cache=visual_alive,
        skip_visual=not visual_alive,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(args.batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_multi_animal,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=2 if int(args.num_workers) > 0 else None,
    )

    amp_enabled = bool(args.amp) and device.type == "cuda"
    amp_dtype = _resolve_amp_dtype(str(args.amp_dtype)) if amp_enabled else None
    amp_dtype_torch = torch.bfloat16 if amp_dtype == "bf16" else torch.float16

    probs_all: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            inputs, _labels = batch
            with autocast("cuda", dtype=amp_dtype_torch, enabled=amp_enabled):
                logits = model(inputs)
                probs = torch.softmax(logits, dim=1).detach().cpu().float().numpy()
            probs_all.append(probs)
    probs_arr = np.concatenate(probs_all, axis=0) if probs_all else np.empty((0, num_classes), dtype=np.float32)
    if probs_arr.shape[0] != len(selected):
        raise RuntimeError(f"prediction count mismatch: probs={probs_arr.shape[0]} samples={len(selected)}")

    by_id_prob = {sample.id: probs_arr[i] for i, sample in enumerate(selected)}

    manifest_rows: list[dict] = []
    failure_rows: list[dict] = []
    summary_rows: list[dict] = []
    per_class_rows_all: list[dict] = []
    gt_bout_rows_all: list[dict] = []
    pred_bout_rows_all: list[dict] = []
    bout_match_rows_all: list[dict] = []

    for (split, video_id), samples in sorted(selected_by_video.items()):
        job = jobs_by_key.get((split, video_id))
        if job is None:
            failure_rows.append({"split": split, "video_id": video_id, "stage": "discover_gt", "error": "no seq/annot job found"})
            continue

        samples = sorted(samples, key=lambda s: (int(s.start_frame), int(s.end_frame), s.id))
        safe_video = _safe_name(video_id)
        video_out = out_root / split / safe_video
        pred_csv = video_out / f"{safe_video}.behavior.csv"
        smoothed_csv = pred_csv.with_suffix(".smoothed_frames.csv")
        if args.skip_existing and pred_csv.is_file() and smoothed_csv.is_file():
            pass
        else:
            rows = []
            window_records = []
            for wi, sample in enumerate(samples):
                probs = by_id_prob[sample.id].astype(np.float64)
                pred_id = int(apply_threshold_decoder_probs(probs, threshold_decoder))
                row = {
                    "window_index": wi,
                    "frame_start": int(sample.start_frame),
                    "frame_end": int(sample.end_frame),
                    "time_s": float(sample.start_frame) / max(float(args.fps), 1e-6),
                    "predicted_class": class_names[pred_id],
                    "predicted_class_id": pred_id,
                    "class_conf": float(probs[pred_id]),
                }
                for ci, cname in enumerate(class_names):
                    row[f"prob_{cname}"] = float(probs[ci])
                rows.append(row)
                window_records.append((int(sample.start_frame), int(sample.end_frame), probs))
            _write_window_csv(pred_csv, rows, class_names)
            _write_smoothed_frames_csv(
                window_records,
                pred_csv,
                class_names,
                idx_to_class,
                float(args.fps),
                int(args.temporal_smoothing_window),
                int(args.bout_min_duration_frames),
                threshold_decoder=threshold_decoder,
                v25_splitter_config=None,
            )

        manifest_rows.append({
            "split": split,
            "video_id": video_id,
            "annot_path": str(job.annot_path),
            "predictions_csv": str(pred_csv),
            "smoothed_csv": str(smoothed_csv),
            "n_windows": len(samples),
        })

        try:
            if smoothed_csv.is_file():
                with smoothed_csv.open("r", newline="", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                frame_col = "frame_idx" if rows and "frame_idx" in rows[0] else "frame_start"
                n_frames = max((int(r[frame_col]) for r in rows), default=-1) + 1
            else:
                n_frames = max((int(s.end_frame) for s in samples), default=-1) + 1
            if n_frames <= 0:
                raise ValueError("could not determine n_frames from prediction CSVs")

            for prediction_type, csv_path in [("raw_windows", pred_csv), ("smoothed_frames", smoothed_csv)]:
                if not csv_path.is_file():
                    continue
                summary, per_class, gt_bouts, pred_bouts, matches = evaluate_prediction_set(
                    split=split,
                    video_id=video_id,
                    prediction_type=prediction_type,
                    predictions_csv=csv_path,
                    annot_path=job.annot_path,
                    n_frames=n_frames,
                    fps=float(args.fps),
                )
                summary_rows.append(summary)
                per_class_rows_all.extend(per_class)
                if prediction_type == "smoothed_frames":
                    gt_bout_rows_all.extend(gt_bouts)
                pred_bout_rows_all.extend(pred_bouts)
                bout_match_rows_all.extend(matches)
        except Exception as exc:
            failure_rows.append({"split": split, "video_id": video_id, "stage": "evaluate", "error": repr(exc)})

    write_csv(out_root / "manifest.csv", manifest_rows)
    write_csv(out_root / "failures.csv", failure_rows)
    write_csv(out_root / "per_video_summary.csv", summary_rows)
    write_csv(out_root / "per_class_metrics.csv", per_class_rows_all)
    write_csv(out_root / "ground_truth_bouts.csv", gt_bout_rows_all)
    write_csv(out_root / "predicted_bouts.csv", pred_bout_rows_all)
    write_csv(out_root / "bout_matches.csv", bout_match_rows_all)
    blocking_failures = [
        row for row in failure_rows
        if str(row.get("stage", "")) != "discover_gt"
    ]
    gt_missing_or_excluded = [
        row for row in failure_rows
        if str(row.get("stage", "")) == "discover_gt"
    ]
    aggregate = {
        "mode": "classifier_only_cached_heldout",
        "manifest_path": str(manifest_path),
        "feature_cache_dir": str(feature_cache_dir),
        "model_path": str(Path(args.model_path).resolve()),
        "n_windows": len(selected),
        "n_manifest_videos": len(selected_by_video),
        "n_evaluated_videos": len(manifest_rows),
        "n_gt_missing_or_excluded_videos": len(gt_missing_or_excluded),
        "n_failures": len(failure_rows),
        "n_blocking_failures": len(blocking_failures),
        "threshold_decoder_disabled": bool(args.disable_threshold_decoder),
        "threshold_decoder_enabled": bool(isinstance(threshold_decoder, dict) and threshold_decoder.get("enabled")),
        "elapsed_s": round(time.time() - t0, 3),
    }
    (out_root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    print(
        f"[classifier-eval] done evaluated_videos={len(manifest_rows)} "
        f"manifest_videos={len(selected_by_video)} windows={len(selected)} "
        f"gt_missing_or_excluded={len(gt_missing_or_excluded)} "
        f"blocking_failures={len(blocking_failures)} "
        f"elapsed={aggregate['elapsed_s']:.1f}s out={out_root}",
        flush=True,
    )
    return 1 if blocking_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
