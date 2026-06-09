#!/usr/bin/env python3
"""Classifier-only Fly-v-Fly held-out evaluation from cached windows.

This mirrors the MARS held-out path: build held-out Fly NPZ windows once,
precompute frozen YOLO/SPPF visual features once, then evaluate each temporal
head from the cached manifest instead of re-running YOLO over the MP4 for every
model. Metrics are delegated to fly_run_all_eval.py so the annotation-aware
primary/secondary/union/intersection/padded-primary modes stay identical to the
existing Fly evaluator.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from torch.amp import autocast
except ImportError:  # torch < 2.0
    from torch.cuda.amp import autocast  # type: ignore[no-redef]

THIS = Path(__file__).resolve().parent
SCRIPTS = THIS / "scripts"
for p in (THIS, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data_x import MultiAnimalSequenceDataset, collate_multi_animal, load_n_manifest  # noqa: E402
from infer_x import (  # noqa: E402
    _resolve_amp_dtype,
    _write_smoothed_frames_csv,
    apply_threshold_decoder_probs,
    load_model_from_checkpoint,
    load_threshold_decoder_config,
)
from utils.pose_features_x import REL_FEATURE_DIM  # noqa: E402
from fly_run_all_eval import (  # noqa: E402
    AGGRESSION_ROOT,
    CLASSES,
    EVAL_PARAMS,
    FLY_FPS,
    IOU_THRESHOLDS,
    YOLO_WEIGHTS,
    discover_test_videos,
    estimate_human_boundary_margin_frames,
    evaluate_video,
    iou_suffix,
    n_frames_for_job,
    summarize_human_human_agreement,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate Fly-v-Fly temporal heads from held-out NPZ and YOLO feature caches.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--training_dir", required=True,
                   help="Directory containing training run folders.")
    p.add_argument("--heldout_npz_root", required=True,
                   help="Root containing w16s4/ and w8s4/ held-out NPZ manifests.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--aggression_root", default=str(AGGRESSION_ROOT))
    p.add_argument("--yolo_weights", default=str(YOLO_WEIGHTS))
    p.add_argument("--runs", nargs="+", default=None,
                   help="Optional exact run names or case-insensitive prefixes.")
    p.add_argument("--include_nonfocused", action="store_true",
                   help="By default, discover only focused Fly runs: hidden=256, heads LSTM/attention, windows 8/16.")
    p.add_argument("--model_path", default=None,
                   help="Direct path to a single model .pt (skips run discovery).")
    p.add_argument("--model_config", default=None,
                   help="Direct config path for --model_path.")
    p.add_argument("--window_stride", type=int, default=EVAL_PARAMS["window_stride"],
                   help="Held-out evaluation stride used in cache folder names.")
    p.add_argument("--split", default="test")
    p.add_argument("--feature_cache_subdir", default="yolo_feature_cache")
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=3)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--amp_dtype", choices=["auto", "fp16", "bf16"], default="auto")
    p.add_argument("--fps", type=float, default=FLY_FPS)
    p.add_argument("--temporal_smoothing_window", type=int, default=EVAL_PARAMS["temporal_smoothing_window"])
    p.add_argument("--bout_min_duration_frames", type=int, default=EVAL_PARAMS["bout_min_duration_frames"])
    p.add_argument(
        "--gt_modes",
        nargs="+",
        default=["primary", "secondary", "union", "intersection", "padded_primary"],
        choices=["primary", "secondary", "union", "intersection", "padded_primary"],
    )
    p.add_argument("--gt_padding_frames", type=int, default=-1,
                   help="-1 estimates padding from primary-vs-secondary boundary disagreement.")
    p.add_argument("--disable_threshold_decoder", action=argparse.BooleanOptionalAction, default=True,
                   help="Use argmax probabilities for controlled comparisons.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Reuse existing behavior/smoothed CSVs when present.")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--continue_on_failure", action="store_true")
    return p.parse_args()


def _safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text)).strip("_") or "video"


def _write_window_csv(path: Path, rows: list[dict], class_names: list[str]) -> None:
    fields = (
        ["window_index", "frame_start", "frame_end", "time_s",
         "predicted_class", "predicted_class_id", "class_conf"]
        + [f"prob_{c}" for c in class_names]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_focused_fly_cfg(cfg: dict) -> bool:
    return (
        int(cfg.get("hidden_dim", -1)) == 256
        and int(cfg.get("num_frames", cfg.get("window_size", -1))) in {8, 16}
        and str(cfg.get("sequence_model", "")).lower() in {"lstm", "attention"}
    )


def discover_runs(training_dir: Path, *, include_nonfocused: bool) -> list[tuple[str, Path, Path, dict]]:
    found: list[tuple[str, Path, Path, dict]] = []
    if not training_dir.is_dir():
        return found
    for run_dir in sorted(training_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        model_pt = run_dir / "best_model_macro_f1.pt"
        config_json = run_dir / "config.json"
        if not (model_pt.is_file() and config_json.is_file()):
            continue
        cfg = _load_config(config_json)
        if not include_nonfocused and not _is_focused_fly_cfg(cfg):
            continue
        found.append((run_dir.name, model_pt, config_json, cfg))
    return found


def select_runs(
    *,
    training_dir: Path,
    include_nonfocused: bool,
    requested: Sequence[str] | None,
    model_path: str | None,
    model_config: str | None,
) -> list[tuple[str, Path, Path, dict]]:
    if model_path:
        model_pt = Path(model_path)
        config_json = Path(model_config) if model_config else model_pt.parent / "config.json"
        if not model_pt.is_file():
            raise SystemExit(f"model not found: {model_pt}")
        if not config_json.is_file():
            raise SystemExit(f"config not found: {config_json}")
        return [(model_pt.parent.name, model_pt, config_json, _load_config(config_json))]

    all_runs = discover_runs(training_dir, include_nonfocused=include_nonfocused)
    if not all_runs:
        raise SystemExit(f"No trained Fly models found in {training_dir}")
    if not requested:
        return all_runs

    selected: list[tuple[str, Path, Path, dict]] = []
    for req in requested:
        key = str(req).lower()
        matches = [r for r in all_runs if r[0].lower() == key or r[0].lower().startswith(key)]
        if not matches:
            available = [r[0] for r in all_runs]
            raise SystemExit(f"Unknown run {req!r}. Available focused runs: {available}")
        selected.append(matches[0])
    return selected


def _load_manifest_lookup(path: Path) -> dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for split_name, rows in payload.get("splits", {}).items():
        for row in rows:
            item = dict(row)
            item.setdefault("dataset_split", split_name)
            out[str(item["id"])] = item
    return out


def _manifest_for_run(heldout_root: Path, cfg: dict, stride: int) -> tuple[Path, Path, str]:
    num_frames = int(cfg.get("num_frames", cfg.get("window_size", 0)))
    if num_frames <= 0:
        raise ValueError("run config lacks num_frames/window_size")
    cache_key = f"w{num_frames}s{int(stride)}"
    window_root = heldout_root / cache_key
    return window_root / "sequence_manifest.json", window_root / "yolo_feature_cache", cache_key


def evaluate_one_run(
    *,
    run_name: str,
    model_pt: Path,
    config_json: Path,
    cfg_hint: dict,
    args: argparse.Namespace,
    jobs_by_video: dict[str, object],
    boundary_margin: dict,
) -> tuple[list[dict], int, float]:
    run_t0 = time.time()
    device = torch.device(args.device)
    out_root = Path(args.output_dir)
    run_out = out_root / run_name
    run_out.mkdir(parents=True, exist_ok=True)

    manifest_path, feature_cache_dir, cache_key = _manifest_for_run(
        Path(args.heldout_npz_root), cfg_hint, int(args.window_stride)
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(f"held-out manifest not found for {run_name}: {manifest_path}")
    if not feature_cache_dir.is_dir():
        raise FileNotFoundError(f"feature cache not found for {run_name}: {feature_cache_dir}")

    model, cfg, idx_to_class, num_classes, n_animals, num_keypoints, checkpoint = load_model_from_checkpoint(
        model_pt,
        config_json,
        device=str(device),
        yolo_weights_fallback=str(args.yolo_weights),
    )
    class_names = [str(idx_to_class[i]) for i in range(num_classes)]
    if class_names != CLASSES:
        raise ValueError(f"class order mismatch for {run_name}: model={class_names}, evaluator={CLASSES}")

    threshold_decoder = load_threshold_decoder_config(
        model_pt,
        config_json,
        cfg,
        decoder_config_path=None,
        disable_decoder=bool(args.disable_threshold_decoder),
        checkpoint=checkpoint,
    )

    splits, _class_to_idx, _idx_to_class, meta = load_n_manifest(manifest_path)
    selected = list(splits.get(str(args.split), []))
    if not selected:
        raise ValueError(f"No samples found in split {args.split!r} for {manifest_path}")
    raw_lookup = _load_manifest_lookup(manifest_path)
    selected_by_video: dict[str, list] = defaultdict(list)
    for sample in selected:
        selected_by_video[str(sample.source_video)].append(sample)

    if args.dry_run:
        print(f"[DRY] {run_name}: {cache_key} samples={len(selected)} videos={sorted(selected_by_video)}")
        return [], 0, time.time() - run_t0

    visual_alive = bool(getattr(model, "_visual_alive", False))
    dataset = MultiAnimalSequenceDataset(
        samples=selected,
        num_frames=int(meta.get("window_size", cfg.get("num_frames", 0))),
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

    amp_enabled = bool(args.amp and not args.no_amp) and device.type == "cuda"
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
        raise RuntimeError(f"{run_name}: prediction count mismatch: {probs_arr.shape[0]} vs {len(selected)}")
    by_id_prob = {sample.id: probs_arr[i] for i, sample in enumerate(selected)}

    video_results: list[dict] = []
    n_failed = 0
    for video_id, samples in sorted(selected_by_video.items()):
        job = jobs_by_video.get(video_id)
        if job is None:
            print(f"  {run_name}/{video_id}: no test job found")
            n_failed += 1
            continue
        samples = sorted(samples, key=lambda s: (int(s.start_frame), int(s.end_frame), s.id))
        safe_video = _safe_name(video_id)
        video_out = run_out / safe_video
        pred_csv = video_out / f"{safe_video}.behavior.csv"
        smoothed_csv = pred_csv.with_suffix(".smoothed_frames.csv")

        if not (args.skip_existing and pred_csv.is_file() and smoothed_csv.is_file()):
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

        try:
            n_frames = n_frames_for_job(job)
            metrics = evaluate_video(
                job=job,
                pred_csv=pred_csv,
                smoothed_csv=smoothed_csv,
                n_frames=n_frames,
                fps=float(args.fps),
                gt_modes=args.gt_modes,
                gt_padding_frames=int(boundary_margin["margin_frames"]),
            )
            sm = metrics.get("primary", {}).get("smoothed", {})
            print(
                f"  {run_name}/{video_id}: "
                f"F1={sm.get('frame_macro_f1', 0):.3f} "
                f"P={sm.get('frame_macro_precision', 0):.3f} "
                f"R={sm.get('frame_macro_recall', 0):.3f}"
            )
            for gt_mode, mode_metrics in metrics.items():
                row = {
                    "run": run_name,
                    "run_name": run_name,
                    "cache_key": cache_key,
                    "manifest_path": str(manifest_path),
                    "feature_cache_dir": str(feature_cache_dir),
                    "movie_id": getattr(job, "movie_id", ""),
                    "video_id": video_id,
                    "gt_mode": gt_mode,
                    "primary_actions": str(getattr(job, "primary_actions_path", "")),
                    "secondary_actions": ";".join(str(p) for p in getattr(job, "secondary_actions_paths", [])),
                    "secondary_annotation_count": len(getattr(job, "secondary_actions_paths", [])),
                    "gt_padding_frames": int(boundary_margin["margin_frames"]) if gt_mode == "padded_primary" else 0,
                    "n_windows": len(samples),
                }
                for pred_type in ["raw", "smoothed"]:
                    m = mode_metrics.get(pred_type, {})
                    row[f"{pred_type}_accuracy"] = m.get("accuracy", 0)
                    row[f"{pred_type}_frame_macro_precision"] = m.get("frame_macro_precision", 0)
                    row[f"{pred_type}_frame_macro_recall"] = m.get("frame_macro_recall", 0)
                    row[f"{pred_type}_frame_macro_f1"] = m.get("frame_macro_f1", 0)
                    for iou_thr in IOU_THRESHOLDS:
                        key_suffix = iou_suffix(iou_thr)
                        row[f"{pred_type}_bout_macro_precision_{key_suffix}"] = m.get(
                            f"bout_macro_precision_iou{key_suffix}", 0)
                        row[f"{pred_type}_bout_macro_recall_{key_suffix}"] = m.get(
                            f"bout_macro_recall_iou{key_suffix}", 0)
                        row[f"{pred_type}_bout_macro_f1_{key_suffix}"] = m.get(
                            f"bout_macro_f1_iou{key_suffix}", 0)
                    for cls in CLASSES:
                        row[f"{pred_type}_precision_{cls}"] = m.get("per_class_precision", {}).get(cls, 0)
                        row[f"{pred_type}_recall_{cls}"] = m.get("per_class_recall", {}).get(cls, 0)
                        row[f"{pred_type}_f1_{cls}"] = m.get("per_class_f1", {}).get(cls, 0)
                video_results.append(row)
        except Exception as exc:
            print(f"  {run_name}/{video_id}: EVAL ERROR {exc}")
            n_failed += 1
            if not args.continue_on_failure:
                raise

    write_csv(run_out / "per_video_metrics.csv", video_results)
    return video_results, n_failed, time.time() - run_t0


def _mean_present(rows: list[dict], key: str) -> float:
    values = [float(r.get(key, 0)) for r in rows if r.get(key, 0) not in ("", None)]
    values = [v for v in values if v != 0]
    return float(np.mean(values)) if values else 0.0


def summarize_run_rows(run_name: str, rows: list[dict], n_failed: int, elapsed_s: float) -> list[dict]:
    out: list[dict] = []
    for gt_mode in sorted({str(r["gt_mode"]) for r in rows}):
        gt_rows = [r for r in rows if str(r["gt_mode"]) == gt_mode]
        out.append({
            "run": run_name,
            "run_name": run_name,
            "gt_mode": gt_mode,
            "n_video_mode_rows": len(gt_rows),
            "n_failed": int(n_failed),
            "mean_raw_frame_macro_precision": _mean_present(gt_rows, "raw_frame_macro_precision"),
            "mean_raw_frame_macro_recall": _mean_present(gt_rows, "raw_frame_macro_recall"),
            "mean_raw_frame_macro_f1": _mean_present(gt_rows, "raw_frame_macro_f1"),
            "mean_raw_bout_macro_precision_25": _mean_present(gt_rows, "raw_bout_macro_precision_025"),
            "mean_raw_bout_macro_recall_25": _mean_present(gt_rows, "raw_bout_macro_recall_025"),
            "mean_raw_bout_macro_f1_25": _mean_present(gt_rows, "raw_bout_macro_f1_025"),
            "mean_smoothed_frame_macro_precision": _mean_present(gt_rows, "smoothed_frame_macro_precision"),
            "mean_smoothed_frame_macro_recall": _mean_present(gt_rows, "smoothed_frame_macro_recall"),
            "mean_smoothed_frame_macro_f1": _mean_present(gt_rows, "smoothed_frame_macro_f1"),
            "mean_smoothed_bout_macro_precision_10": _mean_present(gt_rows, "smoothed_bout_macro_precision_010"),
            "mean_smoothed_bout_macro_recall_10": _mean_present(gt_rows, "smoothed_bout_macro_recall_010"),
            "mean_smoothed_bout_macro_f1_10": _mean_present(gt_rows, "smoothed_bout_macro_f1_010"),
            "mean_smoothed_bout_macro_precision_25": _mean_present(gt_rows, "smoothed_bout_macro_precision_025"),
            "mean_smoothed_bout_macro_recall_25": _mean_present(gt_rows, "smoothed_bout_macro_recall_025"),
            "mean_smoothed_bout_macro_f1_25": _mean_present(gt_rows, "smoothed_bout_macro_f1_025"),
            "mean_smoothed_bout_macro_precision_50": _mean_present(gt_rows, "smoothed_bout_macro_precision_050"),
            "mean_smoothed_bout_macro_recall_50": _mean_present(gt_rows, "smoothed_bout_macro_recall_050"),
            "mean_smoothed_bout_macro_f1_50": _mean_present(gt_rows, "smoothed_bout_macro_f1_050"),
            "elapsed_s": float(elapsed_s),
        })
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = select_runs(
        training_dir=Path(args.training_dir),
        include_nonfocused=bool(args.include_nonfocused),
        requested=args.runs,
        model_path=args.model_path,
        model_config=args.model_config,
    )
    jobs = discover_test_videos(Path(args.aggression_root))
    if not jobs:
        raise SystemExit(f"No Fly-v-Fly test videos found in {args.aggression_root}")
    jobs_by_video = {f"movie{j.movie_id}": j for j in jobs}
    boundary_margin = estimate_human_boundary_margin_frames(jobs)
    if int(args.gt_padding_frames) >= 0:
        boundary_margin["margin_frames"] = int(args.gt_padding_frames)
        boundary_margin["padding_source"] = "fixed_cli"
    else:
        boundary_margin["padding_source"] = "estimated_human_boundary_std"

    human_rows = summarize_human_human_agreement(jobs)
    human_csv = out_dir / "human_human_agreement.csv"
    write_csv(human_csv, human_rows)

    print("=" * 70)
    print("Fly-v-Fly Cached Held-out Evaluation")
    print("=" * 70)
    print(f"Models: {[r[0] for r in runs]}")
    print(f"Held-out cache root: {Path(args.heldout_npz_root)}")
    print(f"GT modes: {args.gt_modes}")
    print(f"Human-human rows: {len(human_rows)} -> {human_csv}")
    print(f"Threshold decoder disabled: {bool(args.disable_threshold_decoder)}")
    print()

    all_per_video: list[dict] = []
    all_summary: list[dict] = []
    total_t0 = time.time()
    failures: list[dict] = []
    for i, (run_name, model_pt, config_json, cfg) in enumerate(runs):
        print(f"[{i + 1}/{len(runs)}] {run_name}")
        try:
            rows, n_failed, elapsed_s = evaluate_one_run(
                run_name=run_name,
                model_pt=model_pt,
                config_json=config_json,
                cfg_hint=cfg,
                args=args,
                jobs_by_video=jobs_by_video,
                boundary_margin=boundary_margin,
            )
            all_per_video.extend(rows)
            all_summary.extend(summarize_run_rows(run_name, rows, n_failed, elapsed_s))
            print(f"  done elapsed={elapsed_s / 60:.1f}m failures={n_failed}\n")
        except Exception as exc:
            failures.append({"run_name": run_name, "error": repr(exc)})
            print(f"  FAILED: {exc}\n")
            if not args.continue_on_failure:
                break

    write_csv(out_dir / "per_video_metrics.csv", all_per_video)
    write_csv(out_dir / "cross_run_summary.csv", all_summary)
    write_csv(out_dir / "failures.csv", failures)
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "mode": "classifier_only_cached_fly_heldout",
        "training_dir": str(Path(args.training_dir)),
        "heldout_npz_root": str(Path(args.heldout_npz_root)),
        "output_dir": str(out_dir),
        "gt_modes_requested": list(args.gt_modes),
        "human_boundary_margin": boundary_margin,
        "human_human_agreement_csv": str(human_csv),
        "human_human_agreement_rows": len(human_rows),
        "runs": all_summary,
        "failures": failures,
        "total_wall_time_s": time.time() - total_t0,
        "threshold_decoder_disabled": bool(args.disable_threshold_decoder),
    }
    (out_dir / "eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Results: {out_dir / 'cross_run_summary.csv'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


