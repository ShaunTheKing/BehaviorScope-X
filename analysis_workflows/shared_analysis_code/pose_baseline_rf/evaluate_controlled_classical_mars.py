#!/usr/bin/env python3
"""Evaluate controlled RF/XGBoost models on held-out MARS windows."""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from controlled_common import (
    add_shared_scripts_to_path,
    dump_json,
    load_config,
    load_pickle,
    output_root,
    resolve_repo_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate controlled classical models with the same bout code as BehaviorScope-Y.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    p.add_argument("--model_paths", nargs="+", required=True)
    p.add_argument("--feature_set", required=True)
    p.add_argument("--splits", nargs="+", default=None)
    p.add_argument("--allow_pred_named_annots", action=argparse.BooleanOptionalAction, default=None)
    return p.parse_args()


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


def _predict_proba_all_classes(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    probs = model.predict_proba(X)
    if isinstance(probs, list):
        probs = np.asarray(probs[0])
    probs = np.asarray(probs, dtype=np.float64)
    if probs.shape[1] == n_classes:
        return probs
    out = np.zeros((probs.shape[0], n_classes), dtype=np.float64)
    classes = getattr(model, "classes_", np.arange(probs.shape[1]))
    for src_i, cls in enumerate(classes):
        out[:, int(cls)] = probs[:, src_i]
    row_sum = out.sum(axis=1, keepdims=True)
    np.divide(out, np.maximum(row_sum, 1e-12), out=out)
    return out


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    shared = add_shared_scripts_to_path(cfg)
    from batch_infer_eval import CLASSES, discover_jobs, evaluate_prediction_set, write_csv  # noqa: WPS433
    from infer_x import _write_smoothed_frames_csv  # noqa: WPS433

    t0 = time.time()
    out = output_root(cfg)
    dataset = str(cfg["features"].get("dataset", "mars"))
    matrix_root = out / "matrices" / dataset
    X = np.load(matrix_root / args.feature_set / "heldout.npz")["X"].astype(np.float32)
    meta_rows = json.loads((matrix_root / args.feature_set / "heldout_metadata.json").read_text(encoding="utf-8"))
    if X.shape[0] != len(meta_rows):
        raise RuntimeError(f"matrix/meta row mismatch: X={X.shape[0]} meta={len(meta_rows)}")

    class_names = list(cfg["evaluation"]["classes"])
    if class_names != CLASSES:
        raise SystemExit(f"class order mismatch: config={class_names}, evaluator={CLASSES}")
    idx_to_class = {i: c for i, c in enumerate(class_names)}
    fps = float(cfg["evaluation"]["fps"])
    splits = args.splits or list(cfg["evaluation"]["splits"])
    allow_pred = bool(cfg["evaluation"].get("allow_pred_named_annots", False))
    if args.allow_pred_named_annots is not None:
        allow_pred = bool(args.allow_pred_named_annots)

    mars_root = resolve_repo_path(cfg["paths"]["mars_data_root"])
    jobs_by_key = {
        (j.split, j.video_id): j
        for j in discover_jobs(mars_root, splits, allow_pred_named_annots=allow_pred)
    }

    by_video: dict[tuple[str, str], list[tuple[int, dict]]] = defaultdict(list)
    split_set = set(splits)
    for idx, row in enumerate(meta_rows):
        split = str(row.get("mars_split", ""))
        if split in split_set:
            by_video[(split, str(row["source_video"]))].append((idx, row))

    all_failures = []
    for model_path_str in args.model_paths:
        model_path = Path(model_path_str).resolve()
        model = load_pickle(model_path)
        probs = _predict_proba_all_classes(model, np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), len(class_names))
        model_tag = model_path.stem
        out_root = out / "eval" / dataset / model_tag / args.feature_set
        manifest_rows = []
        failure_rows = []
        exclusion_rows = []
        summary_rows = []
        per_class_rows_all = []
        gt_bout_rows_all = []
        pred_bout_rows_all = []
        bout_match_rows_all = []

        for (split, video_id), indexed_rows in sorted(by_video.items()):
            job = jobs_by_key.get((split, video_id))
            if job is None:
                exclusion_rows.append({
                    "split": split,
                    "video_id": video_id,
                    "reason": "no accepted human-GT annot under allow_pred_named_annots=false",
                })
                continue

            indexed_rows = sorted(indexed_rows, key=lambda item: (int(item[1]["start_frame"]), int(item[1]["end_frame"]), item[1]["sample_id"]))
            safe_video = _safe_name(video_id)
            video_out = out_root / split / safe_video
            pred_csv = video_out / f"{safe_video}.behavior.csv"
            rows = []
            window_records = []
            for wi, (row_idx, row_meta) in enumerate(indexed_rows):
                row_probs = probs[row_idx].astype(np.float64)
                pred_id = int(np.argmax(row_probs))
                row = {
                    "window_index": wi,
                    "frame_start": int(row_meta["start_frame"]),
                    "frame_end": int(row_meta["end_frame"]),
                    "time_s": float(row_meta["start_frame"]) / max(fps, 1e-6),
                    "predicted_class": class_names[pred_id],
                    "predicted_class_id": pred_id,
                    "class_conf": float(row_probs[pred_id]),
                }
                for ci, cname in enumerate(class_names):
                    row[f"prob_{cname}"] = float(row_probs[ci])
                rows.append(row)
                window_records.append((int(row_meta["start_frame"]), int(row_meta["end_frame"]), row_probs))
            _write_window_csv(pred_csv, rows, class_names)
            smoothed_csv = _write_smoothed_frames_csv(
                window_records,
                pred_csv,
                class_names,
                idx_to_class,
                fps,
                int(cfg["evaluation"]["temporal_smoothing_window"]),
                int(cfg["evaluation"]["bout_min_duration_frames"]),
                threshold_decoder=None,
                v25_splitter_config=None,
            )

            manifest_rows.append({
                "split": split,
                "video_id": video_id,
                "annot_path": str(job.annot_path),
                "predictions_csv": str(pred_csv),
                "smoothed_csv": str(smoothed_csv),
                "n_windows": len(indexed_rows),
                "allow_pred_named_annots": allow_pred,
            })
            try:
                n_frames = max(int(row["end_frame"]) for _, row in indexed_rows) + 1
                for prediction_type, csv_path in [("raw_windows", pred_csv), ("smoothed_frames", smoothed_csv)]:
                    if csv_path is None or not Path(csv_path).is_file():
                        continue
                    summary, per_class, gt_bouts, pred_bouts, matches = evaluate_prediction_set(
                        split=split,
                        video_id=video_id,
                        prediction_type=prediction_type,
                        predictions_csv=Path(csv_path),
                        annot_path=job.annot_path,
                        n_frames=n_frames,
                        fps=fps,
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
        write_csv(out_root / "exclusions.csv", exclusion_rows)
        write_csv(out_root / "failures.csv", failure_rows)
        write_csv(out_root / "per_video_summary.csv", summary_rows)
        write_csv(out_root / "per_class_metrics.csv", per_class_rows_all)
        write_csv(out_root / "ground_truth_bouts.csv", gt_bout_rows_all)
        write_csv(out_root / "predicted_bouts.csv", pred_bout_rows_all)
        write_csv(out_root / "bout_matches.csv", bout_match_rows_all)
        aggregate = {
            "mode": "controlled_classical_cached_heldout",
            "model_path": str(model_path),
            "feature_set": args.feature_set,
            "allow_pred_named_annots": allow_pred,
            "n_windows": int(X.shape[0]),
            "n_videos_considered": int(len(by_video)),
            "n_excluded_videos": int(len(exclusion_rows)),
            "n_failures": int(len(failure_rows)),
            "shared_scripts": str(shared),
            "elapsed_s": round(time.time() - t0, 3),
        }
        dump_json(out_root / "aggregate_summary.json", aggregate)
        print(f"[eval] {model_tag}/{args.feature_set}: failures={len(failure_rows)} out={out_root}", flush=True)
        all_failures.extend(failure_rows)

    return 1 if all_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


