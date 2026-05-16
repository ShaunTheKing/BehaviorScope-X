#!/usr/bin/env python3
"""Benchmark GUI-equivalent BehaviorScope-Y full-model inference throughput.

Runs the same ``BehaviorScope-Y/infer_y.py`` entry point used by the GUI on
five held-out videos, repeated three times by default. The benchmark is designed
for manuscript framing around compute-conscious visual + pose inference:

  - no review MP4 export
  - no per-frame pose CSV export
  - runtime metrics enabled
  - inference AMP enabled by default
  - larger YOLO frame batching
  - buffered CSV flushing

Outputs:
  data/manuscript_v5/inference_benchmark/<run_id>/
    benchmark_runs.csv
    benchmark_summary.json
    selected_videos.csv
    repeat_*/<video>.behavior.csv
    repeat_*/<video>.behavior_metrics.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from batch_infer_eval_behaviorscope_y import convert_seq_to_mp4  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parent.parent
INFER_SCRIPT = REPO_ROOT / "infer_y.py"
DEFAULT_RUN_DIR = REPO_ROOT / "data" / "manuscript_v5" / "training_runs" / "R5_yolo_attn_v5_full"
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "data"
    / "manuscript_v5"
    / "test_eval"
    / "R5_yolo_attn_v5_full_mars_test_eval"
    / "manifest.csv"
)
DEFAULT_YOLO_WEIGHTS = REPO_ROOT / "mars_yolo_pose" / "runs" / "pose" / "train" / "weights" / "best.pt"
DEFAULT_OUT_ROOT = REPO_ROOT / "data" / "manuscript_v5" / "inference_benchmark"
NOTEBOOK_OUT_ROOT = REPO_ROOT / "manuscript_outputs" / "data" / "manuscript_v5" / "inference_benchmark"
QC_EXCLUDED_VIDEO_IDS = {"Mouse060_20160526_18-16-27"}


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _load_selected_videos(manifest_csv: Path, n_videos: int, selection: str) -> list[dict]:
    rows = _read_csv(manifest_csv)
    candidates: list[dict] = []
    for row in rows:
        video_id = row.get("video_id", "")
        if video_id in QC_EXCLUDED_VIDEO_IDS:
            continue
        source = Path(row.get("source_for_infer", ""))
        source_kind = "mp4"
        if not source.is_file():
            seq_path = Path(row.get("seq_path", ""))
            if seq_path.is_file():
                source = seq_path
                source_kind = "seq"
            else:
                continue
        else:
            source_kind = source.suffix.lower().lstrip(".") or "file"
        if not source.is_file():
            continue
        try:
            frames = int(float(row.get("conversion_frames", 0) or 0))
        except Exception:
            frames = 0
        candidates.append(
            {
                "split": row.get("split", ""),
                "video_id": video_id,
                "source_for_infer": str(source),
                "source_kind": source_kind,
                "seq_path": row.get("seq_path", ""),
                "seek_mat_path": row.get("seek_mat_path", ""),
                "conversion_frames": frames,
                "conversion_width": row.get("conversion_width", ""),
                "conversion_height": row.get("conversion_height", ""),
            }
        )
    if len(candidates) < n_videos:
        raise SystemExit(
            f"Only found {len(candidates)} usable source videos in {manifest_csv}; "
            f"need {n_videos}."
        )
    if selection == "shortest":
        candidates.sort(key=lambda r: (int(r["conversion_frames"]), r["split"], r["video_id"]))
    elif selection == "longest":
        candidates.sort(key=lambda r: (-int(r["conversion_frames"]), r["split"], r["video_id"]))
    else:
        candidates.sort(key=lambda r: (r["split"], r["video_id"]))
    return candidates[:n_videos]


def _metrics_summary(metrics_csv: Path) -> dict:
    rows = _read_csv(metrics_csv)
    if not rows:
        return {}
    last = rows[-1]
    frames = float(last.get("frames_done", 0) or 0)
    windows = float(last.get("windows_done", 0) or 0)
    elapsed = float(last.get("elapsed_s", 0) or 0)
    interval_fps = [
        float(r.get("instant_fps", 0) or 0)
        for r in rows
        if float(r.get("instant_fps", 0) or 0) > 0
    ]
    steady_rows = rows[1:-1] if len(rows) > 2 else rows[1:]
    steady_interval_fps = [
        float(r.get("instant_fps", 0) or 0)
        for r in steady_rows
        if float(r.get("instant_fps", 0) or 0) > 0
    ]
    gpu_avg = [
        float(r.get("gpu_util_percent_avg", 0) or 0)
        for r in rows
        if str(r.get("gpu_util_percent_avg", "")).strip() not in ("", "nan")
    ]
    gpu_peak = [
        float(r.get("gpu_util_percent_max", 0) or 0)
        for r in rows
        if str(r.get("gpu_util_percent_max", "")).strip() not in ("", "nan")
    ]
    return {
        "metrics_rows": len(rows),
        "frames_done": int(frames),
        "windows_done": int(windows),
        "metrics_elapsed_s": round(elapsed, 3),
        "metrics_fps": round(frames / elapsed, 3) if elapsed > 0 else None,
        "interval_fps_mean": round(statistics.mean(interval_fps), 3) if interval_fps else None,
        "interval_fps_median": round(statistics.median(interval_fps), 3) if interval_fps else None,
        "interval_fps_max": round(max(interval_fps), 3) if interval_fps else None,
        "steady_interval_fps_mean": round(statistics.mean(steady_interval_fps), 3)
        if steady_interval_fps
        else None,
        "steady_interval_fps_median": round(statistics.median(steady_interval_fps), 3)
        if steady_interval_fps
        else None,
        "steady_interval_fps_max": round(max(steady_interval_fps), 3)
        if steady_interval_fps
        else None,
        "gpu_util_avg_mean": round(statistics.mean(gpu_avg), 3) if gpu_avg else None,
        "gpu_util_peak_max": round(max(gpu_peak), 3) if gpu_peak else None,
    }


def _summarize_runs(rows: list[dict]) -> dict:
    fps_vals = [float(r["metrics_fps"]) for r in rows if r.get("metrics_fps") not in (None, "")]
    steady_mean_vals = [
        float(r["steady_interval_fps_mean"])
        for r in rows
        if r.get("steady_interval_fps_mean") not in (None, "")
    ]
    steady_max_vals = [
        float(r["steady_interval_fps_max"])
        for r in rows
        if r.get("steady_interval_fps_max") not in (None, "")
    ]
    frames_total = sum(int(r.get("frames_done", 0) or 0) for r in rows)
    infer_seconds_total = sum(float(r.get("metrics_elapsed_s", 0) or 0) for r in rows)
    wall_seconds_total = sum(float(r.get("subprocess_wall_s", 0) or 0) for r in rows)
    out = {
        "n_runs": len(rows),
        "n_videos": len({r["video_id"] for r in rows}),
        "repeats_per_video": len({r["repeat"] for r in rows}),
        "total_frames": frames_total,
        "total_inference_seconds": round(infer_seconds_total, 3),
        "aggregate_inference_fps": round(frames_total / infer_seconds_total, 3)
        if infer_seconds_total > 0
        else None,
        "total_subprocess_wall_seconds": round(wall_seconds_total, 3),
        "aggregate_wall_fps": round(frames_total / wall_seconds_total, 3)
        if wall_seconds_total > 0
        else None,
    }
    if fps_vals:
        out.update(
            {
                "run_fps_mean": round(statistics.mean(fps_vals), 3),
                "run_fps_median": round(statistics.median(fps_vals), 3),
                "run_fps_min": round(min(fps_vals), 3),
                "run_fps_max": round(max(fps_vals), 3),
            }
        )
    if steady_mean_vals:
        out.update(
            {
                "steady_interval_fps_mean": round(statistics.mean(steady_mean_vals), 3),
                "steady_interval_fps_median": round(statistics.median(steady_mean_vals), 3),
                "steady_interval_fps_min": round(min(steady_mean_vals), 3),
                "steady_interval_fps_max_mean": round(max(steady_mean_vals), 3),
            }
        )
    if steady_max_vals:
        out["steady_interval_fps_max_observed"] = round(max(steady_max_vals), 3)
    by_video = {}
    for vid in sorted({r["video_id"] for r in rows}):
        vals = [float(r["metrics_fps"]) for r in rows if r["video_id"] == vid and r.get("metrics_fps")]
        by_video[vid] = {
            "n": len(vals),
            "fps_mean": round(statistics.mean(vals), 3) if vals else None,
            "fps_max": round(max(vals), 3) if vals else None,
        }
    out["per_video"] = by_video
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark R5 full-model inference FPS on held-out videos.")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--manifest_csv", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--model_path", type=Path, default=DEFAULT_RUN_DIR / "best_model_macro_f1.pt")
    p.add_argument("--model_config", type=Path, default=DEFAULT_RUN_DIR / "config.json")
    p.add_argument("--yolo_weights", type=Path, default=DEFAULT_YOLO_WEIGHTS)
    p.add_argument("--out_root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--run_id", default=None)
    p.add_argument("--n_videos", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--selection", choices=["shortest", "longest", "alphabetical"], default="shortest")
    p.add_argument("--device", default="cuda")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--window_stride", type=int, default=16)
    p.add_argument("--yolo_batch", type=int, default=64)
    p.add_argument("--pose_conf_threshold", type=float, default=0.2)
    p.add_argument("--metrics_log_interval", type=int, default=100)
    p.add_argument("--csv_flush_interval", type=int, default=50)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--temporal_smoothing_window", type=int, default=0)
    p.add_argument("--bout_min_duration_frames", type=int, default=0)
    p.add_argument("--convert_missing_mp4", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--overwrite_converted_mp4", action="store_true", default=False)
    p.add_argument("--mirror_to_notebook", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    for required in (INFER_SCRIPT, args.manifest_csv, args.model_path, args.model_config, args.yolo_weights):
        if not Path(required).is_file():
            raise SystemExit(f"Required file not found: {required}")

    run_id = args.run_id or datetime.now().strftime("R5_full_model_max_fps_%Y%m%d_%H%M%S")
    out_dir = args.out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = _load_selected_videos(args.manifest_csv, args.n_videos, args.selection)

    converted_dir = out_dir / "source_mp4"
    for video in selected:
        if video.get("source_kind") != "seq":
            video["benchmark_source"] = video["source_for_infer"]
            video["benchmark_source_kind"] = video["source_kind"]
            video["seq_conversion_elapsed_s"] = ""
            continue
        if not args.convert_missing_mp4:
            raise SystemExit(
                f"{video['video_id']} selected a .seq source. Re-run with "
                "--convert_missing_mp4 or choose videos with existing MP4 sources."
            )
        seq_path = Path(video["source_for_infer"])
        seek_path = Path(str(video.get("seek_mat_path") or ""))
        if not seek_path.is_file():
            candidate = seq_path.with_name(seq_path.stem + "-seek.mat")
            seek_path = candidate if candidate.is_file() else Path()
        mp4_path = converted_dir / f"{video['video_id']}.mp4"
        print(f"[benchmark] converting {seq_path.name} -> {mp4_path}", flush=True)
        info = convert_seq_to_mp4(
            seq_path,
            mp4_path,
            seek_mat_path=(seek_path if seek_path.is_file() else None),
            fps=float(args.fps),
            overwrite=bool(args.overwrite_converted_mp4),
        )
        video["benchmark_source"] = str(mp4_path)
        video["benchmark_source_kind"] = "converted_mp4"
        video["seq_conversion_elapsed_s"] = info.get("elapsed_s", "")
        video["conversion_frames"] = int(info.get("frames") or video.get("conversion_frames") or 0)
        video["conversion_width"] = info.get("width", video.get("conversion_width", ""))
        video["conversion_height"] = info.get("height", video.get("conversion_height", ""))

    _write_csv(
        out_dir / "selected_videos.csv",
        selected,
        [
            "split",
            "video_id",
            "source_for_infer",
            "source_kind",
            "benchmark_source",
            "benchmark_source_kind",
            "seq_conversion_elapsed_s",
            "seq_path",
            "seek_mat_path",
            "conversion_frames",
            "conversion_width",
            "conversion_height",
        ],
    )

    run_rows: list[dict] = []
    for repeat in range(1, int(args.repeats) + 1):
        repeat_dir = out_dir / f"repeat_{repeat:02d}"
        repeat_dir.mkdir(parents=True, exist_ok=True)
        for video in selected:
            video_id = video["video_id"]
            output_csv = repeat_dir / f"{video_id}.behavior.csv"
            metrics_csv = repeat_dir / f"{video_id}.behavior_metrics.csv"
            log_path = repeat_dir / f"{video_id}.log"
            cmd = [
                str(args.python),
                str(INFER_SCRIPT),
                "--model_path",
                str(args.model_path),
                "--model_config",
                str(args.model_config),
                "--yolo_weights",
                str(args.yolo_weights),
                "--source",
                video["benchmark_source"],
                "--output",
                str(output_csv),
                "--device",
                str(args.device),
                "--fps",
                str(float(args.fps)),
                "--window_stride",
                str(int(args.window_stride)),
                "--yolo_batch",
                str(int(args.yolo_batch)),
                "--pose_conf_threshold",
                str(float(args.pose_conf_threshold)),
                "--temporal_smoothing_window",
                str(int(args.temporal_smoothing_window)),
                "--bout_min_duration_frames",
                str(int(args.bout_min_duration_frames)),
                "--log_metrics",
                "--metrics_csv",
                str(metrics_csv),
                "--metrics_log_interval",
                str(int(args.metrics_log_interval)),
                "--csv_flush_interval",
                str(int(args.csv_flush_interval)),
            ]
            if args.amp:
                cmd.append("--amp")

            print(f"[benchmark] repeat={repeat} video={video_id}", flush=True)
            t0 = time.time()
            with log_path.open("w", encoding="utf-8") as log_f:
                log_f.write("# " + " ".join(cmd) + "\n\n")
                log_f.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=str(REPO_ROOT),
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    encoding="utf-8",
                    errors="replace",
                )
            wall = time.time() - t0
            metrics = _metrics_summary(metrics_csv) if metrics_csv.is_file() else {}
            row = {
                "repeat": repeat,
                "split": video["split"],
                "video_id": video_id,
                "source_for_infer": video["source_for_infer"],
                "benchmark_source": video["benchmark_source"],
                "benchmark_source_kind": video["benchmark_source_kind"],
                "conversion_frames": video["conversion_frames"],
                "returncode": proc.returncode,
                "subprocess_wall_s": round(wall, 3),
                "output_csv": str(output_csv),
                "metrics_csv": str(metrics_csv),
                "log_path": str(log_path),
                **metrics,
            }
            run_rows.append(row)
            _write_csv(out_dir / "benchmark_runs.csv", run_rows)
            if proc.returncode != 0:
                print(f"[benchmark] FAILED repeat={repeat} video={video_id}; see {log_path}", flush=True)
                return int(proc.returncode)

    summary = {
        "run_id": run_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_profile": {
            "entry_point": str(INFER_SCRIPT),
            "model_path": str(args.model_path),
            "model_config": str(args.model_config),
            "yolo_weights": str(args.yolo_weights),
            "selection": args.selection,
            "amp": bool(args.amp),
            "device": args.device,
            "yolo_batch": int(args.yolo_batch),
            "window_stride": int(args.window_stride),
            "pose_conf_threshold": float(args.pose_conf_threshold),
            "temporal_smoothing_window": int(args.temporal_smoothing_window),
            "bout_min_duration_frames": int(args.bout_min_duration_frames),
            "metrics_log_interval": int(args.metrics_log_interval),
            "csv_flush_interval": int(args.csv_flush_interval),
            "review_mp4_export": False,
            "pose_csv_export": False,
            "convert_missing_mp4": bool(args.convert_missing_mp4),
        },
        "summary": _summarize_runs(run_rows),
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.mirror_to_notebook:
        nb_dir = NOTEBOOK_OUT_ROOT / run_id
        nb_dir.mkdir(parents=True, exist_ok=True)
        for name in ("benchmark_runs.csv", "benchmark_summary.json", "selected_videos.csv"):
            (nb_dir / name).write_text((out_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
        latest = NOTEBOOK_OUT_ROOT / "latest_summary.json"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    s = summary["summary"]
    print(
        "[benchmark] done "
        f"aggregate_inference_fps={s.get('aggregate_inference_fps')} "
        f"steady_interval_fps_mean={s.get('steady_interval_fps_mean')} "
        f"steady_interval_fps_max_observed={s.get('steady_interval_fps_max_observed')} "
        f"run_fps_mean={s.get('run_fps_mean')} "
        f"run_fps_max={s.get('run_fps_max')} "
        f"out={out_dir}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
