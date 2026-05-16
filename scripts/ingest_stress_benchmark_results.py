#!/usr/bin/env python3
"""Ingest portable BehaviorScope GUI benchmark results for the notebook.

The benchmark package writes machine-local paths inside its raw CSV/JSON
outputs. This script mirrors the compact benchmark artifacts into
``manuscript_outputs/data/behaviorscope_stress_benchmark`` with those paths
sanitized and with analysis-ready summary tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_ROOTS = [
    REPO_ROOT / "BehaviorScope_StressTest" / "results",
    Path.home() / "Documents" / "GitHub" / "Stress_Test_BehaviorScope-Y" / "results",
]
DEFAULT_OUT_DIR = REPO_ROOT / "manuscript_outputs" / "data" / "behaviorscope_stress_benchmark"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def _discover_runs(source_roots: list[Path]) -> list[Path]:
    by_run_id: dict[str, Path] = {}
    for root in source_roots:
        if not root.is_dir():
            continue
        for candidate in sorted(root.iterdir()):
            if not candidate.is_dir():
                continue
            if not (candidate / "benchmark_summary.json").is_file():
                continue
            if not (candidate / "benchmark_runs.csv").is_file():
                continue
            summary = _read_json(candidate / "benchmark_summary.json")
            run_id = str(summary.get("run_id") or candidate.name)
            by_run_id.setdefault(run_id, candidate)
    return [by_run_id[k] for k in sorted(by_run_id)]


def _sanitize_profile(summary: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(json.dumps(summary))
    profile = out.get("profile")
    if isinstance(profile, dict):
        profile["entry_point"] = "behaviorscope/infer.py"
        profile["model"] = "models/behavior_classifier.pt"
        profile["pose_detector"] = "models/pose_detector.pt"
    return out


def _sanitize_selected_videos(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = [c for c in ["video", "size_bytes"] if c in df.columns]
    out = df[keep].copy()
    if "size_bytes" in out.columns:
        out["size_mb"] = pd.to_numeric(out["size_bytes"], errors="coerce") / (1024 * 1024)
        out["size_mb"] = out["size_mb"].round(2)
    return out


def _sanitize_benchmark_runs(path: Path, run_id: str, hardware_label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ["run_id", "hardware_label"] if c in df.columns])
    df.insert(0, "hardware_label", hardware_label)
    df.insert(0, "run_id", run_id)
    drop_cols = [c for c in ["output_csv", "metrics_csv", "log_path"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    numeric_cols = [
        "repeat",
        "returncode",
        "subprocess_wall_s",
        "metrics_rows",
        "frames_done",
        "windows_done",
        "metrics_elapsed_s",
        "metrics_fps",
        "interval_fps_mean",
        "interval_fps_median",
        "interval_fps_max",
        "steady_interval_fps_mean",
        "steady_interval_fps_median",
        "steady_interval_fps_max",
        "gpu_util_avg_mean",
        "gpu_util_peak_max",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _collect_interval_metrics(run_dir: Path, run_id: str, hardware_label: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    raw_metric_paths = sorted(run_dir.rglob("*.behavior_metrics.csv"))
    if not raw_metric_paths and (run_dir / "interval_metrics.csv").is_file():
        out = pd.read_csv(run_dir / "interval_metrics.csv")
        out = out.drop(columns=[c for c in ["run_id", "hardware_label"] if c in out.columns])
        out.insert(0, "hardware_label", hardware_label)
        out.insert(0, "run_id", run_id)
        return out
    for metrics_path in raw_metric_paths:
        repeat = metrics_path.parent.name.replace("repeat_", "")
        video = metrics_path.name.replace(".behavior_metrics.csv", ".mp4")
        df = pd.read_csv(metrics_path)
        if "source" in df.columns:
            df = df.drop(columns=["source"])
        df.insert(0, "video", video)
        df.insert(0, "repeat", int(repeat) if repeat.isdigit() else repeat)
        df.insert(0, "hardware_label", hardware_label)
        df.insert(0, "run_id", run_id)
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    text_cols = {"run_id", "hardware_label", "video", "wallclock_iso"}
    for col in out.columns.difference(list(text_cols)):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _hardware_label(summary: dict[str, Any]) -> str:
    hw = summary.get("hardware", {})
    platform = hw.get("platform", {})
    cpu = hw.get("cpu", {})
    gpu = summary.get("hardware", {}).get("gpu", {})
    if gpu.get("cuda_available") and gpu.get("name"):
        name = str(gpu.get("name")).strip()
        return name.replace("NVIDIA GeForce ", "").replace(" ", "_")
    system = str(platform.get("system") or "CPU").strip()
    machine = str(platform.get("machine") or "").strip()
    logical = cpu.get("logical_cores")
    physical = cpu.get("physical_cores")
    if system == "Darwin":
        return f"Apple_CPU_{physical or logical}c"
    if system == "Windows":
        return f"Windows_CPU_{physical or logical}c_{logical}t"
    return f"{system}_{machine}_CPU_{logical}t".strip("_").replace(" ", "_")


def _cpu_model_name(summary: dict[str, Any], hardware_label: str) -> str | None:
    """Return user-confirmed CPU marketing names when benchmark metadata is generic."""
    hw = summary.get("hardware", {})
    platform = hw.get("platform", {})
    processor = str(platform.get("processor") or "")
    if hardware_label == "Windows_CPU_8c_16t" and "Model 141" in processor:
        return "Intel Core i7-11800H @ 2.30 GHz"
    if hardware_label == "Apple_CPU_14c":
        return "Apple M4 Max"
    return None


def _summary_row(summary: dict[str, Any], run_id: str, hardware_label: str) -> dict[str, Any]:
    hw = summary.get("hardware", {})
    platform = hw.get("platform", {})
    gpu = hw.get("gpu", {})
    cpu = hw.get("cpu", {})
    mem = hw.get("memory", {})
    packages = hw.get("packages", {})
    profile = summary.get("profile", {})
    s = summary.get("summary", {})
    return {
        "run_id": run_id,
        "hardware_label": hardware_label,
        "created_at": summary.get("created_at"),
        "platform_system": platform.get("system"),
        "platform_release": platform.get("release"),
        "platform_machine": platform.get("machine"),
        "platform_processor": platform.get("processor"),
        "cpu_model_name": _cpu_model_name(summary, hardware_label),
        "gpu_name": gpu.get("name"),
        "cuda_available": gpu.get("cuda_available"),
        "cuda_version": gpu.get("cuda_version"),
        "driver_query": gpu.get("nvidia_smi_query"),
        "vram_gb": gpu.get("total_vram_gb"),
        "compute_capability": gpu.get("compute_capability"),
        "logical_cores": cpu.get("logical_cores"),
        "physical_cores": cpu.get("physical_cores"),
        "ram_gb": mem.get("total_gb"),
        "torch": packages.get("torch"),
        "opencv": packages.get("opencv"),
        "ultralytics": packages.get("ultralytics"),
        "device": profile.get("device"),
        "amp": profile.get("amp"),
        "yolo_batch": profile.get("yolo_batch"),
        "window_stride": profile.get("window_stride"),
        "n_runs": s.get("n_runs"),
        "n_videos": s.get("n_videos"),
        "repeats_per_video": s.get("repeats_per_video"),
        "total_frames": s.get("total_frames"),
        "aggregate_inference_fps": s.get("aggregate_inference_fps"),
        "aggregate_wall_fps": s.get("aggregate_wall_fps"),
        "run_fps_mean": s.get("run_fps_mean"),
        "run_fps_median": s.get("run_fps_median"),
        "run_fps_min": s.get("run_fps_min"),
        "run_fps_max": s.get("run_fps_max"),
        "steady_interval_fps_mean": s.get("steady_interval_fps_mean"),
        "steady_interval_fps_median": s.get("steady_interval_fps_median"),
        "steady_interval_fps_min": s.get("steady_interval_fps_min"),
        "steady_interval_fps_max_mean": s.get("steady_interval_fps_max_mean"),
        "steady_interval_fps_max_observed": s.get("steady_interval_fps_max_observed"),
    }


def _per_video_summary(runs: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "metrics_fps",
        "steady_interval_fps_mean",
        "steady_interval_fps_max",
        "gpu_util_avg_mean",
        "gpu_util_peak_max",
    ]
    grouped = (
        runs.groupby(["run_id", "hardware_label", "video"], dropna=False)
        .agg(
            n_repeats=("repeat", "count"),
            frames_done=("frames_done", "median"),
            metrics_fps_mean=("metrics_fps", "mean"),
            metrics_fps_sd=("metrics_fps", "std"),
            metrics_fps_min=("metrics_fps", "min"),
            metrics_fps_max=("metrics_fps", "max"),
            steady_fps_mean=("steady_interval_fps_mean", "mean"),
            steady_fps_sd=("steady_interval_fps_mean", "std"),
            steady_fps_max_observed=("steady_interval_fps_max", "max"),
            gpu_util_avg_mean=("gpu_util_avg_mean", "mean"),
            gpu_util_peak_max=("gpu_util_peak_max", "max"),
        )
        .reset_index()
    )
    for col in grouped.select_dtypes(include="number").columns:
        if col not in {"n_repeats", "frames_done"}:
            grouped[col] = grouped[col].round(3)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(description="Mirror BehaviorScope GUI benchmark results into notebook data.")
    parser.add_argument("--source_root", action="append", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    source_roots = args.source_root if args.source_root else DEFAULT_SOURCE_ROOTS
    source_roots = [*source_roots, args.out_dir / "runs"]
    run_dirs = _discover_runs(source_roots)
    if not run_dirs:
        raise SystemExit("No benchmark result folders found.")

    out_dir = args.out_dir
    runs_out = out_dir / "runs"
    analysis_out = out_dir / "analysis"
    runs_out.mkdir(parents=True, exist_ok=True)
    analysis_out.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    all_runs: list[pd.DataFrame] = []
    all_intervals: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        summary = _read_json(run_dir / "benchmark_summary.json")
        run_id = str(summary.get("run_id") or run_dir.name)
        hardware_label = _hardware_label(summary)
        run_out = runs_out / run_id
        run_out.mkdir(parents=True, exist_ok=True)

        sanitized_summary = _sanitize_profile(summary)
        _write_json(run_out / "benchmark_summary.json", sanitized_summary)
        hardware_src = run_dir / "hardware_specs.json"
        hardware_dst = run_out / "hardware_specs.json"
        if hardware_src.resolve() != hardware_dst.resolve():
            shutil.copy2(hardware_src, hardware_dst)

        selected = _sanitize_selected_videos(run_dir / "selected_videos.csv")
        selected.to_csv(run_out / "selected_videos.csv", index=False)

        run_rows = _sanitize_benchmark_runs(run_dir / "benchmark_runs.csv", run_id, hardware_label)
        run_rows.to_csv(run_out / "benchmark_runs.csv", index=False)
        all_runs.append(run_rows)

        intervals = _collect_interval_metrics(run_dir, run_id, hardware_label)
        if not intervals.empty:
            intervals.to_csv(run_out / "interval_metrics.csv", index=False)
            all_intervals.append(intervals)

        summary_rows.append(_summary_row(summary, run_id, hardware_label))

    summaries = pd.DataFrame(summary_rows)
    summaries.to_csv(analysis_out / "run_summaries.csv", index=False)
    if all_runs:
        all_run_rows = pd.concat(all_runs, ignore_index=True)
        all_run_rows.to_csv(analysis_out / "benchmark_runs_all.csv", index=False)
        _per_video_summary(all_run_rows).to_csv(analysis_out / "per_video_summary.csv", index=False)
    if all_intervals:
        all_interval_rows = pd.concat(all_intervals, ignore_index=True)
        all_interval_rows.to_csv(analysis_out / "interval_metrics_all.csv", index=False)

    readme = """# BehaviorScope stress-test benchmark data

Notebook-facing mirror of portable BehaviorScope stress-test results.

This folder contains sanitized, compact benchmark artifacts only. Per-frame
behavior prediction CSVs and logs are intentionally not mirrored here.

## Layout

- `analysis/run_summaries.csv`: one row per hardware benchmark run.
- `analysis/benchmark_runs_all.csv`: one row per video repeat.
- `analysis/per_video_summary.csv`: per-video repeated-run summary.
- `analysis/interval_metrics_all.csv`: interval-level FPS/GPU telemetry.
- `runs/<run_id>/`: sanitized compact artifacts for each benchmark run.

## Benchmark unit

The primary unit is a video-repeat inference run. Each hardware benchmark uses
five bundled MP4 videos repeated three times. FPS summaries report both
end-to-end inference FPS and warm steady-state interval FPS.

## Adding runs

Run `python scripts/ingest_stress_benchmark_results.py` to rebuild the mirror
from the default local stress-test roots. To ingest a result folder on another
drive, pass the results root with `--source_root`. Already mirrored compact runs
are also treated as fallback inputs, so previously ingested hardware profiles
are preserved even when the original external drive is not mounted.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Ingested {len(run_dirs)} benchmark run(s) into {out_dir}")
    for row in summary_rows:
        print(
            f"- {row['run_id']}: {row['gpu_name']} "
            f"aggregate={row['aggregate_inference_fps']} fps "
            f"steady_mean={row['steady_interval_fps_mean']} fps "
            f"steady_max={row['steady_interval_fps_max_observed']} fps"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
