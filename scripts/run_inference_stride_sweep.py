#!/usr/bin/env python3
"""
Inference-stride sweep on the v5 R5 full-video model.

Runs `batch_infer_eval_behaviorscope_y.py` over a fixed 5-video subset of the
MARS held-out test set at multiple inference strides, captures GPU/CPU
hardware telemetry per run, and aggregates frame F1, bout F1, and bout-edge
offset metrics into a single long-format CSV.

The five videos are pre-selected to span attack, investigation, and mount
behaviors with a deliberately wide bout-duration range. All five are part of
the existing v5 R5 + ablation analyses (no exclusion, no QC drop).

Outputs (under --output_root, default data/manuscript_v5/stride_sweep/):
  stride_<S>/<video_id>/                        per-(stride, video) batch output root
  stride_<S>/<video_id>/<split>/<video_id>/     per-video files (behavior CSV, infer_meta.json)
  stride_<S>/<video_id>/per_class_metrics.csv   etc. (run-level summary CSVs scoped to one video)
  stride_sweep_metrics.csv          one row per (stride, video, class)
  stride_sweep_hardware.csv         one row per (stride, video)
  stride_sweep_bout_offsets_raw.csv per-matched-bout signed offsets, for distribution plots
  stride_sweep_reference_parity.csv stride-16 parity check against the full R5 held-out eval
  stride_sweep_runlog.json          provenance: invocations, durations, environment

Each (stride, video) gets its own batch output root so the batch driver's
run-level summary CSVs (per_class_metrics.csv, bout_matches.csv,
predicted_bouts.csv, ground_truth_bouts.csv) are not overwritten by sibling
videos at the same stride.

Run:
  python scripts/run_inference_stride_sweep.py --mars_root <path-to-MARS_data>

Hardware capture is best-effort. nvidia-smi must be on PATH for GPU sampling.

--skip_existing is OFF by default for manuscript sweeps. The batch driver's
resume check looks only at file presence, not at whether the existing files
were produced under the same model/config/AMP/calibration as the current
invocation; for safety, runs are re-executed from scratch unless you opt in.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]

# -- Fixed selection (rationale documented in manuscript_outputs/27_stride_sweep_analysis.qmd) -----
# Five MARS test videos jointly covering attack/investigation/mount with a
# deliberately wide bout-duration range. All present in every v5 ablation arm.
PICKED_VIDEOS: list[tuple[str, str]] = [
    ("test_2", "Mouse156_20151012_20-16-37"),  # 30 attack / 90 invest / 3 mount, rapid behavior switching
    ("test_1", "Mouse069_20160709_16-02-03"),  # 18 / 47 / 1, long attack + long investigation
    ("test_2", "Mouse155_20151129_22-41-43"),  # 2 / 30 / 1, very long attack bouts
    ("test_2", "Mouse159_20151129_20-48-38"),  # 12 / 37 / 1, medium attack + short investigation
    ("test_1", "Mouse063_20160526_19-14-46"),  # 0 / 101 / 43, mount-rich mating-context regime
]
DEFAULT_STRIDES: list[int] = [16, 8, 4]
CLASSES = ("attack", "investigation", "mount", "other")
BEHAVIOR_CLASSES = ("attack", "investigation", "mount")

# -- Default paths (override via CLI) ----------------------------------------------------------------
DEFAULT_MARS_ROOT = str(Path.cwd())
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "manuscript_v5" / "stride_sweep"
DEFAULT_MODEL_PATH = REPO_ROOT / "data" / "manuscript_v5" / "training_runs" / "R5_yolo_attn_v5_full" / "best_model_macro_f1.pt"
DEFAULT_MODEL_CONFIG = REPO_ROOT / "data" / "manuscript_v5" / "training_runs" / "R5_yolo_attn_v5_full" / "config.json"
DEFAULT_YOLO_WEIGHTS = REPO_ROOT / "mars_yolo_pose" / "runs" / "pose" / "train" / "weights" / "best.pt"
DEFAULT_CALIBRATION = REPO_ROOT / "mars_behaviorscope_npz_full" / "calibration.json"
DEFAULT_REFERENCE_EVAL_ROOT = REPO_ROOT / "data" / "manuscript_v5" / "test_eval" / "R5_yolo_attn_v5_full_mars_test_eval"

BATCH_DRIVER = REPO_ROOT / "scripts" / "batch_infer_eval_behaviorscope_y.py"


# -----------------------------------------------------------------------------------------------------
# Hardware sampler
# -----------------------------------------------------------------------------------------------------
@dataclass
class HardwareSample:
    timestamp: float
    gpu_util_pct: float | None
    gpu_mem_used_mb: float | None
    gpu_mem_total_mb: float | None


@dataclass
class HardwareTelemetry:
    samples: list[HardwareSample] = field(default_factory=list)
    gpu_name: str | None = None
    driver_version: str | None = None
    cuda_version: str | None = None
    cpu_brand: str = ""
    cpu_logical_cores: int = 0
    host_ram_total_gb: float = 0.0


def _try_nvidia_smi_query() -> tuple[str | None, str | None, str | None]:
    """One-shot query for static GPU info."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            text=True, timeout=5.0,
        )
        line = out.strip().splitlines()[0]
        name, driver = [p.strip() for p in line.split(",", 1)]
    except Exception:
        return None, None, None
    cuda_version = None
    try:
        out = subprocess.check_output(["nvidia-smi"], text=True, timeout=5.0)
        for ln in out.splitlines():
            if "CUDA Version:" in ln:
                cuda_version = ln.split("CUDA Version:")[-1].strip().split()[0]
                break
    except Exception:
        pass
    return name, driver, cuda_version


def _read_static_host_info(tel: HardwareTelemetry) -> None:
    tel.cpu_brand = platform.processor() or platform.machine()
    tel.cpu_logical_cores = os.cpu_count() or 0
    try:
        import psutil  # type: ignore
        tel.host_ram_total_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        tel.host_ram_total_gb = 0.0
    name, driver, cuda = _try_nvidia_smi_query()
    tel.gpu_name = name
    tel.driver_version = driver
    tel.cuda_version = cuda


class HardwareSampler:
    """Background nvidia-smi sampler at ~1 Hz. Best-effort; silently no-ops if nvidia-smi missing."""

    def __init__(self, interval_s: float = 1.0):
        self.interval_s = interval_s
        self.samples: list[HardwareSample] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._available = shutil.which("nvidia-smi") is not None

    def start(self) -> None:
        if not self._available:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval_s * 2.0)
        self._thread = None

    def _loop(self) -> None:
        while not self._stop.is_set():
            sample = self._sample_once()
            if sample is not None:
                self.samples.append(sample)
            self._stop.wait(self.interval_s)

    def _sample_once(self) -> HardwareSample | None:
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True, timeout=2.0,
            )
            parts = [p.strip() for p in out.strip().splitlines()[0].split(",")]
            return HardwareSample(
                timestamp=time.time(),
                gpu_util_pct=float(parts[0]),
                gpu_mem_used_mb=float(parts[1]),
                gpu_mem_total_mb=float(parts[2]),
            )
        except Exception:
            return HardwareSample(time.time(), None, None, None)


def summarize_samples(samples: list[HardwareSample]) -> dict[str, float | None]:
    util = [s.gpu_util_pct for s in samples if s.gpu_util_pct is not None]
    mem = [s.gpu_mem_used_mb for s in samples if s.gpu_mem_used_mb is not None]
    out: dict[str, float | None] = {
        "n_samples": len(samples),
        "gpu_util_mean": None,
        "gpu_util_p95": None,
        "gpu_util_max": None,
        "gpu_mem_used_mean_mb": None,
        "gpu_mem_used_p95_mb": None,
        "gpu_mem_used_max_mb": None,
    }
    if util:
        out["gpu_util_mean"] = round(statistics.mean(util), 2)
        out["gpu_util_p95"] = round(_percentile(util, 95), 2)
        out["gpu_util_max"] = round(max(util), 2)
    if mem:
        out["gpu_mem_used_mean_mb"] = round(statistics.mean(mem), 1)
        out["gpu_mem_used_p95_mb"] = round(_percentile(mem, 95), 1)
        out["gpu_mem_used_max_mb"] = round(max(mem), 1)
    return out


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(round((q / 100.0) * (len(s) - 1)))))
    return s[idx]


# -----------------------------------------------------------------------------------------------------
# Per-(stride, video) inference
# -----------------------------------------------------------------------------------------------------
def run_one(
    stride: int,
    split: str,
    video_id: str,
    args: argparse.Namespace,
    output_root: Path,
) -> dict:
    """Run inference + eval for one (stride, video) and return a result dict.

    Each (stride, video) gets its own batch output root so that the batch
    driver's run-level summary CSVs (per_class_metrics.csv, bout_matches.csv,
    predicted_bouts.csv, ground_truth_bouts.csv) are not overwritten by the
    next video's invocation. Layout:

        output_root/stride_<S>/<video_id>/                <- batch_driver output_root
        output_root/stride_<S>/<video_id>/<split>/<video_id>/<files>
    """
    stride_dir = output_root / f"stride_{stride}"
    stride_dir.mkdir(parents=True, exist_ok=True)
    run_dir = stride_dir / video_id  # per-(stride, video) batch output root
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python, str(BATCH_DRIVER),
        "--mars_root", str(args.mars_root),
        "--splits", split,
        "--model_path", str(args.model_path),
        "--model_config", str(args.model_config),
        "--yolo_weights", str(args.yolo_weights),
        "--calibration_json", str(args.calibration_json),
        "--output_root", str(run_dir),
        "--video_filter", video_id,
        "--window_stride", str(stride),
        "--yolo_batch", str(args.yolo_batch),
        "--pose_conf_threshold", str(args.pose_conf_threshold),
        "--temporal_smoothing_window", str(args.temporal_smoothing_window),
        "--bout_min_duration_frames", str(args.bout_min_duration_frames),
        "--source_mode", args.source_mode,
        "--no_videos",
        "--log_metrics",
    ]
    if args.amp:
        cmd.append("--amp")
    if args.skip_existing:
        cmd.append("--skip_existing")
    if args.overwrite:
        cmd.append("--overwrite")

    print(f"[stride={stride}] [{split}/{video_id}] launching...", flush=True)
    print("    " + " ".join(cmd), flush=True)

    sampler = HardwareSampler(interval_s=args.hw_interval_s)
    sampler.start()
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    wall_s = time.time() - t0
    sampler.stop()

    if proc.returncode != 0:
        print(f"[stride={stride}] [{split}/{video_id}] FAILED rc={proc.returncode}", flush=True)
        return _failed_result(
            stride=stride, split=split, video_id=video_id, run_dir=run_dir,
            returncode=proc.returncode, wall_seconds=round(wall_s, 2),
            hw=summarize_samples(sampler.samples),
            fail_reason=f"non-zero return code {proc.returncode}",
        )

    result = _validated_run_result(
        stride=stride, split=split, video_id=video_id, run_dir=run_dir,
        returncode=0, wall_seconds=round(wall_s, 2),
        hw=summarize_samples(sampler.samples),
    )
    if result.get("status") != "ok":
        print(f"[stride={stride}] [{split}/{video_id}] FAILED: {result.get('fail_reason')}", flush=True)
        return result
    print(f"[stride={stride}] [{split}/{video_id}] ok wall={wall_s:.1f}s fps={result['end_to_end_fps']}", flush=True)
    return result


def _failed_result(
    *,
    stride: int,
    split: str,
    video_id: str,
    run_dir: Path,
    returncode: int | None,
    wall_seconds: float | None,
    hw: dict,
    fail_reason: str,
) -> dict:
    return {
        "stride": stride,
        "split": split,
        "video_id": video_id,
        "status": "failed",
        "returncode": returncode,
        "fail_reason": fail_reason,
        "wall_seconds": wall_seconds,
        "end_to_end_fps": None,
        "n_frames": None,
        "hw": hw,
        "video_dir": None,
        "run_dir": str(run_dir),
    }


def _validated_run_result(
    *,
    stride: int,
    split: str,
    video_id: str,
    run_dir: Path,
    returncode: int | None,
    wall_seconds: float | None,
    hw: dict,
) -> dict:
    """Validate a completed per-(stride, video) output root before aggregation."""
    video_dir = run_dir / split / video_id

    def failed(reason: str) -> dict:
        return _failed_result(
            stride=stride, split=split, video_id=video_id, run_dir=run_dir,
            returncode=returncode, wall_seconds=wall_seconds, hw=hw,
            fail_reason=reason,
        )

    if not run_dir.is_dir():
        return failed(f"run_dir missing at {run_dir}")
    if not video_dir.is_dir():
        return failed(f"video_dir missing at {video_dir}")

    fail_csv = run_dir / "failures.csv"
    fail_rows = _filter_csv(fail_csv, lambda row: row.get("video_id") == video_id) if fail_csv.is_file() else []
    if fail_rows:
        stages = ", ".join(sorted({r.get("stage", "?") for r in fail_rows}))
        return failed(f"failures.csv has {len(fail_rows)} row(s) for {video_id} (stage(s): {stages})")

    pvs_csv = run_dir / "per_video_summary.csv"
    if not pvs_csv.is_file():
        return failed(f"per_video_summary.csv missing at {pvs_csv}")
    pvs_rows = _filter_csv(pvs_csv, lambda row: row.get("video_id") == video_id)
    if not pvs_rows:
        return failed(f"per_video_summary.csv has no row for {video_id}")

    per_class_csv = run_dir / "per_class_metrics.csv"
    if not per_class_csv.is_file():
        return failed(f"per_class_metrics.csv missing at {per_class_csv}")
    per_class_rows = _filter_csv(
        per_class_csv,
        lambda row: row.get("video_id") == video_id and row.get("prediction_type") == "smoothed_frames",
    )
    if len(per_class_rows) < len(CLASSES):
        return failed(f"per_class_metrics.csv has {len(per_class_rows)} smoothed_frames row(s) for {video_id}")

    gt_bouts_csv = run_dir / "ground_truth_bouts.csv"
    if not gt_bouts_csv.is_file():
        return failed(f"ground_truth_bouts.csv missing at {gt_bouts_csv}")
    gt_bouts = _filter_csv(gt_bouts_csv, lambda row: row.get("video_id") == video_id)
    if not gt_bouts:
        return failed(f"ground_truth_bouts.csv has no row for {video_id}")

    predicted_bouts_csv = run_dir / "predicted_bouts.csv"
    if not predicted_bouts_csv.is_file():
        return failed(f"predicted_bouts.csv missing at {predicted_bouts_csv}")
    bout_matches_csv = run_dir / "bout_matches.csv"
    if not bout_matches_csv.exists():
        return failed(f"bout_matches.csv missing at {bout_matches_csv}")

    n_frames = _count_video_frames(video_dir)
    if n_frames <= 0:
        return failed("per-frame behavior CSV missing or empty (n_frames=0)")

    fps = (n_frames / wall_seconds) if (wall_seconds and wall_seconds > 0) else None
    return {
        "stride": stride,
        "split": split,
        "video_id": video_id,
        "status": "ok",
        "returncode": returncode,
        "wall_seconds": wall_seconds,
        "n_frames": n_frames,
        "end_to_end_fps": round(fps, 2) if fps is not None else None,
        "hw": hw,
        "video_dir": str(video_dir),
        "run_dir": str(run_dir),
    }


def _count_video_frames(video_dir: Path) -> int:
    """Count frames from the per-video behavior CSV (one row per frame)."""
    if not video_dir.is_dir():
        return 0
    candidates = list(video_dir.glob("*.behavior.smoothed_frames.csv"))
    if not candidates:
        candidates = list(video_dir.glob("*.behavior.csv"))
    if not candidates:
        return 0
    n = 0
    with open(candidates[0], encoding="utf-8") as f:
        next(f, None)  # header
        for _ in f:
            n += 1
    return n


# -----------------------------------------------------------------------------------------------------
# Aggregation: read per-stride outputs and produce stride_sweep_metrics.csv + bout offsets
# -----------------------------------------------------------------------------------------------------
def aggregate_metrics(
    output_root: Path,
    stride_results: list[dict],
    iou_thresholds: tuple[float, ...] = (0.25, 0.50),
    edge_tolerance_frames: tuple[int, ...] = (2, 5, 10),
) -> tuple[Path, Path, Path]:
    metrics_rows: list[dict] = []
    raw_offsets_rows: list[dict] = []

    for r in stride_results:
        if r.get("status") != "ok":
            continue
        # run_dir is the per-(stride, video) batch output root. Each (stride,
        # video) has its own copy of the run-level summary CSVs, so reading
        # them does not race with sibling videos at the same stride.
        run_dir = Path(r["run_dir"])
        stride = r["stride"]
        video_id = r["video_id"]
        split = r["split"]

        per_class_csv = run_dir / "per_class_metrics.csv"
        per_class = _filter_csv(per_class_csv, lambda row: row.get("video_id") == video_id and row.get("prediction_type") == "smoothed_frames")
        bout_matches_csv = run_dir / "bout_matches.csv"
        bout_matches = _filter_csv(bout_matches_csv, lambda row: row.get("video_id") == video_id and row.get("prediction_type") == "smoothed_frames")
        predicted_bouts_csv = run_dir / "predicted_bouts.csv"
        predicted_bouts = _filter_csv(predicted_bouts_csv, lambda row: row.get("video_id") == video_id and row.get("prediction_type") == "smoothed_frames")
        gt_bouts_csv = run_dir / "ground_truth_bouts.csv"
        gt_bouts = _filter_csv(gt_bouts_csv, lambda row: row.get("video_id") == video_id)

        # Frame-level metrics by class (rows already pooled per video by per_class_metrics.csv)
        for cls in CLASSES + ("all", "behavior_only"):
            row = _emit_metrics_row(
                stride=stride, split=split, video_id=video_id, cls=cls,
                wall_seconds=r["wall_seconds"], end_to_end_fps=r.get("end_to_end_fps"),
                n_frames=r.get("n_frames"),
                per_class=per_class,
                bout_matches=bout_matches,
                predicted_bouts=predicted_bouts,
                gt_bouts=gt_bouts,
                iou_thresholds=iou_thresholds,
                edge_tolerance_frames=edge_tolerance_frames,
            )
            metrics_rows.append(row)

        # Per-bout signed offsets at iou=0.25 for downstream distribution plots
        for m in bout_matches:
            try:
                if abs(float(m["threshold"]) - 0.25) > 1e-6:
                    continue
                raw_offsets_rows.append({
                    "stride": stride, "split": split, "video_id": video_id,
                    "class": m["class"],
                    "tiou": float(m["tiou"]),
                    "pred_start": int(m["pred_start"]), "pred_end": int(m["pred_end"]),
                    "gt_start": int(m["gt_start"]), "gt_end": int(m["gt_end"]),
                    "start_offset_frames": int(m["start_offset_frames"]),
                    "end_offset_frames": int(m["end_offset_frames"]),
                })
            except Exception:
                continue

    metrics_path = output_root / "stride_sweep_metrics.csv"
    hardware_path = output_root / "stride_sweep_hardware.csv"
    raw_offsets_path = output_root / "stride_sweep_bout_offsets_raw.csv"

    _write_csv(metrics_path, metrics_rows)
    _write_csv(raw_offsets_path, raw_offsets_rows)
    _write_csv(hardware_path, [_emit_hardware_row(r) for r in stride_results])
    return metrics_path, hardware_path, raw_offsets_path


def write_reference_parity(
    *,
    output_root: Path,
    metrics_path: Path,
    reference_eval_root: Path | None,
    baseline_stride: int = 16,
) -> Path | None:
    """Compare baseline-stride rows against the existing full R5 held-out eval."""
    if reference_eval_root is None:
        return None

    parity_path = output_root / "stride_sweep_reference_parity.csv"
    if not reference_eval_root.is_dir():
        _write_csv(parity_path, [{
            "status": "reference_missing",
            "reference_eval_root": str(reference_eval_root),
        }])
        return parity_path
    if not metrics_path.is_file() or metrics_path.stat().st_size == 0:
        _write_csv(parity_path, [{
            "status": "metrics_missing",
            "reference_eval_root": str(reference_eval_root),
        }])
        return parity_path

    current_rows = _read_csv(metrics_path)
    current_by_key = {
        (int(float(r["stride"])), r["split"], r["video_id"], r["class"]): r
        for r in current_rows
        if str(r.get("stride", "")).strip()
    }
    reference_by_key: dict[tuple[int, str, str, str], dict] = {}
    for split, video_id in PICKED_VIDEOS:
        for row in _eval_metric_rows_from_root(
            eval_root=reference_eval_root,
            stride=baseline_stride,
            split=split,
            video_id=video_id,
        ):
            reference_by_key[(baseline_stride, split, video_id, row["class"])] = row

    metric_cols = [
        "frame_f1",
        "bout_f1_iou25",
        "bout_f1_iou50",
        "gt_bouts",
        "pred_bouts",
        "n_matched_bouts_iou25",
        "median_abs_start_offset",
        "median_abs_end_offset",
        "mean_abs_start_offset",
        "mean_abs_end_offset",
        "pct_bouts_within_pm2_frames",
        "pct_bouts_within_pm5_frames",
        "pct_bouts_within_pm10_frames",
    ]
    rows: list[dict] = []
    for split, video_id in PICKED_VIDEOS:
        for cls in CLASSES + ("all", "behavior_only"):
            key = (baseline_stride, split, video_id, cls)
            cur = current_by_key.get(key)
            ref = reference_by_key.get(key)
            for metric in metric_cols:
                cur_val = _safe_float(cur.get(metric)) if cur else None
                ref_val = _safe_float(ref.get(metric)) if ref else None
                if cur is None:
                    status = "missing_current"
                elif ref is None:
                    status = "missing_reference"
                elif cur_val is None and ref_val is None:
                    status = "ok"
                elif cur_val is None or ref_val is None:
                    status = "mismatch"
                else:
                    status = "ok" if abs(cur_val - ref_val) <= 1e-6 else "mismatch"
                rows.append({
                    "status": status,
                    "baseline_stride": baseline_stride,
                    "split": split,
                    "video_id": video_id,
                    "class": cls,
                    "metric": metric,
                    "current": cur_val,
                    "reference": ref_val,
                    "delta": round(cur_val - ref_val, 9) if cur_val is not None and ref_val is not None else None,
                    "reference_eval_root": str(reference_eval_root),
                })
    _write_csv(parity_path, rows)
    return parity_path


def _eval_metric_rows_from_root(
    *,
    eval_root: Path,
    stride: int,
    split: str,
    video_id: str,
) -> list[dict]:
    per_class = _filter_csv(
        eval_root / "per_class_metrics.csv",
        lambda row: row.get("video_id") == video_id and row.get("prediction_type") == "smoothed_frames",
    )
    bout_matches = _filter_csv(
        eval_root / "bout_matches.csv",
        lambda row: row.get("video_id") == video_id and row.get("prediction_type") == "smoothed_frames",
    )
    predicted_bouts = _filter_csv(
        eval_root / "predicted_bouts.csv",
        lambda row: row.get("video_id") == video_id and row.get("prediction_type") == "smoothed_frames",
    )
    gt_bouts = _filter_csv(eval_root / "ground_truth_bouts.csv", lambda row: row.get("video_id") == video_id)
    return [
        _emit_metrics_row(
            stride=stride, split=split, video_id=video_id, cls=cls,
            wall_seconds=None, end_to_end_fps=None, n_frames=None,
            per_class=per_class,
            bout_matches=bout_matches,
            predicted_bouts=predicted_bouts,
            gt_bouts=gt_bouts,
            iou_thresholds=(0.25, 0.50),
            edge_tolerance_frames=(2, 5, 10),
        )
        for cls in CLASSES + ("all", "behavior_only")
    ]


def _filter_csv(path: Path, pred) -> list[dict]:
    if not path.is_file():
        return []
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if pred(row):
                out.append(row)
    return out


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _safe_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        x = float(value)
    except Exception:
        return None
    return None if _isnan(x) else x


def _load_prior_runlog_timings(output_root: Path) -> dict[tuple[int, str], dict]:
    """Recover wall/FPS values when rebuilding summaries in aggregate-only mode."""
    runlog_path = output_root / "stride_sweep_runlog.json"
    if not runlog_path.is_file():
        return {}
    try:
        data = json.loads(runlog_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    out: dict[tuple[int, str], dict] = {}
    for row in data.get("runs", []):
        try:
            key = (int(row["stride"]), str(row["video_id"]))
        except Exception:
            continue
        out[key] = row
    return out


def _emit_metrics_row(
    *,
    stride: int, split: str, video_id: str, cls: str,
    wall_seconds: float | None, end_to_end_fps: float | None, n_frames: int | None,
    per_class: list[dict], bout_matches: list[dict],
    predicted_bouts: list[dict], gt_bouts: list[dict],
    iou_thresholds: tuple[float, ...],
    edge_tolerance_frames: tuple[int, ...],
) -> dict:
    row: dict = {
        "stride": stride, "split": split, "video_id": video_id, "class": cls,
        "wall_seconds": wall_seconds, "end_to_end_fps": end_to_end_fps, "n_frames": n_frames,
    }

    # Frame F1 (per-class rows have frame_f1; pooled "all" averages 4-class; behavior_only averages 3-class)
    frame_f1_by_cls = {r["class"]: float(r.get("frame_f1") or "nan") for r in per_class if r.get("class")}
    if cls in CLASSES:
        row["frame_f1"] = frame_f1_by_cls.get(cls)
    elif cls == "all":
        vals = [frame_f1_by_cls[c] for c in CLASSES if c in frame_f1_by_cls and not _isnan(frame_f1_by_cls[c])]
        row["frame_f1"] = round(statistics.mean(vals), 6) if vals else None
    else:  # behavior_only
        vals = [frame_f1_by_cls[c] for c in BEHAVIOR_CLASSES if c in frame_f1_by_cls and not _isnan(frame_f1_by_cls[c])]
        row["frame_f1"] = round(statistics.mean(vals), 6) if vals else None

    # Bout F1 at each IoU. The eval pipeline (batch_infer_eval_behaviorscope_y.py)
    # builds GT and predicted bouts with include_other=False. As a result:
    #   - "other" rows have no bout support at all -> bout F1 / edge metrics = None.
    #   - "all" rows are an attempted 4-class pool, but bouts only exist for the
    #     three behavior classes, so a bout-side "all" is effectively
    #     "behavior_only" with a misleading name. We null it to avoid that
    #     misread; "behavior_only" is the canonical pooled bout row.
    bout_supported = cls in BEHAVIOR_CLASSES or cls == "behavior_only"
    for thr in iou_thresholds:
        if bout_supported:
            f1 = _bout_f1_for_class(predicted_bouts, gt_bouts, bout_matches, thr, cls)
        else:
            f1 = None
        row[f"bout_f1_iou{int(thr*100):02d}"] = f1

    # GT and prediction counts. Frame-side "all" averages four classes, but
    # bout-side "all" excludes "other"; here we report the bout-pipeline scope.
    if cls in CLASSES:
        row["gt_bouts"] = sum(1 for r in gt_bouts if r.get("class") == cls)
        row["pred_bouts"] = sum(1 for r in predicted_bouts if r.get("class") == cls)
    elif cls == "behavior_only":
        row["gt_bouts"] = sum(1 for r in gt_bouts if r.get("class") in BEHAVIOR_CLASSES)
        row["pred_bouts"] = sum(1 for r in predicted_bouts if r.get("class") in BEHAVIOR_CLASSES)
    else:  # cls == "all": frame-level pool; bout pipeline excludes "other".
        row["gt_bouts"] = None
        row["pred_bouts"] = None

    # Bout-edge offset metrics (only at iou=0.25 to keep the table compact).
    # Same scope rule as bout F1: skip "other" and "all".
    if not bout_supported:
        offsets = []
    elif cls in BEHAVIOR_CLASSES:
        offsets = [m for m in bout_matches if abs(float(m["threshold"]) - 0.25) < 1e-6 and m.get("class") == cls]
    else:  # behavior_only
        offsets = [m for m in bout_matches if abs(float(m["threshold"]) - 0.25) < 1e-6 and m.get("class") in BEHAVIOR_CLASSES]

    if offsets:
        starts_abs = [abs(int(m["start_offset_frames"])) for m in offsets]
        ends_abs = [abs(int(m["end_offset_frames"])) for m in offsets]
        row["n_matched_bouts_iou25"] = len(offsets)
        row["median_abs_start_offset"] = statistics.median(starts_abs)
        row["median_abs_end_offset"] = statistics.median(ends_abs)
        row["mean_abs_start_offset"] = round(statistics.mean(starts_abs), 2)
        row["mean_abs_end_offset"] = round(statistics.mean(ends_abs), 2)
        for k in edge_tolerance_frames:
            row[f"pct_bouts_within_pm{k}_frames"] = round(
                100.0 * sum(
                    1 for m in offsets
                    if abs(int(m["start_offset_frames"])) <= k and abs(int(m["end_offset_frames"])) <= k
                ) / len(offsets),
                2,
            )
    else:
        # Distinguish "no matches" (zero) from "not applicable" (None) so the
        # downstream chapter can render them differently.
        row["n_matched_bouts_iou25"] = 0 if bout_supported else None
        for key in ("median_abs_start_offset", "median_abs_end_offset",
                    "mean_abs_start_offset", "mean_abs_end_offset"):
            row[key] = None
        for k in edge_tolerance_frames:
            row[f"pct_bouts_within_pm{k}_frames"] = None

    return row


def _bout_f1_for_class(
    predicted_bouts: list[dict],
    gt_bouts: list[dict],
    bout_matches: list[dict],
    iou_threshold: float,
    cls: str,
) -> float | None:
    """Recompute bout F1 from predicted/GT bouts and matches at this IoU threshold for a class."""
    if cls in CLASSES:
        target_classes = {cls}
    elif cls == "behavior_only":
        target_classes = set(BEHAVIOR_CLASSES)
    else:
        target_classes = set(CLASSES)

    n_gt = sum(1 for r in gt_bouts if r.get("class") in target_classes)
    n_pred = sum(1 for r in predicted_bouts if r.get("class") in target_classes)
    n_tp = sum(
        1 for m in bout_matches
        if abs(float(m["threshold"]) - iou_threshold) < 1e-6
        and m.get("class") in target_classes
    )
    if n_gt == 0 and n_pred == 0:
        return None
    precision = (n_tp / n_pred) if n_pred > 0 else 0.0
    recall = (n_tp / n_gt) if n_gt > 0 else 0.0
    if (precision + recall) == 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 6)


def _isnan(x) -> bool:
    try:
        return x != x  # NaN check
    except Exception:
        return False


def _emit_hardware_row(r: dict) -> dict:
    hw = r.get("hw") or {}
    return {
        "stride": r.get("stride"),
        "split": r.get("split"),
        "video_id": r.get("video_id"),
        "status": r.get("status"),
        "wall_seconds": r.get("wall_seconds"),
        "end_to_end_fps": r.get("end_to_end_fps"),
        "n_frames": r.get("n_frames"),
        "n_samples": hw.get("n_samples"),
        "gpu_util_mean": hw.get("gpu_util_mean"),
        "gpu_util_p95": hw.get("gpu_util_p95"),
        "gpu_util_max": hw.get("gpu_util_max"),
        "gpu_mem_used_mean_mb": hw.get("gpu_mem_used_mean_mb"),
        "gpu_mem_used_p95_mb": hw.get("gpu_mem_used_p95_mb"),
        "gpu_mem_used_max_mb": hw.get("gpu_mem_used_max_mb"),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in cols:
                cols.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# -----------------------------------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mars_root", default=DEFAULT_MARS_ROOT,
                   help="Path to MARS_data/ root containing test_1/ and test_2/.")
    p.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT),
                   help="Where stride_<S>/ subfolders + summary CSVs are written.")
    p.add_argument("--strides", type=int, nargs="+", default=DEFAULT_STRIDES)
    p.add_argument("--model_path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--model_config", default=str(DEFAULT_MODEL_CONFIG))
    p.add_argument("--yolo_weights", default=str(DEFAULT_YOLO_WEIGHTS))
    p.add_argument("--calibration_json", default=str(DEFAULT_CALIBRATION))
    p.add_argument("--source_mode", choices=["mp4", "seq"], default="mp4")
    p.add_argument("--pose_conf_threshold", type=float, default=0.2,
                   help="Match the v5 batch eval default (overridden from 0.3 to 0.2 for v5).")
    p.add_argument("--yolo_batch", type=int, default=32,
                   help="YOLO inference batch size. Default 32 matches the recorded "
                        "R5 full-video held-out test evaluation; override only when "
                        "intentionally running a separate deployment-throughput variant.")
    p.add_argument("--temporal_smoothing_window", type=int, default=5)
    p.add_argument("--bout_min_duration_frames", type=int, default=15)
    p.add_argument("--amp", action="store_true", default=True,
                   help="Match v5 R5 inference (bf16 AMP). Pass --no-amp to disable.")
    p.add_argument("--no-amp", dest="amp", action="store_false")
    p.add_argument("--skip_existing", action="store_true", default=False,
                   help="Skip a (stride, video) run if its batch output already has "
                        "predicted CSVs. The batch driver's resume check inspects only "
                        "file presence, not model/config/AMP/calibration parity, so for "
                        "manuscript sweeps this is OFF by default - re-run from scratch "
                        "to ensure all outputs reflect the current settings. Use --skip_existing "
                        "only when you know the existing outputs were produced under identical "
                        "settings to the current invocation.")
    p.add_argument("--overwrite", action="store_true",
                   help="Force re-running even if outputs exist.")
    p.add_argument("--hw_interval_s", type=float, default=1.0,
                   help="Hardware sampling interval in seconds (best-effort nvidia-smi).")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--aggregate_only", action="store_true",
                   help="Skip inference; just re-aggregate metrics from existing per-stride outputs.")
    p.add_argument("--reference_eval_root", default=str(DEFAULT_REFERENCE_EVAL_ROOT),
                   help="Existing full R5 held-out evaluation root used for stride-16 parity checks.")
    p.add_argument("--no_reference_parity", action="store_true",
                   help="Do not write stride_sweep_reference_parity.csv.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    runlog_path = output_root / ("stride_sweep_aggregate_runlog.json" if args.aggregate_only else "stride_sweep_runlog.json")

    # Static host info written once
    tel = HardwareTelemetry()
    _read_static_host_info(tel)
    runlog: dict = {
        "start_time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "host": {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_brand": tel.cpu_brand,
            "cpu_logical_cores": tel.cpu_logical_cores,
            "host_ram_total_gb": tel.host_ram_total_gb,
            "gpu_name": tel.gpu_name,
            "driver_version": tel.driver_version,
            "cuda_version": tel.cuda_version,
        },
        "args": vars(args),
        "picked_videos": [{"split": s, "video_id": v} for s, v in PICKED_VIDEOS],
        "strides": args.strides,
        "runs": [],
    }

    print(f"[stride_sweep] picked videos: {len(PICKED_VIDEOS)}")
    for s, v in PICKED_VIDEOS:
        print(f"    {s}/{v}")
    print(f"[stride_sweep] strides: {args.strides}")
    print(f"[stride_sweep] output_root: {output_root}")
    print()

    stride_results: list[dict] = []
    if not args.aggregate_only:
        for stride in args.strides:
            for split, video_id in PICKED_VIDEOS:
                r = run_one(stride, split, video_id, args, output_root)
                stride_results.append(r)
                runlog["runs"].append({k: v for k, v in r.items() if k != "hw"})
                with open(runlog_path, "w", encoding="utf-8") as f:
                    json.dump(runlog, f, indent=2, default=str)
    else:
        # Re-construct stride_results from existing per-(stride, video) batch
        # output roots without re-running inference.
        prior_timings = _load_prior_runlog_timings(output_root)
        for stride in args.strides:
            stride_dir = output_root / f"stride_{stride}"
            for split, video_id in PICKED_VIDEOS:
                run_dir = stride_dir / video_id
                prior = prior_timings.get((stride, video_id), {})
                r = _validated_run_result(
                    stride=stride, split=split, video_id=video_id, run_dir=run_dir,
                    returncode=0,
                    wall_seconds=_safe_float(prior.get("wall_seconds")),
                    hw={"n_samples": 0},
                )
                if r.get("status") == "ok" and prior.get("end_to_end_fps") is not None:
                    r["end_to_end_fps"] = _safe_float(prior.get("end_to_end_fps"))
                if r.get("status") != "ok":
                    print(f"[aggregate_only] failed stride={stride} {split}/{video_id}: {r.get('fail_reason')}")
                stride_results.append(r)

    metrics_path, hw_path, offsets_path = aggregate_metrics(output_root, stride_results)
    reference_eval_root = None if args.no_reference_parity else Path(args.reference_eval_root)
    parity_path = write_reference_parity(
        output_root=output_root,
        metrics_path=metrics_path,
        reference_eval_root=reference_eval_root,
        baseline_stride=16,
    )

    runlog["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    runlog["metrics_csv"] = str(metrics_path)
    runlog["hardware_csv"] = str(hw_path)
    runlog["bout_offsets_raw_csv"] = str(offsets_path)
    runlog["reference_parity_csv"] = str(parity_path) if parity_path else None
    with open(runlog_path, "w", encoding="utf-8") as f:
        json.dump(runlog, f, indent=2, default=str)

    print()
    print("[stride_sweep] done.")
    print(f"  metrics: {metrics_path}")
    print(f"  hardware: {hw_path}")
    print(f"  bout offsets (raw): {offsets_path}")
    if parity_path:
        print(f"  reference parity: {parity_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
