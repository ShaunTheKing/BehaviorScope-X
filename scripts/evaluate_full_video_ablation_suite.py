#!/usr/bin/env python3
"""
evaluate_full_video_ablation_suite.py
========================
Overnight driver: run held-out test evaluation on the MARS test_1+test_2
videos for every v5 (full-video sliding-window) ablation training run, using
each run's ``best_model_macro_f1.pt`` checkpoint.

For each run it invokes ``scripts/batch_infer_eval_behaviorscope_y.py``,
which discovers test videos under MARS_root, runs ``infer_y.py`` per video,
evaluates predictions against the BENTO ``.annot`` ground truth, and writes
per-video / per-class / aggregate metrics into the run's eval output dir.

Pairs with:
  - scripts/evaluate_clip_ablation_suite.py - clip-arm version (same structure)
  - scripts/analyze_full_video_test_results.py - apples-to-apples clip-vs-full analysis

Defaults:
  - --pose_conf_threshold 0.2 (per the v5 inference convention; relaxed from
    the v4 default of 0.3 to recover more keypoints under occlusion)
  - --calibration_json mars_behaviorscope_npz_full/calibration.json (covers
    all 115 MARS videos including test_1/test_2; v5 NPZ build doesn't write
    its own because body-length is per-physical-video, not per-methodology)
  - Skips R5 by default (its test eval was already run on 2026-05-06 and
    sits at data/manuscript_v5/test_eval/R5_yolo_attn_v5_full_mars_test_eval/).
    Pass --include_r5 to redo it.

Resilient to partial completion:
  - skips runs whose ``per_class_metrics.csv`` already exists (unless --rerun)
  - skips individual videos whose ``*.behavior.csv`` already exists
  - logs each run to its own ``run.log`` for post-hoc debugging
  - never aborts the loop on a single-run failure; reports at the end
  - mirrors compact eval CSVs to manuscript_outputs/data/manuscript_v5/test_eval/
    after each successful run (skip with --no_mirror)

Usage:
    python scripts/evaluate_full_video_ablation_suite.py
    python scripts/evaluate_full_video_ablation_suite.py --rerun
    python scripts/evaluate_full_video_ablation_suite.py --runs R5 R4
    python scripts/evaluate_full_video_ablation_suite.py --include_r5
    python scripts/evaluate_full_video_ablation_suite.py --max_videos 2     # smoke test
    python scripts/evaluate_full_video_ablation_suite.py --no_mirror
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
BATCH_SCRIPT = SCRIPTS_DIR / "batch_infer_eval_behaviorscope_y.py"
TRAINING_RUNS_DIR = REPO_ROOT / "data" / "manuscript_v5" / "training_runs"
TEST_EVAL_DIR     = REPO_ROOT / "data" / "manuscript_v5" / "test_eval"
NOTEBOOK_TEST_EVAL_DIR = REPO_ROOT / "manuscript_outputs" / "data" / "manuscript_v5" / "test_eval"

# Run order: ablations first (R5 was already evaluated on 2026-05-06).
# Each tuple is (short, training_run_dirname).
RUNS: list[tuple[str, str]] = [
    ("R1",  "R1_yolo_attn_v5_crops_baseline"),
    ("R2",  "R2_yolo_attn_v5_pose_only"),
    ("R3",  "R3_yolo_attn_v5_visual_only"),
    ("R4",  "R4_yolo_attn_v5_no_group_rgb"),
    ("R4b", "R4b_yolo_attn_v5_no_relations"),
]
R5_ENTRY = ("R5", "R5_yolo_attn_v5_full")

DEFAULT_MARS_ROOT = str(Path.cwd())
DEFAULT_YOLO_WEIGHTS = REPO_ROOT / "mars_yolo_pose" / "runs" / "pose" / "train" / "weights" / "best.pt"
DEFAULT_CALIBRATION  = REPO_ROOT / "mars_behaviorscope_npz_full" / "calibration.json"

# Files to mirror into the notebook after a successful eval. NPZ-style large
# CSVs (predicted_bouts/ground_truth_bouts/bout_matches) get mirrored too -
# they're under ~5 MB each per run and compact relative to the per-video MP4
# overlays we're already skipping via --no_videos.
MIRROR_FILES = [
    "per_class_metrics.csv",
    "per_video_summary.csv",
    "predicted_bouts.csv",
    "ground_truth_bouts.csv",
    "bout_matches.csv",
    "manifest.csv",
    "failures.csv",
    "aggregate_summary.json",
    "headline_metrics.json",
    "fps_summary.json",
    "hw_summary.json",
    "hw_stats.csv",
    "run.log",
]


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------------------------------------------------------
# Hardware monitor - once-per-second sampler of cpu/ram/gpu/vram/temp/power.
# Writes a per-run CSV and computes mean/median/peak/min on stop.
# ----------------------------------------------------------------------------
class HWMonitor:
    COLS = ["t_rel_s", "cpu_pct", "ram_gb_used", "gpu_util_pct",
            "vram_gb_used", "gpu_temp_c", "gpu_power_w"]

    def __init__(self, csv_path: Path, gpu_index: int = 0, interval_s: float = 1.0):
        self.csv_path = csv_path
        self.gpu_index = int(gpu_index)
        self.interval = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[dict] = []
        self._t0 = 0.0
        self._psutil = None
        self._pynvml = None
        self._nvml_handle = None
        self._init_libs()

    def _init_libs(self) -> None:
        try:
            import psutil
            self._psutil = psutil
            psutil.cpu_percent(interval=None)
        except Exception:
            self._psutil = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception:
            self._pynvml = None
            self._nvml_handle = None

    def _sample(self, t_rel: float) -> dict:
        row = {c: float("nan") for c in self.COLS}
        row["t_rel_s"] = round(t_rel, 2)
        if self._psutil is not None:
            try:
                row["cpu_pct"] = float(self._psutil.cpu_percent(interval=None))
                row["ram_gb_used"] = round(self._psutil.virtual_memory().used / 1e9, 2)
            except Exception:
                pass
        if self._pynvml is not None and self._nvml_handle is not None:
            try:
                util = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                mem = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                temp = self._pynvml.nvmlDeviceGetTemperature(
                    self._nvml_handle, self._pynvml.NVML_TEMPERATURE_GPU)
                row["gpu_util_pct"] = float(util.gpu)
                row["vram_gb_used"] = round(mem.used / 1e9, 2)
                row["gpu_temp_c"] = float(temp)
                try:
                    row["gpu_power_w"] = round(
                        self._pynvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0, 1)
                except Exception:
                    pass
            except Exception:
                pass
        return row

    def _run(self) -> None:
        self._t0 = time.time()
        while not self._stop.is_set():
            self._samples.append(self._sample(time.time() - self._t0))
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._stop.clear()
        self._samples = []
        self._thread = threading.Thread(target=self._run, daemon=True, name="hw-mon")
        self._thread.start()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.COLS)
            w.writeheader()
            w.writerows(self._samples)
        warm = self._samples[5:] if len(self._samples) > 10 else self._samples
        summary: dict = {"n_samples": len(self._samples)}
        for col in self.COLS[1:]:
            vals = [r[col] for r in warm if r[col] == r[col]]
            if vals:
                summary[col] = {
                    "mean": round(statistics.mean(vals), 2),
                    "median": round(statistics.median(vals), 2),
                    "peak": round(max(vals), 2),
                    "min": round(min(vals), 2),
                }
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
        return summary


def _compute_fps(metrics_csvs: list[Path]) -> dict:
    frames_total = 0
    seconds_total = 0.0
    per_video_fps: list[float] = []
    for csv_path in metrics_csvs:
        try:
            with open(csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if not rows:
                continue
            last = rows[-1]
            frames_done = float(last.get("frames_done", 0) or 0)
            elapsed = float(last.get("elapsed_s", 0) or 0)
            if elapsed > 0 and frames_done > 0:
                fps = frames_done / elapsed
                per_video_fps.append(fps)
                frames_total += int(frames_done)
                seconds_total += elapsed
        except Exception:
            continue
    if not per_video_fps:
        return {}
    return {
        "n_videos":      len(per_video_fps),
        "fps_mean":      round(statistics.mean(per_video_fps), 2),
        "fps_median":    round(statistics.median(per_video_fps), 2),
        "fps_min":       round(min(per_video_fps), 2),
        "fps_max":       round(max(per_video_fps), 2),
        "total_frames":  frames_total,
        "total_seconds": round(seconds_total, 1),
        "aggregate_fps": round(frames_total / seconds_total, 2) if seconds_total else None,
    }


def _compute_macro_f1(per_class_csv: Path,
                      pred_type: str = "smoothed_frames") -> dict:
    classes = ["attack", "investigation", "mount", "other"]
    if not per_class_csv.exists():
        return {}
    try:
        with open(per_class_csv, encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("prediction_type") == pred_type]
    except Exception:
        return {}
    if not rows:
        return {}
    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}
    for r in rows:
        c = r["class"]
        if c not in classes:
            continue
        tp[c] += int(r.get(f"cm_{c}", 0) or 0)
        for other in classes:
            if other != c:
                fn[c] += int(r.get(f"cm_{other}", 0) or 0)
    for r in rows:
        gt = r["class"]
        if gt not in classes:
            continue
        for predcls in classes:
            if predcls != gt:
                fp[predcls] += int(r.get(f"cm_{predcls}", 0) or 0)
    per_class = {}
    f1s = []
    for c in classes:
        denom_p = tp[c] + fp[c]
        denom_r = tp[c] + fn[c]
        prec = tp[c] / denom_p if denom_p > 0 else 0.0
        rec  = tp[c] / denom_r if denom_r > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "tp": tp[c], "fp": fp[c], "fn": fn[c]}
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)
    behavior_f1s = [per_class[c]["f1"] for c in ["attack", "investigation", "mount"]]
    return {
        "prediction_type":   pred_type,
        "macro_f1":          round(macro_f1, 4),
        "macro_f1_behavior": round(sum(behavior_f1s) / 3, 4),
        "per_class":         per_class,
    }


def _run_one(short: str, dirname: str, args: argparse.Namespace) -> dict:
    run_dir = TRAINING_RUNS_DIR / dirname
    model_path = run_dir / "best_model_macro_f1.pt"
    model_cfg  = run_dir / "config.json"
    out_root   = TEST_EVAL_DIR / f"{dirname}_mars_test_eval"
    out_root.mkdir(parents=True, exist_ok=True)

    log_path = out_root / "run.log"
    per_class_csv = out_root / "per_class_metrics.csv"

    result = {
        "short": short, "dirname": dirname,
        "out_root": str(out_root), "log_path": str(log_path),
        "started_at": _ts(),
        "status": "skipped", "elapsed_s": 0.0,
        "macro_f1": None, "error": None,
    }

    # Resume policy - if per_class_metrics.csv exists, recompute headline and skip
    if not args.rerun and per_class_csv.exists():
        try:
            head_smoothed = _compute_macro_f1(per_class_csv, "smoothed_frames")
            if head_smoothed:
                result["status"] = "already_done"
                result["macro_f1"] = head_smoothed.get("macro_f1")
                result["macro_f1_behavior"] = head_smoothed.get("macro_f1_behavior")
                print(f"[{_ts()}] [skip] {short}: per_class_metrics.csv exists "
                      f"(macro_f1={result['macro_f1']})", flush=True)
                return result
        except Exception:
            pass

    if not model_path.exists() or not model_cfg.exists():
        result["status"] = "missing_checkpoint"
        result["error"] = f"missing {model_path.name} or config.json under {run_dir}"
        print(f"[{_ts()}] [warn] {short}: {result['error']}", flush=True)
        return result

    cmd = [
        args.python, str(BATCH_SCRIPT),
        "--mars_root",       args.mars_root,
        "--splits",          *args.splits,
        "--model_path",      str(model_path),
        "--model_config",    str(model_cfg),
        "--yolo_weights",    args.yolo_weights,
        "--output_root",     str(out_root),
        "--device",          args.device,
        "--fps",             str(args.fps),
        "--window_stride",   str(args.window_stride),
        "--yolo_batch",      str(args.yolo_batch),
        "--pose_conf_threshold",       str(args.pose_conf_threshold),
        "--temporal_smoothing_window", str(args.temporal_smoothing_window),
        "--bout_min_duration_frames",  str(args.bout_min_duration_frames),
    ]
    if args.amp:                 cmd.append("--amp")
    if args.no_videos:           cmd.append("--no_videos")
    if args.calibration_json:    cmd.extend(["--calibration_json", args.calibration_json])
    if args.max_videos:          cmd.extend(["--max_videos", str(args.max_videos)])
    if args.video_filter:        cmd.extend(["--video_filter", args.video_filter])
    if args.skip_existing:       cmd.append("--skip_existing")
    if args.export_pose:         cmd.append("--export_pose")
    if args.log_metrics:         cmd.append("--log_metrics")
    if args.print_each_window:   cmd.append("--print_each_window")

    print(f"[{_ts()}] [run]  {short}  ->  {out_root}", flush=True)
    print(f"[{_ts()}]        cmd: {' '.join(cmd)}", flush=True)

    hw_csv = out_root / "hw_stats.csv"
    hw_summary_path = out_root / "hw_summary.json"
    hw_mon = HWMonitor(hw_csv, gpu_index=0, interval_s=1.0)
    hw_mon.start()

    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"# Command: {' '.join(cmd)}\n# Started: {_ts()}\n\n")
            f.flush()
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1, encoding="utf-8", errors="replace",
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                f.write(line); f.flush()
                sys.stdout.write(f"  [{short}] {line}"); sys.stdout.flush()
            proc.wait()
        result["elapsed_s"] = round(time.time() - t0, 1)
        hw_summary = hw_mon.stop()
        with open(hw_summary_path, "w", encoding="utf-8") as f:
            json.dump(hw_summary, f, indent=2)
        result["hw_summary"] = hw_summary
        metrics_csvs = sorted(out_root.rglob("*.behavior_metrics.csv"))
        if metrics_csvs:
            fps_summary = _compute_fps(metrics_csvs)
            with open(out_root / "fps_summary.json", "w", encoding="utf-8") as f:
                json.dump(fps_summary, f, indent=2)
            result["fps_summary"] = fps_summary
        if per_class_csv.exists():
            head_smoothed = _compute_macro_f1(per_class_csv, "smoothed_frames")
            head_raw      = _compute_macro_f1(per_class_csv, "raw_windows")
            with open(out_root / "headline_metrics.json", "w", encoding="utf-8") as f:
                json.dump({"smoothed_frames": head_smoothed,
                           "raw_windows": head_raw}, f, indent=2)
            result["macro_f1"] = head_smoothed.get("macro_f1")
            result["macro_f1_behavior"] = head_smoothed.get("macro_f1_behavior")
        if proc.returncode == 0:
            result["status"] = "ok"
            extra = ""
            if result.get("fps_summary"):
                extra = f"  fps_mean={result['fps_summary'].get('fps_mean')}"
            if result.get("hw_summary", {}).get("gpu_util_pct"):
                gpu = result["hw_summary"]["gpu_util_pct"]["mean"]
                extra += f"  gpu_util_mean={gpu}%"
            print(f"[{_ts()}] [done] {short}  elapsed={result['elapsed_s']}s  "
                  f"macro_f1={result['macro_f1']}{extra}", flush=True)
        else:
            result["status"] = "failed"
            result["error"] = f"exit code {proc.returncode}; see {log_path}"
            print(f"[{_ts()}] [fail] {short}  exit={proc.returncode}  log={log_path}",
                  flush=True)
    except KeyboardInterrupt:
        result["status"] = "interrupted"
        result["elapsed_s"] = round(time.time() - t0, 1)
        try: hw_mon.stop()
        except Exception: pass
        raise
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        result["elapsed_s"] = round(time.time() - t0, 1)
        try: hw_mon.stop()
        except Exception: pass
        print(f"[{_ts()}] [err]  {short}: {exc}", flush=True)
    return result


def _mirror_to_notebook(out_dir_name: str) -> None:
    """Copy compact eval artefacts into the notebook data tree.

    Notebook path mirrors the live path one-for-one:
      data/manuscript_v5/test_eval/<out_dir_name>/<file>
        -> manuscript_outputs/data/manuscript_v5/test_eval/<out_dir_name>/<file>

    Large per-video MP4 overlays are not in the source by default
    (--no_videos) so don't need filtering here.
    """
    src = TEST_EVAL_DIR / out_dir_name
    dst = NOTEBOOK_TEST_EVAL_DIR / out_dir_name
    if not src.exists():
        return
    dst.mkdir(parents=True, exist_ok=True)
    print(f"[{_ts()}] [mirror] {out_dir_name}", flush=True)
    for fname in MIRROR_FILES:
        sp = src / fname
        if sp.exists():
            try:
                shutil.copy2(sp, dst / fname)
            except Exception as exc:
                print(f"[{_ts()}] [warn]  mirror {fname}: {exc}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", default=None,
                   help="Subset of runs (short names: R1 R2 R3 R4 R4b). "
                        "Default: all 5 ablations (R5 already done).")
    p.add_argument("--include_r5", action="store_true",
                   help="Also re-evaluate R5 v5 full model (default: skip "
                        "since its eval was completed on 2026-05-06).")
    p.add_argument("--rerun", action="store_true",
                   help="Re-run even if per_class_metrics.csv exists.")
    p.add_argument("--no_mirror", action="store_true",
                   help="Skip copying eval CSVs into manuscript_outputs/data/.")
    # Pass-through args to the inner batch script
    p.add_argument("--mars_root", default=DEFAULT_MARS_ROOT)
    p.add_argument("--splits", nargs="+", default=["test_1", "test_2"])
    p.add_argument("--yolo_weights", default=str(DEFAULT_YOLO_WEIGHTS))
    p.add_argument("--calibration_json",
                   default=str(DEFAULT_CALIBRATION) if DEFAULT_CALIBRATION.exists() else None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--window_stride", type=int, default=16)
    p.add_argument("--yolo_batch", type=int, default=32)
    p.add_argument("--pose_conf_threshold", type=float, default=0.2,
                   help="Per-keypoint confidence threshold at inference. v5 "
                        "convention: 0.2 (relaxed from v4's 0.3 to recover "
                        "more keypoints under occlusion).")
    p.add_argument("--temporal_smoothing_window", type=int, default=5)
    p.add_argument("--bout_min_duration_frames", type=int, default=15)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no_videos", action="store_true", default=True,
                   help="Skip annotated MP4 export (saves disk + time).")
    p.add_argument("--export_pose", action="store_true", default=True)
    p.add_argument("--log_metrics", action="store_true", default=True)
    p.add_argument("--max_videos", type=int, default=0,
                   help="Smoke-test by limiting videos per run.")
    p.add_argument("--video_filter", default=None)
    p.add_argument("--skip_existing", action="store_true", default=True,
                   help="Skip per-video inference when behavior CSV already exists.")
    p.add_argument("--print_each_window", action="store_true")
    p.add_argument("--python", default=sys.executable)
    return p.parse_args()


def main() -> int:
    args = parse_args()

    candidates = list(RUNS)
    if args.include_r5:
        candidates = [R5_ENTRY] + candidates
    if args.runs is not None:
        wanted = set(args.runs)
        all_known = dict(candidates + [R5_ENTRY])
        candidates = [(s, all_known[s]) for s in args.runs if s in all_known]
        missing = wanted - set(s for s, _ in candidates)
        if missing:
            print(f"[{_ts()}] [warn] unknown run shorts: {sorted(missing)}", flush=True)
    runs_to_do = candidates

    print(f"[{_ts()}] [start] v5 test eval batch driver")
    print(f"[{_ts()}]        runs:                 {[r[0] for r in runs_to_do]}")
    print(f"[{_ts()}]        splits:               {args.splits}")
    print(f"[{_ts()}]        mars_root:            {args.mars_root}")
    print(f"[{_ts()}]        calibration_json:     {args.calibration_json}")
    print(f"[{_ts()}]        pose_conf_threshold:  {args.pose_conf_threshold}")
    print(f"[{_ts()}]        rerun:                {args.rerun}")
    print(f"[{_ts()}]        mirror:               {not args.no_mirror}")

    TEST_EVAL_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    overall_t0 = time.time()
    for short, dirname in runs_to_do:
        res = _run_one(short, dirname, args)
        results.append(res)
        if not args.no_mirror and res["status"] in ("ok", "already_done"):
            _mirror_to_notebook(f"{dirname}_mars_test_eval")
    overall_elapsed = time.time() - overall_t0

    summary_path = TEST_EVAL_DIR / "v5_test_eval_summary.json"
    summary = {
        "completed_at":  _ts(),
        "elapsed_total_s": round(overall_elapsed, 1),
        "runs": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"[{_ts()}] ============================================================")
    print(f"[{_ts()}]   v5 test eval - final summary")
    print(f"[{_ts()}] ============================================================")
    print(f"  {'Run':<5} {'Status':<14} {'macro_F1':<10} {'beh-only':<10} "
          f"{'elapsed':<10} {'fps':<8} {'GPU%':<8} {'VRAM(GB)':<10}")
    print("  " + "-" * 96)
    for r in results:
        macro = f"{r['macro_f1']:.4f}" if r.get('macro_f1') is not None else "n/a"
        beh = f"{r.get('macro_f1_behavior'):.4f}" if r.get('macro_f1_behavior') is not None else "n/a"
        elapsed = f"{r['elapsed_s']:>6.0f}s"
        fps = r.get("fps_summary", {}).get("fps_mean", "n/a")
        hw = r.get("hw_summary", {})
        gpu = hw.get("gpu_util_pct", {}).get("mean", "n/a")
        vram = hw.get("vram_gb_used", {}).get("peak", "n/a")
        print(f"  {r['short']:<5} {r['status']:<14} {macro:<10} {beh:<10} "
              f"{elapsed:<10} {str(fps):<8} {str(gpu):<8} {str(vram):<10}")
    print()
    print(f"[{_ts()}] total elapsed: {overall_elapsed/60:.1f} min")
    print(f"[{_ts()}] summary saved: {summary_path}")
    if not args.no_mirror:
        try:
            NOTEBOOK_TEST_EVAL_DIR.mkdir(parents=True, exist_ok=True)
            shutil.copy2(summary_path, NOTEBOOK_TEST_EVAL_DIR / "v5_test_eval_summary.json")
            print(f"[{_ts()}] [mirror] v5_test_eval_summary.json", flush=True)
        except Exception as exc:
            print(f"[{_ts()}] [warn]  mirror summary failed: {exc}", flush=True)
    print()
    print("[next step] run scripts/analyze_full_video_test_results.py once all 5 ablations are done")
    print("            (will need to be extended to take a list of arms; current script handles R5).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n[{_ts()}] interrupted; results so far were saved per-run.")
        sys.exit(130)
