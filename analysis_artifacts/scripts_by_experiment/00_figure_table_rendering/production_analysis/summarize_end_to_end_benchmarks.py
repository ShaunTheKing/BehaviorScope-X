#!/usr/bin/env python
"""Summarize portable end-to-end benchmark runs for the V6 supplement."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


PACKAGE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_BENCH_ROOT = (
    PACKAGE_ROOT
    / "verification_files"
    / "by_experiment"
    / "07_deployment_and_gui"
    / "end_to_end_pipeline_benchmark"
    / "outputs"
)
DEFAULT_TABLE_DIR = PACKAGE_ROOT / "tables" / "production" / "supplement"
DEFAULT_INTERNAL_QC_DIR = (
    PACKAGE_ROOT
    / "scripts_by_experiment"
    / "00_figure_table_rendering"
    / "production_analysis"
    / "internal_qc"
    / "end_to_end_benchmark"
)
DEFAULT_FIG_DIR = PACKAGE_ROOT / "figures" / "production" / "supplement"
STYLE_PATH = (
    PACKAGE_ROOT
    / "scripts_by_experiment"
    / "00_figure_table_rendering"
    / "current_analysis"
)

import sys

sys.path.insert(0, str(STYLE_PATH))
from behaviorscope_figure_style import apply_style, save_figure  # noqa: E402


SCENARIOS = {
    "benchmarks_mps": {
        "scenario": "mps_opencv_old_stack",
        "label": "MPS\nOpenCV",
        "hardware": "Apple Silicon",
        "stack": "old_stack",
        "decode_backend": "opencv",
        "color": "#009E73",
        "sort": 1,
    },
    "benchmarks": {
        "scenario": "rtx2080_opencv_current_stack",
        "label": "RTX2080\nOpenCV",
        "hardware": "RTX2080",
        "stack": "current_stack",
        "decode_backend": "opencv",
        "color": "#7C3AED",
        "sort": 2,
    },
    "benchmarks_RTX4080": {
        "scenario": "rtx4080_opencv_old_stack",
        "label": "RTX4080\nOpenCV",
        "hardware": "RTX4080",
        "stack": "old_stack",
        "decode_backend": "opencv",
        "color": "#0072B2",
        "sort": 3,
    },
    "benchmarks_RTX4080_new_stack": {
        "scenario": "rtx4080_ffmpeg_new_stack",
        "label": "RTX4080\nFFmpeg\ncache",
        "hardware": "RTX4080",
        "stack": "new_stack",
        "decode_backend": "ffmpeg-predecode",
        "color": "#D55E00",
        "sort": 4,
    },
}


def as_float(value: object) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def mean(values: list[float]) -> float:
    vals = [v for v in values if not math.isnan(v)]
    return sum(vals) / len(vals) if vals else math.nan


def median(values: list[float]) -> float:
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return math.nan
    mid = len(vals) // 2
    if len(vals) % 2:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def fmt(value: float, digits: int = 4) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench_root", type=Path, default=DEFAULT_BENCH_ROOT)
    parser.add_argument("--table_dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--internal_qc_dir", type=Path, default=DEFAULT_INTERNAL_QC_DIR)
    parser.add_argument("--figure_dir", type=Path, default=DEFAULT_FIG_DIR)
    return parser.parse_args()


def read_rows(bench_root: Path) -> list[dict]:
    rows: list[dict] = []
    for folder, meta in SCENARIOS.items():
        path = bench_root / folder / "benchmark_summary.csv"
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out = dict(row)
                out.update({k: v for k, v in meta.items() if k != "color"})
                try:
                    out["source_summary"] = str(path.relative_to(PACKAGE_ROOT))
                except ValueError:
                    out["source_summary"] = str(path)
                out["decode_backend_observed"] = row.get("decode_backend") or "opencv_or_legacy_blank"
                rows.append(out)
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["scenario"]].append(row)

    summary = []
    for scenario, group in grouped.items():
        frames = sum(as_float(r.get("frames")) for r in group)
        wall = sum(as_float(r.get("wall_time_s")) for r in group)
        predecode_by_video = {}
        for r in group:
            pre = as_float(r.get("predecode_wall_time_s"))
            if not math.isnan(pre):
                predecode_by_video[r.get("video_id", "")] = pre
        unique_predecode = sum(predecode_by_video.values())
        total_with_predecode = wall + unique_predecode
        fps_values = [as_float(r.get("end_to_end_fps")) for r in group]
        summary.append({
            "scenario": scenario,
            "label": group[0]["label"].replace("\n", " "),
            "hardware": group[0]["hardware"],
            "stack": group[0]["stack"],
            "decode_backend": group[0]["decode_backend"],
            "observed_decode_backend": ";".join(sorted({r.get("decode_backend_observed", "") for r in group})),
            "n_model_video_runs": len(group),
            "n_models": len({r.get("model", "") for r in group}),
            "n_videos": len({r.get("video_id", "") for r in group}),
            "total_frames": int(frames),
            "total_inference_wall_s": fmt(wall, 3),
            "unique_predecode_wall_s": fmt(unique_predecode, 3),
            "aggregate_fps_inference_only": fmt(frames / wall, 4),
            "aggregate_fps_with_unique_predecode": fmt(frames / total_with_predecode, 4),
            "mean_run_fps": fmt(mean(fps_values), 4),
            "median_run_fps": fmt(median(fps_values), 4),
            "mean_instant_fps": fmt(mean([as_float(r.get("mean_instant_fps")) for r in group]), 4),
            "median_instant_fps": fmt(mean([as_float(r.get("median_instant_fps")) for r in group]), 4),
            "mean_cpu_percent_normalized": fmt(mean([as_float(r.get("mean_cpu_percent_normalized")) for r in group]), 4),
            "mean_gpu_util_percent": fmt(mean([as_float(r.get("mean_gpu_util_percent")) for r in group]), 4),
            "mean_peak_ram_mb": fmt(mean([as_float(r.get("max_cpu_rss_mb")) for r in group]), 2),
            "mean_peak_gpu_mem_mb": fmt(mean([as_float(r.get("max_gpu_mem_used_mb")) for r in group]), 2),
            "mean_pose_crop_next_share_pct": fmt(mean([as_float(r.get("mean_pose_crop_next_share_pct")) for r in group]), 4),
            "mean_feature_encode_share_pct": fmt(mean([as_float(r.get("mean_feature_encode_share_pct")) for r in group]), 4),
            "mean_window_assembly_share_pct": fmt(mean([as_float(r.get("mean_window_assembly_share_pct")) for r in group]), 4),
            "mean_classify_share_pct": fmt(mean([as_float(r.get("mean_classify_share_pct")) for r in group]), 4),
        })
    return sorted(summary, key=lambda r: SCENARIOS[next(k for k, v in SCENARIOS.items() if v["scenario"] == r["scenario"])]["sort"])


def pairwise(rows: list[dict]) -> list[dict]:
    by_key = {(r["scenario"], r.get("model", ""), r.get("video_id", "")): r for r in rows}
    comparisons = [
        ("mps_vs_rtx4080_opencv_old_stack", "mps_opencv_old_stack", "rtx4080_opencv_old_stack"),
        ("mps_vs_rtx2080_opencv", "mps_opencv_old_stack", "rtx2080_opencv_current_stack"),
        ("rtx4080_opencv_vs_rtx2080_opencv", "rtx4080_opencv_old_stack", "rtx2080_opencv_current_stack"),
        ("rtx4080_new_ffmpeg_vs_old_opencv", "rtx4080_ffmpeg_new_stack", "rtx4080_opencv_old_stack"),
    ]
    out = []
    keys = sorted({(r.get("model", ""), r.get("video_id", "")) for r in rows})
    for name, lhs, rhs in comparisons:
        for model, video in keys:
            a = by_key.get((lhs, model, video))
            b = by_key.get((rhs, model, video))
            if not a or not b:
                continue
            fps_a = as_float(a.get("end_to_end_fps"))
            fps_b = as_float(b.get("end_to_end_fps"))
            out.append({
                "comparison": name,
                "model": model,
                "video_id": video,
                "lhs_scenario": lhs,
                "rhs_scenario": rhs,
                "lhs_fps": fmt(fps_a, 4),
                "rhs_fps": fmt(fps_b, 4),
                "fps_delta": fmt(fps_a - fps_b, 4),
                "fps_ratio": fmt(fps_a / fps_b if fps_b else math.nan, 4),
                "pct_change": fmt(((fps_a - fps_b) / fps_b * 100.0) if fps_b else math.nan, 2),
            })
    return out


def stage_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        stage_values = [
            as_float(r.get("mean_pose_crop_next_share_pct")),
            as_float(r.get("mean_feature_encode_share_pct")),
            as_float(r.get("mean_window_assembly_share_pct")),
            as_float(r.get("mean_classify_share_pct")),
        ]
        if all(math.isnan(v) for v in stage_values):
            continue
        out.append({
            "scenario": r["scenario"],
            "model": r.get("model", ""),
            "video_id": r.get("video_id", ""),
            "pose_crop_next_share_pct": r.get("mean_pose_crop_next_share_pct", ""),
            "feature_encode_share_pct": r.get("mean_feature_encode_share_pct", ""),
            "window_assembly_share_pct": r.get("mean_window_assembly_share_pct", ""),
            "classify_share_pct": r.get("mean_classify_share_pct", ""),
            "csv_write_share_pct": r.get("mean_csv_write_share_pct", ""),
            "pose_crop_next_ms": r.get("mean_pose_crop_next_ms", ""),
            "feature_encode_ms": r.get("mean_feature_encode_ms", ""),
            "window_assembly_ms": r.get("mean_window_assembly_ms", ""),
            "classify_ms": r.get("mean_classify_ms", ""),
            "csv_write_ms": r.get("mean_csv_write_ms", ""),
        })
    return out


def plot_throughput(rows: list[dict], summary: list[dict], figure_dir: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [1.18, 1.0]})
    scenarios = [r["scenario"] for r in summary]
    labels = [SCENARIOS[k]["label"] for k in SCENARIOS if SCENARIOS[k]["scenario"] in scenarios]
    colors = [SCENARIOS[k]["color"] for k in SCENARIOS if SCENARIOS[k]["scenario"] in scenarios]
    scenario_to_summary = {r["scenario"]: r for r in summary}
    ordered = [SCENARIOS[k]["scenario"] for k in SCENARIOS if SCENARIOS[k]["scenario"] in scenario_to_summary]
    x = list(range(len(ordered)))
    agg = [as_float(scenario_to_summary[s]["aggregate_fps_inference_only"]) for s in ordered]
    axes[0].bar(x, agg, color=colors, edgecolor="#111827", linewidth=0.6)
    for xi, value in zip(x, agg):
        axes[0].text(xi, value + 1.1, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    axes[0].axhline(30, color="#6B7280", linestyle="--", linewidth=1.0)
    axes[0].text(len(x) - 0.45, 30.8, "30 fps", color="#374151", fontsize=8, ha="right")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Aggregate throughput (frames/s)")
    axes[0].set_title("(A) End-to-end inference throughput", loc="left")
    axes[0].set_ylim(0, max(agg) * 1.28)

    jitter = {"full_lstm256": -0.08, "full_attention256": 0.08}
    markers = {"full_lstm256": "o", "full_attention256": "s"}
    for i, scenario in enumerate(ordered):
        group = [r for r in rows if r["scenario"] == scenario]
        for r in group:
            model = r.get("model", "")
            axes[0].scatter(
                i + jitter.get(model, 0.0),
                as_float(r.get("end_to_end_fps")),
                marker=markers.get(model, "o"),
                s=24,
                facecolor="white",
                edgecolor="#111827",
                linewidth=0.6,
                zorder=3,
            )

    rtx_old = [r for r in rows if r["scenario"] == "rtx4080_opencv_old_stack"]
    rtx_new = [r for r in rows if r["scenario"] == "rtx4080_ffmpeg_new_stack"]
    pairs = []
    by_new = {(r.get("model", ""), r.get("video_id", "")): r for r in rtx_new}
    for old in rtx_old:
        key = (old.get("model", ""), old.get("video_id", ""))
        if key in by_new:
            pairs.append((old, by_new[key]))
    axes[1].axhline(0, color="#6B7280", linewidth=1.0)
    for old, new in pairs:
        model = old.get("model", "")
        delta = as_float(new.get("end_to_end_fps")) - as_float(old.get("end_to_end_fps"))
        axes[1].scatter(
            0 if model == "full_lstm256" else 1,
            delta,
            marker=markers.get(model, "o"),
            s=36,
            color="#D55E00",
            edgecolor="#111827",
            linewidth=0.5,
        )
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["LSTM-256", "Attention-256"])
    axes[1].set_ylabel("FPS change vs RTX4080 OpenCV")
    axes[1].set_title("(B) FFmpeg-cache diagnostic on RTX4080", loc="left")
    axes[1].set_ylim(-1.05, 0.12)
    axes[1].text(0.98, 0.94, "no throughput gain", transform=axes[1].transAxes,
                 ha="right", va="top", fontsize=8, color="#374151")
    save_figure(fig, figure_dir / "supp_end_to_end_benchmark_throughput")
    plt.close(fig)


def plot_stage(summary: list[dict], stages: list[dict], figure_dir: Path) -> None:
    if not stages:
        return
    apply_style()
    stage_defs = [
        ("pose_crop_next_share_pct", "Pose/tracking/crop", "#0072B2"),
        ("feature_encode_share_pct", "YOLO/SPPF features", "#009E73"),
        ("window_assembly_share_pct", "Window assembly", "#F0E442"),
        ("classify_share_pct", "Temporal classifier", "#D55E00"),
    ]
    scenario_order = [
        SCENARIOS[k]["scenario"]
        for k in SCENARIOS
        if any(r["scenario"] == SCENARIOS[k]["scenario"] for r in stages)
    ]
    labels = [
        SCENARIOS[k]["label"]
        for k in SCENARIOS
        if SCENARIOS[k]["scenario"] in scenario_order
    ]
    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    for yi, scenario in enumerate(scenario_order):
        group = [r for r in stages if r["scenario"] == scenario]
        left = 0.0
        for col, label, color in stage_defs:
            value = mean([as_float(r.get(col)) for r in group])
            ax.barh([yi], [value], left=left, color=color, edgecolor="#111827", linewidth=0.5, label=label if yi == 0 else None)
            if value >= 4:
                ax.text(left + value / 2, yi, f"{value:.1f}%", ha="center", va="center", fontsize=8)
            left += value
    ax.set_xlim(0, 100)
    ax.set_yticks(list(range(len(scenario_order))))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Mean share of interval wall time (%)")
    ax.set_title("Stage-timed diagnostic localizes runtime upstream of classification", loc="left", fontsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2, frameon=False)
    save_figure(fig, figure_dir / "supp_end_to_end_benchmark_stage_timing")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    rows = read_rows(args.bench_root)
    if not rows:
        raise SystemExit(f"No benchmark summaries found under {args.bench_root}.")

    run_fields = [
        "scenario", "label", "hardware", "stack", "decode_backend", "decode_backend_observed",
        "host", "platform", "device", "amp_requested", "amp_used", "model", "repeat", "split",
        "video_id", "frames", "duration_s", "wall_time_s", "predecode_wall_time_s",
        "end_to_end_fps", "mean_instant_fps", "median_instant_fps", "mean_cpu_percent_normalized",
        "mean_gpu_util_percent", "max_cpu_rss_mb", "max_gpu_mem_used_mb",
        "mean_pose_crop_next_ms", "mean_feature_encode_ms", "mean_window_assembly_ms",
        "mean_classify_ms", "mean_pose_crop_next_share_pct", "mean_feature_encode_share_pct",
        "mean_window_assembly_share_pct", "mean_classify_share_pct", "source_summary",
    ]
    summary = summarize(rows)
    pairs = pairwise(rows)
    stages = stage_rows(rows)

    write_csv(args.internal_qc_dir / "end_to_end_benchmark_run_level.csv", rows, run_fields)
    write_csv(args.internal_qc_dir / "end_to_end_benchmark_summary.csv", summary, list(summary[0].keys()))
    write_csv(args.internal_qc_dir / "end_to_end_benchmark_pairwise_video_delta.csv", pairs, list(pairs[0].keys()))
    write_csv(args.internal_qc_dir / "end_to_end_benchmark_stage_timing.csv", stages, list(stages[0].keys()))
    write_csv(args.table_dir / "supp_end_to_end_benchmark_run_level.csv", rows, run_fields)
    write_csv(args.table_dir / "supp_end_to_end_benchmark_summary.csv", summary, list(summary[0].keys()))
    write_csv(args.table_dir / "supp_end_to_end_benchmark_pairwise_video_delta.csv", pairs, list(pairs[0].keys()))
    write_csv(args.table_dir / "supp_end_to_end_benchmark_stage_timing.csv", stages, list(stages[0].keys()))

    plot_throughput(rows, summary, args.figure_dir)
    plot_stage(summary, stages, args.figure_dir)

    for path in [
        args.internal_qc_dir / "end_to_end_benchmark_run_level.csv",
        args.internal_qc_dir / "end_to_end_benchmark_summary.csv",
        args.internal_qc_dir / "end_to_end_benchmark_pairwise_video_delta.csv",
        args.internal_qc_dir / "end_to_end_benchmark_stage_timing.csv",
        args.figure_dir / "supp_end_to_end_benchmark_throughput.png",
        args.figure_dir / "supp_end_to_end_benchmark_throughput.pdf",
        args.figure_dir / "supp_end_to_end_benchmark_stage_timing.png",
        args.figure_dir / "supp_end_to_end_benchmark_stage_timing.pdf",
    ]:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
