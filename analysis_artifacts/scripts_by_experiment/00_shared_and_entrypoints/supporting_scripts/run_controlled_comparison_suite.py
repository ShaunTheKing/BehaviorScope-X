#!/usr/bin/env python3
"""Orchestrate cached-feature model-family comparison runs.

This wrapper is intentionally explicit. It builds one frozen command plan for:

1. MARS annotation provenance audit.
2. Controlled RF/XGBoost matrices with train-only PCA.
3. RF/XGBoost training across configured seeds.
4. RF/XGBoost human-GT-only held-out evaluation across seeds.
5. MARS neural-head training/evaluation across the prespecified ablation matrix.
6. MobileNetV3-native portability training/evaluation.
7. Optional Fly-v-Fly short-bout training/evaluation with annotation-uncertainty modes.
8. Suite-level metric summaries, including mean/SD across seeds.

Use `plan` first; use `run --stage ...` only after reviewing the plan.
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
MAIN_CONFIG = ROOT / "config.yaml"
CLASSICAL_CONFIG = ROOT / "pose_baseline_rf" / "config.yaml"
RUNS_ROOT = ROOT / "outputs" / "controlled_comparison_runs"
CLASSICAL_FEATURE_SETS = [
    "pose_frame",
    "pose_window",
    "pose_visual_frame_pca",
    "pose_visual_window_pca",
]
DEFAULT_NEURAL_HEADS = ["lstm", "attention", "tcn"]
MOBILENET_BACKEND_NAME = "mobilenetv3_native"
MOBILENET_VISUAL_DIM = 960
MOBILENET_RAW_VISUAL_DIM = 3 * MOBILENET_VISUAL_DIM
FULL_MANUSCRIPT_SUITE = "full_manuscript"
MARS_CONTROLLED_SUITE = "mars_controlled"
FLY_GT_MODES = ["primary", "secondary", "union", "intersection", "padded_primary"]


@dataclass
class PlannedCommand:
    stage: str
    name: str
    command: list[str]
    expected_outputs: list[str]
    reason: str


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_path(path: str | Path, *, root: Path = ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (root / p).resolve()


def main_profile(cfg: dict) -> dict:
    profile_name = cfg.get("active_path_profile")
    return cfg["path_profiles"][profile_name]


def resolve_profile_path(cfg: dict, value: str | Path) -> Path:
    profile = main_profile(cfg)
    raw = str(value)
    if raw in profile:
        raw = str(profile[raw])
    first, sep, rest = raw.partition("/")
    if sep and first in profile:
        raw = str(profile[first]) + "/" + rest
    return resolve_path(raw)


def validate_math(classical_cfg: dict, tcn_layers: int, tcn_kernel_size: int) -> dict:
    features = classical_cfg["features"]
    n_animals = int(features["n_animals"])
    n_keypoints = int(features["n_keypoints"])
    relation_dim = int(features["relation_dim"])
    pca_dim = int(features["visual_pca_dim_per_frame"])
    pose_self = 4 * n_keypoints + n_keypoints * (n_keypoints - 1) // 2
    pose_relation = n_animals * (pose_self + 4) + n_animals * (n_animals - 1) * (relation_dim + 2)
    pose_visual_pca = pose_relation + pca_dim
    # Each TCN residual block in shared_scripts/scripts/model_y.py contains
    # two causal Conv1d layers at the same dilation. The receptive-field
    # contribution is therefore doubled relative to the common single-conv
    # shorthand.
    tcn_convs_per_block = 2
    tcn_rf_single_conv_shorthand = 1 + (int(tcn_kernel_size) - 1) * ((2 ** int(tcn_layers)) - 1)
    tcn_rf = 1 + (
        tcn_convs_per_block
        * (int(tcn_kernel_size) - 1)
        * ((2 ** int(tcn_layers)) - 1)
    )
    expected_pose = int(features["expected_pose_relation_dim_per_frame"])
    expected_full = int(features["expected_pose_visual_pca_dim_per_frame"])
    if pose_relation != expected_pose:
        raise SystemExit(f"Pose/relation dim mismatch: computed={pose_relation} config={expected_pose}")
    if pose_visual_pca != expected_full:
        raise SystemExit(f"Pose+PCA dim mismatch: computed={pose_visual_pca} config={expected_full}")
    return {
        "n_animals": n_animals,
        "n_keypoints": n_keypoints,
        "pose_self_per_animal": pose_self,
        "pose_relation_dim_per_frame": pose_relation,
        "pose_visual_pca_dim_per_frame": pose_visual_pca,
        "mars_window": 32,
        "mars_pose_window_dim": 32 * pose_relation,
        "mars_pose_visual_pca_window_dim": 32 * pose_visual_pca,
        "tcn_layers": int(tcn_layers),
        "tcn_kernel_size": int(tcn_kernel_size),
        "tcn_convs_per_block": int(tcn_convs_per_block),
        "tcn_single_conv_shorthand_receptive_field": tcn_rf_single_conv_shorthand,
        "tcn_nominal_receptive_field": tcn_rf,
        "tcn_effective_mars_context_frames": min(tcn_rf, 32),
    }


def make_backend_classical_config(
    base_cfg: dict,
    *,
    output_root_path: Path,
    dataset_name: str,
    train_feature_cache: Path,
    heldout_feature_cache: Path,
    raw_visual_dim_per_frame: int,
    visual_backend: str,
) -> dict:
    cfg = dict(base_cfg)
    cfg["project"] = dict(base_cfg.get("project", {}))
    cfg["project"]["output_root"] = str(output_root_path)
    cfg["paths"] = dict(base_cfg.get("paths", {}))
    cfg["paths"]["train_feature_cache"] = str(train_feature_cache)
    cfg["paths"]["heldout_feature_cache"] = str(heldout_feature_cache)
    cfg["features"] = dict(base_cfg.get("features", {}))
    cfg["features"]["dataset"] = str(dataset_name)
    cfg["features"]["visual_raw_dim_per_frame"] = int(raw_visual_dim_per_frame)
    cfg["features"]["visual_backend"] = str(visual_backend)
    return cfg


def model_flags(model_id: str) -> list[str]:
    flags: list[str] = []
    if model_id.startswith("animal_visual_only_"):
        flags += ["--disable_group_rgb", "--disable_pose_self", "--disable_relations"]
    elif model_id.startswith("pose_relations_only_"):
        flags += ["--disable_visual_streams"]
    elif model_id.startswith("visual_only_"):
        flags += ["--disable_pose_self", "--disable_relations"]
    elif model_id.startswith("no_group_visual_"):
        flags += ["--disable_group_rgb"]
    elif model_id.startswith("no_relations_"):
        flags += ["--disable_relations"]
    elif model_id.startswith("full_") or model_id.startswith("clip_full_"):
        pass
    else:
        raise SystemExit(f"No model flag mapping for {model_id}")
    return flags


def neural_run_name(model_id: str, head: str, seed: int) -> str:
    if "_lstm" in model_id:
        base = model_id.replace("_lstm", f"_{head}")
    else:
        base = f"{model_id}_{head}"
    return f"{base}_seed{seed}"


def neural_backend_run_name(model_id: str, head: str, seed: int, backend: str) -> str:
    return f"{backend}_{neural_run_name(model_id, head, seed)}"


def append_amp(cmd: list[str], cfg: dict, device: str) -> None:
    env = cfg.get("environment", {})
    if bool(env.get("amp", True)) and str(device).startswith("cuda"):
        cmd.append("--amp")
        cmd.extend(["--amp_dtype", str(env.get("amp_dtype", "auto"))])


def capacity_model_ids(prefix: str, hidden_dims: list[int] | tuple[int, ...]) -> list[str]:
    return [f"{prefix}_lstm{int(h)}" for h in hidden_dims]


def apply_suite_preset(args: argparse.Namespace) -> None:
    """Convert a named scientific suite into explicit CLI settings.

    The preset writes ordinary argparse fields so the frozen config records the
    actual matrix instead of hiding behavior behind a magic branch.
    """
    suite = str(getattr(args, "suite", MARS_CONTROLLED_SUITE) or MARS_CONTROLLED_SUITE)
    if suite == MARS_CONTROLLED_SUITE:
        return
    if suite != FULL_MANUSCRIPT_SUITE:
        raise SystemExit(f"Unsupported suite preset: {suite}")

    args.neural_heads = ["lstm", "attention", "tcn"]
    args.lstm_scope = "previous"
    args.all_heads_on_previous_lstm_models = True
    args.mobilenet_neural_heads = ["lstm", "attention", "tcn"]
    if args.mobilenet_model_ids is None:
        args.mobilenet_model_ids = capacity_model_ids("full", [256, 512, 896])
    args.include_fly = True
    args.fly_heads = ["lstm", "attention", "tcn"]
    args.fly_hidden_dims = [256, 512, 896]
    args.fly_window_recipes = ["w16s8", "w8s4"]


def apply_head_exclusions(args: argparse.Namespace) -> None:
    """Apply explicit head-family exclusions after suite preset expansion."""
    if bool(getattr(args, "focused_tcn_control", False)):
        args.neural_heads = ["lstm", "attention", "tcn"]
        args.mobilenet_neural_heads = [h for h in [str(x).lower() for x in args.mobilenet_neural_heads] if h != "tcn"]
        args.fly_heads = [h for h in [str(x).lower() for x in args.fly_heads] if h != "tcn"]
        return
    if not bool(getattr(args, "exclude_tcn", False)):
        return
    for attr in ("neural_heads", "mobilenet_neural_heads", "fly_heads"):
        heads = [str(h).lower() for h in getattr(args, attr, [])]
        setattr(args, attr, [h for h in heads if h != "tcn"])


def command_output_text(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd,
            cwd=str(ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def environment_snapshot() -> dict:
    snap = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "python": sys.version,
        "cpu_count_logical": os.cpu_count(),
        "nvidia_smi_L": command_output_text(["nvidia-smi", "-L"]),
        "nvidia_smi_query_gpu": command_output_text([
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]),
    }
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        snap.update({
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_bytes": int(vm.total),
            "ram_total_gb": round(float(vm.total) / (1024 ** 3), 3),
        })
    except Exception as exc:
        snap["psutil"] = f"unavailable: {exc}"
    return snap


def _query_gpu_rows() -> list[dict]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        rows.append({
            "gpu_index": parts[0],
            "gpu_name": parts[1],
            "gpu_util_pct": parts[2],
            "gpu_mem_util_pct": parts[3],
            "gpu_mem_used_mb": parts[4],
            "gpu_mem_total_mb": parts[5],
            "gpu_power_w": parts[6],
        })
    return rows


def _numeric(value: object) -> float:
    try:
        text = str(value).strip()
        return float(text) if text else 0.0
    except Exception:
        return 0.0


def monitor_resources(pid: int, csv_path: Path, stop_event: threading.Event, interval_s: float) -> None:
    """Sample host and GPU resources while one planned command runs."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "elapsed_s",
        "cpu_percent_total",
        "ram_used_gb",
        "ram_total_gb",
        "proc_tree_rss_gb",
        "gpu_index",
        "gpu_name",
        "gpu_util_pct",
        "gpu_mem_util_pct",
        "gpu_mem_used_mb",
        "gpu_mem_total_mb",
        "gpu_power_w",
    ]
    t0 = time.time()
    try:
        import psutil  # type: ignore
        root_proc = psutil.Process(pid)
        psutil.cpu_percent(interval=None)
    except Exception:
        psutil = None  # type: ignore
        root_proc = None

    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        while not stop_event.is_set():
            elapsed = round(time.time() - t0, 3)
            cpu_percent_total = ""
            ram_used_gb = ""
            ram_total_gb = ""
            proc_tree_rss_gb = ""
            if psutil is not None:
                try:
                    vm = psutil.virtual_memory()
                    cpu_percent_total = psutil.cpu_percent(interval=None)
                    ram_used_gb = round(float(vm.used) / (1024 ** 3), 3)
                    ram_total_gb = round(float(vm.total) / (1024 ** 3), 3)
                    procs = [root_proc] + root_proc.children(recursive=True) if root_proc is not None else []
                    rss = 0
                    for proc in procs:
                        try:
                            rss += int(proc.memory_info().rss)
                        except Exception:
                            pass
                    proc_tree_rss_gb = round(float(rss) / (1024 ** 3), 3)
                except Exception:
                    pass
            gpu_rows = _query_gpu_rows()
            if not gpu_rows:
                writer.writerow({
                    "elapsed_s": elapsed,
                    "cpu_percent_total": cpu_percent_total,
                    "ram_used_gb": ram_used_gb,
                    "ram_total_gb": ram_total_gb,
                    "proc_tree_rss_gb": proc_tree_rss_gb,
                })
            else:
                for gpu in gpu_rows:
                    row = {
                        "elapsed_s": elapsed,
                        "cpu_percent_total": cpu_percent_total,
                        "ram_used_gb": ram_used_gb,
                        "ram_total_gb": ram_total_gb,
                        "proc_tree_rss_gb": proc_tree_rss_gb,
                    }
                    row.update(gpu)
                    writer.writerow(row)
            fh.flush()
            stop_event.wait(max(float(interval_s), 1.0))


def summarize_resource_csv(csv_path: Path) -> dict:
    if not csv_path.is_file() or csv_path.stat().st_size == 0:
        return {}
    rows = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    return {
        "resource_csv": str(csv_path),
        "samples": len(rows),
        "max_cpu_percent_total": max(_numeric(r.get("cpu_percent_total")) for r in rows),
        "max_ram_used_gb": max(_numeric(r.get("ram_used_gb")) for r in rows),
        "max_proc_tree_rss_gb": max(_numeric(r.get("proc_tree_rss_gb")) for r in rows),
        "max_gpu_util_pct": max(_numeric(r.get("gpu_util_pct")) for r in rows),
        "max_gpu_mem_used_mb": max(_numeric(r.get("gpu_mem_used_mb")) for r in rows),
        "max_gpu_power_w": max(_numeric(r.get("gpu_power_w")) for r in rows),
    }


def build_plan(args: argparse.Namespace) -> tuple[list[PlannedCommand], dict]:
    main_cfg = load_yaml(MAIN_CONFIG)
    classical_cfg = load_yaml(CLASSICAL_CONFIG)
    py = str(main_cfg.get("environment", {}).get("python_executable", "python"))
    device = args.device or str(main_cfg.get("environment", {}).get("device_default", "cuda:0"))
    tcn_hidden = int(args.tcn_hidden_dim)
    tcn_layers = int(args.tcn_layers)
    tcn_kernel = int(args.tcn_kernel_size)
    seeds = [int(s) for s in args.seeds]
    if not seeds:
        raise SystemExit("At least one seed is required.")
    neural_heads = [str(h).lower() for h in args.neural_heads]
    unknown_heads = sorted(set(neural_heads) - {"lstm", "attention", "tcn"})
    if unknown_heads:
        raise SystemExit(f"Unsupported neural head(s): {unknown_heads}")
    math_contract = validate_math(classical_cfg, tcn_layers, tcn_kernel)

    recipe = main_cfg["recipes"]["mars_cw15_sqrt_pe"]
    fly_recipe_by_window = {
        "w16s8": main_cfg["recipes"]["fly_full_w16s8"],
        "w8s4": main_cfg["recipes"]["fly_full_w8s4"],
    }
    profile = main_profile(main_cfg)
    mars_yolo = resolve_profile_path(main_cfg, "mars_yolo_weights")
    mars_root = resolve_profile_path(main_cfg, "mars_data_root")
    fly_yolo = resolve_profile_path(main_cfg, "fly_yolo_weights")
    fly_aggression_root = resolve_profile_path(main_cfg, "fly_data_root/Aggression")
    train_manifest = resolve_profile_path(main_cfg, "npz_full_video_root/sequence_manifest.json")
    heldout_manifest = resolve_profile_path(main_cfg, "heldout_eval_npz_root/sequence_manifest.json")
    heldout_cache = resolve_profile_path(main_cfg, "eval_feature_cache_root/mars_stride16")
    train_feature_cache = train_manifest.parent / "yolo_feature_cache"
    suite_root = resolve_path(args.output_root)
    mobilenet_checkpoint = resolve_path(args.mobilenet_checkpoint)
    mobilenet_train_cache = suite_root / "feature_caches" / MOBILENET_BACKEND_NAME / "train"
    mobilenet_heldout_cache = suite_root / "feature_caches" / MOBILENET_BACKEND_NAME / "heldout"
    provenance_root = suite_root / "provenance"
    frozen_classical_config = provenance_root / "pose_baseline_rf_config.yolo_sppf.frozen.yaml"
    frozen_mobilenet_classical_config = provenance_root / f"pose_baseline_rf_config.{MOBILENET_BACKEND_NAME}.frozen.yaml"
    neural_train_project = suite_root / "neural_training"
    fly_npz_root = suite_root / "fly" / "npz_cache"
    fly_heldout_npz_root = suite_root / "fly" / "heldout_npz_cache"
    fly_train_project = suite_root / "fly" / "training"
    fly_eval_root = suite_root / "fly" / "eval"
    classical_out = suite_root / "classical"
    classical_yolo_cfg = make_backend_classical_config(
        classical_cfg,
        output_root_path=classical_out,
        dataset_name="mars_yolo_sppf",
        train_feature_cache=train_feature_cache,
        heldout_feature_cache=heldout_cache,
        raw_visual_dim_per_frame=int(classical_cfg["features"]["visual_raw_dim_per_frame"]),
        visual_backend="yolo_sppf",
    )
    classical_mobilenet_cfg = make_backend_classical_config(
        classical_cfg,
        output_root_path=classical_out,
        dataset_name=f"mars_{MOBILENET_BACKEND_NAME}",
        train_feature_cache=mobilenet_train_cache,
        heldout_feature_cache=mobilenet_heldout_cache,
        raw_visual_dim_per_frame=MOBILENET_RAW_VISUAL_DIM,
        visual_backend=MOBILENET_BACKEND_NAME,
    )
    provenance_root.mkdir(parents=True, exist_ok=True)
    frozen_classical_config.write_text(yaml.safe_dump(classical_yolo_cfg, sort_keys=False), encoding="utf-8")
    frozen_mobilenet_classical_config.write_text(
        yaml.safe_dump(classical_mobilenet_cfg, sort_keys=False), encoding="utf-8"
    )
    previous_lstm_models = list(main_cfg["experiments"]["mars_full_video_canonical"]["models"])
    lstm_model_ids = previous_lstm_models if args.lstm_scope == "previous" else [args.full_model_id]
    neural_runs: list[dict] = []
    for head in neural_heads:
        if head == "lstm":
            model_ids = lstm_model_ids
        elif head == "tcn" and bool(getattr(args, "focused_tcn_control", False)):
            model_ids = ["full_lstm256", "full_lstm512"]
        elif args.all_heads_on_previous_lstm_models:
            model_ids = previous_lstm_models
        else:
            model_ids = [args.full_model_id]
        for model_id in model_ids:
            if model_id not in main_cfg["models"]:
                raise SystemExit(f"Unknown model id for neural run: {model_id}")
            for seed in seeds:
                hidden = int(main_cfg["models"][model_id]["hidden_dim"])
                neural_runs.append({
                    "head": head,
                    "visual_backend": "yolo_sppf",
                    "feature_cache": train_feature_cache,
                    "heldout_feature_cache": heldout_cache,
                    "precomputed_visual_dim": 0,
                    "model_id": model_id,
                    "seed": seed,
                    "hidden_dim": hidden,
                    "name": neural_backend_run_name(model_id, head, seed, "yolo_sppf"),
                })
    if not bool(args.skip_mobilenet_neural):
        for head in [str(h).lower() for h in args.mobilenet_neural_heads]:
            if head not in {"lstm", "attention", "tcn"}:
                raise SystemExit(f"Unsupported MobileNet neural head: {head}")
            for model_id in args.mobilenet_model_ids:
                if model_id not in main_cfg["models"]:
                    raise SystemExit(f"Unknown MobileNet model id: {model_id}")
                for seed in seeds:
                    hidden = int(main_cfg["models"][model_id]["hidden_dim"])
                    neural_runs.append({
                        "head": head,
                        "visual_backend": MOBILENET_BACKEND_NAME,
                        "feature_cache": mobilenet_train_cache,
                        "heldout_feature_cache": mobilenet_heldout_cache,
                        "precomputed_visual_dim": MOBILENET_VISUAL_DIM,
                        "model_id": model_id,
                        "seed": seed,
                        "hidden_dim": hidden,
                        "name": neural_backend_run_name(model_id, head, seed, MOBILENET_BACKEND_NAME),
                    })

    commands: list[PlannedCommand] = []
    commands.append(PlannedCommand(
        stage="audit",
        name="mars_annotation_provenance",
        command=[
            py, str(ROOT / "pose_baseline_rf" / "audit_mars_annotation_provenance.py"),
            "--config", str(frozen_classical_config),
        ],
        expected_outputs=[str(classical_out / "audits" / "mars_annotation_provenance.csv")],
        reason="Separate human-GT held-out videos from pred-named/uncertain annotations before any headline metric.",
    ))
    for cache_split, manifest, out_cache, splits_to_cache in [
        ("train", train_manifest, mobilenet_train_cache, ["train", "val"]),
        ("heldout", heldout_manifest, mobilenet_heldout_cache, ["train", "val"]),
    ]:
        build_cache_cmd = [
            py, str(ROOT / "shared_scripts" / "scripts" / "precompute_mobilenetv3_features_y.py"),
            "--manifest_path", str(manifest),
            "--checkpoint", str(mobilenet_checkpoint),
            "--output_dir", str(out_cache),
            "--splits", *splits_to_cache,
            "--n_animals", "2",
            "--resize_size", str(args.mobilenet_resize_size),
            "--batch", str(args.mobilenet_cache_batch),
            "--num_workers", str(args.mobilenet_cache_num_workers),
            "--device", device,
            "--cache_dtype", str(args.mobilenet_cache_dtype),
        ]
        append_amp(build_cache_cmd, main_cfg, device)
        commands.append(PlannedCommand(
            stage="build_visual_cache",
            name=f"build_{MOBILENET_BACKEND_NAME}_{cache_split}_cache",
            command=build_cache_cmd,
            expected_outputs=[str(out_cache / "cache_manifest.json")],
            reason=(
                "Build native MobileNetV3-large backbone.features.16 visual cache "
                "inside this fresh suite directory; no YOLO/SPPF analogue is constructed."
            ),
        ))
    classical_backends = [
        ("yolo_sppf", "mars_yolo_sppf", frozen_classical_config),
        (MOBILENET_BACKEND_NAME, f"mars_{MOBILENET_BACKEND_NAME}", frozen_mobilenet_classical_config),
    ]
    for backend_name, dataset_name, backend_config in classical_backends:
        commands.append(PlannedCommand(
            stage="matrices",
            name=f"build_controlled_classical_matrices_{backend_name}",
            command=[
                py, str(ROOT / "pose_baseline_rf" / "build_controlled_matrices.py"),
                "--config", str(backend_config),
            ],
            expected_outputs=[str(classical_out / "matrices" / dataset_name / "matrix_build_summary.json")],
            reason=f"Build RF/XGBoost matrices for {backend_name}; visual PCA is fit on training frames only.",
        ))
    for backend_name, dataset_name, backend_config in classical_backends:
        for seed in seeds:
            run_tag = f"seed{seed}"
            commands.append(PlannedCommand(
                stage="train_classical",
                name=f"train_rf_xgb_{backend_name}_{run_tag}",
                command=[
                    py, str(ROOT / "pose_baseline_rf" / "train_controlled_classical.py"),
                    "--config", str(backend_config),
                    "--seed", str(seed),
                    "--run_tag", run_tag,
                ],
                expected_outputs=[str(classical_out / "models" / dataset_name / f"training_summary_{run_tag}.json")],
                reason=f"Train RF and XGBoost seed {seed} on frozen {backend_name} matrices.",
            ))
    for backend_name, dataset_name, backend_config in classical_backends:
        for seed in seeds:
            run_tag = f"seed{seed}"
            for feature_set in CLASSICAL_FEATURE_SETS:
                model_paths = [
                    classical_out / "models" / dataset_name / f"rf_{feature_set}_{run_tag}.pkl",
                    classical_out / "models" / dataset_name / f"xgb_{feature_set}_{run_tag}.pkl",
                ]
                commands.append(PlannedCommand(
                    stage="eval_classical",
                    name=f"eval_rf_xgb_{backend_name}_{feature_set}_{run_tag}",
                    command=[
                        py, str(ROOT / "pose_baseline_rf" / "evaluate_controlled_classical_mars.py"),
                        "--config", str(backend_config),
                        "--feature_set", feature_set,
                        "--model_paths", *[str(p) for p in model_paths],
                        "--no-allow_pred_named_annots",
                    ],
                    expected_outputs=[
                        str(classical_out / "eval" / dataset_name / f"rf_{feature_set}_{run_tag}" / feature_set / "aggregate_summary.json"),
                        str(classical_out / "eval" / dataset_name / f"xgb_{feature_set}_{run_tag}" / feature_set / "aggregate_summary.json"),
                    ],
                    reason=f"Evaluate seed {seed} {backend_name} classical models against human-GT annotations.",
                ))

    for run in neural_runs:
        head = str(run["head"])
        run_name = str(run["name"])
        seed = int(run["seed"])
        train_neural = [
            py, str(ROOT / "shared_scripts" / "scripts" / "train_y.py"),
            "--manifest_path", str(train_manifest),
            "--yolo_weights", str(mars_yolo),
            "--yolo_backbone_end_layer", str(recipe["yolo_backbone_end_layer"]),
            "--use_feature_cache", str(run["feature_cache"]),
            "--allow_feature_cache_without_manifest",
            "--sequence_model", head,
            "--tcn_layers", str(tcn_layers),
            "--tcn_kernel_size", str(tcn_kernel),
            "--hidden_dim", str(run["hidden_dim"]),
            "--positional_encoding", str(recipe["positional_encoding"]),
            "--epochs", str(recipe["epochs"]),
            "--batch", str(recipe["batch"]),
            "--num_workers", str(recipe["num_workers"]),
            "--lr", str(recipe["lr"]),
            "--weight_decay", str(recipe["weight_decay"]),
            "--patience", str(recipe["patience"]),
            "--dropout", str(recipe["dropout"]),
            "--num_lstm_layers", str(recipe["num_lstm_layers"]),
            "--pose_fusion_dim", str(recipe["pose_fusion_dim"]),
            "--pose_fusion_strategy", str(recipe["pose_fusion_strategy"]),
            "--class_weighting", str(recipe["class_weighting"]),
            "--class_weight_clamp", str(recipe["class_weight_clamp"]),
            "--train_sampler", str(recipe["train_sampler"]),
            "--disable_threshold_decoder",
            "--resume",
            "--seed", str(seed),
            "--device", device,
            "--per_class_metrics", "--confusion_matrix",
            "--project", str(neural_train_project),
            "--name", run_name,
        ]
        if int(run.get("precomputed_visual_dim", 0)) > 0:
            train_neural.extend(["--precomputed_visual_dim", str(int(run["precomputed_visual_dim"]))])
        train_neural.extend(model_flags(str(run["model_id"])))
        append_amp(train_neural, main_cfg, device)
        commands.append(PlannedCommand(
            stage="train_neural",
            name=run_name,
            command=train_neural,
            expected_outputs=[
                str(neural_train_project / run_name / "best_model_macro_f1.pt"),
                str(neural_train_project / run_name / "config.json"),
                str(neural_train_project / run_name / "training_complete.json"),
            ],
            reason=(
                f"Train {head} seed {seed} for model template {run['model_id']} "
                f"using the same MARS windows and cached {run['visual_backend']} features."
            ),
        ))

    for run in neural_runs:
        run_name = str(run["name"])
        neural_eval_root = suite_root / "neural_eval" / run_name
        eval_neural = [
            py, str(ROOT / "shared_scripts" / "scripts" / "eval_manifest_classifier_y.py"),
            "--manifest_path", str(heldout_manifest),
            "--feature_cache_dir", str(run["heldout_feature_cache"]),
            "--mars_root", str(mars_root),
            "--splits", *list(classical_cfg["evaluation"]["splits"]),
            "--model_path", str(neural_train_project / run_name / "best_model_macro_f1.pt"),
            "--model_config", str(neural_train_project / run_name / "config.json"),
            "--yolo_weights", str(mars_yolo),
            "--output_root", str(neural_eval_root),
            "--batch", str(main_cfg["global_eval_recipe"].get("batch", 16)),
            "--num_workers", str(main_cfg["global_eval_recipe"].get("num_workers", 0)),
            "--fps", str(main_cfg["global_eval_recipe"]["fps"]),
            "--temporal_smoothing_window", str(main_cfg["global_eval_recipe"]["temporal_smoothing_window"]),
            "--bout_min_duration_frames", str(main_cfg["global_eval_recipe"]["bout_min_duration_frames"]),
            "--disable_threshold_decoder",
            "--device", device,
        ]
        append_amp(eval_neural, main_cfg, device)
        commands.append(PlannedCommand(
            stage="eval_neural",
            name=f"eval_{run_name}",
            command=eval_neural,
            expected_outputs=[str(neural_eval_root / "aggregate_summary.json")],
            reason="Evaluate neural head on the same held-out cached windows and human-GT-only annotation filter.",
        ))
    fly_runs: list[dict] = []
    if bool(args.include_fly):
        fly_heads = [str(h).lower() for h in args.fly_heads]
        unknown_fly_heads = sorted(set(fly_heads) - {"lstm", "attention", "tcn"})
        if unknown_fly_heads:
            raise SystemExit(f"Unsupported Fly-v-Fly head(s): {unknown_fly_heads}")
        fly_windows = [str(w).lower() for w in args.fly_window_recipes]
        unknown_windows = sorted(set(fly_windows) - set(fly_recipe_by_window))
        if unknown_windows:
            raise SystemExit(f"Unsupported Fly-v-Fly window recipe(s): {unknown_windows}")

        for window_key in fly_windows:
            fly_recipe = fly_recipe_by_window[window_key]
            fly_npz_dir = fly_npz_root / window_key
            commands.append(PlannedCommand(
                stage="build_fly_npz",
                name=f"build_fly_npz_{window_key}",
                command=[
                    py, str(ROOT / "shared_scripts" / "fly_to_behaviorscope_npz_full.py"),
                    "--aggression_root", str(fly_aggression_root),
                    "--yolo_weights", str(fly_yolo),
                    "--output_root", str(fly_npz_dir),
                    "--window_size", str(fly_recipe["window_size"]),
                    "--window_stride", str(fly_recipe["window_stride"]),
                    "--crop_size", str(fly_recipe["crop_size"]),
                    "--group_crop_size", str(fly_recipe["crop_size"]),
                    "--animal_scale_factor", str(fly_recipe.get(
                        "animal_scale_factor", main_cfg["fly_eval_recipe"]["animal_scale_factor"]
                    )),
                    "--group_scale_factor", str(fly_recipe.get(
                        "group_scale_factor", main_cfg["fly_eval_recipe"]["group_scale_factor"]
                    )),
                    "--pose_conf_threshold", str(fly_recipe["pose_conf_threshold"]),
                    "--yolo_batch", str(fly_recipe.get("yolo_batch", 16)),
                    "--npz_writers", str(fly_recipe.get("npz_writers", 1)),
                    "--other_subsample", str(fly_recipe["other_subsample"]),
                    "--seed", str(args.fly_npz_seed),
                    "--device", device,
                    "--val_movie_ids", *[str(v) for v in fly_recipe["val_movie_ids"]],
                ],
                expected_outputs=[
                    str(fly_npz_dir / "sequence_manifest.json"),
                    str(fly_npz_dir / "fly_npz_build_summary.json"),
                ],
                reason=(
                    f"Build fresh Fly-v-Fly {window_key} priority-OR training windows "
                    "inside the suite; official test movies remain excluded."
                ),
            ))
            for head in fly_heads:
                for hidden in [int(h) for h in args.fly_hidden_dims]:
                    for seed in seeds:
                        run_name = f"fly_yolo_sppf_full_{head}{hidden}_{window_key}_seed{seed}"
                        fly_runs.append({
                            "name": run_name,
                            "head": head,
                            "hidden_dim": int(hidden),
                            "seed": int(seed),
                            "window_key": window_key,
                            "recipe": fly_recipe,
                            "manifest": fly_npz_dir / "sequence_manifest.json",
                        })

        for run in fly_runs:
            fly_recipe = run["recipe"]
            fly_train = [
                py, str(ROOT / "shared_scripts" / "scripts" / "train_y.py"),
                "--manifest_path", str(run["manifest"]),
                "--yolo_weights", str(fly_yolo),
                "--yolo_backbone_end_layer", str(fly_recipe["yolo_backbone_end_layer"]),
                "--auto_feature_cache",
                "--feature_cache_batch", str(fly_recipe["feature_cache_batch"]),
                "--feature_cache_num_workers", str(fly_recipe["feature_cache_num_workers"]),
                "--sequence_model", str(run["head"]),
                "--tcn_layers", str(tcn_layers),
                "--tcn_kernel_size", str(tcn_kernel),
                "--hidden_dim", str(run["hidden_dim"]),
                "--positional_encoding", str(fly_recipe["positional_encoding"]),
                "--epochs", str(fly_recipe["epochs"]),
                "--batch", str(fly_recipe["batch"]),
                "--num_workers", str(fly_recipe["num_workers"]),
                "--lr", str(fly_recipe["lr"]),
                "--weight_decay", str(fly_recipe["weight_decay"]),
                "--patience", str(fly_recipe["patience"]),
                "--dropout", str(fly_recipe["dropout"]),
                "--num_lstm_layers", str(fly_recipe["num_lstm_layers"]),
                "--pose_fusion_dim", str(fly_recipe["pose_fusion_dim"]),
                "--pose_fusion_strategy", str(fly_recipe["pose_fusion_strategy"]),
                "--class_weighting", str(fly_recipe["class_weighting"]),
                "--class_weight_clamp", str(fly_recipe["class_weight_clamp"]),
                "--train_sampler", str(fly_recipe["train_sampler"]),
                "--disable_threshold_decoder",
                "--resume",
                "--seed", str(run["seed"]),
                "--device", device,
                "--per_class_metrics", "--confusion_matrix",
                "--project", str(fly_train_project),
                "--name", str(run["name"]),
            ]
            append_amp(fly_train, main_cfg, device)
            commands.append(PlannedCommand(
                stage="train_fly",
                name=str(run["name"]),
                command=fly_train,
                expected_outputs=[
                    str(fly_train_project / str(run["name"]) / "best_model_macro_f1.pt"),
                    str(fly_train_project / str(run["name"]) / "config.json"),
                    str(fly_train_project / str(run["name"]) / "training_complete.json"),
                ],
                reason=(
                    f"Train Fly-v-Fly full-stream {run['head']} hidden={run['hidden_dim']} "
                    f"{run['window_key']} seed {run['seed']} on fresh short-bout windows."
                ),
            ))

        fly_eval_stride = int(main_cfg["fly_eval_recipe"]["window_stride"])
        fly_eval_cache_keys: dict[str, dict] = {}
        for window_key in fly_windows:
            fly_recipe = fly_recipe_by_window[window_key]
            eval_key = f"w{int(fly_recipe['window_size'])}s{fly_eval_stride}"
            fly_eval_cache_keys[eval_key] = fly_recipe

        for eval_key, fly_recipe in sorted(fly_eval_cache_keys.items()):
            fly_heldout_dir = fly_heldout_npz_root / eval_key
            commands.append(PlannedCommand(
                stage="build_fly_npz",
                name=f"build_fly_heldout_npz_{eval_key}",
                command=[
                    py, str(ROOT / "shared_scripts" / "fly_to_behaviorscope_npz_full.py"),
                    "--aggression_root", str(fly_aggression_root),
                    "--yolo_weights", str(fly_yolo),
                    "--output_root", str(fly_heldout_dir),
                    "--window_size", str(fly_recipe["window_size"]),
                    "--window_stride", str(fly_eval_stride),
                    "--crop_size", str(fly_recipe["crop_size"]),
                    "--group_crop_size", str(fly_recipe["crop_size"]),
                    "--animal_scale_factor", str(fly_recipe.get(
                        "animal_scale_factor", main_cfg["fly_eval_recipe"]["animal_scale_factor"]
                    )),
                    "--group_scale_factor", str(fly_recipe.get(
                        "group_scale_factor", main_cfg["fly_eval_recipe"]["group_scale_factor"]
                    )),
                    "--pose_conf_threshold", str(main_cfg["fly_eval_recipe"]["pose_conf_threshold"]),
                    "--yolo_batch", str(main_cfg["fly_eval_recipe"]["yolo_batch"]),
                    "--npz_writers", str(fly_recipe.get("npz_writers", 1)),
                    "--other_subsample", "1.0",
                    "--seed", str(args.fly_npz_seed),
                    "--device", device,
                    "--val_movie_ids", *[str(v) for v in fly_recipe["val_movie_ids"]],
                    "--only_test",
                    "--test_split_name", "test",
                    "--skip_existing",
                ],
                expected_outputs=[
                    str(fly_heldout_dir / "sequence_manifest.json"),
                    str(fly_heldout_dir / "fly_npz_build_summary.json"),
                ],
                reason=(
                    f"Build Fly-v-Fly held-out test windows for cached classifier-only "
                    f"evaluation ({eval_key}); this is the MARS-style held-out path."
                ),
            ))
            commands.append(PlannedCommand(
                stage="build_fly_npz",
                name=f"cache_fly_heldout_visual_features_{eval_key}",
                command=[
                    py, str(ROOT / "shared_scripts" / "scripts" / "precompute_visual_features_y.py"),
                    "--manifest_path", str(fly_heldout_dir / "sequence_manifest.json"),
                    "--yolo_weights", str(fly_yolo),
                    "--output_dir", str(fly_heldout_dir / "yolo_feature_cache"),
                    "--splits", "test",
                    "--yolo_backbone_end_layer", str(fly_recipe["yolo_backbone_end_layer"]),
                    "--n_animals", "2",
                    "--batch", str(fly_recipe.get("feature_cache_batch", 16)),
                    "--num_workers", str(fly_recipe.get("feature_cache_num_workers", 0)),
                    "--device", device,
                ],
                expected_outputs=[
                    str(fly_heldout_dir / "yolo_feature_cache" / "cache_manifest.json"),
                ],
                reason=(
                    f"Precompute frozen YOLO/SPPF visual features once for Fly-v-Fly "
                    f"held-out {eval_key}, so each temporal model reuses the same cache."
                ),
            ))
            append_amp(commands[-1].command, main_cfg, device)

        fly_eval = [
            py, str(ROOT / "shared_scripts" / "fly_run_cached_eval.py"),
            "--training_dir", str(fly_train_project),
            "--heldout_npz_root", str(fly_heldout_npz_root),
            "--output_dir", str(fly_eval_root),
            "--aggression_root", str(fly_aggression_root),
            "--yolo_weights", str(fly_yolo),
            "--device", device,
            "--fps", str(main_cfg["fly_eval_recipe"]["fps"]),
            "--window_stride", str(fly_eval_stride),
            "--temporal_smoothing_window", str(main_cfg["fly_eval_recipe"]["temporal_smoothing_window"]),
            "--bout_min_duration_frames", str(main_cfg["fly_eval_recipe"]["bout_min_duration_frames"]),
            "--gt_modes", *FLY_GT_MODES,
            "--runs", *[str(run["name"]) for run in fly_runs],
        ]
        if str(device).startswith("cuda"):
            fly_eval.append("--amp")
        commands.append(PlannedCommand(
            stage="eval_fly",
            name="eval_fly_short_bout_uncertainty",
            command=fly_eval,
            expected_outputs=[
                str(fly_eval_root / "cross_run_summary.csv"),
                str(fly_eval_root / "eval_summary.json"),
            ],
            reason=(
                "Evaluate all freshly trained Fly-v-Fly heads from held-out NPZ and "
                "YOLO/SPPF feature caches, preserving primary/secondary/union/"
                "intersection/padded GT modes and avoiding per-model YOLO reruns."
            ),
        ))

    neural_names = [str(run["name"]) for run in neural_runs]
    commands.append(PlannedCommand(
        stage="summarize",
        name="summarize_ground_truth_evaluations",
        command=[
            py, str(ROOT / "summarize_controlled_comparison_results.py"),
            "--suite_root", str(suite_root),
            "--classical_root", str(classical_out),
            "--neural_names", *neural_names,
        ],
        expected_outputs=[
            str(suite_root / "summaries" / "controlled_metrics_summary.csv"),
            str(suite_root / "summaries" / "controlled_metrics_summary_smoothed_only.csv"),
            str(suite_root / "summaries" / "controlled_seed_summary_smoothed_only.csv"),
        ],
        reason="Collect all RF/XGBoost/neural-head ground-truth evaluation metrics into one suite-level MARS summary.",
    ))

    frozen = {
        "created_unix_s": time.time(),
        "root": str(ROOT),
        "main_config": str(MAIN_CONFIG),
        "classical_config": str(CLASSICAL_CONFIG),
        "frozen_classical_config": str(frozen_classical_config),
        "frozen_mobilenet_classical_config": str(frozen_mobilenet_classical_config),
        "path_profile": main_cfg.get("active_path_profile"),
        "python": py,
        "device": device,
        "suite_output_root": str(suite_root),
        "suite": str(args.suite),
        "frozen_suite_config": str(suite_root / "controlled_suite_config.frozen.json"),
        "classical_output_root": str(classical_out),
        "seeds": seeds,
        "neural_heads": neural_heads,
        "lstm_scope": args.lstm_scope,
        "all_heads_on_previous_lstm_models": bool(args.all_heads_on_previous_lstm_models),
        "full_model_id": args.full_model_id,
        "neural_run_count": len(neural_runs),
        "neural_names": [str(run["name"]) for run in neural_runs],
        "fly_enabled": bool(args.include_fly),
        "fly_run_count": len(fly_runs),
        "fly_names": [str(run["name"]) for run in fly_runs],
        "fly_eval_cache_root": str(fly_heldout_npz_root),
        "human_gt_only": True,
        "allow_pred_named_annots": False,
        "prespecified_decoding": {
            "window_prediction_rule": "argmax_over_class_probabilities",
            "threshold_decoder": "disabled for controlled model-family comparisons",
            "temporal_smoothing_window": int(main_cfg["global_eval_recipe"]["temporal_smoothing_window"]),
            "bout_min_duration_frames": int(main_cfg["global_eval_recipe"]["bout_min_duration_frames"]),
            "raw_window_eval": "overlapping-window probability average then argmax",
            "smoothed_frame_eval": "overlapping-window probability average, argmax, median smoothing, minimum-duration filtering",
        },
        "prespecified_model_matrix": {
            "visual_backends": ["yolo_sppf", MOBILENET_BACKEND_NAME],
            "classical_feature_sets": CLASSICAL_FEATURE_SETS,
            "classical_models": ["rf", "xgb"],
            "neural_heads": neural_heads,
            "mobilenet_neural_heads": [str(h).lower() for h in args.mobilenet_neural_heads],
            "mobilenet_model_ids": [str(m) for m in args.mobilenet_model_ids],
            "mobilenet_neural_deferred": bool(args.skip_mobilenet_neural),
            "neural_templates": sorted({str(run["model_id"]) for run in neural_runs}),
            "fly_heads": [str(h).lower() for h in args.fly_heads],
            "fly_hidden_dims": [int(h) for h in args.fly_hidden_dims],
            "fly_window_recipes": [str(w) for w in args.fly_window_recipes],
            "fly_gt_modes": FLY_GT_MODES,
            "fly_cached_heldout_eval": bool(args.include_fly),
            "fly_eval_window_stride": int(main_cfg["fly_eval_recipe"]["window_stride"]),
            "seeds": seeds,
        },
        "visual_backend_contracts": {
            "yolo_sppf": {
                "role": "primary controlled decoder-comparison backend",
                "pose_model_family": "YOLO-pose",
                "pose_model_size": "nano",
                "feature_layer": f"layers[0:{recipe['yolo_backbone_end_layer']}] through SPPF",
                "pooling": "adaptive_avg_pool2d",
                "feature_dim_per_crop": 256,
                "raw_visual_dim_per_frame": int(classical_yolo_cfg["features"]["visual_raw_dim_per_frame"]),
                "train_cache": str(train_feature_cache),
                "heldout_cache": str(heldout_cache),
            },
            MOBILENET_BACKEND_NAME: {
                "role": "architecture-portability/sensitivity arm, not size-matched to YOLO nano",
                "pose_model_family": "BehaviorScopeZ",
                "pose_model_backbone": "mobilenet_v3_large",
                "pose_model_size": "large",
                "feature_layer": "backbone.features.16",
                "pooling": "adaptive_avg_pool2d",
                "normalization": "imagenet",
                "resize_size": int(args.mobilenet_resize_size),
                "feature_dim_per_crop": MOBILENET_VISUAL_DIM,
                "raw_visual_dim_per_frame": MOBILENET_RAW_VISUAL_DIM,
                "checkpoint": str(mobilenet_checkpoint),
                "train_cache": str(mobilenet_train_cache),
                "heldout_cache": str(mobilenet_heldout_cache),
            },
        },
        "resource_telemetry": {
            "enabled": True,
            "sample_interval_s_default": 30.0,
            "per_stage_csv": "logs/resources/<step>.resources.csv",
            "summary_location": "logs/run_summary.json",
        },
        "environment_snapshot": environment_snapshot(),
        "math_contract": math_contract,
        "paths": {
            "mars_yolo_weights": str(mars_yolo),
            "mars_root": str(mars_root),
            "train_manifest": str(train_manifest),
            "train_feature_cache": str(train_feature_cache),
            "heldout_manifest": str(heldout_manifest),
            "heldout_feature_cache": str(heldout_cache),
            "mobilenet_checkpoint": str(mobilenet_checkpoint),
            "mobilenet_train_feature_cache": str(mobilenet_train_cache),
            "mobilenet_heldout_feature_cache": str(mobilenet_heldout_cache),
            "fly_yolo_weights": str(fly_yolo),
            "fly_aggression_root": str(fly_aggression_root),
            "fly_npz_root": str(fly_npz_root),
            "fly_heldout_npz_root": str(fly_heldout_npz_root),
            "fly_training_root": str(fly_train_project),
            "fly_eval_root": str(fly_eval_root),
            "profile": profile,
        },
    }
    return commands, frozen


def write_plan(commands: list[PlannedCommand], frozen: dict, out_root: Path) -> None:
    provenance = out_root / "provenance"
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MAIN_CONFIG, provenance / "config.yaml")
    shutil.copy2(CLASSICAL_CONFIG, provenance / "pose_baseline_rf_config.yaml")
    payload = {
        "frozen": frozen,
        "commands": [asdict(cmd) for cmd in commands],
    }
    (out_root / "controlled_suite_config.frozen.json").write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    (out_root / "controlled_command_plan.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_root / "controlled_command_plan.txt").write_text(
        "\n\n".join(
            f"[{cmd.stage}] {cmd.name}\nReason: {cmd.reason}\nCommand:\n{format_command(cmd.command)}\nExpected:\n"
            + "\n".join(f"  - {p}" for p in cmd.expected_outputs)
            for cmd in commands
        ),
        encoding="utf-8",
    )


def format_command(cmd: list[str]) -> str:
    return " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd)


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(text)).strip("_") or "step"


def selected_commands(commands: list[PlannedCommand], stage: str) -> list[PlannedCommand]:
    if stage == "all":
        return commands
    return [cmd for cmd in commands if cmd.stage == stage]


def command_values(command: list[str], option: str) -> list[str]:
    if option not in command:
        return []
    idx = command.index(option) + 1
    values: list[str] = []
    while idx < len(command) and not str(command[idx]).startswith("--"):
        values.append(str(command[idx]))
        idx += 1
    return values


def command_value(command: list[str], option: str) -> str | None:
    values = command_values(command, option)
    return values[0] if values else None


def expected_outputs_exist(planned: PlannedCommand) -> bool:
    return bool(planned.expected_outputs) and all(Path(p).is_file() for p in planned.expected_outputs)


def expected_classical_model_paths(planned: PlannedCommand) -> list[Path]:
    config_path = command_value(planned.command, "--config")
    if not config_path:
        return []
    cfg = load_yaml(Path(config_path))
    out = resolve_path(cfg.get("project", {}).get("output_root", "outputs/pose_baseline_rf_controlled"))
    dataset = str(cfg.get("features", {}).get("dataset", "mars"))
    feature_sets = command_values(planned.command, "--feature_sets") or list(cfg.get("features", {}).get("feature_sets", []))
    models = command_values(planned.command, "--models") or ["rf", "xgb"]
    seed_value = command_value(planned.command, "--seed")
    run_tag = command_value(planned.command, "--run_tag") or (f"seed{seed_value}" if seed_value else None)
    suffix = f"_{run_tag}" if run_tag else ""
    model_dir = out / "models" / dataset
    return [
        model_dir / f"{model}_{feature_set}{suffix}.pkl"
        for feature_set in feature_sets
        for model in models
    ]


def _training_run_dir(planned: PlannedCommand) -> Path | None:
    project = command_value(planned.command, "--project")
    name = command_value(planned.command, "--name")
    if not project or not name:
        return None
    return Path(project) / name


def _previous_summary_completed(planned: PlannedCommand, log_root: Path | None) -> bool:
    if log_root is None:
        return False
    summary_path = log_root / "run_summary.json"
    if not summary_path.is_file():
        return False
    try:
        records = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(records, list):
        return False
    for record in records:
        if not isinstance(record, dict):
            continue
        if (
            record.get("stage") == planned.stage
            and record.get("name") == planned.name
            and record.get("status") == "completed"
            and int(record.get("exit_code", -1)) == 0
        ):
            if planned.stage in {"train_neural", "train_fly"}:
                log_path = record.get("log_path")
                if log_path and Path(str(log_path)).is_file():
                    try:
                        text = Path(str(log_path)).read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        return False
                    if "[stop]" in text or "completion marker:" not in text:
                        return False
            return True
    return False


def _successful_legacy_log_exists(planned: PlannedCommand, log_root: Path | None) -> bool:
    if log_root is None or not log_root.is_dir():
        return False
    pattern = f"*_{safe_name(planned.stage)}_{safe_name(planned.name)}.log"
    for log_path in log_root.glob(pattern):
        try:
            with log_path.open("rb") as fh:
                size = fh.seek(0, os.SEEK_END)
                fh.seek(max(0, size - 32768), os.SEEK_SET)
                tail = fh.read().decode("utf-8", errors="replace")
        except Exception:
            continue
        if "[exit_code] 0" in tail and "[done]" in tail and "[stop]" not in tail:
            return True
    return False


def training_command_completed(planned: PlannedCommand, log_root: Path | None) -> bool:
    """Completion check for neural training commands.

    New runs write an explicit training_complete.json marker at the end of
    train_y.py.  Existing completed commands from a live pre-marker wrapper are
    still accepted only if their wrapper summary/log shows a clean exit.  This
    prevents a partially interrupted run from being skipped merely because an
    early config file and an intermediate best checkpoint already exist.
    """
    run_dir = _training_run_dir(planned)
    if run_dir is None:
        return expected_outputs_exist(planned)
    required = [
        run_dir / "best_model_macro_f1.pt",
        run_dir / "config.json",
    ]
    if not all(p.is_file() for p in required):
        return False
    if (run_dir / "training_complete.json").is_file():
        return True
    return (
        _previous_summary_completed(planned, log_root)
        or _successful_legacy_log_exists(planned, log_root)
    )


def command_completed(planned: PlannedCommand, log_root: Path | None = None) -> bool:
    if planned.stage == "train_classical":
        model_paths = expected_classical_model_paths(planned)
        return bool(model_paths) and all(p.is_file() for p in model_paths)
    if planned.stage in {"train_neural", "train_fly"}:
        return training_command_completed(planned, log_root)
    return expected_outputs_exist(planned)


def run_commands(
    commands: list[PlannedCommand],
    *,
    skip_missing_eval_models: bool,
    skip_completed: bool,
    log_root: Path,
    keep_going: bool,
    resource_interval_s: float,
) -> int:
    log_root.mkdir(parents=True, exist_ok=True)
    resource_root = log_root / "resources"
    resource_root.mkdir(parents=True, exist_ok=True)
    results = []
    overall_code = 0
    for step_idx, planned in enumerate(commands, start=1):
        if skip_completed and command_completed(planned, log_root):
            print(f"[skip-completed] {planned.stage}:{planned.name}", flush=True)
            results.append({
                "stage": planned.stage,
                "name": planned.name,
                "status": "skipped_completed",
                "reason": "all expected outputs already exist",
                "expected_outputs": planned.expected_outputs,
            })
            (log_root / "run_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            continue

        if planned.stage == "eval_classical" and skip_missing_eval_models:
            model_paths = planned.command[planned.command.index("--model_paths") + 1:]
            if "--no-allow_pred_named_annots" in model_paths:
                model_paths = model_paths[:model_paths.index("--no-allow_pred_named_annots")]
            existing = [p for p in model_paths if Path(p).is_file()]
            if not existing:
                print(f"[skip] {planned.name}: no trained model files exist yet", flush=True)
                results.append({
                    "stage": planned.stage,
                    "name": planned.name,
                    "status": "skipped",
                    "reason": "no trained model files exist yet",
                })
                continue
            planned.command = planned.command[:planned.command.index("--model_paths") + 1] + existing + ["--no-allow_pred_named_annots"]

        log_path = log_root / f"{step_idx:02d}_{safe_name(planned.stage)}_{safe_name(planned.name)}.log"
        resource_csv = resource_root / f"{step_idx:02d}_{safe_name(planned.stage)}_{safe_name(planned.name)}.resources.csv"
        start = time.time()
        print(f"\n[{planned.stage}] {planned.name}", flush=True)
        print(format_command(planned.command), flush=True)
        print(f"[log] {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8", errors="replace") as log_fh:
            log_fh.write(f"[stage] {planned.stage}\n")
            log_fh.write(f"[name] {planned.name}\n")
            log_fh.write(f"[reason] {planned.reason}\n")
            log_fh.write(f"[command] {format_command(planned.command)}\n\n")
            log_fh.flush()
            proc = subprocess.Popen(
                planned.command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            stop_event = threading.Event()
            monitor_thread = threading.Thread(
                target=monitor_resources,
                args=(int(proc.pid), resource_csv, stop_event, float(resource_interval_s)),
                daemon=True,
            )
            monitor_thread.start()
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line, end="", flush=True)
                log_fh.write(line)
            code = proc.wait()
            stop_event.set()
            monitor_thread.join(timeout=max(float(resource_interval_s), 1.0) + 2.0)
            elapsed = round(time.time() - start, 3)
            log_fh.write(f"\n[exit_code] {code}\n[elapsed_s] {elapsed}\n")

        status = "completed" if code == 0 else "failed"
        resource_summary = summarize_resource_csv(resource_csv)
        results.append({
            "stage": planned.stage,
            "name": planned.name,
            "status": status,
            "exit_code": int(code),
            "elapsed_s": elapsed,
            "log_path": str(log_path),
            "resource_summary": resource_summary,
            "expected_outputs": planned.expected_outputs,
        })
        (log_root / "run_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        if code != 0:
            overall_code = code
            print(f"[failed] {planned.name} exit_code={code}", flush=True)
            if not keep_going:
                return code
    return overall_code


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plan or run the controlled RF/XGBoost/TCN comparison suite.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("action", choices=["plan", "run", "run-all"],
                   help="'run-all' is an explicit alias for 'run --stage all'.")
    p.add_argument("--stage", choices=[
        "all", "audit", "build_visual_cache", "matrices", "train_classical", "eval_classical",
        "train_neural", "eval_neural", "build_fly_npz", "train_fly", "eval_fly", "summarize",
    ], default="all")
    p.add_argument("--suite", choices=[MARS_CONTROLLED_SUITE, FULL_MANUSCRIPT_SUITE],
                   default=MARS_CONTROLLED_SUITE,
                   help=(
                       "Named matrix preset. full_manuscript expands to MARS RF/XGB, "
                       "MARS all neural heads/capacities/stream ablations, MobileNet "
                       "full-stream portability, and Fly-v-Fly short-bout runs."
                   ))
    p.add_argument("--output_root", default=None,
                   help="Fresh run output directory. Default creates outputs/controlled_comparison_runs/run_<timestamp>.")
    p.add_argument("--device", default=None)
    p.add_argument("--tcn_hidden_dim", type=int, default=896)
    p.add_argument("--tcn_layers", type=int, default=5)
    p.add_argument("--tcn_kernel_size", type=int, default=3)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44],
                   help="Seeds for RF, XGBoost, and neural heads. Splits/PCA/eval stay fixed.")
    p.add_argument("--neural_heads", nargs="+", default=DEFAULT_NEURAL_HEADS,
                   choices=["lstm", "attention", "tcn"],
                   help="Neural temporal heads to run.")
    p.add_argument("--lstm_scope", choices=["previous", "full"], default="previous",
                   help="Which model templates to run for the LSTM head.")
    p.add_argument("--full_model_id", default="full_lstm896",
                   help="Full-stream model template used when a head is limited to full-only.")
    p.add_argument("--all_heads_on_previous_lstm_models", action=argparse.BooleanOptionalAction, default=True,
                   help="Run attention/TCN on every previous MARS LSTM model template, not only the full model.")
    p.add_argument("--mobilenet_checkpoint",
                   default=str(ROOT / "OtherPoseModels" / "mars_mobilenetv3_pose_v6_plus_fullval" / "best_pose_map5095.pt"),
                   help="Relabeled MobileNetV3-large BehaviorScopeZ pose checkpoint for the portability arm.")
    p.add_argument("--mobilenet_neural_heads", nargs="+", default=["tcn"],
                   choices=["lstm", "attention", "tcn"],
                   help="Neural heads to run on native MobileNetV3 features. Default keeps this as a focused portability arm.")
    p.add_argument("--mobilenet_model_ids", nargs="+", default=None,
                   help="Model templates to run on MobileNet features. Default: --full_model_id only.")
    p.add_argument("--skip_mobilenet_neural", action="store_true",
                   help=(
                       "Exclude MobileNetV3 neural training/evaluation commands from this plan. "
                       "Useful for running YOLO/SPPF held-out eval first, then choosing one "
                       "MobileNet capacity from those results."
                   ))
    p.add_argument("--mobilenet_resize_size", type=int, default=640,
                   help="Resize cached crops before MobileNet extraction. 640 matches the pose checkpoint training config.")
    p.add_argument("--mobilenet_cache_batch", type=int, default=1,
                   help="Sequence-window batch for MobileNet feature caching. "
                        "Effective crop batch is batch*32*(1+2), so 1 is the safer default.")
    p.add_argument("--mobilenet_cache_num_workers", type=int, default=2)
    p.add_argument("--mobilenet_cache_dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--include_fly", action=argparse.BooleanOptionalAction, default=False,
                   help="Include Fly-v-Fly fresh NPZ/training/evaluation stages.")
    p.add_argument("--fly_heads", nargs="+", default=["tcn"],
                   choices=["lstm", "attention", "tcn"],
                   help="Fly-v-Fly temporal heads when --include_fly is enabled.")
    p.add_argument("--fly_hidden_dims", nargs="+", type=int, default=[896],
                   help="Fly-v-Fly hidden dimensions when --include_fly is enabled.")
    p.add_argument("--fly_window_recipes", nargs="+", default=["w8s4"],
                   choices=["w16s8", "w8s4"],
                   help="Fly-v-Fly window recipes to build/train/evaluate.")
    p.add_argument("--exclude_tcn", action="store_true",
                   help=(
                       "After applying the suite preset, remove TCN from MARS, "
                       "MobileNet, and Fly neural-head matrices. Use this to resume "
                       "a full_manuscript run without the expensive TCN family."
                   ))
    p.add_argument("--focused_tcn_control", action="store_true",
                   help=(
                       "After applying the suite preset, keep TCN only as a focused "
                       "YOLO/SPPF full-stream neural-head control at hidden dimensions "
                       "256 and 512 across seeds. Excludes TCN from full stream "
                       "ablation, MobileNet, and Fly matrices."
                   ))
    p.add_argument("--fly_npz_seed", type=int, default=42,
                   help="Seed for Fly-v-Fly NPZ other-window subsampling. Model seeds use --seeds.")
    p.add_argument("--strict_eval_models", action="store_true",
                   help="Fail eval_classical if an expected RF/XGB model is missing.")
    p.add_argument("--skip_completed", action="store_true",
                   help=(
                       "Skip commands whose expected outputs already exist. Classical training is model-file aware, "
                       "so existing RF models are not enough to skip a command when XGBoost models are still missing."
                   ))
    p.add_argument("--log_root", default=None,
                   help="Directory for per-step logs. Default: <output_root>/logs.")
    p.add_argument("--resource_sample_interval_s", type=float, default=30.0,
                   help="Seconds between CPU/RAM/GPU resource samples for each stage.")
    p.add_argument("--keep_going", action="store_true",
                   help="Continue to later independent commands after a command fails.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.action == "run-all":
        args.action = "run"
        args.stage = "all"
    apply_suite_preset(args)
    apply_head_exclusions(args)
    if args.output_root is None:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_root = str(RUNS_ROOT / f"run_{stamp}")
    if args.mobilenet_model_ids is None:
        args.mobilenet_model_ids = [args.full_model_id]
    commands, frozen = build_plan(args)
    out_root = resolve_path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    write_plan(commands, frozen, out_root)
    chosen = selected_commands(commands, args.stage)
    print(f"Plan written to: {out_root / 'controlled_command_plan.txt'}")
    print(f"JSON plan written to: {out_root / 'controlled_command_plan.json'}")
    print(f"Selected stage: {args.stage} ({len(chosen)} commands)")
    for cmd in chosen:
        print(f"\n[{cmd.stage}] {cmd.name}")
        print(f"  {cmd.reason}")
        print(f"  {format_command(cmd.command)}")
    if args.action == "plan":
        return 0
    log_root = resolve_path(args.log_root) if args.log_root else out_root / "logs"
    return run_commands(
        chosen,
        skip_missing_eval_models=not args.strict_eval_models,
        skip_completed=bool(args.skip_completed),
        log_root=log_root,
        keep_going=bool(args.keep_going),
        resource_interval_s=float(args.resource_sample_interval_s),
    )


if __name__ == "__main__":
    raise SystemExit(main())
