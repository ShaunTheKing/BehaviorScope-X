#!/usr/bin/env python3
"""Release runner for the DLC SuperAnimal top-down BehaviorScope-X analysis."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SUPPORT = THIS_DIR / "supporting_scripts"
DEFAULT_CONFIG = THIS_DIR / "final_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or run the DLC SuperAnimal top-down BehaviorScope-X analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("action", choices=["plan", "run", "run-all"])
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "train_dlc_pose_detector",
            "build_trainval_npz",
            "build_trainval_hrnet_cache",
            "train_classifier",
            "prepare_heldout_manifest",
            "build_heldout_npz",
            "build_heldout_hrnet_cache",
            "evaluate_heldout",
        ],
        default="all",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip_completed", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as fh:
        return json.load(fh)


def resolve_path(value: str | None, base: Path | None = None) -> str:
    if not value:
        return ""
    text = str(value)
    path = Path(text)
    if path.is_absolute():
        return str(path)
    return str(((base or THIS_DIR) / path).resolve())


def py(script: Path, *args: str) -> list[str]:
    return [sys.executable, str(script), *[str(a) for a in args if str(a) != ""]]


def env_for(cfg: dict) -> dict:
    env = os.environ.copy()
    scripts_dir = resolve_path(cfg["paths"].get("behaviorscope_scripts"))
    if scripts_dir:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = scripts_dir + (os.pathsep + existing if existing else "")
    return env


def cfg_device(cfg: dict, override: str | None) -> str:
    return str(override or cfg.get("runtime", {}).get("device", "cuda:0"))


def build_commands(cfg: dict, device: str) -> list[dict]:
    paths = cfg["paths"]
    train = cfg["dlc_training"]
    topdown = cfg["topdown_npz"]
    features = cfg["hrnet_feature_cache"]
    behavior = cfg["behavior_classifier"]
    heldout = cfg["heldout"]

    project_root = resolve_path(paths["dlc_project_root"])
    behaviorscope_root = resolve_path(paths["behaviorscope_root"])
    behaviorscope_scripts = resolve_path(paths["behaviorscope_scripts"])
    dlc_config = resolve_path(paths["dlc_config"])
    dlc_model_config = resolve_path(paths.get("dlc_model_config", paths["dlc_config"]))
    conversion_table = resolve_path(paths["conversion_table"], THIS_DIR)
    pose_snapshot = resolve_path(paths["pose_snapshot"])
    detector_snapshot = resolve_path(paths["detector_snapshot"])

    trainval_npz_root = resolve_path(paths["trainval_npz_root"])
    trainval_manifest = str(Path(trainval_npz_root) / "sequence_manifest.json")
    trainval_feature_cache = resolve_path(paths["trainval_hrnet_feature_cache"])
    heldout_npz_root = resolve_path(paths["heldout_npz_root"])
    heldout_manifest = str(Path(heldout_npz_root) / "sequence_manifest.json")
    heldout_feature_cache = resolve_path(paths["heldout_hrnet_feature_cache"])
    training_root = resolve_path(paths["classifier_training_root"])
    run_name = str(paths["classifier_run_name"])
    model_path = str(Path(training_root) / run_name / "best_model_macro_f1.pt")
    model_config = str(Path(training_root) / run_name / "config.json")

    commands: list[dict] = []
    commands.append({
        "stage": "train_dlc_pose_detector",
        "description": "Fine-tune pose and detector; best checkpoints are selected by declared validation metrics inside the training script.",
        "command": py(
            SUPPORT / "dlc_superanimal_train_mars.py",
            "--config_path", dlc_config,
            "--project_root", project_root,
            "--conversion_table_csv", conversion_table,
            "--epochs", train["pose_epochs"],
            "--detector_epochs", train["detector_epochs"],
            "--batch_size", train["pose_batch_size"],
            "--detector_batch_size", train["detector_batch_size"],
            "--eval_interval", train["pose_eval_interval"],
            "--detector_eval_interval", train["detector_eval_interval"],
            "--save_epochs", train["save_epochs"],
            "--detector_save_epochs", train["detector_save_epochs"],
            "--early_stop_patience", train["early_stop_patience"],
            "--early_stop_metric", train["early_stop_metric"],
            "--dataloader_workers", train["dataloader_workers"],
            "--max_snapshots_to_keep", train["max_snapshots_to_keep"],
            "--device", device,
            "--stages", "convert", "base_dataset", "finetune_dataset", "train", "detector_train", "evaluate",
            "--resume",
        ),
    })
    commands.append({
        "stage": "build_trainval_npz",
        "description": "Build DLC top-down train/validation sequence NPZ cache.",
        "command": py(
            SUPPORT / "dlc_topdown_full_video_npz.py",
            "--mars_root", resolve_path(paths["mars_root"]),
            "--output_root", trainval_npz_root,
            "--pose_config", dlc_model_config,
            "--pose_snapshot", pose_snapshot,
            "--detector_snapshot", detector_snapshot,
            "--device", device,
            "--pose_batch_size", topdown["pose_batch_size"],
            "--detector_batch_size", topdown["detector_batch_size"],
            "--npz_writers", topdown["npz_writers"],
            "--npz_compression", topdown["npz_compression"],
            "--npz_compresslevel", topdown["npz_compresslevel"],
            "--validation_sample_limit", topdown["validation_sample_limit"],
        ),
    })
    commands.append({
        "stage": "build_trainval_hrnet_cache",
        "description": "Build HRNet-W32 semantic_concat feature cache for train/validation windows.",
        "command": py(
            SUPPORT / "dlc_feature_cache_adapter.py",
            "--manifest_path", trainval_manifest,
            "--dlc_config", dlc_config,
            "--dlc_project_root", project_root,
            "--shuffle", train["finetune_shuffle"],
            "--output_dir", trainval_feature_cache,
            "--splits", "train", "val",
            "--n_animals", features["n_animals"],
            "--num_keypoints", features["num_keypoints"],
            "--mode", "dlc_backbone",
            "--feature_tap", features["feature_tap"],
            "--snapshot_index", features["snapshot_index"],
            "--batch", features["batch_size"],
            "--device", device,
            "--cache_dtype", features["cache_dtype"],
        ),
    })
    commands.append({
        "stage": "train_classifier",
        "description": "Train Attention-256 classifier from precomputed DLC-HRNet visual features and DLC pose-derived features.",
        "command": py(
            Path(behaviorscope_scripts) / "train_y.py",
            "--manifest_path", trainval_manifest,
            "--yolo_weights", behavior["dummy_yolo_weights"],
            "--use_feature_cache", trainval_feature_cache,
            "--precomputed_visual_dim", features["feature_dim"],
            "--sequence_model", behavior["sequence_model"],
            "--hidden_dim", behavior["hidden_dim"],
            "--positional_encoding", behavior["positional_encoding"],
            "--epochs", behavior["epochs"],
            "--batch", behavior["batch_size"],
            "--num_workers", behavior["num_workers"],
            "--lr", behavior["lr"],
            "--weight_decay", behavior["weight_decay"],
            "--patience", behavior["patience"],
            "--dropout", behavior["dropout"],
            "--pose_fusion_dim", behavior["pose_fusion_dim"],
            "--pose_fusion_strategy", behavior["pose_fusion_strategy"],
            "--class_weighting", behavior["class_weighting"],
            "--class_weight_clamp", behavior["class_weight_clamp"],
            "--train_sampler", behavior["train_sampler"],
            "--disable_threshold_decoder",
            "--resume",
            "--seed", behavior["seed"],
            "--device", device,
            "--per_class_metrics",
            "--confusion_matrix",
            "--project", training_root,
            "--name", run_name,
            "--amp",
            "--amp_dtype", "auto",
        ),
    })
    commands.append({
        "stage": "prepare_heldout_manifest",
        "description": "Prepare held-out MP4/.annot manifest; test_1/test_2 are retained as mars_split metadata.",
        "command": py(
            SUPPORT / "prepare_mars_heldout_mp4_manifest.py",
            "--mars_root", resolve_path(paths["mars_root"]),
            "--dest_root", resolve_path(paths["heldout_mp4_manifest_root"]),
            "--splits", "test_1", "test_2",
            "--fps", heldout["fps"],
            "--conversion_retries", heldout["conversion_retries"],
            "--conversion_retry_delay_s", heldout["conversion_retry_delay_s"],
        ),
    })
    commands.append({
        "stage": "build_heldout_npz",
        "description": "Build DLC top-down held-out NPZ cache for the manuscript MARS test videos.",
        "command": py(
            SUPPORT / "dlc_topdown_full_video_npz.py",
            "--mars_root", resolve_path(paths["mars_root"]),
            "--source_manifest_csv", resolve_path(paths["heldout_source_manifest"]),
            "--output_root", heldout_npz_root,
            "--pose_config", dlc_model_config,
            "--pose_snapshot", pose_snapshot,
            "--detector_snapshot", detector_snapshot,
            "--device", device,
            "--pose_batch_size", topdown["pose_batch_size"],
            "--detector_batch_size", topdown["detector_batch_size"],
            "--npz_writers", topdown["npz_writers"],
            "--npz_compression", topdown["npz_compression"],
            "--npz_compresslevel", topdown["npz_compresslevel"],
            "--validation_sample_limit", topdown["validation_sample_limit"],
            "--exclude_splits", "none",
        ),
    })
    commands.append({
        "stage": "build_heldout_hrnet_cache",
        "description": "Build HRNet-W32 semantic_concat feature cache for held-out DLC top-down windows.",
        "command": py(
            SUPPORT / "dlc_feature_cache_adapter.py",
            "--manifest_path", heldout_manifest,
            "--dlc_config", dlc_config,
            "--dlc_project_root", project_root,
            "--shuffle", train["finetune_shuffle"],
            "--output_dir", heldout_feature_cache,
            "--splits", "train", "val",
            "--n_animals", features["n_animals"],
            "--num_keypoints", features["num_keypoints"],
            "--mode", "dlc_backbone",
            "--feature_tap", features["feature_tap"],
            "--snapshot_index", features["snapshot_index"],
            "--batch", features["batch_size"],
            "--device", device,
            "--cache_dtype", features["cache_dtype"],
        ),
    })
    commands.append({
        "stage": "evaluate_heldout",
        "description": "Evaluate the trained DLC-HRNet classifier on the held-out DLC top-down cache.",
        "command": py(
            Path(behaviorscope_scripts) / "eval_manifest_classifier_y.py",
            "--manifest_path", heldout_manifest,
            "--feature_cache_dir", heldout_feature_cache,
            "--mars_root", resolve_path(paths["mars_root"]),
            "--splits", "test_1", "test_2",
            "--model_path", model_path,
            "--model_config", model_config,
            "--yolo_weights", behavior["dummy_yolo_weights"],
            "--output_root", resolve_path(paths["heldout_eval_root"]),
            "--batch", heldout["eval_batch_size"],
            "--num_workers", heldout["eval_num_workers"],
            "--device", device,
            "--amp",
            "--amp_dtype", "auto",
            "--temporal_smoothing_window", heldout["temporal_smoothing_window"],
            "--bout_min_duration_frames", heldout["bout_min_duration_frames"],
            "--disable_threshold_decoder",
        ),
    })
    return commands


def select_commands(commands: list[dict], stage: str) -> list[dict]:
    if stage == "all":
        return commands
    return [cmd for cmd in commands if cmd["stage"] == stage]


def write_plan(commands: list[dict]) -> None:
    json_path = THIS_DIR / "release_command_plan.json"
    txt_path = THIS_DIR / "release_command_plan.txt"
    json_path.write_text(json.dumps(commands, indent=2), encoding="utf-8")
    lines = []
    for i, item in enumerate(commands, start=1):
        lines.append(f"[{i:02d}] {item['stage']}")
        lines.append(str(item.get("description", "")))
        lines.append(" ".join(item["command"]))
        lines.append("")
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[plan] wrote {json_path}")
    print(f"[plan] wrote {txt_path}")


def run_commands(commands: list[dict], cfg: dict) -> None:
    env = env_for(cfg)
    for item in commands:
        print(f"\n[run] {item['stage']}", flush=True)
        print(" ".join(item["command"]), flush=True)
        subprocess.run(item["command"], check=True, env=env)


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    commands = select_commands(build_commands(cfg, cfg_device(cfg, args.device)), args.stage)
    write_plan(commands)
    if args.action in {"run", "run-all"}:
        run_commands(commands, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
