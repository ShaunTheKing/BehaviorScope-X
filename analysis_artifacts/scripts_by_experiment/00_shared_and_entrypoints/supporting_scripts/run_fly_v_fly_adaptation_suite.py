#!/usr/bin/env python3
"""Focused Fly-v-Fly adaptation rerun.

This wrapper runs the adaptation experiment needed for the manuscript:

- attention head only
- hidden dimension 256
- Fly windows w16/s8 and w8/s4
- one seed by default
- Fly-specific class-balance recipe
- reuse of existing Fly NPZ windows and held-out feature caches by default
- cached held-out evaluation over movies 6--10
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "config.yaml"
RUNS_ROOT = ROOT / "outputs" / "controlled_comparison_runs"
FLY_GT_MODES = ["primary", "secondary", "union", "intersection", "padded_primary"]
WINDOWS = ["w16s8", "w8s4"]


@dataclass
class PlannedCommand:
    stage: str
    name: str
    command: list[str]
    expected_outputs: list[str]
    reason: str


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    p = Path(value)
    return p if p.is_absolute() else (ROOT / p).resolve()


def profile_path(cfg: dict[str, Any], key: str) -> Path:
    profile = cfg["path_profiles"][cfg["active_path_profile"]]
    raw = str(profile[key])
    return resolve_path(raw)


def append_amp(cmd: list[str], cfg: dict[str, Any], device: str) -> None:
    env = cfg.get("environment", {})
    if bool(env.get("amp", True)) and str(device).startswith("cuda"):
        cmd.append("--amp")
        cmd.extend(["--amp_dtype", str(env.get("amp_dtype", "auto"))])


def safe_recipe_tag(class_weighting: str, clamp: float, sampler: str) -> str:
    return f"cw-{class_weighting}_clamp-{clamp:g}_sampler-{sampler}".replace(".", "p")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the focused Fly-v-Fly adaptation analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("action", choices=["plan", "run", "run-all"])
    p.add_argument(
        "--stage",
        choices=["all", "build_train_npz", "train", "build_heldout_npz", "eval"],
        default="all",
    )
    p.add_argument("--suite_root", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fly_npz_seed", type=int, default=42)
    p.add_argument("--windows", nargs="+", choices=WINDOWS, default=WINDOWS)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--head", choices=["attention"], default="attention")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--class_weighting", choices=["none", "sqrt_inverse", "inverse"], default="sqrt_inverse")
    p.add_argument("--class_weight_clamp", type=float, default=1.2)
    p.add_argument("--train_sampler", choices=["random", "weighted"], default="random")
    p.add_argument("--other_subsample", type=float, default=None)
    p.add_argument(
        "--train_npz_root",
        default=str(ROOT / "outputs" / "controlled_comparison_runs" / "fly_run_20260602" / "fly" / "npz_cache"),
        help="Existing Fly train/validation NPZ root with w16s8 and w8s4 subfolders.",
    )
    p.add_argument(
        "--heldout_npz_root",
        default=str(ROOT / "outputs" / "controlled_comparison_runs" / "fly_run_20260602" / "fly" / "heldout_npz_cache"),
        help="Existing Fly held-out NPZ/cache root with w16s4 and w8s4 subfolders.",
    )
    p.add_argument(
        "--rebuild_npz",
        action="store_true",
        help="Build new NPZ/cache under this suite root. Default reuses existing NPZ/cache roots.",
    )
    p.add_argument("--skip_completed", action="store_true")
    p.add_argument("--keep_going", action="store_true")
    return p.parse_args()


def selected_stages(action: str, stage: str, *, rebuild_npz: bool) -> list[str]:
    if action == "run-all" or stage == "all":
        if not rebuild_npz:
            return ["train", "eval"]
        return ["build_train_npz", "train", "build_heldout_npz", "eval"]
    return [stage]


def suite_root_from_args(value: str | None, tag: str) -> Path:
    if value:
        return resolve_path(value)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return RUNS_ROOT / f"fly_adaptation_{tag}_{stamp}"


def command_text(cmd: list[str]) -> str:
    return " ".join(f'"{x}"' if " " in str(x) else str(x) for x in cmd)


def expected_done(paths: list[str]) -> bool:
    return bool(paths) and all(Path(p).exists() for p in paths)


def build_plan(args: argparse.Namespace, suite_root: Path) -> tuple[list[PlannedCommand], dict[str, Any]]:
    cfg = load_yaml(CONFIG)
    py = str(cfg.get("environment", {}).get("python_executable", sys.executable))
    device = args.device or str(cfg.get("environment", {}).get("device_default", "cuda:0"))
    fly_yolo = profile_path(cfg, "fly_yolo_weights")
    fly_aggression_root = profile_path(cfg, "fly_data_root") / "Aggression"
    eval_recipe = cfg["fly_eval_recipe"]
    recipe_by_window = {
        "w16s8": dict(cfg["recipes"]["fly_full_w16s8"]),
        "w8s4": dict(cfg["recipes"]["fly_full_w8s4"]),
    }

    commands: list[PlannedCommand] = []
    npz_root = suite_root / "fly" / "npz_cache" if args.rebuild_npz else resolve_path(args.train_npz_root)
    heldout_root = suite_root / "fly" / "heldout_npz_cache" if args.rebuild_npz else resolve_path(args.heldout_npz_root)
    train_root = suite_root / "fly" / "training"
    eval_root = suite_root / "fly" / "eval"

    runs: list[dict[str, Any]] = []
    for window in args.windows:
        recipe = recipe_by_window[window]
        if args.epochs is not None:
            recipe["epochs"] = int(args.epochs)
        if args.patience is not None:
            recipe["patience"] = int(args.patience)
        recipe["batch"] = int(args.batch)
        if args.other_subsample is not None:
            recipe["other_subsample"] = float(args.other_subsample)

        train_npz = npz_root / window
        if args.rebuild_npz:
            commands.append(PlannedCommand(
                stage="build_train_npz",
                name=f"build_train_npz_{window}",
                command=[
                    py, str(ROOT / "shared_scripts" / "fly_to_behaviorscope_npz_full.py"),
                    "--aggression_root", str(fly_aggression_root),
                    "--yolo_weights", str(fly_yolo),
                    "--output_root", str(train_npz),
                    "--window_size", str(recipe["window_size"]),
                    "--window_stride", str(recipe["window_stride"]),
                    "--crop_size", str(recipe["crop_size"]),
                    "--group_crop_size", str(recipe["crop_size"]),
                    "--animal_scale_factor", str(recipe.get("animal_scale_factor", eval_recipe["animal_scale_factor"])),
                    "--group_scale_factor", str(recipe.get("group_scale_factor", eval_recipe["group_scale_factor"])),
                    "--pose_conf_threshold", str(recipe["pose_conf_threshold"]),
                    "--yolo_batch", str(recipe.get("yolo_batch", 16)),
                    "--npz_writers", str(recipe.get("npz_writers", 1)),
                    "--other_subsample", str(recipe["other_subsample"]),
                    "--seed", str(args.fly_npz_seed),
                    "--device", device,
                    "--val_movie_ids", *[str(v) for v in recipe["val_movie_ids"]],
                ],
                expected_outputs=[
                    str(train_npz / "sequence_manifest.json"),
                    str(train_npz / "fly_npz_build_summary.json"),
                ],
                reason=(
                    f"Build Fly-v-Fly {window} train/validation windows for the "
                    "focused adaptation rerun; official test movies remain excluded."
                ),
            ))

        run_name = f"fly_yolo_sppf_full_{args.head}{int(args.hidden_dim)}_{window}_seed{int(args.seed)}"
        runs.append({"name": run_name, "window": window, "recipe": recipe})
        train_cmd = [
            py, str(ROOT / "shared_scripts" / "scripts" / "train_y.py"),
            "--manifest_path", str(train_npz / "sequence_manifest.json"),
            "--yolo_weights", str(fly_yolo),
            "--yolo_backbone_end_layer", str(recipe["yolo_backbone_end_layer"]),
            "--auto_feature_cache",
            "--feature_cache_batch", str(recipe["feature_cache_batch"]),
            "--feature_cache_num_workers", str(recipe["feature_cache_num_workers"]),
            "--sequence_model", args.head,
            "--hidden_dim", str(args.hidden_dim),
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
            "--class_weighting", str(args.class_weighting),
            "--class_weight_clamp", str(args.class_weight_clamp),
            "--train_sampler", str(args.train_sampler),
            "--disable_threshold_decoder",
            "--resume",
            "--seed", str(args.seed),
            "--device", device,
            "--per_class_metrics", "--confusion_matrix",
            "--project", str(train_root),
            "--name", run_name,
        ]
        append_amp(train_cmd, cfg, device)
        commands.append(PlannedCommand(
            stage="train",
            name=run_name,
            command=train_cmd,
            expected_outputs=[
                str(train_root / run_name / "best_model_macro_f1.pt"),
                str(train_root / run_name / "config.json"),
                str(train_root / run_name / "training_complete.json"),
            ],
            reason=(
                f"Train Fly-v-Fly {args.head}-256 {window} seed {args.seed} "
                f"with class_weighting={args.class_weighting}, "
                f"class_weight_clamp={args.class_weight_clamp:g}, "
                f"train_sampler={args.train_sampler}."
            ),
        ))

        eval_key = f"w{int(recipe['window_size'])}s{int(eval_recipe['window_stride'])}"
        heldout_npz = heldout_root / eval_key
        if args.rebuild_npz:
            commands.append(PlannedCommand(
                stage="build_heldout_npz",
                name=f"build_heldout_npz_{eval_key}",
                command=[
                    py, str(ROOT / "shared_scripts" / "fly_to_behaviorscope_npz_full.py"),
                    "--aggression_root", str(fly_aggression_root),
                    "--yolo_weights", str(fly_yolo),
                    "--output_root", str(heldout_npz),
                    "--window_size", str(recipe["window_size"]),
                    "--window_stride", str(eval_recipe["window_stride"]),
                    "--crop_size", str(recipe["crop_size"]),
                    "--group_crop_size", str(recipe["crop_size"]),
                    "--animal_scale_factor", str(recipe.get("animal_scale_factor", eval_recipe["animal_scale_factor"])),
                    "--group_scale_factor", str(recipe.get("group_scale_factor", eval_recipe["group_scale_factor"])),
                    "--pose_conf_threshold", str(eval_recipe["pose_conf_threshold"]),
                    "--yolo_batch", str(eval_recipe["yolo_batch"]),
                    "--npz_writers", str(recipe.get("npz_writers", 1)),
                    "--other_subsample", "1.0",
                    "--seed", str(args.fly_npz_seed),
                    "--device", device,
                    "--val_movie_ids", *[str(v) for v in recipe["val_movie_ids"]],
                    "--only_test",
                    "--test_split_name", "test",
                    "--skip_existing",
                ],
                expected_outputs=[
                    str(heldout_npz / "sequence_manifest.json"),
                    str(heldout_npz / "fly_npz_build_summary.json"),
                ],
                reason=f"Build held-out Fly-v-Fly {eval_key} windows for cached classifier-only evaluation.",
            ))

            cache_cmd = [
                py, str(ROOT / "shared_scripts" / "scripts" / "precompute_visual_features_y.py"),
                "--manifest_path", str(heldout_npz / "sequence_manifest.json"),
                "--yolo_weights", str(fly_yolo),
                "--output_dir", str(heldout_npz / "yolo_feature_cache"),
                "--splits", "test",
                "--yolo_backbone_end_layer", str(recipe["yolo_backbone_end_layer"]),
                "--n_animals", "2",
                "--batch", str(recipe.get("feature_cache_batch", 16)),
                "--num_workers", str(recipe.get("feature_cache_num_workers", 0)),
                "--device", device,
            ]
            append_amp(cache_cmd, cfg, device)
            commands.append(PlannedCommand(
                stage="build_heldout_npz",
                name=f"cache_heldout_features_{eval_key}",
                command=cache_cmd,
                expected_outputs=[str(heldout_npz / "yolo_feature_cache" / "cache_manifest.json")],
                reason=f"Precompute frozen YOLO/SPPF held-out features once for {eval_key}.",
            ))

    eval_cmd = [
        py, str(ROOT / "shared_scripts" / "fly_run_cached_eval.py"),
        "--training_dir", str(train_root),
        "--heldout_npz_root", str(heldout_root),
        "--output_dir", str(eval_root),
        "--aggression_root", str(fly_aggression_root),
        "--yolo_weights", str(fly_yolo),
        "--device", device,
        "--fps", str(eval_recipe["fps"]),
        "--window_stride", str(eval_recipe["window_stride"]),
        "--temporal_smoothing_window", str(eval_recipe["temporal_smoothing_window"]),
        "--bout_min_duration_frames", str(eval_recipe["bout_min_duration_frames"]),
        "--gt_modes", *FLY_GT_MODES,
        "--runs", *[str(r["name"]) for r in runs],
    ]
    if str(device).startswith("cuda"):
        eval_cmd.append("--amp")
    commands.append(PlannedCommand(
        stage="eval",
        name="eval_fly_adaptation_attention256",
        command=eval_cmd,
        expected_outputs=[
            str(eval_root / "cross_run_summary.csv"),
            str(eval_root / "per_video_metrics.csv"),
            str(eval_root / "eval_summary.json"),
        ],
        reason="Evaluate the focused attention-only Fly adaptation models with annotation-aware GT modes.",
    ))

    frozen = {
        "created_unix_s": dt.datetime.now().timestamp(),
        "script": str(Path(__file__).resolve()),
        "config": str(CONFIG),
        "suite_root": str(suite_root),
        "device": device,
        "fly_aggression_root": str(fly_aggression_root),
        "fly_yolo_weights": str(fly_yolo),
        "adaptation_design": {
            "head": args.head,
            "hidden_dim": int(args.hidden_dim),
            "windows": list(args.windows),
            "model_seed": int(args.seed),
            "npz_seed": int(args.fly_npz_seed),
            "batch": int(args.batch),
            "reused_train_npz_root": str(npz_root),
            "reused_heldout_npz_root": str(heldout_root),
            "rebuild_npz": bool(args.rebuild_npz),
            "class_weighting": str(args.class_weighting),
            "class_weight_clamp": float(args.class_weight_clamp),
            "train_sampler": str(args.train_sampler),
            "reason": (
                "MARS multi-seed/model-family experiments established attention as the "
                "primary temporal head; Fly-v-Fly is rerun as a focused short-bout "
                "adaptation with a Fly-specific class-balance recipe."
            ),
        },
        "runs": runs,
        "gt_modes": FLY_GT_MODES,
    }
    return commands, frozen


def write_plan(commands: list[PlannedCommand], frozen: dict[str, Any], suite_root: Path) -> None:
    suite_root.mkdir(parents=True, exist_ok=True)
    text_lines = []
    for i, cmd in enumerate(commands, start=1):
        text_lines.append(f"[{i:02d}] {cmd.stage}:{cmd.name}")
        text_lines.append(f"reason: {cmd.reason}")
        text_lines.append(command_text(cmd.command))
        text_lines.append("")
    (suite_root / "fly_adaptation_command_plan.txt").write_text("\n".join(text_lines), encoding="utf-8")
    (suite_root / "fly_adaptation_command_plan.json").write_text(
        json.dumps([asdict(c) for c in commands], indent=2), encoding="utf-8"
    )
    (suite_root / "fly_adaptation_suite_config.frozen.json").write_text(
        json.dumps(frozen, indent=2), encoding="utf-8"
    )


def run_commands(
    commands: list[PlannedCommand],
    *,
    stages: list[str],
    skip_completed: bool,
    keep_going: bool,
) -> int:
    selected = [c for c in commands if c.stage in set(stages)]
    for cmd in selected:
        if skip_completed and expected_done(cmd.expected_outputs):
            print(f"[skip-completed] {cmd.stage}:{cmd.name}")
            continue
        print()
        print(f"[fly-adaptation] {cmd.stage}:{cmd.name}")
        print(command_text(cmd.command), flush=True)
        completed = subprocess.run(cmd.command, cwd=str(ROOT))
        if completed.returncode != 0:
            if keep_going:
                print(f"[failed] {cmd.stage}:{cmd.name} exit={completed.returncode}", flush=True)
                continue
            return int(completed.returncode)
    return 0


def main() -> int:
    args = parse_args()
    tag = safe_recipe_tag(args.class_weighting, args.class_weight_clamp, args.train_sampler)
    suite_root = suite_root_from_args(args.suite_root, tag)
    commands, frozen = build_plan(args, suite_root)
    stages = selected_stages(args.action, args.stage, rebuild_npz=bool(args.rebuild_npz))
    write_plan(commands, frozen, suite_root)

    print(f"Fly adaptation suite root: {suite_root}")
    print(
        "Matrix: "
        f"head={args.head} hidden={args.hidden_dim} windows={args.windows} seed={args.seed} "
        f"class_weighting={args.class_weighting} clamp={args.class_weight_clamp:g} "
        f"sampler={args.train_sampler}"
    )
    print(f"Stages: {stages}")
    print(f"Plan: {suite_root / 'fly_adaptation_command_plan.txt'}")

    if args.action == "plan":
        for cmd in commands:
            if cmd.stage in set(stages):
                print()
                print(f"[{cmd.stage}] {cmd.name}")
                print(f"  {cmd.reason}")
                print(f"  {command_text(cmd.command)}")
        return 0
    return run_commands(
        commands,
        stages=stages,
        skip_completed=bool(args.skip_completed),
        keep_going=bool(args.keep_going),
    )


if __name__ == "__main__":
    raise SystemExit(main())
