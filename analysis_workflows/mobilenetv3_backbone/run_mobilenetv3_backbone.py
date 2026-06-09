#!/usr/bin/env python3
"""Analysis runner for the final MobileNetV3-native backbone workflow."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = THIS_DIR.parent
APP_ROOT = ANALYSIS_ROOT.parent
SUPPORT_ROOT = ANALYSIS_ROOT / "shared_analysis_code"
sys.path.insert(0, str(SUPPORT_ROOT))

import run_controlled_comparison_suite as controlled  # noqa: E402


DEFAULT_SUITE_ROOT = APP_ROOT / "outputs" / "controlled_comparison_runs" / "release_mobilenetv3_backbone"
BACKEND = controlled.MOBILENET_BACKEND_NAME


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or run the final MobileNetV3-native backbone analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("action", choices=["plan", "run", "run-all"])
    parser.add_argument(
        "--stage",
        choices=[
            "all",
            "build_npz",
            "audit",
            "build_visual_cache",
            "matrices",
            "train_classical",
            "eval_classical",
            "train_neural",
            "eval_neural",
            "summarize",
        ],
        default="all",
    )
    parser.add_argument("--suite_root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--heads", nargs="+", choices=["lstm", "attention"], default=["lstm", "attention"])
    parser.add_argument("--model_ids", nargs="+", default=["full_lstm256"])
    parser.add_argument("--cache_batch", type=int, default=1)
    parser.add_argument("--cache_num_workers", type=int, default=2)
    parser.add_argument("--cache_dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--resize_size", type=int, default=640)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument(
        "--rebuild_npz",
        action="store_true",
        help="Prepend MARS train/held-out NPZ build commands. Default reuses final caches.",
    )
    parser.add_argument("--resource_sample_interval_s", type=float, default=30.0)
    return parser.parse_args()


def build_controlled_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        action="run",
        stage="all",
        suite=controlled.MARS_CONTROLLED_SUITE,
        output_root=str(args.suite_root),
        device=args.device,
        tcn_hidden_dim=896,
        tcn_layers=5,
        tcn_kernel_size=3,
        seeds=list(args.seeds),
        neural_heads=[],
        lstm_scope="full",
        full_model_id="full_lstm256",
        all_heads_on_previous_lstm_models=False,
        mobilenet_checkpoint=str(
            APP_ROOT / "OtherPoseModels" / "mars_mobilenetv3_pose_v6_plus_fullval" / "best_pose_map5095.pt"
        ),
        mobilenet_neural_heads=list(args.heads),
        mobilenet_model_ids=list(args.model_ids),
        skip_mobilenet_neural=False,
        mobilenet_resize_size=int(args.resize_size),
        mobilenet_cache_batch=int(args.cache_batch),
        mobilenet_cache_num_workers=int(args.cache_num_workers),
        mobilenet_cache_dtype=str(args.cache_dtype),
        include_fly=False,
        fly_heads=[],
        fly_hidden_dims=[],
        fly_window_recipes=[],
        exclude_tcn=True,
        focused_tcn_control=False,
        fly_npz_seed=42,
        strict_eval_models=False,
        skip_completed=bool(args.skip_completed),
        log_root=None,
        resource_sample_interval_s=float(args.resource_sample_interval_s),
        keep_going=bool(args.keep_going),
    )


def is_mobilenet_command(command: controlled.PlannedCommand) -> bool:
    if command.stage in {"audit", "summarize"}:
        return True
    if command.stage == "build_visual_cache":
        return BACKEND in command.name
    if command.stage in {"matrices", "train_classical", "eval_classical", "train_neural", "eval_neural"}:
        return BACKEND in command.name
    return False


def stage_selected(command: controlled.PlannedCommand, stage: str) -> bool:
    return stage == "all" or command.stage == stage


def strip_unused_tcn_flags(command: controlled.PlannedCommand) -> controlled.PlannedCommand:
    cleaned: list[str] = []
    skip_next = False
    for item in command.command:
        if skip_next:
            skip_next = False
            continue
        if item in {"--tcn_layers", "--tcn_kernel_size"}:
            skip_next = True
            continue
        cleaned.append(item)
    command.command = cleaned
    return command


def build_npz_commands(args: argparse.Namespace, frozen: dict) -> list[controlled.PlannedCommand]:
    main_cfg = controlled.load_yaml(controlled.MAIN_CONFIG)
    py = str(main_cfg.get("environment", {}).get("python_executable", "python"))
    device = args.device or str(main_cfg.get("environment", {}).get("device_default", "cuda:0"))
    profile = controlled.main_profile(main_cfg)
    preprocessing = main_cfg["preprocessing"]
    full_npz = preprocessing["full_video_npz"]
    heldout = preprocessing["heldout_eval_cache"]
    recipe = main_cfg["recipes"]["mars_cw15_sqrt_pe"]

    mars_root = controlled.resolve_profile_path(main_cfg, "mars_data_root")
    yolo_weights = controlled.resolve_profile_path(main_cfg, "mars_yolo_weights")
    source_mp4_cache = controlled.resolve_profile_path(main_cfg, profile["source_mp4_cache_root"])
    train_npz_root = controlled.resolve_profile_path(main_cfg, "npz_full_video_root")
    heldout_npz_root = controlled.resolve_profile_path(main_cfg, "heldout_eval_npz_root")
    train_manifest = train_npz_root / "sequence_manifest.json"
    heldout_manifest = heldout_npz_root / "sequence_manifest.json"

    def append_common_npz_flags(cmd: list[str], *, output_root: Path, other_subsample: float) -> None:
        cmd.extend([
            "--mars_root", str(mars_root),
            "--yolo_weights", str(yolo_weights),
            "--output_root", str(output_root),
            "--source_mode", str(full_npz.get("source_mode", "mp4")),
            "--source_mp4_cache", str(source_mp4_cache),
            "--window_size", "32",
            "--window_stride", "16",
            "--crop_size", "224",
            "--group_crop_size", "224",
            "--animal_scale_factor", str(main_cfg["global_eval_recipe"]["animal_scale_factor"]),
            "--group_scale_factor", str(main_cfg["global_eval_recipe"]["group_scale_factor"]),
            "--pose_conf_threshold", str(main_cfg["global_eval_recipe"]["pose_conf_threshold"]),
            "--yolo_batch", str(full_npz.get("yolo_batch", 16)),
            "--npz_writers", str(full_npz.get("npz_writers", 1)),
            "--other_subsample", str(other_subsample),
            "--seed", "42",
            "--device", device,
            "--skip_existing",
        ])

    build_train = [
        py, str(controlled.APP_ROOT / "prepare_full_video_npz.py"),
        "--train_splits", *[str(s) for s in full_npz["train_splits"]],
        "--val_splits", *[str(s) for s in full_npz["val_splits"]],
        "--exclude_splits", *[str(s) for s in full_npz["exclude_splits"]],
    ]
    append_common_npz_flags(
        build_train,
        output_root=train_npz_root,
        other_subsample=float(full_npz.get("other_subsample", 0.30)),
    )

    build_heldout = [
        py, str(controlled.APP_ROOT / "prepare_full_video_npz.py"),
        "--train_splits", "test_1",
        "--val_splits", "test_2",
        "--exclude_splits", "train", "validation",
        "--allow_pred_named_annots",
        "--tolerate_bad_seq_frames",
    ]
    append_common_npz_flags(
        build_heldout,
        output_root=heldout_npz_root,
        other_subsample=1.0,
    )

    frozen.setdefault("release_rebuild_npz", {
        "training_cache": str(train_npz_root),
        "heldout_cache": str(heldout_npz_root),
        "heldout_manifest_split_policy": (
            "MARS test_1/test_2 are built through the NPZ builder train/val slots, "
            "with original mars_split retained for held-out evaluation."
        ),
    })

    return [
        controlled.PlannedCommand(
            stage="build_npz",
            name="build_mars_training_npz",
            command=build_train,
            expected_outputs=[str(train_manifest)],
            reason="Build MARS train/validation full-video windows used by the MobileNetV3-native comparison.",
        ),
        controlled.PlannedCommand(
            stage="build_npz",
            name="build_mars_heldout_eval_npz",
            command=build_heldout,
            expected_outputs=[str(heldout_manifest)],
            reason="Build MARS held-out test_1/test_2 windows while excluding train/validation sources.",
        ),
    ]


def write_release_plan(commands: list[controlled.PlannedCommand], frozen: dict, suite_root: Path) -> None:
    THIS_DIR.joinpath("release_command_plan.json").write_text(
        json.dumps({"frozen": frozen, "commands": [asdict(command) for command in commands]}, indent=2),
        encoding="utf-8",
    )
    THIS_DIR.joinpath("release_command_plan.txt").write_text(
        "\n\n".join(
            f"[{command.stage}] {command.name}\n"
            f"Reason: {command.reason}\n"
            f"Command:\n{controlled.format_command(command.command)}\n"
            f"Expected outputs:\n" + "\n".join(f"  - {path}" for path in command.expected_outputs)
            for command in commands
        ),
        encoding="utf-8",
    )
    suite_root.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    if args.action == "run-all":
        args.action = "run"
        args.stage = "all"
    controlled_args = build_controlled_args(args)
    commands, frozen = controlled.build_plan(controlled_args)
    selected = [
        strip_unused_tcn_flags(command)
        for command in commands
        if is_mobilenet_command(command) and stage_selected(command, args.stage)
    ]
    if args.rebuild_npz or args.stage == "build_npz":
        npz_commands = build_npz_commands(args, frozen)
        selected = npz_commands if args.stage == "build_npz" else npz_commands + selected
    write_release_plan(selected, frozen, args.suite_root)
    print(f"Command plan: {THIS_DIR / 'release_command_plan.txt'}")
    print(f"Suite root: {args.suite_root}")
    print(f"Selected MobileNetV3 stage: {args.stage} ({len(selected)} commands)")
    for command in selected:
        print(f"\n[{command.stage}] {command.name}")
        print(f"  {command.reason}")
        print(f"  {controlled.format_command(command.command)}")
    if args.action == "plan":
        return 0
    return controlled.run_commands(
        selected,
        skip_missing_eval_models=True,
        skip_completed=bool(args.skip_completed),
        log_root=args.suite_root / "logs_mobilenetv3",
        keep_going=bool(args.keep_going),
        resource_interval_s=float(args.resource_sample_interval_s),
    )


if __name__ == "__main__":
    raise SystemExit(main())


