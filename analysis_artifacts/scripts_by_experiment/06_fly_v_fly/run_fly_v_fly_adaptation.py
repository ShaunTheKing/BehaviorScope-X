#!/usr/bin/env python3
"""Clean release runner for the final focused Fly-v-Fly adaptation."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUITE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "controlled_comparison_runs"
    / "release_fly_v_fly_attention256_cw12_seed42"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plan or run the final Fly-v-Fly attention-256 short-bout adaptation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("action", choices=["plan", "run", "run-all"])
    parser.add_argument(
        "--stage",
        choices=["all", "build_train_npz", "train", "build_heldout_npz", "eval"],
        default="all",
    )
    parser.add_argument("--suite_root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip_completed", action="store_true")
    parser.add_argument("--keep_going", action="store_true")
    parser.add_argument(
        "--rebuild_npz",
        action="store_true",
        help="Regenerate Fly train and held-out NPZ/cache under the suite root. Default reuses final cached NPZ.",
    )
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    action = "run-all" if args.action == "run-all" else args.action
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_fly_v_fly_adaptation_suite.py"),
        action,
        "--stage",
        args.stage,
        "--suite_root",
        str(args.suite_root),
        "--seed",
        "42",
        "--fly_npz_seed",
        "42",
        "--windows",
        "w16s8",
        "w8s4",
        "--hidden_dim",
        "256",
        "--head",
        "attention",
        "--batch",
        "64",
        "--class_weighting",
        "sqrt_inverse",
        "--class_weight_clamp",
        "1.2",
        "--train_sampler",
        "random",
        "--train_npz_root",
        str(PROJECT_ROOT / "outputs" / "controlled_comparison_runs" / "fly_run_20260602" / "fly" / "npz_cache"),
        "--heldout_npz_root",
        str(PROJECT_ROOT / "outputs" / "controlled_comparison_runs" / "fly_run_20260602" / "fly" / "heldout_npz_cache"),
    ]
    if args.device:
        cmd.extend(["--device", str(args.device)])
    if args.skip_completed:
        cmd.append("--skip_completed")
    if args.keep_going:
        cmd.append("--keep_going")
    if args.rebuild_npz:
        cmd.append("--rebuild_npz")
    return cmd


def quote_command(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in cmd)


def main() -> int:
    args = parse_args()
    cmd = build_command(args)
    plan_path = Path(__file__).resolve().parent / "release_command.txt"
    plan_path.write_text(quote_command(cmd) + "\n", encoding="utf-8")
    print(f"Release command: {plan_path}")
    print(quote_command(cmd))
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
