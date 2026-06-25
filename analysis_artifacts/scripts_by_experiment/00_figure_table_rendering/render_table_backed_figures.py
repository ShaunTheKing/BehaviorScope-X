"""Render manuscript figures that can be redrawn from packaged tables.

Run this from the analysis_artifacts root:

    python scripts_by_experiment/00_figure_table_rendering/render_table_backed_figures.py

The script writes new outputs under ``figures_reproduced/production`` and
``tables_reproduced/production``. It intentionally excludes figure panels that
require raw videos, full controlled-run folders, trained model outputs, or
external pose-project folders.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION = PACKAGE_ROOT / "scripts_by_experiment" / "00_figure_table_rendering" / "production_analysis"
YOLO_TABLES = PACKAGE_ROOT / "tables" / "production" / "yolo_sppf_main"
SUPP_TABLES = PACKAGE_ROOT / "tables" / "production" / "supplement"
FIG_ROOT = PACKAGE_ROOT / "figures_reproduced" / "production"
TABLE_ROOT = PACKAGE_ROOT / "tables_reproduced" / "production"


@dataclass(frozen=True)
class RenderTask:
    name: str
    command: list[str]


def p(path: Path) -> str:
    return str(path)


def tasks(python_exe: str) -> list[RenderTask]:
    yolo_figs = FIG_ROOT / "yolo_sppf_main"
    supp_figs = FIG_ROOT / "supplement"
    return [
        RenderTask(
            "framework_stream_schematics",
            [
                python_exe,
                p(PRODUCTION / "plot_framework_stream_schematics.py"),
                "--figure_dir",
                p(yolo_figs),
                "--table_dir",
                p(YOLO_TABLES),
            ],
        ),
        RenderTask(
            "yolo_full_ablation_matrix",
            [
                python_exe,
                p(PRODUCTION / "plot_yolo_full_ablation_matrix.py"),
                "--table_root",
                p(YOLO_TABLES),
                "--figure_dir",
                p(supp_figs),
                "--table_dir",
                p(TABLE_ROOT / "supplement"),
            ],
        ),
        RenderTask(
            "yolo_per_video_paired_effects",
            [
                python_exe,
                p(PRODUCTION / "plot_yolo_per_video_paired_effects.py"),
                "--table_dir",
                p(YOLO_TABLES),
                "--figure_dir",
                p(yolo_figs),
            ],
        ),
        RenderTask(
            "yolo_full_video_ethogram_comparison",
            [
                python_exe,
                p(PRODUCTION / "plot_yolo_full_video_ethogram_comparison.py"),
                "--table_dir",
                p(YOLO_TABLES),
                "--figure_dir",
                p(yolo_figs),
            ],
        ),
        RenderTask(
            "yolo_undersegmentation_merge_audit",
            [
                python_exe,
                p(PRODUCTION / "audit_yolo_undersegmentation_merges.py"),
                "--table_dir",
                p(YOLO_TABLES),
                "--figure_dir",
                p(yolo_figs),
                "--smoothed_all_bouts_csv",
                p(YOLO_TABLES / "supp_yolo_bout_fragmentation_audit_all_capacities_all_bouts_long.csv"),
                "--raw_all_bouts_csv",
                p(YOLO_TABLES / "supp_yolo_bout_fragmentation_audit_all_capacities_raw_windows_all_bouts_long.csv"),
            ],
        ),
        RenderTask(
            "yolo_sequence_structure",
            [
                python_exe,
                p(PRODUCTION / "plot_yolo_ethogram_sequence_structure.py"),
                "--table_dir",
                p(YOLO_TABLES),
                "--figure_dir",
                p(yolo_figs),
            ],
        ),
        RenderTask(
            "yolo_operating_envelope",
            [
                python_exe,
                p(PRODUCTION / "plot_yolo_operating_envelope.py"),
                "--table_dir",
                p(YOLO_TABLES),
                "--figure_dir",
                p(yolo_figs),
            ],
        ),
        RenderTask(
            "yolo_bout_fragmentation_audit",
            [
                python_exe,
                p(PRODUCTION / "audit_yolo_bout_fragmentation.py"),
                "--table_dir",
                p(YOLO_TABLES),
                "--figure_dir",
                p(supp_figs),
                "--output_name",
                "supp_yolo_bout_fragmentation_audit_all_capacities",
            ],
        ),
        RenderTask(
            "yolo_bout_sequence_ngram_audit",
            [
                python_exe,
                p(PRODUCTION / "analyze_yolo_bout_sequence_ngrams.py"),
                "--table_dir",
                p(YOLO_TABLES),
                "--figure_dir",
                p(supp_figs),
            ],
        ),
        RenderTask(
            "yolo_bout_boundary_agreement",
            [
                python_exe,
                p(PRODUCTION / "plot_yolo_bout_boundary_agreement.py"),
                "--table_dir",
                p(YOLO_TABLES),
                "--figure_dir",
                p(yolo_figs),
            ],
        ),
        RenderTask(
            "end_to_end_benchmark",
            [
                python_exe,
                p(PRODUCTION / "summarize_end_to_end_benchmarks.py"),
                "--table_dir",
                p(TABLE_ROOT / "supplement"),
                "--internal_qc_dir",
                p(TABLE_ROOT / "supplement" / "internal_qc" / "end_to_end_benchmark"),
                "--figure_dir",
                p(supp_figs),
            ],
        ),
        RenderTask(
            "dlc_heldout_summary",
            [
                python_exe,
                p(PACKAGE_ROOT / "scripts_by_experiment" / "04_dlc_hrnet_topdown" / "make_dlc_heldout_summary_figure.py"),
                "--table_dir",
                p(PACKAGE_ROOT / "tables" / "production" / "dlc_superanimal_topdown"),
                "--figure_dir",
                p(FIG_ROOT / "dlc_superanimal_topdown"),
            ],
        ),
        RenderTask(
            "descriptor_provenance",
            [
                python_exe,
                p(PACKAGE_ROOT / "scripts_by_experiment" / "05_descriptor_provenance" / "make_descriptor_provenance_figure.py"),
                "--table_dir",
                p(PACKAGE_ROOT / "tables" / "production" / "descriptor_provenance"),
                "--figure_dir",
                p(FIG_ROOT / "descriptor_provenance"),
            ],
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default=sys.executable, help="Python executable used for child plotting scripts.")
    parser.add_argument("--list", action="store_true", help="List available table-backed render tasks and exit.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--only", nargs="*", default=None, help="Optional task-name subset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    all_tasks = tasks(args.python)
    selected = all_tasks
    if args.only:
        wanted = set(args.only)
        selected = [task for task in all_tasks if task.name in wanted]
        missing = sorted(wanted - {task.name for task in selected})
        if missing:
            raise SystemExit(f"Unknown task name(s): {', '.join(missing)}")
    if args.list:
        for task in all_tasks:
            print(task.name)
        return 0
    for directory in [FIG_ROOT, TABLE_ROOT, TABLE_ROOT / "supplement"]:
        directory.mkdir(parents=True, exist_ok=True)
    for task in selected:
        print(f"[render] {task.name}")
        print(" ".join(task.command))
        if not args.dry_run:
            subprocess.run(task.command, cwd=PACKAGE_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
