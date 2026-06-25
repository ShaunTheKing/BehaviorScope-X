#!/usr/bin/env python
"""Copy completed focused Fly-v-Fly eval CSVs into reproducibility-facing tables."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "outputs" / "controlled_comparison_runs" / "fly_run_20260602" / "fly" / "eval"
TABLE_DIR = ROOT / "Manuscript_penultimate" / "tables" / "production" / "fly_v_fly"

COPIES = {
    "cross_run_summary.csv": "fly_focused_eval_cross_run_summary.csv",
    "per_video_metrics.csv": "fly_focused_eval_per_video_metrics.csv",
    "failures.csv": "fly_focused_eval_failures.csv",
}


def update_manifest(manifest_path: Path) -> None:
    if not manifest_path.is_file():
        return
    rows = []
    with manifest_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        for row in reader:
            table_name = row.get("table_name", "")
            if table_name in COPIES.values() and (TABLE_DIR / table_name).is_file():
                row["status"] = "ready"
            rows.append(row)
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    summary = EVAL_DIR / "eval_summary.json"
    if not summary.is_file():
        raise SystemExit(f"Eval is not complete yet: missing {summary}")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    missing = [src for src in COPIES if not (EVAL_DIR / src).is_file()]
    if missing:
        raise SystemExit(f"Eval is incomplete: missing {missing}")

    for src_name, dst_name in COPIES.items():
        src = EVAL_DIR / src_name
        dst = TABLE_DIR / dst_name
        shutil.copy2(src, dst)
        print(f"{src} -> {dst}")

    update_manifest(TABLE_DIR / "fly_reproduction_table_manifest.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
