#!/usr/bin/env python3
"""
mirror_to_notebook.py
======================
Mirror a manuscript artifact from the project root to the notebook-internal
``manuscript_outputs/data/manuscript_v4/`` so the deck and chapters render
without depending on the project root layout.

Creates **independent copies** (not hardlinks) so deletion of one location
never affects the other. Idempotent: re-running on the same source overwrites
the destination cleanly.

Usage:
    # Mirror a training run
    python scripts/mirror_to_notebook.py training_runs/R5_yolo_attn_v4_full

    # Mirror a test eval
    python scripts/mirror_to_notebook.py test_eval/R5_test_aggregate

    # Mirror everything currently under data/manuscript_v4/
    python scripts/mirror_to_notebook.py --all
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
SRC_BASE = ROOT / "data" / "manuscript_v4"
DST_BASE = ROOT / "manuscript_outputs" / "data" / "manuscript_v4"


def _mirror_one(rel_path: str) -> tuple[int, float]:
    src = SRC_BASE / rel_path
    dst = DST_BASE / rel_path
    if not src.exists():
        print(f"[skip] source missing: {src}")
        return 0, 0.0
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst)
        files = [p for p in dst.rglob("*") if p.is_file()]
    else:
        shutil.copy2(src, dst)
        files = [dst]
    n = len(files)
    sz_mb = sum(p.stat().st_size for p in files) / 1e6
    print(f"[ok]   {rel_path}  ->  {n} files, {sz_mb:.1f} MB")
    return n, sz_mb


def _mirror_all() -> None:
    if not SRC_BASE.exists():
        print(f"[abort] no source root: {SRC_BASE}")
        sys.exit(1)
    total_n, total_mb = 0, 0.0
    # Walk top-level subfolders (training_runs, test_eval, keypoint_eval, ...)
    for sub in sorted(SRC_BASE.iterdir()):
        if not sub.is_dir():
            continue
        # Mirror each child of the subfolder one at a time so progress is visible
        for child in sorted(sub.iterdir()):
            rel = child.relative_to(SRC_BASE).as_posix()
            n, mb = _mirror_one(rel)
            total_n += n
            total_mb += mb
    print(f"\n[done] total {total_n} files, {total_mb:.1f} MB mirrored")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default=None,
                   help="Path under data/manuscript_v4/ to mirror "
                        "(e.g. 'training_runs/R5_yolo_attn_v4_full').")
    p.add_argument("--all", action="store_true",
                   help="Mirror everything currently under data/manuscript_v4/.")
    args = p.parse_args()
    if args.all:
        _mirror_all()
        return
    if not args.path:
        p.error("provide a path or use --all")
    _mirror_one(args.path)


if __name__ == "__main__":
    main()
