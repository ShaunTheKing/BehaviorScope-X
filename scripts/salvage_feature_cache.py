#!/usr/bin/env python3
"""
salvage_feature_cache.py
=========================
Re-key a feature cache directory whose filenames were computed against an
earlier version of the manifest, so that the current manifest's
``feature_cache_path()`` lookups hit the right files.

Why this exists
---------------
``feature_cache_path()`` derives each cache filename from a sha1 of
``sample.id | sequence_npz | start_frame | end_frame``. If the manifest is
rebuilt with a different path normalisation (relative vs absolute) or
different frame conventions (source-video vs clip-relative), the hash flips
even though the actual cached features are still valid (they only depend on
the NPZ pixel data and the YOLO backbone, both unchanged).

This script:
  1. Walks the current manifest (train+val).
  2. For each sample, computes the expected cache filename via the production
     ``feature_cache_path()``.
  3. Groups expected-by-stem and existing-by-stem (multiple per stem allowed —
     the production ``_safe_cache_stem`` truncates at 200 chars so different
     windows of long-named clips legitimately share a stem).
  4. For each stem:
       - First, mark any sample whose exact expected file already exists as
         "already correct".
       - Then, for the remaining samples, rename one available disk candidate
         (with the same stem) to that sample's expected hash. Cached features
         are deterministic from the underlying NPZ + frozen YOLO backbone, so
         any candidate with the right stem produces identical features.
       - Samples beyond available candidates -> "missing" (auto_feature_cache
         rebuilds them at training time).
       - Disk files beyond manifest demand -> "orphans". KEPT by default;
         pass ``--delete_orphans`` to remove them.

Critical safety property: this script will NEVER delete a cache file just
because it shares a stem with another file. Stem collisions are expected when
clip names are long (the hash differentiates them). Earlier versions of this
script destroyed valid cache entries by treating same-stem files as "duplicates"
— the current implementation explicitly avoids that.

Idempotent: if files are already correctly named, nothing happens.

Usage:
    python scripts/salvage_feature_cache.py ^
        --manifest mars_behaviorscope_npz_yolo_kp/sequence_manifest.json ^
        --cache_dir mars_behaviorscope_npz_yolo_kp/yolo_feature_cache ^
        --dry_run    # print plan without modifying anything

    # Once dry-run looks correct:
    python scripts/salvage_feature_cache.py ^
        --manifest mars_behaviorscope_npz_yolo_kp/sequence_manifest.json ^
        --cache_dir mars_behaviorscope_npz_yolo_kp/yolo_feature_cache
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Use the actual production functions so the hash logic cannot drift.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "BehaviorScope-Y"))
from data_y import feature_cache_path, load_n_manifest  # noqa: E402


def _expected_path(sample, cache_dir: Path) -> Path:
    """Use the production function so the hash matches exactly what
    train_y.py + _missing_feature_cache_count compute at runtime."""
    return feature_cache_path(cache_dir, sample)


_HASH_TAIL = re.compile(r"_([0-9a-f]{12})\.npz$")


def _strip_hash(filename: str) -> str | None:
    """Return the stem prefix (everything before the 12-char hex hash)."""
    m = _HASH_TAIL.search(filename)
    if not m:
        return None
    return filename[: m.start()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",  required=True, type=Path)
    p.add_argument("--cache_dir", required=True, type=Path)
    p.add_argument("--splits",    nargs="+", default=["train", "val"])
    p.add_argument("--dry_run",   action="store_true",
                   help="Print plan only; do not rename or delete.")
    p.add_argument("--delete_orphans", action="store_true",
                   help="Also delete cache files whose stems do not appear in the "
                        "current manifest (or that exceed manifest demand for a "
                        "given stem). OFF by default — orphans are only logged.")
    args = p.parse_args()

    if not args.manifest.exists():
        print(f"[abort] manifest not found: {args.manifest}")
        sys.exit(1)
    if not args.cache_dir.exists():
        print(f"[abort] cache dir not found: {args.cache_dir}")
        sys.exit(1)

    # CRITICAL: resolve to absolute paths up front. The hash key in
    # feature_cache_path() uses the (string form of the) absolute sequence_npz
    # path that load_n_manifest() builds from manifest.parent. If we pass a
    # relative manifest path, the resulting hashes won't match what train_y.py
    # sees when it's invoked with the absolute --manifest_path string. Force
    # absolute here so this script and train always agree.
    args.manifest = args.manifest.resolve()
    args.cache_dir = args.cache_dir.resolve()
    print(f"[salvage] resolved manifest:  {args.manifest}")
    print(f"[salvage] resolved cache_dir: {args.cache_dir}")

    # 1. Load manifest the same way training does (absolute paths, dataclasses)
    splits, _class_to_idx, _idx_to_class, _meta = load_n_manifest(args.manifest)
    samples = []
    for sp in args.splits:
        samples.extend(splits.get(sp, []))
    if not samples:
        print(f"[abort] no samples found in splits {args.splits}")
        sys.exit(1)

    print(f"[salvage] manifest:  {args.manifest}")
    print(f"[salvage] cache_dir: {args.cache_dir}")
    print(f"[salvage] samples in manifest (splits {args.splits}): {len(samples):,}")

    # Pre-compute the canonical (production) target path for every sample.
    # IMPORTANT: never collapse to one-per-stem. The production helper
    # _safe_cache_stem(text, max_len=200) truncates long sample.ids, so multiple
    # distinct windows of the same long-named clip legitimately share a stem
    # (they differ only in the trailing _f00XXXX_sXXXX which gets cut off).
    # The hash differentiates them in the full filename.
    expected_paths: list[tuple[Path, object]] = []
    expected_by_stem: dict[str, list[tuple[Path, object]]] = {}
    truncation_collisions = 0
    for s in samples:
        target = _expected_path(s, args.cache_dir)
        stem = _strip_hash(target.name)
        if stem is None:
            continue
        bucket = expected_by_stem.setdefault(stem, [])
        if bucket:
            truncation_collisions += 1
        bucket.append((target, s))
        expected_paths.append((target, s))

    if truncation_collisions:
        print(f"[salvage] note: {truncation_collisions:,} manifest samples share a "
              f"truncated stem with another sample (this is expected — the "
              f"underlying hash still differentiates them; we will NEVER delete "
              f"a file because its stem matches another).")

    # 2. Walk cache_dir -> group existing files by stem (multiple per stem allowed)
    existing_by_stem: dict[str, list[Path]] = {}
    for p in args.cache_dir.glob("*.npz"):
        stem = _strip_hash(p.name)
        if stem is None:
            continue
        existing_by_stem.setdefault(stem, []).append(p)

    n_disk_files = sum(len(v) for v in existing_by_stem.values())
    print(f"[salvage] existing cache stems on disk:           {len(existing_by_stem):,}")
    print(f"[salvage] existing cache files on disk:           {n_disk_files:,}")
    print(f"[salvage] expected manifest samples:              {len(expected_paths):,}")

    # 3. Plan: rename, delete, missing
    #    For each stem we have:
    #      - K manifest samples (expected_paths with that stem)
    #      - M disk files with that stem
    #    Rules:
    #      - If a sample's exact expected file already exists -> already_correct
    #      - Else, pick an unused disk candidate with the same stem and rename
    #        to that sample's expected hash. The cached features are derived
    #        from the underlying NPZ + frozen YOLO backbone (deterministic),
    #        so any candidate with the right stem produces identical features
    #        regardless of which sample it was originally written for.
    #      - If K > M, the K-M leftover samples are missing (auto_feature_cache).
    #      - If M > K, the M-K leftover files are orphans (no manifest match).
    #        We do NOT auto-delete orphans; we only log them.
    rename_plan: list[tuple[Path, Path]] = []
    orphan_files: list[Path] = []
    already_correct = 0
    missing: list[tuple[str, object]] = []  # (stem, sample)

    for stem, samples_for_stem in expected_by_stem.items():
        candidates = list(existing_by_stem.get(stem, []))
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        used_paths: set[Path] = set()

        # First pass: claim already-correct files
        remaining_samples: list[tuple[Path, object]] = []
        for target, sample in samples_for_stem:
            if target.exists():
                already_correct += 1
                used_paths.add(target)
            else:
                remaining_samples.append((target, sample))

        # Second pass: rename remaining candidates -> remaining samples
        unused_candidates = [p for p in candidates if p not in used_paths]
        for (target, sample), candidate in zip(remaining_samples, unused_candidates):
            rename_plan.append((candidate, target))
            used_paths.add(candidate)

        # Leftover samples (no candidate) -> missing
        for target, sample in remaining_samples[len(unused_candidates):]:
            missing.append((stem, sample))

        # Leftover candidates beyond what samples need -> orphans
        for p in unused_candidates[len(remaining_samples):]:
            orphan_files.append(p)

    # Disk files with stems that don't appear in the manifest at all
    for stem, files in existing_by_stem.items():
        if stem not in expected_by_stem:
            orphan_files.extend(files)

    # 4. Validate one renamed file by trying to load it (sanity)
    if rename_plan:
        sample_path = rename_plan[0][0]
        try:
            with np.load(sample_path, allow_pickle=False) as data:
                keys = list(data.files)
            print(f"[salvage] sample cache loads OK: {sample_path.name}  keys={keys}")
        except Exception as exc:
            print(f"[abort] sample cache file failed to load: {sample_path}: {exc}")
            sys.exit(2)

    # 5. Report
    print()
    print(f"  Already correct:                       {already_correct:,}")
    print(f"  Will rename (stem match -> right hash) {len(rename_plan):,}")
    print(f"  Missing (auto_feature_cache rebuilds): {len(missing):,}")
    print(f"  Orphan files (no manifest match):      {len(orphan_files):,}")
    cov = (already_correct + len(rename_plan)) / max(len(expected_paths), 1)
    print(f"  Coverage after salvage (samples):      {cov*100:.1f}%")
    if not args.delete_orphans and orphan_files:
        print(f"  (orphan files will be KEPT; pass --delete_orphans to remove)")

    if args.dry_run:
        print("\n[dry-run] no changes made.")
        return

    print("\n[salvage] applying...")
    for src, dst in rename_plan:
        src.rename(dst)
    deleted_bytes = 0
    if args.delete_orphans:
        for p in orphan_files:
            try:
                deleted_bytes += p.stat().st_size
                p.unlink()
            except FileNotFoundError:
                pass
        print(f"[salvage] renamed {len(rename_plan):,}, deleted {len(orphan_files):,} "
              f"orphans ({deleted_bytes/1e6:.1f} MB freed)")
    else:
        print(f"[salvage] renamed {len(rename_plan):,}; orphans preserved "
              f"({len(orphan_files):,} files, kept on disk).")

    # Final tally
    final = list(args.cache_dir.glob("*.npz"))
    print(f"[salvage] cache now contains {len(final):,} files.")
    print(f"[salvage] remaining {len(missing):,} samples will be built by "
          f"--auto_feature_cache when training starts.")


if __name__ == "__main__":
    main()
