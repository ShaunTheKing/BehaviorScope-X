from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

THIS = Path(__file__).resolve().parent
APP_ROOT = THIS.parents[2]
SHARED_ANALYSIS = APP_ROOT / "analysis_workflows" / "shared_analysis_code"
sys.path.insert(0, str(APP_ROOT))
sys.path.insert(0, str(SHARED_ANALYSIS))

from batch_infer_eval import (  # noqa: E402
    convert_seq_to_mp4_with_retries,
    discover_jobs,
    probe_video,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prepare held-out MARS MP4/.annot source manifest for DLC top-down cache building.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mars_root", required=True)
    p.add_argument("--dest_root", required=True)
    p.add_argument("--splits", nargs="+", default=["test_1", "test_2"])
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--conversion_retries", type=int, default=3)
    p.add_argument("--conversion_retry_delay_s", type=float, default=2.0)
    p.add_argument("--tolerate_bad_seq_frames", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_videos", type=int, default=0)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def schema_split(mars_split: str) -> str:
    # The downstream sequence_manifest schema has only train/val containers.
    # Keep the real source split in mars_split for held-out evaluation.
    return "train" if mars_split == "test_1" else "val"


def main() -> int:
    args = parse_args()
    mars_root = Path(args.mars_root).resolve()
    dest_root = Path(args.dest_root).resolve()
    jobs = discover_jobs(mars_root, args.splits, allow_pred_named_annots=False)
    if int(args.max_videos) > 0:
        jobs = jobs[: int(args.max_videos)]
    if not jobs:
        raise SystemExit(f"No held-out jobs discovered under {mars_root} for splits={args.splits}")

    print(f"[heldout-mp4] discovered {len(jobs)} jobs", flush=True)
    print(f"[heldout-mp4] dest_root={dest_root}", flush=True)
    if args.dry_run:
        for job in jobs:
            print(f"[dry] {job.split}/{job.video_id} seq={job.seq_path} annot={job.annot_path}", flush=True)
        return 0

    dest_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    failures: list[dict] = []
    t0 = time.time()

    for i, job in enumerate(jobs, start=1):
        if job.seq_path is None:
            failures.append({"split": job.split, "video_id": job.video_id, "stage": "discover", "error": "missing seq"})
            continue
        video_dest_dir = dest_root / job.split / job.video_id
        video_dest_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = video_dest_dir / f"{job.video_id}.mp4"
        annot_path = video_dest_dir / job.annot_path.name

        print(f"[heldout-mp4] [{i}/{len(jobs)}] {job.split}/{job.video_id}", flush=True)
        try:
            info = convert_seq_to_mp4_with_retries(
                job.seq_path,
                mp4_path,
                seek_mat_path=job.seek_mat_path,
                fps=float(args.fps),
                overwrite=bool(args.overwrite),
                retries=int(args.conversion_retries),
                retry_delay_s=float(args.conversion_retry_delay_s),
                tolerate_bad_frames=bool(args.tolerate_bad_seq_frames),
            )
            probe = probe_video(mp4_path)
            if not probe.get("valid"):
                raise RuntimeError(f"converted MP4 probe failed: {probe}")
            shutil.copy2(job.annot_path, annot_path)
            rows.append(
                {
                    "split": schema_split(job.split),
                    "video_path": str(mp4_path),
                    "annot_path": str(annot_path),
                    "video_id": job.video_id,
                    "mars_split": job.split,
                    "fps": float(args.fps),
                    "seq_path": str(job.seq_path),
                    "seek_mat_path": str(job.seek_mat_path or ""),
                    "converted": bool(info.get("converted")),
                    "frames": int(probe.get("frames") or info.get("frames") or 0),
                }
            )
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "split": job.split,
                    "video_id": job.video_id,
                    "stage": "convert_seq_to_mp4",
                    "error": repr(exc),
                }
            )
            print(f"[heldout-mp4] ERROR {job.split}/{job.video_id}: {exc}", flush=True)

    manifest_path = dest_root / "source_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = [
            "split",
            "video_path",
            "annot_path",
            "video_id",
            "mars_split",
            "fps",
            "seq_path",
            "seek_mat_path",
            "converted",
            "frames",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    failures_path = dest_root / "source_manifest_failures.csv"
    if failures:
        with failures_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted({k for row in failures for k in row}))
            writer.writeheader()
            writer.writerows(failures)

    total_bytes = sum(p.stat().st_size for p in dest_root.rglob("*") if p.is_file())
    summary = {
        "mars_root": str(mars_root),
        "dest_root": str(dest_root),
        "splits": list(args.splits),
        "jobs_discovered": len(jobs),
        "sources_written": len(rows),
        "failures": len(failures),
        "manifest_path": str(manifest_path),
        "failures_path": str(failures_path) if failures else "",
        "total_gib": round(total_bytes / (1024**3), 3),
        "elapsed_s": round(time.time() - t0, 1),
    }
    (dest_root / "prepare_mars_heldout_mp4_manifest_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} held-out MP4 conversions failed; see {failures_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


