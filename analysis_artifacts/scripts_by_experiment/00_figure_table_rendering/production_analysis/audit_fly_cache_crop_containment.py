"""Audit whether Fly-v-Fly cached animal crops contain predicted keypoints.

This distinguishes three geometries:
  1. the raw YOLO detection bbox stored in the NPZ;
  2. the same bbox expanded around its center by a configurable factor;
  3. the actual BehaviorScope per-animal crop window used for visual features.

The audit is intended to answer whether wing keypoints that fall outside the
YOLO body box are nevertheless visible to the per-animal crop stream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = REPO_ROOT / "Manuscript_penultimate" / "tables" / "production" / "fly_v_fly"

DEFAULT_MANIFESTS = [
    REPO_ROOT / "outputs" / "controlled_comparison_runs" / "fly_run_20260602" / "fly" / "npz_cache" / "w8s4" / "sequence_manifest.json",
    REPO_ROOT / "outputs" / "controlled_comparison_runs" / "fly_run_20260602" / "fly" / "npz_cache" / "w16s8" / "sequence_manifest.json",
    REPO_ROOT / "outputs" / "controlled_comparison_runs" / "fly_run_20260602" / "fly" / "heldout_npz_cache" / "w8s4" / "sequence_manifest.json",
    REPO_ROOT / "outputs" / "controlled_comparison_runs" / "fly_run_20260602" / "fly" / "heldout_npz_cache" / "w16s4" / "sequence_manifest.json",
]

KEYPOINT_NAMES = ["centroid", "left_wing", "right_wing", "axis_endpoint_1", "axis_endpoint_2"]


def iter_manifest_items(manifest_path: Path) -> Iterable[tuple[str, dict]]:
    manifest = json.loads(manifest_path.read_text())
    for split, items in manifest.get("splits", {}).items():
        for item in items:
            yield split, item


def expand_xyxy(boxes: np.ndarray, factor: float) -> np.ndarray:
    boxes = boxes.astype(np.float32)
    center = (boxes[..., :2] + boxes[..., 2:]) * 0.5
    half = (boxes[..., 2:] - boxes[..., :2]) * 0.5 * float(factor)
    return np.concatenate([center - half, center + half], axis=-1)


def inside_xyxy(points: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    return (
        (points[..., 0] >= boxes[..., 0:1])
        & (points[..., 0] <= boxes[..., 2:3])
        & (points[..., 1] >= boxes[..., 1:2])
        & (points[..., 1] <= boxes[..., 3:4])
    )


def summarize_counts(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    grouped = (
        df.groupby(["cache", "split", "geometry", "keypoint"], as_index=False)
        .agg(
            n_keypoints=("n_keypoints", "sum"),
            contained_n=("contained_n", "sum"),
        )
    )
    grouped["contained_pct"] = grouped["contained_n"] / grouped["n_keypoints"].replace(0, np.nan) * 100.0
    return grouped


def audit_manifest(manifest_path: Path, *, bbox_expand: float, max_npz: int | None) -> pd.DataFrame:
    cache_root = manifest_path.parent
    cache_name = str(cache_root.relative_to(REPO_ROOT)).replace("\\", "/")
    rows: list[dict] = []
    n_seen = 0

    for split, item in iter_manifest_items(manifest_path):
        if max_npz is not None and n_seen >= max_npz:
            break
        npz_path = cache_root / item["sequence_npz"]
        with np.load(npz_path) as z:
            keypoints = z["animal_keypoints_raw"].astype(np.float32)
            keypoint_conf = z["animal_keypoints_conf"].astype(np.float32)
            raw_boxes = z["bbox_xyxy_animals"].astype(np.float32)
            crop_boxes = z["crop_xyxy_animals"].astype(np.float32)
            expanded_boxes = expand_xyxy(raw_boxes, bbox_expand)

        valid = np.isfinite(keypoints[..., 0]) & np.isfinite(keypoints[..., 1]) & (keypoint_conf > 0)
        geometries = {
            "raw_bbox": raw_boxes,
            f"bbox_x{bbox_expand:.2f}": expanded_boxes,
            "actual_crop": crop_boxes,
        }
        for geometry, boxes in geometries.items():
            contained = inside_xyxy(keypoints, boxes) & valid
            for ki, name in enumerate(KEYPOINT_NAMES[: keypoints.shape[-2]]):
                valid_k = valid[..., ki]
                rows.append(
                    {
                        "cache": cache_name,
                        "split": split,
                        "geometry": geometry,
                        "keypoint": name,
                        "n_keypoints": int(valid_k.sum()),
                        "contained_n": int(contained[..., ki].sum()),
                    }
                )
            rows.append(
                {
                    "cache": cache_name,
                    "split": split,
                    "geometry": geometry,
                    "keypoint": "all_keypoints",
                    "n_keypoints": int(valid.sum()),
                    "contained_n": int(contained.sum()),
                }
            )
        n_seen += 1

    return summarize_counts(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, default=None)
    parser.add_argument("--bbox-expand", type=float, default=1.20)
    parser.add_argument("--max-npz", type=int, default=None, help="Optional per-run cap for quick sampling.")
    parser.add_argument("--output", type=Path, default=TABLE_DIR / "fly_cache_crop_keypoint_containment.csv")
    args = parser.parse_args()

    manifests = args.manifest or DEFAULT_MANIFESTS
    frames = []
    for manifest in manifests:
        manifest = manifest.resolve()
        if not manifest.is_file():
            print(f"[skip] missing manifest: {manifest}")
            continue
        print(f"[audit] {manifest}")
        frames.append(audit_manifest(manifest, bbox_expand=args.bbox_expand, max_npz=args.max_npz))

    if not frames:
        raise SystemExit("No manifests audited.")
    out = pd.concat(frames, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"[write] {args.output}")
    print(out[out["keypoint"].eq("all_keypoints")].to_string(index=False))


if __name__ == "__main__":
    main()
