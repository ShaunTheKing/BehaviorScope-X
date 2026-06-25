"""Audit Fly YOLO-pose keypoints that fall outside their YOLO bbox.

This is a label-geometry diagnostic, not a model-performance metric. It checks
whether the 5 Fly pose keypoints lie inside the associated YOLO xywh box in
normalized image coordinates for:

1. a full-movie prediction label directory, and
2. the original YOLO-pose validation labels.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = ROOT / "Manuscript_penultimate" / "tables" / "production" / "fly_v_fly"
DEFAULT_PRED = Path(r"F:\BehaviorScope-N\Fly_v_Fly_2021\Aggression\Aggression\movie2\predict-4\labels")
DEFAULT_VAL = Path(r"F:\BehaviorScope-N\Fly_v_Fly_2021\fly_yolo_pose_30k\labels\val")
KPT_NAMES = ["centroid", "left_wing", "right_wing", "axis_endpoint_1", "axis_endpoint_2"]


def percentile(values: list[float], q: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    return vals[int(q * (len(vals) - 1))]


def audit_dir(label_dir: Path, source: str, image_width: int = 144, image_height: int = 144) -> tuple[list[dict], list[dict]]:
    by_key = defaultdict(lambda: {"n": 0, "outside": 0, "dist": []})
    files = 0
    detections = 0
    detections_any_outside = 0
    for path in label_dir.glob("*.txt"):
        files += 1
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            vals = [float(x) for x in line.split()]
            if len(vals) < 20:
                continue
            detections += 1
            xc, yc, bw, bh = vals[1:5]
            x1, x2 = xc - bw / 2.0, xc + bw / 2.0
            y1, y2 = yc - bh / 2.0, yc + bh / 2.0
            any_outside = False
            for k, name in enumerate(KPT_NAMES):
                x, y, conf = vals[5 + 3 * k : 8 + 3 * k]
                if conf < 0.2:
                    continue
                dx = max(x1 - x, 0.0, x - x2) * image_width
                dy = max(y1 - y, 0.0, y - y2) * image_height
                outside = dx > 1e-9 or dy > 1e-9
                any_outside = any_outside or outside
                dist = math.hypot(dx, dy)
                for key in [("all", "all_keypoints"), ("keypoint", name)]:
                    row = by_key[key]
                    row["n"] += 1
                    row["outside"] += int(outside)
                    if outside:
                        row["dist"].append(dist)
            detections_any_outside += int(any_outside)

    summary = [{
        "source": source,
        "label_dir": str(label_dir),
        "label_files": files,
        "detections": detections,
        "detections_with_any_keypoint_outside": detections_any_outside,
        "detections_with_any_keypoint_outside_pct": 100.0 * detections_any_outside / detections if detections else 0.0,
    }]
    detail = []
    for (_, key_name), stats in sorted(by_key.items()):
        n = stats["n"]
        outside = stats["outside"]
        dists = stats["dist"]
        detail.append({
            "source": source,
            "keypoint": key_name,
            "n_keypoints": n,
            "outside_n": outside,
            "outside_pct": 100.0 * outside / n if n else 0.0,
            "median_outside_px": percentile(dists, 0.50),
            "p90_outside_px": percentile(dists, 0.90),
            "max_outside_px": max(dists) if dists else 0.0,
        })
    return summary, detail


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction_labels", type=Path, default=DEFAULT_PRED)
    parser.add_argument("--validation_labels", type=Path, default=DEFAULT_VAL)
    parser.add_argument("--table_dir", type=Path, default=TABLE_DIR)
    args = parser.parse_args()

    all_summary = []
    all_detail = []
    for source, label_dir in [
        ("movie2_prediction_labels", args.prediction_labels),
        ("original_yolo_validation_labels", args.validation_labels),
    ]:
        summary, detail = audit_dir(label_dir, source)
        all_summary.extend(summary)
        all_detail.extend(detail)
    write_csv(args.table_dir / "fly_pose_keypoints_outside_bbox_summary.csv", all_summary)
    write_csv(args.table_dir / "fly_pose_keypoints_outside_bbox_by_keypoint.csv", all_detail)
    for row in all_summary:
        print(row)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
