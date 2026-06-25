"""Write reproducibility-facing Fly-v-Fly annotation and interannotator tables."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import median

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - only used when OpenCV is unavailable.
    cv2 = None


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[2]
SHARED = ROOT / "shared_scripts"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from fly_to_behaviorscope_npz_full import (  # noqa: E402
    BEHAVIOR_NAMES,
    CLASS_NAMES,
    CLASS_TO_ID,
    FLY_FPS,
    expand_actions_to_frame_labels,
    load_traintest,
)
from fly_run_all_eval import (  # noqa: E402
    actions_to_bouts,
    bouts_to_frames,
    evaluate_frame_arrays,
    find_action_files,
    frames_to_bouts,
    load_actions_file,
    estimate_n_frames_for_actions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aggression_root",
        type=Path,
        default=ROOT / "Fly-v-Fly" / "Aggression",
    )
    parser.add_argument(
        "--table_dir",
        type=Path,
        default=ROOT / "Manuscript_penultimate" / "tables" / "production" / "fly_v_fly",
    )
    return parser.parse_args()


def video_frame_count(movie_dir: Path, actions: dict) -> tuple[int, str]:
    mp4 = movie_dir / f"{movie_dir.name}.mp4"
    if cv2 is not None and mp4.is_file():
        cap = cv2.VideoCapture(str(mp4))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if frames > 0:
            return frames, "mp4_frame_count"
    return int(estimate_n_frames_for_actions(actions)), "annotation_extent"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(numer: int | float, denom: int | float) -> float | str:
    if not denom:
        return ""
    return round(float(numer) / float(denom) * 100.0, 3)


def summarize() -> dict[str, Path]:
    args = parse_args()
    aggression_root = args.aggression_root
    table_dir = args.table_dir
    train_ids, test_ids = load_traintest(aggression_root)
    train_set = set(train_ids)
    test_set = set(test_ids)
    movie_ids = sorted(train_set | test_set)

    video_rows: list[dict] = []
    heldout_rows: list[dict] = []
    availability_rows: list[dict] = []
    bout_rows: list[dict] = []
    frame_agreement_rows: list[dict] = []
    class_agreement_rows: list[dict] = []
    human_metric_rows: list[dict] = []

    for movie_id in movie_ids:
        movie_dir = aggression_root / f"movie{movie_id}"
        primary_path, secondary_paths = find_action_files(movie_dir)
        primary_actions = load_actions_file(primary_path)
        n_frames, frame_source = video_frame_count(movie_dir, primary_actions)
        split = "heldout" if movie_id in test_set else "train_or_validation"

        primary_priority_frames = expand_actions_to_frame_labels(
            primary_actions, n_frames, CLASS_TO_ID
        )
        priority_bouts = frames_to_bouts(primary_priority_frames)
        durations = [
            int(b["end_frame"] - b["start_frame"] + 1)
            for b in priority_bouts
        ]
        per_class = {name: [] for name in BEHAVIOR_NAMES}
        for bout in priority_bouts:
            per_class[bout["class"]].append(
                int(bout["end_frame"] - bout["start_frame"] + 1)
            )

        base_video_row = {
            "movie_id": movie_id,
            "split": split,
            "n_frames": n_frames,
            "frame_count_source": frame_source,
            "fps": FLY_FPS,
            "primary_actions_path": str(primary_path),
            "secondary_annotation_available": bool(secondary_paths),
            "secondary_annotation_count": len(secondary_paths),
            "secondary_actions_paths": ";".join(str(p) for p in secondary_paths),
            "primary_priority_bouts_total": len(priority_bouts),
            "primary_priority_median_bout_frames": median(durations) if durations else "",
            "primary_priority_median_bout_seconds": (
                round(float(median(durations)) / FLY_FPS, 6) if durations else ""
            ),
        }
        for name in BEHAVIOR_NAMES:
            vals = per_class[name]
            base_video_row[f"{name}_bouts"] = len(vals)
            base_video_row[f"{name}_median_frames"] = median(vals) if vals else ""

            bout_rows.append({
                "movie_id": movie_id,
                "split": split,
                "behavior": name,
                "primary_priority_bouts": len(vals),
                "median_duration_frames": median(vals) if vals else "",
                "median_duration_seconds": (
                    round(float(median(vals)) / FLY_FPS, 6) if vals else ""
                ),
                "min_duration_frames": min(vals) if vals else "",
                "max_duration_frames": max(vals) if vals else "",
            })

        video_rows.append(base_video_row)
        if split == "heldout":
            heldout_rows.append(base_video_row)

        availability_rows.append({
            "movie_id": movie_id,
            "split": split,
            "primary_actions_path": str(primary_path),
            "secondary_annotation_available": bool(secondary_paths),
            "secondary_annotation_count": len(secondary_paths),
            "secondary_actions_paths": ";".join(str(p) for p in secondary_paths),
            "note": "" if secondary_paths else "no secondary annotation file found",
        })

        frame_row = {
            "movie_id": movie_id,
            "split": split,
            "n_frames": n_frames,
            "secondary_annotation_available": bool(secondary_paths),
            "secondary_annotation_count": len(secondary_paths),
            "secondary_actions_path": str(secondary_paths[0]) if secondary_paths else "",
            "overall_frame_agreement_pct": "",
            "primary_nonother_same_label_pct": "",
            "either_nonother_same_label_pct": "",
            "both_nonother_same_label_pct": "",
            "primary_nonother_frames": "",
            "either_nonother_frames": "",
            "both_nonother_frames": "",
            "note": "" if secondary_paths else "no secondary annotation file found",
        }

        if not secondary_paths:
            frame_agreement_rows.append(frame_row)
            for name in BEHAVIOR_NAMES:
                class_agreement_rows.append({
                    "movie_id": movie_id,
                    "split": split,
                    "behavior": name,
                    "secondary_annotation_available": False,
                    "primary_frames": "",
                    "same_label_frames": "",
                    "same_label_pct": "",
                    "secondary_other_frames": "",
                    "secondary_other_pct": "",
                    "secondary_different_behavior_frames": "",
                    "secondary_different_behavior_pct": "",
                    "note": "no secondary annotation file found",
                })
            human_metric_rows.append({
                "movie_id": movie_id,
                "split": split,
                "secondary_annotation_available": False,
                "direction": "",
                "reference": "",
                "comparison": "",
                "accuracy": "",
                "frame_macro_precision": "",
                "frame_macro_recall": "",
                "frame_macro_f1": "",
                "gt_bouts_total": "",
                "pred_bouts_total": "",
                "bout_macro_precision_iou010": "",
                "bout_macro_recall_iou010": "",
                "bout_macro_f1_iou010": "",
                "bout_macro_precision_iou025": "",
                "bout_macro_recall_iou025": "",
                "bout_macro_f1_iou025": "",
                "bout_macro_precision_iou050": "",
                "bout_macro_recall_iou050": "",
                "bout_macro_f1_iou050": "",
                "note": "no secondary annotation file found",
            })
            continue

        secondary_path = secondary_paths[0]
        secondary_actions = load_actions_file(secondary_path)
        primary_frames = bouts_to_frames(actions_to_bouts(primary_actions), n_frames)
        secondary_frames = bouts_to_frames(actions_to_bouts(secondary_actions), n_frames)
        same = primary_frames == secondary_frames
        other_idx = CLASS_TO_ID["other"]
        primary_nonother = primary_frames != other_idx
        either_nonother = (primary_frames != other_idx) | (secondary_frames != other_idx)
        both_nonother = (primary_frames != other_idx) & (secondary_frames != other_idx)

        frame_row.update({
            "overall_frame_agreement_pct": pct(int(same.sum()), n_frames),
            "primary_nonother_same_label_pct": pct(
                int((same & primary_nonother).sum()), int(primary_nonother.sum())
            ),
            "either_nonother_same_label_pct": pct(
                int((same & either_nonother).sum()), int(either_nonother.sum())
            ),
            "both_nonother_same_label_pct": pct(
                int((same & both_nonother).sum()), int(both_nonother.sum())
            ),
            "primary_nonother_frames": int(primary_nonother.sum()),
            "either_nonother_frames": int(either_nonother.sum()),
            "both_nonother_frames": int(both_nonother.sum()),
        })
        frame_agreement_rows.append(frame_row)

        for name in BEHAVIOR_NAMES:
            idx = CLASS_TO_ID[name]
            mask = primary_frames == idx
            same_label = (secondary_frames == idx) & mask
            secondary_other = (secondary_frames == other_idx) & mask
            secondary_diff_behavior = (
                (secondary_frames != idx) & (secondary_frames != other_idx) & mask
            )
            class_agreement_rows.append({
                "movie_id": movie_id,
                "split": split,
                "behavior": name,
                "secondary_annotation_available": True,
                "primary_frames": int(mask.sum()),
                "same_label_frames": int(same_label.sum()),
                "same_label_pct": pct(int(same_label.sum()), int(mask.sum())),
                "secondary_other_frames": int(secondary_other.sum()),
                "secondary_other_pct": pct(int(secondary_other.sum()), int(mask.sum())),
                "secondary_different_behavior_frames": int(secondary_diff_behavior.sum()),
                "secondary_different_behavior_pct": pct(
                    int(secondary_diff_behavior.sum()), int(mask.sum())
                ),
                "note": "",
            })

        for direction, gt_frames, pred_frames, reference, comparison in [
            ("primary_as_reference", primary_frames, secondary_frames, "primary", "secondary"),
            ("secondary_as_reference", secondary_frames, primary_frames, "secondary", "primary"),
        ]:
            metrics = evaluate_frame_arrays(gt_frames, pred_frames)
            human_metric_rows.append({
                "movie_id": movie_id,
                "split": split,
                "secondary_annotation_available": True,
                "direction": direction,
                "reference": reference,
                "comparison": comparison,
                "accuracy": round(metrics["accuracy"], 6),
                "frame_macro_precision": round(metrics["frame_macro_precision"], 6),
                "frame_macro_recall": round(metrics["frame_macro_recall"], 6),
                "frame_macro_f1": round(metrics["frame_macro_f1"], 6),
                "gt_bouts_total": metrics["gt_bouts_total"],
                "pred_bouts_total": metrics["pred_bouts_total"],
                "bout_macro_precision_iou010": round(metrics["bout_macro_precision_iou010"], 6),
                "bout_macro_recall_iou010": round(metrics["bout_macro_recall_iou010"], 6),
                "bout_macro_f1_iou010": round(metrics["bout_macro_f1_iou010"], 6),
                "bout_macro_precision_iou025": round(metrics["bout_macro_precision_iou025"], 6),
                "bout_macro_recall_iou025": round(metrics["bout_macro_recall_iou025"], 6),
                "bout_macro_f1_iou025": round(metrics["bout_macro_f1_iou025"], 6),
                "bout_macro_precision_iou050": round(metrics["bout_macro_precision_iou050"], 6),
                "bout_macro_recall_iou050": round(metrics["bout_macro_recall_iou050"], 6),
                "bout_macro_f1_iou050": round(metrics["bout_macro_f1_iou050"], 6),
                "note": "",
            })

    video_fields = [
        "movie_id", "split", "n_frames", "frame_count_source", "fps",
        "primary_actions_path", "secondary_annotation_available",
        "secondary_annotation_count", "secondary_actions_paths",
        "primary_priority_bouts_total", "primary_priority_median_bout_frames",
        "primary_priority_median_bout_seconds",
    ]
    for name in BEHAVIOR_NAMES:
        video_fields += [f"{name}_bouts", f"{name}_median_frames"]

    outputs = {
        "all_video_annotation_surface": table_dir / "fly_video_annotation_surface_all.csv",
        "heldout_annotation_surface": table_dir / "fly_video_annotation_surface_heldout.csv",
        "bout_distribution_by_video_class": table_dir / "fly_gt_bout_distribution_by_video_class.csv",
        "secondary_annotation_availability": table_dir / "fly_secondary_annotation_availability.csv",
        "interannotator_frame_agreement": table_dir / "fly_interannotator_frame_agreement_by_video.csv",
        "interannotator_class_agreement": table_dir / "fly_interannotator_class_agreement_by_video.csv",
        "human_human_eval_metrics": table_dir / "fly_human_human_eval_metrics_by_video.csv",
        "original_thesis_human_benchmark": table_dir / "fly_original_thesis_human_benchmark.csv",
    }
    write_csv(outputs["all_video_annotation_surface"], video_rows, video_fields)
    write_csv(outputs["heldout_annotation_surface"], heldout_rows, video_fields)
    write_csv(outputs["bout_distribution_by_video_class"], bout_rows, [
        "movie_id", "split", "behavior", "primary_priority_bouts",
        "median_duration_frames", "median_duration_seconds",
        "min_duration_frames", "max_duration_frames",
    ])
    write_csv(outputs["secondary_annotation_availability"], availability_rows, [
        "movie_id", "split", "primary_actions_path",
        "secondary_annotation_available", "secondary_annotation_count",
        "secondary_actions_paths", "note",
    ])
    write_csv(outputs["interannotator_frame_agreement"], frame_agreement_rows, [
        "movie_id", "split", "n_frames", "secondary_annotation_available",
        "secondary_annotation_count", "secondary_actions_path",
        "overall_frame_agreement_pct", "primary_nonother_same_label_pct",
        "either_nonother_same_label_pct", "both_nonother_same_label_pct",
        "primary_nonother_frames", "either_nonother_frames",
        "both_nonother_frames", "note",
    ])
    write_csv(outputs["interannotator_class_agreement"], class_agreement_rows, [
        "movie_id", "split", "behavior", "secondary_annotation_available",
        "primary_frames", "same_label_frames", "same_label_pct",
        "secondary_other_frames", "secondary_other_pct",
        "secondary_different_behavior_frames",
        "secondary_different_behavior_pct", "note",
    ])
    write_csv(outputs["human_human_eval_metrics"], human_metric_rows, [
        "movie_id", "split", "secondary_annotation_available", "direction",
        "reference", "comparison", "accuracy", "frame_macro_precision",
        "frame_macro_recall", "frame_macro_f1", "gt_bouts_total",
        "pred_bouts_total", "bout_macro_precision_iou010",
        "bout_macro_recall_iou010", "bout_macro_f1_iou010",
        "bout_macro_precision_iou025", "bout_macro_recall_iou025",
        "bout_macro_f1_iou025", "bout_macro_precision_iou050",
        "bout_macro_recall_iou050", "bout_macro_f1_iou050", "note",
    ])
    write_csv(outputs["original_thesis_human_benchmark"], [{
        "source": "Eyjolfsdottir MS thesis, Section 6.1 Human vs. human",
        "source_url": "https://thesis.caltech.edu/8195/",
        "dataset_subset": "Fly-vs-Fly Aggression",
        "second_layer_coverage": "1/10 of Aggression test movies",
        "reported_recall": "generally above 80%",
        "reported_precision": "60-80%",
        "metric_context": "trained novice annotators compared with expert annotations",
        "manuscript_use": "human-reference benchmark for Fly-v-Fly precision/recall in addition to frame and bout macro-F1",
    }], [
        "source", "source_url", "dataset_subset", "second_layer_coverage",
        "reported_recall", "reported_precision", "metric_context",
        "manuscript_use",
    ])
    return outputs


if __name__ == "__main__":
    outputs = summarize()
    print(json.dumps({k: str(v) for k, v in outputs.items()}, indent=2))
