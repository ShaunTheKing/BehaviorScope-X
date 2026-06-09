"""Build model-agnostic ethogram and bout summaries from prediction CSVs.

The input can be a frame-level CSV such as ``*.smoothed_frames.csv`` from
``infer_x.py`` or a window-level prediction CSV with ``frame_start`` and
``frame_end`` columns. Frame-level CSVs are preferred for exact ethograms.
Window CSVs are converted by averaging available ``prob_*`` columns across
covered frames, or by majority-voting predicted labels when probabilities are
not present.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"{path} has no prediction rows.")
    missing = {"predicted_class", "frame_start"} - set(fieldnames)
    if missing and "frame_idx" not in fieldnames:
        raise ValueError(f"{path} is missing required prediction columns: {', '.join(sorted(missing))}")
    return rows, fieldnames


def _int_value(row: dict[str, str], key: str, default: int = 0) -> int:
    value = row.get(key, "")
    if value == "":
        return default
    return int(float(value))


def _row_span(row: dict[str, str]) -> tuple[int, int]:
    if row.get("frame_idx", "") != "":
        idx = _int_value(row, "frame_idx")
        return idx, idx
    start = _int_value(row, "frame_start")
    end = _int_value(row, "frame_end", start)
    if end < start:
        end = start
    return start, end


def _class_from_row(row: dict[str, str], fallback_id: int = 0) -> tuple[int, str]:
    label = str(row.get("predicted_class", "") or fallback_id)
    class_id_text = row.get("predicted_class_id", "")
    if class_id_text != "":
        class_id = int(float(class_id_text))
    else:
        class_id = fallback_id
    return class_id, label


def _frame_labels(rows: list[dict[str, str]], fieldnames: list[str], background_label: str) -> list[tuple[int, str]]:
    max_frame = max(end for _, end in (_row_span(row) for row in rows))
    n_frames = max_frame + 1
    prob_cols = [name for name in fieldnames if name.startswith("prob_")]
    if prob_cols:
        class_labels = [name.removeprefix("prob_") for name in prob_cols]
        prob_sums = [[0.0 for _ in prob_cols] for _ in range(n_frames)]
        coverage = [0 for _ in range(n_frames)]
        for row in rows:
            start, end = _row_span(row)
            probs = [float(row.get(col, "") or 0.0) for col in prob_cols]
            for frame in range(max(0, start), min(n_frames - 1, end) + 1):
                values = prob_sums[frame]
                for idx, prob in enumerate(probs):
                    values[idx] += prob
                coverage[frame] += 1
        labels: list[tuple[int, str]] = []
        for frame, count in enumerate(coverage):
            if count <= 0:
                labels.append((-1, background_label))
                continue
            best_idx = max(range(len(prob_cols)), key=lambda idx: prob_sums[frame][idx])
            labels.append((best_idx, class_labels[best_idx]))
        return labels

    votes: list[Counter[tuple[int, str]]] = [Counter() for _ in range(n_frames)]
    label_ids: dict[str, int] = {}
    for row in rows:
        if row.get("predicted_class_id", "") == "":
            label = str(row.get("predicted_class", "") or background_label)
            label_ids.setdefault(label, len(label_ids))
            class_id, label = label_ids[label], label
        else:
            class_id, label = _class_from_row(row)
        start, end = _row_span(row)
        for frame in range(max(0, start), min(n_frames - 1, end) + 1):
            votes[frame][(class_id, label)] += 1
    return [counter.most_common(1)[0][0] if counter else (-1, background_label) for counter in votes]


def _segments(frame_labels: list[tuple[int, str]], *, include_background: bool, background_label: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    if not frame_labels:
        return segments
    start = 0
    current = frame_labels[0]
    for idx, value in enumerate(frame_labels[1:], start=1):
        if value == current:
            continue
        class_id, label = current
        if include_background or label != background_label:
            segments.append({"class_id": class_id, "behavior": label, "start_frame": start, "end_frame": idx - 1})
        start = idx
        current = value
    class_id, label = current
    if include_background or label != background_label:
        segments.append({"class_id": class_id, "behavior": label, "start_frame": start, "end_frame": len(frame_labels) - 1})
    return segments


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_csv(
    path: Path,
    *,
    fps: float,
    include_background: bool,
    background_label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows, fieldnames = _read_rows(path)
    frame_labels = _frame_labels(rows, fieldnames, background_label)
    segments = _segments(frame_labels, include_background=include_background, background_label=background_label)
    video_id = path.name.removesuffix(".smoothed_frames.csv")
    if video_id == path.name:
        video_id = path.stem
    segment_rows: list[dict[str, Any]] = []
    for segment in segments:
        start = int(segment["start_frame"])
        end = int(segment["end_frame"])
        n_frames = end - start + 1
        segment_rows.append(
            {
                "video_id": video_id,
                "source_csv": str(path),
                "behavior": segment["behavior"],
                "class_id": segment["class_id"],
                "start_frame": start,
                "end_frame": end,
                "start_s": round(start / fps, 6),
                "end_s": round(end / fps, 6),
                "duration_s": round(n_frames / fps, 6),
                "n_frames": n_frames,
            }
        )

    total_frames = len(frame_labels)
    by_behavior: dict[str, dict[str, Any]] = {}
    for segment in segment_rows:
        behavior = str(segment["behavior"])
        item = by_behavior.setdefault(
            behavior,
            {
                "video_id": video_id,
                "source_csv": str(path),
                "behavior": behavior,
                "bouts": 0,
                "frames": 0,
                "seconds": 0.0,
                "fraction": 0.0,
            },
        )
        item["bouts"] += 1
        item["frames"] += int(segment["n_frames"])
    summary_rows = []
    for item in sorted(by_behavior.values(), key=lambda row: str(row["behavior"])):
        item["seconds"] = round(int(item["frames"]) / fps, 6)
        item["fraction"] = round(int(item["frames"]) / total_frames, 6) if total_frames else 0.0
        summary_rows.append(item)

    metadata = {
        "source_csv": str(path),
        "video_id": video_id,
        "fps": fps,
        "total_frames": total_frames,
        "total_seconds": round(total_frames / fps, 6),
        "segments": len(segment_rows),
        "behaviors": sorted(by_behavior),
    }
    return segment_rows, summary_rows, metadata


def discover_inputs(args: argparse.Namespace) -> list[Path]:
    paths = [Path(value) for value in args.predictions_csv]
    if args.predictions_dir:
        root = Path(args.predictions_dir)
        paths.extend(sorted(path for path in root.rglob(args.pattern) if path.is_file()))
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ethogram timelines and bout-duration summaries from BehaviorScope-X prediction CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--predictions_csv", nargs="*", default=[], help="One or more prediction CSVs.")
    parser.add_argument("--predictions_dir", default="", help="Folder to search recursively for prediction CSVs.")
    parser.add_argument("--pattern", default="*.smoothed_frames.csv", help="Glob used with --predictions_dir.")
    parser.add_argument("--output_dir", required=True, help="Folder for ethogram_segments.csv and ethogram_behavior_summary.csv.")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second used when converting frame indices to seconds.")
    parser.add_argument("--background_label", default="other", help="Label treated as background when --exclude_background is used.")
    parser.add_argument("--exclude_background", action="store_true", help="Omit background/other segments from CSV outputs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0")
    inputs = discover_inputs(args)
    if not inputs:
        raise SystemExit("No prediction CSVs found. Select a CSV or adjust --predictions_dir/--pattern.")

    output_dir = Path(args.output_dir)
    all_segments: list[dict[str, Any]] = []
    all_summary: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for path in inputs:
        segments, summary_rows, meta = summarize_csv(
            path,
            fps=float(args.fps),
            include_background=not bool(args.exclude_background),
            background_label=str(args.background_label),
        )
        all_segments.extend(segments)
        all_summary.extend(summary_rows)
        metadata.append(meta)
        print(f"[ethogram] {path} -> {len(segments)} segments", flush=True)

    _write_csv(
        output_dir / "ethogram_segments.csv",
        all_segments,
        ["video_id", "source_csv", "behavior", "class_id", "start_frame", "end_frame", "start_s", "end_s", "duration_s", "n_frames"],
    )
    _write_csv(
        output_dir / "ethogram_behavior_summary.csv",
        all_summary,
        ["video_id", "source_csv", "behavior", "bouts", "frames", "seconds", "fraction"],
    )
    summary = {
        "input_csvs": len(inputs),
        "output_dir": str(output_dir),
        "segments_csv": str(output_dir / "ethogram_segments.csv"),
        "behavior_summary_csv": str(output_dir / "ethogram_behavior_summary.csv"),
        "inputs": metadata,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ethogram_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

