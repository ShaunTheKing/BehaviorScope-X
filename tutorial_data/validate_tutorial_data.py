from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


REQUIRED_MODEL_FILES = [
    Path("tutorial_models/full_pose_mobilenetv3/best_pose_map5095.pt"),
    Path("tutorial_models/full_pose_mobilenetv3/best_pose_map5095.json"),
    Path("tutorial_models/full_attn_classifier_mobilenetv3/best_model_macro_f1.pt"),
    Path("tutorial_models/full_attn_classifier_mobilenetv3/config.json"),
    Path("tutorial_models/model_manifest.json"),
    Path("tutorial_models/README.md"),
]

PORTABLE_TEXT_SUFFIXES = {".json", ".md", ".txt", ".yaml", ".yml", ".csv"}
LOCAL_PATH_MARKERS = [
    ":\\",
    "C:/",
    "D:/",
    "E:/",
    "F:/",
    "G:/",
    "Users/Aegis",
    "Aegis-MSI",
]


def _format_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _find_local_paths(root: Path) -> list[tuple[Path, str]]:
    hits: list[tuple[Path, str]] = []
    if not root.exists():
        return hits
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in PORTABLE_TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in LOCAL_PATH_MARKERS:
            if marker in text:
                hits.append((path, marker))
    return hits


def _validate_models(tutorial_root: Path) -> int:
    models_root = tutorial_root / "tutorial_models"
    missing_models = [path for path in REQUIRED_MODEL_FILES if not (tutorial_root / path).is_file()]
    print(f"[tutorial] model bundle: {models_root}")
    if missing_models:
        print("[tutorial] missing tutorial model files:")
        for path in missing_models:
            print(f"  - {path}")
        return 1

    manifest_path = models_root / "model_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        print(f"[tutorial] invalid model manifest JSON: {exc}")
        return 1

    if not isinstance(manifest.get("models"), dict):
        print("[tutorial] model manifest is missing the 'models' object")
        return 1

    print("[tutorial] tutorial model files:")
    for rel_path in REQUIRED_MODEL_FILES:
        path = tutorial_root / rel_path
        print(f"  - {rel_path} ({_format_size(path.stat().st_size)})")

    local_path_hits = _find_local_paths(models_root)
    if local_path_hits:
        print("[tutorial] local absolute path markers found in tutorial model text files:")
        for path, marker in local_path_hits[:20]:
            print(f"  - {path.relative_to(tutorial_root)} contains {marker!r}")
        return 1

    return 0


def main() -> int:
    tutorial_root = Path(__file__).resolve().parent / "BehaviorScope-Y_tutorial"
    manifest_path = tutorial_root / "source_manifest.csv"
    class_names_path = tutorial_root / "class_names.txt"
    annotations_root = tutorial_root / "annotations"
    videos_root = tutorial_root / "videos"

    required = [manifest_path, class_names_path, annotations_root, videos_root]
    missing_required = [path for path in required if not path.exists()]
    if missing_required:
        print("[tutorial] missing required paths:")
        for path in missing_required:
            print(f"  - {path}")
        return 1

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        print(f"[tutorial] no rows found in {manifest_path}")
        return 1

    missing_videos = []
    missing_annotations = []
    split_counts: dict[str, int] = {}
    expected_spans = 0
    for row in rows:
        split = str(row.get("split", "")).strip() or "unknown"
        split_counts[split] = split_counts.get(split, 0) + 1
        video_path = tutorial_root / str(row.get("video_path", ""))
        annot_path = tutorial_root / str(row.get("annot_path", ""))
        if not video_path.is_file():
            missing_videos.append(video_path)
        if not annot_path.is_file():
            missing_annotations.append(annot_path)
        try:
            expected_spans += int(row.get("approved_annotation_count", "0") or 0)
        except ValueError:
            pass

    print(f"[tutorial] root: {tutorial_root}")
    print(f"[tutorial] manifest rows: {len(rows)}")
    print(f"[tutorial] split counts: {split_counts}")
    print(f"[tutorial] expected annotation spans: {expected_spans}")
    print(f"[tutorial] missing videos: {len(missing_videos)}")
    print(f"[tutorial] missing annotation files: {len(missing_annotations)}")

    if missing_videos:
        print("[tutorial] first missing videos:")
        for path in missing_videos[:10]:
            print(f"  - {path.relative_to(tutorial_root)}")
    if missing_annotations:
        print("[tutorial] first missing annotation files:")
        for path in missing_annotations[:10]:
            print(f"  - {path.relative_to(tutorial_root)}")

    if missing_videos or missing_annotations:
        print("[tutorial] incomplete. Use the GUI tutorial downloader or place the media package in this folder.")
        return 1

    model_status = _validate_models(tutorial_root)
    if model_status != 0:
        return model_status

    print("[tutorial] complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

