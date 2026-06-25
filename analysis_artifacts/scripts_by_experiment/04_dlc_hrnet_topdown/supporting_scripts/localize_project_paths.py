"""Make the MARS DLC project portable after copying it to another drive."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


def as_posix_path(path: Path) -> str:
    return path.resolve().as_posix()


def rewrite_config_project_path(config_path: Path, project_root: Path) -> None:
    text = config_path.read_text(encoding="utf-8")
    lines = []
    replaced = False
    skip_project_path_continuation = False
    for line in text.splitlines():
        if skip_project_path_continuation:
            if line.startswith((" ", "\t")) and line.strip():
                continue
            skip_project_path_continuation = False
        if line.startswith("project_path:"):
            lines.append(f"project_path: {as_posix_path(project_root)}")
            replaced = True
            if not line.split(":", 1)[1].strip():
                skip_project_path_continuation = True
        elif line.startswith("date:"):
            raw_value = line.split(":", 1)[1].strip()
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw_value):
                lines.append(f'date: "{raw_value}"')
            else:
                lines.append(line)
        elif line.startswith("iteration:") and not line.split(":", 1)[1].strip():
            lines.append("iteration: 0")
        else:
            lines.append(line)
    if not replaced:
        lines.insert(0, f"project_path: {as_posix_path(project_root)}")
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def relativize_path_value(value: str, project_root: Path) -> str:
    if not value:
        return value
    path = Path(value)
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        marker = "labeled-data"
        normalized = value.replace("\\", "/")
        idx = normalized.find(marker + "/")
        if idx >= 0:
            return normalized[idx:]
        marker = "MARS-data"
        idx = normalized.find(marker + "/")
        if idx >= 0:
            return normalized[idx:]
    return value


def copy_metadata(metadata_source: Path, metadata_dest: Path, project_root: Path) -> None:
    metadata_dest.mkdir(parents=True, exist_ok=True)

    summary_src = metadata_source / "build_summary.json"
    if summary_src.is_file():
        summary = json.loads(summary_src.read_text(encoding="utf-8"))
        (metadata_dest / "build_summary.original.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        summary["output_root"] = "."
        summary["portable_note"] = (
            "output_root is relative to the DLC project folder containing config.yaml"
        )
        (metadata_dest / "build_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    manifest_src = metadata_source / "frame_manifest.csv"
    if manifest_src.is_file():
        with manifest_src.open("r", newline="", encoding="utf-8") as src, (
            metadata_dest / "frame_manifest.csv"
        ).open("w", newline="", encoding="utf-8") as dst:
            reader = csv.DictReader(src)
            fieldnames = list(reader.fieldnames or [])
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                for column in ("image_path", "pose_json", "source_video"):
                    if column in row:
                        row[column] = relativize_path_value(row[column], project_root)
                writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_root",
        default=".",
        help="DLC project folder containing config.yaml and labeled-data.",
    )
    parser.add_argument(
        "--metadata_source",
        default="",
        help="Folder containing build_summary.json and frame_manifest.csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    config_path = project_root / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing DLC config: {config_path}")

    rewrite_config_project_path(config_path, project_root)

    if args.metadata_source:
        metadata_source = Path(args.metadata_source).resolve()
    else:
        metadata_source = (project_root.parent / "metadata").resolve()

    if metadata_source.is_dir():
        copy_metadata(metadata_source, project_root / "metadata", project_root)
    else:
        print(f"[portable] metadata source not found, skipped: {metadata_source}")

    print(f"[portable] config project_path set to {project_root}")
    print(f"[portable] metadata folder: {project_root / 'metadata'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
