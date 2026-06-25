"""Create production Figure 1 and Figure 2 from validated source assets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
STYLE_DIR = THIS_DIR.parent / "current_analysis"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import apply_style, panel_label, save_figure  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("Manuscript_penultimate/figures/production/source_assets"),
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=Path("Manuscript_penultimate/figures/production/yolo_sppf_main"),
    )
    parser.add_argument(
        "--table_dir",
        type=Path,
        default=Path("Manuscript_penultimate/tables/production/yolo_sppf_main"),
    )
    parser.add_argument(
        "--suite_root",
        type=Path,
        default=Path("outputs/controlled_comparison_runs/run_20260528_205727"),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def render_full_image(source: Path, out_base: Path) -> None:
    apply_style()
    image = plt.imread(source)
    height, width = image.shape[:2]
    fig_w = 11.0
    fig_h = max(4.5, fig_w * height / width)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), constrained_layout=True)
    ax.imshow(image)
    ax.set_axis_off()
    save_figure(fig, out_base)
    plt.close(fig)


def render_pose_quality(source_dir: Path, figure_dir: Path) -> None:
    apply_style()
    training = plt.imread(source_dir / "pose_model_training_curves.png")
    keypoints = plt.imread(source_dir / "keypoint_accuracy.png")
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.3), constrained_layout=True)
    for ax, image, label, title in [
        (axes[0], training, "A", "YOLO-pose training convergence"),
        (axes[1], keypoints, "B", "Held-out keypoint accuracy"),
    ]:
        ax.imshow(image)
        ax.set_axis_off()
        ax.set_title(title, fontsize=11, pad=8)
        panel_label(ax, label, x=-0.04, y=1.04)
    save_figure(fig, figure_dir / "fig2_yolo_pose_quality")
    plt.close(fig)


def write_metadata(args: argparse.Namespace) -> None:
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    frozen = read_json(args.suite_root / "controlled_suite_config.frozen.json")
    matrix = read_json(args.suite_root / "classical" / "matrices" / "mars_yolo_sppf" / "matrix_build_summary.json")
    yolo_contract = frozen.get("visual_backend_contracts", {}).get("yolo_sppf", {})
    paths = frozen.get("paths", {})

    fig1_rows = [
        {
            "figure": "fig1_framework_streams",
            "source_asset": str(args.source_dir / "input_stream_decomposition_mouse061.png"),
            "backend": "yolo_sppf",
            "purpose": "BehaviorScope-Y framework and cached input streams",
            "panels": "source frame, group-scene crop, per-animal crops, pose-self stream, relational geometry stream",
        }
    ]
    pd.DataFrame(fig1_rows).to_csv(args.table_dir / "fig1_framework_streams_source_metadata.csv", index=False)

    provenance_rows = [
        {"quantity": "pose_model_family", "value": yolo_contract.get("pose_model_family", "YOLO-pose")},
        {"quantity": "pose_model_size", "value": yolo_contract.get("pose_model_size", "nano")},
        {"quantity": "feature_layer", "value": yolo_contract.get("feature_layer", "")},
        {"quantity": "pooling", "value": yolo_contract.get("pooling", "")},
        {"quantity": "feature_dim_per_crop", "value": yolo_contract.get("feature_dim_per_crop", "")},
        {"quantity": "raw_visual_dim_per_frame", "value": yolo_contract.get("raw_visual_dim_per_frame", "")},
        {"quantity": "pose_relation_dim_per_frame", "value": matrix.get("pose_relation_dim_per_frame", "")},
        {"quantity": "pose_visual_pca_dim_per_frame", "value": matrix.get("pose_visual_pca_dim_per_frame", "")},
        {"quantity": "train_feature_cache", "value": paths.get("train_feature_cache", "")},
        {"quantity": "heldout_feature_cache", "value": paths.get("heldout_feature_cache", "")},
        {"quantity": "train_windows", "value": matrix.get("split_provenance", {}).get("train", {}).get("n_windows", "")},
        {"quantity": "validation_windows", "value": matrix.get("split_provenance", {}).get("val", {}).get("n_windows", "")},
        {"quantity": "heldout_windows", "value": matrix.get("split_provenance", {}).get("heldout", {}).get("n_windows", "")},
    ]
    pd.DataFrame(provenance_rows).to_csv(args.table_dir / "fig2_yolo_feature_provenance.csv", index=False)
    pd.DataFrame(
        [
            {
                "figure": "fig2_yolo_pose_quality",
                "source_asset": str(args.source_dir / "pose_model_training_curves.png"),
                "panel": "A",
                "description": "YOLO-pose training convergence curves",
            },
            {
                "figure": "fig2_yolo_pose_quality",
                "source_asset": str(args.source_dir / "keypoint_accuracy.png"),
                "panel": "B",
                "description": "YOLO-pose held-out keypoint accuracy summary",
            },
        ]
    ).to_csv(args.table_dir / "fig2_yolo_pose_quality_metrics.csv", index=False)


def main() -> None:
    args = parse_args()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    render_full_image(args.source_dir / "input_stream_decomposition_mouse061.png", args.figure_dir / "fig1_framework_streams")
    render_pose_quality(args.source_dir, args.figure_dir)
    write_metadata(args)
    print(f"Wrote {args.figure_dir / 'fig1_framework_streams.png'}")
    print(f"Wrote {args.figure_dir / 'fig2_yolo_pose_quality.png'}")


if __name__ == "__main__":
    main()
