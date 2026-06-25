#!/usr/bin/env python3
"""Collect controlled RF/XGBoost/TCN evaluation metrics into suite summaries."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
DEFAULT_SUITE_ROOT = ROOT / "outputs" / "controlled_comparison_suite"
DEFAULT_CLASSICAL_ROOT = ROOT / "outputs" / "pose_baseline_rf_controlled"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Summarize controlled comparison eval outputs against ground truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--suite_root", default=str(DEFAULT_SUITE_ROOT))
    p.add_argument("--classical_root", default=str(DEFAULT_CLASSICAL_ROOT))
    p.add_argument("--neural_names", nargs="*", default=None,
                   help="Neural eval run names to collect. Default: all directories under <suite_root>/neural_eval.")
    p.add_argument("--tcn_names", nargs="*", default=None,
                   help="Backward-compatible alias for old TCN-only eval roots.")
    return p.parse_args()


def read_csv(path: Path) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_metric_prefix(prefix: str, metrics: dict, row: dict) -> None:
    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            row[f"{prefix}{key}"] = value


def collect_pose_backend_provenance(suite_root: Path) -> list[dict]:
    suite_cfg = read_json(suite_root / "controlled_suite_config.frozen.json")
    contracts = suite_cfg.get("visual_backend_contracts", {})
    rows: list[dict] = []
    for backend, contract in sorted(contracts.items()):
        row = {
            "visual_backend": backend,
            "role": contract.get("role", ""),
            "pose_model_family": contract.get("pose_model_family", ""),
            "pose_model_backbone": contract.get("pose_model_backbone", ""),
            "pose_model_size": contract.get("pose_model_size", ""),
            "feature_layer": contract.get("feature_layer", ""),
            "pooling": contract.get("pooling", ""),
            "feature_dim_per_crop": contract.get("feature_dim_per_crop", ""),
            "raw_visual_dim_per_frame": contract.get("raw_visual_dim_per_frame", ""),
            "train_cache": contract.get("train_cache", ""),
            "heldout_cache": contract.get("heldout_cache", ""),
        }
        if backend == "mobilenetv3_native":
            cache_manifest = read_json(Path(str(contract.get("train_cache", ""))) / "cache_manifest.json")
            row["parameter_count"] = cache_manifest.get("parameter_count", "")
            row["pose_metrics_source"] = str(Path(str(contract.get("train_cache", ""))) / "cache_manifest.json")
            _flatten_metric_prefix("pose_", cache_manifest.get("checkpoint_metrics", {}), row)
        elif backend == "yolo_sppf":
            results_path = suite_root.parents[2] / "MARS_YOLO_Pose_Model" / "results.csv"
            # suite_root is <repo>/outputs/controlled_comparison_runs/run_... .
            if not results_path.is_file():
                results_path = ROOT / "MARS_YOLO_Pose_Model" / "results.csv"
            rows_csv = read_csv(results_path)
            if rows_csv:
                last = rows_csv[-1]
                row["pose_metrics_source"] = str(results_path)
                row["pose_box_map50"] = last.get("metrics/mAP50(B)", "")
                row["pose_box_map5095"] = last.get("metrics/mAP50-95(B)", "")
                row["pose_pose_map50"] = last.get("metrics/mAP50(P)", "")
                row["pose_pose_map5095"] = last.get("metrics/mAP50-95(P)", "")
                row["pose_precision"] = last.get("metrics/precision(P)", "")
                row["pose_recall"] = last.get("metrics/recall(P)", "")
        rows.append(row)
    return rows


def collect_pca_provenance(classical_root: Path) -> list[dict]:
    rows: list[dict] = []
    for meta_path in sorted((classical_root / "pca").glob("*/visual_pca_metadata.json")):
        meta = read_json(meta_path)
        row = {
            "dataset": meta_path.parent.name,
            "metadata_path": str(meta_path),
            "algorithm": meta.get("algorithm", ""),
            "n_components": meta.get("n_components", ""),
            "raw_visual_dim": meta.get("raw_visual_dim", ""),
            "components_shape": json.dumps(meta.get("components_shape", [])),
            "explained_variance_ratio_sum": meta.get("explained_variance_ratio_sum", ""),
            "fit_split": meta.get("fit_split", ""),
            "fit_train_windows": meta.get("fit_train_windows", ""),
            "fit_train_frames_for_scaler": meta.get("fit_train_frames_for_scaler", ""),
            "fit_train_source_video_count": meta.get("fit_train_source_video_count", ""),
            "fit_train_class_counts_json": json.dumps(meta.get("fit_train_class_counts", {}), sort_keys=True),
            "incremental_pca_skipped_tail_frames": meta.get("incremental_pca_skipped_tail_frames", ""),
        }
        rows.append(row)
    return rows


def as_float(row: dict, key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def parse_seed(model_name: str) -> tuple[str, str]:
    match = re.search(r"_seed(\d+)$", str(model_name))
    if not match:
        return str(model_name), ""
    return str(model_name)[: match.start()], match.group(1)


def neural_family(model_name: str) -> str:
    lower = str(model_name).lower()
    for head in ("lstm", "attention", "tcn"):
        if f"_{head}" in lower:
            return head
    return "neural"


def discover_eval_roots(
    suite_root: Path,
    classical_root: Path,
    neural_names: list[str] | None,
    tcn_names: list[str] | None,
) -> list[dict]:
    roots = []
    eval_parent = classical_root / "eval"
    if eval_parent.is_dir():
        for dataset_root in sorted(p for p in eval_parent.iterdir() if p.is_dir()):
            dataset = dataset_root.name
            for model_root in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
                model_name = model_root.name
                family = "rf" if model_name.startswith("rf_") else "xgb" if model_name.startswith("xgb_") else "classical"
                for feature_root in sorted(p for p in model_root.iterdir() if p.is_dir()):
                    roots.append({
                        "family": family,
                        "model": model_name,
                        "model_base": parse_seed(model_name)[0],
                        "seed": parse_seed(model_name)[1],
                        "dataset": dataset,
                        "feature_set": f"{dataset}/{feature_root.name}",
                        "eval_root": feature_root,
                    })
    neural_eval_root = suite_root / "neural_eval"
    names = neural_names
    if names is None and neural_eval_root.is_dir():
        names = [p.name for p in sorted(neural_eval_root.iterdir()) if p.is_dir()]
    for model_name in names or []:
        eval_root = neural_eval_root / model_name
        if eval_root.is_dir():
            roots.append({
                "family": neural_family(model_name),
                "model": model_name,
                "model_base": parse_seed(model_name)[0],
                "seed": parse_seed(model_name)[1],
                "dataset": "",
                "feature_set": "full_sequence",
                "eval_root": eval_root,
            })
    # Backward compatibility for plans created before neural_eval existed.
    tcn_eval_root = suite_root / "tcn_eval"
    for tcn_name in tcn_names or []:
        eval_root = tcn_eval_root / tcn_name
        if eval_root.is_dir():
            roots.append({
                "family": "tcn",
                "model": tcn_name,
                "model_base": parse_seed(tcn_name)[0],
                "seed": parse_seed(tcn_name)[1],
                "dataset": "",
                "feature_set": "full_sequence",
                "eval_root": eval_root,
            })
    return roots


def summarize_group(rows: list[dict], group_keys: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row.get(k, "") for k in group_keys)
        groups[key].append(row)

    out = []
    for key, group in sorted(groups.items()):
        total_frames = sum(as_float(r, "n_frames") for r in group)
        weighted_acc = (
            sum(as_float(r, "accuracy") * as_float(r, "n_frames") for r in group) / total_frames
            if total_frames > 0 else 0.0
        )
        n = len(group)
        row = {k: v for k, v in zip(group_keys, key)}
        row.update({
            "n_videos": n,
            "n_frames": int(total_frames),
            "accuracy_weighted_by_frames": weighted_acc,
            "frame_macro_f1_present_behaviors_mean": sum(as_float(r, "frame_macro_f1_present_behaviors") for r in group) / max(n, 1),
            "bout_macro_f1_iou10_present_behaviors_mean": sum(as_float(r, "bout_macro_f1_iou10_present_behaviors") for r in group) / max(n, 1),
            "bout_macro_f1_iou25_present_behaviors_mean": sum(as_float(r, "bout_macro_f1_iou25_present_behaviors") for r in group) / max(n, 1),
            "bout_macro_f1_iou50_present_behaviors_mean": sum(as_float(r, "bout_macro_f1_iou50_present_behaviors") for r in group) / max(n, 1),
            "gt_bouts_total": int(sum(as_float(r, "gt_bouts_total") for r in group)),
            "pred_bouts_total": int(sum(as_float(r, "pred_bouts_total") for r in group)),
        })
        out.append(row)
    return out


def summarize_across_seeds(model_summary_rows: list[dict]) -> list[dict]:
    metric_keys = [
        "accuracy_weighted_by_frames",
        "frame_macro_f1_present_behaviors_mean",
        "bout_macro_f1_iou10_present_behaviors_mean",
        "bout_macro_f1_iou25_present_behaviors_mean",
        "bout_macro_f1_iou50_present_behaviors_mean",
    ]
    groups = defaultdict(list)
    for row in model_summary_rows:
        seed = str(row.get("seed", ""))
        if not seed:
            continue
        key = (
            row.get("family", ""),
            row.get("model_base", ""),
            row.get("feature_set", ""),
            row.get("prediction_type", ""),
        )
        groups[key].append(row)
    out = []
    for key, group in sorted(groups.items()):
        row = {
            "family": key[0],
            "model_base": key[1],
            "feature_set": key[2],
            "prediction_type": key[3],
            "n_seeds": len(group),
            "seeds": ";".join(str(r.get("seed", "")) for r in group),
        }
        for metric in metric_keys:
            vals = [as_float(r, metric) for r in group]
            mean = sum(vals) / max(len(vals), 1)
            var = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1) if len(vals) > 1 else 0.0
            row[f"{metric}_seed_mean"] = mean
            row[f"{metric}_seed_sd"] = var ** 0.5
        out.append(row)
    return out


def main() -> int:
    args = parse_args()
    suite_root = Path(args.suite_root).resolve()
    classical_root = Path(args.classical_root).resolve()
    eval_roots = discover_eval_roots(suite_root, classical_root, args.neural_names, args.tcn_names)

    per_video_rows = []
    manifest = []
    for item in eval_roots:
        eval_root = Path(item["eval_root"])
        summary_csv = eval_root / "per_video_summary.csv"
        rows = read_csv(summary_csv)
        failures = read_csv(eval_root / "failures.csv")
        exclusions = read_csv(eval_root / "exclusions.csv")
        aggregate_path = eval_root / "aggregate_summary.json"
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8")) if aggregate_path.is_file() else {}
        manifest.append({
            "family": item["family"],
            "model": item["model"],
            "dataset": item.get("dataset", ""),
            "feature_set": item["feature_set"],
            "eval_root": str(eval_root),
            "per_video_summary_csv": str(summary_csv),
            "n_per_video_rows": len(rows),
            "n_failures": len(failures),
            "n_exclusions": len(exclusions),
            "aggregate_summary_json": str(aggregate_path) if aggregate_path.is_file() else "",
            "allow_pred_named_annots": aggregate.get("allow_pred_named_annots", False),
        })
        for row in rows:
            enriched = dict(item)
            enriched.pop("eval_root", None)
            enriched.update(row)
            enriched["eval_root"] = str(eval_root)
            per_video_rows.append(enriched)

    summary_rows = summarize_group(
        per_video_rows,
        ["family", "model", "model_base", "seed", "feature_set", "prediction_type"],
    )
    across_seed_rows = summarize_across_seeds(summary_rows)
    smoothed_summary = [r for r in summary_rows if r.get("prediction_type") == "smoothed_frames"]
    smoothed_across_seed = [r for r in across_seed_rows if r.get("prediction_type") == "smoothed_frames"]
    pose_backend_rows = collect_pose_backend_provenance(suite_root)
    pca_provenance_rows = collect_pca_provenance(classical_root)

    out_dir = suite_root / "summaries"
    write_csv(out_dir / "controlled_eval_manifest.csv", manifest)
    write_csv(out_dir / "controlled_per_video_summary.csv", per_video_rows)
    write_csv(out_dir / "controlled_metrics_summary.csv", summary_rows)
    write_csv(out_dir / "controlled_metrics_summary_smoothed_only.csv", smoothed_summary)
    write_csv(out_dir / "controlled_seed_summary.csv", across_seed_rows)
    write_csv(out_dir / "controlled_seed_summary_smoothed_only.csv", smoothed_across_seed)
    write_csv(out_dir / "pose_backbone_provenance.csv", pose_backend_rows)
    write_csv(out_dir / "pca_provenance.csv", pca_provenance_rows)
    payload = {
        "suite_root": str(suite_root),
        "classical_root": str(classical_root),
        "neural_names": args.neural_names,
        "tcn_names": args.tcn_names,
        "n_eval_roots": len(eval_roots),
        "n_per_video_rows": len(per_video_rows),
        "outputs": {
            "manifest": str(out_dir / "controlled_eval_manifest.csv"),
            "per_video": str(out_dir / "controlled_per_video_summary.csv"),
            "all_prediction_types": str(out_dir / "controlled_metrics_summary.csv"),
            "smoothed_only": str(out_dir / "controlled_metrics_summary_smoothed_only.csv"),
            "seed_summary": str(out_dir / "controlled_seed_summary.csv"),
            "seed_summary_smoothed_only": str(out_dir / "controlled_seed_summary_smoothed_only.csv"),
            "pose_backbone_provenance": str(out_dir / "pose_backbone_provenance.csv"),
            "pca_provenance": str(out_dir / "pca_provenance.csv"),
        },
    }
    (out_dir / "controlled_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[summary] eval_roots={len(eval_roots)} per_video_rows={len(per_video_rows)} out={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
