#!/usr/bin/env python3
"""Build controlled window-level matrices for RF/XGBoost reruns."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import StandardScaler

from controlled_common import (
    dump_json,
    dump_pickle,
    extract_pose_relation_frame_features,
    feature_cache_path_for_sample,
    iter_samples,
    load_config,
    load_manifest_samples,
    load_raw_visual_frame_features,
    output_root,
    pose_relation_dim,
    pose_visual_pca_dim,
    resolve_repo_path,
    sample_metadata,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create RF/XGBoost matrices with train-only PCA-compressed visual features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    p.add_argument("--max_samples_per_split", type=int, default=0,
                   help="Optional smoke-test cap; 0 means use all samples.")
    p.add_argument("--skip_visual", action="store_true",
                   help="Build only pose_frame and pose_window matrices.")
    return p.parse_args()


def _take_cap(samples: list, cap: int) -> list:
    return samples if int(cap) <= 0 else samples[: int(cap)]


def _visual_batches(samples: list, cache_dir: Path, cfg: dict, batch_size: int):
    buf: list[np.ndarray] = []
    n_buf = 0
    for sample in samples:
        cache_path = feature_cache_path_for_sample(cache_dir, sample, cfg)
        visual = load_raw_visual_frame_features(cache_path)
        buf.append(visual)
        n_buf += int(visual.shape[0])
        if n_buf >= int(batch_size):
            yield np.concatenate(buf, axis=0)
            buf = []
            n_buf = 0
    if buf:
        yield np.concatenate(buf, axis=0)


def fit_visual_pca(train_samples: list, cache_dir: Path, cfg: dict, pca_dir: Path) -> dict:
    n_components = int(cfg["pca"]["n_components"])
    batch_size = max(int(cfg["pca"].get("batch_size", 8192)), n_components)
    algorithm = str(cfg["pca"].get("algorithm", "incremental")).lower()

    scaler = StandardScaler(copy=True)
    n_frames = 0
    for batch in _visual_batches(train_samples, cache_dir, cfg, batch_size):
        scaler.partial_fit(batch)
        n_frames += int(batch.shape[0])
    if n_frames < n_components:
        raise RuntimeError(f"Need at least {n_components} train frames for PCA, got {n_frames}")

    if algorithm == "full":
        chunks = [batch for batch in _visual_batches(train_samples, cache_dir, cfg, batch_size)]
        raw = np.concatenate(chunks, axis=0)
        scaled = scaler.transform(raw)
        pca = PCA(n_components=n_components, svd_solver="full")
        pca.fit(scaled)
        skipped = 0
    else:
        pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        skipped = 0
        for batch in _visual_batches(train_samples, cache_dir, cfg, batch_size):
            scaled = scaler.transform(batch)
            if scaled.shape[0] < n_components:
                skipped += int(scaled.shape[0])
                continue
            pca.partial_fit(scaled)

    dump_pickle(pca_dir / "visual_scaler.pkl", scaler)
    dump_pickle(pca_dir / "visual_pca.pkl", pca)
    fit_source_videos = sorted({str(s.source_video) for s in train_samples})
    fit_class_counts: dict[str, int] = {}
    for sample in train_samples:
        fit_class_counts[str(sample.class_name)] = fit_class_counts.get(str(sample.class_name), 0) + 1
    meta = {
        "algorithm": algorithm,
        "n_components": n_components,
        "raw_visual_dim": int(cfg["features"]["visual_raw_dim_per_frame"]),
        "components_shape": list(getattr(pca, "components_", np.empty((0, 0))).shape),
        "explained_variance_ratio_sum": float(np.sum(getattr(pca, "explained_variance_ratio_", [0.0]))),
        "fit_split": "train",
        "fit_train_windows": int(len(train_samples)),
        "fit_train_frames_for_scaler": int(n_frames),
        "fit_train_source_video_count": int(len(fit_source_videos)),
        "fit_train_source_videos": fit_source_videos,
        "fit_train_class_counts": fit_class_counts,
        "incremental_pca_skipped_tail_frames": int(skipped),
    }
    dump_json(pca_dir / "visual_pca_metadata.json", meta)
    return {"scaler": scaler, "pca": pca, "metadata": meta}


def summarize_samples(split_name: str, samples: list, raw_lookup: dict) -> dict:
    class_counts: dict[str, int] = {}
    source_videos = sorted({str(s.source_video) for s in samples})
    dataset_splits = sorted({
        str(raw_lookup.get(s.id, {}).get("dataset_split", ""))
        for s in samples
        if str(raw_lookup.get(s.id, {}).get("dataset_split", ""))
    })
    mars_splits = sorted({
        str(raw_lookup.get(s.id, {}).get("mars_split", ""))
        for s in samples
        if str(raw_lookup.get(s.id, {}).get("mars_split", ""))
    })
    for sample in samples:
        class_counts[str(sample.class_name)] = class_counts.get(str(sample.class_name), 0) + 1
    return {
        "split": split_name,
        "n_windows": int(len(samples)),
        "n_source_videos": int(len(source_videos)),
        "source_videos": source_videos,
        "dataset_splits": dataset_splits,
        "mars_splits": mars_splits,
        "class_counts": class_counts,
    }


def transform_visual(cache_path: Path, scaler, pca) -> np.ndarray:
    raw = load_raw_visual_frame_features(cache_path)
    return pca.transform(scaler.transform(raw)).astype(np.float32)


def make_rows_for_sample(sample, cfg: dict, cache_dir: Path | None, scaler=None, pca=None) -> dict[str, np.ndarray]:
    pose = extract_pose_relation_frame_features(sample.sequence_npz, cfg)
    center = int(pose.shape[0] // 2)
    rows = {
        "pose_frame": pose[center],
        "pose_window": pose.reshape(-1),
    }
    if scaler is not None and pca is not None and cache_dir is not None:
        visual_pca = transform_visual(feature_cache_path_for_sample(cache_dir, sample, cfg), scaler, pca)
        full = np.concatenate([pose, visual_pca], axis=1).astype(np.float32)
        rows["pose_visual_frame_pca"] = full[center]
        rows["pose_visual_window_pca"] = full.reshape(-1)
    return rows


def build_split_matrices(
    split_name: str,
    samples: list,
    raw_lookup: dict,
    cfg: dict,
    out_dir: Path,
    feature_sets: list[str],
    cache_dir: Path | None,
    scaler=None,
    pca=None,
) -> dict:
    buckets = {fs: [] for fs in feature_sets}
    labels = []
    meta_rows = []
    t0 = time.time()
    for idx, sample in enumerate(samples, start=1):
        row_map = make_rows_for_sample(sample, cfg, cache_dir, scaler=scaler, pca=pca)
        for fs in feature_sets:
            buckets[fs].append(row_map[fs])
        labels.append(int(sample.label))
        meta_rows.append(sample_metadata(sample, raw_lookup))
        if idx % 1000 == 0 or idx == len(samples):
            rate = idx / max(time.time() - t0, 1e-6)
            print(f"[{split_name}] {idx}/{len(samples)} windows ({rate:.1f} win/s)", flush=True)

    y = np.asarray(labels, dtype=np.int16)
    summary = {"split": split_name, "n_windows": int(len(samples)), "feature_sets": {}}
    for fs, rows in buckets.items():
        X = np.stack(rows, axis=0).astype(np.float32)
        fs_dir = out_dir / fs
        fs_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fs_dir / f"{split_name}.npz", X=X, y=y)
        dump_json(fs_dir / f"{split_name}_metadata.json", meta_rows)
        summary["feature_sets"][fs] = {"shape": [int(X.shape[0]), int(X.shape[1])]}
    return summary


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    out = output_root(cfg)
    matrix_parent = "matrices_smoke" if int(args.max_samples_per_split) > 0 else "matrices"
    matrix_root = out / matrix_parent / str(cfg["features"].get("dataset", "mars"))
    pca_dir = out / "pca" / str(cfg["features"].get("dataset", "mars"))
    train_manifest = resolve_repo_path(cfg["paths"]["train_manifest"])
    heldout_manifest = resolve_repo_path(cfg["paths"]["heldout_manifest"])
    train_cache = resolve_repo_path(cfg["paths"]["train_feature_cache"])
    heldout_cache = resolve_repo_path(cfg["paths"]["heldout_feature_cache"])

    train_splits, class_to_idx, idx_to_class, train_manifest_meta, train_raw = load_manifest_samples(train_manifest, cfg)
    heldout_splits, _cti, _itc, heldout_manifest_meta, heldout_raw = load_manifest_samples(heldout_manifest, cfg)
    train_samples = _take_cap(list(iter_samples(train_splits, ["train"])), args.max_samples_per_split)
    val_samples = _take_cap(list(iter_samples(train_splits, ["val"])), args.max_samples_per_split)
    heldout_samples = _take_cap([s for rows in heldout_splits.values() for s in rows], args.max_samples_per_split)

    feature_sets = list(cfg["features"]["feature_sets"])
    if args.skip_visual:
        feature_sets = [fs for fs in feature_sets if "visual" not in fs]

    pca_bundle = None
    if any("visual" in fs for fs in feature_sets):
        pca_bundle = fit_visual_pca(train_samples, train_cache, cfg, pca_dir)

    summaries = []
    summaries.append(build_split_matrices(
        "train", train_samples, train_raw, cfg, matrix_root, feature_sets, train_cache,
        scaler=pca_bundle["scaler"] if pca_bundle else None,
        pca=pca_bundle["pca"] if pca_bundle else None,
    ))
    summaries.append(build_split_matrices(
        "val", val_samples, train_raw, cfg, matrix_root, feature_sets, train_cache,
        scaler=pca_bundle["scaler"] if pca_bundle else None,
        pca=pca_bundle["pca"] if pca_bundle else None,
    ))
    summaries.append(build_split_matrices(
        "heldout", heldout_samples, heldout_raw, cfg, matrix_root, feature_sets, heldout_cache,
        scaler=pca_bundle["scaler"] if pca_bundle else None,
        pca=pca_bundle["pca"] if pca_bundle else None,
    ))

    n_animals = int(cfg["features"]["n_animals"])
    n_keypoints = int(cfg["features"]["n_keypoints"])
    relation_dim = int(cfg["features"]["relation_dim"])
    pca_dim = int(cfg["features"]["visual_pca_dim_per_frame"])
    contract = {
        "pose_relation_dim_per_frame": pose_relation_dim(n_animals, n_keypoints, relation_dim),
        "pose_visual_pca_dim_per_frame": pose_visual_pca_dim(n_animals, n_keypoints, relation_dim, pca_dim),
        "train_manifest_window_size": train_manifest_meta.get("window_size"),
        "heldout_manifest_window_size": heldout_manifest_meta.get("window_size"),
        "split_provenance": {
            "train": summarize_samples("train", train_samples, train_raw),
            "val": summarize_samples("val", val_samples, train_raw),
            "heldout": summarize_samples("heldout", heldout_samples, heldout_raw),
            "notes": (
                "PCA/scaler are fit on the train split only. Held-out samples come from the "
                "heldout manifest; its internal split names may be train/val while mars_split "
                "records the true MARS test split."
            ),
        },
        "pca_metadata": pca_bundle["metadata"] if pca_bundle else None,
        "class_to_idx": class_to_idx,
        "idx_to_class": idx_to_class,
        "summaries": summaries,
    }
    dump_json(matrix_root / "matrix_build_summary.json", contract)
    print(f"[matrix-build] wrote matrices to {matrix_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
