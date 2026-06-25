"""Shared utilities for controlled RF/XGBoost baseline reruns.

The controlled path is intentionally window-level: one feature row per
manifest sample, one label per row, and evaluation spreads each row's
probabilities over the same frame span used by the neural classifier.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
CONFIG_PATH = THIS_DIR / "config.yaml"


def load_config(config_path: Path | str = CONFIG_PATH) -> dict:
    cfg_path = Path(config_path)
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["_config_path"] = str(cfg_path.resolve())
    return cfg


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def add_shared_scripts_to_path(cfg: dict) -> Path:
    shared = resolve_repo_path(cfg["paths"]["shared_scripts"])
    if str(shared) not in sys.path:
        sys.path.insert(0, str(shared))
    return shared


def output_root(cfg: dict) -> Path:
    root = cfg.get("project", {}).get("output_root", "outputs/pose_baseline_rf_controlled")
    out = resolve_repo_path(root)
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_manifest_samples(manifest_path: Path, cfg: dict):
    add_shared_scripts_to_path(cfg)
    from data_y import load_n_manifest  # noqa: WPS433

    splits, class_to_idx, idx_to_class, meta = load_n_manifest(manifest_path)
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_lookup = {}
    for split_name, rows in raw.get("splits", {}).items():
        for row in rows:
            item = dict(row)
            item.setdefault("dataset_split", split_name)
            raw_lookup[str(item["id"])] = item
    return splits, class_to_idx, idx_to_class, meta, raw_lookup


def feature_cache_path_for_sample(cache_dir: Path, sample, cfg: dict) -> Path:
    add_shared_scripts_to_path(cfg)
    from data_y import feature_cache_path  # noqa: WPS433

    return feature_cache_path(cache_dir, sample)


def pose_relation_dim(n_animals: int, n_keypoints: int, relation_dim: int) -> int:
    per_animal_pose = 4 * n_keypoints + n_keypoints * (n_keypoints - 1) // 2
    animal_total = n_animals * (per_animal_pose + 4)
    relation_total = n_animals * (n_animals - 1) * (relation_dim + 2)
    return int(animal_total + relation_total)


def pose_visual_pca_dim(n_animals: int, n_keypoints: int, relation_dim: int, pca_dim: int) -> int:
    return int(pose_relation_dim(n_animals, n_keypoints, relation_dim) + pca_dim)


def extract_pose_relation_frame_features(npz_path: Path | str, cfg: dict) -> np.ndarray:
    """Return [T, D] explicit pose/relation features matching model_y inputs.

    Per animal: rich pose features plus pose_conf, crop_conf, track_conf,
    track_age. Per ordered non-self pair: 11 relation features plus
    relation_pose_conf and relation_present.
    """
    add_shared_scripts_to_path(cfg)
    from utils.pose_features_y import extract_per_animal_pose_features  # noqa: WPS433

    n_animals = int(cfg["features"]["n_animals"])
    relation_dim = int(cfg["features"]["relation_dim"])
    expected = int(cfg["features"]["expected_pose_relation_dim_per_frame"])

    with np.load(str(npz_path)) as d:
        kpts = d["animal_keypoints"].astype(np.float32)
        if kpts.shape[1] != n_animals:
            raise ValueError(f"{npz_path}: n_animals={kpts.shape[1]}, expected {n_animals}")
        pose_self = extract_per_animal_pose_features(kpts)
        parts: list[np.ndarray] = []
        for a in range(n_animals):
            reliability = np.stack(
                [
                    d["pose_conf"][:, a],
                    d["crop_conf"][:, a],
                    d["track_conf"][:, a],
                    d["track_age"][:, a],
                ],
                axis=1,
            ).astype(np.float32)
            parts.append(np.concatenate([pose_self[:, a, :], reliability], axis=1))

        rel = d["relation_features"].astype(np.float32)
        rel_pose_conf = d["relation_pose_conf"].astype(np.float32)
        rel_present = d["relation_present"].astype(np.float32)
        if rel.shape[-1] != relation_dim:
            raise ValueError(f"{npz_path}: relation_dim={rel.shape[-1]}, expected {relation_dim}")
        for i in range(n_animals):
            for j in range(n_animals):
                if i == j:
                    continue
                pair_rel = rel[:, i, j, :]
                pair_reliability = np.stack([rel_pose_conf[:, i, j], rel_present[:, i, j]], axis=1)
                parts.append(np.concatenate([pair_rel, pair_reliability], axis=1))

    features = np.concatenate(parts, axis=1).astype(np.float32)
    if features.shape[1] != expected:
        raise ValueError(f"{npz_path}: pose feature dim={features.shape[1]}, expected {expected}")
    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def load_raw_visual_frame_features(cache_path: Path | str) -> np.ndarray:
    """Return [T, 768] concat(group_feat, animal0_feat, animal1_feat)."""
    with np.load(str(cache_path)) as d:
        group = d["group_feat"].astype(np.float32)
        animal = d["animal_feat"].astype(np.float32)
    if animal.ndim != 3:
        raise ValueError(f"{cache_path}: animal_feat shape must be [T,N,D], got {animal.shape}")
    visual = np.concatenate([group, animal.reshape(animal.shape[0], -1)], axis=1)
    return np.nan_to_num(visual.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def sample_metadata(sample, raw_lookup: dict) -> dict:
    raw = raw_lookup.get(sample.id, {})
    return {
        "sample_id": str(sample.id),
        "label": int(sample.label),
        "class_name": str(sample.class_name),
        "source_video": str(sample.source_video),
        "start_frame": int(sample.start_frame),
        "end_frame": int(sample.end_frame),
        "fps": float(sample.fps),
        "dataset_split": str(raw.get("dataset_split", "")),
        "mars_split": str(raw.get("mars_split", "")),
        "sequence_npz": str(sample.sequence_npz),
    }


def dump_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def dump_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def iter_samples(splits: dict, split_names: Iterable[str]):
    for name in split_names:
        yield from splits.get(name, [])
