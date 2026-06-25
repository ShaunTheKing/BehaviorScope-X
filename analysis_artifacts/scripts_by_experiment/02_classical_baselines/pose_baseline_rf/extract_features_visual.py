#!/usr/bin/env python3
"""Extract pose + visual features from BehaviorScope-Y NPZ windows.

Extends the 132-dim pose-only feature vector with 768 dims of frozen
YOLO-pose backbone SPPF visual descriptors:
  - group_feat:  256-dim  (scene-level feature, shared across animals)
  - animal_feat: 512-dim  (2 animals Ã— 256-dim per-animal crop features)

Total per-frame feature vector: 132 (pose) + 256 (group) + 512 (animal) = 900

This answers the external question: do the YOLO-pose backbone's learned
visual representations help classical ML classifiers (RF/XGBoost), or only
the temporal LSTM?

Usage:
    cd pose_baseline_rf
    python extract_features_visual.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# â”€â”€â”€ Setup paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
THIS_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((THIS_DIR / "config.yaml").read_text(encoding="utf-8"))

SHARED_SCRIPTS = Path(CONFIG["shared_scripts"])
if str(SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS))

from utils.pose_features import extract_rich_pose_features  # noqa: E402

OUTPUT_DIR = Path(CONFIG["output_root"]) / "features_visual"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# â”€â”€â”€ Visual cache filename mapping (from data_y.py) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _safe_cache_stem(text: str, max_len: int = 96) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return (stem[:max_len] or "sample")


def visual_cache_path(cache_dir: Path, sample_id: str,
                      start_frame: int, end_frame: int) -> Path:
    """Compute visual feature cache filename for a manifest sample.

    Replicates the logic from data_y.feature_cache_path().
    """
    key = f"{sample_id}|{start_frame}|{end_frame}"
    digest = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return cache_dir / f"{_safe_cache_stem(sample_id)}_{digest}.npz"


# â”€â”€â”€ Feature extraction from a single window â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def extract_pose_features(npz_path: str | Path) -> np.ndarray:
    """Extract [T, 132] pose-only features from one NPZ window.

    Identical to the pose-only extract_features.py.
    """
    d = np.load(str(npz_path))
    T = int(d["animal_keypoints"].shape[0])
    N = int(d["n_animals"])
    assert N == 2, f"Expected 2 animals, got {N}"

    # Pose-self per animal
    kpts = d["animal_keypoints"]  # [T, N, K, 3]
    pose_feats = []
    for a in range(N):
        pf = extract_rich_pose_features(kpts[:, a, :, :])  # [T, 49]
        pose_feats.append(pf)
    pose_self = np.concatenate(pose_feats, axis=1)  # [T, 98]

    # Relational features
    rel = d["relation_features"]  # [T, N, N, 11]
    relational = np.concatenate([rel[:, 0, 1, :], rel[:, 1, 0, :]], axis=1)  # [T, 22]

    # Reliability scalars
    reliability = np.stack([
        d["pose_conf"][:, 0], d["pose_conf"][:, 1],
        d["crop_conf"][:, 0], d["crop_conf"][:, 1],
        d["track_conf"][:, 0], d["track_conf"][:, 1],
        d["track_age"][:, 0], d["track_age"][:, 1],
    ], axis=1)  # [T, 8]

    # Masks
    masks = np.stack([
        d["animal_mask"][:, 0].astype(np.float32),
        d["animal_mask"][:, 1].astype(np.float32),
        d["pose_mask"][:, 0].astype(np.float32),
        d["pose_mask"][:, 1].astype(np.float32),
    ], axis=1)  # [T, 4]

    return np.concatenate([pose_self, relational, reliability, masks],
                          axis=1).astype(np.float32)  # [T, 132]


def load_visual_features(visual_npz_path: Path) -> np.ndarray:
    """Load and flatten visual features from a YOLO feature cache NPZ.

    Returns [T, 768] float32:  group_feat [T,256] + animal_feat [T,2,256] flattened.
    """
    vc = np.load(str(visual_npz_path))
    group_feat = vc["group_feat"]      # [T, 256]
    animal_feat = vc["animal_feat"]    # [T, 2, 256]

    T = group_feat.shape[0]
    animal_flat = animal_feat.reshape(T, -1)  # [T, 512]

    return np.concatenate([group_feat, animal_flat], axis=1).astype(np.float32)  # [T, 768]


def extract_pose_visual_features(
    npz_path: str | Path,
    visual_npz_path: Path,
) -> np.ndarray:
    """Extract [T, 900] pose + visual features from one window.

    Concatenates 132-dim pose features with 768-dim visual features.
    """
    pose = extract_pose_features(npz_path)      # [T, 132]
    visual = load_visual_features(visual_npz_path)  # [T, 768]

    assert pose.shape[0] == visual.shape[0], (
        f"Frame count mismatch: pose {pose.shape[0]} vs visual {visual.shape[0]}"
    )

    combined = np.concatenate([pose, visual], axis=1).astype(np.float32)  # [T, 900]
    assert combined.shape[1] == 900, (
        f"Expected 900-dim combined features, got {combined.shape[1]}")
    return combined


# â”€â”€â”€ Process a full manifest split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def process_manifest(manifest_path: str | Path, visual_cache_dir: str | Path,
                     label: str) -> dict:
    """Extract pose+visual features from all windows in a manifest.

    Returns dict with keys: X, y, window_meta
      X: [total_frames, 900] float32
      y: [total_frames] int16
      window_meta: list of dicts
    """
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    visual_cache_dir = Path(visual_cache_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    all_X = []
    all_y = []
    window_meta = []

    splits = manifest["splits"]
    n_total = sum(len(v) for v in splits.values())
    done = 0
    skipped_visual = 0
    t0 = time.time()

    for split_name, samples in splits.items():
        for sample in samples:
            npz_path = manifest_dir / sample["sequence_npz"]
            if not npz_path.is_file():
                print(f"  WARNING: missing window NPZ {npz_path}, skipping")
                continue

            # Compute visual cache path
            vc_path = visual_cache_path(
                visual_cache_dir,
                sample["id"],
                int(sample["start_frame"]),
                int(sample["end_frame"]),
            )
            if not vc_path.is_file():
                skipped_visual += 1
                if skipped_visual <= 5:
                    print(f"  WARNING: missing visual cache {vc_path.name}, skipping")
                continue

            features = extract_pose_visual_features(npz_path, vc_path)  # [T, 900]
            T = features.shape[0]

            label_idx = int(sample["label"])
            labels = np.full(T, label_idx, dtype=np.int16)

            all_X.append(features)
            all_y.append(labels)
            window_meta.append({
                "split": split_name,
                "source_video": sample.get("source_video", ""),
                "start_frame": int(sample["start_frame"]),
                "end_frame": int(sample["end_frame"]),
                "n_frames": T,
                "label": label_idx,
                "class_name": sample["class_name"],
                "sample_id": sample["id"],
            })

            done += 1
            if done % 500 == 0 or done == n_total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{label}] {done}/{n_total} windows "
                      f"({rate:.1f} win/s, {elapsed:.0f}s elapsed)")

    if skipped_visual > 5:
        print(f"  [{label}] WARNING: {skipped_visual} windows skipped "
              f"(missing visual cache)")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"  [{label}] done: {X.shape[0]} frames, {X.shape[1]} features, "
          f"{len(window_meta)} windows  ({skipped_visual} skipped)")
    return {"X": X, "y": y, "window_meta": window_meta}


# â”€â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    print("=" * 70)
    print("pose_baseline_rf â€” Visual Feature Extraction (pose + SPPF)")
    print("=" * 70)

    train_visual_cache = Path(CONFIG["visual_feature_cache"]["train"])
    heldout_visual_cache = Path(CONFIG["visual_feature_cache"]["heldout"])

    # Verify cache directories exist
    assert train_visual_cache.is_dir(), f"Train visual cache not found: {train_visual_cache}"
    assert heldout_visual_cache.is_dir(), f"Heldout visual cache not found: {heldout_visual_cache}"

    # â”€â”€ Training + validation features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    train_manifest = Path(CONFIG["npz_cache"]["train_manifest"])
    print(f"\n[1/2] Processing training manifest: {train_manifest}")
    print(f"       Visual cache: {train_visual_cache}")
    train_data = process_manifest(train_manifest, train_visual_cache, "train+val")

    # Split back into train / val using the manifest split keys
    train_mask = np.array([m["split"] == "train" for m in train_data["window_meta"]])
    val_mask = np.array([m["split"] == "val" for m in train_data["window_meta"]])

    # Build per-window frame indices
    frame_idx = 0
    train_frame_mask = np.zeros(len(train_data["y"]), dtype=bool)
    val_frame_mask = np.zeros(len(train_data["y"]), dtype=bool)
    for i, meta in enumerate(train_data["window_meta"]):
        n = meta["n_frames"]
        if train_mask[i]:
            train_frame_mask[frame_idx:frame_idx + n] = True
        else:
            val_frame_mask[frame_idx:frame_idx + n] = True
        frame_idx += n

    X_train = train_data["X"][train_frame_mask]
    y_train = train_data["y"][train_frame_mask]
    X_val = train_data["X"][val_frame_mask]
    y_val = train_data["y"][val_frame_mask]

    print(f"\n  Train: {X_train.shape[0]} frames Ã— {X_train.shape[1]} features  "
          f"(attack={np.sum(y_train==0)}, invest={np.sum(y_train==1)}, "
          f"mount={np.sum(y_train==2)}, other={np.sum(y_train==3)})")
    print(f"  Val:   {X_val.shape[0]} frames Ã— {X_val.shape[1]} features  "
          f"(attack={np.sum(y_val==0)}, invest={np.sum(y_val==1)}, "
          f"mount={np.sum(y_val==2)}, other={np.sum(y_val==3)})")

    out_train = OUTPUT_DIR / "train_features.npz"
    np.savez_compressed(out_train, X=X_train, y=y_train)
    print(f"  -> {out_train}")

    out_val = OUTPUT_DIR / "val_features.npz"
    np.savez_compressed(out_val, X=X_val, y=y_val)
    print(f"  -> {out_val}")

    # â”€â”€ Held-out features â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    heldout_manifest = Path(CONFIG["npz_cache"]["heldout_manifest"])
    print(f"\n[2/2] Processing held-out manifest: {heldout_manifest}")
    print(f"       Visual cache: {heldout_visual_cache}")
    heldout_data = process_manifest(heldout_manifest, heldout_visual_cache, "heldout")

    out_heldout = OUTPUT_DIR / "heldout_features.npz"
    np.savez_compressed(out_heldout, X=heldout_data["X"], y=heldout_data["y"])
    print(f"  -> {out_heldout}")

    # Save window metadata
    meta_path = OUTPUT_DIR / "heldout_window_meta.json"
    meta_path.write_text(
        json.dumps(heldout_data["window_meta"], indent=2), encoding="utf-8")
    print(f"  -> {meta_path}")

    train_val_meta_path = OUTPUT_DIR / "train_val_window_meta.json"
    train_val_meta_path.write_text(
        json.dumps(train_data["window_meta"], indent=2), encoding="utf-8")
    print(f"  -> {train_val_meta_path}")

    # â”€â”€ Summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    summary = {
        "feature_mode": "pose_visual",
        "pose_dim": 132,
        "visual_dim": 768,
        "total_dim": int(X_train.shape[1]),
        "train_frames": int(X_train.shape[0]),
        "val_frames": int(X_val.shape[0]),
        "heldout_frames": int(heldout_data["X"].shape[0]),
        "train_windows": int(train_mask.sum()),
        "val_windows": int(val_mask.sum()),
        "heldout_windows": len(heldout_data["window_meta"]),
    }
    summary_path = OUTPUT_DIR / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary}")
    print(f"  -> {summary_path}")
    print("\nVisual feature extraction complete.")


if __name__ == "__main__":
    main()
