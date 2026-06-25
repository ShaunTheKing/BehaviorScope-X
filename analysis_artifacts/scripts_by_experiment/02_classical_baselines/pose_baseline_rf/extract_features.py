#!/usr/bin/env python3
"""Extract flat per-frame pose features from BehaviorScope-Y NPZ windows.

Reads the windowed NPZ cache (produced by prepare_full_video_npz.py) and
extracts the same pose_self + relational features used by the LSTM's
pose_relations_only stream. Saves flat arrays ready for RF / XGBoost training.

Per-frame feature vector (132-d for 2 animals, 7 keypoints):
  pose_self per animal:  49 × 2 = 98
    - centered coords:      K×3 = 21
    - pairwise distances:   K(K-1)/2 = 21
    - velocities:           K = 7
  relational features:   11 × 2 directed pairs = 22
  reliability scalars:   4 × 2 animals = 8   (pose_conf, crop_conf, track_conf, track_age)
  masks:                 2 × 2 animals = 4   (animal_mask, pose_mask as float)

Usage:
    cd pose_baseline_rf
    python extract_features.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ─── Setup paths ────────────────────────────────────────────────────────────
THIS_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((THIS_DIR / "config.yaml").read_text(encoding="utf-8"))

SHARED_SCRIPTS = Path(CONFIG["shared_scripts"])
if str(SHARED_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SHARED_SCRIPTS))

from utils.pose_features import extract_rich_pose_features  # noqa: E402

OUTPUT_DIR = Path(CONFIG["output_root"]) / "features"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ─── Feature extraction from a single NPZ ──────────────────────────────────

def extract_frame_features(npz_path: str | Path) -> np.ndarray:
    """Extract a [T, 132] flat feature matrix from one NPZ window.

    Returns float32 array of shape [T, feature_dim].
    """
    d = np.load(str(npz_path))

    T = int(d["animal_keypoints"].shape[0])
    N = int(d["n_animals"])
    assert N == 2, f"Expected 2 animals, got {N}"

    # ── 1. Pose-self features: extract_rich_pose_features per animal ────
    kpts = d["animal_keypoints"]  # [T, N, K, 3]  crop-relative normalized
    pose_feats = []
    for a in range(N):
        # extract_rich_pose_features expects [T, K, 3] → returns [T, D_pose]
        pf = extract_rich_pose_features(kpts[:, a, :, :])  # [T, 49]
        pose_feats.append(pf)
    pose_self = np.concatenate(pose_feats, axis=1)  # [T, 98]

    # ── 2. Relational features: directed pairs (0→1) and (1→0) ─────────
    rel = d["relation_features"]  # [T, N, N, 11]
    rel_01 = rel[:, 0, 1, :]     # [T, 11]
    rel_10 = rel[:, 1, 0, :]     # [T, 11]
    relational = np.concatenate([rel_01, rel_10], axis=1)  # [T, 22]

    # ── 3. Reliability scalars ──────────────────────────────────────────
    reliability = np.stack([
        d["pose_conf"][:, 0],     # [T]
        d["pose_conf"][:, 1],
        d["crop_conf"][:, 0],
        d["crop_conf"][:, 1],
        d["track_conf"][:, 0],
        d["track_conf"][:, 1],
        d["track_age"][:, 0],
        d["track_age"][:, 1],
    ], axis=1)  # [T, 8]

    # ── 4. Boolean masks as float ───────────────────────────────────────
    masks = np.stack([
        d["animal_mask"][:, 0].astype(np.float32),
        d["animal_mask"][:, 1].astype(np.float32),
        d["pose_mask"][:, 0].astype(np.float32),
        d["pose_mask"][:, 1].astype(np.float32),
    ], axis=1)  # [T, 4]

    # ── Concatenate ─────────────────────────────────────────────────────
    features = np.concatenate([pose_self, relational, reliability, masks], axis=1)
    return features.astype(np.float32)  # [T, 132]


# ─── Process a full manifest split ──────────────────────────────────────────

def process_manifest(manifest_path: str | Path, label: str) -> dict:
    """Extract features from all windows in a manifest.

    Returns dict with keys: X, y, window_meta
      X: [total_frames, 132] float32
      y: [total_frames] int16  (per-frame label from window's dominant class)
      window_meta: list of dicts with source_video, start_frame, end_frame, n_frames
    """
    manifest_path = Path(manifest_path)
    manifest_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    all_X = []
    all_y = []
    window_meta = []

    splits = manifest["splits"]
    n_total = sum(len(v) for v in splits.values())
    done = 0
    t0 = time.time()

    for split_name, samples in splits.items():
        for sample in samples:
            npz_path = manifest_dir / sample["sequence_npz"]
            if not npz_path.is_file():
                print(f"  WARNING: missing {npz_path}, skipping")
                continue

            features = extract_frame_features(npz_path)  # [T, 132]
            T = features.shape[0]

            # Per-frame labels: use the window's dominant class for all frames
            # This matches how the LSTM training dataloader labels windows
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

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"  [{label}] done: {X.shape[0]} frames, {X.shape[1]} features, "
          f"{len(window_meta)} windows")
    return {"X": X, "y": y, "window_meta": window_meta}


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("pose_baseline_rf — Feature Extraction")
    print("=" * 70)

    # ── Training + validation features ──────────────────────────────────
    train_manifest = Path(CONFIG["npz_cache"]["train_manifest"])
    print(f"\n[1/2] Processing training manifest: {train_manifest}")
    train_data = process_manifest(train_manifest, "train+val")

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

    print(f"\n  Train: {X_train.shape[0]} frames  "
          f"(attack={np.sum(y_train==0)}, invest={np.sum(y_train==1)}, "
          f"mount={np.sum(y_train==2)}, other={np.sum(y_train==3)})")
    print(f"  Val:   {X_val.shape[0]} frames  "
          f"(attack={np.sum(y_val==0)}, invest={np.sum(y_val==1)}, "
          f"mount={np.sum(y_val==2)}, other={np.sum(y_val==3)})")

    out_train = OUTPUT_DIR / "train_features.npz"
    np.savez_compressed(out_train, X=X_train, y=y_train)
    print(f"  -> {out_train}")

    out_val = OUTPUT_DIR / "val_features.npz"
    np.savez_compressed(out_val, X=X_val, y=y_val)
    print(f"  -> {out_val}")

    # ── Held-out features (preserve per-window structure for evaluation) ─
    heldout_manifest = Path(CONFIG["npz_cache"]["heldout_manifest"])
    print(f"\n[2/2] Processing held-out manifest: {heldout_manifest}")
    heldout_data = process_manifest(heldout_manifest, "heldout")

    out_heldout = OUTPUT_DIR / "heldout_features.npz"
    np.savez_compressed(out_heldout, X=heldout_data["X"], y=heldout_data["y"])
    print(f"  -> {out_heldout}")

    # Save window metadata for evaluation (need to reconstruct per-video predictions)
    meta_path = OUTPUT_DIR / "heldout_window_meta.json"
    meta_path.write_text(
        json.dumps(heldout_data["window_meta"], indent=2),
        encoding="utf-8",
    )
    print(f"  -> {meta_path}")

    # Also save train+val window metadata
    train_val_meta_path = OUTPUT_DIR / "train_val_window_meta.json"
    train_val_meta_path.write_text(
        json.dumps(train_data["window_meta"], indent=2),
        encoding="utf-8",
    )
    print(f"  -> {train_val_meta_path}")

    # ── Summary ─────────────────────────────────────────────────────────
    summary = {
        "train_frames": int(X_train.shape[0]),
        "val_frames": int(X_val.shape[0]),
        "heldout_frames": int(heldout_data["X"].shape[0]),
        "feature_dim": int(X_train.shape[1]),
        "train_windows": int(train_mask.sum()),
        "val_windows": int(val_mask.sum()),
        "heldout_windows": len(heldout_data["window_meta"]),
    }
    summary_path = OUTPUT_DIR / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary}")
    print(f"  -> {summary_path}")
    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()
