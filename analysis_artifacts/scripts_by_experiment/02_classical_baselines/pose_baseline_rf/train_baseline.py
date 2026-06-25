#!/usr/bin/env python3
"""Train RF and XGBoost baselines on extracted pose features.

Trains classifiers in two modes:
  1. Per-frame (132 features) — no temporal context
  2. Sliding-window (132 × (2K+1) features) — concatenated neighbor frames

Model selection uses validation macro-F1 over behavior classes (attack,
investigation, mount), matching the BehaviorScope-Y LSTM criterion.

Usage:
    cd pose_baseline_rf
    python train_baseline.py
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("WARNING: xgboost not installed. Only RF will be trained.")

THIS_DIR = Path(__file__).resolve().parent
CONFIG = yaml.safe_load((THIS_DIR / "config.yaml").read_text(encoding="utf-8"))
FEATURE_DIR = Path(CONFIG["output_root"]) / "features"
MODEL_DIR = Path(CONFIG["output_root"]) / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = CONFIG["evaluation"]["classes"]
BEHAVIOR_CLASSES = CONFIG["evaluation"]["behavior_classes"]
BEHAVIOR_IDX = [CLASSES.index(c) for c in BEHAVIOR_CLASSES]


# ─── Temporal context augmentation ──────────────────────────────────────────

def add_temporal_context(X: np.ndarray, half_window: int,
                         window_boundaries: list[int]) -> np.ndarray:
    """Concatenate ±half_window neighboring frames around each frame.

    Respects window boundaries: padding with zeros at the edges of each
    NPZ window (not across windows, since adjacent windows may come from
    different videos or non-contiguous regions).

    Args:
        X: [N_frames, D] feature matrix
        half_window: K, so context is 2K+1 frames
        window_boundaries: cumulative frame counts marking window starts
            e.g., [0, 32, 64, ...] if each window has 32 frames

    Returns:
        X_aug: [N_frames, D * (2K+1)]
    """
    if half_window <= 0:
        return X

    N, D = X.shape
    K = int(half_window)
    width = 2 * K + 1
    X_aug = np.zeros((N, D * width), dtype=X.dtype)

    # Build (start, end) pairs for each NPZ window
    segments = []
    for i in range(len(window_boundaries) - 1):
        segments.append((window_boundaries[i], window_boundaries[i + 1]))

    for seg_start, seg_end in segments:
        seg_len = seg_end - seg_start
        for local_i in range(seg_len):
            global_i = seg_start + local_i
            for offset_idx, offset in enumerate(range(-K, K + 1)):
                neighbor = local_i + offset
                if 0 <= neighbor < seg_len:
                    src = seg_start + neighbor
                    X_aug[global_i, offset_idx * D:(offset_idx + 1) * D] = X[src]
                # else: stays zero (padding)

    return X_aug


def build_window_boundaries(meta_path: Path) -> list[int]:
    """Build cumulative frame-count boundaries from window metadata."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    boundaries = [0]
    for entry in meta:
        boundaries.append(boundaries[-1] + entry["n_frames"])
    return boundaries


# ─── Macro-F1 over behavior classes only ────────────────────────────────────

def behavior_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro-F1 over attack, investigation, mount only.

    Matches BehaviorScope-Y's convention: only classes present in ground
    truth contribute to the macro average.
    """
    per_class_f1 = f1_score(y_true, y_pred, labels=list(range(len(CLASSES))),
                            average=None, zero_division=0.0)
    present = [i for i in BEHAVIOR_IDX if np.any(y_true == i)]
    if not present:
        return 0.0
    return float(np.mean([per_class_f1[i] for i in present]))


# ─── Train a single classifier ─────────────────────────────────────────────

def train_rf(X_train, y_train, X_val, y_val, tag: str) -> dict:
    """Train Random Forest, evaluate on validation, save model."""
    cfg = CONFIG["random_forest"]
    print(f"\n  Training RF ({tag}) ...")
    print(f"    n_estimators={cfg['n_estimators']}, "
          f"max_features={cfg['max_features']}, "
          f"class_weight={cfg['class_weight']}")

    clf = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_leaf=cfg["min_samples_leaf"],
        max_features=cfg["max_features"],
        class_weight=cfg["class_weight"],
        n_jobs=cfg["n_jobs"],
        random_state=cfg["random_state"],
        verbose=1,
    )

    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"    Training time: {train_time:.1f}s")

    y_val_pred = clf.predict(X_val)
    val_f1 = behavior_macro_f1(y_val, y_val_pred)
    val_acc = float(np.mean(y_val == y_val_pred))

    y_train_pred = clf.predict(X_train)
    train_f1 = behavior_macro_f1(y_train, y_train_pred)

    print(f"    Train macro-F1 (behavior): {train_f1:.4f}")
    print(f"    Val   macro-F1 (behavior): {val_f1:.4f}")
    print(f"    Val   accuracy:            {val_acc:.4f}")

    model_path = MODEL_DIR / f"rf_{tag}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    -> {model_path}")

    return {
        "model": "RandomForest",
        "tag": tag,
        "train_time_s": round(train_time, 1),
        "train_macro_f1": round(train_f1, 4),
        "val_macro_f1": round(val_f1, 4),
        "val_accuracy": round(val_acc, 4),
        "n_estimators": cfg["n_estimators"],
        "model_path": str(model_path),
    }


def train_xgb(X_train, y_train, X_val, y_val, tag: str) -> dict:
    """Train XGBoost with early stopping, evaluate on validation, save model."""
    if not HAS_XGBOOST:
        print(f"\n  Skipping XGBoost ({tag}): package not installed")
        return {}

    cfg = CONFIG["xgboost"]
    print(f"\n  Training XGBoost ({tag}) ...")
    print(f"    n_estimators={cfg['n_estimators']}, "
          f"max_depth={cfg['max_depth']}, "
          f"early_stopping={cfg['early_stopping_rounds']}")

    clf = XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        eval_metric=cfg["eval_metric"],
        early_stopping_rounds=cfg["early_stopping_rounds"],
        n_jobs=cfg["n_jobs"],
        random_state=cfg["random_state"],
        num_class=len(CLASSES),
        objective="multi:softprob",
        verbosity=1,
    )

    t0 = time.time()
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    train_time = time.time() - t0
    best_round = clf.best_iteration if hasattr(clf, "best_iteration") else cfg["n_estimators"]
    print(f"    Training time: {train_time:.1f}s  (best round: {best_round})")

    y_val_pred = clf.predict(X_val)
    val_f1 = behavior_macro_f1(y_val, y_val_pred)
    val_acc = float(np.mean(y_val == y_val_pred))

    y_train_pred = clf.predict(X_train)
    train_f1 = behavior_macro_f1(y_train, y_train_pred)

    print(f"    Train macro-F1 (behavior): {train_f1:.4f}")
    print(f"    Val   macro-F1 (behavior): {val_f1:.4f}")
    print(f"    Val   accuracy:            {val_acc:.4f}")

    model_path = MODEL_DIR / f"xgb_{tag}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"    -> {model_path}")

    return {
        "model": "XGBoost",
        "tag": tag,
        "train_time_s": round(train_time, 1),
        "train_macro_f1": round(train_f1, 4),
        "val_macro_f1": round(val_f1, 4),
        "val_accuracy": round(val_acc, 4),
        "best_round": best_round,
        "model_path": str(model_path),
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("pose_baseline_rf — Training")
    print("=" * 70)

    # ── Load features ───────────────────────────────────────────────────
    print("\nLoading features ...")
    train_data = np.load(FEATURE_DIR / "train_features.npz")
    val_data = np.load(FEATURE_DIR / "val_features.npz")
    X_train_raw = train_data["X"]
    y_train = train_data["y"].astype(np.int32)
    X_val_raw = val_data["X"]
    y_val = val_data["y"].astype(np.int32)

    print(f"  Train: {X_train_raw.shape}  Val: {X_val_raw.shape}")

    # NaN/Inf cleanup
    X_train_raw = np.nan_to_num(X_train_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_raw = np.nan_to_num(X_val_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Load window boundaries for temporal context ─────────────────────
    train_val_meta_path = FEATURE_DIR / "train_val_window_meta.json"
    train_val_meta = json.loads(train_val_meta_path.read_text(encoding="utf-8"))

    # Separate train/val window metadata to build boundaries per split
    train_metas = [m for m in train_val_meta if m["split"] == "train"]
    val_metas = [m for m in train_val_meta if m["split"] == "val"]

    train_boundaries = [0]
    for m in train_metas:
        train_boundaries.append(train_boundaries[-1] + m["n_frames"])
    val_boundaries = [0]
    for m in val_metas:
        val_boundaries.append(val_boundaries[-1] + m["n_frames"])

    # ── Train for each temporal context setting ─────────────────────────
    results = []
    half_windows = CONFIG["temporal_context"]["half_windows"]

    for K in half_windows:
        tag = f"perframe" if K == 0 else f"context{2*K+1}"
        print(f"\n{'─' * 60}")
        print(f"Temporal context: K={K} ({tag})")
        print(f"{'─' * 60}")

        if K == 0:
            X_train = X_train_raw
            X_val = X_val_raw
        else:
            print(f"  Adding ±{K} frame context ...")
            X_train = add_temporal_context(X_train_raw, K, train_boundaries)
            X_val = add_temporal_context(X_val_raw, K, val_boundaries)

        print(f"  Feature dim: {X_train.shape[1]}")

        # Random Forest
        rf_result = train_rf(X_train, y_train, X_val, y_val, tag)
        if rf_result:
            results.append(rf_result)

        # XGBoost
        xgb_result = train_xgb(X_train, y_train, X_val, y_val, tag)
        if xgb_result:
            results.append(xgb_result)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Training Summary")
    print(f"{'=' * 70}")
    print(f"{'Model':<12} {'Tag':<14} {'Val macro-F1':>12} {'Val acc':>10} {'Time':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['model']:<12} {r['tag']:<14} {r['val_macro_f1']:>12.4f} "
              f"{r['val_accuracy']:>10.4f} {r['train_time_s']:>7.1f}s")

    summary_path = Path(CONFIG["output_root"]) / "results" / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  -> {summary_path}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
