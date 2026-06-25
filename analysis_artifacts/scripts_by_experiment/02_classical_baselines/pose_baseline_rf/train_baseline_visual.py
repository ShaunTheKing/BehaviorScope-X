#!/usr/bin/env python3
"""Train RF and XGBoost baselines on pose + visual features.

Per-frame only (K=0, 900 features). Temporal context is intentionally
omitted: at 900-dim/frame, context15 would produce 13,500 features —
introducing curse-of-dimensionality confounds that muddy the comparison.
Per-frame+visual vs per-frame pose-only cleanly isolates whether visual
features help non-LSTM classifiers.

Usage:
    cd pose_baseline_rf
    python train_baseline_visual.py
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
FEATURE_DIR = Path(CONFIG["output_root"]) / "features_visual"
MODEL_DIR = Path(CONFIG["output_root"]) / "models_visual"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = CONFIG["evaluation"]["classes"]
BEHAVIOR_CLASSES = CONFIG["evaluation"]["behavior_classes"]
BEHAVIOR_IDX = [CLASSES.index(c) for c in BEHAVIOR_CLASSES]


def behavior_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over attack, investigation, mount only."""
    per_class_f1 = f1_score(y_true, y_pred, labels=list(range(len(CLASSES))),
                            average=None, zero_division=0.0)
    present = [i for i in BEHAVIOR_IDX if np.any(y_true == i)]
    if not present:
        return 0.0
    return float(np.mean([per_class_f1[i] for i in present]))


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
        "feature_mode": "pose_visual",
        "feature_dim": int(X_train.shape[1]),
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
        "feature_mode": "pose_visual",
        "feature_dim": int(X_train.shape[1]),
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
    print("pose_baseline_rf — Training (pose + visual, per-frame only)")
    print("=" * 70)

    # ── Load visual features ────────────────────────────────────────────
    print("\nLoading pose+visual features ...")
    train_data = np.load(FEATURE_DIR / "train_features.npz")
    val_data = np.load(FEATURE_DIR / "val_features.npz")
    X_train = train_data["X"]
    y_train = train_data["y"].astype(np.int32)
    X_val = val_data["X"]
    y_val = val_data["y"].astype(np.int32)

    print(f"  Train: {X_train.shape}  Val: {X_val.shape}")
    expected_dim = 900
    assert X_train.shape[1] == expected_dim, (
        f"Expected {expected_dim}-dim features, got {X_train.shape[1]}")

    # NaN/Inf cleanup
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)

    # ── Per-frame only (K=0) ───────────────────────────────────────────
    tag = "perframe"
    print(f"\n{'─' * 60}")
    print(f"Per-frame pose+visual ({X_train.shape[1]} features)")
    print(f"{'─' * 60}")

    results = []

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
    print("Training Summary (pose + visual)")
    print(f"{'=' * 70}")
    print(f"{'Model':<12} {'Tag':<22} {'Dim':>5} {'Val macro-F1':>12} {'Val acc':>10} {'Time':>8}")
    print("-" * 75)
    for r in results:
        print(f"{r['model']:<12} {r['tag']:<22} {r['feature_dim']:>5} "
              f"{r['val_macro_f1']:>12.4f} {r['val_accuracy']:>10.4f} "
              f"{r['train_time_s']:>7.1f}s")

    summary_path = Path(CONFIG["output_root"]) / "results_visual" / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n  -> {summary_path}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
