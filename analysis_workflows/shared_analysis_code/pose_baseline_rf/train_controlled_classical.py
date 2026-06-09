#!/usr/bin/env python3
"""Train controlled RF/XGBoost models from saved window-level matrices."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from controlled_common import dump_json, dump_pickle, load_config, output_root

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train controlled classical baselines.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    p.add_argument("--feature_sets", nargs="*", default=None,
                   help="Subset of feature sets to train; default reads config.yaml.")
    p.add_argument("--models", nargs="+", choices=["rf", "xgb"], default=["rf", "xgb"])
    p.add_argument("--seed", type=int, default=None,
                   help="Override RF/XGBoost random_state for this run.")
    p.add_argument("--run_tag", default=None,
                   help="Suffix for model filenames and summaries. Default: seed<seed> when --seed is set.")
    return p.parse_args()


def behavior_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str], behavior_classes: list[str]) -> float:
    labels = list(range(len(classes)))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0.0)
    behavior_idx = [classes.index(c) for c in behavior_classes]
    present = [idx for idx in behavior_idx if np.any(y_true == idx)]
    return float(np.mean([per_class[idx] for idx in present])) if present else 0.0


def load_split(matrix_root: Path, feature_set: str, split: str):
    data = np.load(matrix_root / feature_set / f"{split}.npz")
    X = np.nan_to_num(data["X"].astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = data["y"].astype(np.int32)
    return X, y


def _suffix(run_tag: str | None) -> str:
    return f"_{run_tag}" if run_tag else ""


def load_existing_summary(summary_path: Path) -> dict[tuple[str, str], dict]:
    if not summary_path.is_file():
        return {}
    try:
        import json

        rows = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        print(f"[train] could not read prior summary {summary_path}: {exc}", flush=True)
        return {}
    prior: dict[tuple[str, str], dict] = {}
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            model = str(row.get("model", ""))
            feature_set = str(row.get("feature_set", ""))
            if model and feature_set:
                prior[(model, feature_set)] = dict(row)
    return prior


def existing_model_row(
    *,
    prior: dict[tuple[str, str], dict],
    model: str,
    feature_set: str,
    model_path: Path,
    seed: int | None,
    run_tag: str | None,
) -> dict | None:
    if not model_path.is_file():
        return None
    row = dict(prior.get((model, feature_set), {}))
    row.update({
        "model": model,
        "feature_set": feature_set,
        "model_path": str(model_path),
        "seed": seed,
        "run_tag": run_tag,
        "skipped_existing": True,
    })
    row.setdefault("train_time_s", 0.0)
    print(f"[train] skip existing {model}:{feature_set} -> {model_path}", flush=True)
    return row


def train_rf(
    X_train,
    y_train,
    X_val,
    y_val,
    cfg: dict,
    model_dir: Path,
    feature_set: str,
    classes,
    behavior_classes,
    seed: int | None,
    run_tag: str | None,
    prior: dict[tuple[str, str], dict] | None = None,
) -> dict:
    model_path = model_dir / f"rf_{feature_set}{_suffix(run_tag)}.pkl"
    existing = existing_model_row(
        prior=prior or {},
        model="rf",
        feature_set=feature_set,
        model_path=model_path,
        seed=seed,
        run_tag=run_tag,
    )
    if existing is not None:
        return existing
    params = dict(cfg["random_forest"])
    if seed is not None:
        params["random_state"] = int(seed)
    clf = RandomForestClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=params["max_depth"],
        min_samples_leaf=int(params["min_samples_leaf"]),
        max_features=params["max_features"],
        class_weight=params["class_weight"],
        n_jobs=int(params["n_jobs"]),
        random_state=int(params["random_state"]),
        verbose=1,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    train_time = time.time() - t0
    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)
    dump_pickle(model_path, clf)
    return {
        "model": "rf",
        "feature_set": feature_set,
        "model_path": str(model_path),
        "train_time_s": round(train_time, 3),
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "val_shape": [int(X_val.shape[0]), int(X_val.shape[1])],
        "train_behavior_macro_f1": behavior_macro_f1(y_train, train_pred, classes, behavior_classes),
        "val_behavior_macro_f1": behavior_macro_f1(y_val, val_pred, classes, behavior_classes),
        "val_accuracy": float(np.mean(y_val == val_pred)),
        "params": params,
        "seed": seed,
        "run_tag": run_tag,
        "skipped_existing": False,
    }


def train_xgb(
    X_train,
    y_train,
    X_val,
    y_val,
    cfg: dict,
    model_dir: Path,
    feature_set: str,
    classes,
    behavior_classes,
    seed: int | None,
    run_tag: str | None,
    prior: dict[tuple[str, str], dict] | None = None,
) -> dict:
    model_path = model_dir / f"xgb_{feature_set}{_suffix(run_tag)}.pkl"
    existing = existing_model_row(
        prior=prior or {},
        model="xgb",
        feature_set=feature_set,
        model_path=model_path,
        seed=seed,
        run_tag=run_tag,
    )
    if existing is not None:
        return existing
    if not HAS_XGBOOST:
        raise RuntimeError("xgboost is not installed, but XGBoost training was requested.")
    params = dict(cfg["xgboost"])
    if seed is not None:
        params["random_state"] = int(seed)
    clf = XGBClassifier(
        n_estimators=int(params["n_estimators"]),
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        eval_metric=params["eval_metric"],
        early_stopping_rounds=int(params["early_stopping_rounds"]),
        n_jobs=int(params["n_jobs"]),
        random_state=int(params["random_state"]),
        num_class=len(classes),
        objective="multi:softprob",
        verbosity=1,
    )
    t0 = time.time()
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    train_time = time.time() - t0
    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)
    dump_pickle(model_path, clf)
    return {
        "model": "xgb",
        "feature_set": feature_set,
        "model_path": str(model_path),
        "train_time_s": round(train_time, 3),
        "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "val_shape": [int(X_val.shape[0]), int(X_val.shape[1])],
        "train_behavior_macro_f1": behavior_macro_f1(y_train, train_pred, classes, behavior_classes),
        "val_behavior_macro_f1": behavior_macro_f1(y_val, val_pred, classes, behavior_classes),
        "val_accuracy": float(np.mean(y_val == val_pred)),
        "best_iteration": int(getattr(clf, "best_iteration", int(params["n_estimators"]))),
        "params": params,
        "seed": seed,
        "run_tag": run_tag,
        "skipped_existing": False,
    }


def main() -> int:
    args = parse_args()
    cfg = load_config(args.config)
    out = output_root(cfg)
    dataset = str(cfg["features"].get("dataset", "mars"))
    matrix_root = out / "matrices" / dataset
    model_dir = out / "models" / dataset
    model_dir.mkdir(parents=True, exist_ok=True)
    classes = list(cfg["evaluation"]["classes"])
    behavior_classes = list(cfg["evaluation"]["behavior_classes"])
    feature_sets = args.feature_sets or list(cfg["features"]["feature_sets"])
    seed = int(args.seed) if args.seed is not None else None
    run_tag = str(args.run_tag) if args.run_tag else (f"seed{seed}" if seed is not None else None)
    summary_path = model_dir / f"training_summary{_suffix(run_tag)}.json"
    prior_summary = load_existing_summary(summary_path)

    results = []
    for feature_set in feature_sets:
        X_train, y_train = load_split(matrix_root, feature_set, "train")
        X_val, y_val = load_split(matrix_root, feature_set, "val")
        print(f"[train] {feature_set}: train={X_train.shape} val={X_val.shape} seed={seed}", flush=True)
        if "rf" in args.models:
            results.append(train_rf(
                X_train, y_train, X_val, y_val, cfg, model_dir, feature_set,
                classes, behavior_classes, seed, run_tag, prior_summary,
            ))
        if "xgb" in args.models and bool(cfg.get("xgboost", {}).get("enabled", True)):
            results.append(train_xgb(
                X_train, y_train, X_val, y_val, cfg, model_dir, feature_set,
                classes, behavior_classes, seed, run_tag, prior_summary,
            ))

    dump_json(summary_path, results)
    print(f"[train] wrote {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


