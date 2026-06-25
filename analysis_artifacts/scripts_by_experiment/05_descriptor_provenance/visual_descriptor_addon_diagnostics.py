from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("outputs")
TABLES = Path("tables/production/mobilenet_portability")
ROUTES = {
    "YOLO-SPPF": ROOT / "eval_feature_cache" / "mars_stride16",
    "MobileNetV3": ROOT / "controlled_comparison_runs" / "run_20260528_205727" / "feature_caches" / "mobilenetv3_native" / "heldout",
    "DLC-HRNet": ROOT / "npz_cache" / "mars_dlc_topdown_hrnet_feature_cache_heldout",
}
OUT = Path("descriptor_mechanism_addon")
CLASS_ORDER = ["attack", "investigation", "mount", "other"]
LABEL_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}
COMPONENTS = ["combined", "group", "animal_pair"]
RNG = np.random.default_rng(123)


def parse_sample_id_from_name(path: Path) -> str | None:
    m = re.match(r"^(attack|investigation|mount|other)_(.+)_([0-9a-f]{12})$", path.stem)
    return m.group(2) if m else None


def file_index(root: Path) -> dict[str, Path]:
    out = {}
    if not root.exists():
        raise FileNotFoundError(root)
    for p in root.glob("*.npz"):
        sid = parse_sample_id_from_name(p)
        if sid is not None:
            out[sid] = p
    if not out:
        raise RuntimeError(f"No NPZ cache files indexed from {root}")
    return out


def label_from_path(path: Path) -> str:
    for label in CLASS_ORDER:
        if path.name.startswith(label + "_"):
            return label
    raise ValueError(f"Cannot parse label from {path.name}")


def video_from_sample_id(sample_id: str) -> str:
    return re.sub(r"_f\d+$", "", sample_id)


def balanced_ids(common_ids: list[str], labels: np.ndarray, max_per_class: int = 300) -> list[str]:
    buckets = defaultdict(list)
    for sid, y in zip(common_ids, labels):
        buckets[int(y)].append(sid)
    if set(buckets) != set(range(len(CLASS_ORDER))):
        raise RuntimeError(f"Missing classes in matched sample: {sorted(buckets)}")
    n = min(max_per_class, min(len(v) for v in buckets.values()))
    chosen = []
    for y in sorted(buckets):
        vals = np.array(buckets[y], dtype=object)
        chosen.extend(RNG.choice(vals, size=n, replace=False).tolist())
    return sorted(chosen)


def load_components(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        group = np.asarray(z["group_feat"], dtype=np.float32)
        animal = np.asarray(z["animal_feat"], dtype=np.float32)
    animal_pair_frames = np.concatenate([animal[:, 0, :], animal[:, 1, :]], axis=1)
    combined_frames = np.concatenate([group, animal_pair_frames], axis=1)
    return {
        "combined": combined_frames,
        "group": group,
        "animal_pair": animal_pair_frames,
    }


def load_route(route: str, index: dict[str, Path], sample_ids: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    vectors = {c: [] for c in COMPONENTS}
    smooth_rows = {c: [] for c in COMPONENTS}
    for sid in sample_ids:
        comps = load_components(index[sid])
        for comp, frames in comps.items():
            vectors[comp].append(frames.mean(axis=0))
            diffs = frames[1:] - frames[:-1]
            l2 = float(np.mean(np.linalg.norm(diffs, axis=1)))
            norms = np.linalg.norm(frames, axis=1)
            denom = norms[1:] * norms[:-1]
            valid = denom > 1e-8
            cosine = 1.0 - np.sum(frames[1:] * frames[:-1], axis=1)[valid] / denom[valid]
            smooth_rows[comp].append([l2, float(np.mean(cosine)) if cosine.size else np.nan])
    X = {comp: np.vstack(rows).astype(np.float32, copy=False) for comp, rows in vectors.items()}
    T = {comp: np.asarray(rows, dtype=np.float32) for comp, rows in smooth_rows.items()}
    return X, T


def fisher_ratio(X: np.ndarray, y: np.ndarray) -> float:
    Xs = StandardScaler().fit_transform(X)
    overall = Xs.mean(axis=0)
    between = 0.0
    within = 0.0
    for cls in np.unique(y):
        Xc = Xs[y == cls]
        mu = Xc.mean(axis=0)
        between += len(Xc) * np.sum((mu - overall) ** 2)
        within += np.sum((Xc - mu) ** 2)
    denom = within / max(1, len(y) - len(np.unique(y)))
    return float((between / max(1, len(np.unique(y)) - 1)) / denom) if denom > 0 else np.nan


def classifier_pipeline(X: np.ndarray, n_train: int):
    ncomp = min(64, X.shape[1], n_train - 1)
    return make_pipeline(
        StandardScaler(),
        PCA(n_components=ncomp, svd_solver="randomized", random_state=42),
        RidgeClassifier(class_weight="balanced"),
    )


def behavior_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[dict[str, float], np.ndarray, pd.DataFrame]:
    folds = GroupKFold(n_splits=min(5, len(np.unique(groups))))
    preds = np.full_like(y, -1)
    fold_scores = []
    for tr, te in folds.split(X, y, groups):
        pipe = classifier_pipeline(X, len(tr))
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        preds[te] = pred
        fold_scores.append(f1_score(y[te], pred, average="macro", zero_division=0))
    cm = confusion_matrix(y, preds, labels=list(range(len(CLASS_ORDER))))
    report = classification_report(
        y,
        preds,
        labels=list(range(len(CLASS_ORDER))),
        target_names=CLASS_ORDER,
        output_dict=True,
        zero_division=0,
    )
    class_rows = []
    for cls in CLASS_ORDER:
        class_rows.append({
            "class": cls,
            "precision": report[cls]["precision"],
            "recall": report[cls]["recall"],
            "f1": report[cls]["f1-score"],
            "support": report[cls]["support"],
        })
    summary = {
        "behavior_probe_macro_f1": float(f1_score(y, preds, average="macro", zero_division=0)),
        "behavior_probe_accuracy": float(accuracy_score(y, preds)),
        "behavior_probe_fold_macro_f1_mean": float(np.mean(fold_scores)),
        "behavior_probe_fold_macro_f1_sd": float(np.std(fold_scores, ddof=1)),
    }
    return summary, cm, pd.DataFrame(class_rows)


def video_probe(X: np.ndarray, video_codes: np.ndarray, min_count: int) -> dict[str, float]:
    if min_count < 3:
        return {"video_probe_accuracy": np.nan, "video_probe_balanced_accuracy": np.nan, "video_probe_n_classes": len(np.unique(video_codes))}
    n_splits = min(5, min_count)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = np.full_like(video_codes, -1)
    for tr, te in skf.split(X, video_codes):
        pipe = classifier_pipeline(X, len(tr))
        pipe.fit(X[tr], video_codes[tr])
        preds[te] = pipe.predict(X[te])
    return {
        "video_probe_accuracy": float(accuracy_score(video_codes, preds)),
        "video_probe_balanced_accuracy": float(balanced_accuracy_score(video_codes, preds)),
        "video_probe_n_classes": int(len(np.unique(video_codes))),
        "video_probe_chance": float(1 / len(np.unique(video_codes))),
    }


def per_video_fisher(X: np.ndarray, y: np.ndarray, videos: np.ndarray, min_per_video: int = 12) -> pd.DataFrame:
    rows = []
    for vid in sorted(np.unique(videos)):
        idx = np.flatnonzero(videos == vid)
        if len(idx) < min_per_video or len(np.unique(y[idx])) < 2:
            continue
        rows.append({
            "video_id": vid,
            "n_windows": int(len(idx)),
            "n_classes": int(len(np.unique(y[idx]))),
            "visual_fisher_ratio": fisher_ratio(X[idx], y[idx]),
        })
    return pd.DataFrame(rows)


def load_mobilenet_deltas() -> pd.DataFrame:
    path = TABLES / "mobilenet_portability_paired_video_deltas.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"video_id", "head", "seed", "delta_frame_macro_f1_present_behaviors", "delta_bout_macro_f1_iou25_present_behaviors"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Missing columns in {path}: {missing}")
    df = df[(df["head"] == "attention") & (df["seed"] == 42)].copy()
    if df.empty:
        raise RuntimeError("No attention seed 42 rows in MobileNet paired deltas")
    return df


def corr_rows(df: pd.DataFrame, xcols: list[str], ycols: list[str]) -> pd.DataFrame:
    rows = []
    for x in xcols:
        for y in ycols:
            sub = df[[x, y]].dropna()
            if len(sub) < 4:
                continue
            rows.append({
                "x": x,
                "y": y,
                "n_videos": len(sub),
                "pearson": float(sub[x].corr(sub[y], method="pearson")),
                "spearman": float(sub[x].corr(sub[y], method="spearman")),
            })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(exist_ok=True)
    indices = {route: file_index(path) for route, path in ROUTES.items()}
    all_common = sorted(set.intersection(*(set(v) for v in indices.values())))
    all_labels = np.array([LABEL_TO_INT[label_from_path(indices["DLC-HRNet"][sid])] for sid in all_common], dtype=int)
    sample_ids = balanced_ids(all_common, all_labels, max_per_class=300)
    y = np.array([LABEL_TO_INT[label_from_path(indices["DLC-HRNet"][sid])] for sid in sample_ids], dtype=int)
    videos = np.array([video_from_sample_id(sid) for sid in sample_ids])
    video_labels, video_codes = np.unique(videos, return_inverse=True)
    min_video_count = min(Counter(video_codes).values())

    meta = {
        "n_common_available": len(all_common),
        "n_loaded": len(sample_ids),
        "class_counts_loaded": {CLASS_ORDER[k]: int(v) for k, v in Counter(y).items()},
        "n_videos_loaded": int(len(video_labels)),
        "min_windows_per_video_loaded": int(min_video_count),
        "routes": {k: str(v) for k, v in ROUTES.items()},
        "scope": "visual descriptors only",
        "components": COMPONENTS,
    }
    (OUT / "addon_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    all_X = {}
    all_T = {}
    for route, idx in indices.items():
        print(f"Loading {route}: {len(sample_ids)} windows", flush=True)
        all_X[route], all_T[route] = load_route(route, idx, sample_ids)

    summary_rows = []
    class_metric_rows = []
    smooth_rows = []
    for route in ROUTES:
        for comp in COMPONENTS:
            print(f"Analyzing {route} / {comp}", flush=True)
            X = all_X[route][comp]
            behavior_summary, cm, class_df = behavior_probe(X, y, videos)
            nuisance = video_probe(X, video_codes, min_video_count)
            row = {
                "route": route,
                "component": comp,
                "n_features": X.shape[1],
                "visual_fisher_ratio": fisher_ratio(X, y),
            }
            row.update(behavior_summary)
            row.update(nuisance)
            row["behavior_minus_video_balanced_accuracy"] = row["behavior_probe_accuracy"] - row["video_probe_balanced_accuracy"]
            summary_rows.append(row)
            pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_csv(
                OUT / f"{route.lower().replace('-', '_')}_{comp}_behavior_probe_confusion.csv"
            )
            class_df.insert(0, "component", comp)
            class_df.insert(0, "route", route)
            class_metric_rows.append(class_df)

            T = all_T[route][comp]
            for cls_idx, cls in enumerate(CLASS_ORDER):
                mask = y == cls_idx
                smooth_rows.append({
                    "route": route,
                    "component": comp,
                    "class": cls,
                    "n_windows": int(mask.sum()),
                    "adjacent_l2_mean": float(np.nanmean(T[mask, 0])),
                    "adjacent_cosine_distance_mean": float(np.nanmean(T[mask, 1])),
                })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "visual_component_probe_summary.csv", index=False)
    pd.concat(class_metric_rows, ignore_index=True).to_csv(OUT / "visual_component_class_metrics.csv", index=False)
    pd.DataFrame(smooth_rows).to_csv(OUT / "visual_temporal_smoothness_by_class.csv", index=False)

    # Per-video descriptor separability versus production MobileNetV3 performance deltas.
    pv = []
    for route in ["YOLO-SPPF", "MobileNetV3"]:
        df = per_video_fisher(all_X[route]["combined"], y, videos)
        df["route"] = route
        pv.append(df)
    pv_long = pd.concat(pv, ignore_index=True)
    pv_long.to_csv(OUT / "per_video_visual_fisher_long.csv", index=False)
    pv_wide = pv_long.pivot(index="video_id", columns="route", values="visual_fisher_ratio").reset_index()
    pv_wide["mobilenet_minus_yolo_visual_fisher"] = pv_wide["MobileNetV3"] - pv_wide["YOLO-SPPF"]
    deltas = load_mobilenet_deltas()
    joined = deltas.merge(pv_wide, on="video_id", how="inner")
    joined.to_csv(OUT / "per_video_visual_fisher_vs_mobilenet_delta.csv", index=False)
    corr = corr_rows(
        joined,
        ["YOLO-SPPF", "MobileNetV3", "mobilenet_minus_yolo_visual_fisher"],
        ["delta_frame_macro_f1_present_behaviors", "delta_bout_macro_f1_iou25_present_behaviors"],
    )
    corr.to_csv(OUT / "per_video_fisher_delta_correlations.csv", index=False)

    # Figures.
    plot_df = summary[summary["component"].isin(["combined", "group", "animal_pair"])]
    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for i, comp in enumerate(COMPONENTS):
        sub = plot_df[plot_df["component"] == comp]
        x = np.arange(len(sub)) + (i - 1) * 0.22
        ax.bar(x, sub["behavior_probe_macro_f1"], width=0.22, label=comp)
    ax.set_xticks(np.arange(len(ROUTES)))
    ax.set_xticklabels(list(ROUTES.keys()), rotation=20, ha="right")
    ax.set_ylabel("Visual-only behavior probe macro-F1")
    ax.legend(frameon=False)
    ax.set_title("Group and animal visual descriptor probes")
    fig.tight_layout()
    fig.savefig(OUT / "group_animal_behavior_probe.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    combined = summary[summary["component"] == "combined"].copy()
    x = np.arange(len(combined))
    ax.bar(x - 0.18, combined["behavior_probe_accuracy"], width=0.36, label="Behavior accuracy")
    ax.bar(x + 0.18, combined["video_probe_balanced_accuracy"], width=0.36, label="Video balanced accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(combined["route"], rotation=20, ha="right")
    ax.set_ylabel("Probe accuracy")
    ax.legend(frameon=False)
    ax.set_title("Behavior signal versus video-identity signal")
    fig.tight_layout()
    fig.savefig(OUT / "behavior_vs_video_probe.png", dpi=200)
    plt.close(fig)

    if not joined.empty:
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        ax.scatter(joined["mobilenet_minus_yolo_visual_fisher"], joined["delta_frame_macro_f1_present_behaviors"])
        ax.axhline(0, color="#999999", linewidth=1)
        ax.axvline(0, color="#999999", linewidth=1)
        ax.set_xlabel("MobileNetV3 minus YOLO visual Fisher ratio")
        ax.set_ylabel("MobileNetV3 minus YOLO frame macro-F1")
        ax.set_title("Per-video visual separability and performance delta")
        fig.tight_layout()
        fig.savefig(OUT / "per_video_fisher_vs_performance_delta.png", dpi=200)
        plt.close(fig)

    report = [
        "# Visual Descriptor Add-On Diagnostics",
        "",
        "## Scope",
        "",
        "This exploratory analysis uses visual descriptors only. It evaluates group descriptors, paired animal descriptors, and combined descriptors on a balanced subset of matched held-out MARS windows.",
        "",
        f"- Matched windows available: {len(all_common)}",
        f"- Loaded windows: {len(sample_ids)}",
        f"- Videos represented: {len(video_labels)}",
        "",
        "## Outputs",
        "",
        "- `visual_component_probe_summary.csv`",
        "- `visual_component_class_metrics.csv`",
        "- `visual_temporal_smoothness_by_class.csv`",
        "- `per_video_visual_fisher_vs_mobilenet_delta.csv`",
        "- `per_video_fisher_delta_correlations.csv`",
        "- `*_behavior_probe_confusion.csv`",
        "- `group_animal_behavior_probe.png`",
        "- `behavior_vs_video_probe.png`",
        "- `per_video_fisher_vs_performance_delta.png`",
        "",
        "## Caveats",
        "",
        "Feature channels are route-local and are not homologous across backbones. Video-identity probes are nuisance diagnostics, not biological endpoints. Per-video correlations are exploratory and limited by the number of held-out videos represented in the sampled subset.",
    ]
    (OUT / "README_addon_diagnostics.md").write_text("\n".join(report), encoding="utf-8")

    print(summary.to_string(index=False))
    print(f"Outputs written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
