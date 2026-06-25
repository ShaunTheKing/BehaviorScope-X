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
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path("outputs")
ROUTES = {
    "YOLO-SPPF": ROOT / "eval_feature_cache" / "mars_stride16",
    "MobileNetV3": ROOT / "controlled_comparison_runs" / "run_20260528_205727" / "feature_caches" / "mobilenetv3_native" / "heldout",
    "DLC-HRNet": ROOT / "npz_cache" / "mars_dlc_topdown_hrnet_feature_cache_heldout",
}
OUT = Path("descriptor_mechanism_exploratory_fast")
CLASS_ORDER = ["attack", "investigation", "mount", "other"]
LABEL_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}
RNG = np.random.default_rng(42)


def safe_name(route: str) -> str:
    return route.lower().replace("/", "_").replace("-", "_")


def parse_sample_id_from_name(path: Path) -> str | None:
    stem = path.stem
    m = re.match(r"^(attack|investigation|mount|other)_(.+)_([0-9a-f]{12})$", stem)
    return m.group(2) if m else None


def file_index(root: Path) -> dict[str, Path]:
    out = {}
    for p in root.glob("*.npz"):
        sid = parse_sample_id_from_name(p)
        if sid is not None:
            out[sid] = p
    return out


def label_from_path(path: Path) -> str:
    for label in CLASS_ORDER:
        if path.name.startswith(label + "_"):
            return label
    raise ValueError(f"Cannot parse label from {path.name}")


def video_from_sample_id(sample_id: str) -> str:
    return re.sub(r"_f\d+$", "", sample_id)


def start_frame_from_sample_id(sample_id: str) -> int:
    m = re.search(r"_f(\d+)$", sample_id)
    return int(m.group(1)) if m else -1


def load_visual(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        group = np.asarray(z["group_feat"], dtype=np.float32)
        animal = np.asarray(z["animal_feat"], dtype=np.float32)
    framewise = np.concatenate([group, animal[:, 0, :], animal[:, 1, :]], axis=1)
    return framewise.mean(axis=0), framewise


def load_matrix(route: str, index: dict[str, Path], sample_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    vectors = []
    frame_accum = None
    frame_count = 0
    for sid in sample_ids:
        vec, framewise = load_visual(index[sid])
        vectors.append(vec)
        if frame_accum is None:
            frame_accum = np.zeros((framewise.shape[0], framewise.shape[1]), dtype=np.float64)
            frame_sq_accum = np.zeros_like(frame_accum)
        frame_accum += framewise
        frame_sq_accum += framewise * framewise
        frame_count += 1
    X = np.vstack(vectors).astype(np.float32, copy=False)
    frame_var = (frame_sq_accum / frame_count) - (frame_accum / frame_count) ** 2
    return X, frame_var.astype(np.float32)


def balanced_indices(labels: np.ndarray, max_per_class: int = 500) -> np.ndarray:
    buckets = defaultdict(list)
    for i, y in enumerate(labels):
        buckets[int(y)].append(i)
    n = min(max_per_class, min(len(v) for v in buckets.values()))
    chosen = []
    for y in sorted(buckets):
        vals = np.array(buckets[y], dtype=int)
        chosen.extend(RNG.choice(vals, size=n, replace=False).tolist())
    return np.array(sorted(chosen), dtype=int)


def pca_summary(X: np.ndarray, max_components: int = 128) -> tuple[dict[str, float], PCA, np.ndarray]:
    Xs = StandardScaler().fit_transform(X)
    ncomp = min(max_components, Xs.shape[0] - 1, Xs.shape[1])
    pca = PCA(n_components=ncomp, svd_solver="randomized", random_state=42)
    scores = pca.fit_transform(Xs)
    ev = pca.explained_variance_ratio_
    c = np.cumsum(ev)
    total = float(pca.explained_variance_.sum())
    pr = float(total * total / np.square(pca.explained_variance_).sum())
    out = {
        "effective_dim_top128_participation_ratio": pr,
        "pc50": int(np.searchsorted(c, 0.50) + 1) if c[-1] >= 0.50 else np.nan,
        "pc80": int(np.searchsorted(c, 0.80) + 1) if c[-1] >= 0.80 else np.nan,
        "pc90": int(np.searchsorted(c, 0.90) + 1) if c[-1] >= 0.90 else np.nan,
        "explained_by_top16": float(c[min(15, len(c) - 1)]),
        "explained_by_top32": float(c[min(31, len(c) - 1)]),
        "explained_by_top64": float(c[min(63, len(c) - 1)]),
    }
    return out, pca, scores


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
    return float((between / max(1, len(np.unique(y)) - 1)) / (within / max(1, len(y) - len(np.unique(y)))))


def top_channels_by_fisher(X: np.ndarray, y: np.ndarray, top_n: int = 30) -> pd.DataFrame:
    Xs = StandardScaler().fit_transform(X)
    rows = []
    for j in range(Xs.shape[1]):
        vals = Xs[:, j]
        overall = vals.mean()
        between = 0.0
        within = 0.0
        for cls in np.unique(y):
            vc = vals[y == cls]
            mu = vc.mean()
            between += len(vc) * (mu - overall) ** 2
            within += np.sum((vc - mu) ** 2)
        denom = within / max(1, len(y) - len(np.unique(y)))
        score = np.nan if denom <= 1e-12 else (between / max(1, len(np.unique(y)) - 1)) / denom
        rows.append((j, score, float(vals.var())))
    df = pd.DataFrame(rows, columns=["channel_index", "fisher_ratio", "z_variance"])
    return df.sort_values("fisher_ratio", ascending=False).head(top_n)


def grouped_probe(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    n_splits = min(5, len(np.unique(groups)))
    f1s, accs = [], []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        ncomp = min(64, X.shape[1], len(tr) - 1)
        pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=ncomp, svd_solver="randomized", random_state=42),
            RidgeClassifier(class_weight="balanced"),
        )
        pipe.fit(X[tr], y[tr])
        pred = pipe.predict(X[te])
        f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
        accs.append(accuracy_score(y[te], pred))
    return {
        "probe_macro_f1_mean": float(np.mean(f1s)),
        "probe_macro_f1_sd": float(np.std(f1s, ddof=1)),
        "probe_accuracy_mean": float(np.mean(accs)),
        "probe_accuracy_sd": float(np.std(accs, ddof=1)),
        "n_group_folds": n_splits,
    }


def linear_cka(X: np.ndarray, Y: np.ndarray, sample_idx: np.ndarray) -> float:
    Xs = StandardScaler().fit_transform(X[sample_idx])
    Ys = StandardScaler().fit_transform(Y[sample_idx])
    numerator = np.linalg.norm(Xs.T @ Ys, ord="fro") ** 2
    denominator = np.linalg.norm(Xs.T @ Xs, ord="fro") * np.linalg.norm(Ys.T @ Ys, ord="fro")
    return float(numerator / denominator)


def write_leverage(route: str, scores: np.ndarray, sample_ids: list[str], labels: np.ndarray):
    # Leverage is squared standardized PC score across the top PCs.
    k = min(16, scores.shape[1])
    leverage = np.sum(scores[:, :k] ** 2, axis=1)
    df = pd.DataFrame({
        "sample_id": sample_ids,
        "video_id": [video_from_sample_id(s) for s in sample_ids],
        "start_frame": [start_frame_from_sample_id(s) for s in sample_ids],
        "label": [CLASS_ORDER[i] for i in labels],
        "top16_pc_leverage": leverage,
    })
    df.sort_values("top16_pc_leverage", ascending=False).head(100).to_csv(
        OUT / f"{safe_name(route)}_top_window_leverage.csv", index=False
    )
    video = df.groupby("video_id", as_index=False).agg(
        n_windows=("sample_id", "size"),
        mean_top16_pc_leverage=("top16_pc_leverage", "mean"),
        total_top16_pc_leverage=("top16_pc_leverage", "sum"),
    )
    video.sort_values("mean_top16_pc_leverage", ascending=False).to_csv(
        OUT / f"{safe_name(route)}_video_leverage.csv", index=False
    )


def main():
    OUT.mkdir(exist_ok=True)
    indices = {route: file_index(path) for route, path in ROUTES.items()}
    all_common_ids = sorted(set.intersection(*(set(idx) for idx in indices.values())))
    all_labels = np.array([LABEL_TO_INT[label_from_path(indices["DLC-HRNet"][sid])] for sid in all_common_ids], dtype=int)
    all_groups = np.array([video_from_sample_id(sid) for sid in all_common_ids])
    selected = balanced_indices(all_labels, max_per_class=500)
    common_ids = [all_common_ids[i] for i in selected]
    labels = all_labels[selected]
    groups = all_groups[selected]
    bal = np.arange(len(common_ids), dtype=int)

    meta = {
        "n_common_samples_available": len(all_common_ids),
        "n_samples_loaded": len(common_ids),
        "n_videos_loaded": int(len(np.unique(groups))),
        "class_counts_available": {CLASS_ORDER[k]: int(v) for k, v in Counter(all_labels).items()},
        "class_counts_balanced": {CLASS_ORDER[k]: int(v) for k, v in Counter(labels[bal]).items()},
        "routes": {r: str(p) for r, p in ROUTES.items()},
        "unit": "matched held-out behavior window; balanced sample for exploratory descriptor analysis",
        "vectorization": "visual only; concatenate group, animal 1, animal 2 descriptors per frame, then average across 32 frames",
        "caveat": "feature channels are route-local and not semantically aligned across backbones; video/window leverage is computed on the balanced loaded subset",
    }
    (OUT / "analysis_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    matrices = {}
    frame_vars = {}
    for route, idx in indices.items():
        print(f"Loading {route}: {len(common_ids)} matched windows", flush=True)
        matrices[route], frame_vars[route] = load_matrix(route, idx, common_ids)

    summaries = []
    for route, X in matrices.items():
        print(f"Analyzing {route}", flush=True)
        Xb, yb, gb = X[bal], labels[bal], groups[bal]
        pca_stats, pca, scores = pca_summary(Xb)
        probe = grouped_probe(Xb, yb, gb)
        emb2 = scores[:, :2]
        sil = silhouette_score(emb2, yb) if len(np.unique(yb)) > 1 else np.nan
        row = {
            "route": route,
            "n_common_samples_available": len(all_common_ids),
            "n_samples_loaded": len(common_ids),
            "n_features": X.shape[1],
            "fisher_ratio_balanced": fisher_ratio(Xb, yb),
            "silhouette_pca2_balanced": float(sil),
        }
        row.update(pca_stats)
        row.update(probe)
        summaries.append(row)

        write_leverage(route, scores, [common_ids[i] for i in bal], yb)
        top_channels_by_fisher(Xb, yb).to_csv(OUT / f"{safe_name(route)}_top_behavior_separating_channels.csv", index=False)

        fv = frame_vars[route]
        frame_summary = pd.DataFrame({
            "frame_index_in_window": np.arange(fv.shape[0]),
            "total_feature_variance": fv.sum(axis=1),
            "mean_feature_variance": fv.mean(axis=1),
        }).sort_values("total_feature_variance", ascending=False)
        frame_summary.to_csv(OUT / f"{safe_name(route)}_frame_position_variance.csv", index=False)

        embed = pd.DataFrame({
            "pc1": emb2[:, 0],
            "pc2": emb2[:, 1],
            "label": [CLASS_ORDER[i] for i in yb],
            "video_id": [video_from_sample_id(common_ids[i]) for i in bal],
            "sample_id": [common_ids[i] for i in bal],
        })
        embed.to_csv(OUT / f"{safe_name(route)}_pca2_embedding_balanced.csv", index=False)

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "visual_descriptor_summary.csv", index=False)

    cka_sample = balanced_indices(labels, max_per_class=250)
    cka_rows = []
    for r1, X1 in matrices.items():
        for r2, X2 in matrices.items():
            cka_rows.append({"route_1": r1, "route_2": r2, "linear_cka_balanced": linear_cka(X1, X2, cka_sample)})
    pd.DataFrame(cka_rows).to_csv(OUT / "visual_descriptor_linear_cka.csv", index=False)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x = np.arange(len(summary))
    ax.bar(x - 0.2, summary["fisher_ratio_balanced"], width=0.4, label="Fisher ratio")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, summary["probe_macro_f1_mean"], width=0.4, color="#777777", label="Linear probe macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["route"], rotation=20, ha="right")
    ax.set_ylabel("Behavior Fisher ratio")
    ax2.set_ylabel("Grouped linear-probe macro-F1")
    ax.set_title("Visual descriptor behavior structure")
    fig.tight_layout()
    fig.savefig(OUT / "visual_descriptor_behavior_structure.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(summary["route"], summary["effective_dim_top128_participation_ratio"])
    ax.set_ylabel("Effective dimensionality, top 128 PCs")
    ax.set_title("Visual descriptor dimensionality")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(OUT / "visual_descriptor_effective_dimensionality.png", dpi=200)
    plt.close(fig)

    print(summary.to_string(index=False))
    print(f"Outputs written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
