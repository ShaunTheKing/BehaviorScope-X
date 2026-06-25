"""Aggregate YOLO/SPPF LSTM and attention held-out analyses.

This production script reads the current controlled-comparison run outputs and
writes reproducibility-facing CSV tables plus manuscript-style summary figures.
Legacy scripts are reference-only; this script uses current run schemas and
explicit paths.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
STYLE_DIR = THIS_DIR.parent / "current_analysis"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import (  # noqa: E402
    BEHAVIOR_COLORS,
    HEAD_COLORS,
    STREAM_COLORS,
    STREAM_LABELS,
    STREAM_SHORT_LABELS,
    apply_style,
    save_figure,
    title_with_panel,
)


MODEL_RE = re.compile(
    r"^yolo_sppf_(?P<stream>.+)_(?P<head>lstm|attention)(?P<capacity>\d+)_seed(?P<seed>\d+)$"
)
STREAM_ORDER = [
    "full",
    "no_group_visual",
    "pose_relations_only",
    "no_relations",
    "animal_visual_only",
    "visual_only",
]
HEAD_ORDER = ["lstm", "attention"]
CAPACITY_ORDER = [256, 512, 896]
PREDICTION_ORDER = ["raw_windows", "smoothed_frames"]
METRIC_COLS = [
    "accuracy",
    "frame_macro_f1_present_behaviors",
    "bout_macro_f1_iou10_present_behaviors",
    "bout_macro_f1_iou25_present_behaviors",
    "bout_macro_f1_iou50_present_behaviors",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate current YOLO/SPPF LSTM+attention held-out evals."
    )
    parser.add_argument(
        "--suite_root",
        type=Path,
        default=Path("outputs/controlled_comparison_runs/run_20260528_205727"),
        help="Controlled comparison run root.",
    )
    parser.add_argument(
        "--table_dir",
        type=Path,
        default=Path("Manuscript_penultimate/tables/production/yolo_sppf_main"),
        help="Output directory for CSV tables.",
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=Path("Manuscript_penultimate/figures/production/yolo_sppf_main"),
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--prediction_type",
        default="smoothed_frames",
        choices=PREDICTION_ORDER,
        help="Prediction type used for primary summary figures.",
    )
    return parser.parse_args()


def parse_model_name(name: str) -> dict[str, object] | None:
    match = MODEL_RE.match(name)
    if not match:
        return None
    out = match.groupdict()
    out["capacity"] = int(out["capacity"])
    out["seed"] = int(out["seed"])
    out["model_name"] = name
    return out


def add_model_fields(df: pd.DataFrame, meta: dict[str, object]) -> pd.DataFrame:
    df = df.copy()
    for key, value in meta.items():
        df[key] = value
    return df


def read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def discover_models(suite_root: Path) -> list[dict[str, object]]:
    eval_root = suite_root / "neural_eval"
    models: list[dict[str, object]] = []
    for model_dir in sorted(eval_root.iterdir()):
        if not model_dir.is_dir():
            continue
        meta = parse_model_name(model_dir.name)
        if meta is None:
            continue
        if meta["head"] not in HEAD_ORDER:
            continue
        if meta["stream"] not in STREAM_ORDER:
            continue
        meta["eval_dir"] = model_dir
        meta["train_dir"] = suite_root / "neural_training" / model_dir.name
        models.append(meta)
    return models


def load_all_tables(suite_root: Path) -> dict[str, pd.DataFrame]:
    models = discover_models(suite_root)
    if not models:
        raise FileNotFoundError(f"No YOLO/SPPF LSTM/attention eval models under {suite_root}")

    per_video = []
    per_class = []
    bout_matches = []
    predicted_bouts = []
    ground_truth_bouts = []
    training_logs = []
    run_rows = []

    for meta in models:
        eval_dir = Path(meta["eval_dir"])
        train_dir = Path(meta["train_dir"])
        run_rows.append(
            {
                **{k: v for k, v in meta.items() if k not in {"eval_dir", "train_dir"}},
                **{f"aggregate_{k}": v for k, v in read_json(eval_dir / "aggregate_summary.json").items()},
            }
        )
        for file_name, sink in [
            ("per_video_summary.csv", per_video),
            ("per_class_metrics.csv", per_class),
            ("bout_matches.csv", bout_matches),
            ("predicted_bouts.csv", predicted_bouts),
            ("ground_truth_bouts.csv", ground_truth_bouts),
        ]:
            path = eval_dir / file_name
            if path.exists():
                sink.append(add_model_fields(pd.read_csv(path), meta))
        train_path = train_dir / "training_log.csv"
        if train_path.exists():
            training_logs.append(add_model_fields(pd.read_csv(train_path), meta))

    return {
        "run_inventory": pd.DataFrame(run_rows),
        "per_video": pd.concat(per_video, ignore_index=True),
        "per_class": pd.concat(per_class, ignore_index=True),
        "bout_matches": pd.concat(bout_matches, ignore_index=True),
        "predicted_bouts": pd.concat(predicted_bouts, ignore_index=True),
        "ground_truth_bouts": pd.concat(ground_truth_bouts, ignore_index=True),
        "training_logs": pd.concat(training_logs, ignore_index=True),
    }


def make_model_summary(per_video: pd.DataFrame, training_logs: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["model_name", "stream", "head", "capacity", "seed", "prediction_type"]
    agg = (
        per_video.groupby(group_cols, as_index=False)
        .agg(
            evaluated_videos=("video_id", "nunique"),
            total_frames=("n_frames", "sum"),
            gt_bouts_total=("gt_bouts_total", "sum"),
            pred_bouts_total=("pred_bouts_total", "sum"),
            **{f"mean_{c}": (c, "mean") for c in METRIC_COLS},
            **{f"sd_{c}": (c, "std") for c in METRIC_COLS},
        )
    )
    best_train = (
        training_logs.sort_values(["model_name", "val_macroF1", "epoch"], ascending=[True, False, True])
        .groupby("model_name", as_index=False)
        .first()[
            [
                "model_name",
                "epoch",
                "train_loss",
                "val_loss",
                "val_macroF1",
                "elapsed_s",
                "cpu_pct",
                "ram_gb_used",
                "gpu_util_pct",
                "vram_gb_used",
                "gpu_power_w",
            ]
        ]
        .rename(
            columns={
                "epoch": "best_val_epoch",
                "train_loss": "best_epoch_train_loss",
                "val_loss": "best_epoch_val_loss",
                "val_macroF1": "best_validation_macroF1",
                "elapsed_s": "best_epoch_elapsed_s",
            }
        )
    )
    return agg.merge(best_train, on="model_name", how="left")


def make_seed_summary(model_summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["stream", "head", "capacity", "prediction_type"]
    metrics = [c for c in model_summary.columns if c.startswith("mean_")]
    return (
        model_summary.groupby(group_cols, as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            evaluated_videos=("evaluated_videos", "mean"),
            **{f"{m}_seed_mean": (m, "mean") for m in metrics},
            **{f"{m}_seed_sd": (m, "std") for m in metrics},
        )
    )


def make_per_class_summary(per_class: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["stream", "head", "capacity", "seed", "prediction_type", "class"]
    value_cols = [
        "gt_frames",
        "pred_frames",
        "frame_precision",
        "frame_recall",
        "frame_f1",
        "bout_f1_iou10",
        "bout_f1_iou25",
        "bout_f1_iou50",
    ]
    out = (
        per_class.groupby(group_cols, as_index=False)
        .agg(
            evaluated_videos=("video_id", "nunique"),
            **{f"mean_{c}": (c, "mean") for c in value_cols},
            **{f"sum_{c}": (c, "sum") for c in ["gt_frames", "pred_frames"]},
        )
    )
    return out


def make_confusion_fingerprints(per_class: pd.DataFrame) -> pd.DataFrame:
    cm_cols = [c for c in per_class.columns if c.startswith("cm_")]
    rows = []
    group_cols = ["stream", "head", "capacity", "seed", "prediction_type"]
    for keys, sub in per_class.groupby(group_cols):
        key_row = dict(zip(group_cols, keys))
        for true_cls, cls_df in sub.groupby("class"):
            totals = cls_df[cm_cols].sum(numeric_only=True)
            total = float(totals.sum())
            row = {**key_row, "true_class": true_cls, "total_frames": total}
            for col in cm_cols:
                pred_cls = col.replace("cm_", "")
                count = float(totals[col])
                row[f"pred_{pred_cls}_frames"] = count
                row[f"pred_{pred_cls}_fraction"] = count / total if total > 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def write_tables(tables: dict[str, pd.DataFrame], table_dir: Path) -> dict[str, Path]:
    table_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for name, df in tables.items():
        path = table_dir / f"{name}.csv"
        df.to_csv(path, index=False)
        outputs[name] = path
    return outputs


def stream_order_index(values: pd.Series) -> pd.Series:
    order = {name: i for i, name in enumerate(STREAM_ORDER)}
    return values.map(order)


def plot_stream_ablation(seed_summary: pd.DataFrame, figure_dir: Path, prediction_type: str) -> None:
    apply_style()
    df = seed_summary[seed_summary["prediction_type"].eq(prediction_type)].copy()
    metric = "mean_frame_macro_f1_present_behaviors_seed_mean"
    err = "mean_frame_macro_f1_present_behaviors_seed_sd"
    bout = "mean_bout_macro_f1_iou25_present_behaviors_seed_mean"
    bout_err = "mean_bout_macro_f1_iou25_present_behaviors_seed_sd"

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=False, constrained_layout=True)
    x = np.arange(len(STREAM_ORDER))
    width = 0.36
    for ax, y_col, e_col, title, ylabel in [
        (axes[0], metric, err, "Held-out frame macro-F1", "Frame macro-F1"),
        (axes[1], bout, bout_err, "Held-out bout F1 @ TIoU 0.25", "Bout F1"),
    ]:
        for offset, head in [(-width / 2, "lstm"), (width / 2, "attention")]:
            sub = (
                df[df["head"].eq(head)]
                .sort_values("stream", key=stream_order_index)
                .drop_duplicates(["stream", "head", "capacity"], keep="last")
            )
            # Best capacity per stream/head for this headline view.
            sub = (
                df[df["head"].eq(head)]
                .sort_values(y_col, ascending=False)
                .groupby(["stream", "head"], as_index=False)
                .first()
                .sort_values("stream", key=stream_order_index)
            )
            ax.bar(
                x + offset,
                sub[y_col].to_numpy(),
                width,
                yerr=sub[e_col].fillna(0).to_numpy(),
                capsize=3,
                color=HEAD_COLORS[head],
                alpha=0.85,
                label=head.upper() if head == "lstm" else "Attention",
                edgecolor="white",
                linewidth=0.7,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([STREAM_SHORT_LABELS[s] for s in STREAM_ORDER], rotation=0)
        ax.set_ylabel(ylabel)
        title_with_panel(ax, "A" if ax is axes[0] else "B", title)
        ax.legend(frameon=False, ncol=2, loc="upper right")
    save_figure(fig, figure_dir / "fig4_yolo_stream_ablation_headline")
    plt.close(fig)


def plot_head_capacity(seed_summary: pd.DataFrame, figure_dir: Path, prediction_type: str) -> None:
    apply_style()
    df = seed_summary[
        seed_summary["prediction_type"].eq(prediction_type) & seed_summary["stream"].eq("full")
    ].copy()
    metric = "mean_frame_macro_f1_present_behaviors_seed_mean"
    bout = "mean_bout_macro_f1_iou25_present_behaviors_seed_mean"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), constrained_layout=True)
    x = np.arange(len(CAPACITY_ORDER))
    for ax, y_col, title, ylabel in [
        (axes[0], metric, "Full-stream frame macro-F1", "Frame macro-F1"),
        (axes[1], bout, "Full-stream bout F1 @ TIoU 0.25", "Bout F1"),
    ]:
        for head in HEAD_ORDER:
            sub = df[df["head"].eq(head)].sort_values("capacity")
            ax.plot(
                x,
                sub[y_col].to_numpy(),
                marker="o",
                linewidth=2.0,
                color=HEAD_COLORS[head],
                label=head.upper() if head == "lstm" else "Attention",
            )
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in CAPACITY_ORDER])
        ax.set_xlabel("Hidden dimension")
        ax.set_ylabel(ylabel)
        title_with_panel(ax, "A" if ax is axes[0] else "B", title)
        ax.legend(frameon=False)
    save_figure(fig, figure_dir / "fig6_yolo_temporal_head_capacity_full_stream")
    plt.close(fig)


def plot_training_stability(
    training_logs: pd.DataFrame,
    seed_summary: pd.DataFrame,
    figure_dir: Path,
    prediction_type: str,
) -> None:
    apply_style()
    train = training_logs.copy()
    train["capacity"] = train["capacity"].astype(int)
    full = train[train["stream"].eq("full")].copy()
    full_agg = (
        full.groupby(["head", "capacity", "epoch"], as_index=False)
        .agg(
            train_loss_mean=("train_loss", "mean"),
            val_loss_mean=("val_loss", "mean"),
            val_macroF1_mean=("val_macroF1", "mean"),
            val_macroF1_sd=("val_macroF1", "std"),
        )
        .sort_values(["head", "capacity", "epoch"])
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    for head in HEAD_ORDER:
        for capacity in CAPACITY_ORDER:
            sub = full_agg[full_agg["head"].eq(head) & full_agg["capacity"].eq(capacity)]
            alpha = {256: 1.0, 512: 0.68, 896: 0.42}[capacity]
            label = f"{head.upper() if head == 'lstm' else 'Attention'}-{capacity}"
            color = HEAD_COLORS[head]
            ax_a.plot(sub["epoch"], sub["train_loss_mean"], color=color, alpha=alpha, linewidth=2.0, label=label)
            ax_b.plot(sub["epoch"], sub["val_loss_mean"], color=color, alpha=alpha, linewidth=2.0, label=label)
            ax_c.plot(sub["epoch"], sub["val_macroF1_mean"], color=color, alpha=alpha, linewidth=2.0, label=label)
            ax_c.fill_between(
                sub["epoch"],
                sub["val_macroF1_mean"] - sub["val_macroF1_sd"].fillna(0),
                sub["val_macroF1_mean"] + sub["val_macroF1_sd"].fillna(0),
                color=color,
                alpha=0.08 * alpha,
                linewidth=0,
            )

    title_with_panel(ax_a, "A", "Full-stream training loss")
    title_with_panel(ax_b, "B", "Full-stream validation loss")
    title_with_panel(ax_c, "C", "Full-stream validation macro-F1")
    for ax in [ax_a, ax_b, ax_c]:
        ax.set_xlabel("Epoch")
        ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax_a.set_ylabel("Cross-entropy loss")
    ax_b.set_ylabel("Cross-entropy loss")
    ax_c.set_ylabel("Validation macro-F1")

    df = seed_summary[seed_summary["prediction_type"].eq(prediction_type)].copy()
    metric = "mean_frame_macro_f1_present_behaviors_seed_mean"
    best = (
        df.sort_values(metric, ascending=False)
        .groupby(["stream", "head"], as_index=False)
        .first()
        .sort_values("stream", key=stream_order_index)
    )
    x = np.arange(len(STREAM_ORDER))
    width = 0.36
    for offset, head in [(-width / 2, "lstm"), (width / 2, "attention")]:
        sub = best[best["head"].eq(head)].sort_values("stream", key=stream_order_index)
        ax_d.bar(
            x + offset,
            sub["mean_best_validation_macroF1_seed_mean"].to_numpy()
            if "mean_best_validation_macroF1_seed_mean" in sub.columns
            else sub[metric].to_numpy(),
            width,
            color=HEAD_COLORS[head],
            alpha=0.85,
            edgecolor="white",
            linewidth=0.7,
            label=head.upper() if head == "lstm" else "Attention",
        )
    title_with_panel(ax_d, "D", "Best held-out setting by stream")
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([STREAM_SHORT_LABELS[s] for s in STREAM_ORDER])
    ax_d.set_ylabel("Held-out frame macro-F1")
    ax_d.legend(frameon=False, ncol=2)

    save_figure(fig, figure_dir / "fig3_yolo_neural_training_stability")
    plt.close(fig)


def plot_confusion_fingerprints(confusion: pd.DataFrame, figure_dir: Path, prediction_type: str) -> None:
    apply_style()
    df = confusion[confusion["prediction_type"].eq(prediction_type)].copy()
    # Use best capacity/head per stream by diagonal behavior accuracy averaged over seeds.
    behavior_classes = ["attack", "investigation", "mount", "other"]
    diag_rows = []
    for keys, sub in df.groupby(["stream", "head", "capacity", "seed"]):
        score = []
        for cls in behavior_classes:
            row = sub[sub["true_class"].eq(cls)]
            if not row.empty:
                score.append(float(row.iloc[0].get(f"pred_{cls}_fraction", np.nan)))
        diag_rows.append({**dict(zip(["stream", "head", "capacity", "seed"], keys)), "diag_mean": np.nanmean(score)})
    best = (
        pd.DataFrame(diag_rows)
        .groupby(["stream", "head", "capacity"], as_index=False)["diag_mean"]
        .mean()
        .sort_values("diag_mean", ascending=False)
        .groupby("stream", as_index=False)
        .first()
    )

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    for ax, stream in zip(axes.ravel(), STREAM_ORDER):
        sel = best[best["stream"].eq(stream)].iloc[0]
        sub = df[
            df["stream"].eq(stream)
            & df["head"].eq(sel["head"])
            & df["capacity"].eq(int(sel["capacity"]))
        ]
        mat = np.zeros((4, 4), dtype=float)
        for i, true_cls in enumerate(behavior_classes):
            row = sub[sub["true_class"].eq(true_cls)]
            for j, pred_cls in enumerate(behavior_classes):
                mat[i, j] = row[f"pred_{pred_cls}_fraction"].mean() if not row.empty else np.nan
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels([c[:4] for c in behavior_classes], rotation=45, ha="right")
        ax.set_yticklabels([c[:4] for c in behavior_classes])
        ax.set_title(STREAM_LABELS[stream], fontsize=10)
        for i in range(4):
            for j in range(4):
                txt = "" if np.isnan(mat[i, j]) else f"{mat[i, j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8, color="white" if mat[i, j] > 0.55 else "#111827")
        ax.grid(False)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, label="Fraction of true-class frames")
    save_figure(fig, figure_dir / "supp_yolo_stream_confusion_fingerprints")
    plt.close(fig)


def print_headline(seed_summary: pd.DataFrame, prediction_type: str) -> None:
    df = seed_summary[seed_summary["prediction_type"].eq(prediction_type)].copy()
    metric = "mean_frame_macro_f1_present_behaviors_seed_mean"
    bout = "mean_bout_macro_f1_iou25_present_behaviors_seed_mean"
    best = df.sort_values(metric, ascending=False).head(12)
    cols = [
        "stream",
        "head",
        "capacity",
        "n_seeds",
        metric,
        "mean_frame_macro_f1_present_behaviors_seed_sd",
        bout,
        "mean_bout_macro_f1_iou25_present_behaviors_seed_sd",
    ]
    print("\nTop held-out configurations by frame macro-F1:")
    print(best[cols].to_string(index=False, float_format=lambda x: f"{x:.4f}"))


def main() -> None:
    args = parse_args()
    suite_root = args.suite_root.resolve()
    table_dir = args.table_dir.resolve()
    figure_dir = args.figure_dir.resolve()
    figure_dir.mkdir(parents=True, exist_ok=True)

    raw = load_all_tables(suite_root)
    model_summary = make_model_summary(raw["per_video"], raw["training_logs"])
    seed_summary = make_seed_summary(model_summary)
    per_class_summary = make_per_class_summary(raw["per_class"])
    confusion = make_confusion_fingerprints(raw["per_class"])

    tables = {
        "yolo_lstm_attention_run_inventory": raw["run_inventory"],
        "yolo_lstm_attention_per_video_long": raw["per_video"],
        "yolo_lstm_attention_per_class_long": raw["per_class"],
        "yolo_lstm_attention_bout_matches_long": raw["bout_matches"],
        "yolo_lstm_attention_predicted_bouts_long": raw["predicted_bouts"],
        "yolo_lstm_attention_ground_truth_bouts_long": raw["ground_truth_bouts"],
        "yolo_lstm_attention_training_logs_long": raw["training_logs"],
        "yolo_lstm_attention_model_summary": model_summary,
        "yolo_lstm_attention_seed_summary": seed_summary,
        "yolo_lstm_attention_per_class_summary": per_class_summary,
        "yolo_lstm_attention_confusion_fingerprints": confusion,
    }
    outputs = write_tables(tables, table_dir)

    plot_training_stability(raw["training_logs"], seed_summary, figure_dir, args.prediction_type)
    plot_stream_ablation(seed_summary, figure_dir, args.prediction_type)
    plot_head_capacity(seed_summary, figure_dir, args.prediction_type)
    plot_confusion_fingerprints(confusion, figure_dir, args.prediction_type)
    print_headline(seed_summary, args.prediction_type)
    print(f"\nWrote {len(outputs)} CSV tables to {table_dir}")
    print(f"Wrote figures to {figure_dir}")


if __name__ == "__main__":
    main()
