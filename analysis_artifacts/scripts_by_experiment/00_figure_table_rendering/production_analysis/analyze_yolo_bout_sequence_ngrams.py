"""Analyze behavior bout sequence n-grams across YOLO/SPPF full-stream heads."""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_DIR = SCRIPT_DIR.parent / "current_analysis"
sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import apply_style, save_figure  # noqa: E402


BEHAVIORS = ["attack", "investigation", "mount"]
MODEL_RE = re.compile(r"^yolo_sppf_full_(?P<head>lstm|attention)(?P<capacity>\d+)_seed(?P<seed>\d+)$")
HEAD_ORDER = ["attention", "lstm"]
CAPACITY_ORDER = [256, 512, 896]
TRACK_COLORS = {
    "Attention-256": "#10B981",
    "Attention-512": "#34D399",
    "Attention-896": "#6EE7B7",
    "LSTM-256": "#2271B2",
    "LSTM-512": "#60A5FA",
    "LSTM-896": "#93C5FD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--table_dir",
        type=Path,
        default=Path("Manuscript_penultimate/tables/production/yolo_sppf_main"),
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=Path("Manuscript_penultimate/figures/production/supplement"),
    )
    parser.add_argument("--prediction_type", default="smoothed_frames")
    parser.add_argument("--reference_model", default="yolo_sppf_full_attention256_seed42")
    parser.add_argument("--max_n", type=int, default=4)
    parser.add_argument("--output_name", default="supp_yolo_bout_sequence_ngram_audit")
    return parser.parse_args()


def load_inputs(table_dir: Path, prediction_type: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_video = pd.read_csv(table_dir / "yolo_lstm_attention_per_video_long.csv")
    per_video = per_video[per_video["prediction_type"].eq(prediction_type)].copy()
    cols = [
        "split",
        "video_id",
        "prediction_type",
        "class",
        "start_frame",
        "end_frame",
        "duration_frames",
        "model_name",
    ]
    gt = pd.read_csv(table_dir / "yolo_lstm_attention_ground_truth_bouts_long.csv.gz", usecols=cols)
    pred = pd.read_csv(table_dir / "yolo_lstm_attention_predicted_bouts_long.csv.gz", usecols=cols)
    gt = gt[gt["prediction_type"].eq(prediction_type)].copy()
    pred = pred[pred["prediction_type"].eq(prediction_type)].copy()
    for df in [gt, pred]:
        df["start_frame"] = pd.to_numeric(df["start_frame"], errors="coerce").astype(int)
        df["end_frame"] = pd.to_numeric(df["end_frame"], errors="coerce").astype(int)
    return per_video, gt, pred


def video_units(per_video: pd.DataFrame, reference_model: str) -> pd.DataFrame:
    units = (
        per_video[per_video["model_name"].eq(reference_model)][["split", "video_id"]]
        .drop_duplicates()
        .copy()
    )
    if units.empty:
        raise ValueError(f"No video units found for reference model {reference_model}")
    return units


def discover_models(per_video: pd.DataFrame) -> list[dict]:
    rows = []
    for model_name in sorted(per_video["model_name"].dropna().unique()):
        match = MODEL_RE.match(str(model_name))
        if not match:
            continue
        head = match.group("head")
        capacity = int(match.group("capacity"))
        seed = int(match.group("seed"))
        if capacity not in CAPACITY_ORDER:
            continue
        track = f"{head.capitalize() if head == 'attention' else 'LSTM'}-{capacity}"
        rows.append({"model_name": str(model_name), "head": head, "capacity": capacity, "seed": seed, "track": track})
    return sorted(rows, key=lambda r: (HEAD_ORDER.index(r["head"]), r["capacity"], r["seed"]))


def bout_sequence(table: pd.DataFrame, split: str, video_id: str, model_name: str | None) -> list[str]:
    sub = table[table["split"].eq(split) & table["video_id"].eq(video_id)].copy()
    if model_name is None:
        sub = sub.drop_duplicates(["split", "video_id", "class", "start_frame", "end_frame"]).copy()
    else:
        sub = sub[sub["model_name"].eq(model_name)].copy()
    sub = sub[sub["class"].isin(BEHAVIORS)].sort_values(["start_frame", "end_frame", "class"])
    return [str(x) for x in sub["class"].tolist()]


def ngrams(seq: list[str], n: int) -> Counter:
    if len(seq) < n:
        return Counter()
    return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))


def multiset_f1(gt_seq: list[str], pred_seq: list[str], n: int) -> dict:
    gt_counts = ngrams(gt_seq, n)
    pred_counts = ngrams(pred_seq, n)
    tp = sum(min(pred_counts[k], gt_counts.get(k, 0)) for k in pred_counts)
    pred_total = sum(pred_counts.values())
    gt_total = sum(gt_counts.values())
    precision = tp / pred_total if pred_total else np.nan
    recall = tp / gt_total if gt_total else np.nan
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else np.nan
    return {
        "n": n,
        "ngram_tp": tp,
        "pred_ngram_total": pred_total,
        "gt_ngram_total": gt_total,
        "ngram_precision": precision,
        "ngram_recall": recall,
        "ngram_f1": f1,
    }


def levenshtein(a: list[str], b: list[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + (0 if token_a == token_b else 1)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def sequence_rows(gt: pd.DataFrame, pred: pd.DataFrame, videos: pd.DataFrame, models: list[dict], max_n: int) -> list[dict]:
    rows = []
    for _, video in videos.iterrows():
        split = str(video["split"])
        video_id = str(video["video_id"])
        gt_seq = bout_sequence(gt, split, video_id, None)
        for spec in models:
            pred_seq = bout_sequence(pred, split, video_id, spec["model_name"])
            edit = levenshtein(gt_seq, pred_seq)
            norm_edit_similarity = 1.0 - (edit / max(1, max(len(gt_seq), len(pred_seq))))
            base = {
                **spec,
                "split": split,
                "video_id": video_id,
                "gt_sequence_length": len(gt_seq),
                "pred_sequence_length": len(pred_seq),
                "sequence_length_ratio_pred_over_gt": len(pred_seq) / max(1, len(gt_seq)),
                "levenshtein_distance": edit,
                "normalized_edit_similarity": norm_edit_similarity,
            }
            for n in range(1, max_n + 1):
                rows.append({**base, **multiset_f1(gt_seq, pred_seq, n)})
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "ngram_precision",
        "ngram_recall",
        "ngram_f1",
        "sequence_length_ratio_pred_over_gt",
        "normalized_edit_similarity",
    ]
    return (
        rows.groupby(["track", "head", "capacity", "seed", "n"], as_index=False)
        .agg(**{f"{m}_video_mean": (m, "mean") for m in metrics})
        .groupby(["track", "head", "capacity", "n"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            **{f"{m}_mean": (f"{m}_video_mean", "mean") for m in metrics},
            **{f"{m}_sd": (f"{m}_video_mean", "std") for m in metrics},
        )
    )


def plot(summary: pd.DataFrame, figure_dir: Path, output_name: str) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    axes = axes.ravel()
    tracks = [
        f"{head.capitalize() if head == 'attention' else 'LSTM'}-{cap}"
        for head in HEAD_ORDER
        for cap in CAPACITY_ORDER
    ]

    for track in tracks:
        sub = summary[summary["track"].eq(track)].sort_values("n")
        axes[0].errorbar(
            sub["n"],
            sub["ngram_f1_mean"],
            yerr=sub["ngram_f1_sd"],
            marker="o",
            color=TRACK_COLORS[track],
            label=track,
            linewidth=1.4,
        )
    axes[0].set_xticks([1, 2, 3, 4])
    axes[0].set_ylim(0, 1)
    axes[0].set_xlabel("N-gram length")
    axes[0].set_ylabel("Sequence n-gram F1")
    axes[0].set_title("A. Bout-sequence n-gram recovery")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)

    n2 = summary[summary["n"].eq(2)].set_index("track").reindex(tracks).reset_index()
    x = np.arange(len(tracks))
    axes[1].bar(
        x,
        n2["normalized_edit_similarity_mean"],
        yerr=n2["normalized_edit_similarity_sd"],
        color=[TRACK_COLORS[t] for t in tracks],
        alpha=0.88,
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tracks, rotation=35, ha="right")
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Normalized edit similarity")
    axes[1].set_title("B. Full bout-sequence edit similarity")

    axes[2].bar(
        x,
        n2["sequence_length_ratio_pred_over_gt_mean"],
        yerr=n2["sequence_length_ratio_pred_over_gt_sd"],
        color=[TRACK_COLORS[t] for t in tracks],
        alpha=0.88,
    )
    axes[2].axhline(1.0, color="#6B7280", linewidth=1)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(tracks, rotation=35, ha="right")
    axes[2].set_ylabel("Predicted / GT sequence length")
    axes[2].set_title("C. Bout-sequence length recovery")

    for track in tracks:
        sub = summary[summary["track"].eq(track) & summary["n"].isin([1, 2, 3])].sort_values("n")
        axes[3].plot(
            sub["ngram_precision_mean"],
            sub["ngram_recall_mean"],
            marker="o",
            color=TRACK_COLORS[track],
            label=track,
            linewidth=1.2,
            alpha=0.85,
        )
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)
    axes[3].set_xlabel("N-gram precision")
    axes[3].set_ylabel("N-gram recall")
    axes[3].set_title("D. Precision-recall trajectory for n=1-3")

    fig.suptitle("Bout-sequence n-gram audit across YOLO/SPPF full-stream capacities", fontsize=13)
    save_figure(fig, figure_dir / output_name)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    per_video, gt, pred = load_inputs(args.table_dir, args.prediction_type)
    videos = video_units(per_video, args.reference_model)
    models = discover_models(per_video)
    rows = pd.DataFrame(sequence_rows(gt, pred, videos, models, args.max_n))
    summary = summarize(rows)
    rows.to_csv(args.table_dir / f"{args.output_name}_video_level.csv", index=False)
    summary.to_csv(args.table_dir / f"{args.output_name}_summary.csv", index=False)
    plot(summary, args.figure_dir, args.output_name)
    n2_rank = (
        summary[summary["n"].eq(2)][["track", "ngram_f1_mean", "normalized_edit_similarity_mean"]]
        .sort_values("ngram_f1_mean", ascending=False)
    )
    print("Bout bigram F1 ranking:")
    print(n2_rank.to_string(index=False))
    print(f"\nWrote {args.figure_dir / (args.output_name + '.png')}")


if __name__ == "__main__":
    main()
