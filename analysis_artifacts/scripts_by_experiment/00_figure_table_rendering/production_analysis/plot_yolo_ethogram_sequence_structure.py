"""Main-manuscript bout-sequence structure analysis for YOLO/SPPF ethograms."""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
STYLE_DIR = SCRIPT_DIR.parent / "current_analysis"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import (  # noqa: E402
    BEHAVIOR_FILL_COLORS,
    HEAD_COLORS,
    apply_style,
    save_figure,
)


BEHAVIORS = ["attack", "investigation", "mount"]
BEHAVIOR_LABELS = {
    "attack": "Attack",
    "investigation": "Investigation",
    "mount": "Mount",
}
MODEL_RE = re.compile(r"^yolo_sppf_full_(?P<head>lstm|attention)(?P<capacity>\d+)_seed(?P<seed>\d+)$")
HEAD_ORDER = ["attention", "lstm"]
CAPACITY_ORDER = [256, 512, 896]
FOCUS_TRACKS = ["Attention-256", "LSTM-512"]
BASELINE_COLORS = {
    "Composition shuffle": "#6B7280",
    "Markov baseline": "#111827",
}
TRACK_COLORS = {
    "Attention-256": HEAD_COLORS["attention"],
    "Attention-512": "#34D399",
    "Attention-896": "#6EE7B7",
    "LSTM-256": HEAD_COLORS["lstm"],
    "LSTM-512": "#1D4ED8",
    "LSTM-896": "#60A5FA",
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
        default=Path("Manuscript_penultimate/figures/production/yolo_sppf_main"),
    )
    parser.add_argument("--prediction_type", default="smoothed_frames")
    parser.add_argument("--reference_model", default="yolo_sppf_full_attention256_seed42")
    parser.add_argument("--output_name", default="fig10_yolo_ethogram_sequence_structure")
    parser.add_argument("--max_n", type=int, default=4)
    parser.add_argument("--null_repeats", type=int, default=250)
    parser.add_argument("--random_seed", type=int, default=1729)
    parser.add_argument("--motif_n", type=int, default=3)
    parser.add_argument("--token_video_id", default="Mouse062_20160526_18-56-26")
    parser.add_argument("--token_split", default="test_1")
    parser.add_argument("--token_max_bouts", type=int, default=90)
    return parser.parse_args()


def model_track(head: str, capacity: int) -> str:
    return f"{head.capitalize() if head == 'attention' else 'LSTM'}-{capacity}"


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
    videos = (
        per_video[per_video["model_name"].eq(reference_model)][["split", "video_id"]]
        .drop_duplicates()
        .copy()
    )
    if videos.empty:
        raise ValueError(f"No held-out videos found for reference model {reference_model}")
    return videos


def discover_models(per_video: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    for model_name in sorted(per_video["model_name"].dropna().unique()):
        match = MODEL_RE.match(str(model_name))
        if not match:
            continue
        head = match.group("head")
        capacity = int(match.group("capacity"))
        seed = int(match.group("seed"))
        if capacity not in CAPACITY_ORDER:
            continue
        rows.append(
            {
                "model_name": str(model_name),
                "head": head,
                "capacity": capacity,
                "seed": seed,
                "track": model_track(head, capacity),
            }
        )
    return sorted(rows, key=lambda row: (HEAD_ORDER.index(row["head"]), row["capacity"], row["seed"]))


def bout_sequence(table: pd.DataFrame, split: str, video_id: str, model_name: str | None) -> list[str]:
    sub = table[table["split"].eq(split) & table["video_id"].eq(video_id)].copy()
    if model_name is None:
        sub = sub.drop_duplicates(["split", "video_id", "class", "start_frame", "end_frame"]).copy()
    else:
        sub = sub[sub["model_name"].eq(model_name)].copy()
    sub = sub[sub["class"].isin(BEHAVIORS)].sort_values(["start_frame", "end_frame", "class"])
    return [str(value) for value in sub["class"].tolist()]


def ngrams(seq: list[str], n: int) -> Counter:
    if len(seq) < n:
        return Counter()
    return Counter(tuple(seq[idx : idx + n]) for idx in range(len(seq) - n + 1))


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
        "ngram_tp": tp,
        "pred_ngram_total": pred_total,
        "gt_ngram_total": gt_total,
        "ngram_precision": precision,
        "ngram_recall": recall,
        "ngram_f1": f1,
    }


def aggregate_gt_transition_model(gt_sequences: dict[tuple[str, str], list[str]]) -> tuple[np.ndarray, np.ndarray]:
    start_counts = Counter()
    transition_counts = {behavior: Counter() for behavior in BEHAVIORS}
    unigram_counts = Counter()
    for seq in gt_sequences.values():
        if not seq:
            continue
        start_counts[seq[0]] += 1
        unigram_counts.update(seq)
        for src, dst in zip(seq[:-1], seq[1:]):
            transition_counts[src][dst] += 1
    start = np.array([start_counts[b] for b in BEHAVIORS], dtype=float)
    start = start / start.sum() if start.sum() else np.full(len(BEHAVIORS), 1 / len(BEHAVIORS))
    unigram = np.array([unigram_counts[b] for b in BEHAVIORS], dtype=float)
    unigram = unigram / unigram.sum() if unigram.sum() else np.full(len(BEHAVIORS), 1 / len(BEHAVIORS))
    trans = np.zeros((len(BEHAVIORS), len(BEHAVIORS)), dtype=float)
    for i, src in enumerate(BEHAVIORS):
        row = np.array([transition_counts[src][dst] for dst in BEHAVIORS], dtype=float)
        trans[i] = row / row.sum() if row.sum() else unigram
    return start, trans


def generate_markov(length: int, start: np.ndarray, trans: np.ndarray, rng: np.random.Generator) -> list[str]:
    if length <= 0:
        return []
    states = np.arange(len(BEHAVIORS))
    current = int(rng.choice(states, p=start))
    seq = [BEHAVIORS[current]]
    for _ in range(1, length):
        current = int(rng.choice(states, p=trans[current]))
        seq.append(BEHAVIORS[current])
    return seq


def collect_sequences(
    gt: pd.DataFrame,
    pred: pd.DataFrame,
    videos: pd.DataFrame,
    models: list[dict],
) -> tuple[dict[tuple[str, str], list[str]], dict[tuple[str, str, str], list[str]]]:
    gt_sequences: dict[tuple[str, str], list[str]] = {}
    pred_sequences: dict[tuple[str, str, str], list[str]] = {}
    for _, video in videos.iterrows():
        split = str(video["split"])
        video_id = str(video["video_id"])
        gt_seq = bout_sequence(gt, split, video_id, None)
        gt_sequences[(split, video_id)] = gt_seq
        for spec in models:
            pred_sequences[(split, video_id, spec["model_name"])] = bout_sequence(
                pred,
                split,
                video_id,
                spec["model_name"],
            )
    return gt_sequences, pred_sequences


def model_metric_rows(
    gt_sequences: dict[tuple[str, str], list[str]],
    pred_sequences: dict[tuple[str, str, str], list[str]],
    models: list[dict],
    max_n: int,
) -> list[dict]:
    rows: list[dict] = []
    for (split, video_id), gt_seq in gt_sequences.items():
        for spec in models:
            pred_seq = pred_sequences[(split, video_id, spec["model_name"])]
            base = {
                "source": "model",
                "track": spec["track"],
                "head": spec["head"],
                "capacity": spec["capacity"],
                "seed": spec["seed"],
                "split": split,
                "video_id": video_id,
                "gt_sequence_length": len(gt_seq),
                "pred_sequence_length": len(pred_seq),
                "sequence_length_ratio_pred_over_gt": len(pred_seq) / max(1, len(gt_seq)),
            }
            for n in range(1, max_n + 1):
                rows.append({**base, "n": n, **multiset_f1(gt_seq, pred_seq, n)})
    return rows


def null_metric_rows(
    gt_sequences: dict[tuple[str, str], list[str]],
    max_n: int,
    repeats: int,
    rng: np.random.Generator,
) -> list[dict]:
    rows: list[dict] = []
    start, trans = aggregate_gt_transition_model(gt_sequences)
    for repeat in range(repeats):
        for (split, video_id), gt_seq in gt_sequences.items():
            shuffled = list(rng.permutation(gt_seq)) if gt_seq else []
            markov = generate_markov(len(gt_seq), start, trans, rng)
            for label, seq in [("Composition shuffle", shuffled), ("Markov baseline", markov)]:
                base = {
                    "source": "null",
                    "track": label,
                    "head": "null",
                    "capacity": 0,
                    "seed": repeat,
                    "split": split,
                    "video_id": video_id,
                    "gt_sequence_length": len(gt_seq),
                    "pred_sequence_length": len(seq),
                    "sequence_length_ratio_pred_over_gt": len(seq) / max(1, len(gt_seq)),
                }
                for n in range(1, max_n + 1):
                    rows.append({**base, "n": n, **multiset_f1(gt_seq, seq, n)})
    return rows


def summarize_for_panel(rows: pd.DataFrame) -> pd.DataFrame:
    per_seed = (
        rows.groupby(["source", "track", "head", "capacity", "seed", "n"], as_index=False)
        .agg(
            ngram_f1=("ngram_f1", "mean"),
            ngram_precision=("ngram_precision", "mean"),
            ngram_recall=("ngram_recall", "mean"),
            sequence_length_ratio_pred_over_gt=("sequence_length_ratio_pred_over_gt", "mean"),
        )
    )
    return (
        per_seed.groupby(["source", "track", "head", "capacity", "n"], as_index=False)
        .agg(
            n_samples=("seed", "nunique"),
            ngram_f1_mean=("ngram_f1", "mean"),
            ngram_f1_sd=("ngram_f1", "std"),
            ngram_precision_mean=("ngram_precision", "mean"),
            ngram_precision_sd=("ngram_precision", "std"),
            ngram_recall_mean=("ngram_recall", "mean"),
            ngram_recall_sd=("ngram_recall", "std"),
            sequence_length_ratio_pred_over_gt_mean=("sequence_length_ratio_pred_over_gt", "mean"),
            sequence_length_ratio_pred_over_gt_sd=("sequence_length_ratio_pred_over_gt", "std"),
        )
    )


def corpus_counts(sequences: list[list[str]], n: int) -> Counter:
    counts = Counter()
    for seq in sequences:
        counts.update(ngrams(seq, n))
    return counts


def motif_bins(gt_counts: Counter) -> dict[tuple[str, ...], str]:
    ordered = sorted(gt_counts.items(), key=lambda item: (item[1], item[0]))
    if not ordered:
        return {}
    bins = {}
    for idx, (motif, _count) in enumerate(ordered):
        frac = (idx + 0.5) / len(ordered)
        if frac <= 1 / 3:
            label = "Rare"
        elif frac <= 2 / 3:
            label = "Intermediate"
        else:
            label = "Common"
        bins[motif] = label
    return bins


def recall_by_bin(gt_counts: Counter, pred_counts: Counter, bins: dict[tuple[str, ...], str]) -> dict[str, float]:
    rows = {}
    for label in ["Rare", "Intermediate", "Common"]:
        motifs = [motif for motif, bin_label in bins.items() if bin_label == label]
        denom = sum(gt_counts[motif] for motif in motifs)
        numer = sum(min(pred_counts.get(motif, 0), gt_counts[motif]) for motif in motifs)
        rows[label] = numer / denom if denom else np.nan
    return rows


def rare_sequence_rows(
    gt_sequences: dict[tuple[str, str], list[str]],
    pred_sequences: dict[tuple[str, str, str], list[str]],
    models: list[dict],
    motif_n: int,
    repeats: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gt_counts = corpus_counts(list(gt_sequences.values()), motif_n)
    bins = motif_bins(gt_counts)
    motif_table = pd.DataFrame(
        [
            {
                "motif": " -> ".join(BEHAVIOR_LABELS[token] for token in motif),
                "gt_count": count,
                "frequency_bin": bins[motif],
            }
            for motif, count in sorted(gt_counts.items(), key=lambda item: item[1])
        ]
    )
    rows = []
    for spec in models:
        pred_counts = corpus_counts(
            [pred_sequences[(split, video_id, spec["model_name"])] for split, video_id in gt_sequences],
            motif_n,
        )
        values = recall_by_bin(gt_counts, pred_counts, bins)
        for bin_label, recall in values.items():
            rows.append({**spec, "source": "model", "frequency_bin": bin_label, "motif_recall": recall})
    start, trans = aggregate_gt_transition_model(gt_sequences)
    for repeat in range(repeats):
        shuffled_sequences = []
        markov_sequences = []
        for seq in gt_sequences.values():
            shuffled_sequences.append(list(rng.permutation(seq)) if seq else [])
            markov_sequences.append(generate_markov(len(seq), start, trans, rng))
        for label, seqs in [("Composition shuffle", shuffled_sequences), ("Markov baseline", markov_sequences)]:
            values = recall_by_bin(gt_counts, corpus_counts(seqs, motif_n), bins)
            for bin_label, recall in values.items():
                rows.append(
                    {
                        "track": label,
                        "head": "null",
                        "capacity": 0,
                        "seed": repeat,
                        "model_name": label,
                        "source": "null",
                        "frequency_bin": bin_label,
                        "motif_recall": recall,
                    }
                )
    recall_df = pd.DataFrame(rows)
    summary = (
        recall_df.groupby(["source", "track", "head", "capacity", "frequency_bin"], as_index=False)
        .agg(
            n_samples=("seed", "nunique"),
            motif_recall_mean=("motif_recall", "mean"),
            motif_recall_sd=("motif_recall", "std"),
        )
    )
    return recall_df, summary.merge(motif_table.groupby("frequency_bin", as_index=False).agg(n_motifs=("motif", "size")), on="frequency_bin", how="left")


def transition_probability(counts: Counter) -> np.ndarray:
    mat = np.zeros((len(BEHAVIORS), len(BEHAVIORS)), dtype=float)
    for i, src in enumerate(BEHAVIORS):
        row_total = sum(counts.get((src, dst), 0) for dst in BEHAVIORS)
        if row_total:
            for j, dst in enumerate(BEHAVIORS):
                mat[i, j] = counts.get((src, dst), 0) / row_total
    return mat


def bigram_edge_table(
    gt_sequences: dict[tuple[str, str], list[str]],
    pred_sequences: dict[tuple[str, str, str], list[str]],
    models: list[dict],
    focus_track: str = "Attention-256",
) -> pd.DataFrame:
    gt_counts = corpus_counts(list(gt_sequences.values()), 2)
    focus_models = [spec for spec in models if spec["track"] == focus_track]
    pred_counts = Counter()
    for spec in focus_models:
        pred_counts.update(
            corpus_counts(
                [pred_sequences[(split, video_id, spec["model_name"])] for split, video_id in gt_sequences],
                2,
            )
        )
    # Average counts over seeds before row normalization.
    if focus_models:
        for key in list(pred_counts.keys()):
            pred_counts[key] = pred_counts[key] / len(focus_models)
    gt_prob = transition_probability(gt_counts)
    pred_prob = transition_probability(pred_counts)
    rows = []
    for i, src in enumerate(BEHAVIORS):
        for j, dst in enumerate(BEHAVIORS):
            rows.append(
                {
                    "from_behavior": src,
                    "to_behavior": dst,
                    "gt_probability": gt_prob[i, j],
                    "model_probability": pred_prob[i, j],
                    "difference_model_minus_gt": pred_prob[i, j] - gt_prob[i, j],
                    "focus_track": focus_track,
                }
            )
    return pd.DataFrame(rows)


def draw_node_arrow_panel(ax, edge_df: pd.DataFrame) -> None:
    ax.set_title("C. Bout-transition organization", loc="left")
    ax.axis("off")
    positions = {
        "attack": np.array([0.12, 0.48]),
        "investigation": np.array([0.52, 0.80]),
        "mount": np.array([0.88, 0.48]),
    }
    for _, edge in edge_df.iterrows():
        src = str(edge["from_behavior"])
        dst = str(edge["to_behavior"])
        prob = float(edge["gt_probability"])
        diff = float(edge["difference_model_minus_gt"])
        if prob < 0.03 and abs(diff) < 0.05:
            continue
        color = "#047857" if diff > 0.035 else "#C2410C" if diff < -0.035 else "#6B7280"
        linewidth = 0.8 + 7.0 * prob
        alpha = 0.45 + min(0.45, prob)
        if src == dst:
            center = positions[src]
            loop = FancyArrowPatch(
                center + np.array([-0.035, 0.065]),
                center + np.array([0.035, 0.065]),
                connectionstyle="arc3,rad=1.8",
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=linewidth,
                color=color,
                alpha=alpha,
            )
            ax.add_patch(loop)
            label_pos = center + np.array([0.0, 0.16])
        else:
            start = positions[src]
            end = positions[dst]
            vec = end - start
            start = start + vec * 0.16
            end = end - vec * 0.16
            rad = 0.18 if (BEHAVIORS.index(src) < BEHAVIORS.index(dst)) else -0.18
            arrow = FancyArrowPatch(
                start,
                end,
                connectionstyle=f"arc3,rad={rad}",
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=linewidth,
                color=color,
                alpha=alpha,
            )
            ax.add_patch(arrow)
            label_pos = (start + end) / 2
        if prob >= 0.10 or abs(diff) >= 0.08:
            ax.text(
                label_pos[0],
                label_pos[1],
                f"{prob * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=7,
                color="#111827",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 1.0},
            )
    for behavior, pos in positions.items():
        circle = Circle(pos, 0.095, facecolor=BEHAVIOR_FILL_COLORS[behavior], edgecolor="#111827", linewidth=1.0)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], BEHAVIOR_LABELS[behavior], ha="center", va="center", fontsize=8, color="white")
    ax.text(
        0.02,
        0.06,
        "Arrow width: GT bigram probability\nEdge color: attention-256 minus GT",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.5,
        color="#374151",
    )


def token_window(gt_seq: list[str], max_bouts: int) -> tuple[int, int]:
    if len(gt_seq) <= max_bouts:
        return 0, len(gt_seq)
    attack_indices = [idx for idx, value in enumerate(gt_seq) if value == "attack"]
    center = attack_indices[0] if attack_indices else max_bouts // 2
    start = max(0, min(center - max_bouts // 3, len(gt_seq) - max_bouts))
    return start, start + max_bouts


def draw_token_strip(ax, sequences: list[tuple[str, list[str]]], start: int, end: int) -> None:
    ax.set_title("D. Behavior-order tokens from one held-out video", loc="left")
    ax.set_xlim(start, end)
    ax.set_ylim(-0.5, len(sequences) - 0.5)
    y_positions = np.arange(len(sequences))[::-1]
    for y, (label, seq) in zip(y_positions, sequences):
        for idx in range(start, min(end, len(seq))):
            ax.add_patch(
                Rectangle(
                    (idx, y - 0.32),
                    0.92,
                    0.64,
                    facecolor=BEHAVIOR_FILL_COLORS[seq[idx]],
                    edgecolor="none",
                )
            )
        ax.text(start - 1.5, y, label, ha="right", va="center", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Bout-order index")
    ax.grid(axis="x", alpha=0.18)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=BEHAVIOR_FILL_COLORS[b], edgecolor="none", label=BEHAVIOR_LABELS[b])
        for b in BEHAVIORS
    ]
    ax.legend(handles=handles, frameon=False, ncol=3, loc="upper right", fontsize=7)


def plot_figure(
    summary: pd.DataFrame,
    rare_summary: pd.DataFrame,
    edge_df: pd.DataFrame,
    gt_sequences: dict[tuple[str, str], list[str]],
    pred_sequences: dict[tuple[str, str, str], list[str]],
    models: list[dict],
    args: argparse.Namespace,
) -> None:
    apply_style()
    fig = plt.figure(figsize=(14.0, 10.0), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, top=0.91, bottom=0.08, hspace=0.36, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    plot_tracks = ["Composition shuffle", "Markov baseline", *FOCUS_TRACKS]
    for track in plot_tracks:
        sub = summary[summary["track"].eq(track)].sort_values("n")
        if sub.empty:
            continue
        color = BASELINE_COLORS.get(track, TRACK_COLORS.get(track, "#6B7280"))
        linestyle = "--" if track in BASELINE_COLORS else "-"
        marker = "s" if track in BASELINE_COLORS else "o"
        ax_a.errorbar(
            sub["n"],
            sub["ngram_f1_mean"],
            yerr=sub["ngram_f1_sd"].fillna(0),
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            color=color,
            label=track,
            capsize=2,
        )
    ax_a.set_xticks(range(1, args.max_n + 1))
    ax_a.set_ylim(0, 1.0)
    ax_a.set_xlabel("Bout n-gram length")
    ax_a.set_ylabel("Weighted n-gram F1")
    ax_a.set_title("A. Weighted n-gram benchmark against null ethograms", loc="left")
    ax_a.legend(frameon=False, fontsize=8)

    bins = ["Rare", "Intermediate", "Common"]
    x = np.arange(len(bins))
    width = 0.18
    for offset, track in zip([-1.5, -0.5, 0.5, 1.5], plot_tracks):
        sub = rare_summary[rare_summary["track"].eq(track)].set_index("frequency_bin").reindex(bins)
        if sub.empty:
            continue
        color = BASELINE_COLORS.get(track, TRACK_COLORS.get(track, "#6B7280"))
        ax_b.bar(
            x + offset * width,
            sub["motif_recall_mean"],
            width,
            yerr=sub["motif_recall_sd"].fillna(0),
            color=color,
            alpha=0.85,
            capsize=2,
            label=track,
        )
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"{b}\n(n={int(rare_summary[rare_summary['frequency_bin'].eq(b)]['n_motifs'].max())})" for b in bins])
    ax_b.set_ylim(0, 1.0)
    ax_b.set_ylabel(f"{args.motif_n}-gram recall")
    ax_b.set_title("B. Rare motif recovery exceeds null ethograms", loc="left")

    draw_node_arrow_panel(ax_c, edge_df)

    gt_seq = gt_sequences.get((args.token_split, args.token_video_id), [])
    attn_model = "yolo_sppf_full_attention256_seed42"
    lstm_model = "yolo_sppf_full_lstm512_seed42"
    sequences = [
        ("Ground truth", gt_seq),
        ("Attention-256", pred_sequences.get((args.token_split, args.token_video_id, attn_model), [])),
        ("LSTM-512", pred_sequences.get((args.token_split, args.token_video_id, lstm_model), [])),
    ]
    start, end = token_window(gt_seq, args.token_max_bouts)
    draw_token_strip(ax_d, sequences, start, end)

    fig.suptitle("Predicted ethograms recover rare sequence motifs while compressing dense bout structure", fontsize=14)
    save_figure(fig, args.figure_dir / args.output_name)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)
    per_video, gt, pred = load_inputs(args.table_dir, args.prediction_type)
    videos = video_units(per_video, args.reference_model)
    models = discover_models(per_video)
    gt_sequences, pred_sequences = collect_sequences(gt, pred, videos, models)

    model_rows = model_metric_rows(gt_sequences, pred_sequences, models, args.max_n)
    null_rows = null_metric_rows(gt_sequences, args.max_n, args.null_repeats, rng)
    rows = pd.DataFrame([*model_rows, *null_rows])
    summary = summarize_for_panel(rows)
    rare_rows, rare_summary = rare_sequence_rows(
        gt_sequences,
        pred_sequences,
        models,
        args.motif_n,
        args.null_repeats,
        rng,
    )
    edge_df = bigram_edge_table(gt_sequences, pred_sequences, models, focus_track="Attention-256")

    rows.to_csv(args.table_dir / f"{args.output_name}_video_level_and_nulls.csv", index=False)
    summary.to_csv(args.table_dir / f"{args.output_name}_summary.csv", index=False)
    rare_rows.to_csv(args.table_dir / f"{args.output_name}_rare_motif_recall_long.csv", index=False)
    rare_summary.to_csv(args.table_dir / f"{args.output_name}_rare_motif_recall_summary.csv", index=False)
    edge_df.to_csv(args.table_dir / f"{args.output_name}_bigram_edge_table.csv", index=False)
    plot_figure(summary, rare_summary, edge_df, gt_sequences, pred_sequences, models, args)

    print("Main sequence-structure n-gram summary:")
    print(
        summary[
            summary["track"].isin(["Composition shuffle", "Markov baseline", *FOCUS_TRACKS])
            & summary["n"].isin([2, 3, 4])
        ][["track", "n", "ngram_f1_mean", "ngram_f1_sd"]].to_string(index=False)
    )
    print(f"\nWrote {args.figure_dir / (args.output_name + '.png')}")


if __name__ == "__main__":
    main()
