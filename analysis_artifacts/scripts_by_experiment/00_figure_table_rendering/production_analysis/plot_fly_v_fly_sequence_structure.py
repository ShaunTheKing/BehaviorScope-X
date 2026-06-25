"""Supplemental Fly-v-Fly bout-sequence structure analysis.

This is a reproducibility-facing analysis for the focused Fly adaptation run. It asks
whether the selected temporal windows recover local behavior-order structure in
the short-bout Fly-v-Fly held-out videos beyond simple null ethograms.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
REPO_ROOT = THIS_DIR.parents[2]
STYLE_DIR = THIS_DIR.parent / "current_analysis"
SHARED_DIR = REPO_ROOT / "shared_scripts"
for path in (STYLE_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from behaviorscope_figure_style import (  # noqa: E402
    HEAD_COLORS,
    NEUTRAL,
    apply_style,
    panel_label,
    save_figure,
)
from fly_run_all_eval import (  # noqa: E402
    BEHAVIOR_CLASSES,
    actions_to_bouts,
    bouts_to_frames,
    frames_to_bouts,
    load_actions_file,
    load_raw_window_predictions,
    load_smoothed_predictions,
)


DEFAULT_EVAL_DIR = (
    REPO_ROOT
    / "outputs"
    / "controlled_comparison_runs"
    / "fly_adaptation_attention256_cw12_seed42"
    / "fly"
    / "eval"
)
TABLE_DIR = ROOT / "tables" / "production" / "fly_v_fly"
FIG_DIR = ROOT / "figures" / "production" / "supplement"

BEHAVIOR_LABELS = {
    "lunge": "Lunge",
    "wing_threat": "Wing threat",
    "charge": "Charge",
    "hold": "Hold",
    "tussle": "Tussle",
}
BEHAVIOR_COLORS = {
    "lunge": "#0072B2",
    "wing_threat": "#009E73",
    "charge": "#D55E00",
    "hold": "#8E6C88",
    "tussle": "#C9A227",
}
WINDOW_LABELS = {"w16s8": "16-frame window", "w8s4": "8-frame window"}
WINDOW_COLORS = {"w16s8": NEUTRAL["dark"], "w8s4": HEAD_COLORS["attention"]}
NULL_COLORS = {"GT composition shuffle": NEUTRAL["mid"], "GT Markov baseline": NEUTRAL["black"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval_dir", type=Path, default=DEFAULT_EVAL_DIR)
    parser.add_argument("--table_dir", type=Path, default=TABLE_DIR)
    parser.add_argument("--figure_dir", type=Path, default=FIG_DIR)
    parser.add_argument("--output_name", default="supp_fly_v_fly_sequence_structure")
    parser.add_argument("--max_n", type=int, default=4)
    parser.add_argument("--null_repeats", type=int, default=500)
    parser.add_argument("--random_seed", type=int, default=2407)
    parser.add_argument("--prediction_type", choices=["raw", "smoothed"], default="raw")
    parser.add_argument("--token_movie_id", type=int, default=6)
    parser.add_argument("--token_max_bouts", type=int, default=120)
    return parser.parse_args()


def run_window(run_name: str) -> str:
    match = re.search(r"_(w\d+s\d+)_seed", run_name)
    if not match:
        raise ValueError(f"Cannot parse Fly window from {run_name}")
    return match.group(1)


def style_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    if grid_axis:
        ax.grid(axis=grid_axis, color=NEUTRAL["grid"], linewidth=0.8, alpha=0.28)
    ax.set_axisbelow(True)


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


def transition_model(gt_sequences: dict[int, list[str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_counts = Counter()
    unigram_counts = Counter()
    transition_counts = {behavior: Counter() for behavior in BEHAVIOR_CLASSES}
    for seq in gt_sequences.values():
        if not seq:
            continue
        start_counts[seq[0]] += 1
        unigram_counts.update(seq)
        for src, dst in zip(seq[:-1], seq[1:]):
            transition_counts[src][dst] += 1
    start = np.array([start_counts[b] for b in BEHAVIOR_CLASSES], dtype=float)
    unigram = np.array([unigram_counts[b] for b in BEHAVIOR_CLASSES], dtype=float)
    start = start / start.sum() if start.sum() else np.full(len(BEHAVIOR_CLASSES), 1 / len(BEHAVIOR_CLASSES))
    unigram = unigram / unigram.sum() if unigram.sum() else np.full(len(BEHAVIOR_CLASSES), 1 / len(BEHAVIOR_CLASSES))
    trans = np.zeros((len(BEHAVIOR_CLASSES), len(BEHAVIOR_CLASSES)), dtype=float)
    for i, src in enumerate(BEHAVIOR_CLASSES):
        row = np.array([transition_counts[src][dst] for dst in BEHAVIOR_CLASSES], dtype=float)
        trans[i] = row / row.sum() if row.sum() else unigram
    return start, trans, unigram


def generate_markov(length: int, start: np.ndarray, trans: np.ndarray, rng: np.random.Generator) -> list[str]:
    if length <= 0:
        return []
    states = np.arange(len(BEHAVIOR_CLASSES))
    current = int(rng.choice(states, p=start))
    seq = [BEHAVIOR_CLASSES[current]]
    for _ in range(1, length):
        current = int(rng.choice(states, p=trans[current]))
        seq.append(BEHAVIOR_CLASSES[current])
    return seq


def sequence_from_frames(frames: np.ndarray) -> list[str]:
    bouts = frames_to_bouts(frames)
    return [str(row["class"]) for row in bouts if row["class"] in BEHAVIOR_CLASSES]


def load_sequences(eval_dir: Path, heldout: pd.DataFrame, prediction_type: str) -> tuple[dict[int, list[str]], dict[tuple[int, str], list[str]], list[dict]]:
    gt_sequences: dict[int, list[str]] = {}
    pred_sequences: dict[tuple[int, str], list[str]] = {}
    specs: list[dict] = []
    run_dirs = sorted(p for p in eval_dir.iterdir() if p.is_dir() and p.name.startswith("fly_yolo"))
    for run_dir in run_dirs:
        window = run_window(run_dir.name)
        specs.append({"run": run_dir.name, "window": window, "track": WINDOW_LABELS[window]})

    for _, video in heldout.iterrows():
        movie_id = int(video["movie_id"])
        n_frames = int(video["n_frames"])
        gt_frames = bouts_to_frames(actions_to_bouts(load_actions_file(Path(str(video["primary_actions_path"])))), n_frames)
        gt_sequences[movie_id] = sequence_from_frames(gt_frames)
        for spec in specs:
            run_dir = eval_dir / spec["run"]
            pred_csv = run_dir / f"movie{movie_id}" / f"movie{movie_id}.behavior.csv"
            smoothed_csv = run_dir / f"movie{movie_id}" / f"movie{movie_id}.behavior.smoothed_frames.csv"
            if prediction_type == "raw":
                pred_frames = load_raw_window_predictions(pred_csv, n_frames)
            else:
                pred_frames = load_smoothed_predictions(smoothed_csv, n_frames)
            pred_sequences[(movie_id, spec["run"])] = sequence_from_frames(pred_frames)
    return gt_sequences, pred_sequences, specs


def model_metric_rows(
    gt_sequences: dict[int, list[str]],
    pred_sequences: dict[tuple[int, str], list[str]],
    specs: list[dict],
    max_n: int,
    prediction_type: str,
) -> list[dict]:
    rows = []
    for movie_id, gt_seq in gt_sequences.items():
        for spec in specs:
            pred_seq = pred_sequences[(movie_id, spec["run"])]
            base = {
                "source": "model",
                **spec,
                "seed": 0,
                "prediction_type": prediction_type,
                "movie_id": movie_id,
                "gt_sequence_length": len(gt_seq),
                "pred_sequence_length": len(pred_seq),
                "sequence_length_ratio_pred_over_gt": len(pred_seq) / max(1, len(gt_seq)),
            }
            for n in range(1, max_n + 1):
                rows.append({**base, "n": n, **multiset_f1(gt_seq, pred_seq, n)})
    return rows


def null_metric_rows(
    gt_sequences: dict[int, list[str]],
    pred_sequences: dict[tuple[int, str], list[str]],
    specs: list[dict],
    max_n: int,
    repeats: int,
    rng: np.random.Generator,
    prediction_type: str,
) -> list[dict]:
    rows = []
    start, trans, unigram = transition_model(gt_sequences)
    for repeat in range(repeats):
        for movie_id, gt_seq in gt_sequences.items():
            shuffled = list(rng.permutation(gt_seq)) if gt_seq else []
            markov = generate_markov(len(gt_seq), start, trans, rng)
            for label, seq in [("GT composition shuffle", shuffled), ("GT Markov baseline", markov)]:
                base = {
                    "source": "null",
                    "run": label,
                    "window": "null",
                    "track": label,
                    "prediction_type": prediction_type,
                    "movie_id": movie_id,
                    "seed": repeat,
                    "gt_sequence_length": len(gt_seq),
                    "pred_sequence_length": len(seq),
                    "sequence_length_ratio_pred_over_gt": len(seq) / max(1, len(gt_seq)),
                }
                for n in range(1, max_n + 1):
                    rows.append({**base, "n": n, **multiset_f1(gt_seq, seq, n)})
            for spec in specs:
                pred_seq = pred_sequences[(movie_id, spec["run"])]
                model_length = len(pred_seq)
                if model_length:
                    sampled = [BEHAVIOR_CLASSES[i] for i in rng.choice(np.arange(len(BEHAVIOR_CLASSES)), size=model_length, p=unigram)]
                else:
                    sampled = []
                markov_model_len = generate_markov(model_length, start, trans, rng)
                for label, seq in [
                    (f"{spec['track']} length unigram", sampled),
                    (f"{spec['track']} length Markov", markov_model_len),
                ]:
                    base = {
                        "source": "length_matched_null",
                        "run": label,
                        "window": spec["window"],
                        "track": label,
                        "prediction_type": prediction_type,
                        "movie_id": movie_id,
                        "seed": repeat,
                        "gt_sequence_length": len(gt_seq),
                        "pred_sequence_length": len(seq),
                        "sequence_length_ratio_pred_over_gt": len(seq) / max(1, len(gt_seq)),
                    }
                    for n in range(1, max_n + 1):
                        rows.append({**base, "n": n, **multiset_f1(gt_seq, seq, n)})
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    per_sample = (
        rows.groupby(["source", "track", "window", "run", "seed", "n"], as_index=False)
        .agg(
            ngram_f1=("ngram_f1", "mean"),
            ngram_precision=("ngram_precision", "mean"),
            ngram_recall=("ngram_recall", "mean"),
            sequence_length_ratio_pred_over_gt=("sequence_length_ratio_pred_over_gt", "mean"),
        )
    )
    return (
        per_sample.groupby(["source", "track", "window", "n"], as_index=False)
        .agg(
            n_samples=("run", "nunique"),
            ngram_f1_mean=("ngram_f1", "mean"),
            ngram_f1_sd=("ngram_f1", "std"),
            ngram_precision_mean=("ngram_precision", "mean"),
            ngram_recall_mean=("ngram_recall", "mean"),
            sequence_length_ratio_pred_over_gt_mean=("sequence_length_ratio_pred_over_gt", "mean"),
            sequence_length_ratio_pred_over_gt_sd=("sequence_length_ratio_pred_over_gt", "std"),
        )
    )


def corpus_counts(sequences: list[list[str]], n: int) -> Counter:
    counts = Counter()
    for seq in sequences:
        counts.update(ngrams(seq, n))
    return counts


def transition_probability(counts: Counter) -> np.ndarray:
    mat = np.zeros((len(BEHAVIOR_CLASSES), len(BEHAVIOR_CLASSES)), dtype=float)
    for i, src in enumerate(BEHAVIOR_CLASSES):
        row_total = sum(counts.get((src, dst), 0) for dst in BEHAVIOR_CLASSES)
        if row_total:
            for j, dst in enumerate(BEHAVIOR_CLASSES):
                mat[i, j] = counts.get((src, dst), 0) / row_total
    return mat


def transition_table(gt_sequences: dict[int, list[str]], pred_sequences: dict[tuple[int, str], list[str]], specs: list[dict]) -> pd.DataFrame:
    gt_counts = corpus_counts(list(gt_sequences.values()), 2)
    gt_prob = transition_probability(gt_counts)
    rows = []
    for spec in specs:
        pred_counts = corpus_counts([pred_sequences[(movie_id, spec["run"])] for movie_id in gt_sequences], 2)
        pred_prob = transition_probability(pred_counts)
        for i, src in enumerate(BEHAVIOR_CLASSES):
            for j, dst in enumerate(BEHAVIOR_CLASSES):
                rows.append({
                    **spec,
                    "from_behavior": src,
                    "to_behavior": dst,
                    "gt_probability": gt_prob[i, j],
                    "model_probability": pred_prob[i, j],
                    "difference_model_minus_gt": pred_prob[i, j] - gt_prob[i, j],
                })
    return pd.DataFrame(rows)


def token_window(seq: list[str], max_bouts: int) -> tuple[int, int]:
    if len(seq) <= max_bouts:
        return 0, len(seq)
    # Movie 6 has many lunge bouts; center on the first non-lunge motif if present.
    center = next((i for i, token in enumerate(seq) if token != "lunge"), max_bouts // 2)
    start = max(0, min(center - max_bouts // 3, len(seq) - max_bouts))
    return start, start + max_bouts


def draw_token_strip(ax: plt.Axes, sequences: list[tuple[str, list[str]]], start: int, end: int) -> None:
    y_positions = np.arange(len(sequences))[::-1]
    ax.set_xlim(start, end)
    ax.set_ylim(-0.6, len(sequences) - 0.4)
    for y, (label, seq) in zip(y_positions, sequences):
        for idx in range(start, min(end, len(seq))):
            token = seq[idx]
            if token not in BEHAVIOR_COLORS:
                continue
            ax.add_patch(Rectangle((idx, y - 0.33), 0.9, 0.66, facecolor=BEHAVIOR_COLORS[token], edgecolor="none"))
        ax.text(start - 1.5, y, label, ha="right", va="center", fontsize=8)
    ax.set_yticks([])
    ax.set_xlabel("Bout-order index")
    ax.grid(axis="x", color=NEUTRAL["grid"], alpha=0.22)
    handles = [
        Rectangle((0, 0), 1, 1, facecolor=BEHAVIOR_COLORS[b], edgecolor="none", label=BEHAVIOR_LABELS[b])
        for b in BEHAVIOR_CLASSES
    ]
    ax.legend(handles=handles, frameon=False, ncol=3, fontsize=7, loc="upper right")
    style_axis(ax, grid_axis="")


def plot_figure(
    summary: pd.DataFrame,
    transition: pd.DataFrame,
    gt_sequences: dict[int, list[str]],
    pred_sequences: dict[tuple[int, str], list[str]],
    specs: list[dict],
    args: argparse.Namespace,
) -> None:
    apply_style()
    fig = plt.figure(figsize=(13.8, 9.2))
    gs = fig.add_gridspec(2, 2, left=0.07, right=0.98, top=0.94, bottom=0.08, hspace=0.38, wspace=0.30)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(2)]

    ax = axes[0]
    order = [
        "GT composition shuffle",
        "GT Markov baseline",
        "16-frame window length unigram",
        "8-frame window length unigram",
        "16-frame window",
        "8-frame window",
    ]
    for track in order:
        sub = summary[summary["track"].eq(track)].sort_values("n")
        if sub.empty:
            continue
        color = NULL_COLORS.get(
            track,
            WINDOW_COLORS.get("w16s8" if str(track).startswith("16") else "w8s4"),
        )
        linestyle = "--" if "baseline" in track or "shuffle" in track or "unigram" in track else "-"
        marker = "s" if "baseline" in track or "shuffle" in track or "unigram" in track else "o"
        alpha = 0.55 if "unigram" in track else 1.0
        ax.errorbar(
            sub["n"],
            sub["ngram_f1_mean"],
            yerr=sub["ngram_f1_sd"].fillna(0),
            marker=marker,
            linestyle=linestyle,
            linewidth=2.0,
            capsize=2,
            color=color,
            label=track,
            alpha=alpha,
        )
    ax.set_xticks(range(1, args.max_n + 1))
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Bout n-gram length")
    ax.set_ylabel("Multiset n-gram F1")
    panel_label(ax, "A")
    ax.set_title("Bout-sequence recovery against null ethograms", loc="left")
    ax.legend(frameon=False, fontsize=8)
    style_axis(ax)

    ax = axes[1]
    length_order = [
        "GT composition shuffle",
        "GT Markov baseline",
        "16-frame window",
        "8-frame window",
    ]
    n2 = summary[summary["n"].eq(2)].set_index("track").reindex(length_order).reset_index()
    x = np.arange(len(n2))
    colors = [NULL_COLORS.get(t, WINDOW_COLORS.get("w16s8" if str(t).startswith("16") else "w8s4")) for t in n2["track"]]
    ax.bar(x, n2["sequence_length_ratio_pred_over_gt_mean"], yerr=n2["sequence_length_ratio_pred_over_gt_sd"].fillna(0), color=colors, width=0.68)
    ax.axhline(1.0, color=NEUTRAL["dark"], linewidth=1.0, linestyle="--")
    ax.set_xticks(x, n2["track"], rotation=25, ha="right")
    ax.set_ylabel("Predicted / GT sequence length")
    panel_label(ax, "B")
    ax.set_title("Bout-sequence length recovery", loc="left")
    style_axis(ax)

    ax = axes[2]
    w8 = transition[transition["window"].eq("w8s4")].copy()
    matrix = np.zeros((len(BEHAVIOR_CLASSES), len(BEHAVIOR_CLASSES)))
    for _, row in w8.iterrows():
        i = BEHAVIOR_CLASSES.index(row["from_behavior"])
        j = BEHAVIOR_CLASSES.index(row["to_behavior"])
        matrix[i, j] = row["difference_model_minus_gt"]
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-0.35, vmax=0.35)
    ax.set_xticks(np.arange(len(BEHAVIOR_CLASSES)), [BEHAVIOR_LABELS[b] for b in BEHAVIOR_CLASSES], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(BEHAVIOR_CLASSES)), [BEHAVIOR_LABELS[b] for b in BEHAVIOR_CLASSES])
    for i in range(len(BEHAVIOR_CLASSES)):
        for j in range(len(BEHAVIOR_CLASSES)):
            ax.text(j, i, f"{matrix[i, j]:+.2f}", ha="center", va="center", fontsize=7)
    panel_label(ax, "C")
    ax.set_title("8-frame transition difference from primary labels", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Model - GT probability")

    ax = axes[3]
    token_movie = args.token_movie_id
    gt_seq = gt_sequences[token_movie]
    w16_run = next(spec["run"] for spec in specs if spec["window"] == "w16s8")
    w8_run = next(spec["run"] for spec in specs if spec["window"] == "w8s4")
    start, end = token_window(gt_seq, args.token_max_bouts)
    draw_token_strip(
        ax,
        [
            ("Primary labels", gt_seq),
            ("16-frame", pred_sequences[(token_movie, w16_run)]),
            ("8-frame", pred_sequences[(token_movie, w8_run)]),
        ],
        start,
        end,
    )
    panel_label(ax, "D")
    ax.set_title(f"Movie {token_movie} bout-order tokens", loc="left")

    save_figure(fig, args.figure_dir / args.output_name)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.random_seed)
    heldout = pd.read_csv(args.table_dir / "fly_video_annotation_surface_heldout.csv")
    gt_sequences, pred_sequences, specs = load_sequences(args.eval_dir, heldout, args.prediction_type)
    model_rows = model_metric_rows(gt_sequences, pred_sequences, specs, args.max_n, args.prediction_type)
    null_rows = null_metric_rows(
        gt_sequences,
        pred_sequences,
        specs,
        args.max_n,
        args.null_repeats,
        rng,
        args.prediction_type,
    )
    rows = pd.DataFrame([*model_rows, *null_rows])
    summary = summarize(rows)
    transitions = transition_table(gt_sequences, pred_sequences, specs)

    rows.to_csv(args.table_dir / f"{args.output_name}_video_level_and_nulls.csv", index=False)
    summary.to_csv(args.table_dir / f"{args.output_name}_summary.csv", index=False)
    transitions.to_csv(args.table_dir / f"{args.output_name}_bigram_transition_table.csv", index=False)
    plot_figure(summary, transitions, gt_sequences, pred_sequences, specs, args)

    print("Fly-v-Fly sequence-structure summary:")
    print(summary[summary["n"].isin([2, 3, 4])][["track", "n", "ngram_f1_mean", "ngram_f1_sd", "sequence_length_ratio_pred_over_gt_mean"]].to_string(index=False))
    print(f"Wrote {args.figure_dir / (args.output_name + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
