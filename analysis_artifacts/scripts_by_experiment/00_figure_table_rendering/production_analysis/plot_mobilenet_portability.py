"""Create production MobileNetV3 cross-backbone analysis figures and tables."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


THIS_DIR = Path(__file__).resolve().parent
STYLE_DIR = THIS_DIR.parent / "current_analysis"
if str(STYLE_DIR) not in sys.path:
    sys.path.insert(0, str(STYLE_DIR))

from behaviorscope_figure_style import (  # noqa: E402
    BACKBONE_COLORS,
    HEAD_COLORS,
    NEUTRAL,
    apply_style,
    panel_label,
    save_figure,
)


METRICS = {
    "frame_macro_f1_present_behaviors": "Frame macro-F1",
    "bout_macro_f1_iou25_present_behaviors": "Bout F1 @0.25",
    "bout_macro_f1_iou50_present_behaviors": "Bout F1 @0.50",
}
BACKBONE_LABELS = {
    "yolo_sppf": "YOLO/SPPF",
    "mobilenetv3_native": "MobileNetV3",
}
HEAD_LABELS = {"attention": "Attention-256", "lstm": "LSTM-256"}
MARKERS = {"attention": "o", "lstm": "s"}
MINUS = "\N{MINUS SIGN}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite_root",
        type=Path,
        default=Path("outputs/controlled_comparison_runs/run_20260528_205727"),
    )
    parser.add_argument(
        "--yolo_table_dir",
        type=Path,
        default=Path("Manuscript_penultimate/tables/production/yolo_sppf_main"),
    )
    parser.add_argument(
        "--figure_dir",
        type=Path,
        default=Path("Manuscript_penultimate/figures/production/mobilenet_portability"),
    )
    parser.add_argument(
        "--table_dir",
        type=Path,
        default=Path("Manuscript_penultimate/tables/production/mobilenet_portability"),
    )
    parser.add_argument("--prediction_type", default="smoothed_frames")
    parser.add_argument("--output_name", default="fig_mobilenet_portability")
    return parser.parse_args()


def numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_yolo_per_video(table_dir: Path, prediction_type: str) -> pd.DataFrame:
    df = pd.read_csv(table_dir / "yolo_lstm_attention_per_video_long.csv")
    df = df[
        df["prediction_type"].eq(prediction_type)
        & df["stream"].eq("full")
        & df["capacity"].eq(256)
        & df["head"].isin(["attention", "lstm"])
    ].copy()
    df["backbone"] = "yolo_sppf"
    df["model_base"] = "yolo_sppf_full_" + df["head"] + "256"
    return numeric(df, ["seed", "n_frames", "gt_bouts_total", "pred_bouts_total", *METRICS.keys()])


def load_mobilenet_per_video(suite_root: Path, prediction_type: str) -> pd.DataFrame:
    path = suite_root / "summaries" / "controlled_per_video_summary.csv"
    df = pd.read_csv(path)
    df = df[
        df["prediction_type"].eq(prediction_type)
        & df["family"].isin(["attention", "lstm"])
        & df["model_base"].str.startswith("mobilenetv3_native_full_", na=False)
    ].copy()
    df["head"] = df["family"]
    df["backbone"] = "mobilenetv3_native"
    return numeric(df, ["seed", "n_frames", "gt_bouts_total", "pred_bouts_total", *METRICS.keys()])


def seed_level(per_video: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["backbone", "head", "model_base", "seed"]
    for keys, grp in per_video.groupby(group_cols, sort=True):
        row = dict(zip(group_cols, keys))
        row["n_videos"] = int(grp[["split", "video_id"]].drop_duplicates().shape[0])
        row["n_frames"] = int(grp["n_frames"].sum())
        row["gt_bouts_total"] = int(grp["gt_bouts_total"].sum())
        row["pred_bouts_total"] = int(grp["pred_bouts_total"].sum())
        for metric in METRICS:
            row[metric] = float(grp[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def model_summary(seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (backbone, head), grp in seed_df.groupby(["backbone", "head"], sort=True):
        row = {
            "backbone": backbone,
            "head": head,
            "n_seeds": int(grp["seed"].nunique()),
            "n_videos": int(grp["n_videos"].iloc[0]),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(grp[metric].mean())
            row[f"{metric}_sd"] = float(grp[metric].std(ddof=1))
        row["pred_bout_ratio_mean"] = float((grp["pred_bouts_total"] / grp["gt_bouts_total"]).mean())
        row["pred_bout_ratio_sd"] = float((grp["pred_bouts_total"] / grp["gt_bouts_total"]).std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 5000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    mean = float(values.mean())
    if values.size < 2:
        return mean, mean, mean
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return mean, float(lo), float(hi)


def paired_effects(per_video: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(20260602)
    rows = []
    long_rows = []
    for head in ["attention", "lstm"]:
        yolo = per_video[(per_video["backbone"].eq("yolo_sppf")) & (per_video["head"].eq(head))]
        mobile = per_video[(per_video["backbone"].eq("mobilenetv3_native")) & (per_video["head"].eq(head))]
        merged = yolo.merge(
            mobile,
            on=["split", "video_id", "seed"],
            suffixes=("_yolo", "_mobilenet"),
            validate="one_to_one",
        )
        if merged.empty:
            raise ValueError(f"No paired rows for {head}")
        for metric in METRICS:
            delta_col = f"delta_{metric}"
            merged[delta_col] = merged[f"{metric}_mobilenet"] - merged[f"{metric}_yolo"]
            mean, lo, hi = bootstrap_mean_ci(merged[delta_col].to_numpy(), rng)
            rows.append(
                {
                    "head": head,
                    "metric": metric,
                    "comparison": "MobileNetV3 - YOLO/SPPF",
                    "mean_delta": mean,
                    "ci_low": lo,
                    "ci_high": hi,
                    "n_paired_video_seed_units": int(len(merged)),
                }
            )
        merged["bout_count_error_reduction"] = (
            (merged["pred_bouts_total_yolo"] - merged["gt_bouts_total_yolo"]).abs()
            - (merged["pred_bouts_total_mobilenet"] - merged["gt_bouts_total_mobilenet"]).abs()
        )
        mean, lo, hi = bootstrap_mean_ci(merged["bout_count_error_reduction"].to_numpy(), rng)
        rows.append(
            {
                "head": head,
                "metric": "bout_count_error_reduction",
                "comparison": "MobileNetV3 - YOLO/SPPF",
                "mean_delta": mean,
                "ci_low": lo,
                "ci_high": hi,
                "n_paired_video_seed_units": int(len(merged)),
            }
        )
        keep = ["split", "video_id", "seed", "head", *[f"delta_{m}" for m in METRICS], "bout_count_error_reduction"]
        merged["head"] = head
        long_rows.append(merged[keep].copy())
    return pd.concat(long_rows, ignore_index=True), pd.DataFrame(rows)


def load_training(suite_root: Path, seed_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    epoch_rows = []
    for _, row in seed_df.iterrows():
        backbone = row["backbone"]
        head = row["head"]
        seed = int(row["seed"])
        if backbone == "yolo_sppf":
            model_name = f"yolo_sppf_full_{head}256_seed{seed}"
        else:
            model_name = f"mobilenetv3_native_full_{head}256_seed{seed}"
        train_dir = suite_root / "neural_training" / model_name
        log_path = train_dir / "training_log.csv"
        config_path = train_dir / "config.json"
        log = pd.read_csv(log_path)
        log["backbone"] = backbone
        log["head"] = head
        log["seed"] = seed
        log["model_name"] = model_name
        epoch_rows.append(log.copy())
        best = log.loc[pd.to_numeric(log["val_macroF1"], errors="coerce").idxmax()]
        total_elapsed = float(pd.to_numeric(log["elapsed_s"], errors="coerce").sum())
        params = np.nan
        combined_dim = np.nan
        feature_dim = np.nan
        if config_path.is_file():
            with config_path.open("r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            params = cfg.get("total_params", np.nan)
            combined_dim = cfg.get("combined_dim", np.nan)
            feature_dim = cfg.get("feature_dim", np.nan)
        rows.append(
            {
                "backbone": backbone,
                "head": head,
                "seed": seed,
                "model_name": model_name,
                "best_val_macro_f1": float(best["val_macroF1"]),
                "best_val_loss": float(best["val_loss"]),
                "total_train_elapsed_s": total_elapsed,
                "mean_gpu_util_pct": float(pd.to_numeric(log.get("gpu_util_pct"), errors="coerce").mean()),
                "max_vram_gb_used": float(pd.to_numeric(log.get("vram_gb_used"), errors="coerce").max()),
                "mean_cpu_pct": float(pd.to_numeric(log.get("cpu_pct"), errors="coerce").mean()),
                "max_ram_gb_used": float(pd.to_numeric(log.get("ram_gb_used"), errors="coerce").max()),
                "total_params": params,
                "combined_dim": combined_dim,
                "feature_dim": feature_dim,
            }
        )
    return pd.DataFrame(rows), pd.concat(epoch_rows, ignore_index=True)


def load_provenance(suite_root: Path) -> pd.DataFrame:
    path = suite_root / "summaries" / "pose_backbone_provenance.csv"
    df = pd.read_csv(path)
    return numeric(
        df,
        [
            "feature_dim_per_crop",
            "raw_visual_dim_per_frame",
            "parameter_count",
            "pose_box_map5095",
            "pose_pose_map5095",
            "pose_pck05",
        ],
    )


def draw_provenance(ax, prov: pd.DataFrame) -> None:
    x = np.arange(2)
    width = 0.34
    for i, backbone in enumerate(["yolo_sppf", "mobilenetv3_native"]):
        row = prov[prov["visual_backend"].eq(backbone)].iloc[0]
        vals = [row["feature_dim_per_crop"], row["raw_visual_dim_per_frame"]]
        ax.bar(x + (i - 0.5) * width, vals, width=width, color=BACKBONE_COLORS[backbone], alpha=0.88)
        for xi, val in zip(x + (i - 0.5) * width, vals):
            ax.annotate(
                f"{int(val)}",
                xy=(xi, val),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.82),
            )
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(["Feature dim\nper crop", "Raw visual dim\nper frame"])
    ax.set_ylabel("Dimensions (log scale)")
    ax.set_title("Backbone feature provenance", loc="left")
    handles = [
        Line2D([0], [0], color=BACKBONE_COLORS[b], lw=8, label=BACKBONE_LABELS[b])
        for b in ["yolo_sppf", "mobilenetv3_native"]
    ]
    ax.legend(handles=handles, frameon=False, loc="upper left")


def draw_validation_scatter(ax, seed_df: pd.DataFrame, training: pd.DataFrame) -> None:
    merged = seed_df.merge(training, on=["backbone", "head", "seed"], validate="one_to_one")
    for _, row in merged.iterrows():
        ax.scatter(
            row["best_val_macro_f1"],
            row["frame_macro_f1_present_behaviors"],
            s=70,
            marker=MARKERS[row["head"]],
            color=BACKBONE_COLORS[row["backbone"]],
            edgecolor=NEUTRAL["dark"],
            linewidth=0.6,
            alpha=0.88,
        )
    lo = min(merged["best_val_macro_f1"].min(), merged["frame_macro_f1_present_behaviors"].min()) - 0.02
    hi = max(merged["best_val_macro_f1"].max(), merged["frame_macro_f1_present_behaviors"].max()) + 0.02
    ax.plot([lo, hi], [lo, hi], ls="--", color=NEUTRAL["mid"], lw=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Validation macro-F1")
    ax.set_ylabel("Held-out frame macro-F1")
    ax.set_title("Seed-level validation vs held-out behavior", loc="left")
    handles = [
        Line2D([0], [0], marker=MARKERS[h], color="none", markerfacecolor=NEUTRAL["mid"], markeredgecolor=NEUTRAL["dark"], markersize=7, label=HEAD_LABELS[h])
        for h in ["attention", "lstm"]
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right")


def draw_metric_bars(ax, summary: pd.DataFrame) -> None:
    metrics = list(METRICS)
    centers = np.arange(len(metrics))
    offsets = {
        ("yolo_sppf", "attention"): -0.27,
        ("yolo_sppf", "lstm"): -0.09,
        ("mobilenetv3_native", "attention"): 0.09,
        ("mobilenetv3_native", "lstm"): 0.27,
    }
    width = 0.16
    for backbone in ["yolo_sppf", "mobilenetv3_native"]:
        for head in ["attention", "lstm"]:
            row = summary[(summary["backbone"].eq(backbone)) & (summary["head"].eq(head))].iloc[0]
            vals = [row[f"{m}_mean"] for m in metrics]
            errs = [row[f"{m}_sd"] for m in metrics]
            color = BACKBONE_COLORS[backbone]
            hatch = "" if head == "attention" else "//"
            ax.bar(
                centers + offsets[(backbone, head)],
                vals,
                width=width,
                yerr=errs,
                color=color,
                alpha=0.82 if head == "attention" else 0.48,
                edgecolor=NEUTRAL["dark"],
                linewidth=0.4,
                hatch=hatch,
                capsize=2,
            )
    ax.set_xticks(centers)
    ax.set_xticklabels([METRICS[m] for m in metrics])
    ax.set_ylim(0.40, 0.80)
    ax.set_ylabel("Held-out score")
    ax.set_title("Held-out MARS performance by backbone and head", loc="left")
    handles = [
        Line2D([0], [0], color=BACKBONE_COLORS["yolo_sppf"], lw=8, label="YOLO/SPPF attention"),
        Line2D([0], [0], color=BACKBONE_COLORS["yolo_sppf"], lw=8, alpha=0.45, label="YOLO/SPPF LSTM"),
        Line2D([0], [0], color=BACKBONE_COLORS["mobilenetv3_native"], lw=8, label="MobileNetV3 attention"),
        Line2D([0], [0], color=BACKBONE_COLORS["mobilenetv3_native"], lw=8, alpha=0.45, label="MobileNetV3 LSTM"),
    ]
    ax.legend(handles=handles, ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))


def draw_paired_effects(ax, paired_summary: pd.DataFrame) -> None:
    rows = paired_summary[paired_summary["metric"].isin(METRICS.keys())].copy()
    order = [
        ("attention", "frame_macro_f1_present_behaviors"),
        ("lstm", "frame_macro_f1_present_behaviors"),
        ("attention", "bout_macro_f1_iou25_present_behaviors"),
        ("lstm", "bout_macro_f1_iou25_present_behaviors"),
        ("attention", "bout_macro_f1_iou50_present_behaviors"),
        ("lstm", "bout_macro_f1_iou50_present_behaviors"),
    ]
    labels = []
    ys = np.arange(len(order))[::-1]
    for y, (head, metric) in zip(ys, order):
        row = rows[(rows["head"].eq(head)) & (rows["metric"].eq(metric))].iloc[0]
        mean = row["mean_delta"]
        lo = row["ci_low"]
        hi = row["ci_high"]
        color = HEAD_COLORS[head]
        ax.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]], fmt=MARKERS[head], color=color, capsize=3, markersize=6)
        ax.annotate(
            f"{mean:+.3f}".replace("-", MINUS),
            xy=(mean, y),
            xytext=(10, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
            color=NEUTRAL["dark"],
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.86),
        )
        labels.append(f"{HEAD_LABELS[head]}\n{METRICS[metric]}")
    ax.axvline(0, color=NEUTRAL["mid"], lw=1)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Paired delta: MobileNetV3 " + MINUS + " YOLO/SPPF")
    ax.set_title("Per-video paired cross-backbone effects", loc="left")
    ax.grid(axis="x", alpha=0.25)


def draw_resource_panel(ax, training: pd.DataFrame) -> None:
    grouped = training.groupby(["backbone", "head"], as_index=False).agg(
        train_hours=("total_train_elapsed_s", lambda x: float(np.mean(x) / 3600.0)),
        train_hours_sd=("total_train_elapsed_s", lambda x: float(np.std(x, ddof=1) / 3600.0)),
        vram=("max_vram_gb_used", "mean"),
        gpu=("mean_gpu_util_pct", "mean"),
    )
    labels = []
    x = np.arange(len(grouped))
    colors = [BACKBONE_COLORS[b] for b in grouped["backbone"]]
    alphas = [0.88 if h == "attention" else 0.50 for h in grouped["head"]]
    for i, row in grouped.iterrows():
        ax.bar(i, row["train_hours"], yerr=row["train_hours_sd"], color=colors[i], alpha=alphas[i], edgecolor=NEUTRAL["dark"], linewidth=0.4, capsize=2)
        labels.append(f"{BACKBONE_LABELS[row['backbone']]}\n{HEAD_LABELS[row['head']]}")
        ax.text(
            i,
            max(0.055, row["train_hours"] * 0.08),
            f"VRAM {row['vram']:.1f} GB",
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=0,
            bbox=dict(boxstyle="round,pad=0.16", facecolor="white", edgecolor="none", alpha=0.82),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Training time per seed (h)")
    ax.set_title("Measured training footprint", loc="left")


def draw_training_qc(epoch_df: pd.DataFrame, figure_dir: Path, output_name: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 7.2), constrained_layout=True)
    specs = [
        ("train_loss", "Training loss", "Loss"),
        ("val_loss", "Validation loss", "Loss"),
        ("val_macroF1", "Validation macro-F1", "Macro-F1"),
        ("vram_gb_used", "VRAM use", "GB"),
    ]
    for ax, (col, title, ylabel) in zip(axes.flat, specs):
        for backbone in ["yolo_sppf", "mobilenetv3_native"]:
            for head in ["attention", "lstm"]:
                grp = epoch_df[(epoch_df["backbone"].eq(backbone)) & (epoch_df["head"].eq(head))].copy()
                if grp.empty:
                    continue
                grp[col] = pd.to_numeric(grp[col], errors="coerce")
                mean = grp.groupby("epoch", as_index=False)[col].mean()
                sd = grp.groupby("epoch", as_index=False)[col].std()
                color = BACKBONE_COLORS[backbone]
                ls = "-" if head == "attention" else "--"
                label = f"{BACKBONE_LABELS[backbone]} {HEAD_LABELS[head].replace('-256', '')}"
                ax.plot(mean["epoch"], mean[col], color=color, ls=ls, lw=1.9, label=label)
                if sd[col].notna().any():
                    ax.fill_between(
                        mean["epoch"].to_numpy(),
                        (mean[col] - sd[col].fillna(0)).to_numpy(),
                        (mean[col] + sd[col].fillna(0)).to_numpy(),
                        color=color,
                        alpha=0.10,
                        linewidth=0,
                    )
        ax.set_title(title, loc="left")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
    for label, ax in zip(["A", "B", "C", "D"], axes.flat):
        panel_label(ax, label, x=-0.12, y=1.13)
    axes[0, 0].legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(1.05, 1.35))
    fig.suptitle("MobileNetV3 cross-backbone training QC relative to YOLO/SPPF", fontsize=13)
    save_figure(fig, figure_dir / output_name)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    apply_style()
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)

    yolo = load_yolo_per_video(args.yolo_table_dir, args.prediction_type)
    mobile = load_mobilenet_per_video(args.suite_root, args.prediction_type)
    per_video = pd.concat([yolo, mobile], ignore_index=True, sort=False)
    seed_df = seed_level(per_video)
    summary = model_summary(seed_df)
    paired_long, paired_summary = paired_effects(per_video)
    training, epoch_training = load_training(args.suite_root, seed_df)
    provenance = load_provenance(args.suite_root)

    per_video.to_csv(args.table_dir / "mobilenet_portability_per_video_long.csv", index=False)
    seed_df.to_csv(args.table_dir / "mobilenet_portability_seed_summary.csv", index=False)
    summary.to_csv(args.table_dir / "mobilenet_portability_model_summary.csv", index=False)
    paired_long.to_csv(args.table_dir / "mobilenet_portability_paired_video_deltas.csv", index=False)
    paired_summary.to_csv(args.table_dir / "mobilenet_portability_paired_effects.csv", index=False)
    training.to_csv(args.table_dir / "mobilenet_portability_training_resource_summary.csv", index=False)
    epoch_training.to_csv(args.table_dir / "mobilenet_portability_training_epoch_long.csv", index=False)
    provenance.to_csv(args.table_dir / "mobilenet_portability_pose_backbone_provenance.csv", index=False)

    fig = plt.figure(figsize=(13.2, 9.2), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.04], hspace=0.52, wspace=0.32)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    draw_provenance(ax_a, provenance)
    draw_validation_scatter(ax_b, seed_df, training)
    draw_metric_bars(ax_c, summary)
    draw_paired_effects(ax_d, paired_summary)
    for label, ax in zip(["A", "B", "C", "D"], [ax_a, ax_b, ax_c, ax_d]):
        panel_label(ax, label, x=-0.12, y=1.12)
    fig.suptitle("MobileNetV3 cross-backbone generalization: native pose features reused for MARS behavior decoding", fontsize=14, y=0.98)
    save_figure(fig, args.figure_dir / args.output_name)
    plt.close(fig)

    fig2 = plt.figure(figsize=(8.6, 4.6), constrained_layout=True)
    ax = fig2.add_subplot(1, 1, 1)
    draw_resource_panel(ax, training)
    panel_label(ax, "A", x=-0.08, y=1.12)
    save_figure(fig2, args.figure_dir / f"{args.output_name}_training_footprint")
    plt.close(fig2)
    draw_training_qc(epoch_training, args.figure_dir, f"{args.output_name}_training_qc")


if __name__ == "__main__":
    main()
