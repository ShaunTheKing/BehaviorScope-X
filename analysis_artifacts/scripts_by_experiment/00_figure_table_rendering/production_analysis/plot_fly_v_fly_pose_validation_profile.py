"""Render a MARS-style Fly-v-Fly pose validation profile.

The Fly YOLO-pose training dataset is not stored in the final manuscript
workspace, but the held-out Fly-v-Fly NPZ cache contains raw YOLO keypoints for
the same checkpoint used by the behavior classifiers. This script compares
those cached keypoints with geometry reconstructed from each public
`movie*_track.mat` file:

  keypoint 0: body centroid
  keypoint 1: left wing
  keypoint 2: right wing
  keypoint 3: positive body-axis endpoint
  keypoint 4: negative body-axis endpoint

The resulting profile mirrors the MARS pose figure: per-keypoint PCK at three
body-length-normalized thresholds, PCK@0.10 by behavior class, and an
OKS-style distribution.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
REPO_ROOT = THIS_DIR.parents[2]
STYLE_DIR = THIS_DIR.parent / "current_analysis"
SHARED_DIR = REPO_ROOT / "shared_scripts"
for path in (STYLE_DIR, SHARED_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from behaviorscope_figure_style import (  # noqa: E402
    NEUTRAL,
    apply_style,
    despine,
    panel_label,
    save_figure,
)
from fly_run_all_eval import (  # noqa: E402
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    actions_to_bouts,
    bouts_to_frames,
    load_actions_file,
)


DEFAULT_MANIFEST = (
    REPO_ROOT
    / "outputs"
    / "controlled_comparison_runs"
    / "fly_run_20260602"
    / "fly"
    / "heldout_npz_cache"
    / "w16s4"
    / "sequence_manifest.json"
)
DEFAULT_AGGRESSION_ROOT = REPO_ROOT / "Fly-v-Fly" / "Aggression"
DEFAULT_TABLE_DIR = ROOT / "tables" / "production" / "fly_v_fly"
DEFAULT_FIG_DIR = ROOT / "figures" / "production" / "fly_v_fly"

KPT_NAMES = ["Centroid", "Left wing", "Right wing", "Axis endpoint 1", "Axis endpoint 2"]
PCK_THRESHOLDS = [(0.05, "PCK@0.05"), (0.10, "PCK@0.10"), (0.20, "PCK@0.20")]
PCK_COLORS = ["#94A3B8", "#3B82F6", "#10B981"]
BEHAVIOR_ORDER = ["lunge", "wing_threat", "charge", "hold", "tussle", "other"]
BEHAVIOR_LABELS = {
    "lunge": "Lunge",
    "wing_threat": "Wing threat",
    "charge": "Charge",
    "hold": "Hold",
    "tussle": "Tussle",
    "other": "Other",
}
OKS_SIGMA = 0.10


def style_axis(ax: plt.Axes) -> None:
    despine(ax)
    ax.grid(axis="y", alpha=0.25)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--aggression_root", type=Path, default=DEFAULT_AGGRESSION_ROOT)
    parser.add_argument("--table_dir", type=Path, default=DEFAULT_TABLE_DIR)
    parser.add_argument("--figure_dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument("--output_name", default="fig_fly_v_fly_pose_profile")
    parser.add_argument(
        "--sample_stride_windows",
        type=int,
        default=1,
        help="Use every nth manifest window. The default uses every window start.",
    )
    return parser.parse_args()


def load_manifest_entries(path: Path, sample_stride_windows: int) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    entries = manifest["splits"]["test"]
    stride = max(1, int(sample_stride_windows))
    return entries[::stride]


def load_tracks(aggression_root: Path, movie_id: int) -> tuple[np.ndarray, np.ndarray]:
    track_path = aggression_root / f"movie{movie_id}" / f"movie{movie_id}_track.mat"
    trk = sio.loadmat(str(track_path), squeeze_me=True, struct_as_record=False)["trk"]
    return trk.data.astype(np.float32), trk.flag_frames.astype(bool)


def gt_keypoints_for_frame(track_data: np.ndarray, frame: int) -> tuple[np.ndarray, np.ndarray]:
    frame_data = track_data[:, frame, :]
    center = frame_data[:, [0, 1]]
    ori = frame_data[:, 2]
    major = frame_data[:, 3]
    left_wing = frame_data[:, [5, 6]]
    right_wing = frame_data[:, [7, 8]]
    axis_vec = np.stack([np.cos(ori), np.sin(ori)], axis=1) * (major[:, None] / 2.0)
    endpoint_pos = center + axis_vec
    endpoint_neg = center - axis_vec
    keypoints = np.stack([center, left_wing, right_wing, endpoint_pos, endpoint_neg], axis=1)
    return keypoints.astype(np.float32), major.astype(np.float32)


def load_primary_labels(aggression_root: Path, movie_id: int, n_frames: int) -> np.ndarray:
    action_path = aggression_root / f"movie{movie_id}" / f"movie{movie_id}_actions.mat"
    return bouts_to_frames(actions_to_bouts(load_actions_file(action_path)), n_frames)


def match_two_animals(gt_xy: np.ndarray, pred_xy: np.ndarray, pred_valid: np.ndarray) -> list[tuple[int, int]]:
    valid_pred = [i for i, ok in enumerate(pred_valid) if bool(ok) and np.all(np.isfinite(pred_xy[i, 0]))]
    if not valid_pred:
        return []
    costs = np.full((2, len(valid_pred)), np.inf, dtype=np.float32)
    for gi in range(2):
        for pj, pi in enumerate(valid_pred):
            costs[gi, pj] = np.linalg.norm(gt_xy[gi, 0] - pred_xy[pi, 0])
    if len(valid_pred) == 1:
        gi = int(np.argmin(costs[:, 0]))
        return [(gi, valid_pred[0])]
    option_a = costs[0, 0] + costs[1, 1]
    option_b = costs[0, 1] + costs[1, 0]
    if option_a <= option_b:
        return [(0, valid_pred[0]), (1, valid_pred[1])]
    return [(0, valid_pred[1]), (1, valid_pred[0])]


def compare_rows(entries: list[dict], manifest_root: Path, aggression_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    tracks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    labels: dict[int, np.ndarray] = {}
    for idx, entry in enumerate(entries, start=1):
        movie_id = int(str(entry["source_video"]).replace("movie", ""))
        frame = int(entry["start_frame"])
        if movie_id not in tracks:
            tracks[movie_id] = load_tracks(aggression_root, movie_id)
            n_frames = tracks[movie_id][0].shape[1]
            labels[movie_id] = load_primary_labels(aggression_root, movie_id, n_frames)
        track_data, flag_frames = tracks[movie_id]
        if frame >= track_data.shape[1] or bool(flag_frames[frame]):
            continue

        npz_path = manifest_root / entry["sequence_npz"]
        with np.load(npz_path, allow_pickle=False) as data:
            pred_xy = data["animal_keypoints_raw"][0].astype(np.float32)
            pred_conf = data["animal_keypoints_conf"][0].astype(np.float32)
            pred_valid = data["pose_mask"][0].astype(bool)

        gt_xy, body_len = gt_keypoints_for_frame(track_data, frame)
        class_idx = int(labels[movie_id][frame])
        class_name = IDX_TO_CLASS[class_idx]
        pairs = match_two_animals(gt_xy, pred_xy, pred_valid)
        matched_gt = {gi for gi, _ in pairs}

        for gi, pi in pairs:
            row = {
                "movie_id": movie_id,
                "frame": frame,
                "gt_animal": gi,
                "pred_slot": pi,
                "matched": True,
                "class_name": class_name,
                "body_len_px": float(body_len[gi]),
            }
            visible = np.all(np.isfinite(gt_xy[gi]), axis=1)
            oks_terms = []
            for k in range(len(KPT_NAMES)):
                if (
                    not visible[k]
                    or not np.all(np.isfinite(pred_xy[pi, k]))
                    or float(pred_conf[pi, k]) <= 0.0
                    or float(body_len[gi]) <= 0.0
                ):
                    row[f"rmse_kpt{k}"] = np.nan
                    for _, label in PCK_THRESHOLDS:
                        row[f"{label.lower().replace('@', '').replace('.', '')}_kpt{k}"] = np.nan
                    continue
                dist = float(np.linalg.norm(gt_xy[gi, k] - pred_xy[pi, k]))
                d_norm = dist / float(body_len[gi])
                row[f"rmse_kpt{k}"] = dist
                row[f"pck005_kpt{k}"] = float(d_norm <= 0.05)
                row[f"pck010_kpt{k}"] = float(d_norm <= 0.10)
                row[f"pck020_kpt{k}"] = float(d_norm <= 0.20)
                oks_terms.append(np.exp(-(d_norm ** 2) / (2.0 * OKS_SIGMA ** 2)))
            row["oks"] = float(np.mean(oks_terms)) if oks_terms else np.nan
            rows.append(row)

        for gi in range(2):
            if gi not in matched_gt:
                rows.append({
                    "movie_id": movie_id,
                    "frame": frame,
                    "gt_animal": gi,
                    "pred_slot": -1,
                    "matched": False,
                    "class_name": class_name,
                    "body_len_px": float(body_len[gi]),
                    "oks": np.nan,
                    **{f"rmse_kpt{k}": np.nan for k in range(len(KPT_NAMES))},
                    **{f"pck005_kpt{k}": np.nan for k in range(len(KPT_NAMES))},
                    **{f"pck010_kpt{k}": np.nan for k in range(len(KPT_NAMES))},
                    **{f"pck020_kpt{k}": np.nan for k in range(len(KPT_NAMES))},
                })

        if idx % 10000 == 0:
            print(f"[fly-pose] compared {idx:,}/{len(entries):,} window starts", flush=True)
    return pd.DataFrame(rows)


def write_summary_tables(raw: pd.DataFrame, table_dir: Path) -> None:
    matched = raw[raw["matched"].astype(bool)].copy()
    keypoint_rows = []
    for k, name in enumerate(KPT_NAMES):
        keypoint_rows.append({
            "keypoint": name,
            "rmse_px": matched[f"rmse_kpt{k}"].mean(),
            "pck_at_0_05": matched[f"pck005_kpt{k}"].mean(),
            "pck_at_0_10": matched[f"pck010_kpt{k}"].mean(),
            "pck_at_0_20": matched[f"pck020_kpt{k}"].mean(),
        })
    pd.DataFrame(keypoint_rows).to_csv(table_dir / "fly_pose_track_keypoint_summary.csv", index=False)

    class_rows = []
    for cname in BEHAVIOR_ORDER:
        sub = matched[matched["class_name"].eq(cname)]
        if sub.empty:
            continue
        row = {"class_name": cname, "n_matched_animals": len(sub), "mean_oks": sub["oks"].mean()}
        for k, name in enumerate(KPT_NAMES):
            row[f"pck_at_0_10_{name.lower().replace(' ', '_')}"] = sub[f"pck010_kpt{k}"].mean()
        class_rows.append(row)
    pd.DataFrame(class_rows).to_csv(table_dir / "fly_pose_track_class_keypoint_pck10.csv", index=False)

    pd.DataFrame([{
        "n_rows": len(raw),
        "n_matched_rows": len(matched),
        "mean_oks": matched["oks"].mean(),
        "mean_body_length_px": matched["body_len_px"].mean(),
        "source": "Fly held-out w16s4 NPZ cached YOLO keypoints versus movie*_track.mat geometry",
    }]).to_csv(table_dir / "fly_pose_track_overall_summary.csv", index=False)


def plot_pose_profile(raw: pd.DataFrame, fig_dir: Path, output_name: str) -> None:
    matched = raw[raw["matched"].astype(bool)].copy()
    fig = plt.figure(figsize=(13.2, 8.8))
    gs = gridspec.GridSpec(
        2,
        2,
        height_ratios=[1.0, 1.0],
        width_ratios=[1.08, 1.0],
        wspace=0.48,
        hspace=0.48,
    )

    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(KPT_NAMES))
    width = 0.27
    pck_cols = [("pck005", "PCK@0.05"), ("pck010", "PCK@0.10"), ("pck020", "PCK@0.20")]
    for i, ((col_prefix, label), color) in enumerate(zip(pck_cols, PCK_COLORS)):
        vals = [matched[f"{col_prefix}_kpt{k}"].mean() for k in range(len(KPT_NAMES))]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=color)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_xticks(x, KPT_NAMES)
    ax.set_ylabel("PCK")
    ax.set_ylim(0, 1.10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    panel_label(ax, "A")
    ax.set_title("Per-keypoint PCK", loc="left", pad=18)
    style_axis(ax)

    ax = fig.add_subplot(gs[1, 0])
    classes = [c for c in BEHAVIOR_ORDER if c in set(matched["class_name"])]
    z = np.full((len(classes), len(KPT_NAMES)), np.nan, dtype=float)
    for ci, cname in enumerate(classes):
        sub = matched[matched["class_name"].eq(cname)]
        for k in range(len(KPT_NAMES)):
            z[ci, k] = sub[f"pck010_kpt{k}"].mean()
    im = ax.imshow(z, aspect="auto", cmap="RdYlGn", vmin=0.60, vmax=1.0)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            val = z[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color="white" if val < 0.78 else "black")
    ax.set_xticks(range(len(KPT_NAMES)), KPT_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(classes)), [BEHAVIOR_LABELS[c] for c in classes])
    ax.set_xlabel("Keypoint")
    ax.set_ylabel("Behavior class")
    ax.grid(False)
    panel_label(ax, "B")
    ax.set_title("PCK@0.10 by behavior class", loc="left")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("PCK@0.10")

    ax = fig.add_subplot(gs[1, 1])
    oks = matched["oks"].dropna().to_numpy(dtype=float)
    mean_oks = float(np.mean(oks))
    ax.hist(oks, bins=60, color="#10B981", edgecolor="none")
    ax.axvline(mean_oks, color="#EF4444", linestyle="--", linewidth=1.2)
    ax.text(mean_oks + 0.012, ax.get_ylim()[1] * 0.95, f"mean = {mean_oks:.3f}", color="#EF4444", fontsize=9, va="top")
    ax.set_xlabel("OKS")
    ax.set_ylabel("Matched animal-frame count")
    ax.set_xlim(0, 1.0)
    panel_label(ax, "C")
    ax.set_title("OKS distribution", loc="left")
    style_axis(ax)

    save_figure(fig, fig_dir / output_name)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    apply_style()
    args.table_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    raw_path = args.table_dir / "fly_pose_track_keypoint_eval_raw.csv"
    entries = load_manifest_entries(args.manifest, args.sample_stride_windows)
    raw = compare_rows(entries, args.manifest.parent, args.aggression_root)
    raw.to_csv(raw_path, index=False)
    write_summary_tables(raw, args.table_dir)
    plot_pose_profile(raw, args.figure_dir, args.output_name)
    print(f"[fly-pose] wrote {raw_path}")
    print(f"[fly-pose] wrote {args.figure_dir / (args.output_name + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
