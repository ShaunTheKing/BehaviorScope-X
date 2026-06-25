"""
Precompute frozen YOLO visual features for BehaviorScope-Y training.

This writes one small NPZ per sequence window containing:
  group_feat   float32/float16 [T, D]
  animal_feat  float32/float16 [T, N, D]

train_y.py can then use --use_feature_cache <output_dir> to train the
temporal/pose/classification head without re-running the frozen YOLO trunk
every epoch.
"""
from __future__ import annotations

import warnings
# Suppress the noisy torch.cuda FutureWarning caused by environments that
# still have the deprecated `pynvml` package installed. Root fix:
# pip uninstall pynvml && pip install nvidia-ml-py.
warnings.filterwarnings(
    "ignore",
    message=".*pynvml.*deprecated.*",
    category=FutureWarning,
)

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from torch.amp import autocast
except ImportError:  # torch < 2.0
    from torch.cuda.amp import autocast  # type: ignore[no-redef]

try:
    from data_y import (
        load_n_manifest,
        MultiAnimalSequenceDataset,
        collate_multi_animal,
        feature_cache_path,
    )
    from model_y import MultiAnimalBehaviorSequenceClassifier, inspect_yolo_keypoint_count
    from utils.pose_features_y import REL_FEATURE_DIM
except Exception:  # pragma: no cover
    from .data_y import (
        load_n_manifest,
        MultiAnimalSequenceDataset,
        collate_multi_animal,
        feature_cache_path,
    )
    from .model_y import MultiAnimalBehaviorSequenceClassifier, inspect_yolo_keypoint_count
    from .utils.pose_features_y import REL_FEATURE_DIM


def parse_args():
    p = argparse.ArgumentParser(description="Precompute BehaviorScope-Y YOLO visual feature cache.")
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--yolo_weights", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"],
                   help="Manifest splits to cache. Default: train val.")
    p.add_argument("--yolo_backbone_end_layer", type=int, default=10)
    p.add_argument("--n_animals", type=int, default=2)
    p.add_argument("--num_keypoints", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action=argparse.BooleanOptionalAction, default=False,
                   help="Pin DataLoader host memory. Disabled by default to avoid CUDA resource mapping failures.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache_dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--skip_invalid_samples", action="store_true")
    p.add_argument("--amp", action="store_true",
                   help="Use CUDA autocast while encoding features. Saved cache dtype is controlled by --cache_dtype.")
    p.add_argument("--amp_dtype", choices=["auto", "fp16", "bf16"], default="auto")
    return p.parse_args()


class IndexedDataset(Dataset):
    def __init__(self, base: Dataset):
        self.base = base

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        return idx, self.base[idx]


def collate_indexed(batch):
    idxs = []
    items = []
    for idx, item in batch:
        if item is None:
            continue
        idxs.append(int(idx))
        items.append(item)
    if not items:
        return None
    inputs, labels = collate_multi_animal(items)
    return idxs, inputs, labels


def resolve_amp_dtype(requested: str) -> str:
    if requested == "auto":
        return "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp16"
    return requested


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits, class_to_idx, idx_to_class, meta = load_n_manifest(args.manifest_path)
    num_frames = int(meta.get("window_size", 32))
    yolo_num_keypoints = inspect_yolo_keypoint_count(args.yolo_weights)
    if "keypoints_per_animal" in meta:
        num_keypoints = int(meta["keypoints_per_animal"])
        if yolo_num_keypoints is not None and yolo_num_keypoints != num_keypoints:
            raise SystemExit(
                f"Manifest keypoints_per_animal={num_keypoints}, but YOLO checkpoint emits "
                f"{yolo_num_keypoints} keypoints."
            )
    elif int(args.num_keypoints) > 0:
        num_keypoints = int(args.num_keypoints)
    elif yolo_num_keypoints is not None:
        num_keypoints = int(yolo_num_keypoints)
    else:
        num_keypoints = 7

    manifest_n_animals = int(meta.get("n_animals", args.n_animals))
    if manifest_n_animals != int(args.n_animals):
        raise SystemExit(
            f"Manifest n_animals={manifest_n_animals}, but --n_animals={args.n_animals}."
        )

    selected_samples = []
    for split in args.splits:
        if split not in splits:
            raise SystemExit(f"Split {split!r} not found. Available: {sorted(splits)}")
        selected_samples.extend(splits[split])
    if not selected_samples:
        raise SystemExit(f"No samples found for splits={args.splits!r}")

    # FAST PRE-FILTER: skip already-cached samples BEFORE building the dataset.
    # The original loop loaded every NPZ + ran the encoder forward for every
    # sample, then checked if the cache file already existed and skipped the
    # write. That wasted N x (NPZ load + RGB normalisation + GPU forward) when
    # only the missing fraction actually needed compute. One glob() of the
    # cache directory + N set lookups is O(N) with zero per-sample syscalls.
    n_total = len(selected_samples)
    if not args.overwrite and out_dir.exists():
        cached_names = {p.name for p in out_dir.glob("*.npz")}
        selected_samples = [
            s for s in selected_samples
            if feature_cache_path(out_dir, s).name not in cached_names
        ]
        n_skip = n_total - len(selected_samples)
        if n_skip:
            print(
                f"[cache] pre-filter: skipping {n_skip:,} already-cached samples; "
                f"will encode {len(selected_samples):,} of {n_total:,}.",
                flush=True,
            )
    if not selected_samples:
        print("[cache] nothing to do — all requested samples already cached.", flush=True)
        return

    dataset = MultiAnimalSequenceDataset(
        samples=selected_samples,
        num_frames=num_frames,
        n_animals=int(args.n_animals),
        num_keypoints=num_keypoints,
        rel_feature_dim=REL_FEATURE_DIM,
        pose_dropout_p_uniform=0.0,
        strict_schema=True,
        skip_invalid=bool(args.skip_invalid_samples),
    )
    loader = DataLoader(
        IndexedDataset(dataset),
        batch_size=int(args.batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_indexed,
        pin_memory=(bool(args.pin_memory) and device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=2 if int(args.num_workers) > 0 else None,
    )

    model = MultiAnimalBehaviorSequenceClassifier(
        num_classes=len(class_to_idx),
        yolo_weights_path=str(args.yolo_weights),
        yolo_backbone_end_layer=int(args.yolo_backbone_end_layer),
        train_backbone=False,
        n_animals=int(args.n_animals),
        num_keypoints=num_keypoints,
        rel_feature_dim=REL_FEATURE_DIM,
        disable_pose_self=True,
        disable_relations=True,
    ).to(device)
    model.eval()
    if not getattr(model, "_visual_alive", False):
        raise SystemExit("Model has no visual stream to precompute.")

    amp_enabled = bool(args.amp) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(args.amp_dtype) if amp_enabled else None
    amp_dtype_torch = (
        (torch.bfloat16 if amp_dtype == "bf16" else torch.float16)
        if amp_enabled else torch.float16
    )
    save_dtype = np.float16 if args.cache_dtype == "float16" else np.float32

    t0 = time.time()
    written = 0
    skipped = 0
    seen = 0
    print(
        f"[cache] samples={len(selected_samples)} splits={args.splits} "
        f"out={out_dir} dtype={args.cache_dtype} feature_dim={model.feature_dim}",
        flush=True,
    )

    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            idxs, inputs, _labels = batch
            B = len(idxs)
            group = inputs["group"].to(device, non_blocking=(device.type == "cuda"))
            animal = inputs["animal"].to(device, non_blocking=(device.type == "cuda"))
            _, N, T = animal.shape[:3]
            with autocast("cuda", dtype=amp_dtype_torch, enabled=amp_enabled):
                group_feat = model._encode_frames(group).detach().cpu().float().numpy()
                an = animal.reshape(B * N, T, *animal.shape[-3:])
                animal_feat = (
                    model._encode_frames(an)
                    .reshape(B, N, T, -1)
                    .permute(0, 2, 1, 3)
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )
            for row_i, sample_idx in enumerate(idxs):
                sample = selected_samples[int(sample_idx)]
                path = feature_cache_path(out_dir, sample)
                seen += 1
                if path.exists() and not args.overwrite:
                    skipped += 1
                    continue
                np.savez_compressed(
                    path,
                    group_feat=group_feat[row_i].astype(save_dtype, copy=False),
                    animal_feat=animal_feat[row_i].astype(save_dtype, copy=False),
                    label=np.asarray(sample.label, dtype=np.int64),
                    sample_id=np.asarray(sample.id),
                )
                written += 1
            if seen and seen % max(int(args.batch) * 10, 100) == 0:
                print(
                    f"[cache] seen={seen}/{len(selected_samples)} written={written} "
                    f"skipped={skipped} elapsed={time.time() - t0:.1f}s",
                    flush=True,
                )

    manifest = {
        "version": "behaviorscope-y-visual-feature-cache-v1",
        "manifest_path": str(args.manifest_path),
        "yolo_weights": str(args.yolo_weights),
        "yolo_backbone_end_layer": int(args.yolo_backbone_end_layer),
        "num_frames": int(num_frames),
        "n_animals": int(args.n_animals),
        "num_keypoints": int(num_keypoints),
        "feature_dim": int(model.feature_dim),
        "cache_dtype": str(args.cache_dtype),
        "splits": list(args.splits),
        "samples_total": int(len(selected_samples)),
        "samples_written": int(written),
        "samples_skipped_existing": int(skipped),
        "elapsed_s": float(time.time() - t0),
    }
    (out_dir / "cache_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[cache] done written={written} skipped={skipped} elapsed={manifest['elapsed_s']:.1f}s "
        f"manifest={out_dir / 'cache_manifest.json'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
