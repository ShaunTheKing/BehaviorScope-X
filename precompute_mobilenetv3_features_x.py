"""Precompute native MobileNetV3-large pose-backbone visual features.

This cache intentionally uses a native MobileNetV3 representation, not a
YOLO/SPPF analogue:

    backbone.features.16 -> adaptive average pooling -> 960-d vector

The output schema matches the existing BehaviorScope-X visual cache so the
downstream behavior heads can swap only the frozen visual descriptor backend:

    group_feat   [T, 960]
    animal_feat  [T, N, 960]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from torch.amp import autocast
except ImportError:  # torch < 2.0
    from torch.cuda.amp import autocast  # type: ignore[no-redef]

try:
    from torchvision import models
except Exception as exc:  # pragma: no cover
    raise ImportError("precompute_mobilenetv3_features_x.py requires torchvision") from exc

try:
    from data_x import load_n_manifest, feature_cache_path, feature_cache_candidate_paths
except Exception:  # pragma: no cover
    from .data_x import load_n_manifest, feature_cache_path, feature_cache_candidate_paths


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Precompute MobileNetV3-large native late-backbone features for BehaviorScope-X.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--checkpoint", required=True,
                   help="BehaviorScopeZ pose checkpoint containing backbone.features.* weights.")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--n_animals", type=int, default=2)
    p.add_argument("--resize_size", type=int, default=640,
                   help="Resize crop tensors before MobileNet. 640 matches the pose-checkpoint training config.")
    p.add_argument("--batch", type=int, default=2,
                   help="Batch of sequence windows. Effective crop batch is batch*T*(1+N).")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache_dtype", choices=["float32", "float16"], default="float32")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_samples", type=int, default=0,
                   help="Optional smoke-test limit. 0 caches all requested samples.")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--amp_dtype", choices=["auto", "fp16", "bf16"], default="auto")
    return p.parse_args()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_amp_dtype(requested: str) -> str:
    if requested == "auto":
        return "bf16" if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else "fp16"
    return requested


class MobileNetFeatureEncoder(torch.nn.Module):
    def __init__(self, checkpoint: Path):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(weights=None).features
        ckpt = torch.load(checkpoint, map_location="cpu")
        if not isinstance(ckpt, dict) or "model" not in ckpt:
            raise ValueError(f"{checkpoint} is not a BehaviorScopeZ checkpoint with a 'model' state dict")
        cfg = ckpt.get("config") if isinstance(ckpt.get("config"), dict) else {}
        backbone_cfg = cfg.get("model", {}).get("backbone", {})
        backbone_name = str(backbone_cfg.get("name", ""))
        if backbone_name != "mobilenet_v3_large":
            raise ValueError(
                f"{checkpoint} embedded backbone={backbone_name!r}; expected 'mobilenet_v3_large'"
            )
        state = {}
        for key, value in ckpt["model"].items():
            if key.startswith("backbone.features."):
                state[key.removeprefix("backbone.features.")] = value
        missing, unexpected = self.backbone.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"MobileNet backbone load mismatch: missing={missing}, unexpected={unexpected}")
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.feature_dim = 960
        self.checkpoint_config = cfg
        self.checkpoint_metrics = ckpt.get("metrics", {})
        self.checkpoint_epoch = ckpt.get("epoch")
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.pool(feat).flatten(1)


class RawWindowDataset(Dataset):
    def __init__(self, samples: list, n_animals: int):
        self.samples = list(samples)
        self.n_animals = int(n_animals)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        with np.load(sample.sequence_npz, allow_pickle=False) as data:
            group = data["group_frames"].astype(np.uint8)
            animal = data["animal_frames"].astype(np.uint8)
        if animal.shape[1] < self.n_animals:
            pad = np.zeros((animal.shape[0], self.n_animals - animal.shape[1], *animal.shape[2:]), dtype=animal.dtype)
            animal = np.concatenate([animal, pad], axis=1)
        elif animal.shape[1] > self.n_animals:
            animal = animal[:, :self.n_animals]
        return idx, group, animal


def collate_raw(batch):
    idxs, groups, animals = [], [], []
    for idx, group, animal in batch:
        idxs.append(int(idx))
        groups.append(torch.from_numpy(group.astype(np.float32) / 255.0).permute(0, 3, 1, 2))
        animals.append(torch.from_numpy(animal.astype(np.float32) / 255.0).permute(1, 0, 4, 2, 3))
    return idxs, torch.stack(groups, dim=0), torch.stack(animals, dim=0)


def normalize_and_resize(x: torch.Tensor, resize_size: int, device: torch.device) -> torch.Tensor:
    x = x.to(device, non_blocking=(device.type == "cuda"))
    if int(resize_size) > 0 and (x.shape[-1] != int(resize_size) or x.shape[-2] != int(resize_size)):
        x = F.interpolate(x, size=(int(resize_size), int(resize_size)), mode="bilinear", align_corners=False)
    mean = IMAGENET_MEAN.to(device=device, dtype=x.dtype)
    std = IMAGENET_STD.to(device=device, dtype=x.dtype)
    return (x - mean) / std


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    splits, class_to_idx, _idx_to_class, meta = load_n_manifest(manifest_path)
    samples = []
    for split in args.splits:
        if split not in splits:
            raise SystemExit(f"Split {split!r} not found in manifest. Available: {sorted(splits)}")
        samples.extend(splits[split])
    if not samples:
        raise SystemExit(f"No samples found for splits={args.splits!r}")
    if int(args.max_samples) > 0:
        samples = samples[:int(args.max_samples)]

    n_total = len(samples)
    if not args.overwrite and out_dir.exists():
        cached_names = {p.name for p in out_dir.glob("*.npz")}
        samples = [
            s for s in samples
            if not any(path.name in cached_names for path in feature_cache_candidate_paths(out_dir, s))
        ]
        skipped_prefilter = n_total - len(samples)
    else:
        skipped_prefilter = 0
    if not samples:
        print("[mobilenet-cache] nothing to do; all requested samples already cached.", flush=True)
    device = torch.device(args.device)
    encoder = MobileNetFeatureEncoder(checkpoint).to(device)
    save_dtype = np.float16 if args.cache_dtype == "float16" else np.float32
    amp_enabled = bool(args.amp) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(args.amp_dtype) if amp_enabled else None
    amp_dtype_torch = (torch.bfloat16 if amp_dtype == "bf16" else torch.float16) if amp_enabled else torch.float16

    loader = DataLoader(
        RawWindowDataset(samples, int(args.n_animals)),
        batch_size=int(args.batch),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_raw,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=2 if int(args.num_workers) > 0 else None,
    )

    t0 = time.time()
    written = 0
    seen = 0
    print(
        f"[mobilenet-cache] samples={len(samples):,}/{n_total:,} splits={args.splits} "
        f"checkpoint={checkpoint} out={out_dir} feature_dim={encoder.feature_dim} "
        f"resize={args.resize_size} dtype={args.cache_dtype}",
        flush=True,
    )
    with torch.no_grad():
        for idxs, group, animal in loader:
            B, T = group.shape[:2]
            N = animal.shape[1]
            flat_group = group.reshape(B * T, *group.shape[2:])
            flat_animal = animal.reshape(B * N * T, *animal.shape[3:])
            with autocast("cuda", dtype=amp_dtype_torch, enabled=amp_enabled):
                group_feat = encoder(normalize_and_resize(flat_group, int(args.resize_size), device))
                animal_feat = encoder(normalize_and_resize(flat_animal, int(args.resize_size), device))
            group_np = group_feat.reshape(B, T, -1).detach().cpu().float().numpy()
            animal_np = (
                animal_feat.reshape(B, N, T, -1)
                .permute(0, 2, 1, 3)
                .detach()
                .cpu()
                .float()
                .numpy()
            )
            for row_i, sample_idx in enumerate(idxs):
                sample = samples[int(sample_idx)]
                path = feature_cache_path(out_dir, sample)
                if path.exists() and not args.overwrite:
                    continue
                np.savez_compressed(
                    path,
                    group_feat=group_np[row_i].astype(save_dtype, copy=False),
                    animal_feat=animal_np[row_i].astype(save_dtype, copy=False),
                    label=np.asarray(sample.label, dtype=np.int64),
                    sample_id=np.asarray(sample.id),
                )
                written += 1
                seen += 1
            if seen and seen % max(int(args.batch) * 10, 50) == 0:
                print(f"[mobilenet-cache] seen={seen}/{len(samples)} written={written}", flush=True)

    param_count = sum(p.numel() for p in encoder.parameters())
    cache_manifest = {
        "version": "behaviorscope-x-visual-feature-cache-v1",
        "visual_backend": "mobilenetv3_large_native",
        "manifest_path": str(manifest_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": encoder.checkpoint_epoch,
        "checkpoint_metrics": encoder.checkpoint_metrics,
        "pose_model_family": "BehaviorScopeZ",
        "pose_model_backbone": "mobilenet_v3_large",
        "pose_model_size": "large",
        "parameter_count": int(param_count),
        "feature_layer": "backbone.features.16",
        "pooling": "adaptive_avg_pool2d",
        "normalization": "imagenet",
        "resize_size": int(args.resize_size),
        "num_frames": int(meta.get("window_size", 32)),
        "n_animals": int(args.n_animals),
        "num_keypoints": int(meta.get("keypoints_per_animal", 7)),
        "feature_dim": int(encoder.feature_dim),
        "raw_visual_dim_per_frame": int((1 + int(args.n_animals)) * encoder.feature_dim),
        "cache_dtype": str(args.cache_dtype),
        "splits": list(args.splits),
        "samples_requested": int(n_total),
        "samples_prefilter_skipped_existing": int(skipped_prefilter),
        "samples_written": int(written),
        "elapsed_s": float(time.time() - t0),
        "class_to_idx": class_to_idx,
    }
    (out_dir / "cache_manifest.json").write_text(json.dumps(cache_manifest, indent=2), encoding="utf-8")
    print(
        f"[mobilenet-cache] done written={written} elapsed={cache_manifest['elapsed_s']:.1f}s "
        f"manifest={out_dir / 'cache_manifest.json'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



