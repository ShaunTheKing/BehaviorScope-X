"""Build BehaviorScope-Y visual feature caches from a fine-tuned DLC model.

This adapter feeds the BehaviorScope-Y RGB crops through the trained DLC pose
model and writes the feature-cache schema consumed by ``train_x.py``:

  group_feat   [T, D] float32/float16
  animal_feat  [T, N, D] float32/float16
  label        scalar int64
  sample_id    scalar string

The default feature tap is ``semantic_concat``: capture the underlying TIMM
HRNet multi-branch feature output inside DLC's backbone, global-average-pool
each branch, and concatenate the pooled vectors. For HRNet-W32 on 224x224
crops this is expected to be 32+64+128+256 = 480 dimensions. This is
deliberately analogous in role to a YOLO-pose SPPF tap: high-level visual
features before the task-specific keypoint decoder.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

try:
    from torch.amp import autocast
except ImportError:  # pragma: no cover
    from torch.cuda.amp import autocast  # type: ignore[no-redef]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Precompute DLC visual feature cache.")
    p.add_argument("--manifest_path", required=True)
    p.add_argument("--dlc_config", required=True)
    p.add_argument("--dlc_project_root", required=True)
    p.add_argument("--shuffle", type=int, default=2)
    p.add_argument("--trainset_index", type=int, default=0)
    p.add_argument("--modelprefix", default="")
    p.add_argument("--snapshot_index", default="-1")
    p.add_argument("--snapshot_path", default="")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--n_animals", type=int, default=2)
    p.add_argument("--num_keypoints", type=int, default=7)
    p.add_argument(
        "--mode",
        choices=["dlc_backbone", "existing_cache"],
        default="dlc_backbone",
        help="Use DLC model features, or explicitly mirror an existing cache.",
    )
    p.add_argument(
        "--source_cache_dir",
        default="",
        help="Existing cache to hardlink/copy when --mode existing_cache is used.",
    )
    p.add_argument(
        "--feature_tap",
        default="semantic_concat",
        help=(
            "DLC backbone feature selection: semantic_concat/auto, last, or a "
            "zero-based branch index. semantic_concat pools and concatenates all "
            "raw TIMM HRNet branches before DLC collapses them."
        ),
    )
    p.add_argument("--feature_dim", type=int, default=0)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache_dtype", choices=["float32", "float16"], default="float16")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp_dtype", choices=["auto", "fp16", "bf16"], default="auto")
    p.add_argument("--strict_hrnet_tap", action=argparse.BooleanOptionalAction, default=True)
    # Accepted for backward compatibility with the old guarded stub.
    p.add_argument("--allow_unvalidated_dlc_feature_tap", action="store_true")
    return p.parse_args()


def find_behaviorscope_scripts(manifest_path: Path) -> Path:
    for parent in [manifest_path.parent, *manifest_path.parents]:
        candidate = parent / "data_x.py"
        if candidate.is_file():
            return candidate.parent
        candidate = parent / "shared_scripts" / "scripts" / "data_x.py"
        if candidate.is_file():
            return candidate.parent
    raise FileNotFoundError(
        "Could not locate BehaviorScope-Y data_x.py "
        f"from manifest path {manifest_path}"
    )


def import_behaviorscope_helpers(manifest_path: Path):
    scripts_dir = find_behaviorscope_scripts(manifest_path)
    sys.path.insert(0, str(scripts_dir))
    from data_x import feature_cache_path, load_n_manifest  # type: ignore

    return feature_cache_path, load_n_manifest, scripts_dir


def selected_manifest_samples(splits: dict[str, list[Any]], split_names: list[str]) -> list[Any]:
    samples: list[Any] = []
    for split in split_names:
        if split not in splits:
            raise SystemExit(f"Split {split!r} not found. Available: {sorted(splits)}")
        samples.extend(splits[split])
    if not samples:
        raise SystemExit(f"No samples found for splits={split_names!r}")
    return samples


def parse_snapshot_index(value: str) -> int | str:
    text = str(value).strip()
    if text.lower() in {"best", "all"}:
        return text.lower()
    try:
        return int(text)
    except ValueError:
        return text


def resolve_dlc_snapshot(args: argparse.Namespace):
    from deeplabcut.pose_estimation_pytorch import data as dlc_data
    from deeplabcut.pose_estimation_pytorch.apis import utils as dlc_utils

    loader = dlc_data.DLCLoader(
        config=str(Path(args.dlc_config).resolve()),
        shuffle=int(args.shuffle),
        trainset_index=int(args.trainset_index),
        modelprefix=str(args.modelprefix),
    )
    if args.snapshot_path:
        snapshot_path = Path(args.snapshot_path).resolve()
        if not snapshot_path.is_file():
            raise FileNotFoundError(f"Missing DLC snapshot: {snapshot_path}")
    else:
        snapshots = dlc_utils.get_model_snapshots(
            parse_snapshot_index(args.snapshot_index),
            loader.model_folder,
            loader.pose_task,
        )
        if len(snapshots) != 1:
            raise SystemExit(
                f"Expected one DLC snapshot for feature extraction, got {len(snapshots)}"
            )
        snapshot_path = snapshots[0].path
    return loader, snapshot_path


def build_dlc_feature_model(args: argparse.Namespace):
    from deeplabcut.pose_estimation_pytorch.apis import utils as dlc_utils
    from deeplabcut.pose_estimation_pytorch.data.preprocessor import AugmentImage, ToTensor
    from deeplabcut.pose_estimation_pytorch.data.transforms import build_transforms

    loader, snapshot_path = resolve_dlc_snapshot(args)
    runner = dlc_utils.get_pose_inference_runner(
        model_config=loader.model_cfg,
        snapshot_path=snapshot_path,
        batch_size=int(args.batch),
        device=args.device,
    )
    model = runner.model
    model.output_features = True
    model.eval()
    model._behaviorscope_raw_backbone_features = None
    model._behaviorscope_backbone_hooked = False
    model._behaviorscope_hook_module = ""
    if hasattr(model.backbone, "model"):
        def _capture_raw_backbone(_module, _inputs, output):
            model._behaviorscope_raw_backbone_features = output

        hook_module = model.backbone.model
        hook_module.register_forward_hook(_capture_raw_backbone)
        model._behaviorscope_backbone_hooked = True
        model._behaviorscope_hook_module = (
            f"{hook_module.__class__.__module__}.{hook_module.__class__.__name__}"
        )
    transform = build_transforms(loader.model_cfg["data"]["inference"])
    preproc = (AugmentImage(transform), ToTensor())
    return model, preproc, loader, snapshot_path


def preprocess_rgb_batch(frames: np.ndarray, preproc: tuple[Any, Any]) -> torch.Tensor:
    if frames.dtype != np.uint8:
        raise ValueError(f"DLC input frames must be uint8 RGB; got {frames.dtype}")
    augment, to_tensor = preproc
    context: dict[str, Any] = {}
    image, context = augment(frames, context)
    tensor, _context = to_tensor(image, context)
    return tensor


def pooled_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 4:
        return torch.nn.functional.adaptive_avg_pool2d(tensor, 1).flatten(1)
    if tensor.ndim > 2:
        return tensor.flatten(2).mean(dim=2)
    raise ValueError(f"Cannot pool feature tensor with shape {tuple(tensor.shape)}")


def flatten_feature_tree(features: Any) -> list[torch.Tensor]:
    if torch.is_tensor(features):
        return [features]
    if isinstance(features, (list, tuple)):
        out: list[torch.Tensor] = []
        for item in features:
            out.extend(flatten_feature_tree(item))
        return out
    if isinstance(features, dict):
        out = []
        for key in sorted(features):
            out.extend(flatten_feature_tree(features[key]))
        return out
    raise TypeError(f"Unsupported DLC feature object: {type(features)!r}")


def select_and_pool_features(features: Any, feature_tap: str) -> tuple[torch.Tensor, str, list[list[int]]]:
    branches = flatten_feature_tree(features)
    if not branches:
        raise ValueError("DLC model returned no backbone feature tensors.")
    branch_shapes = [list(branch.shape) for branch in branches]

    tap = str(feature_tap).lower()
    if tap in {"auto", "semantic_concat", "concat", "hrnet_concat"}:
        pooled = [pooled_tensor(branch) for branch in branches]
        return torch.cat(pooled, dim=1), f"semantic_concat:{len(branches)}_branches", branch_shapes
    if tap in {"last", "deepest", "semantic_last"}:
        return pooled_tensor(branches[-1]), f"last_branch:{len(branches) - 1}", branch_shapes
    try:
        index = int(tap)
    except ValueError as exc:
        raise ValueError(
            f"Unknown --feature_tap {feature_tap!r}; use semantic_concat, last, or an integer branch index."
        ) from exc
    if index < 0:
        index += len(branches)
    if index < 0 or index >= len(branches):
        raise IndexError(f"Feature branch index {index} outside 0..{len(branches) - 1}")
    return pooled_tensor(branches[index]), f"branch:{index}", branch_shapes


def validate_hrnet_w32_semantic_tap(
    *,
    args: argparse.Namespace,
    loader: Any,
    model: torch.nn.Module,
    feature_dim: int,
    feature_branch_shapes: list[list[int]],
    feature_source: str,
) -> dict[str, Any]:
    backbone_cfg = loader.model_cfg.get("model", {}).get("backbone", {})
    backbone_type = str(backbone_cfg.get("type", ""))
    model_name = str(backbone_cfg.get("model_name", ""))
    tap = str(args.feature_tap).lower()
    strict = bool(args.strict_hrnet_tap) and not bool(args.allow_unvalidated_dlc_feature_tap)
    validation = {
        "strict": strict,
        "backbone_type": backbone_type,
        "model_name": model_name,
        "requested_feature_tap": str(args.feature_tap),
        "feature_source": feature_source,
        "hooked_module": getattr(model, "_behaviorscope_hook_module", ""),
        "hook_fired": feature_source == "forward_hook:PoseModel.backbone.model",
        "branch_count": len(feature_branch_shapes),
        "branch_channels": [
            int(shape[1]) for shape in feature_branch_shapes if len(shape) >= 2
        ],
        "feature_dim": int(feature_dim),
    }
    if not strict or tap not in {"auto", "semantic_concat", "concat", "hrnet_concat"}:
        validation["status"] = "not_required"
        return validation
    if backbone_type != "HRNet" or model_name != "hrnet_w32":
        validation["status"] = "not_hrnet_w32"
        return validation

    expected_channels = [32, 64, 128, 256]
    expected_dim = sum(expected_channels)
    failures = []
    if not validation["hook_fired"]:
        failures.append("forward hook on PoseModel.backbone.model did not provide features")
    if validation["branch_count"] != 4:
        failures.append(f"expected 4 HRNet branches, got {validation['branch_count']}")
    if validation["branch_channels"] != expected_channels:
        failures.append(
            f"expected branch channels {expected_channels}, got {validation['branch_channels']}"
        )
    if int(feature_dim) != expected_dim:
        failures.append(f"expected feature_dim {expected_dim}, got {feature_dim}")
    if failures:
        validation["status"] = "failed"
        validation["failures"] = failures
        raise RuntimeError("Invalid HRNet-W32 semantic feature tap: " + "; ".join(failures))

    validation["status"] = "ok"
    validation["expected_branch_channels"] = expected_channels
    validation["expected_feature_dim"] = expected_dim
    return validation


def resolve_amp_dtype(requested: str) -> torch.dtype:
    if requested == "auto":
        return torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    return torch.bfloat16 if requested == "bf16" else torch.float16


def encode_frames(
    model: torch.nn.Module,
    preproc: tuple[Any, Any],
    frames: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    feature_tap: str,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[np.ndarray, str, list[list[int]], str]:
    outputs: list[np.ndarray] = []
    resolved_tap = ""
    feature_source = ""
    feature_branch_shapes: list[list[int]] = []
    for start in range(0, int(frames.shape[0]), int(batch_size)):
        chunk = frames[start : start + int(batch_size)]
        tensor = preprocess_rgb_batch(chunk, preproc).to(device, non_blocking=(device.type == "cuda"))
        with torch.no_grad(), autocast("cuda", dtype=amp_dtype, enabled=amp_enabled):
            model._behaviorscope_raw_backbone_features = None
            model_out = model(tensor)
            raw_features = getattr(model, "_behaviorscope_raw_backbone_features", None)
            if raw_features is not None:
                features = raw_features
                feature_source = "forward_hook:PoseModel.backbone.model"
            else:
                features = model_out["backbone"]["features"]
                feature_source = "fallback:PoseModel.outputs.backbone.features"
            pooled, resolved_tap, branch_shapes = select_and_pool_features(
                features,
                feature_tap,
            )
        if not feature_branch_shapes:
            feature_branch_shapes = branch_shapes
        outputs.append(pooled.detach().cpu().float().numpy())
    return np.concatenate(outputs, axis=0), resolved_tap, feature_branch_shapes, feature_source


def write_cache_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def prefilter_missing_samples(out_dir: Path, samples: list[Any], feature_cache_path, overwrite: bool):
    if overwrite:
        return samples, 0
    cached_names = {p.name for p in out_dir.glob("*.npz")}
    missing = [s for s in samples if feature_cache_path(out_dir, s).name not in cached_names]
    return missing, len(samples) - len(missing)


def hardlink_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        return "existing"
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def mirror_existing_cache(args: argparse.Namespace, samples: list[Any], meta: dict[str, Any], feature_cache_path) -> int:
    source = Path(args.source_cache_dir).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Missing --source_cache_dir: {source}")
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = source / "cache_manifest.json"
    source_meta = json.loads(source_manifest.read_text(encoding="utf-8")) if source_manifest.is_file() else {}

    written = 0
    linked = 0
    copied = 0
    existing = 0
    for sample in samples:
        src = feature_cache_path(source, sample)
        if not src.is_file():
            raise FileNotFoundError(f"Source cache missing for {sample.id}: {src}")
        dst = feature_cache_path(out_dir, sample)
        if dst.exists() and args.overwrite:
            dst.unlink()
        action = hardlink_or_copy(src, dst)
        written += action in {"hardlink", "copy"}
        linked += action == "hardlink"
        copied += action == "copy"
        existing += action == "existing"

    feature_dim = int(args.feature_dim or source_meta.get("feature_dim", 0))
    if feature_dim <= 0:
        first = feature_cache_path(out_dir, samples[0])
        with np.load(first, allow_pickle=False) as data:
            feature_dim = int(data["group_feat"].shape[-1])

    manifest = {
        "version": "behaviorscope-y-visual-feature-cache-v1",
        "status": "ok",
        "mode": "existing_cache",
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "source_cache_dir": str(source),
        "source_cache_manifest": str(source_manifest) if source_manifest.is_file() else "",
        "num_frames": int(meta.get("window_size", 32)),
        "n_animals": int(args.n_animals),
        "num_keypoints": int(args.num_keypoints),
        "feature_dim": int(feature_dim),
        "cache_dtype": str(args.cache_dtype),
        "splits": list(args.splits),
        "samples_total": int(len(samples)),
        "samples_written": int(written),
        "samples_hardlinked": int(linked),
        "samples_copied": int(copied),
        "samples_skipped_existing": int(existing),
        "tap_documentation": (
            "Explicit fallback: reused a precomputed cache. This is not a DLC HRNet "
            "feature extraction run."
        ),
    }
    write_cache_manifest(out_dir / "cache_manifest.json", manifest)
    print(f"[cache] mirrored existing cache manifest={out_dir / 'cache_manifest.json'}", flush=True)
    return 0


def compute_dlc_cache(args: argparse.Namespace, samples: list[Any], meta: dict[str, Any], feature_cache_path) -> int:
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_to_encode, skipped_existing = prefilter_missing_samples(
        out_dir, samples, feature_cache_path, bool(args.overwrite)
    )
    if int(args.max_samples) > 0:
        samples_to_encode = samples_to_encode[: int(args.max_samples)]
    if skipped_existing:
        print(
            f"[cache] skipping {skipped_existing:,} existing samples; "
            f"encoding {len(samples_to_encode):,}.",
            flush=True,
        )

    device = torch.device(args.device)
    model, preproc, loader, snapshot_path = build_dlc_feature_model(args)
    amp_enabled = bool(args.amp) and device.type == "cuda"
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    save_dtype = np.float16 if args.cache_dtype == "float16" else np.float32

    num_frames = int(meta.get("window_size", 32))
    manifest_n_animals = int(meta.get("n_animals", args.n_animals))
    if manifest_n_animals != int(args.n_animals):
        raise SystemExit(f"Manifest n_animals={manifest_n_animals}, but --n_animals={args.n_animals}.")
    if "keypoints_per_animal" in meta:
        num_keypoints = int(meta["keypoints_per_animal"])
    else:
        num_keypoints = int(args.num_keypoints)

    t0 = time.time()
    written = 0
    resolved_tap = ""
    feature_source = ""
    feature_branch_shapes: list[list[int]] = []
    tap_validation: dict[str, Any] = {}
    feature_dim = int(args.feature_dim)
    first_group_frames_shape: list[int] = []
    first_animal_frames_shape: list[int] = []
    print(
        f"[cache] DLC features samples={len(samples_to_encode):,} tap={args.feature_tap} "
        f"snapshot={snapshot_path}",
        flush=True,
    )

    for i, sample in enumerate(samples_to_encode, start=1):
        dst = feature_cache_path(out_dir, sample)
        if dst.exists() and not args.overwrite:
            continue
        with np.load(sample.sequence_npz, allow_pickle=False) as data:
            group_frames = data["group_frames"]
            animal_frames = data["animal_frames"]
            if not first_group_frames_shape:
                first_group_frames_shape = list(group_frames.shape)
                first_animal_frames_shape = list(animal_frames.shape)
            if int(group_frames.shape[0]) != num_frames:
                raise ValueError(f"{sample.id}: group_frames T={group_frames.shape[0]}, expected {num_frames}")
            if int(animal_frames.shape[1]) != int(args.n_animals):
                raise ValueError(f"{sample.id}: animal_frames N={animal_frames.shape[1]}, expected {args.n_animals}")
            group_feat, resolved_tap, group_branch_shapes, feature_source = encode_frames(
                model,
                preproc,
                group_frames,
                device=device,
                batch_size=int(args.batch),
                feature_tap=str(args.feature_tap),
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            if not feature_branch_shapes:
                feature_branch_shapes = group_branch_shapes
            t, n = animal_frames.shape[:2]
            animal_flat = animal_frames.reshape(t * n, *animal_frames.shape[2:])
            animal_feat_flat, resolved_tap, _animal_branch_shapes, _animal_feature_source = encode_frames(
                model,
                preproc,
                animal_flat,
                device=device,
                batch_size=int(args.batch),
                feature_tap=str(args.feature_tap),
                amp_enabled=amp_enabled,
                amp_dtype=amp_dtype,
            )
            animal_feat = animal_feat_flat.reshape(t, n, -1)

        if feature_dim <= 0:
            feature_dim = int(group_feat.shape[-1])
            if not tap_validation:
                tap_validation = validate_hrnet_w32_semantic_tap(
                    args=args,
                    loader=loader,
                    model=model,
                    feature_dim=feature_dim,
                    feature_branch_shapes=feature_branch_shapes,
                    feature_source=feature_source,
                )
        if int(group_feat.shape[-1]) != feature_dim or int(animal_feat.shape[-1]) != feature_dim:
            raise ValueError(
                f"{sample.id}: feature dim mismatch group={group_feat.shape[-1]} "
                f"animal={animal_feat.shape[-1]} expected={feature_dim}"
            )
        np.savez(
            dst,
            group_feat=group_feat.astype(save_dtype, copy=False),
            animal_feat=animal_feat.astype(save_dtype, copy=False),
            label=np.asarray(sample.label, dtype=np.int64),
            sample_id=np.asarray(sample.id),
        )
        written += 1
        if i == 1 or i % 100 == 0:
            print(
                f"[cache] encoded={i:,}/{len(samples_to_encode):,} written={written:,} "
                f"feature_dim={feature_dim} elapsed={time.time() - t0:.1f}s",
                flush=True,
            )

    manifest = {
        "version": "behaviorscope-y-visual-feature-cache-v1",
        "status": "ok",
        "mode": "dlc_backbone",
        "manifest_path": str(Path(args.manifest_path).resolve()),
        "dlc_config": str(Path(args.dlc_config).resolve()),
        "dlc_project_root": str(Path(args.dlc_project_root).resolve()),
        "shuffle": int(args.shuffle),
        "trainset_index": int(args.trainset_index),
        "modelprefix": str(args.modelprefix),
        "snapshot_index": str(args.snapshot_index),
        "snapshot_path": str(snapshot_path),
        "model_folder": str(loader.model_folder),
        "pose_task": str(loader.pose_task),
        "feature_tap": str(args.feature_tap),
        "resolved_feature_tap": resolved_tap,
        "feature_source": feature_source,
        "hooked_module": getattr(model, "_behaviorscope_hook_module", ""),
        "tap_validation": tap_validation,
        "feature_tap_source": (
            "Forward hook on PoseModel.backbone.model, the TIMM HRNet features_only "
            "module inside DLC's HRNet backbone. This captures raw HRNet branch "
            "features before DLC HRNet.prepare_output collapses them and before "
            "pose/keypoint heads run. If a model has no nested backbone.model, the "
            "adapter falls back to PoseModel.forward outputs['backbone']['features']."
        ),
        "feature_pooling": (
            "global_average_pool_spatial_dims; semantic_concat concatenates pooled "
            "vectors from every returned HRNet feature branch"
        ),
        "feature_branch_shapes_first_batch": feature_branch_shapes,
        "first_group_frames_shape": first_group_frames_shape,
        "first_animal_frames_shape": first_animal_frames_shape,
        "feature_dim_formula": (
            "sum(channel_dim of each tapped branch after spatial pooling); for "
            "standard HRNet-W32 branches on 224x224 crops this is 32+64+128+256=480"
        ),
        "feature_dim": int(feature_dim),
        "num_frames": int(num_frames),
        "n_animals": int(args.n_animals),
        "num_keypoints": int(num_keypoints),
        "cache_dtype": str(args.cache_dtype),
        "cache_container": "npz_uncompressed",
        "splits": list(args.splits),
        "samples_total": int(len(samples)),
        "samples_requested_for_encoding": int(len(samples_to_encode)),
        "samples_written": int(written),
        "samples_skipped_existing": int(skipped_existing),
        "elapsed_s": float(time.time() - t0),
        "tap_documentation": (
            "The adapter hooks PoseModel.backbone.model to capture the raw TIMM "
            "HRNet multi-branch feature output before DLC HRNet.prepare_output "
            "collapses it and before keypoint heads run. semantic_concat "
            "global-average-pools each branch and concatenates the pooled vectors; "
            "for standard HRNet-W32 on 224x224 crops this is 32+64+128+256=480."
        ),
    }
    write_cache_manifest(out_dir / "cache_manifest.json", manifest)
    print(
        f"[cache] done written={written:,} feature_dim={feature_dim} "
        f"manifest={out_dir / 'cache_manifest.json'}",
        flush=True,
    )
    return 0


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest_path).resolve()
    feature_cache_path, load_n_manifest, scripts_dir = import_behaviorscope_helpers(manifest_path)
    splits, _class_to_idx, _idx_to_class, meta = load_n_manifest(manifest_path)
    samples = selected_manifest_samples(splits, list(args.splits))
    if int(args.max_samples) > 0 and args.mode == "existing_cache":
        samples = samples[: int(args.max_samples)]
    print(f"[cache] BehaviorScope helpers: {scripts_dir}", flush=True)

    if args.mode == "existing_cache":
        return mirror_existing_cache(args, samples, meta, feature_cache_path)
    return compute_dlc_cache(args, samples, meta, feature_cache_path)


if __name__ == "__main__":
    raise SystemExit(main())


