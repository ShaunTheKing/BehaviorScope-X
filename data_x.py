"""
BehaviorScope-X data loader.

Standalone module so the existing single-animal `data.py` is untouched.

Consumes NPZs produced by `cropping_n.py` /
`scripts/mars_to_behaviorscope_npz.py` with `--emit-mode n_animal`. The
expected schema (from §7.1 of the proposal):

    group_frames        uint8   [T, H, W, 3]
    animal_frames       uint8   [T, N, H, W, 3]
    animal_keypoints    float32 [T, N, K, 3]    crop-relative, normalized
    animal_keypoints_raw  float32 [T, N, K, 2]  original-frame px (debug)
    animal_keypoints_conf float32 [T, N, K]
    animal_mask         bool    [T, N]          HARD MASK (animal present)
    pose_mask           bool    [T, N]          SOFT GATE (pose valid)
    pose_conf           float32 [T, N]          mean keypoint conf per animal
    crop_conf           float32 [T, N]          per-animal crop validity
    track_conf          float32 [T, N]          track confidence
    track_age           float32 [T, N]          age, normalized to [0,1]
    relation_features   float32 [T, N, N, R]    per-pair features
    relation_pose_mask  bool    [T, N, N]
    relation_pose_conf  float32 [T, N, N]
    relation_present    bool    [T, N, N]
    bbox_xyxy_animals   int32   [T, N, 4]       (debug)
    bbox_xyxy_group     int32   [T, 4]          (debug)
    crop_xyxy_animals   int32   [T, N, 4]       (debug)
    crop_xyxy_group     int32   [T, 4]          (debug)
    track_ids           int32   [T, N]
    body_length_px      float32 scalar
    n_animals           int32   scalar

The dataset returns a dict-of-tensors per sample, ready for batching by
`collate_multi_animal()`.
"""
from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from config import IMAGENET_MEAN, IMAGENET_STD
    from utils.pose_features_x import extract_per_animal_pose_features
except Exception:  # pragma: no cover
    from .config import IMAGENET_MEAN, IMAGENET_STD
    from .utils.pose_features_x import extract_per_animal_pose_features


# ----------------------------------------------------------------------------
# Manifest entry (mirrors the shape used in the existing data.py)
# ----------------------------------------------------------------------------

@dataclass
class MultiAnimalSequenceSample:
    id: str
    label: int
    class_name: str
    sequence_npz: Path
    start_frame: int
    end_frame: int
    fps: float
    source_video: str


REQUIRED_SCHEMA_VERSION = "behaviorscope-n-v1"


def _safe_cache_stem(text: str, max_len: int = 96) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("._")
    return (stem[:max_len] or "sample")


def feature_cache_path(cache_dir: Path | str, sample: MultiAnimalSequenceSample) -> Path:
    """Stable cache filename for a manifest sample."""
    cache_dir = Path(cache_dir)
    key = f"{sample.id}|{sample.sequence_npz}|{sample.start_frame}|{sample.end_frame}"
    digest = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()[:12]
    return cache_dir / f"{_safe_cache_stem(sample.id)}_{digest}.npz"


def feature_cache_candidate_paths(cache_dir: Path | str, sample: MultiAnimalSequenceSample) -> List[Path]:
    """Return current and legacy cache filenames for a manifest sample.

    Older manuscript-era caches used only sample id and frame bounds in the
    hash. Current caches include the resolved NPZ path to avoid collisions.
    Keep the legacy lookup so copied/released caches remain readable.
    """
    cache_dir = Path(cache_dir)
    current = feature_cache_path(cache_dir, sample)
    legacy_key = f"{sample.id}|{sample.start_frame}|{sample.end_frame}"
    legacy_digest = hashlib.sha1(legacy_key.encode("utf-8", errors="ignore")).hexdigest()[:12]
    legacy = cache_dir / f"{_safe_cache_stem(sample.id)}_{legacy_digest}.npz"
    return [current] if legacy == current else [current, legacy]


def _shape_str(arr: np.ndarray) -> str:
    return "x".join(str(v) for v in arr.shape)


def validate_n_npz_payload(
    data,
    *,
    sample_id: str,
    num_frames: int,
    n_animals: int,
    num_keypoints: int,
    rel_feature_dim: int,
    require_schema_version: bool = False,
    require_visual: bool = True,
) -> None:
    """Fail-fast validator for BehaviorScope-X NPZ windows.

    This checks the tensor schema that training and inference assume. It is
    intentionally strict by default because silent padding/trimming can change
    the analytical unit without making the training loop fail.
    """
    required = {
        "group_frames": (num_frames, None, None, 3),
        "animal_frames": (num_frames, n_animals, None, None, 3),
        "animal_keypoints": (num_frames, n_animals, num_keypoints, 3),
        "animal_keypoints_raw": (num_frames, n_animals, num_keypoints, 2),
        "animal_keypoints_conf": (num_frames, n_animals, num_keypoints),
        "animal_mask": (num_frames, n_animals),
        "pose_mask": (num_frames, n_animals),
        "pose_conf": (num_frames, n_animals),
        "crop_conf": (num_frames, n_animals),
        "track_conf": (num_frames, n_animals),
        "track_age": (num_frames, n_animals),
        "relation_features": (num_frames, n_animals, n_animals, rel_feature_dim),
        "relation_pose_mask": (num_frames, n_animals, n_animals),
        "relation_pose_conf": (num_frames, n_animals, n_animals),
        "relation_present": (num_frames, n_animals, n_animals),
        "track_ids": (num_frames, n_animals),
    }
    if not require_visual:
        required.pop("group_frames", None)
        required.pop("animal_frames", None)
    missing = [name for name in required if name not in data.files]
    if missing:
        raise ValueError(f"{sample_id}: missing required NPZ keys: {missing}")

    if require_schema_version:
        if "schema_version" not in data.files:
            raise ValueError(f"{sample_id}: missing schema_version")
        schema = str(np.asarray(data["schema_version"]).item())
        if schema != REQUIRED_SCHEMA_VERSION:
            raise ValueError(
                f"{sample_id}: schema_version={schema!r}, expected {REQUIRED_SCHEMA_VERSION!r}"
            )

    if "n_animals" in data.files:
        npz_n_animals = int(np.asarray(data["n_animals"]).item())
        if npz_n_animals != int(n_animals):
            raise ValueError(
                f"{sample_id}: n_animals={npz_n_animals}, expected {n_animals}"
            )

    for name, expected in required.items():
        arr = data[name]
        if arr.ndim != len(expected):
            raise ValueError(
                f"{sample_id}: {name} ndim={arr.ndim}, expected {len(expected)} "
                f"(shape={_shape_str(arr)})"
            )
        for axis, exp in enumerate(expected):
            if exp is not None and int(arr.shape[axis]) != int(exp):
                raise ValueError(
                    f"{sample_id}: {name} shape={_shape_str(arr)}, "
                    f"expected axis {axis} == {exp}"
                )

    if require_visual:
        if data["group_frames"].dtype != np.uint8 or data["animal_frames"].dtype != np.uint8:
            raise ValueError(f"{sample_id}: RGB frame arrays must be uint8")

    animal_mask = data["animal_mask"].astype(bool)
    pose_mask = data["pose_mask"].astype(bool)
    if np.any(pose_mask & ~animal_mask):
        raise ValueError(f"{sample_id}: pose_mask contains True where animal_mask is False")

    rel_present = data["relation_present"].astype(bool)
    expected_rel_present = animal_mask[:, :, None] & animal_mask[:, None, :]
    diag = np.arange(n_animals)
    expected_rel_present[:, diag, diag] = False
    if not np.array_equal(rel_present, expected_rel_present):
        raise ValueError(f"{sample_id}: relation_present is inconsistent with animal_mask")


def load_n_manifest(manifest_path: Path | str):
    """Load a sequence_manifest.json and return splits + class maps + meta.

    Returns: (splits: dict[str, list[MultiAnimalSequenceSample]],
              class_to_idx: dict[str, int],
              idx_to_class: dict[int, str],
              meta: dict)
    """
    manifest_path = Path(manifest_path)
    base = manifest_path.parent
    with manifest_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    class_to_idx = {str(k): int(v) for k, v in payload["class_to_idx"].items()}
    idx_to_class = {int(k): v for k, v in payload["idx_to_class"].items()}
    meta = payload.get("meta", {})

    splits: Dict[str, List[MultiAnimalSequenceSample]] = {}
    for split_name, samples in payload["splits"].items():
        out: List[MultiAnimalSequenceSample] = []
        for s in samples:
            npz_rel = s.get("sequence_npz")
            if not npz_rel:
                continue
            npz_path = (base / Path(npz_rel)).resolve()
            out.append(MultiAnimalSequenceSample(
                id=s["id"],
                label=int(s["label"]),
                class_name=s["class_name"],
                sequence_npz=npz_path,
                start_frame=int(s.get("start_frame", 0)),
                end_frame=int(s.get("end_frame", 0)),
                fps=float(s.get("fps", 30.0)),
                source_video=s.get("source_video", ""),
            ))
        splits[split_name] = out
    return splits, class_to_idx, idx_to_class, meta


# ----------------------------------------------------------------------------
# Tensor helpers
# ----------------------------------------------------------------------------

def _normalize_rgb(frames_uint8: np.ndarray) -> torch.Tensor:
    """[T, H, W, 3] uint8 -> [T, 3, H, W] float32 in [0, 1].

    BehaviorScope-X uses Ultralytics YOLO normalization (x/255 only — no
    mean/std subtraction) so the YOLO trunk sees inputs in the same
    distribution it was trained on. Applying ImageNet mean/std on top would
    break the domain-pretrained feature hypothesis.

    `frames_uint8` may be HWC RGB or BGR; the upstream pipeline writes RGB,
    so we trust that.
    """
    arr = frames_uint8.astype(np.float32) / 255.0
    arr = arr.transpose(0, 3, 1, 2)  # [T, 3, H, W]
    return torch.from_numpy(arr.copy())


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class MultiAnimalSequenceDataset(Dataset):
    """Loads BehaviorScope-X NPZs and returns multi-stream tensors with masks.

    No augmentation in v1 (resize+normalize only). Per the proposal,
    augmentation should apply identical params across all streams within a
    clip; the basic loader can defer that to a later iteration.
    """

    def __init__(
        self,
        samples: Sequence[MultiAnimalSequenceSample],
        num_frames: int,
        n_animals: int,
        num_keypoints: int,
        rel_feature_dim: int = 11,
        pose_dropout_p_uniform: float = 0.0,
        strict_schema: bool = True,
        skip_invalid: bool = False,
        feature_cache_dir: Path | str | None = None,
        require_feature_cache: bool = False,
        skip_visual: bool = False,
        rng_seed: int = 0,
    ):
        self.samples = list(samples)
        self.num_frames = int(num_frames)
        self.n_animals = int(n_animals)
        self.num_keypoints = int(num_keypoints)
        self.rel_feature_dim = int(rel_feature_dim)
        # R6b: random per-animal per-frame pose dropout, rate sampled per window
        self.pose_dropout_p_uniform = float(pose_dropout_p_uniform)
        self.strict_schema = bool(strict_schema)
        self.skip_invalid = bool(skip_invalid)
        self.feature_cache_dir = Path(feature_cache_dir) if feature_cache_dir else None
        self.require_feature_cache = bool(require_feature_cache)
        self.rng_seed = int(rng_seed)
        # When true, never read group_frames / animal_frames from the NPZ and
        # never emit "group" / "animal" tensors. Used by visual-disabled
        # ablations (R4 pose-only, R4b relations-only) where the model's
        # visual encoder is None, so loading raw RGB is pure wasted disk I/O,
        # CPU normalization, and CPU->GPU transfer.
        self.skip_visual = bool(skip_visual)

    def __len__(self) -> int:
        return len(self.samples)

    def _rng_for_sample(self, idx: int, sample: MultiAnimalSequenceSample) -> np.random.Generator:
        """Deterministic NumPy RNG for per-sample augmentation.

        DataLoader workers can schedule samples differently across processes.
        Seeding from run seed + stable sample identity makes pose-dropout
        augmentation reproducible regardless of worker count.
        """
        key = f"{self.rng_seed}|{idx}|{sample.id}|{sample.start_frame}|{sample.end_frame}"
        seed = int(hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest(), 16)
        return np.random.default_rng(seed)

    def _apply_pose_dropout(
        self,
        pose_mask: np.ndarray,
        pose_conf: np.ndarray,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """R6b training: drop pose per-frame, per-animal at a per-window rate."""
        if self.pose_dropout_p_uniform <= 0.0:
            return pose_mask, pose_conf
        p = float(rng.uniform(0.0, self.pose_dropout_p_uniform))
        if p <= 1e-9:
            return pose_mask, pose_conf
        drop = rng.random(pose_mask.shape) < p
        new_mask = pose_mask & ~drop
        # When pose is dropped, also zero its confidence so reliability features
        # carry the right signal.
        new_conf = pose_conf.copy()
        new_conf[drop] = 0.0
        return new_mask, new_conf

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        try:
            data = np.load(sample.sequence_npz, allow_pickle=False)
            use_cache = self.feature_cache_dir is not None
            # When skip_visual is on, the validator's shape check on
            # group_frames / animal_frames forces a full RGB load just to
            # inspect dtype — defeating the whole point of skipping visual.
            if self.strict_schema:
                validate_n_npz_payload(
                    data,
                    sample_id=sample.id,
                    num_frames=self.num_frames,
                    n_animals=self.n_animals,
                    num_keypoints=self.num_keypoints,
                    rel_feature_dim=self.rel_feature_dim,
                    require_schema_version=True,
                    require_visual=not (use_cache or self.skip_visual),
                )

            group_feat = None
            animal_feat = None
            if use_cache:
                cache_candidates = feature_cache_candidate_paths(self.feature_cache_dir, sample)
                cache_path = next((p for p in cache_candidates if p.is_file()), cache_candidates[0])
                if not cache_path.is_file():
                    if self.require_feature_cache:
                        raise FileNotFoundError(
                            f"feature cache missing for {sample.id}: "
                            f"{'; '.join(str(p) for p in cache_candidates)}. "
                            "Run precompute_visual_features_x.py first, or omit --use_feature_cache."
                        )
                else:
                    with np.load(cache_path, allow_pickle=False) as cache:
                        group_feat = cache["group_feat"].astype(np.float32)
                        animal_feat = cache["animal_feat"].astype(np.float32)
                    T_clip = int(group_feat.shape[0])
            if self.skip_visual and group_feat is None and animal_feat is None:
                # Visual streams disabled at the model level — never load the
                # raw RGB tensors from disk. Derive T_clip from any per-frame
                # array that's guaranteed to be present (animal_keypoints lives
                # in every NPZ).
                group_rgb = None
                animal_rgb = None
                if "animal_keypoints" in data.files:
                    T_clip = int(data["animal_keypoints"].shape[0])
                elif "animal_mask" in data.files:
                    T_clip = int(data["animal_mask"].shape[0])
                else:
                    T_clip = self.num_frames
            elif group_feat is None or animal_feat is None:
                # ---- frames ----
                group_rgb = data["group_frames"]       # [T, H, W, 3] uint8
                animal_rgb = data["animal_frames"]     # [T, N, H, W, 3] uint8
                T_clip = int(group_rgb.shape[0])
            else:
                group_rgb = None
                animal_rgb = None

            if T_clip != self.num_frames:
                raise ValueError(
                    f"window size mismatch for {sample.id}: {T_clip} vs {self.num_frames}"
                )

            if animal_feat is not None:
                N_clip = int(animal_feat.shape[1])
            elif animal_rgb is not None:
                N_clip = int(animal_rgb.shape[1])
            else:
                N_clip = self.n_animals
            # If the dataset's N differs from the model's expected N, pad or trim
            N = self.n_animals
            if animal_rgb is not None:
                if N_clip < N:
                    # pad missing animals with zeros + animal_mask=False
                    pad_animal = np.zeros((T_clip, N - N_clip, *animal_rgb.shape[2:]), dtype=animal_rgb.dtype)
                    animal_rgb = np.concatenate([animal_rgb, pad_animal], axis=1)
                elif N_clip > N:
                    animal_rgb = animal_rgb[:, :N]
            if animal_feat is not None:
                if animal_feat.ndim != 3:
                    raise ValueError(f"{sample.id}: animal_feat ndim={animal_feat.ndim}, expected 3")
                if group_feat.ndim != 2:
                    raise ValueError(f"{sample.id}: group_feat ndim={group_feat.ndim}, expected 2")
                if animal_feat.shape[0] != T_clip:
                    raise ValueError(f"{sample.id}: animal_feat T={animal_feat.shape[0]}, expected {T_clip}")
                if N_clip < N:
                    pad = np.zeros((T_clip, N - N_clip, animal_feat.shape[-1]), dtype=animal_feat.dtype)
                    animal_feat = np.concatenate([animal_feat, pad], axis=1)
                elif N_clip > N:
                    animal_feat = animal_feat[:, :N]

            # ---- per-animal arrays ----
            def _take(name: str, fallback_shape, dtype):
                arr = data[name] if name in data.files else np.zeros(fallback_shape, dtype=dtype)
                # Pad/trim N axis where applicable
                if arr.ndim >= 2 and arr.shape[1] != N:
                    if arr.shape[1] < N:
                        pad_shape = list(arr.shape); pad_shape[1] = N - arr.shape[1]
                        arr = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=1)
                    elif arr.shape[1] > N:
                        arr = arr[:, :N]
                return arr

            animal_keypoints = _take("animal_keypoints", (T_clip, N, self.num_keypoints, 3), np.float32)
            animal_kp_raw = _take("animal_keypoints_raw", (T_clip, N, self.num_keypoints, 2), np.float32)
            animal_kp_conf = _take("animal_keypoints_conf", (T_clip, N, self.num_keypoints), np.float32)

            animal_mask = _take("animal_mask", (T_clip, N), np.bool_).astype(bool)
            pose_mask = _take("pose_mask", (T_clip, N), np.bool_).astype(bool)
            pose_conf = _take("pose_conf", (T_clip, N), np.float32).astype(np.float32)
            crop_conf = _take("crop_conf", (T_clip, N), np.float32).astype(np.float32)
            track_conf = _take("track_conf", (T_clip, N), np.float32).astype(np.float32)
            track_age = _take("track_age", (T_clip, N), np.float32).astype(np.float32)

            # ---- relation arrays ----
            rel = data["relation_features"] if "relation_features" in data.files else np.zeros((T_clip, N, N, self.rel_feature_dim), dtype=np.float32)
            rel_pose_mask = data["relation_pose_mask"] if "relation_pose_mask" in data.files else np.zeros((T_clip, N, N), dtype=bool)
            rel_pose_conf = data["relation_pose_conf"] if "relation_pose_conf" in data.files else np.zeros((T_clip, N, N), dtype=np.float32)
            rel_present = data["relation_present"] if "relation_present" in data.files else np.zeros((T_clip, N, N), dtype=bool)

            # Trim N x N axes to model's N
            if rel.shape[1] != N or rel.shape[2] != N:
                trimmed = np.zeros((T_clip, N, N, rel.shape[-1]), dtype=rel.dtype)
                copy_n = min(N, rel.shape[1])
                trimmed[:, :copy_n, :copy_n, :] = rel[:, :copy_n, :copy_n, :]
                rel = trimmed
            for arr_name, arr_in in (("rel_pose_mask", rel_pose_mask), ("rel_pose_conf", rel_pose_conf), ("rel_present", rel_present)):
                if arr_in.shape[1] != N or arr_in.shape[2] != N:
                    trimmed = np.zeros((T_clip, N, N), dtype=arr_in.dtype)
                    copy_n = min(N, arr_in.shape[1])
                    trimmed[:, :copy_n, :copy_n] = arr_in[:, :copy_n, :copy_n]
                    if arr_name == "rel_pose_mask":
                        rel_pose_mask = trimmed.astype(bool)
                    elif arr_name == "rel_pose_conf":
                        rel_pose_conf = trimmed.astype(np.float32)
                    elif arr_name == "rel_present":
                        rel_present = trimmed.astype(bool)

            # ---- pose-self features (compute on the fly per-animal) ----
            # animal_keypoints is [T, N, K, 3] in crop-normalized coords (x,y,conf)
            # extract_per_animal_pose_features returns [T, N, D]
            try:
                pose_self_feats = extract_per_animal_pose_features(animal_keypoints)
            except Exception:
                K = self.num_keypoints
                D = K * 4 + K * (K - 1) // 2
                pose_self_feats = np.zeros((T_clip, N, D), dtype=np.float32)

            # R6b training-time pose dropout (no-op if rate=0)
            rng = self._rng_for_sample(idx, sample)
            pose_mask, pose_conf = self._apply_pose_dropout(pose_mask, pose_conf, rng)

            # ---- to tensors ----
            out = {
                "animal_mask": torch.from_numpy(animal_mask),        # [T, N]
                "pose_mask": torch.from_numpy(pose_mask),            # [T, N]
                "pose_conf": torch.from_numpy(pose_conf),            # [T, N]
                "crop_conf": torch.from_numpy(crop_conf),            # [T, N]
                "track_conf": torch.from_numpy(track_conf),          # [T, N]
                "track_age": torch.from_numpy(track_age),            # [T, N]
                "pose_self": torch.from_numpy(pose_self_feats),      # [T, N, D_pose]
                "relation_features": torch.from_numpy(rel.astype(np.float32)),       # [T, N, N, R]
                "relation_pose_mask": torch.from_numpy(rel_pose_mask),               # [T, N, N]
                "relation_pose_conf": torch.from_numpy(rel_pose_conf.astype(np.float32)),  # [T, N, N]
                "relation_present": torch.from_numpy(rel_present),                   # [T, N, N]
            }
            if group_feat is not None and animal_feat is not None:
                out["group_feat"] = torch.from_numpy(group_feat.astype(np.float32))    # [T, D]
                out["animal_feat"] = torch.from_numpy(animal_feat.astype(np.float32))  # [T, N, D]
            elif self.skip_visual:
                # Visual streams disabled — model.forward() never reads
                # batch["group"] or batch["animal"], so we emit nothing.
                pass
            else:
                # YOLO-style normalization (x/255 only, no mean/std subtraction)
                # so the YOLO trunk sees inputs matching its training distribution.
                group_t = _normalize_rgb(group_rgb)                     # [T, 3, H, W]
                # animal_rgb: [T, N, H, W, 3] -> [N, T, 3, H, W]
                ar = animal_rgb.transpose(1, 0, 4, 2, 3).astype(np.float32) / 255.0
                animal_t = torch.from_numpy(ar)                          # [N, T, 3, H, W]
                out["group"] = group_t
                out["animal"] = animal_t
            label = torch.tensor(int(sample.label), dtype=torch.long)
            return out, label
        except Exception as exc:
            if self.skip_invalid:
                print(f"[data_n] skipping {sample.id}: {exc}")
                return None
            raise


def collate_multi_animal(batch):
    """Stack a batch of dict samples. Drops failed samples (None)."""
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    inputs = [b[0] for b in batch]
    labels = torch.stack([b[1] for b in batch], dim=0)
    out: Dict[str, torch.Tensor] = {}
    for key in inputs[0].keys():
        out[key] = torch.stack([s[key] for s in inputs], dim=0)
    return out, labels


# ----------------------------------------------------------------------------
# Class-weight helpers (sqrt_inverse with optional clamp)
# ----------------------------------------------------------------------------

def compute_class_weights(
    samples: Sequence[MultiAnimalSequenceSample],
    num_classes: int,
    strategy: str = "sqrt_inverse",
    clamp_max: float | None = 2.0,
) -> torch.Tensor:
    """Compute per-class loss weights from the training samples."""
    counts = np.zeros(num_classes, dtype=np.float64)
    for s in samples:
        if 0 <= int(s.label) < num_classes:
            counts[int(s.label)] += 1.0
    counts = np.maximum(counts, 1.0)
    if strategy == "inverse":
        w = 1.0 / counts
    elif strategy == "sqrt_inverse":
        w = 1.0 / np.sqrt(counts)
    elif strategy in ("none", "uniform"):
        w = np.ones_like(counts)
    else:
        raise ValueError(f"Unknown class_weighting strategy: {strategy}")
    # Normalize so the minimum weight is 1.0
    w = w / w.min()
    if clamp_max is not None and clamp_max > 0:
        w = np.minimum(w, float(clamp_max))
    return torch.from_numpy(w.astype(np.float32))


def make_weighted_sampler(
    samples: Sequence[MultiAnimalSequenceSample],
    class_weights: torch.Tensor,
    generator: torch.Generator | None = None,
):
    """Return a WeightedRandomSampler that draws each sample with weight =
    class_weights[label]."""
    from torch.utils.data import WeightedRandomSampler

    sample_weights = np.array(
        [float(class_weights[int(s.label)]) for s in samples],
        dtype=np.float64,
    )
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
        generator=generator,
    )



