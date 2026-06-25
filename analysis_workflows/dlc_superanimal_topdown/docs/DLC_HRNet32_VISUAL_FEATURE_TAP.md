# DLC HRNet-W32 Visual Feature Tap

This project extracts visual descriptors from the fine-tuned DeepLabCut
`superanimal_topviewmouse` `hrnet_w32` pose model by hooking the HRNet backbone
inside DLC's PyTorch `PoseModel`.

## Reportable Tap Definition

For the DLC/HRNet-W32 run, visual features are tapped from the TIMM HRNet
feature extractor inside DLC's HRNet backbone:

`PoseModel.backbone.model`

This is the `features_only=True` TIMM HRNet module created by:

`deeplabcut.pose_estimation_pytorch.models.backbones.hrnet.HRNet._load_hrnet`

The adapter registers a PyTorch forward hook on that module and captures its raw
multi-branch output before DLC calls:

`HRNet.prepare_output(y_list)`

DLC's default `HRNet.prepare_output()` returns only `y_list[0]` when
`interpolate_branches` is false. The adapter deliberately taps one step earlier
so the visual descriptor can include all four HRNet-W32 branches.

The tap is still before any DLC pose/keypoint head is applied.

## Default Pooling Rule

The default adapter setting is:

`--feature_tap semantic_concat`

For HRNet-W32, the raw TIMM feature output contains four branches. On the
BehaviorScope-X 224x224 RGB crops, the installed DLC/TIMM implementation emits:

| Branch | Tensor Shape Per Crop | Pooled Width |
| --- | ---: | ---: |
| 0 | `[32, 56, 56]` | 32 |
| 1 | `[64, 28, 28]` | 64 |
| 2 | `[128, 14, 14]` | 128 |
| 3 | `[256, 7, 7]` | 256 |

The adapter global-average-pools each branch over its spatial dimensions:

`[C, H, W] -> [C]`

Then it concatenates the branch vectors:

`32 + 64 + 128 + 256 = 480`

So the default DLC HRNet-W32 visual feature vector size is:

`D = 480`

This is produced separately for each group crop and each individual-animal crop:

- `group_feat`: `[T, 480]`
- `animal_feat`: `[T, N, 480]`

With the current MARS/BehaviorScope-X windows:

- `T = 32` frames
- `N = 2` animals
- `group_feat` per sample: `[32, 480]`
- `animal_feat` per sample: `[32, 2, 480]`

In cache metadata this is recorded as:

`resolved_feature_tap: semantic_concat:<N>_branches`

The manifest also records:

- `feature_tap_source`
- `feature_source`
- `hooked_module`
- `tap_validation`
- `feature_pooling`
- `feature_branch_shapes_first_batch`
- `first_group_frames_shape`
- `first_animal_frames_shape`
- `feature_dim_formula`
- `feature_dim`
- `snapshot_path`
- `shuffle`
- `snapshot_index`

For the default `semantic_concat` HRNet-W32 tap, the adapter runs strict
validation unless `--allow_unvalidated_dlc_feature_tap` is explicitly passed. The
strict check requires:

- the `PoseModel.backbone.model` forward hook to fire
- four raw HRNet branches
- branch channel widths `[32, 64, 128, 256]`
- pooled concatenated feature dimension `480`

If any of these checks fail, feature-cache creation stops instead of silently
falling back to a different representation.

## Relationship To YOLO-Pose SPPF Tap

HRNet-W32 does not contain a YOLO-style SPPF module. The closest deliberate
comparison used here is therefore not a named SPPF layer, but the same functional
location in the visual pipeline:

high-level backbone representation, before the task-specific pose decoder/head,
pooled into a fixed-width visual descriptor.

For YOLO-pose this was the SPPF-layer tap. For DLC HRNet-W32 this is the raw
four-branch HRNet feature output inside DLC's backbone, with `semantic_concat`
pooling. It is not the same named architecture layer, because HRNet-W32 has no
SPPF module.

## Reproducible Command Surface

The orchestration script exposes the tap explicitly:

```powershell
.\orchestration\RUN_ME_DLC_TO_ATTENTION256.ps1 `
  -DlcBatch 42 `
  -DlcEvalInterval 1 `
  -DlcSaveEpochs 1 `
  -DlcEarlyStopPatience 5 `
  -DlcWorkers 8 `
  -DlcPinMemory `
  -FeatureCacheMode dlc_backbone `
  -FeatureTap semantic_concat `
  -DlcSnapshotIndex best
```

## Training Protocol Notes

The end-to-end pipeline has two main training phases:

1. DLC SuperAnimal TopViewMouse HRNet-W32 pose fine-tuning.
2. BehaviorScope-X attention-256 behavior-classifier training from the cached DLC
   visual features plus pose/relation inputs.

Within the DLC portion, the script first builds the base and fine-tune datasets.
Those are dataset/model-config creation stages, not separate GPU training phases.
The current DLC command records:

- DLC pose fine-tuning batch size initially tested: `DlcBatch = 32`
- DLC pose fine-tuning batch size used for the faster overnight run:
  `DlcBatch = 42`
- feature extraction batch size: `FeatureBatch = 16`
- BehaviorScope-X classifier batch size: `BehaviorBatch = 64`
- DLC pose fine-tuning images reported by DLC: `41,066` training images and
  `2,162` testing/validation images
- Verbatim DLC terminal line: `Using 41066 images and 2162 for testing`
- DLC pose fine-tuning optimizer steps per epoch with the initial `DlcBatch = 32`:
  `ceil(41066 / 32) = 1,284` iterations per epoch
- DLC pose fine-tuning optimizer steps per epoch with the faster overnight
  `DlcBatch = 42`: `ceil(41066 / 42) = 978` iterations per epoch
- Observed early-run DLC pose fine-tuning throughput on the RTX 4080 system:
  approximately `1.8` iterations/second. At that rate, one epoch is roughly
  `1,284 / 1.8 = 713` seconds, or about `12` minutes. The full `50` epoch run is
  therefore roughly `10` hours before downstream feature extraction/classifier
  training, subject to validation/checkpoint overhead and normal throughput drift.
- DLC pose fine-tuning early stopping: enabled in this orchestration with
  `DlcEarlyStopPatience = 5`.
- DLC dataloader workers initially tested: `DlcWorkers = 4`.
- DLC dataloader workers used for the faster overnight run: `DlcWorkers = 8`.
- DLC dataloader pinned memory: enabled with `DlcPinMemory`.
- DLC validation cadence: `DlcEvalInterval = 1`, so the validation/test split is
  evaluated after every fine-tuning epoch.
- DLC snapshot save interval passed by orchestration: `DlcSaveEpochs = 1`.
- DLC pose checkpoint-selection metric: `metrics/test.mAP`, which DLC computes as
  OKS mean average precision across thresholds `0.50, 0.55, ..., 0.95`. This is the
  pose-estimation analogue of mAP50-95.
- DLC retains the best pose snapshot as `snapshot-best-<epoch>.pt` when
  `metrics/test.mAP` improves. The orchestration stops when this metric has not
  improved for `5` validation epochs.
- The downstream DLC feature-cache stage defaults to `DlcSnapshotIndex = best`, so
  visual features are extracted from the validation-selected best pose checkpoint,
  not merely the last completed checkpoint.
- BehaviorScope-X classifier early stopping patience: `15` epochs, but the
  classifier command currently requests `10` total epochs, so patience is unlikely
  to trigger in this run.

Detector training was not active during the pose fine-tuning run because
`detector_epochs` remained `0`. Detector fine-tuning is handled as a separate
stage so pose checkpoint selection is not disturbed.

DLC may print an advisory warning mentioning "batch size 1 and/or
freeze_bn_stats". In this run, the active pose model training batch size is
`42` for the faster overnight run. The warning is expected because HRNet-W32 uses
frozen batch-normalization statistics (`freeze_bn_stats: true`) in the DLC model
configuration. The detector configuration can still show `batch_size: 1`, but
detector training is disabled with `detector_epochs: 0`.

During the faster overnight DLC fine-tuning run, the user updated the run to
`DlcBatch = 42` with `DlcWorkers = 8` and pinned memory enabled. Reported steady
state utilization was approximately `96-100%` GPU utilization and `31-35%` CPU
utilization, with visibly faster iteration throughput than the earlier low-worker
run.

## Detector Fine-Tuning Notes

The pose fine-tune completed with early stopping at epoch `32`; the best pose
checkpoint by validation mAP was epoch `27`:

- best pose metric: `metrics/test.mAP = 91.76826675673418`
- best pose checkpoint: `snapshot-best-027.pt`
- final saved pose snapshots retained: `snapshot-best-027.pt`,
  `snapshot-028.pt`, `snapshot-029.pt`, `snapshot-030.pt`, `snapshot-031.pt`,
  `snapshot-032.pt`

Detector fine-tuning is a separate DLC top-down requirement. It should create
`snapshot-detector*.pt` files and must not change the retained pose checkpoints.
The orchestration script includes a detector-stage guard that records the pose
snapshot set before detector training and refuses to continue if detector
training changes the pose snapshot inventory or selected best pose checkpoint.

The detector-only stage can be run with:

```powershell
.\orchestration\RUN_ME_DLC_TO_ATTENTION256.ps1 `
  -Device cuda:0 `
  -DlcDetectorEpochs 10 `
  -DlcDetectorBatch 2 `
  -DlcDetectorEvalInterval 1 `
  -DlcDetectorSaveEpochs 1 `
  -DlcWorkers 8 `
  -DlcPinMemory `
  -DlcStages detector_train `
  -SkipFeatureCache `
  -SkipClassifierTraining
```

The detector run was manually stopped after a satisfactory validated checkpoint
was saved. The manual marker verified:

- best detector checkpoint: `snapshot-detector-best-004.pt`
- best detector metric: `metrics/test.mAP@50:95 = 71.02195519430495`
- best detector epoch: `4`
- preserved best pose checkpoint: `snapshot-best-027.pt`
- preserved best pose metric: `metrics/test.mAP = 91.76826675673418`

Best detector validation metrics, epoch `4/10`:

- train loss: `0.12946`
- `metrics/test.mAP@50:95 = 71.02`
- `metrics/test.mAP@50 = 95.95`
- `metrics/test.mAP@75 = 86.66`
- `metrics/test.mAR@50:95 = 75.73`
- `metrics/test.mAR@50 = 96.11`
- `metrics/test.mAR@75 = 89.18`

Detector validation metrics by epoch:

| Detector Epoch | Train Loss | mAP@50:95 | mAP@50 | mAP@75 | mAR@50:95 | mAR@50 | mAR@75 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.18840 | 64.40 | 95.90 | 78.70 | 69.51 | 96.16 | 83.35 |
| 2 | 0.15905 | 67.31 | 96.96 | 83.22 | 72.41 | 97.09 | 86.63 |
| 3 | 0.14070 | 70.61 | 95.95 | 85.97 | 75.36 | 96.53 | 88.99 |
| 4 | 0.12946 | 71.02 | 95.95 | 86.66 | 75.73 | 96.11 | 89.18 |

For the current BehaviorScope-X classifier pipeline, the DLC feature-cache stage
extracts visual features from the fine-tuned HRNet backbone on existing
BehaviorScope sequence crops. The classifier still reads pose/relation inputs
from the BehaviorScope NPZ samples. A trained detector is required for a full DLC
top-down raw-image/video analysis workflow; using DLC detector-derived poses as
the classifier pose inputs would require an additional explicit conversion stage.

Alternative tap settings are available for sensitivity checks:

- `semantic_concat`: pool and concatenate all returned HRNet feature branches.
- `last`: pool only the last returned branch.
- integer branch index, for example `0` or `-1`: pool one selected branch.

The default for the planned comparison is `semantic_concat`.

## Native DLC Top-Down NPZ Cache

A full-video native DLC top-down cache builder was added for the scientifically
separate condition where both the detector/crops and pose keypoints come from
the fine-tuned DLC models.

- Script: `orchestration/dlc_topdown_full_video_npz.py`
- Entry point: `orchestration/RUN_ME_DLC_TOPDOWN_NPZ.ps1`
- Source videos: `path/to/MARS_data`
- Included source splits: `train` and `validation`
- Excluded leakage-guard splits: `test_1` and `test_2`
- Source media used: per-video `_Top.mp4` files, not `.seq` conversion
- Pose checkpoint: `snapshot-best-027.pt`
- Detector checkpoint: `snapshot-detector-best-004.pt`
- DLC output path:
  `BehaviorScope_X_minimal/outputs/npz_cache/mars_full_video_dlc_topdown`
- NPZ schema: same `behaviorscope-n-v1` window schema consumed by the existing
  BehaviorScope-X loader
- Compression default: deflate level 1. This is preferred for the full RGB-heavy
  cache because the existing YOLO-derived full-video cache is about 70 GiB
  compressed, so uncompressed full NPZs would risk exhausting the available
  C-drive space.
- Resume behavior: completed videos write markers under
  `build_state/completed_videos`; rerunning the same command skips matching
  completed videos and rebuilds only incomplete/stale videos.
- Smoke validation completed on 2026-06-07 with one train source and one
  validation source, two windows per source, and manifest validation passed.

Default full-cache command:

```powershell
.\orchestration\RUN_ME_DLC_TOPDOWN_NPZ.ps1 `
  -MarsRoot path\to\MARS_data `
  -Device cuda:0 `
  -PoseBatch 16 `
  -DetectorBatch 2 `
  -NpzWriters 3 `
  -NpzCompression deflated `
  -NpzCompressLevel 1 `
  -ValidationSampleLimit 64
```


