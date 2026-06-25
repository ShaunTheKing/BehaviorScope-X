# DLC Top-Down BehaviorScope-X Next Step

Date prepared: 2026-06-07

The DLC full-video top-down NPZ cache has been built. The next step is not to
train directly from the NPZ cache with the default YOLO visual backbone. Instead,
we first build a DLC-HRNet visual feature cache from the DLC top-down crops, then
train the Attention-256 behavior classifier from those precomputed DLC features.

## Inputs

- DLC top-down manifest:
  `BehaviorScope_X_minimal/outputs/npz_cache/mars_full_video_dlc_topdown/sequence_manifest.json`
- Train windows: 20,414
- Validation windows: 10,079
- Pose snapshot:
  `dlc-models-pytorch/iteration-0/MARS_DLC_SuperAnimal2026-06-06-trainset95shuffle2/train/snapshot-best-027.pt`
- Detector snapshot:
  `dlc-models-pytorch/iteration-0/MARS_DLC_SuperAnimal2026-06-06-trainset95shuffle2/train/snapshot-detector-best-004.pt`

## Feature Tap

Use `-FeatureTap semantic_concat`.

This hooks the raw TIMM HRNet multi-branch output inside DLC's HRNet backbone,
before DLC collapses the branches and before keypoint heads run. Each branch is
global-average-pooled and concatenated.

For HRNet-W32, the feature vector is:

```text
32 + 64 + 128 + 256 = 480 visual features per view per frame
```

Each BehaviorScope-X window has 32 frames and 3 visual streams:

```text
1 group crop + 2 animal crops = 3 views
32 frames x 3 views x 480 features x 2 bytes float16 = 92,160 bytes
```

For 30,493 windows, the raw feature arrays are about 2.62 GiB before NPZ
container overhead. A practical expectation is roughly 3 GiB, not tens of GiB.

## Entry Point

Dry-run preflight:

```powershell
.\orchestration\RUN_ME_DLC_TOPDOWN_TO_ATTENTION256.ps1 `
  -Device cuda:0 `
  -FeatureBatch 16 `
  -BehaviorBatch 64 `
  -BehaviorWorkers 8 `
  -FeatureTap semantic_concat `
  -DlcSnapshotIndex best `
  -DryRun
```

Full run:

```powershell
.\orchestration\RUN_ME_DLC_TOPDOWN_TO_ATTENTION256.ps1 `
  -Device cuda:0 `
  -FeatureBatch 16 `
  -BehaviorBatch 64 `
  -BehaviorWorkers 8 `
  -FeatureTap semantic_concat `
  -DlcSnapshotIndex best
```

## Outputs

Feature cache:

```text
behaviorscope_outputs/dlc_topdown_hrnet_feature_cache_trainval
```

Behavior classifier run:

```text
behaviorscope_outputs/dlc_topdown_attention256_training/mars_dlc_topdown_hrnet_attention256_seed42
```

## Resume Behavior

Rerunning the command is safe:

- The feature-cache step skips existing feature NPZs unless
  `-OverwriteFeatureCache` is supplied.
- The classifier step passes `--resume` to `train_x.py`.
- To train only after the feature cache is complete, use:

```powershell
.\orchestration\RUN_ME_DLC_TOPDOWN_TO_ATTENTION256.ps1 `
  -Device cuda:0 `
  -SkipFeatureCache `
  -BehaviorBatch 64 `
  -BehaviorWorkers 8
```

## Disk Note

The local DLC top-down NPZ cache should remain in place until the behavior run
finishes or the E: backup has been verified. The already verified YOLO full-video
NPZ cache is the safer deletion candidate if more C: drive space is needed.


