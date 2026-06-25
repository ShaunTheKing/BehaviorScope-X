# Reproducibility Entry Points

These scripts provide stable reproducibility-facing entry points for reproducing the
DLC SuperAnimal TopViewMouse HRNet-W32 fine-tuning and downstream
BehaviorScope-Y feature-cache/classifier workflow.

Run them from an activated `deeplabcut` conda environment at the project root:

```powershell
conda activate deeplabcut
cd <DLC_WORK_ROOT>
```

## 1. Fine-Tune Pose And Detector

```powershell
.\reproducibility\01_finetune_dlc_pose_and_detector.ps1
```

This calls the maintained orchestration code with explicit settings:

- pose model: DLC SuperAnimal TopViewMouse `hrnet_w32`
- pose batch size: `42`
- pose validation interval: every epoch
- pose early stopping: `5` validation epochs without improved `metrics/test.mAP`
- pose best checkpoint: `snapshot-best-<epoch>.pt`
- detector model: Faster R-CNN ResNet50 FPN v2 from the DLC SuperAnimal detector
  checkpoint
- detector batch size: `2`
- detector validation interval: every epoch
- dataloader workers: `8`
- pinned memory: enabled

The detector stage is guarded so it refuses to continue if detector training
changes the retained pose checkpoints or selected best pose checkpoint.

If detector training is intentionally stopped after a satisfactory best detector
checkpoint has been saved, mark the detector stage complete with:

```powershell
python .\orchestration\mark_detector_train_complete.py
```

This verifies that a `snapshot-detector-best-<epoch>.pt` exists and records the
best detector metric plus the preserved best pose checkpoint in
`pipeline_state\dlc_training\detector_train.json`.

For the current run, the validated best pose checkpoint was:

```text
snapshot-best-027.pt
metrics/test.mAP = 91.76826675673418
```

## 2. Build Feature Cache And Train Classifier

```powershell
.\reproducibility\02_build_dlc_feature_cache_and_train_classifier.ps1
```

This skips DLC training, resolves the best pose snapshot, builds the
BehaviorScope-Y visual feature cache from the fine-tuned DLC HRNet backbone, and
then trains the attention-256 classifier.

The feature tap is strict by default. Cache generation stops if the HRNet-W32
tap is not exactly:

- forward hook source: `PoseModel.backbone.model`
- four HRNet branches
- branch channels: `[32, 64, 128, 256]`
- pooled concatenated feature dimension: `480`

The cache manifest records the resolved snapshot path, feature source, hooked
module, tap validation result, branch shapes, input crop shapes, and feature
dimension.

## Notes

The current cache workflow extracts DLC HRNet visual features from existing
BehaviorScope sequence crops. The classifier still reads pose/relation inputs
from the BehaviorScope NPZ samples. A trained detector is needed for a full DLC
top-down raw-video/image workflow; using detector-derived DLC poses as classifier
pose inputs would require a separate explicit conversion stage.
