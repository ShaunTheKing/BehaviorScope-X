# DLC SuperAnimal Top-Down BehaviorScope-X Analysis

This folder contains the release scripts for the DLC SuperAnimal top-down
BehaviorScope-X analysis. It is organized as a clean, Python-first command path
for reproducing the DLC portability/stress-test of amortized pose vision.

The intended entry point is:

```bash
python run_dlc_superanimal_topdown.py plan
```

Use `plan` first. It writes `release_command_plan.json` and
`release_command_plan.txt` without running long jobs. To run a specific stage:

```bash
python run_dlc_superanimal_topdown.py run --stage build_heldout_npz
```

## Analysis Stages

1. `train_dlc_pose_detector`: fine-tune the DLC SuperAnimal pose model and detector. Best checkpoints are selected inside the training script from declared validation metrics. No manual detector-completion marker is part of this release path.
2. `build_trainval_npz`: build the DLC top-down train/validation full-video NPZ cache.
3. `build_trainval_hrnet_cache`: extract HRNet-W32 `semantic_concat` visual descriptors from the DLC top-down crops.
4. `train_classifier`: train the Attention-256 BehaviorScope-X classifier from DLC pose-derived and HRNet visual features.
5. `prepare_heldout_manifest`: prepare a held-out MP4/.annot manifest if converted held-out MP4s are not already available.
6. `build_heldout_npz`: build the DLC top-down held-out NPZ cache for the 28 manuscript MARS test videos.
7. `build_heldout_hrnet_cache`: extract held-out HRNet-W32 visual descriptors.
8. `evaluate_heldout`: evaluate held-out frame, bout, and per-video metrics using the same evaluator family as the YOLO/SPPF and MobileNetV3 analyses.

## Feature Tap

The DLC visual descriptor uses HRNet-W32 `semantic_concat`: four HRNet branches
are global-average-pooled and concatenated:

```text
32 + 64 + 128 + 256 = 480 features per crop
```

This is the DLC/HRNet analogue of using a late semantic pose-backbone tap in
the YOLO/SPPF analysis. It is not a claim that DLC is better or worse than
YOLO-pose; it tests whether the same amortized pose vision principle transfers
to a widely used external pose-estimation framework.

## Held-Out Set

The held-out DLC manifest is aligned to the manuscript production YOLO/SPPF and
MobileNetV3 tables: 28 human-labeled MARS test videos. Two `test_2` folders with
prediction-named annotation files are excluded from the production comparison.

The held-out sequence manifest uses the existing two-container cache schema:
`test_1` videos are stored in the `train` container and `test_2` videos are
stored in the `val` container. The original source split is retained in each row
as `mars_split`, and the evaluator selects held-out videos by that field.

## Files

- `final_config.json`: frozen current configuration for the DLC analysis path.
- `run_dlc_superanimal_topdown.py`: Python entry point for planning and running stages.
- `supporting_scripts/`: copied support scripts used by the runner.
- `configs/`: conversion table and small configuration inputs.
- `docs/`: feature-tap and next-step documentation.
- `provenance_templates/`: expected output notes.
- `SCRIPT_MANIFEST.csv`: SHA256 hashes for the copied scripts and docs.
