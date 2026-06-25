# Upstream Output Paths

Some manuscript figures can be redrawn directly from bundled tables. Others
need outputs from model training, full-video inference, pose-cache generation,
or held-out evaluation. This file lists the folder layout those upstream
figures expect after a full rerun.

The folders do not need to live on the author machine. They can be generated
under any local `<RUN_ROOT>` or `<DLC_WORK_ROOT>` as long as the same structure
is preserved, or the plotting scripts are called with explicit path flags.

## Placeholder Roots

- `<PACKAGE_ROOT>`: unpacked `analysis_artifacts` folder.
- `<RUN_ROOT>`: local working root for generated caches, training runs,
  evaluation runs, and source MP4 caches.
- `<MARS_DATA_ROOT>`: local MARS behavior dataset folder containing
  `train/`, `validation/`, `test_1/`, and `test_2/`.
- `<DATA_ROOT>`: local parent folder for optional raw source data trees.
- `<DLC_WORK_ROOT>`: local DLC SuperAnimal working root.

When a script flag is named `--mars_root`, pass `<MARS_DATA_ROOT>` directly.

## MARS YOLO/SPPF and MobileNetV3

The controlled-comparison command templates write the primary MARS run here:

```text
<RUN_ROOT>/outputs/controlled_comparison_runs/run_20260528_205727/
```

Figure scripts that depend on this run use these subfolders:

```text
neural_eval/
neural_training/
manuscript_figures/interim_qc/classical_rf_xgb/
manuscript_figures/interim_qc/throughput/
feature_caches/mobilenetv3_native/
```

Qualitative bout-example figures also expect source MP4s here:

```text
<RUN_ROOT>/outputs/source_mp4_cache/mars/
```

## Fly-v-Fly

The focused Fly-v-Fly evaluation is expected here:

```text
<RUN_ROOT>/outputs/controlled_comparison_runs/fly_adaptation_attention256_cw12_seed42/fly/eval/
```

The Fly pose-validation figure uses:

```text
<RUN_ROOT>/Fly_YOLO_Pose_Model/weights/Fly_model_val.txt
```

## DLC-HRNet

The DLC held-out evaluation folder is expected here:

```text
<DLC_WORK_ROOT>/behaviorscope_outputs/dlc_topdown_heldout_eval/mars_dlc_topdown_hrnet_attention256_seed42/
```

The DLC command template also uses the DLC project and checkpoint folders under
the selected `<DLC_WORK_ROOT>`.

## Descriptor Provenance

The descriptor-provenance scripts can use generated outputs under:

```text
<RUN_ROOT>/outputs/
```

The manuscript descriptor-provenance multiplot can also be redrawn from the
bundled tables:

```text
tables/production/descriptor_provenance/
```

## Reproduced Output Targets

Keep regenerated figures and tables outside the bundled expected-output
folders:

```text
<PACKAGE_ROOT>/figures_reproduced/production/
<PACKAGE_ROOT>/tables_reproduced/production/
```

This keeps `figures/production/` and `tables/production/` available as the
reference outputs for comparison.
