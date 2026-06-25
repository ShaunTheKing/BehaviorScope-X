# Manuscript Reproduction Inputs

This file maps each manuscript-facing analysis to the materials users need to rebuild the data products: split/window loading, feature-cache construction, model training, held-out evaluation, and table/figure verification.

External data and model checkpoint binaries are not bundled. Paths in command templates use placeholders defined in `PATH_AND_REPRODUCTION_POLICY.md`.

## Shared Runtime Scripts

- Window/split loading: `scripts_by_experiment/00_shared_and_entrypoints/supporting_scripts/shared_scripts/scripts/prepare_full_video_npz.py`
- YOLO/SPPF feature cache: `scripts_by_experiment/00_shared_and_entrypoints/supporting_scripts/shared_scripts/scripts/precompute_visual_features_y.py`
- MobileNetV3 feature cache: `scripts_by_experiment/00_shared_and_entrypoints/supporting_scripts/shared_scripts/scripts/precompute_mobilenetv3_features_y.py`
- Temporal classifier training: `scripts_by_experiment/00_shared_and_entrypoints/supporting_scripts/shared_scripts/scripts/train_y.py`
- MARS held-out evaluation: `scripts_by_experiment/00_shared_and_entrypoints/supporting_scripts/shared_scripts/scripts/eval_manifest_classifier_y.py`
- Fly NPZ/cache preparation: `scripts_by_experiment/00_shared_and_entrypoints/supporting_scripts/shared_scripts/fly_to_behaviorscope_npz_full.py`
- Fly cached evaluation: `scripts_by_experiment/00_shared_and_entrypoints/supporting_scripts/shared_scripts/fly_run_cached_eval.py`
- Classical baseline scripts: `scripts_by_experiment/02_classical_baselines/pose_baseline_rf/`

## MARS YOLO/SPPF

- Primary run config: `scripts_by_experiment/01_mars_yolo_sppf/final_config.json`
- Frozen suite config: `scripts_by_experiment/01_mars_yolo_sppf/controlled_suite_config.filtered.json`
- Classical baseline config: `scripts_by_experiment/01_mars_yolo_sppf/pose_baseline_rf_config.yolo_sppf.frozen.yaml`
- Reproduction command template: `commands/01_mars_yolo_sppf_command_template.txt`
- Historical release plan: `scripts_by_experiment/01_mars_yolo_sppf/release_command_plan.txt`
- Rebuild stages covered by the command template: MARS train/validation NPZ, YOLO/SPPF train feature cache, held-out NPZ, held-out feature cache, annotation audit, classical matrices, RF/XGBoost training/evaluation, LSTM/attention temporal training/evaluation, summary generation.
- Verification artifacts: `verification_files/by_experiment/01_mars_yolo_sppf/` and `verification_files/by_experiment/00_manuscript_tables/yolo_sppf_main/`

## MARS Classical Baselines

- Primary baseline config: `scripts_by_experiment/02_classical_baselines/pose_baseline_rf/config.yaml`
- Baseline scripts: `scripts_by_experiment/02_classical_baselines/pose_baseline_rf/`
- Reproduction command template: `commands/01_mars_yolo_sppf_command_template.txt`
- Verification artifacts: `verification_files/by_experiment/02_classical_baselines/` and `verification_files/by_experiment/00_manuscript_tables/classical_baselines/`

## MARS MobileNetV3

- Primary run config: `scripts_by_experiment/03_mobilenetv3_backbone/final_config.json`
- Frozen suite config: `scripts_by_experiment/03_mobilenetv3_backbone/controlled_suite_config.filtered.json`
- Classical baseline config: `scripts_by_experiment/03_mobilenetv3_backbone/pose_baseline_rf_config.mobilenetv3_native.frozen.yaml`
- Reproduction command template: `commands/03_mobilenetv3_backbone_command_template.txt`
- Historical release plan: `scripts_by_experiment/03_mobilenetv3_backbone/release_command_plan.txt`
- Rebuild stages covered by the command template: MARS train/validation NPZ, held-out NPZ, annotation audit, MobileNetV3 train/held-out feature caches, classical matrices, RF/XGBoost training/evaluation, LSTM/attention temporal training/evaluation, summary generation.
- Verification artifacts: `verification_files/by_experiment/03_mobilenetv3_backbone/` and `verification_files/by_experiment/00_manuscript_tables/mobilenet_portability/`

## DLC-HRNet Top-Down

- Primary run config: `scripts_by_experiment/04_dlc_hrnet_topdown/final_config.json`
- DLC conversion table: `scripts_by_experiment/04_dlc_hrnet_topdown/mars_to_superanimal_conversion_table.csv`
- DLC scripts and documentation: `scripts_by_experiment/04_dlc_hrnet_topdown/`
- Reproduction command template: `commands/04_dlc_hrnet_topdown_command_template.txt`
- Historical release plan: `scripts_by_experiment/04_dlc_hrnet_topdown/release_command_plan.txt`
- Rebuild stages covered by the command template: DLC pose/detector fine-tuning, train/validation NPZ, HRNet feature cache, attention-256 classifier training, held-out manifest preparation, held-out NPZ, held-out HRNet feature cache, held-out evaluation.
- Verification artifacts: `verification_files/by_experiment/04_dlc_hrnet_topdown/` and `verification_files/by_experiment/00_manuscript_tables/dlc_superanimal_topdown/`

## Descriptor Provenance

- Primary metadata: `scripts_by_experiment/05_descriptor_provenance/analysis_metadata.json`
- Component metadata: `scripts_by_experiment/05_descriptor_provenance/addon_metadata.json`
- Analysis scripts and outputs: `scripts_by_experiment/05_descriptor_provenance/`
- Verification artifacts: `verification_files/by_experiment/05_descriptor_provenance/` and `verification_files/by_experiment/00_manuscript_tables/descriptor_provenance/`

## Fly-v-Fly

- Primary run config: `scripts_by_experiment/06_fly_v_fly/final_config.json`
- Frozen suite config: `scripts_by_experiment/06_fly_v_fly/fly_adaptation_suite_config.frozen.json`
- Reproduction command template: `commands/06_fly_v_fly_command_template.txt`
- Historical release plan: `scripts_by_experiment/06_fly_v_fly/release_command_plan.txt`
- Rebuild stages covered by the command template: Fly train-window loading/cache reuse, attention-256 training for w16/s8 and w8/s4, cached held-out evaluation for movies 6-10 and manuscript annotation modes.
- Verification artifacts: `verification_files/by_experiment/06_fly_v_fly/release_fly_v_fly_attention256_cw12_seed42/` and `verification_files/by_experiment/00_manuscript_tables/fly_v_fly/`

## Figure and Table Rendering

- Production scripts: `scripts_by_experiment/00_figure_table_rendering/production_analysis/`
- Final rendered figures: `figures/production/`
- Production table inputs: `tables/production/`
- Table/figure verification data: `verification_files/by_experiment/00_manuscript_tables/`
- Manuscript mapping: `MANUSCRIPT_FIGURE_TABLE_MAP.csv`
- Graph reproduction guide: `GRAPH_REPRODUCTION.md`
- Per-figure redraw status: `GRAPH_REPRODUCTION_STATUS.csv`
- Table-backed redraw command: `python scripts_by_experiment/00_figure_table_rendering/render_table_backed_figures.py`
- Upstream output path contract: `UPSTREAM_OUTPUT_PATHS.md`
- Full figure rendering template after upstream runs: `commands/00_render_all_manuscript_figures_from_upstream_outputs_template.ps1`
