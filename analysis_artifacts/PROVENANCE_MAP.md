# BehaviorScope-X v4 provenance map

This map links the manuscript-facing experiment families to the configs, command templates, and verification artifacts included in `analysis_artifacts`.

## 01_mars_yolo_sppf

- Manuscript scope: MARS YOLO/SPPF temporal sequence classifiers and YOLO/SPPF classical baselines.
- Manuscript evidence: main experiment matrix and supplement temporal-classifier configuration table.
- Primary configs: `scripts_by_experiment/01_mars_yolo_sppf/final_config.json`; `scripts_by_experiment/01_mars_yolo_sppf/controlled_suite_config.filtered.json`; `scripts_by_experiment/01_mars_yolo_sppf/pose_baseline_rf_config.yolo_sppf.frozen.yaml`.
- Commands: `commands/01_mars_yolo_sppf_command_template.txt`.
- Verification artifacts: `verification_files/by_experiment/01_mars_yolo_sppf`; `verification_files/by_experiment/00_manuscript_tables/yolo_sppf_main`; `verification_files/by_experiment/00_manuscript_tables/classical_baselines`.
- Design summary: six stream conditions x LSTM/attention x hidden dimensions 256/512/896 x seeds 42/43/44; 30 held-out MARS videos. Classical baselines use the manuscript-facing RF/XGBoost feature sets and seeds.

## 03_mobilenetv3_backbone

- Manuscript scope: MARS MobileNetV3-native backbone portability.
- Manuscript evidence: main MobileNetV3 portability figure and supplement temporal-classifier configuration table.
- Primary configs: `scripts_by_experiment/03_mobilenetv3_backbone/final_config.json`; `scripts_by_experiment/03_mobilenetv3_backbone/controlled_suite_config.filtered.json`; `scripts_by_experiment/03_mobilenetv3_backbone/pose_baseline_rf_config.mobilenetv3_native.frozen.yaml`.
- Commands: `commands/03_mobilenetv3_backbone_command_template.txt`.
- Verification artifacts: `verification_files/by_experiment/03_mobilenetv3_backbone`; `verification_files/by_experiment/00_manuscript_tables/mobilenet_portability`.
- Design summary: full-stream MobileNetV3-native descriptors x LSTM/attention x hidden dimension 256 x seeds 42/43/44 on the same 30 held-out MARS videos.

## 04_dlc_hrnet_topdown

- Manuscript scope: DLC SuperAnimal HRNet-W32 top-down route.
- Manuscript evidence: main DLC-HRNet held-out summary and supplement DLC held-out/confusion tables.
- Primary config: `scripts_by_experiment/04_dlc_hrnet_topdown/final_config.json`.
- Commands: `commands/04_dlc_hrnet_topdown_command_template.txt`.
- Verification artifacts: `verification_files/by_experiment/04_dlc_hrnet_topdown`; `verification_files/by_experiment/00_manuscript_tables/dlc_superanimal_topdown`.
- Design summary: DeepLabCut SuperAnimal TopViewMouse HRNet-W32 detector-plus-pose cache with four-branch pre-head 480-dimensional descriptor tap, attention-256 seed 42, 28 accepted held-out MARS videos.

## 05_descriptor_provenance

- Manuscript scope: descriptor route and component analyses.
- Manuscript evidence: main descriptor-provenance figure and supplement descriptor route/component tables.
- Primary configs: `scripts_by_experiment/05_descriptor_provenance/analysis_metadata.json`; `scripts_by_experiment/05_descriptor_provenance/addon_metadata.json`.
- Verification artifacts: `verification_files/by_experiment/05_descriptor_provenance`; `verification_files/by_experiment/00_manuscript_tables/descriptor_provenance`.
- Design summary: matched held-out behavior-window probes comparing YOLO/SPPF, MobileNetV3, and DLC-HRNet descriptor routes and descriptor components.

## 06_fly_v_fly

- Manuscript scope: Fly-v-Fly short-bout cross-assay stress test.
- Manuscript evidence: main Fly-v-Fly section and supplement annotation-aware Fly-v-Fly evaluation details.
- Primary configs: `scripts_by_experiment/06_fly_v_fly/final_config.json`; `scripts_by_experiment/06_fly_v_fly/fly_adaptation_suite_config.frozen.json`.
- Commands: `commands/06_fly_v_fly_command_template.txt`.
- Verification artifacts: `verification_files/by_experiment/06_fly_v_fly/release_fly_v_fly_attention256_cw12_seed42`; `verification_files/by_experiment/00_manuscript_tables/fly_v_fly`.
- Design summary: two full-stream attention-256 windows, w16/s8 and w8/s4, seed 42, Fly-specific class weighting and random sampler; held-out movies 6-10.

## 00_figure_table_rendering

- Manuscript scope: figure and table rendering.
- Verification artifacts: `verification_files/by_experiment/00_manuscript_tables`.
- Design summary: production plotting, table, and QC scripts used to assemble manuscript-facing artifacts.

## 07_deployment_and_gui

- Manuscript scope: BehaviorScope-X GUI and deployment documentation.
- Verification artifacts: `verification_files/by_experiment/07_deployment_and_gui`.
- Design summary: GUI/tutorial support corresponding to the manuscript supplement.
