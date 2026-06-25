# BehaviorScope-Y Release Scripts

This folder contains clean manuscript and reproducibility runners for the final analyses.
They are separated from historical orchestration files so each command path
corresponds to a documented analysis family.

## Subfolders

- `mars_yolo_sppf_ablation`: MARS YOLO/SPPF stream ablation, LSTM/attention
  temporal sequence classifiers, and YOLO/SPPF RF/XGBoost classical baselines.
- `mobilenetv3_backbone`: MobileNetV3-native feature extraction, LSTM/attention
  temporal sequence classifiers, and MobileNetV3 RF/XGBoost classical baselines.
- `fly_v_fly_adaptation`: focused Fly-v-Fly short-bout adaptation using
  attention-256, seed 42, class-weight clamp 1.2, random sampling, and cached
  held-out evaluation.
- `dlc_superanimal_topdown`: DLC SuperAnimal HRNet-W32 top-down cache creation,
  HRNet semantic feature extraction, Attention-256 classifier training, and
  held-out evaluation.
- `gui_application`: launch wrapper and deployment notes for the Qt
  annotation and analysis desktop application, including tutorial metadata,
  dataset import, timeline annotation, export, training, and inference panels.
- `gui_application_backup`: source-level backup mirror of the current GUI
  application.
- `gui_application_backup_barebones_20260604`: archived copy of the earlier
  minimal GUI before the richer release app was ported in.
- `supporting_scripts`: fresh copies of the lower-level scripts called by the
  release runners.

Each analysis folder contains its own `final_config.json` and runner. Run
`python <runner>.py plan` before running any stage.
