# Production Figure Assets

This folder contains peer-review-facing figure assets for the V9
BehaviorScope-X manuscript.

## Layout

- `source_assets/`: existing manuscript figures copied from the previous
  root-level figure folder so they are tracked alongside production outputs.
- `yolo_sppf_main/`: figures generated from the current YOLO/SPPF LSTM +
  attention held-out analysis.
- `production_figure_manifest.csv`: file-level manifest for all production
  figure assets.

The package preserves the production subfolder layout:

- `classical_baselines/`
- `mobilenet_portability/`
- `fly_v_fly/`
- `supplement/`

Reproduction-redrawn outputs should be written to `figures_reproduced/production/`
so the expected outputs in this folder remain unchanged for comparison.
