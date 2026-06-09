# Supporting Script Copies

This folder contains analysis helper code called by the model-family runners.
The runners in the sibling analysis folders are the recommended entry points.
Core GUI, cache, training, and inference scripts live once at the package root;
this folder keeps only the comparison-suite, classical-baseline, and Fly-v-Fly
helpers that are specific to the manuscript analyses.

## Included groups

- `run_controlled_comparison_suite.py`: shared planner used by the MARS
  YOLO/SPPF and MobileNetV3 analysis runners.
- `run_fly_v_fly_adaptation_suite.py`: independent final Fly-v-Fly adaptation
  suite used by the Fly analysis runner.
- `summarize_controlled_comparison_results.py`: controlled MARS summary helper.
- `pose_baseline_rf`: RF/XGBoost matrix construction, training, evaluation, and
  annotation-provenance audit scripts.
- `fly_v_fly/fly_to_behaviorscope_npz_full.py`: Fly-v-Fly NPZ builder.
- `fly_v_fly/fly_run_cached_eval.py`: Fly-v-Fly cached held-out evaluator.
- package-root scripts: MARS full-video NPZ construction, temporal sequence
  classifier training, held-out evaluation, visual-feature precomputation,
  model definitions, crop helpers, and utility modules.

The analysis-specific `final_config.json` files live in the sibling runner
folders, not here.


