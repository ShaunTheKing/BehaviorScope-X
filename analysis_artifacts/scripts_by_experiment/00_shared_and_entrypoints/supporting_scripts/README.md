# Supporting Script Copies

This folder contains fresh copies of the lower-level scripts called by the
clean release runners. The runners in the sibling analysis folders are the
recommended entry points; these files are included so the public release has an
auditable copy of the training, evaluation, feature-cache, Fly NPZ, and
classical-baseline code paths.

## Included groups

- `run_controlled_comparison_suite.py`: shared planner used by the MARS
  YOLO/SPPF and MobileNetV3 release runners.
- `run_fly_v_fly_adaptation_suite.py`: independent final Fly-v-Fly adaptation
  suite used by the Fly release runner.
- `summarize_controlled_comparison_results.py`: controlled MARS summary helper.
- `pose_baseline_rf`: RF/XGBoost matrix construction, training, evaluation, and
  annotation-provenance audit scripts.
- `shared_scripts/fly_to_behaviorscope_npz_full.py`: Fly-v-Fly NPZ builder.
- `shared_scripts/fly_run_cached_eval.py`: Fly-v-Fly cached held-out evaluator.
- `shared_scripts/scripts`: Qt GUI application, MARS full-video NPZ
  construction, temporal sequence classifier training, held-out evaluation,
  visual-feature precomputation, model definitions, crop helpers, and utility
  modules.

The analysis-specific `final_config.json` files live in the sibling runner
folders, not here.
