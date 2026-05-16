# BehaviorScope-Y Utility Scripts

This folder contains command-line utilities for preparing behavior datasets,
building YOLO-pose inputs, evaluating trained models, and reproducing the
manuscript analyses. Paths are intentionally user supplied or relative to the
current working directory; the scripts should not assume a local lab folder.

For the primary user workflow, start with the repository-level `README.md`.
The usual sequence is:

1. Convert full-video annotations to BehaviorScope-Y NPZ files.
2. Build or reuse a YOLO visual feature cache.
3. Train a temporal classifier with configurable LSTM or attention heads.
4. Optionally export a single bundled `.pt` file containing both the YOLO-pose
   checkpoint and the behavior classifier.
5. Run inference and evaluate against held-out annotations.

## Dataset Preparation

- `mars_to_behaviorscope_npz.py` converts MARS-style annotations into NPZ
  sequence files for model training.
- `mars_to_behaviorscope_npz_full.py` builds full-video NPZ files when the
  full temporal context should be preserved.
- `mars_to_behaviorscope_clips.py` extracts class-specific clip folders from
  MARS annotations.
- `mars_to_behaviorscope_clips_full.py` extracts full-video clip variants.
- `mars_to_yolo_pose.py` builds an Ultralytics YOLO-pose dataset from MARS
  pose annotations.
- `add_other_class.py` adds non-bout `other` samples to an existing YOLO-pose
  dataset.
- `link_mp4_cache.py` links or mirrors decoded videos into a cache location.
- `salvage_feature_cache.py` repairs or consolidates partial feature-cache
  runs when long preprocessing jobs were interrupted.

## Quality Control

- `qc_overlay.py` renders bounding boxes, keypoints, and class labels for
  sampled YOLO-pose frames.
- `compare_inference_screenshots.py` compares visual inference outputs.
- `build_kpt_eval_figures.py` creates keypoint-evaluation figures.
- `eval_yolo_kpt_vs_mars_gt_fast.py` evaluates YOLO keypoints against MARS
  ground-truth keypoints.
- `evaluate_yolo_pipeline_smoke.py` runs a small YOLO-pose smoke test.

## Model Evaluation

- `batch_infer_eval_behaviorscope_y.py` runs inference and evaluation over a
  batch of videos.
- `evaluate_against_annot.py` compares model predictions against annotation
  files.
- `evaluate_full_video_ablation_suite.py` evaluates the full-video ablation
  suite used for manuscript comparisons.
- `evaluate_clip_ablation_suite.py` evaluates the clip-trained ablation suite.
- `analyze_full_video_test_results.py` summarizes full-video evaluation
  outputs.
- `analyze_clip_test_results.py` summarizes clip-trained evaluation outputs.
- `evaluate_full_model_test_set.py` evaluates the full multimodal model on a
  held-out test set.
- `evaluate_attention_model_test_set.py` evaluates the attention-head model on
  a held-out test set.
- `evaluate_single_video_full_model.py` evaluates the full model on one video.
- `threshold_sweep_eval.py` sweeps post-processing thresholds and evaluates
  bout-level behavior metrics.
- `evaluate_stride_comparison.py` compares inference stride settings.
- `run_inference_stride_sweep.py` runs a configurable stride sweep.
- `benchmark_inference_fps.py` measures end-to-end inference throughput.
- `ingest_stress_benchmark_results.py` consolidates stress-test benchmark
  outputs.

## Bout and Post-Processing Analyses

- `analyze_deep_bout_metrics.py` performs detailed bout-level error analysis.
- `probe_merged_bouts.py` inspects cases where predicted bouts merge separate
  annotated events.
- `sweep_investigation_thresholds.py` sweeps investigation-specific decision
  thresholds.
- `sweep_posthoc_smoothing.py` evaluates smoothing, gap-fill, and minimum-bout
  parameters.
- `sweep_temporal_splitter.py` evaluates temporal splitting rules for merged
  bouts.
- `tune_confidence_temporal_splitter.py` tunes confidence-based bout splitting.

## Notes For Public Use

- Run scripts from the repository root unless a script says otherwise.
- Prefer explicit arguments such as `--mars_root`, `--npz_dir`,
  `--feature_cache`, `--classifier`, and `--out_dir` over editing paths inside
  scripts.
- The repository does not ship a YOLO-pose checkpoint. Users should provide
  their own checkpoint or use a compatible public Ultralytics pose model.
- Manuscript reproduction scripts may expect the same directory structure used
  by the generated outputs, but they should not require private absolute paths.
