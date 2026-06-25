# GUI Overview

BehaviorScope-X presents the workflow as one annotation surface followed by model-family tabs. This makes the GUI the single validated workflow surface while keeping the differences between pose backbones explicit. Each model-family tab is a pose-model-flexible route into the same downstream cache, classifier, evaluation, and output structure.

## Top-Level Tabs

### Annotate + Clip

Use this tab to import videos, assign splits, define behaviors, draw bout spans, review boundaries, and export full-video annotations. The legacy clip path remains available for older class-folder datasets, but full-video annotation export is the recommended route for new work.

### YOLO-pose

Use this tab when the pose model is an Ultralytics-compatible YOLO-pose checkpoint:

1. Build a full-video sequence cache.
2. Precompute YOLO-pose visual features.
3. Train and validate a temporal behavior classifier.
4. Run single-video or batch inference.
5. Create ethogram timelines and bout summaries from prediction CSVs.
6. Inspect outputs.

This path includes the most interactive inference surface because the trained classifier can be bundled with the YOLO-pose checkpoint.

### MobileNetV3

Use this tab when the pose model is a compatible MobileNetV3 pose checkpoint:

1. Build a full-video sequence cache directly from videos with the MobileNetV3 pose checkpoint.
2. Extract MobileNetV3 pose-backbone visual features.
3. Train and validate the temporal behavior classifier from the MobileNetV3 feature cache.
4. Create ethogram timelines and bout summaries from prediction CSVs.
5. Use the staged runners for larger held-out suites, static negative-control baselines, and suite-summary collection.

### DeepLabCut-HRNet

Use this tab when the pose workflow is a DeepLabCut SuperAnimal/HRNet project. The stages cover pose and detector fine-tuning, top-down cache creation, HRNet feature-cache extraction, classifier training, held-out evaluation, model-agnostic ethogram/bout summaries, and output inspection.

## Ethograms + Bouts

Each temporal pose-model workflow has an **Ethograms + Bouts** tab. The tab accepts prediction CSVs from any trained BehaviorScope-X temporal classifier and writes `ethogram_segments.csv`, `ethogram_behavior_summary.csv`, and `ethogram_summary.json`. Use `*.smoothed_frames.csv` files when you need frame-exact timelines; window-level prediction CSVs are converted by probability averaging or label voting.

## Plan Versus Run

MobileNetV3 and DeepLabCut-HRNet stages have an **Action** selector:

- `plan` prints or writes the commands that would be executed.
- `run` starts the stage.

Use `plan` before long GPU jobs, especially cache building and classifier training.

## Stop Buttons

- **Stop after step** asks the process to terminate cleanly when supported.
- **Force stop** terminates the process immediately. Use it only when the job is stuck or a clean stop is not possible.

## Outputs Tabs

Each model-family tab has an Outputs panel. Use it to inspect sequence caches, feature caches, checkpoints, training logs, confusion matrices, held-out metrics, ethogram exports, bout summaries, keypoint exports, and behavior prediction files.



