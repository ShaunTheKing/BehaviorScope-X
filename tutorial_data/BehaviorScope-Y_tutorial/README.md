# BehaviorScope-X Tutorial Testbed

This folder is a small curated MARS subset for learning and smoke-testing the BehaviorScope-X GUI workflow. The directory name is retained for compatibility with existing tutorial manifests. It is not a benchmark dataset and is not intended to reproduce manuscript-scale model performance.

The Git repository keeps the tutorial structure, manifests, class labels, provenance, and reference annotations. MP4 video files are distributed separately because regular GitHub repositories block files larger than 100 MiB. The GUI can download the full tutorial package from Hugging Face into `tutorial_data`, or you can manually place/extract it so this folder contains `videos/train`, `videos/val`, and `videos/test`.

## Contents

- `videos/train`, `videos/val`, `videos/test`: tutorial MP4 videos.
- `annotations/train`, `annotations/val`, `annotations/test`: matching BENTO `.annot` reference files.
- `source_manifest.csv`: relative-path manifest for raw tutorial videos and annotations.
- `class_names.txt`: behavior class order used by the tutorial.
- `provenance.json`: source paths, split assignment, file sizes, and bout counts.
- `tutorial_models`: optional MobileNetV3 pose and behavior-classifier checkpoints for tutorial workflow checks.

## Recommended GUI Workflow

1. Open the BehaviorScope-X Qt GUI.
2. Use `Tutorial > Download/Load BehaviorScope-X tutorial...`.
3. If the MP4s are not present locally, choose `Download` to fetch the Hugging Face tutorial package or `Locate Folder` if you already downloaded it manually.
4. The GUI imports the tutorial videos, assigns the train/val/test splits, adds the MARS behavior labels, converts the `.annot` reference spans into BehaviorScope-X timeline annotations, and fills the first raw-video cache stage.
5. For the YOLO-pose path, add an Ultralytics YOLO-pose `.pt` checkpoint in the `YOLO-pose > Full-Video Cache` tab. Do not place the bundled MobileNetV3 checkpoint in this field.
6. Choose whether to keep the suggested output folder, then run Full-Video Cache.
7. Continue through Train/Val Feature Cache, Train + Validation Eval, Bundle/Export, and Inference.

The bundled `tutorial_models` folder is for MobileNetV3 feature-cache and classifier workflow checks after a compatible window NPZ cache exists. Those checkpoints are not YOLO-pose weights.

## Split Policy

The tutorial uses:

- `train`: three short annotated examples
- `val`: one short validation example
- `test`: two held-out examples

The training tab should keep its default split settings: train on `train`, validate on `val`. The `test` split is preserved for inference/evaluation practice and should not be used for training.

## GUI Annotations

`Tutorial > Download/Load BehaviorScope-X tutorial...` converts all 206 BENTO `.annot` spans into approved BehaviorScope-X annotations so the timeline shows what the completed tutorial labels should look like. Reloading the tutorial is idempotent and does not duplicate existing spans.

## Notes

The MP4 filenames and annotation filenames are normalized to `<video_id>.mp4` and `<video_id>.annot` so the structure is easy to follow. Original source locations are recorded in `provenance.json`.


