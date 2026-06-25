# BehaviorScope-Y Tutorial Testbed

This folder is a small curated MARS subset for learning and smoke-testing the BehaviorScope-Y GUI workflow. It is not a benchmark dataset and is not intended to reproduce manuscript-scale model performance.

This folder is complete when `videos/train`, `videos/val`, and `videos/test` contain the six MP4s listed in `source_manifest.csv`. Public repository distributions may keep only the tutorial structure, manifests, class labels, provenance, and ground-truth annotations because regular GitHub repositories block files larger than 100 MiB.

## Contents

- `videos/train`, `videos/val`, `videos/test`: tutorial MP4 videos.
- `annotations/train`, `annotations/val`, `annotations/test`: matching BENTO `.annot` ground-truth files.
- `source_manifest.csv`: relative-path manifest for the Prepare Full Video workflow.
- `class_names.txt`: behavior class order used by the tutorial.
- `provenance.json`: source paths, split assignment, file sizes, and bout counts.

## Recommended GUI Workflow

1. Open the BehaviorScope-Y Qt GUI.
2. Use `Tutorial > Download/Load BehaviorScope-Y tutorial...`.
3. If the MP4s are not present locally, place them under the paths listed in `source_manifest.csv`, then choose `Locate Folder`.
4. The GUI imports the tutorial videos, assigns the train/val/test splits, adds the MARS behavior labels, converts the `.annot` ground-truth spans into BehaviorScope-Y timeline annotations, and fills the `Prepare Full Video` paths.
5. Prepare the full-video NPZ windows using a pose checkpoint that is appropriate for your licensing context.
6. For the public tutorial default, build the `MobileNetV3 Cache` using a MobileNetV3-native pose-backbone checkpoint. This avoids making YOLO-pose weights part of the tutorial distribution.
7. Continue through Train, Bundle/Export, and Inference. The YOLO/SPPF cache remains available for users who supply their own compatible YOLO-pose checkpoint.

## Split Policy

The tutorial uses:

- `train`: three short annotated examples
- `val`: one short validation example
- `test`: two held-out examples

The training tab should keep its default split settings: train on `train`, validate on `val`. The `test` split is preserved for inference/evaluation practice and should not be used for training.

## GUI Annotations

`Tutorial > Download/Load BehaviorScope-Y tutorial...` converts all 206 BENTO `.annot` spans into approved BehaviorScope-Y annotations so the timeline shows what the completed tutorial labels should look like. Reloading the tutorial is idempotent and does not duplicate existing spans.

## Notes

The MP4 filenames and annotation filenames are normalized to `<video_id>.mp4` and `<video_id>.annot` so the structure is easy to follow. Original source locations are recorded in `provenance.json`.
