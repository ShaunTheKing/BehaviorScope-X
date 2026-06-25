# Tutorial Models

This folder contains optional tutorial checkpoints for trying the MobileNetV3 workflow on the included MARS tutorial videos. These files are demonstration assets for workflow testing, not manuscript-scale benchmark models.

## Contents

- `full_pose_mobilenetv3/best_pose_map5095.pt`: MobileNetV3 pose checkpoint used to build tutorial visual-feature caches.
- `full_pose_mobilenetv3/best_pose_map5095.json`: metadata for the MobileNetV3 pose checkpoint.
- `full_attn_classifier_mobilenetv3/best_model_macro_f1.pt`: sample attention-256 temporal behavior classifier.
- `full_attn_classifier_mobilenetv3/config.json`: portable classifier configuration with tutorial-relative paths.
- `model_manifest.json`: machine-readable summary of the tutorial model bundle.

## Notes

The MobileNetV3 classifier expects a MobileNetV3 feature cache. It should not be used with the YOLO-pose or DeepLabCut-HRNet cache workflows unless a matching cache has first been created.

Run `python tutorial_data/validate_tutorial_data.py` from the package root to check that tutorial videos, reference annotations, and tutorial model files are present.
