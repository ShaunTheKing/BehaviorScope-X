# DLC-HRNet Held-Out Evaluation Analysis

Date generated: 2026-06-08

## Input

- Evaluated videos: 28
- Manifest videos: 28
- Windows: 11960
- Blocking failures: 0
- Threshold decoder disabled: True

## Pooled Per-Video Summary

| Prediction | Accuracy | Frame macro F1 | Bout F1 IoU@0.25 | Bout F1 IoU@0.50 |
|---|---:|---:|---:|---:|
| Raw windows | 0.895 | 0.749 | 0.670 | 0.555 |
| Smoothed frames | 0.595 | 0.638 | 0.537 | 0.443 |

Scores are means over held-out videos, not pooled frame-level aggregates.

## Confusion Counts

| True \ Pred | Attack | Investigation | Mount | Other |
|---|---:|---:|---:|---:|
| Attack | 15941 | 2216 | 608 | 303 |
| Investigation | 5300 | 62743 | 2008 | 6647 |
| Mount | 606 | 1656 | 32891 | 126 |
| Other | 3305 | 13545 | 1005 | 198828 |

## Pose-Backbone Feature Pipeline Comparison

| Backbone | Prediction | Frame macro F1 | Bout F1 IoU@0.25 | Bout F1 IoU@0.50 |
|---|---|---:|---:|---:|
| YOLO-SPPF | raw_windows | 0.758 | 0.691 | 0.549 |
| YOLO-SPPF | smoothed_frames | 0.757 | 0.691 | 0.549 |
| MobileNetV3 | smoothed_frames | 0.677 | 0.601 | 0.439 |
| DLC-HRNet | raw_windows | 0.749 | 0.670 | 0.555 |
| DLC-HRNet | smoothed_frames | 0.638 | 0.537 | 0.443 |

The raw-window DLC-HRNet comparison is directly paired with YOLO-SPPF on the same 28 held-out videos. MobileNetV3 production rows are available as smoothed-frame outputs, so the three-backbone comparison should be interpreted as a pipeline-level comparison rather than an isolated visual-backbone-only ablation.

## Generated Files

- `dlc_yolo_mobilenet_per_video_comparison_long.csv`
- `dlc_yolo_mobilenet_metric_comparison_summary.csv`
- `dlc_heldout_per_video_summary.csv`
- `dlc_heldout_per_class_summary.csv`
- `dlc_heldout_confusion_counts_raw.csv`
- `dlc_heldout_confusion_recall_raw.csv`
- `dlc_heldout_metric_summary.png`
- `dlc_heldout_metric_summary.pdf`
- `dlc_heldout_per_class_scores.png`
- `dlc_heldout_per_class_scores.pdf`
- `dlc_heldout_confusion_recall_raw.png`
- `dlc_heldout_confusion_recall_raw.pdf`
- `dlc_vs_yolo_mobilenet_metric_comparison.png`
- `dlc_vs_yolo_mobilenet_metric_comparison.pdf`
- `dlc_vs_yolo_raw_per_video_delta.png`
- `dlc_vs_yolo_raw_per_video_delta.pdf`
