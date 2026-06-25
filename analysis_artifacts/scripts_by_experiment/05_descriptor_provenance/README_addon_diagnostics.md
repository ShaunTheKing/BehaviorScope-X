# Visual Descriptor Add-On Diagnostics

## Scope

This exploratory analysis uses visual descriptors only. It evaluates group descriptors, paired animal descriptors, and combined descriptors on a balanced subset of matched held-out MARS windows.

- Matched windows available: 11,960
- Loaded windows: 1,200, balanced at 300 per behavior class
- Videos represented: 28
- Components: combined group+animal descriptors, group-only descriptors, and paired animal descriptors

## Main Findings

1. MobileNetV3 visual descriptors were less behavior-separable than YOLO-SPPF visual descriptors across all component tests. For combined descriptors, MobileNetV3 reached visual Fisher ratio 17.85 and visual-only macro-F1 0.750, compared with YOLO-SPPF Fisher ratio 32.71 and macro-F1 0.828.

2. The animal-centered descriptor stream carried most of the YOLO-SPPF behavior signal. YOLO-SPPF animal-pair descriptors reached macro-F1 0.824, close to the combined descriptor value of 0.828, whereas group-only descriptors reached 0.663. MobileNetV3 animal-pair descriptors reached 0.715 and group-only descriptors reached 0.684.

3. MobileNetV3 descriptors were highly predictive of video identity. For combined descriptors, MobileNetV3 video-identity balanced accuracy was 0.746, compared with behavior-probe accuracy 0.753. For animal-pair descriptors, video-identity balanced accuracy was 0.773, exceeding behavior-probe accuracy 0.720. This supports a nuisance-variance interpretation: MobileNetV3 visual descriptors preserved substantial route/session/video-specific information that was not preferentially organized around behavior labels.

4. YOLO-SPPF descriptors showed a larger behavior-over-video margin. For combined descriptors, YOLO-SPPF behavior accuracy was 0.828 while video-identity balanced accuracy was 0.594. DLC-HRNet showed the largest behavior-over-video margin, with combined behavior accuracy 0.728 and video-identity balanced accuracy 0.338, but its visual-only behavior probe remained below YOLO-SPPF.

5. Per-video visual Fisher ratios did not strongly explain the MobileNetV3-vs-YOLO per-video performance deltas in this sampled subset. Correlations between per-video visual separability and frame/bout F1 deltas were weak, so this analysis is not the best manuscript-facing result.

6. MobileNetV3 descriptors had larger adjacent-frame descriptor changes than YOLO-SPPF descriptors in this sampled analysis. This may indicate noisier or less temporally stable visual evidence, but it should be treated as exploratory unless repeated on the full matched set.

## Outputs

- `visual_component_probe_summary.csv`
- `visual_component_class_metrics.csv`
- `visual_temporal_smoothness_by_class.csv`
- `per_video_visual_fisher_vs_mobilenet_delta.csv`
- `per_video_fisher_delta_correlations.csv`
- `*_behavior_probe_confusion.csv`
- `group_animal_behavior_probe.png`
- `behavior_vs_video_probe.png`
- `per_video_fisher_vs_performance_delta.png`

## Best Manuscript Candidates

The strongest candidate result is the combination of behavior-probe and nuisance-probe evidence. It supports the statement that MobileNetV3 underperformance is consistent with descriptor provenance, specifically lower behavior-label accessibility and stronger video/session identity structure in the visual descriptor space.

The group-vs-animal result is also useful because it explains where the primary YOLO-SPPF visual signal sits: animal-centered descriptors carry most of the behavior-accessible visual information.

The per-video correlation result is weaker and should probably remain internal unless the full matched-set analysis strengthens it.

## Caveats

Feature channels are route-local and are not homologous across backbones. Video-identity probes are nuisance diagnostics, not biological endpoints. Per-video correlations are exploratory and limited by the number of held-out videos represented in the sampled subset. These analyses support an accessible-information and nuisance-variance interpretation, not a definitive causal mechanism.
