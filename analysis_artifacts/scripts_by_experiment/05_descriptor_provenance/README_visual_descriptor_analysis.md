# Visual Descriptor Mechanism Analysis

This exploratory analysis compares cached visual descriptors from matched held-out MARS windows across three pose-backbone routes: YOLO-SPPF, MobileNetV3, and DLC-HRNet. It does not modify manuscript-facing outputs.

## Unit And Inputs

- Unit: matched held-out behavior window.
- Sample: 2,000 windows, balanced at 500 per class, drawn from 11,960 windows common to all three visual feature caches.
- Vectorization: concatenate group, animal 1, and animal 2 visual descriptors per frame, then average across the 32-frame window.
- Scope: visual descriptors only. Pose/relation streams are intentionally excluded.

## Main Result

MobileNetV3 had weaker behavior-relevant visual descriptor structure than YOLO-SPPF in this sampled matched-window analysis, despite having a larger raw descriptor.

| Route | Visual dims | Fisher ratio | Visual-only probe macro-F1 | Effective dim, top 128 PCs | Top 64 PCs explained |
|---|---:|---:|---:|---:|---:|
| YOLO-SPPF | 768 | 54.00 | 0.849 +/- 0.015 | 14.60 | 0.902 |
| MobileNetV3 | 2,880 | 26.06 | 0.757 +/- 0.028 | 7.65 | 0.841 |
| DLC-HRNet | 1,440 | 63.33 | 0.704 +/- 0.059 | 13.25 | 0.911 |

## Interpretation

The MobileNetV3 descriptor was larger but less behavior-separable by Fisher ratio and less decodable by a grouped linear probe than YOLO-SPPF. This supports a descriptor-provenance interpretation of the MobileNetV3 performance gap: the native MobileNetV3 feature tap produced usable visual evidence, but the evidence was less accessible for behavior labels than the YOLO-SPPF visual descriptor under this analysis.

The DLC-HRNet result should be interpreted separately. Its Fisher ratio was high, but the grouped linear probe macro-F1 was lower and more variable. That pattern is consistent with a route whose descriptor geometry differs substantially and may require its route-specific classifier/postprocessing context.

## Representation Similarity

Linear CKA on the balanced sample showed:

- YOLO-SPPF vs MobileNetV3: 0.779
- YOLO-SPPF vs DLC-HRNet: 0.376
- MobileNetV3 vs DLC-HRNet: 0.218

This supports the manuscript claim that visual descriptors from different pose routes are not interchangeable. It does not prove a causal mechanism by itself.

## Diagnostic Outputs

- `visual_descriptor_summary.csv`: route-level variance, dimensionality, separability, and probe metrics.
- `visual_descriptor_linear_cka.csv`: pairwise representation similarity.
- `*_video_leverage.csv`: videos contributing most to top-PC descriptor variance.
- `*_top_window_leverage.csv`: individual windows contributing most to top-PC descriptor variance.
- `*_frame_position_variance.csv`: within-window frame positions with largest descriptor variance.
- `*_top_behavior_separating_channels.csv`: route-local channels with strongest behavior separation.
- `visual_descriptor_behavior_structure.png`: Fisher ratio and probe macro-F1.
- `visual_descriptor_effective_dimensionality.png`: effective dimensionality.

## Important Caveats

- Feature channels are route-local. A high-ranking MobileNetV3 channel is not homologous to a YOLO-SPPF or HRNet channel.
- This is a sampled exploratory analysis. It is appropriate for deciding whether to build a manuscript-facing analysis, but a final supplement should rerun the same code on either the full matched set or a predeclared balanced sample.
- The analysis supports an accessible-information explanation, not a complete mechanistic proof of why MobileNetV3 underperformed.
