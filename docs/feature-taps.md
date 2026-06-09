# Feature Taps

BehaviorScope-X reuses visual representations from pose models before those representations are compressed into keypoint coordinates. The goal is not to claim that one pose backbone is universally best. The goal is to make pose-trained visual information available to downstream behavior classifiers without training a separate video backbone.

## What Is a Feature Tap?

A feature tap is the point inside a pose model where intermediate activations are extracted, pooled, cached, and supplied to a behavior classifier. A useful tap should preserve:

- assay-specific animal appearance,
- contact and overlap context,
- local scene information,
- enough semantic depth to support behavior discrimination,
- enough spatial information to remain tied to the animal crop.

Very early layers may be too local and texture-heavy. Very late heatmap or coordinate layers may be too compressed. The preferred region is usually late backbone or neck features before the final pose-output heads.

## YOLO-pose Tap

The YOLO-pose workflow uses a late YOLO-pose backbone/neck representation analogous to the feature region that supports the final pose heads. In the MARS runs, this was chosen to resemble an SPPF-level visual summary: semantically rich, downstream of much of the spatial feature aggregation, and upstream of final keypoint decoding.

Operationally, the GUI and training scripts use the YOLO-pose checkpoint for:

- detection,
- keypoint estimation,
- crop construction,
- frozen visual-feature extraction.

The exact layer index is controlled by the YOLO feature-cache settings. Keep that setting fixed across train/validation and held-out extraction.

## MobileNetV3 Tap

The MobileNetV3 workflow extracts from the final MobileNetV3-large feature block before the pose-output layer. The descriptor is global-average-pooled to create a compact visual vector for each crop.

For the controlled MARS workflow:

- descriptor size is 960 dimensions per crop,
- the workflow uses group and animal crops,
- with one group crop and two individual-animal crops, the raw visual descriptor is 3 x 960 = 2880 dimensions per frame before downstream sequence aggregation.

This tap is deliberately late in the backbone because the comparison asks whether pose-trained visual representations can be reused, not whether raw pixels alone are sufficient.

## DeepLabCut-HRNet Tap

The DeepLabCut-HRNet workflow extracts from HRNet-W32 multi-resolution backbone branches before final keypoint heads. Each branch is spatially pooled, and the branch descriptors are concatenated.

For HRNet-W32, the branch channel widths are:

- 32 channels,
- 64 channels,
- 128 channels,
- 256 channels.

After global average pooling and concatenation, the visual descriptor is:

```text
32 + 64 + 128 + 256 = 480 dimensions per crop
```

With one group crop and two individual-animal crops, the raw DLC-HRNet visual descriptor is:

```text
3 x 480 = 1440 dimensions per frame
```

Some internal command-line tools refer to this setting as `semantic_concat`. In user-facing terms, it means pooled multi-resolution HRNet branch concatenation.

## General Tap-Selection Rubric

When adapting BehaviorScope-X to another pose architecture, use the following rubric.

| Architecture family | Recommended tap region | Avoid | Reason |
| --- | --- | --- | --- |
| YOLO-style pose | Late backbone or neck features before pose heads | final keypoint tensors only | retains semantic context while staying tied to detections |
| ResNet-style pose | C4 or C5 features before deconvolution or heatmap heads | final heatmaps only | captures high-level visual context before coordinate compression |
| HRNet | pooled multi-resolution branches before final keypoint heads | only the last heatmap layer | preserves multi-scale spatial and semantic information |
| U-Net/hourglass | bottleneck plus late decoder features before confidence maps | only final confidence maps | balances semantic compression with spatial localization |
| Lightweight mobile backbones | final convolutional feature block before task head | logits or coordinates only | gives compact descriptors suitable for caching |

## Reproducibility Checklist

For every feature-cache run, record:

- pose model family and checkpoint source,
- crop definition and crop size,
- layer or branch selection,
- pooling method,
- descriptor dimension per crop,
- number of crops per frame,
- final visual descriptor dimension,
- dtype used in the feature cache,
- train/validation and held-out manifests.

These details make downstream comparisons interpretable and reproducible.



