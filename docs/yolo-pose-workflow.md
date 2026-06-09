# YOLO-pose Workflow

The YOLO-pose path is one validated BehaviorScope-X pose workflow. It uses one YOLO-pose checkpoint for detection, keypoint estimation, and frozen visual-feature extraction, then trains a temporal classifier on pose-derived and visual streams.

## Inputs

You need:

- full-video annotation export from the GUI,
- a class-name file,
- a YOLO-pose `.pt` checkpoint,
- a destination folder for sequence caches and training outputs.

## Recommended Stage Order

### 1. Full-Video Cache

Open `YOLO-pose > Full-Video Cache`.

This stage reads full videos and `.annot` files, runs YOLO-pose, creates crops and pose-derived features, and writes sliding-window NPZ files plus a `sequence_manifest.json`.

Recommended settings:

- use the same keypoint layout used to train the YOLO-pose checkpoint,
- keep manifest validation enabled,
- use `skip existing` when resuming an interrupted cache build,
- keep the output root specific to the dataset and split.

### 2. Train/Val Feature Cache

Open `YOLO-pose > Train/Val Feature Cache`.

This stage precomputes frozen visual descriptors from the YOLO-pose backbone. Precomputing features makes classifier training faster and easier to reproduce because the temporal model sees a fixed feature cache.

### 3. Train + Validation Eval

Open `YOLO-pose > Train + Validation Eval`.

This stage trains the temporal classifier. The classifier can use:

- pose geometry streams,
- visual descriptors from the YOLO-pose backbone,
- class weighting,
- temporal smoothing or threshold decoding,
- optional model bundling.

When bundling is enabled, training writes a single `.pt` file containing both the classifier and the YOLO-pose checkpoint reference needed for inference.

### 4. Inference

Open `YOLO-pose > Inference` for one video or `YOLO-pose > Batch` for a folder.

Use bundled models when possible. They reduce path mistakes because users do not need to supply a separate classifier, config, and YOLO checkpoint.

### 5. Ethograms + Bouts

Open `YOLO-pose > Ethograms + Bouts` to create ethogram timelines and bout-duration summaries from YOLO-pose temporal prediction CSVs. Use `*.smoothed_frames.csv` files for frame-exact timelines; window prediction CSVs are converted by probability averaging or label voting.

## Outputs

Typical outputs include:

- sequence NPZ cache,
- feature cache,
- classifier checkpoint,
- `config.json`,
- training log,
- confusion matrices,
- behavior CSV,
- ethogram and bout summary CSVs,
- optional keypoint CSV,
- optional annotated review MP4.

Use `YOLO-pose > Outputs` to inventory these artifacts.

## When To Use Legacy Clip Cache

Use `YOLO-pose > Legacy Clip Cache` only when your dataset is already organized as class-folder clips. For new full-video behavioral work, use full-video annotations and full-video caches.



