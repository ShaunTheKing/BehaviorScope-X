# MobileNetV3 Workflow

The MobileNetV3 workflow tests whether visual descriptors from a different pose backbone can support the same pose-plus-vision behavior-classification principle. The GUI can build MobileNetV3 sequence windows directly from raw videos, extract MobileNetV3 visual descriptors, and train the temporal classifier from those cached features. Larger held-out suites and static negative-control baselines remain available through staged runners.

The tutorial package can include a small MobileNetV3 pose checkpoint and a sample attention-256 classifier in `tutorial_data/BehaviorScope-X_tutorial/tutorial_models`. These files are for workflow checks on the tutorial videos; manuscript-scale evaluation should use the full recorded runs and provenance tables.

## Inputs

You need:

- train/validation and held-out manifests, or GUI-exported `source_manifest.csv` files for rebuilding them,
- source videos and `.annot` files when building sequence NPZ caches directly,
- the MobileNetV3 pose-backbone checkpoint,
- an output suite root,
- the shared analysis helper scripts included in `analysis_workflows/shared_analysis_code`.

## Recommended Stage Order

For the direct GUI panels, use the tabs in order. For staged runners, use `Action=plan` first and switch to `Action=run` only after the command and output root look correct.

### 1. Full-Video + Held-Out Caches

Use `MobileNetV3 > Full-Video Cache` to build train/validation sequence NPZ caches directly from source videos using the MobileNetV3 pose checkpoint. The emitted `sequence_manifest.json` has the same schema as the YOLO-pose and DLC cache builders, so the downstream trainer can reuse the same pose-derived streams, crop tensors, labels, and split metadata.

For manuscript-scale held-out runs, use the staged runner or a separate source manifest so held-out windows are kept separate from train/validation windows.

### 2. Train/Val Feature Cache

Use `MobileNetV3 > Train/Val Feature Cache` to extract MobileNetV3 visual descriptors for each cached window. The workflow uses the MobileNetV3 pose backbone as a frozen feature extractor.

### 3. Train Temporal Classifiers

Use `MobileNetV3 > Train + Validation Eval` to train LSTM or attention classifiers from pose-derived streams and MobileNetV3 visual descriptors. The GUI presents this as cached-feature training and requires the MobileNetV3 feature-cache directory. Use repeated seeds when you need stable aggregate comparisons.

### 4. Held-Out Evaluation

Evaluate trained temporal classifiers on held-out videos. This stage should write frame metrics, bout metrics, per-video summaries, and confusion matrices.

### 5. Static Baseline Training

Train RF/XGBoost static-baseline models. These are best interpreted as negative controls for tabularized, non-sequential reuse. PCA-compressed visual descriptors and flattened pose features do not preserve temporal dynamics the way a sequence model does.

### 6. Static Baseline Evaluation

Evaluate static baselines on held-out videos. Report them as evidence that visual descriptors require an appropriate temporal model, not as a direct substitute for the sequence classifier.

### 7. Suite Summary Tables

Collect frame-level, bout-level, per-video, and confusion-matrix summary tables across MobileNetV3 suite runs.

### 8. Ethograms + Bouts

Use `MobileNetV3 > Ethograms + Bouts` to create model-agnostic ethogram timelines and bout-duration summaries from temporal prediction CSVs. Use `*.smoothed_frames.csv` files for frame-exact timelines; window prediction CSVs are converted by probability averaging or label voting.

## Practical Notes

- Keep the suite root stable across stages.
- Use the same seeds, model IDs, and temporal heads when comparing runs.
- Increase feature-cache batch size only if GPU memory and data loading remain stable.
- Keep resource logging enabled for long runs when possible.

## Outputs

Use `MobileNetV3 > Outputs` to inspect the suite root. Expected artifacts include caches, feature arrays, trained classifiers, evaluation tables, confusion matrices, ethogram summaries, bout summaries, and resource logs.


