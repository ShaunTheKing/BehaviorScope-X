# Outputs and Provenance

BehaviorScope-X workflows produce several artifact types. Keeping them organized makes it easier to reproduce training, audit held-out evaluation, and share a compact result package.

## Sequence Caches

Sequence caches are usually NPZ folders with a `sequence_manifest.json`. They store sliding windows derived from full videos. Depending on the workflow, a window may include:

- frame indices,
- class label,
- pose coordinates,
- pose confidence,
- crop metadata,
- pose-derived geometry features,
- cached crops or crop references,
- source video and split metadata.

Train/validation and held-out caches should be kept separate.

## Visual Feature Caches

Feature caches store frozen visual descriptors extracted from a pose backbone. They make classifier training faster and keep the visual stream fixed across repeated temporal-model runs.

Expected metadata includes:

- source sequence manifest,
- pose checkpoint or model source,
- tap description,
- descriptor dimension,
- dtype,
- split names,
- number of samples.

## Classifier Outputs

Classifier training usually writes:

- checkpoint files,
- `config.json`,
- `training_log.csv`,
- validation metrics,
- confusion matrices,
- completion marker,
- optional bundled model.

Keep the configuration file with the checkpoint. It records the streams, model dimensions, decoding settings, class names, and feature-cache paths used during training.

## Held-Out Evaluation Outputs

Held-out evaluation should write:

- frame accuracy and macro F1,
- bout F1 at the configured IoU thresholds,
- per-class precision, recall, and F1,
- row-normalized and count confusion matrices,
- per-video metrics,
- ethogram-level summaries,
- behavior CSVs,
- optional keypoint CSVs,
- optional review videos.

Frame metrics and bout metrics answer different questions. Frame metrics evaluate per-frame classification. Bout metrics evaluate whether behavior episodes were detected with reasonable temporal overlap.

## Provenance Checklist

Before archiving a run, confirm that the folder contains:

- command plan or command log,
- exact config files,
- source manifests,
- sequence-cache manifest,
- feature-cache manifest,
- training log,
- validation and held-out metrics,
- confusion matrices,
- summary tables,
- environment notes,
- model checkpoint selection criteria.

For large cache folders, a separate checksum manifest is useful before upload or transfer.



