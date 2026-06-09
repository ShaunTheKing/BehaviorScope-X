# Troubleshooting

## The GUI Opens, But a Stage Fails Immediately

Check the command shown in the log. Most immediate failures are caused by missing paths, the wrong Python environment, or missing optional dependencies.

Recommended checks:

```bash
python -c "import torch, cv2, pandas, sklearn; print('ok')"
python -c "import ultralytics; print('ultralytics ok')"
```

For DeepLabCut stages:

```bash
python -c "import deeplabcut; print('dlc ok')"
```

Run that command in the same environment used to start the GUI or workflow runner.

## A Long Stage Is About To Start

Use `Action=plan` first. Confirm:

- the input manifest,
- the output root,
- the device,
- the checkpoint paths,
- whether `Skip completed stages` is enabled,
- whether existing caches will be reused or rebuilt.

Then switch to `Action=run`.

## GPU Utilization Is Low

Low GPU utilization can come from video decoding, disk I/O, small batch sizes, CPU-side preprocessing, or compression. Practical adjustments:

- move source videos and output caches to an SSD when possible,
- increase batch size gradually while watching VRAM,
- increase data-loader workers if CPU and memory allow,
- reduce progress logging frequency only if logging is extremely frequent,
- avoid rebuilding caches that already validate.

If the GPU is already near 100 percent but VRAM is low, the job may be compute-bound rather than memory-bound.

## Disk Space Is Tight

Cache folders can be large. Before deleting a local cache, confirm that:

- the backup copy exists,
- the manifest files are present,
- a few NPZ files can be opened,
- file counts and total sizes are plausible,
- held-out and train/validation caches are not confused.

Feature caches are often smaller than full-video sequence caches, especially when descriptors are stored as float16.

## DeepLabCut Evaluation Mentions Missing Detector Snapshots

Top-down DLC evaluation requires detector snapshots for detector-based evaluation and video analysis. If the message appears after pose training, train or select a detector checkpoint before running full top-down cache construction.

## Static Baselines Underperform

RF/XGBoost static baselines flatten or compress inputs and do not model temporal dynamics. Treat them as negative controls for non-sequential reuse rather than as direct replacements for LSTM or attention classifiers.

## Outputs Do Not Match Expected Splits

Check the source manifest. Confirm that:

- split names are normalized,
- train/validation/held-out videos are assigned intentionally,
- held-out files are not accidentally included in training caches,
- all referenced `.annot` and `.mp4` files exist.

The annotation export should be the source of truth for split assignment.


