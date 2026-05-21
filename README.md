# BehaviorScope-Y

BehaviorScope-Y is a desktop application for building animal behavior classifiers from video. It combines YOLO-pose detection with a temporal behavior model so researchers can annotate videos, train a classifier, and run inference from a single GUI-driven workflow.

The application supports the full path from raw videos to reviewable predictions:

- import and annotate videos,
- assign videos to train, validation, and held-out test splits,
- prepare full-video training windows,
- train a temporal behavior classifier,
- bundle the classifier and YOLO-pose model into one `.pt` file,
- run inference on one video or a folder of videos,
- export prediction CSVs and annotated review MP4s.

Command-line scripts are included for automation and reproducible batch runs, but the recommended starting point is the Qt GUI.

## Quick Start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python behaviorscope_y_qt.py
```

For GPU training, install the CUDA build of PyTorch before installing the remaining dependencies:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`ffmpeg` is recommended for video conversion and some export workflows.

## Requirements

- Python 3.10 or newer is recommended.
- A YOLO-pose `.pt` checkpoint for the target animal/video setup.
- Videos readable by OpenCV/Qt, such as `.mp4`.
- An NVIDIA GPU is recommended for training and faster inference.

## GUI Workflow

Launch the application:

```bash
python behaviorscope_y_qt.py
```

The main tabs are ordered by workflow:

- `Annotate + Clip`: import videos, annotate behavior spans, review labels, and export full-video annotation manifests.
- `Prepare Full Video`: convert full-video annotations into training windows.
- `Feature Cache`: precompute YOLO visual features for faster training.
- `Train`: train the temporal behavior classifier and export a bundled model.
- `Inference`: run prediction on a single video and optionally save an annotated review MP4.
- `Batch`: run prediction on a folder of videos.

After a step completes successfully, the GUI fills the next tab's paths where possible.

## GUI Screenshots

The screenshots below show the main GUI workflow with annotated callouts for new users. The image files are stored in `docs/screenshots/annotated/` so GitHub can render them directly from the repository.

### Annotation Workspace

![Annotated screenshot of the annotation workspace](docs/screenshots/annotated/01_annotate_tutorial_annotated.png)

### Prepare Full-Video Dataset

![Annotated screenshot of the Prepare Full Video tab](docs/screenshots/annotated/02_prepare_full_video_annotated.png)

### Feature Cache

![Annotated screenshot of the Feature Cache tab](docs/screenshots/annotated/03_feature_cache_annotated.png)

### Train

![Annotated screenshot of the Train tab](docs/screenshots/annotated/04_train_annotated.png)

### Inference

![Annotated screenshot of the Inference tab](docs/screenshots/annotated/05_inference_annotated.png)

### Batch Inference

![Annotated screenshot of the Batch tab](docs/screenshots/annotated/06_batch_annotated.png)

## Tutorial Dataset

BehaviorScope-Y includes a guided tutorial based on a small MARS mouse-behavior subset. In the GUI, choose:

```text
Tutorial > Download/Load BehaviorScope-Y tutorial...
```

The tutorial loader downloads or locates the tutorial data, imports the videos, adds behavior labels, assigns train/validation/test splits, converts `.annot` files into timeline annotations, and fills the downstream workflow paths.

If the automatic download fails, manual download instructions are available in [`tutorial_data/README.md`](tutorial_data/README.md).

## Annotation And Splits

The annotation workspace supports:

- importing individual videos or folders,
- creating and editing behavior spans on a timeline,
- marking spans as draft, ready, approved, or rejected,
- locking reviewed spans,
- assigning videos to `train`, `val`, `test`, or `exclude`,
- exporting full-video annotations for training.

For full-video training, use:

```text
Project > Export full-video annotations...
```

Legacy clip extraction remains available from the Project menu, but full-video annotation export is the recommended training path.

## Model Bundling

BehaviorScope-Y uses two model components during training:

- a YOLO-pose model for detection, keypoints, and visual feature extraction,
- a temporal classifier for behavior prediction.

Training can export a bundled `.pt` file containing both components. Existing classifier and YOLO-pose checkpoints can also be bundled from the GUI:

```text
Model Tools > Bundle existing classifier + YOLO...
```

The bundled model is the preferred inference format because users do not need to manage separate classifier, config, and YOLO paths.

## Inference Outputs

Inference can produce:

- behavior prediction CSV files,
- smoothed per-frame outputs,
- runtime metrics,
- optional pose exports,
- annotated review MP4s.

Review MP4s make it easier to inspect predictions visually and share model outputs with collaborators.

## Command-Line Use

The GUI is the recommended entry point. The CLI remains useful for scripted runs, remote machines, and reproducible experiments.

Prepare full-video training windows:

```bash
python prepare_full_video_npz.py ^
  --source_manifest_csv path\to\source_manifest.csv ^
  --class_names_file path\to\class_names.txt ^
  --yolo_weights path\to\yolo_pose_best.pt ^
  --output_root runs\my_dataset_npz ^
  --validate_manifest
```

Train and export a bundled model:

```bash
python train_y.py ^
  --manifest_path runs\my_dataset_npz\sequence_manifest.json ^
  --yolo_weights path\to\yolo_pose_best.pt ^
  --auto_feature_cache ^
  --sequence_model lstm ^
  --project runs ^
  --name my_behavior_model ^
  --export_single_model
```

Run inference:

```bash
python infer_y.py ^
  --model_path runs\my_behavior_model\behaviorscope_y_single_model.pt ^
  --source path\to\video.mp4 ^
  --output runs\my_behavior_model\inference_outputs\video.behavior.csv ^
  --output_video runs\my_behavior_model\inference_outputs\video.annotated.mp4
```

## Repository Layout

- `behaviorscope_y_qt.py`: Qt GUI launcher.
- `annotation_app/`: GUI application code.
- `prepare_full_video_npz.py`: full-video dataset preparation.
- `precompute_visual_features_y.py`: YOLO feature-cache generation.
- `train_y.py`: behavior classifier training.
- `infer_y.py`: inference and review-video export.
- `package_single_model_y.py`: single-file model bundling.
- `docs/screenshots/`: reusable raw and annotated GUI screenshots for documentation.
- `tutorial_data/`: tutorial metadata and local tutorial cache.

## Data License

The tutorial media and annotations are derived from MARS data and are distributed under `CC BY-NC 4.0` terms with attribution. The tutorial dataset is separate from the software license.

## License

BehaviorScope-Y source code is licensed under the GNU Affero General Public License v3.0. See [`LICENSE`](LICENSE) for the full license text.





