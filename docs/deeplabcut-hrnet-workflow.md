# DeepLabCut-HRNet Workflow

The DeepLabCut-HRNet workflow tests whether the BehaviorScope pose-plus-vision principle transfers to a widely used external pose-estimation framework. It uses DLC top-down detections and HRNet-W32 pose features to build sequence caches, visual-feature caches, temporal classifiers, and held-out evaluations.

Run these stages from an environment where DeepLabCut imports successfully.

## Inputs

You need:

- a DLC project with `config.yaml`,
- a DLC SuperAnimal-compatible HRNet-W32 pose model,
- a detector checkpoint or training configuration,
- train/validation source videos and annotations,
- held-out source videos and annotations,
- a DLC workflow config JSON,
- enough disk space for full-video sequence caches.

## Recommended Stage Order

Use `Action=plan` first for every stage. Switch to `Action=run` only after confirming paths, device, and output roots.

### 1. Pose + Detector Fine-Tuning

Fine-tune the DLC SuperAnimal pose model and the detector. The workflow should select checkpoints based on validation metrics rather than relying on the last epoch. In the MARS DLC run, the pose model selected the best validation epoch, and the detector selected the best detector validation epoch.

### 2. Full-Video Train/Val Cache

Run the DLC top-down detector and pose model over train/validation videos. This creates full-video sliding-window sequence NPZs from DLC-derived detections, poses, and crops.

### 3. Train/Val HRNet Feature Cache

Extract frozen HRNet-W32 visual descriptors from DLC top-down crops. This is the visual stream used by the downstream temporal classifier.

### 4. Classifier Train + Validation

Train the temporal behavior classifier using DLC pose-derived geometry streams plus HRNet visual descriptors. Validation metrics are used to select classifier checkpoints and decoding parameters.

### 5. Held-Out Manifest

Prepare the held-out source manifest. Confirm that the held-out videos and `.annot` files match the intended evaluation set.

### 6. Held-Out Full-Video Cache

Build DLC top-down sequence windows for held-out videos.

### 7. Held-Out HRNet Feature Cache

Extract HRNet visual descriptors for the held-out windows.

### 8. Held-Out Evaluation

Evaluate the trained classifier on held-out videos. This stage should write frame metrics, bout metrics, per-video metrics, confusion matrices, ethogram summaries, and behavior/keypoint exports when configured.

### 9. Ethograms + Bouts

Use `DeepLabCut-HRNet > Ethograms + Bouts` to create model-agnostic ethogram timelines and bout-duration summaries from temporal prediction CSVs. Use `*.smoothed_frames.csv` files for frame-exact timelines; window prediction CSVs are converted by probability averaging or label voting.

## Practical Notes

- DLC cache construction is usually slower than YOLO-pose because it runs the detector and pose model through DLC's top-down workflow.
- Keep pose and detector checkpoints separate. Detector training should not overwrite pose checkpoints.
- Preserve the DLC workflow config and completion markers with the output directory.
- If disk space is limited, back up caches before deleting any local copy.

## Outputs

Use `DeepLabCut-HRNet > Outputs` to inspect DLC sequence caches, HRNet feature caches, classifier checkpoints, held-out evaluation tables, confusion matrices, ethograms, bout summaries, and provenance files.


