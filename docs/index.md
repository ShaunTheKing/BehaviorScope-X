# BehaviorScope-X Documentation

BehaviorScope-X is a desktop application and script bundle for building animal behavior classifiers from video. The central workflow is simple: annotate behavior bouts, choose a supported pose-model workflow, build full-video sequence caches, extract pose-derived and pose-backbone visual features, train temporal classifiers, and inspect frame-level and bout-level outputs.

The GUI is the shared validated workflow surface. It follows a pose-model-flexible design: YOLO-pose, MobileNetV3, and DeepLabCut-HRNet each provide an independent route to compatible sequence caches, feature caches, classifier training, evaluation, ethogram/bout summaries, and output inspection.

The GUI is organized around model families:

- **Annotate + Clip** imports videos, assigns splits, labels bouts, tracks quality-control state, and exports full-video annotation manifests.
- **YOLO-pose** supports full-video cache construction, feature-cache generation, classifier training, single-video inference, folder-level batch inference, ethogram/bout summaries, and output inspection.
- **MobileNetV3** supports native full-video cache construction, MobileNetV3 feature-cache generation, cached-feature classifier training, staged held-out evaluation, ethogram/bout summaries, and output inspection.
- **DeepLabCut-HRNet** exposes staged workflows for DLC SuperAnimal fine-tuning, top-down sequence-cache generation, HRNet feature extraction, classifier training, held-out evaluation, ethogram/bout summaries, and output inspection.

Use the GUI when you want a guided desktop workflow. Use the analysis runners when you want scripted, reproducible long-running jobs.

## Recommended Reading Order

1. [Installation](installation.md)
2. [GUI Overview](gui-overview.md)
3. [Annotation Workflow](annotation-workflow.md)
4. One model-family workflow:
   - [YOLO-pose Workflow](yolo-pose-workflow.md)
   - [MobileNetV3 Workflow](mobilenetv3-workflow.md)
   - [DeepLabCut-HRNet Workflow](deeplabcut-hrnet-workflow.md)
5. [Feature Taps](feature-taps.md)
6. [Outputs and Provenance](outputs-and-provenance.md)
7. [Troubleshooting](troubleshooting.md)

## Start the GUI

```bash
python behaviorscope_x_qt.py
```

## Start the Documentation Server

```bash
mkdocs serve
```

Then open the local URL printed by MkDocs.



