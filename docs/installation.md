# Installation

## Python Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install a CUDA-enabled PyTorch build before installing the rest of the requirements:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## System Dependencies

Install `ffmpeg` if you plan to extract clips, convert videos, or write annotated review videos. BehaviorScope-X can read common MP4 files through OpenCV/Qt, but `ffmpeg` makes video handling more reliable.

## Model-Specific Dependencies

### YOLO-pose

The YOLO-pose workflow requires:

- `ultralytics`
- a YOLO-pose `.pt` checkpoint trained for your animal, view, and keypoint layout
- an NVIDIA GPU for practical training and faster cache building

### MobileNetV3

The MobileNetV3 workflow requires:

- the MobileNetV3 pose-backbone checkpoint expected by the workflow runner
- the shared controlled-comparison helper scripts included under `analysis_workflows/shared_analysis_code`
- `scikit-learn` and `xgboost` for static-baseline training

### DeepLabCut-HRNet

The DeepLabCut-HRNet workflow requires:

- a working DeepLabCut 3 environment
- a DLC-format project with `config.yaml`
- DLC SuperAnimal-compatible pose and detector checkpoints or a workflow configuration that can locate them
- enough disk space for full-video NPZ caches and feature caches

Run DeepLabCut stages from the environment where `deeplabcut` imports successfully. The GUI can launch the Python runner, but the active interpreter must match the dependencies for the workflow being run.

## Documentation Build

MkDocs is included in `requirements.txt`.

```bash
mkdocs serve
mkdocs build
```

Use `mkdocs build --strict` before packaging a release.



