# BehaviorScope-Y GUI Application

This folder contains the public release copy of the BehaviorScope-Y Qt desktop
application plus a launcher. The GUI mirrors the command-line workflow used by
the manuscript analyses while giving users an approachable workspace for their
own behavior datasets:

- create projects, define behavior labels, assign colors/hotkeys, and import
  individual videos or folders;
- load the included tutorial project structure when its MP4 files are present,
  or build a new dataset from user-selected videos;
- review videos in the Qt player, mark behavior spans on timeline lanes, and
  track span review states;
- export approved spans as BENTO `.annot` files and `source_manifest.csv`;
- prepare class-folder clip datasets or full-video NPZ windows;
- precompute frozen MobileNetV3-native or YOLO/SPPF feature caches;
- train temporal sequence classifiers;
- run single-video or folder-level inference with optional review MP4 output.

Launch from this folder:

```bash
python launch_behaviorscope_y_qt.py
```

Use a specific project database:

```bash
python launch_behaviorscope_y_qt.py --project_db path/to/annotation_workspace.sqlite
```

The GUI requires PySide6. Clip export requires `ffmpeg` on `PATH`; the
remaining tabs work without ffmpeg. For licensing-sensitive public tutorials,
the recommended default visual-backbone route is the MobileNetV3-native cache.
YOLO/SPPF remains available for users who supply their own compatible YOLO-pose
checkpoint.

The local release tutorial expects the six MP4 files listed in
`tutorial_data/BehaviorScope-Y_tutorial/source_manifest.csv` to be present
under `tutorial_data/BehaviorScope-Y_tutorial/videos`. If those files are not
present, use `Tutorial > Download/Load BehaviorScope-Y tutorial...` and choose
`Locate Folder` after placing the MP4s locally. A download option is shown only
when `BEHAVIORSCOPE_Y_TUTORIAL_HF_REPO` is configured.

Included source files:

- `behaviorscope_y_qt.py`: Qt application entry point.
- `annotation_app/`: project database, annotation workspace, export, training,
  and inference command builders.
- `tutorial_data/`: lightweight tutorial metadata and annotation examples.
- `package_single_model_y.py`: optional utility for bundling a classifier and
  YOLO-pose checkpoint into a single `.pt` file for inference.
