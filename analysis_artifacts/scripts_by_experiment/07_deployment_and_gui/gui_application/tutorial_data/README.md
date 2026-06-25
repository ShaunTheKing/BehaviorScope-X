# BehaviorScope-Y Tutorial Data

This directory is the default local cache for the GUI tutorial dataset.

The release tutorial is complete when `BehaviorScope-Y_tutorial/videos` contains the six MP4s listed in `source_manifest.csv`. For public repository distribution, the lightweight metadata can be kept without MP4s because regular GitHub repositories block files larger than 100 MiB.

## Automatic Download

In the GUI, use:

`Tutorial > Download/Load BehaviorScope-Y tutorial...`

If the tutorial videos are missing locally, place the six MP4s listed in `source_manifest.csv` under the matching `videos/train`, `videos/val`, and `videos/test` folders. The GUI can also download a configured external tutorial dataset, but no public Hugging Face tutorial-video package is assumed by default.

For licensing-sensitive public tutorials, use the MobileNetV3-native pose-backbone feature cache as the default visual-backbone route. The YOLO/SPPF path remains available for users who supply their own compatible YOLO-pose checkpoint under terms suitable for their use case.

Advanced users can point the GUI to a hosted mirror by setting `BEHAVIORSCOPE_Y_TUTORIAL_HF_REPO` before launching the application. When this variable is not set, the GUI offers local folder selection rather than a download path.

## Manual Download

Place or extract the tutorial videos so the final structure is:

```text
tutorial_data/
  BehaviorScope-Y_tutorial/
    source_manifest.csv
    class_names.txt
    provenance.json
    annotations/
    videos/
      train/
      val/
      test/
```

Then launch the GUI and choose `Tutorial > Download/Load BehaviorScope-Y tutorial...`. If the folder is not detected automatically, choose `Locate Folder` and select `BehaviorScope-Y_tutorial`.

## License

The tutorial media and annotations are derived from MARS data and should be distributed under the original MARS-compatible `CC BY-NC 4.0` terms with attribution. The tutorial data is separate from the BehaviorScope-Y software license and is not AGPL-licensed.

No YOLO-pose weights are bundled with the tutorial data. Users who choose the YOLO/SPPF workflow must provide their own compatible checkpoint.
