# BehaviorScope-X Tutorial Data

This directory is the default local cache for the GUI tutorial dataset.

The GitHub repository keeps lightweight tutorial metadata in `BehaviorScope-Y_tutorial`, including manifests, class names, provenance, reference `.annot` files, and optional tutorial model manifests. The folder name is retained for compatibility with existing tutorial manifests, while the GUI presents the workflow as BehaviorScope-X. The MP4 videos and large tutorial checkpoints may be distributed separately because regular GitHub repositories block files larger than 100 MiB.

## Automatic Download

In the GUI, use:

`Tutorial > Download/Load BehaviorScope-X tutorial...`

If the tutorial videos are missing locally, the GUI can download the Hugging Face dataset into this `tutorial_data` directory and then import the videos, splits, labels, and timeline annotations.

The public tutorial dataset is hosted on Hugging Face as:

`farhanaugustine/BehaviorScope-Y_tutorial`

Advanced users can point the GUI to a mirror by setting `BEHAVIORSCOPE_Y_TUTORIAL_HF_REPO` before launching the application.

## Manual Download

If you download the tutorial manually, place or extract it so the final structure is:

```text
tutorial_data/
  BehaviorScope-Y_tutorial/
    source_manifest.csv
    class_names.txt
    provenance.json
    annotations/
    tutorial_models/
      full_pose_mobilenetv3/
      full_attn_classifier_mobilenetv3/
    videos/
      train/
      val/
      test/
```

Then launch the GUI and choose `Tutorial > Download/Load BehaviorScope-X tutorial...`. If the folder is not detected automatically, choose `Locate Folder` and select `BehaviorScope-Y_tutorial`.

You can check whether the local tutorial cache is complete with:

```bash
python tutorial_data/validate_tutorial_data.py
```

The validator checks tutorial videos, reference annotations, MobileNetV3 tutorial checkpoints, classifier configuration portability, and the tutorial model manifest.

## License

The tutorial media and annotations are derived from MARS data and should be distributed under the original MARS-compatible `CC BY-NC 4.0` terms with attribution. The tutorial data is separate from the BehaviorScope-X software license and is not AGPL-licensed.



