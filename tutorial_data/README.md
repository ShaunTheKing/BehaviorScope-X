# BehaviorScope-Y Tutorial Data

This directory is the default local cache for the GUI tutorial dataset.

The GitHub repository keeps lightweight tutorial metadata in `BehaviorScope-Y_tutorial`, including manifests, class names, provenance, and ground-truth `.annot` files. The MP4 videos are distributed separately because regular GitHub repositories block files larger than 100 MiB.

## Automatic Download

In the GUI, use:

`Tutorial > Download/Load BehaviorScope-Y tutorial...`

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
    videos/
      train/
      val/
      test/
```

Then launch the GUI and choose `Tutorial > Download/Load BehaviorScope-Y tutorial...`. If the folder is not detected automatically, choose `Locate Folder` and select `BehaviorScope-Y_tutorial`.

## License

The tutorial media and annotations are derived from MARS data and should be distributed under the original MARS-compatible `CC BY-NC 4.0` terms with attribution. The tutorial data is separate from the BehaviorScope-Y software license and is not AGPL-licensed.


