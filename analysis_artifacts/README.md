# BehaviorScope-X Analysis Artifacts

This folder contains the manuscript-facing analysis artifacts for the
BehaviorScope-X manuscript and supplement. It includes the scripts, frozen
configuration files, command templates, verification tables, and final rendered
figures used for the manuscript-facing analyses.

The artifacts support two levels of checking:

1. Redraw figures that can be regenerated from bundled CSV/JSON tables.
2. Rebuild upstream training, cache, evaluation, and figure products after the
   reviewer supplies the raw datasets and any required checkpoint files.

Model checkpoint binaries are not bundled. Some full reruns also require large
raw video/data folders. The command templates make those external paths
explicit instead of preserving author-machine paths.

## What This Folder Provides

This folder provides the manuscript-facing analysis code and the small
verification artifacts needed to inspect and redraw many reported results. It
does not include the large raw video datasets or the trained upstream pose
checkpoint binaries.

Without downloading any external data, a reviewer can:

- inspect the exact scripts, command templates, and frozen configuration files;
- inspect the reported experiment matrix and provenance maps;
- compare manuscript figures against the bundled production tables;
- redraw the table-backed figures into `figures_reproduced/production/`;
- check the bundled CSV/JSON/log verification artifacts.

With the raw datasets and required upstream model assets, a reviewer can also
use the command templates to rebuild the full analysis path:

- build split/window manifests and NPZ caches;
- construct YOLO/SPPF, MobileNetV3, DLC-HRNet, or Fly feature caches;
- train the temporal behavior classifiers;
- evaluate held-out frame, bout, and per-video behavior metrics;
- regenerate manuscript-facing summary tables and figures.

The required external inputs depend on the route being reproduced:

- **MARS YOLO/SPPF:** MARS videos/annotations plus the YOLO-pose checkpoint used
  for pose tracking and SPPF feature extraction.
- **MARS MobileNetV3:** MARS videos/annotations plus the MobileNetV3
  feature-cache route and its dependencies.
- **DLC-HRNet:** MARS videos/annotations plus a DLC SuperAnimal working setup;
  the provided commands cover the DLC fine-tuning/cache/evaluation path when
  the required DLC data and checkpoints are available.
- **Fly-v-Fly:** Fly-v-Fly videos/annotations plus the Fly pose model or the
  generated Fly NPZ/cache outputs.

In short: this archive contains the analysis code, configuration, provenance,
verification data, and expected outputs. Large datasets and checkpoint binaries
are supplied separately by the reviewer or by the corresponding data/model
release location.


## GitHub Packaging Note

This GitHub copy keeps the same analysis content as the local manuscript archive, with one storage-oriented change: six large long-form CSV tables are stored as `.csv.gz` files. These are ordinary gzip-compressed CSV files. The included Python scripts read them with `pandas.read_csv`, which handles `.gz` inputs directly.

The compression is only for repository hosting. It avoids GitHub's regular Git file-size limit while preserving the table contents for reviewers who want to inspect or rerun the table-backed analyses.

## Quick Start

From the `analysis_artifacts` folder:

```powershell
python -m pip install -r requirements.txt
python scripts_by_experiment/00_figure_table_rendering/render_table_backed_figures.py --list
python scripts_by_experiment/00_figure_table_rendering/render_table_backed_figures.py
```

The renderer writes regenerated outputs to:

```text
figures_reproduced/production/
tables_reproduced/production/
```

Compare those files with the expected manuscript outputs in:

```text
figures/production/
tables/production/
```

The table-backed renderer does not rerun model training or full-video
inference. It is intended as a fast check that the packaged table inputs and
plotting scripts reproduce the table-backed manuscript figures.

## Full Rerun Workflow

For a full rerun, first decide which experiment family you want to reproduce:
YOLO/SPPF, MobileNetV3, DLC-HRNet, Fly-v-Fly, or final figure rendering from
previously generated upstream outputs.

Copy the path template, fill in local paths, and generate localized commands:

```powershell
Copy-Item .\commands\reproducibility_paths.template.yml .\commands\reproducibility_paths.yml
powershell -ExecutionPolicy Bypass -File .\commands\00_configure_paths.ps1 `
  -ConfigYaml .\commands\reproducibility_paths.yml
```

The generated commands are written to:

```text
commands/localized/
```

Open `commands/localized/RUNNABILITY_REPORT.md` before running anything. It
lists which generated command files are runnable and which still contain
unresolved placeholders.

Use the `include_*` values in `commands/reproducibility_paths.yml` to enable
only the experiment families you intend to run. A blank path in an enabled
family is left as a placeholder and is reported as not runnable.

## Expected External Folder Layout

The MARS data root should point to the dataset folder itself, not to its
parent. The path should contain these split folders:

```text
<MARS_DATA_ROOT>/
  train/
  validation/
  test_1/
  test_2/
```

Use a separate working root for generated caches, training runs, evaluation
runs, and reproduced figures:

```text
<RUN_ROOT>/
  outputs/
  Fly_YOLO_Pose_Model/
```

For DLC-HRNet reruns, use a DLC working root that contains or can generate the
DLC project, detector/pose checkpoints, top-down NPZ caches, HRNet feature
caches, and held-out evaluation folder:

```text
<DLC_WORK_ROOT>/
  dlc-models-pytorch/
  behaviorscope_outputs/
```

Additional expected upstream output folders are listed in
`UPSTREAM_OUTPUT_PATHS.md`.

## Where To Look

- `PROVENANCE_MAP.md`: which experiment family supports each major manuscript result.
- `REPORTED_EXPERIMENTS.csv`: the reported experiment matrix and config locations.
- `MANUSCRIPT_REPRODUCTION_INPUTS.md`: scripts and inputs for rebuilding split loading, caches, model training, evaluation, and verification outputs.
- `GRAPH_REPRODUCTION.md`: how to redraw figures from bundled tables or upstream outputs.
- `GRAPH_REPRODUCTION_STATUS.csv`: per-figure reproduction status and required inputs.
- `MANUSCRIPT_FIGURE_TABLE_MAP.csv`: manuscript figure/table labels mapped to package files and experiment families.
- `PATH_AND_REPRODUCTION_POLICY.md`: path placeholders and rules for localizing commands.

## Package Contents

- `commands/`: command templates and the path-localization script.
- `scripts_by_experiment/`: scripts, frozen configs, and per-experiment notes.
- `verification_files/by_experiment/`: CSV/JSON/log artifacts used to verify manuscript results.
- `figures/production/`: final rendered manuscript-facing figures.
- `tables/production/`: production table inputs used by figure/table scripts.
- `metadata/v4_package_manifest.csv`: file manifest and SHA-256 hashes.
- `requirements.txt`: baseline Python packages for command-line reproduction.
- `requirements-optional-dlc.txt`: extra packages for the DLC-HRNet path.
- `requirements-optional-gui.txt`: extra packages for GUI-assisted review.
