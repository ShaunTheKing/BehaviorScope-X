# Path and Reproduction Policy

The command plans in older working folders preserve the paths used on the
author machine. Those paths are useful provenance, but they are not the paths a
reviewer should run.

For this package, use the templates in `commands/`. They replace local machine
roots with named placeholders that can be filled in on another computer.

## Placeholder Roots

- `<PACKAGE_ROOT>` or `<MANUSCRIPT_PACKAGE_ROOT>`: the unpacked
  `analysis_artifacts` folder.
- `<RUN_ROOT>`: a working folder for generated caches, checkpoints, training
  runs, evaluation outputs, and reproduced figures.
- `<MARS_DATA_ROOT>`: the MARS behavior dataset folder itself.
- `<DATA_ROOT>`: a parent folder for optional external data trees.
- `<DLC_WORK_ROOT>`: the DLC SuperAnimal project/work folder.
- `<BEHAVIORSCOPE_X_GUI_ROOT>`: a BehaviorScope-X GUI checkout, if using the
  GUI-assisted material.
- `<PYTHON>`: the Python executable or environment launcher used to run the
  commands.

## Required MARS Layout

Set `<MARS_DATA_ROOT>` to the folder that directly contains the MARS split
folders:

```text
<MARS_DATA_ROOT>/
  train/
  validation/
  test_1/
  test_2/
```

Do not set `<MARS_DATA_ROOT>` to a parent folder that then contains another
`MARS-data` or `MARS_data` folder. The commands pass `<MARS_DATA_ROOT>`
directly to script flags named `--mars_root`.

## Working Folder Layout

Use `<RUN_ROOT>` for generated outputs rather than writing into the raw data
folder:

```text
<RUN_ROOT>/
  outputs/
  Fly_YOLO_Pose_Model/
```

The exact upstream output folders expected by figure scripts are listed in
`UPSTREAM_OUTPUT_PATHS.md`.

## Generate Local Commands

From the package root:

```powershell
Copy-Item .\commands\reproducibility_paths.template.yml .\commands\reproducibility_paths.yml
powershell -ExecutionPolicy Bypass -File .\commands\00_configure_paths.ps1 `
  -ConfigYaml .\commands\reproducibility_paths.yml
```

The generated command files are written to:

```text
commands/localized/
```

The configurator also writes:

```text
commands/localized/RUNNABILITY_REPORT.md
```

Read that report before running commands. It marks files as `runnable`,
`not runnable`, or `skipped`.

Blank paths intentionally leave placeholders unresolved. That is useful when a
reviewer wants to inspect a command family before supplying every external
path. Disabled command families are skipped and can be enabled by changing the
corresponding `include_*` value to `true` in `reproducibility_paths.yml`.

## Checkpoint Scope

Model checkpoint binaries are not bundled because redistribution has unresolved
license interactions. The package includes scripts, configs, command templates,
and verification artifacts so that reviewers can inspect the exact analysis
logic and rerun it when the required data and checkpoints are available.
