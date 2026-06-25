# Graph Reproduction

This package includes the final manuscript figures, the production tables used
by the plotting scripts, and a small wrapper that redraws the figures that do
not require rerunning model training or full-video inference.

## Fast Table-Backed Check

From the package root, list the available redraw tasks:

```powershell
python scripts_by_experiment/00_figure_table_rendering/render_table_backed_figures.py --list
```

Run all table-backed redraw tasks:

```powershell
python scripts_by_experiment/00_figure_table_rendering/render_table_backed_figures.py
```

The regenerated files are written to:

```text
figures_reproduced/production/
tables_reproduced/production/
```

The expected manuscript outputs remain in:

```text
figures/production/
tables/production/
```

Use the expected outputs as visual and tabular references when comparing a
rerun.

## What The Renderer Covers

The table-backed renderer covers figures that can be redrawn from bundled
CSV/JSON files and the shared matplotlib style module. It includes the
table-backed YOLO/SPPF figures, DLC held-out summary, descriptor-provenance
multiplot, and the end-to-end benchmark supplement figures.

It intentionally does not rerun:

- pose-model training,
- temporal model training,
- full-video inference,
- raw-video frame extraction,
- DLC project training,
- checkpoint-dependent feature-cache construction.

Those steps are covered by the command templates and upstream-output contract.

## Full Figure Reproduction

For figures marked `upstream_required` in `GRAPH_REPRODUCTION_STATUS.csv`,
first run the relevant experiment command template in `commands/`. Once the
training/evaluation outputs exist, use the script listed in
`GRAPH_REPRODUCTION_STATUS.csv` or the PowerShell template:

```text
commands/00_render_all_manuscript_figures_from_upstream_outputs_template.ps1
```

The expected upstream folder layout is documented in `UPSTREAM_OUTPUT_PATHS.md`.

## Status Table

Use `GRAPH_REPRODUCTION_STATUS.csv` as the figure-by-figure index. Its
`redraw_status` field means:

- `table_backed`: can be redrawn from files included in this package.
- `upstream_required`: final output is bundled, but redraw requires generated
  upstream run/evaluation folders.
- `archived_output_only`: final output is bundled; full redraw requires source
  material that is not part of this package, such as raw training logs, raw
  videos, or manual GUI screenshots.
