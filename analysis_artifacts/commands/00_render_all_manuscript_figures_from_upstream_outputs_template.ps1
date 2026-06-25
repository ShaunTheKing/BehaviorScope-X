# Render manuscript figures after upstream runs/evaluations have been generated.
#
# Generate a localized copy with 00_configure_paths.ps1 before running.
$PACKAGE_ROOT = "<PACKAGE_ROOT>"
$RUN_ROOT = "<RUN_ROOT>"
$DLC_WORK_ROOT = "<DLC_WORK_ROOT>"
$PYTHON = "<PYTHON>"

$PROD = Join-Path $PACKAGE_ROOT "scripts_by_experiment\00_figure_table_rendering\production_analysis"
$FIG_ROOT = Join-Path $PACKAGE_ROOT "figures_reproduced\production"
$TABLE_ROOT = Join-Path $PACKAGE_ROOT "tables_reproduced\production"
$EXPECTED_TABLE_ROOT = Join-Path $PACKAGE_ROOT "tables\production"

$MARS_SUITE = Join-Path $RUN_ROOT "outputs\controlled_comparison_runs\run_20260528_205727"
$MARS_MP4_CACHE = Join-Path $RUN_ROOT "outputs\source_mp4_cache\mars"
$FLY_EVAL = Join-Path $RUN_ROOT "outputs\controlled_comparison_runs\fly_adaptation_attention256_cw12_seed42\fly\eval"
$FLY_POSE_VAL = Join-Path $RUN_ROOT "Fly_YOLO_Pose_Model\weights\Fly_model_val.txt"
$DLC_EVAL = Join-Path $DLC_WORK_ROOT "behaviorscope_outputs\dlc_topdown_heldout_eval\mars_dlc_topdown_hrnet_attention256_seed42"

New-Item -ItemType Directory -Force -Path $FIG_ROOT | Out-Null
New-Item -ItemType Directory -Force -Path $TABLE_ROOT | Out-Null

# Table-backed figures that can be redrawn from bundled production tables.
& $PYTHON (Join-Path $PACKAGE_ROOT "scripts_by_experiment\00_figure_table_rendering\render_table_backed_figures.py")

# MARS YOLO/SPPF figures and tables rebuilt from the controlled-comparison run.
& $PYTHON (Join-Path $PROD "analyze_yolo_sppf_lstm_attention.py") `
  --suite_root $MARS_SUITE `
  --table_dir (Join-Path $TABLE_ROOT "yolo_sppf_main") `
  --figure_dir (Join-Path $FIG_ROOT "yolo_sppf_main")

& $PYTHON (Join-Path $PROD "plot_yolo_classical_baselines.py") `
  --source_dir (Join-Path $MARS_SUITE "manuscript_figures\interim_qc\classical_rf_xgb") `
  --suite_root $MARS_SUITE `
  --figure_dir (Join-Path $FIG_ROOT "classical_baselines") `
  --table_dir (Join-Path $TABLE_ROOT "classical_baselines")

& $PYTHON (Join-Path $PROD "plot_yolo_ethogram_bout_examples.py") `
  --table_dir (Join-Path $TABLE_ROOT "yolo_sppf_main") `
  --figure_dir (Join-Path $FIG_ROOT "yolo_sppf_main") `
  --mp4_cache $MARS_MP4_CACHE

& $PYTHON (Join-Path $PROD "plot_mars_model_type_ethograms.py") `
  --suite_root $MARS_SUITE `
  --figure_dir (Join-Path $FIG_ROOT "supplement") `
  --table_dir (Join-Path $TABLE_ROOT "supplement")

& $PYTHON (Join-Path $PROD "plot_yolo_training_throughput.py") `
  --source_dir (Join-Path $MARS_SUITE "manuscript_figures\interim_qc\throughput") `
  --figure_dir (Join-Path $FIG_ROOT "supplement") `
  --table_dir (Join-Path $TABLE_ROOT "supplement")

# MobileNetV3 cross-backbone figures rebuilt from the controlled-comparison run.
& $PYTHON (Join-Path $PROD "plot_mobilenet_portability.py") `
  --suite_root $MARS_SUITE `
  --yolo_table_dir (Join-Path $TABLE_ROOT "yolo_sppf_main") `
  --figure_dir (Join-Path $FIG_ROOT "mobilenet_portability") `
  --table_dir (Join-Path $TABLE_ROOT "mobilenet_portability")

& $PYTHON (Join-Path $PROD "plot_mobilenet_ethogram_bout_detail.py") `
  --suite_root $MARS_SUITE `
  --figure_dir (Join-Path $FIG_ROOT "mobilenet_portability") `
  --table_dir (Join-Path $TABLE_ROOT "mobilenet_portability")

# Fly-v-Fly focused attention-256 figures rebuilt from the focused eval folder.
& $PYTHON (Join-Path $PROD "plot_fly_v_fly_focused_results.py") `
  --eval_dir $FLY_EVAL `
  --table_dir (Join-Path $TABLE_ROOT "fly_v_fly") `
  --fig_dir (Join-Path $FIG_ROOT "fly_v_fly") `
  --supp_fig_dir (Join-Path $FIG_ROOT "supplement") `
  --pose_val_txt $FLY_POSE_VAL

& $PYTHON (Join-Path $PROD "plot_fly_v_fly_ethogram_examples.py") `
  --eval_dir $FLY_EVAL `
  --table_dir (Join-Path $TABLE_ROOT "fly_v_fly") `
  --figure_dir (Join-Path $FIG_ROOT "fly_v_fly")

# DLC-HRNet held-out analysis and summary figures rebuilt from the DLC eval folder.
& $PYTHON (Join-Path $PACKAGE_ROOT "scripts_by_experiment\04_dlc_hrnet_topdown\analyze_dlc_heldout_eval.py") `
  --eval_dir $DLC_EVAL `
  --analysis_dir (Join-Path $TABLE_ROOT "dlc_superanimal_topdown_analysis") `
  --manuscript_root $PACKAGE_ROOT `
  --no_manuscript_copy

& $PYTHON (Join-Path $PACKAGE_ROOT "scripts_by_experiment\04_dlc_hrnet_topdown\make_dlc_heldout_summary_figure.py") `
  --table_dir (Join-Path $TABLE_ROOT "dlc_superanimal_topdown_analysis\tables") `
  --figure_dir (Join-Path $FIG_ROOT "dlc_superanimal_topdown")

# Descriptor-provenance manuscript figure from the packaged descriptor tables.
& $PYTHON (Join-Path $PACKAGE_ROOT "scripts_by_experiment\05_descriptor_provenance\make_descriptor_provenance_figure.py") `
  --table_dir (Join-Path $EXPECTED_TABLE_ROOT "descriptor_provenance") `
  --figure_dir (Join-Path $FIG_ROOT "descriptor_provenance")
