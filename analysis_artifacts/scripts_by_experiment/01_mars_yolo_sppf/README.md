# MARS YOLO/SPPF Ablation Runner

Use this folder for the final MARS YOLO/SPPF stream-ablation analysis and
YOLO/SPPF RF/XGBoost baselines.

The runner excludes TCN, MobileNetV3, and Fly-v-Fly by construction. It keeps
the final manuscript comparison: LSTM and attention temporal sequence
classifiers across the MARS stream/capacity/seed matrix, plus YOLO/SPPF
classical baselines with train-fit PCA compression.

Plan first:

```bash
python run_mars_yolo_sppf_ablation.py plan
```

To include cache construction in the command plan, add `--rebuild_npz`:

```bash
python run_mars_yolo_sppf_ablation.py plan --rebuild_npz
```

This prepends the final MARS training NPZ build, training YOLO/SPPF feature
cache, held-out NPZ build, and held-out YOLO/SPPF feature cache. The held-out
NPZ build uses MARS `test_1` and `test_2` while excluding the training and
validation splits; the original MARS split names are retained in the manifest
for held-out evaluation filtering.

Run all selected stages:

```bash
python run_mars_yolo_sppf_ablation.py run-all --skip_completed --keep_going
```

Use `--stage` to run a subset such as `train_neural`, `eval_neural`,
`train_classical`, `eval_classical`, or `build_npz`.
