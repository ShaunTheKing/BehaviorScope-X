# MobileNetV3 Backbone Runner

Use this folder for the final MobileNetV3-native backbone generalization
analysis.

The runner keeps MobileNetV3 feature-cache construction, MobileNetV3
LSTM/attention temporal sequence classifiers, and MobileNetV3 RF/XGBoost
classical baselines. It excludes YOLO/SPPF temporal ablations, Fly-v-Fly, and
TCN.

Plan first:

```bash
python run_mobilenetv3_backbone.py plan
```

To include cache construction in the command plan, add `--rebuild_npz`:

```bash
python run_mobilenetv3_backbone.py plan --rebuild_npz
```

This prepends the final MARS training and held-out NPZ builds. MobileNetV3
feature-cache construction remains in the runner's `build_visual_cache` stage,
which consumes those manifests.

Run all selected stages:

```bash
python run_mobilenetv3_backbone.py run-all --skip_completed --keep_going
```

The default classifier setting is the final manuscript capacity,
`full_lstm256`, with LSTM and attention heads across seeds 42, 43, and 44.
