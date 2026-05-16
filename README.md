# BehaviorScope-Y

BehaviorScope-Y is a YOLO-backed behavior-classification workflow for multi-animal videos. A YOLO-pose checkpoint is reused for both keypoint detection and frozen visual feature extraction, while a temporal classifier learns behavior from visual tokens, pose-self features, and inter-animal relational geometry.

The public workflow is:

1. Annotate full videos or export class-folder clips with the Qt app.
2. Build BehaviorScope-Y NPZ windows.
3. Build or reuse a YOLO visual feature cache.
4. Train a classifier with configurable LSTM or attention temporal heads.
5. Export a bundled single `.pt` that contains both the classifier and YOLO-pose weights.
6. Run inference with that single `.pt`, without juggling a separate YOLO path.

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

For GPU training, install the CUDA build of PyTorch first, then install the rest of the requirements:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`ffmpeg` is recommended for clip extraction and video conversion.

## Launch The App

```bash
python behaviorscope_y_qt.py
```

The Qt workflow tabs are ordered for the normal user path:

- `Annotate + Clip`: import videos, annotate spans, and export full-video annotation manifests.
- `Prepare Full Video`: converts full-video annotation exports into NPZ windows.
- `Feature Cache`: precomputes frozen YOLO visual tokens for selected splits.
- `Train`: trains the temporal classifier and can export a bundled single `.pt`.
- `Inference` / `Batch`: runs prediction on one video or a folder.

After each successful step, the app fills the next tab's paths automatically.

## Command Line Workflow

Prepare full-video NPZs:

```bash
python prepare_full_video_npz.py ^
  --source_manifest_csv path\to\source_manifest.csv ^
  --class_names_file path\to\class_names.txt ^
  --yolo_weights path\to\yolo_pose_best.pt ^
  --output_root runs\my_dataset_npz ^
  --validate_manifest
```

Build the visual feature cache:

```bash
python precompute_visual_features_y.py ^
  --manifest_path runs\my_dataset_npz\sequence_manifest.json ^
  --yolo_weights path\to\yolo_pose_best.pt ^
  --output_dir runs\my_dataset_npz\yolo_feature_cache ^
  --splits train val ^
  --cache_dtype float32
```

Train with an LSTM temporal head:

```bash
python train_y.py ^
  --manifest_path runs\my_dataset_npz\sequence_manifest.json ^
  --yolo_weights path\to\yolo_pose_best.pt ^
  --use_feature_cache runs\my_dataset_npz\yolo_feature_cache ^
  --sequence_model lstm ^
  --hidden_dim 896 ^
  --num_lstm_layers 1 ^
  --attention_heads 4 ^
  --project runs ^
  --name my_lstm896_run ^
  --export_single_model
```

Train with an attention temporal head:

```bash
python train_y.py ^
  --manifest_path runs\my_dataset_npz\sequence_manifest.json ^
  --yolo_weights path\to\yolo_pose_best.pt ^
  --auto_feature_cache ^
  --sequence_model attention ^
  --hidden_dim 896 ^
  --attention_heads 8 ^
  --positional_encoding sinusoidal ^
  --project runs ^
  --name my_attention_run ^
  --export_single_model
```

Run inference with the bundled model:

```bash
python infer_y.py ^
  --model_path runs\my_lstm896_run\behaviorscope_y_single_model.pt ^
  --source path\to\video.mp4 ^
  --output runs\my_lstm896_run\inference_outputs\video.behavior.csv ^
  --output_video runs\my_lstm896_run\inference_outputs\video.annotated.mp4
```

For older two-file checkpoints, pass `--model_path`, optional `--model_config`, and `--yolo_weights`.

## Architecture Knobs

The CLI and Qt training panel expose the manuscript-relevant model controls:

- Temporal head: `--sequence_model lstm|attention`
- Recurrent capacity: `--hidden_dim`, `--num_lstm_layers`, `--bidirectional_lstm`
- Attention behavior: `--attention_heads`, `--use_attention_pool`, `--positional_encoding`
- Fusion: `--pose_fusion_dim`, `--pose_fusion_strategy`
- YOLO visual trunk: `--yolo_backbone_end_layer`, `--train_backbone`, `--backbone_lr`
- Stream ablations: `--disable_group_rgb`, `--disable_per_animal_rgb`, `--disable_pose_self`, `--disable_relations`, `--disable_visual_streams`
- Cache behavior: `--auto_feature_cache`, `--use_feature_cache`, `--feature_cache_dtype`

## Single-File Model Packaging

Training can write the bundled model automatically:

```bash
python train_y.py ... --export_single_model
```

You can also package an existing run:

```bash
python package_single_model_y.py ^
  --classifier_checkpoint runs\my_run\best_model_macro_f1.pt ^
  --model_config runs\my_run\config.json ^
  --yolo_weights path\to\yolo_pose_best.pt ^
  --output runs\my_run\behaviorscope_y_single_model.pt
```

The bundled `.pt` embeds the classifier checkpoint, training config, decoder metadata, temporal splitter metadata when present, and the raw YOLO-pose checkpoint bytes.
