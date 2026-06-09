from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtWidgets import QWidget

from app_metadata import APP_NAME, APP_SLUG, LOCAL_STATE_DIR
from .workflow_base import REPO_ROOT, THIS_DIR, ProcessRunner, WorkflowPanel, python_invocation as _python_invocation, script_path as _script_path
from .workflow_dataset_panels import _YOLO_WEIGHTS_TOOLTIP

class TrainPanel(WorkflowPanel):
    def __init__(
        self,
        parent: QWidget | None = None,
        on_success: Callable[[list[str]], None] | None = None,
        *,
        workflow_label: str = "YOLO-pose",
        feature_mode: str = "yolo",
    ) -> None:
        super().__init__(parent)
        self.workflow_label = workflow_label
        self.feature_mode = feature_mode
        self.manifest_path = self._path_row("Manifest path", mode="file", file_filter="JSON (*.json);;All files (*.*)")
        self.manifest_path.setToolTip("sequence_manifest.json from the same pose workflow used to build the training windows.")
        self.yolo_weights = self._path_row("YOLO pose weights", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)")
        self.yolo_weights.setPlaceholderText("Required only for YOLO-backed training or auto YOLO feature caching")
        self.yolo_weights.setToolTip(_YOLO_WEIGHTS_TOOLTIP)
        self.project = self._path_row("Run project folder", mode="dir", default=str(THIS_DIR / "runs"))
        self.name = self._line_row("Run name", f"{APP_SLUG}_run")
        self.epochs = self._spin_row("Epochs", 10, 1)
        self.patience = self._spin_row("Patience", 5, 0)
        self.batch = self._spin_row("Batch size", 24, 1)
        self.num_workers = self._spin_row("Workers", 2, 0)
        self.lr = self._line_row("Learning rate", "5e-5")
        self.weight_decay = self._line_row("Weight decay", "5e-4")
        self.dropout = self._double_row("Dropout", 0.6, 0.0, 1.0, 3)
        self.scheduler = self._combo_row("Scheduler", ["plateau", "cosine", "none"], "plateau")
        self.class_weighting = self._combo_row("Class weighting", ["sqrt_inverse", "inverse", "none"], "sqrt_inverse")
        self.class_weight_clamp = self._double_row("Class weight clamp", 1.5, 0.0, 100.0, 3)
        self.train_sampler = self._combo_row("Train sampler", ["random", "weighted"], "weighted")
        self.n_animals = self._spin_row("Animals", 2, 1)
        self.num_keypoints = self._spin_row("Keypoints (0 = auto)", 0, 0)
        self.hidden_dim = self._spin_row("Hidden dim", 256, 16)
        self.num_lstm_layers = self._spin_row("LSTM layers", 1, 1)
        self.bidirectional_lstm = self._checkbox_row("Bidirectional LSTM", False)
        self.pose_fusion_dim = self._spin_row("Pose fusion dim", 128, 16)
        self.pose_fusion_strategy = self._combo_row("Pose fusion", ["gated_attention", "concat"], "gated_attention")
        self.sequence_model = self._combo_row("Sequence model", ["lstm", "attention"], "lstm")
        self.use_attention_pool = self._checkbox_row("Attention pool after LSTM", False)
        self.attention_heads = self._spin_row("Attention heads", 4, 1)
        self.positional_encoding = self._combo_row("Positional encoding", ["learned", "sinusoidal", "none"], "learned")
        self.yolo_backbone_end_layer = self._spin_row("YOLO backbone end layer", 10, 1)
        self.train_backbone = self._checkbox_row("Fine-tune YOLO visual backbone", False)
        self.backbone_lr = self._line_row("Backbone learning rate", "1e-5")
        self.device = self._line_row("Device", "cuda")
        self.seed = self._spin_row("Seed", 0, 0)
        self.log_interval = self._spin_row("Log interval", 200, 0)
        self.auto_feature_cache = self._checkbox_row("Auto-build/reuse YOLO feature cache", True)
        self.use_feature_cache = self._path_row("Existing feature cache", mode="dir")
        self.feature_cache_batch = self._spin_row("Feature cache batch (0 = train batch)", 0, 0)
        self.feature_cache_num_workers = self._spin_row("Feature cache workers (-1 = train workers)", -1, -1)
        self.feature_cache_dtype = self._combo_row("Feature cache dtype", ["float32", "float16"], "float32")
        self.train_splits = self._line_row("Train splits", "train")
        self.val_splits = self._line_row("Validation splits", "val")
        self.allow_test_split_training = self._checkbox_row("Allow test split training", False)
        self.skip_invalid_samples = self._checkbox_row("Skip invalid samples", False)
        self.amp = self._checkbox_row("AMP", True)
        self.amp_dtype = self._combo_row("AMP dtype", ["auto", "bf16", "fp16"], "auto")
        self.per_class_metrics = self._checkbox_row("Per-class metrics", True)
        self.confusion_matrix = self._checkbox_row("Confusion matrix", True)
        self.disable_threshold_decoder = self._checkbox_row("Disable auto threshold decoder", False)
        self.background_class = self._line_row("Background class", "auto")
        self.threshold_grid_min = self._double_row("Threshold grid min", 0.30, 0.0, 1.0, 3)
        self.threshold_grid_max = self._double_row("Threshold grid max", 0.95, 0.0, 1.0, 3)
        self.threshold_grid_step = self._double_row("Threshold grid step", 0.05, 0.001, 1.0, 3)
        self.threshold_fit_rounds = self._spin_row("Threshold fit rounds", 3, 1)
        self.save_decoder_json = self._checkbox_row("Save decoder JSON sidecar", False)
        self.save_temporal_splitter_json = self._checkbox_row("Save temporal splitter JSON sidecar", False)
        self.disable_visual_streams = self._checkbox_row("Disable all visual streams", False)
        self.disable_group_rgb = self._checkbox_row("Disable group RGB stream", False)
        self.disable_per_animal_rgb = self._checkbox_row("Disable per-animal RGB stream", False)
        self.disable_pose_self = self._checkbox_row("Disable pose-self stream", False)
        self.disable_relations = self._checkbox_row("Disable relations stream", False)
        self.relations_pose_only = self._checkbox_row("Relations use pose only", False)
        self.pose_dropout_p_uniform = self._double_row("Pose dropout p uniform", 0.0, 0.0, 1.0, 3)
        self.export_single_model = self._checkbox_row("Export single bundled .pt after training", True)
        self.single_model_path = self._path_row("Bundled .pt output", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)", save=True)
        # Temporal splitter - annotation-aware fitting
        self.temporal_splitter_annot_root = self._path_row(
            "Splitter annot root (optional)",
            mode="dir",
            default="",
        )
        self.temporal_splitter_strict_annot = self._checkbox_row(
            "Strict annot mode (skip videos missing .annot)", False
        )
        self.disable_temporal_splitter = self._checkbox_row("Disable temporal splitter", False)
        self._configure_feature_mode()
        runner = ProcessRunner(
            command_builder=self.build_command,
            graceful_stop=self.request_stop_after_epoch,
            on_success=on_success,
        )
        self._wrap(f"Train {APP_NAME} temporal classifier ({workflow_label})", runner)

    def _configure_feature_mode(self) -> None:
        if self.feature_mode == "yolo":
            self.use_feature_cache.setToolTip("Optional precomputed YOLO feature cache. Leave empty to auto-build or train from raw visual streams.")
            self.auto_feature_cache.setToolTip("Build or reuse a YOLO visual feature cache before classifier training.")
            return
        self.yolo_weights.clear()
        self.auto_feature_cache.setChecked(False)
        self.train_backbone.setChecked(False)
        self.use_feature_cache.setPlaceholderText(f"Required {self.workflow_label} feature cache")
        self.use_feature_cache.setToolTip(
            f"Feature cache produced by the {self.workflow_label} feature-cache stage. "
            "Training uses the same temporal classifier, but the visual descriptors come from this pose workflow."
        )
        self.export_single_model.setChecked(False)
        self.export_single_model.setEnabled(False)
        self.export_single_model.setToolTip(
            "Single-file bundled inference is only available for YOLO-backed models in the current GUI. "
            f"{self.workflow_label} predictions are produced by the workflow evaluation runners."
        )
        for field in (
            self.yolo_weights,
            self.yolo_backbone_end_layer,
            self.train_backbone,
            self.backbone_lr,
            self.auto_feature_cache,
            self.feature_cache_batch,
            self.feature_cache_num_workers,
            self.feature_cache_dtype,
            self.single_model_path,
        ):
            self._set_field_visible(field, False)

    def build_command(self) -> list[str]:
        script = _script_path("train_x.py")
        if not script.exists():
            raise ValueError(f"Missing script: {script}")
        has_yolo_weights = bool(self.yolo_weights.text().strip())
        has_feature_cache = bool(self.use_feature_cache.text().strip())
        if self.train_backbone.isChecked() and (self.auto_feature_cache.isChecked() or has_feature_cache):
            raise ValueError("Fine-tuning the YOLO backbone is incompatible with frozen feature cache use.")
        if self.train_backbone.isChecked() and not has_yolo_weights:
            raise ValueError("Fine-tuning the YOLO backbone requires YOLO pose weights.")
        if self.auto_feature_cache.isChecked() and not has_yolo_weights and not has_feature_cache:
            raise ValueError("Auto-building a YOLO feature cache requires YOLO pose weights. For MobileNetV3 or DeepLabCut, select an existing feature cache and turn off auto-build.")
        if self.feature_mode != "yolo" and not has_feature_cache:
            raise ValueError(f"{self.workflow_label} training requires an existing feature cache.")
        cmd = [
            *_python_invocation(),
            str(script),
            "--manifest_path",
            self._require(self.manifest_path, "Manifest path"),
            "--epochs",
            str(self.epochs.value()),
            "--patience",
            str(self.patience.value()),
            "--scheduler",
            self.scheduler.currentText(),
            "--batch",
            str(self.batch.value()),
            "--num_workers",
            str(self.num_workers.value()),
            "--lr",
            self.lr.text().strip() or "5e-5",
            "--weight_decay",
            self.weight_decay.text().strip() or "5e-4",
            "--dropout",
            str(self.dropout.value()),
            "--class_weighting",
            self.class_weighting.currentText(),
            "--class_weight_clamp",
            str(self.class_weight_clamp.value()),
            "--train_sampler",
            self.train_sampler.currentText(),
            "--n_animals",
            str(self.n_animals.value()),
            "--hidden_dim",
            str(self.hidden_dim.value()),
            "--num_lstm_layers",
            str(self.num_lstm_layers.value()),
            "--pose_fusion_dim",
            str(self.pose_fusion_dim.value()),
            "--pose_fusion_strategy",
            self.pose_fusion_strategy.currentText(),
            "--sequence_model",
            self.sequence_model.currentText(),
            "--attention_heads",
            str(self.attention_heads.value()),
            "--positional_encoding",
            self.positional_encoding.currentText(),
            "--yolo_backbone_end_layer",
            str(self.yolo_backbone_end_layer.value()),
            "--log_interval",
            str(self.log_interval.value()),
            "--seed",
            str(self.seed.value()),
            "--device",
            self.device.text().strip() or "cuda",
            "--project",
            self._require(self.project, "Run project folder"),
            "--name",
            self._require(self.name, "Run name"),
        ]
        if has_yolo_weights:
            cmd.extend(["--yolo_weights", self.yolo_weights.text().strip()])
        if self.num_keypoints.value() > 0:
            cmd.extend(["--num_keypoints", str(self.num_keypoints.value())])
        if self.bidirectional_lstm.isChecked():
            cmd.append("--bidirectional_lstm")
        if self.use_attention_pool.isChecked():
            cmd.append("--use_attention_pool")
        if self.train_backbone.isChecked():
            cmd.extend(["--train_backbone", "--backbone_lr", self.backbone_lr.text().strip() or "1e-5"])
        if self.auto_feature_cache.isChecked():
            cmd.append("--auto_feature_cache")
            cmd.extend(["--feature_cache_batch", str(self.feature_cache_batch.value())])
            cmd.extend(["--feature_cache_num_workers", str(self.feature_cache_num_workers.value())])
            cmd.extend(["--feature_cache_dtype", self.feature_cache_dtype.currentText()])
        if self.use_feature_cache.text().strip():
            cmd.extend(["--use_feature_cache", self.use_feature_cache.text().strip()])
        train_splits = [part for part in self.train_splits.text().replace(",", " ").split() if part]
        val_splits = [part for part in self.val_splits.text().replace(",", " ").split() if part]
        if train_splits:
            cmd.extend(["--train_splits", *train_splits])
        if val_splits:
            cmd.extend(["--val_splits", *val_splits])
        if self.allow_test_split_training.isChecked():
            cmd.append("--allow_test_split_training")
        if self.skip_invalid_samples.isChecked():
            cmd.append("--skip_invalid_samples")
        if self.amp.isChecked():
            cmd.append("--amp")
            cmd.extend(["--amp_dtype", self.amp_dtype.currentText()])
        if self.per_class_metrics.isChecked():
            cmd.append("--per_class_metrics")
        if self.confusion_matrix.isChecked():
            cmd.extend(["--confusion_matrix", "--confusion_matrix_interval", "1"])
        cmd.extend(["--background_class", self.background_class.text().strip() or "auto"])
        cmd.extend(["--threshold_grid_min", str(self.threshold_grid_min.value())])
        cmd.extend(["--threshold_grid_max", str(self.threshold_grid_max.value())])
        cmd.extend(["--threshold_grid_step", str(self.threshold_grid_step.value())])
        cmd.extend(["--threshold_fit_rounds", str(self.threshold_fit_rounds.value())])
        if self.disable_threshold_decoder.isChecked():
            cmd.append("--disable_threshold_decoder")
        if self.save_decoder_json.isChecked():
            cmd.append("--save_decoder_json")
        if self.save_temporal_splitter_json.isChecked():
            cmd.append("--save_temporal_splitter_json")
        if self.disable_visual_streams.isChecked():
            cmd.append("--disable_visual_streams")
        if self.disable_group_rgb.isChecked():
            cmd.append("--disable_group_rgb")
        if self.disable_per_animal_rgb.isChecked():
            cmd.append("--disable_per_animal_rgb")
        if self.disable_pose_self.isChecked():
            cmd.append("--disable_pose_self")
        if self.disable_relations.isChecked():
            cmd.append("--disable_relations")
        if self.relations_pose_only.isChecked():
            cmd.append("--relations_pose_only")
        if self.pose_dropout_p_uniform.value() > 0:
            cmd.extend(["--pose_dropout_p_uniform", str(self.pose_dropout_p_uniform.value())])
        if self.export_single_model.isChecked():
            cmd.append("--export_single_model")
            if self.single_model_path.text().strip():
                cmd.extend(["--single_model_path", self.single_model_path.text().strip()])
        # Temporal splitter
        annot_root = self.temporal_splitter_annot_root.text().strip()
        if annot_root:
            cmd.extend(["--temporal_splitter_annot_root", annot_root])
        if self.temporal_splitter_strict_annot.isChecked():
            cmd.append("--temporal_splitter_strict_annot")
        if self.disable_temporal_splitter.isChecked():
            cmd.append("--disable_temporal_splitter")
        cmd.extend(self._extra())
        return cmd

    def request_stop_after_epoch(self) -> str:
        project = Path(self.project.text().strip() or ".")
        name = self.name.text().strip()
        if not name:
            return "stop not requested: run name is empty"
        run_dir = project / name
        run_dir.mkdir(parents=True, exist_ok=True)
        flag = run_dir / "stop_training.flag"
        flag.write_text("stop after current epoch\n", encoding="utf-8")
        return f"requested stop after current epoch via {flag}"


class InferencePanel(WorkflowPanel):
    def __init__(self, *, batch_mode: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.batch_mode = batch_mode
        self.model_path = self._path_row("Model .pt", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)")
        self.model_config = self._path_row("Model config.json", mode="file", file_filter="JSON (*.json);;All files (*.*)")
        self.yolo_weights = self._path_row("YOLO pose weights", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)")
        self.yolo_weights.setPlaceholderText("Optional for bundled single-model .pt")
        self.yolo_weights.setToolTip(_YOLO_WEIGHTS_TOOLTIP)
        source_label = "Source video folder" if batch_mode else "Source video/file/folder"
        self.source = self._path_row(source_label, mode="dir" if batch_mode else "file", file_filter="Videos (*.mp4 *.avi *.mov *.mkv *.seq);;All files (*.*)")
        self.output = self._path_row("Output CSV (single video)", mode="file", file_filter="CSV (*.csv);;All files (*.*)", save=True)
        self.output_dir = self._path_row("Output CSV directory", mode="dir")
        self.output_video = self._path_row("Output MP4 (single video)", mode="file", file_filter="MP4 (*.mp4);;All files (*.*)", save=True)
        self.output_video_dir = self._path_row("Output MP4 directory", mode="dir")
        self.device = self._line_row("Device", "cuda")
        self.yolo_batch = self._spin_row("YOLO batch", 64, 1)
        self.fps = self._double_row("Fallback FPS", 30.0, 0.1, 1000.0, 3)
        self.crop_size = self._spin_row("Animal crop size", 224, 16)
        self.animal_scale_factor = self._double_row("Animal scale factor", 4.0, 0.01, 100.0, 3)
        self.group_scale_factor = self._double_row("Group scale factor", 8.0, 0.01, 100.0, 3)
        self.body_length_px = self._double_row("Body length px (0 = config/calibration)", 0.0, 0.0, 100000.0, 3)
        self.calibration_json = self._path_row("Calibration JSON", mode="file", file_filter="JSON (*.json);;All files (*.*)")
        self.pose_conf_threshold = self._double_row("Pose confidence threshold", 0.3, 0.0, 1.0, 3)
        self.num_frames = self._spin_row("Window frames (0 = model config)", 0, 0)
        self.window_stride = self._spin_row("Window stride (0 = model config)", 0, 0)
        self.temporal_smoothing_window = self._spin_row("Median smoothing window", 0, 0)
        self.bout_min_duration = self._spin_row("Bout min duration frames", 0, 0)
        self.csv_flush_interval = self._spin_row("CSV flush interval", 50, 1)
        self.decoder_config = self._path_row("Decoder config JSON", mode="file", file_filter="JSON (*.json);;All files (*.*)")
        self.disable_threshold_decoder = self._checkbox_row("Disable threshold decoder", False)
        self.temporal_splitter_config = self._path_row("Temporal splitter JSON", mode="file", file_filter="JSON (*.json);;All files (*.*)")
        self.disable_temporal_splitter = self._checkbox_row("Disable temporal splitter", False)
        self.smooth_bouts = self._checkbox_row("Smooth bouts for MP4/frame CSV", True)
        self.log_metrics = self._checkbox_row("Log runtime metrics", True)
        self.metrics_log_interval = self._spin_row("Metrics log interval", 30, 1)
        self.amp = self._checkbox_row("Inference AMP", True)
        self.amp_dtype = self._combo_row("AMP dtype", ["auto", "bf16", "fp16"], "auto")
        self.export_pose = self._checkbox_row("Export pose CSV (keypoints + bbox per frame)", False)
        self.no_bboxes = self._checkbox_row("Hide boxes in review MP4", False)
        self.no_keypoints = self._checkbox_row("Hide keypoints in review MP4", False)
        self.no_video_header = self._checkbox_row("Do not write video metadata header", False)
        runner = ProcessRunner(command_builder=self.build_command, graceful_stop=self.request_stop)
        title = f"Batch process a folder with {APP_NAME}" if batch_mode else f"Run {APP_NAME} inference"
        self._wrap(title, runner)

    def build_command(self) -> list[str]:
        script = _script_path("infer_x.py")
        if not script.exists():
            raise ValueError(f"Missing script: {script}")
        cmd = [
            *_python_invocation(),
            str(script),
            "--model_path",
            self._require(self.model_path, "Model .pt"),
            "--source",
            self._require(self.source, "Source"),
            "--device",
            self.device.text().strip() or "cuda",
            "--yolo_batch",
            str(self.yolo_batch.value()),
            "--fps",
            str(self.fps.value()),
            "--crop_size",
            str(self.crop_size.value()),
            "--animal_scale_factor",
            str(self.animal_scale_factor.value()),
            "--group_scale_factor",
            str(self.group_scale_factor.value()),
            "--pose_conf_threshold",
            str(self.pose_conf_threshold.value()),
            "--num_frames",
            str(self.num_frames.value()),
            "--window_stride",
            str(self.window_stride.value()),
            "--temporal_smoothing_window",
            str(self.temporal_smoothing_window.value()),
            "--bout_min_duration_frames",
            str(self.bout_min_duration.value()),
            "--csv_flush_interval",
            str(self.csv_flush_interval.value()),
        ]
        if self.yolo_weights.text().strip():
            cmd.extend(["--yolo_weights", self.yolo_weights.text().strip()])
        if self.model_config.text().strip():
            cmd.extend(["--model_config", self.model_config.text().strip()])
        if self.body_length_px.value() > 0:
            cmd.extend(["--body_length_px", str(self.body_length_px.value())])
        if self.calibration_json.text().strip():
            cmd.extend(["--calibration_json", self.calibration_json.text().strip()])
        if self.output.text().strip() and not self.batch_mode:
            cmd.extend(["--output", self.output.text().strip()])
        if self.output_dir.text().strip():
            cmd.extend(["--output_dir", self.output_dir.text().strip()])
        if self.output_video.text().strip() and not self.batch_mode:
            cmd.extend(["--output_video", self.output_video.text().strip()])
        if self.output_video_dir.text().strip():
            cmd.extend(["--output_video_dir", self.output_video_dir.text().strip()])
        if self.decoder_config.text().strip():
            cmd.extend(["--decoder_config", self.decoder_config.text().strip()])
        if self.disable_threshold_decoder.isChecked():
            cmd.append("--disable_threshold_decoder")
        if self.temporal_splitter_config.text().strip():
            cmd.extend(["--temporal_splitter_config", self.temporal_splitter_config.text().strip()])
        if self.disable_temporal_splitter.isChecked():
            cmd.append("--disable_temporal_splitter")
        if self.smooth_bouts.isChecked():
            cmd.append("--smooth_bouts")
        if self.log_metrics.isChecked():
            cmd.append("--log_metrics")
            cmd.extend(["--metrics_log_interval", str(self.metrics_log_interval.value())])
        if self.amp.isChecked():
            cmd.append("--amp")
            cmd.extend(["--amp_dtype", self.amp_dtype.currentText()])
        if self.export_pose.isChecked():
            cmd.append("--export_pose")
        if self.no_bboxes.isChecked():
            cmd.append("--no_bboxes")
        if self.no_keypoints.isChecked():
            cmd.append("--no_keypoints")
        if self.no_video_header.isChecked():
            cmd.append("--no_video_header")
        stop_flag = self._stop_flag_path()
        cmd.extend(["--stop_flag_file", str(stop_flag)])
        cmd.extend(self._extra())
        return cmd

    def _stop_flag_path(self) -> Path:
        base = Path(self.output_dir.text().strip() or self.output.text().strip() or REPO_ROOT / LOCAL_STATE_DIR)
        if base.suffix:
            base = base.parent
        base.mkdir(parents=True, exist_ok=True)
        return base / "stop_inference.flag"

    def request_stop(self) -> str:
        flag = self._stop_flag_path()
        flag.write_text("stop inference\n", encoding="utf-8")
        return f"requested graceful inference stop via {flag}"

