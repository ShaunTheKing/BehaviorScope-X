from __future__ import annotations

from typing import Callable

from PySide6.QtWidgets import QWidget

from app_metadata import APP_NAME
from .workflow_base import ProcessRunner, WorkflowPanel, python_invocation as _python_invocation, script_path as _script_path

_YOLO_WEIGHTS_TOOLTIP = (
    "Ultralytics YOLO-pose checkpoint (.pt) used for keypoint detection, cropping, and YOLO visual feature extraction.\n"
    "Do not place MobileNetV3 or DeepLabCut checkpoints in this field; use their model-family tabs.\n"
    "\n"
    "- MARS top-down mice, the reference dataset:\n"
    "    mars_yolo_pose/runs/pose/train/weights/best.pt\n"
    "  or download a pre-trained MARS YOLO-pose checkpoint from the project release.\n"
    "\n"
    "- Your own dataset or another species:\n"
    "    Train with Ultralytics first:\n"
    "      yolo train task=pose model=yolo11n-pose.pt data=your_pose.yaml epochs=100\n"
    "    then point this field at the resulting best.pt.\n"
    "\n"
    "See README section 'Getting YOLO weights' for a step-by-step guide."
)


class PrepareDatasetPanel(WorkflowPanel):
    def __init__(self, parent: QWidget | None = None, on_success: Callable[[list[str]], None] | None = None) -> None:
        super().__init__(parent)
        self.dataset_root = self._path_row("Class-folder clip root", mode="dir")
        self.yolo_weights = self._path_row("YOLO pose weights", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)")
        self.yolo_weights.setPlaceholderText("Required - see README 'Getting YOLO weights'")
        self.yolo_weights.setToolTip(_YOLO_WEIGHTS_TOOLTIP)
        self.output_root = self._path_row("Output dataset root", mode="dir")
        self.manifest_path = self._path_row("Manifest path", mode="file", file_filter="JSON (*.json);;All files (*.*)", save=True)
        self.clip_metadata_csv = self._path_row("clips_metadata.csv", mode="file", file_filter="CSV (*.csv);;All files (*.*)")
        self.split_map_csv = self._path_row("Split map CSV (explicit split)", mode="file", file_filter="CSV (*.csv);;All files (*.*)")
        self.window_size = self._spin_row("Window size", 32, 1)
        self.window_stride = self._spin_row("Window stride", 16, 1)
        self.n_animals = self._spin_row("Animals", 2, 1)
        self.crop_size = self._spin_row("Animal crop size", 224, 16)
        self.group_crop_size = self._spin_row("Group crop size", 224, 16)
        self.animal_scale_factor = self._double_row("Animal scale factor", 4.0, 0.01, 100.0, 3)
        self.group_scale_factor = self._double_row("Group scale factor", 8.0, 0.01, 100.0, 3)
        self.body_length_px = self._double_row("Body length px (0 = auto)", 0.0, 0.0, 100000.0, 3)
        self.pose_conf_threshold = self._double_row("Pose confidence threshold", 0.3, 0.0, 1.0, 3)
        self.yolo_conf = self._double_row("YOLO confidence", 0.25, 0.0, 1.0, 3)
        self.yolo_iou = self._double_row("YOLO IoU", 0.45, 0.0, 1.0, 3)
        self.yolo_imgsz = self._spin_row("YOLO image size", 640, 32)
        self.device = self._line_row("Device", "cuda:0")
        self.train_ratio = self._double_row("Train ratio", 0.8, 0.0, 1.0, 3)
        self.val_ratio = self._double_row("Val ratio", 0.2, 0.0, 1.0, 3)
        self.test_ratio = self._double_row("Test ratio", 0.0, 0.0, 1.0, 3)
        self.split_strategy = self._combo_row("Split strategy", ["clip", "source", "window", "explicit"], "clip")
        self.source_grouping = self._combo_row("Source grouping", ["clip", "metadata", "filename_prefix"], "clip")
        self.skip_existing = self._checkbox_row("Skip existing NPZ windows", True)
        self.keep_last_box = self._checkbox_row("Keep last box when YOLO misses", True)
        self.validate_manifest = self._checkbox_row("Validate manifest after build", True)
        runner = ProcessRunner(command_builder=self.build_command, on_success=on_success)
        self._wrap(f"Prepare class-folder clips into {APP_NAME} windows", runner)

    def build_command(self) -> list[str]:
        script = _script_path("prepare_clips_x.py")
        if not script.exists():
            raise ValueError(f"Missing script: {script}")
        cmd = [
            *_python_invocation(),
            str(script),
            "--dataset_root",
            self._require(self.dataset_root, "Class-folder clip root"),
            "--yolo_weights",
            self._require(self.yolo_weights, "YOLO pose weights"),
            "--window_size",
            str(self.window_size.value()),
            "--window_stride",
            str(self.window_stride.value()),
            "--crop_size",
            str(self.crop_size.value()),
            "--n_animals",
            str(self.n_animals.value()),
            "--yolo_imgsz",
            str(self.yolo_imgsz.value()),
            "--device",
            self.device.text().strip() or "cuda:0",
            "--train_ratio",
            str(self.train_ratio.value()),
            "--val_ratio",
            str(self.val_ratio.value()),
            "--test_ratio",
            str(self.test_ratio.value()),
            "--split_strategy",
            self.split_strategy.currentText(),
            "--source_grouping",
            self.source_grouping.currentText(),
            "--validate_manifest",
            "true" if self.validate_manifest.isChecked() else "false",
        ]
        if self.output_root.text().strip():
            cmd.extend(["--output_root", self.output_root.text().strip()])
        if self.manifest_path.text().strip():
            cmd.extend(["--manifest_path", self.manifest_path.text().strip()])
        if self.clip_metadata_csv.text().strip():
            cmd.extend(["--clip_metadata_csv", self.clip_metadata_csv.text().strip()])
        if self.split_strategy.currentText() == "explicit" and not (self.clip_metadata_csv.text().strip() or self.split_map_csv.text().strip()):
            raise ValueError("Explicit split strategy requires clips_metadata.csv or a split map CSV.")
        if self.split_map_csv.text().strip():
            cmd.extend(["--split_map_csv", self.split_map_csv.text().strip()])
        cmd.extend(["--group_crop_size", str(self.group_crop_size.value())])
        cmd.extend(["--animal_scale_factor", str(self.animal_scale_factor.value())])
        cmd.extend(["--group_scale_factor", str(self.group_scale_factor.value())])
        if self.body_length_px.value() > 0:
            cmd.extend(["--body_length_px", str(self.body_length_px.value())])
        cmd.extend(["--pose_conf_threshold", str(self.pose_conf_threshold.value())])
        cmd.extend(["--yolo_conf", str(self.yolo_conf.value())])
        cmd.extend(["--yolo_iou", str(self.yolo_iou.value())])
        if self.skip_existing.isChecked():
            cmd.append("--skip_existing")
        if self.keep_last_box.isChecked():
            cmd.append("--keep_last_box")
        cmd.extend(self._extra())
        return cmd


class PrepareFullVideoDatasetPanel(WorkflowPanel):
    def __init__(self, parent: QWidget | None = None, on_success: Callable[[list[str]], None] | None = None) -> None:
        super().__init__(parent)
        self.source_manifest_csv = self._path_row(
            "source_manifest.csv",
            mode="file",
            file_filter="CSV (*.csv);;All files (*.*)",
        )
        self.class_names_file = self._path_row(
            "class_names.txt",
            mode="file",
            file_filter="Text (*.txt);;All files (*.*)",
        )
        self.yolo_weights = self._path_row("YOLO pose weights", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)")
        self.yolo_weights.setPlaceholderText("Required")
        self.yolo_weights.setToolTip(_YOLO_WEIGHTS_TOOLTIP)
        self.output_root = self._path_row("Output dataset root", mode="dir")
        self.manifest_path = self._path_row("Manifest path", mode="file", file_filter="JSON (*.json);;All files (*.*)", save=True)
        self.window_size = self._spin_row("Window size", 32, 1)
        self.window_stride = self._spin_row("Window stride", 16, 1)
        self.source_mode = self._combo_row("Source mode", ["mp4", "seq"], "mp4")
        self.n_animals = self._spin_row("Animals", 2, 1)
        self.crop_size = self._spin_row("Animal crop size", 224, 16)
        self.group_crop_size = self._spin_row("Group crop size", 224, 16)
        self.animal_scale_factor = self._double_row("Animal scale factor", 4.0, 0.01, 100.0, 3)
        self.group_scale_factor = self._double_row("Group scale factor", 8.0, 0.01, 100.0, 3)
        self.body_length_px = self._double_row("Body length px (0 = estimate)", 0.0, 0.0, 100000.0, 3)
        self.pose_conf_threshold = self._double_row("Pose confidence threshold", 0.3, 0.0, 1.0, 3)
        self.yolo_conf = self._double_row("YOLO confidence", 0.25, 0.0, 1.0, 3)
        self.yolo_iou = self._double_row("YOLO IoU", 0.45, 0.0, 1.0, 3)
        self.yolo_imgsz = self._spin_row("YOLO image size", 640, 32)
        self.yolo_batch = self._spin_row("YOLO batch", 64, 1)
        self.fps = self._double_row("Fallback FPS", 30.0, 0.1, 1000.0, 3)
        self.npz_writers = self._spin_row("NPZ writers", 4, 1)
        self.npz_compresslevel = self._spin_row("NPZ deflate level", 1, 0, 9)
        self.label_min_dominance = self._double_row("Label min dominance", 0.5, 0.0, 1.0, 3)
        self.other_subsample = self._double_row("Other subsample", 0.3, 0.0, 1.0, 3)
        self.device = self._line_row("Device", "cuda:0")
        self.skip_existing = self._checkbox_row("Resume completed videos", True)
        self.validate_manifest = self._checkbox_row("Validate manifest after build", True)
        self.validation_sample_limit = self._spin_row("Validation sample limit (0 = all)", 1000, 0)
        runner = ProcessRunner(command_builder=self.build_command, on_success=on_success)
        self._wrap(f"Prepare full-video annotations into {APP_NAME} windows", runner)

    def build_command(self) -> list[str]:
        script = _script_path("prepare_full_video_npz.py")
        if not script.exists():
            raise ValueError(f"Missing script: {script}")
        cmd = [
            *_python_invocation(),
            str(script),
            "--source_manifest_csv",
            self._require(self.source_manifest_csv, "source_manifest.csv"),
            "--class_names_file",
            self._require(self.class_names_file, "class_names.txt"),
            "--yolo_weights",
            self._require(self.yolo_weights, "YOLO pose weights"),
            "--window_size",
            str(self.window_size.value()),
            "--window_stride",
            str(self.window_stride.value()),
            "--crop_size",
            str(self.crop_size.value()),
            "--group_crop_size",
            str(self.group_crop_size.value()),
            "--source_mode",
            self.source_mode.currentText(),
            "--n_animals",
            str(self.n_animals.value()),
            "--animal_scale_factor",
            str(self.animal_scale_factor.value()),
            "--group_scale_factor",
            str(self.group_scale_factor.value()),
            "--pose_conf_threshold",
            str(self.pose_conf_threshold.value()),
            "--yolo_conf",
            str(self.yolo_conf.value()),
            "--yolo_iou",
            str(self.yolo_iou.value()),
            "--yolo_imgsz",
            str(self.yolo_imgsz.value()),
            "--yolo_batch",
            str(self.yolo_batch.value()),
            "--fps",
            str(self.fps.value()),
            "--npz_writers",
            str(self.npz_writers.value()),
            "--npz_compresslevel",
            str(self.npz_compresslevel.value()),
            "--label_min_dominance",
            str(self.label_min_dominance.value()),
            "--other_subsample",
            str(self.other_subsample.value()),
            "--device",
            self.device.text().strip() or "cuda:0",
            "--validation_sample_limit",
            str(self.validation_sample_limit.value()),
        ]
        if self.output_root.text().strip():
            cmd.extend(["--output_root", self.output_root.text().strip()])
        if self.manifest_path.text().strip():
            cmd.extend(["--manifest_path", self.manifest_path.text().strip()])
        if self.body_length_px.value() > 0:
            cmd.extend(["--body_length_px", str(self.body_length_px.value())])
        if self.skip_existing.isChecked():
            cmd.append("--skip_existing")
        if self.validate_manifest.isChecked():
            cmd.append("--validate_manifest")
        else:
            cmd.append("--no-validate_manifest")
        cmd.extend(self._extra())
        return cmd


class MobileNetFullVideoDatasetPanel(WorkflowPanel):
    def __init__(self, parent: QWidget | None = None, on_success: Callable[[list[str]], None] | None = None) -> None:
        super().__init__(parent)
        self.source_manifest_csv = self._path_row(
            "source_manifest.csv",
            mode="file",
            file_filter="CSV (*.csv);;All files (*.*)",
        )
        self.class_names_file = self._path_row(
            "class_names.txt",
            mode="file",
            file_filter="Text (*.txt);;All files (*.*)",
        )
        self.checkpoint = self._path_row("MobileNetV3 pose checkpoint", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)")
        self.checkpoint.setPlaceholderText("Required")
        self.checkpoint.setToolTip(f"{APP_NAME} MobileNetV3-large pose checkpoint used to detect animals and keypoints before windowing.")
        self.output_root = self._path_row("Output dataset root", mode="dir")
        self.manifest_path = self._path_row("Manifest path", mode="file", file_filter="JSON (*.json);;All files (*.*)", save=True)
        self.window_size = self._spin_row("Window size", 32, 1)
        self.window_stride = self._spin_row("Window stride", 16, 1)
        self.source_mode = self._combo_row("Source mode", ["mp4", "seq"], "mp4")
        self.n_animals = self._spin_row("Animals", 2, 1)
        self.crop_size = self._spin_row("Animal crop size", 224, 16)
        self.group_crop_size = self._spin_row("Group crop size", 224, 16)
        self.animal_scale_factor = self._double_row("Animal scale factor", 4.0, 0.01, 100.0, 3)
        self.group_scale_factor = self._double_row("Group scale factor", 8.0, 0.01, 100.0, 3)
        self.body_length_px = self._double_row("Body length px (0 = estimate)", 0.0, 0.0, 100000.0, 3)
        self.pose_conf_threshold = self._double_row("Keypoint confidence threshold", 0.3, 0.0, 1.0, 3)
        self.pose_model_conf = self._double_row("Detector confidence", 0.25, 0.0, 1.0, 3)
        self.pose_model_iou = self._double_row("NMS IoU", 0.45, 0.0, 1.0, 3)
        self.pose_model_batch = self._spin_row("Pose batch", 8, 1)
        self.pre_nms_topk = self._spin_row("Pre-NMS top-k", 1000, 1)
        self.fps = self._double_row("Fallback FPS", 30.0, 0.1, 1000.0, 3)
        self.npz_writers = self._spin_row("NPZ writers", 4, 1)
        self.npz_compresslevel = self._spin_row("NPZ deflate level", 1, 0, 9)
        self.label_min_dominance = self._double_row("Label min dominance", 0.5, 0.0, 1.0, 3)
        self.other_subsample = self._double_row("Other subsample", 0.3, 0.0, 1.0, 3)
        self.device = self._line_row("Device", "cuda:0")
        self.skip_existing = self._checkbox_row("Resume completed videos", True)
        self.validate_manifest = self._checkbox_row("Validate manifest after build", True)
        self.validation_sample_limit = self._spin_row("Validation sample limit (0 = all)", 1000, 0)
        runner = ProcessRunner(command_builder=self.build_command, on_success=on_success)
        self._wrap("Prepare full-video annotations into MobileNetV3 pose windows", runner)

    def build_command(self) -> list[str]:
        script = _script_path("prepare_full_video_npz.py")
        if not script.exists():
            raise ValueError(f"Missing script: {script}")
        cmd = [
            *_python_invocation(),
            str(script),
            "--pose_backend",
            "mobilenetv3",
            "--source_manifest_csv",
            self._require(self.source_manifest_csv, "source_manifest.csv"),
            "--class_names_file",
            self._require(self.class_names_file, "class_names.txt"),
            "--mobilenetv3_checkpoint",
            self._require(self.checkpoint, "MobileNetV3 pose checkpoint"),
            "--window_size",
            str(self.window_size.value()),
            "--window_stride",
            str(self.window_stride.value()),
            "--crop_size",
            str(self.crop_size.value()),
            "--group_crop_size",
            str(self.group_crop_size.value()),
            "--source_mode",
            self.source_mode.currentText(),
            "--n_animals",
            str(self.n_animals.value()),
            "--animal_scale_factor",
            str(self.animal_scale_factor.value()),
            "--group_scale_factor",
            str(self.group_scale_factor.value()),
            "--pose_conf_threshold",
            str(self.pose_conf_threshold.value()),
            "--pose_model_conf",
            str(self.pose_model_conf.value()),
            "--pose_model_iou",
            str(self.pose_model_iou.value()),
            "--pose_model_batch",
            str(self.pose_model_batch.value()),
            "--mobilenetv3_pre_nms_topk",
            str(self.pre_nms_topk.value()),
            "--fps",
            str(self.fps.value()),
            "--npz_writers",
            str(self.npz_writers.value()),
            "--npz_compresslevel",
            str(self.npz_compresslevel.value()),
            "--label_min_dominance",
            str(self.label_min_dominance.value()),
            "--other_subsample",
            str(self.other_subsample.value()),
            "--device",
            self.device.text().strip() or "cuda:0",
            "--validation_sample_limit",
            str(self.validation_sample_limit.value()),
        ]
        if self.output_root.text().strip():
            cmd.extend(["--output_root", self.output_root.text().strip()])
        if self.manifest_path.text().strip():
            cmd.extend(["--manifest_path", self.manifest_path.text().strip()])
        if self.body_length_px.value() > 0:
            cmd.extend(["--body_length_px", str(self.body_length_px.value())])
        if self.skip_existing.isChecked():
            cmd.append("--skip_existing")
        if self.validate_manifest.isChecked():
            cmd.append("--validate_manifest")
        else:
            cmd.append("--no-validate_manifest")
        cmd.extend(self._extra())
        return cmd


class FeatureCachePanel(WorkflowPanel):
    def __init__(self, parent: QWidget | None = None, on_success: Callable[[list[str]], None] | None = None) -> None:
        super().__init__(parent)
        self.manifest_path = self._path_row("Manifest path", mode="file", file_filter="JSON (*.json);;All files (*.*)")
        self.yolo_weights = self._path_row("YOLO pose weights", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)")
        self.yolo_weights.setPlaceholderText("Required")
        self.yolo_weights.setToolTip(_YOLO_WEIGHTS_TOOLTIP)
        self.output_dir = self._path_row("Feature cache output", mode="dir")
        self.splits = self._line_row("Splits", "train val")
        self.batch = self._spin_row("Batch", 24, 1)
        self.num_workers = self._spin_row("Workers", 2, 0)
        self.device = self._line_row("Device", "cuda")
        self.cache_dtype = self._combo_row("Cache dtype", ["float32", "float16"], "float32")
        self.amp_dtype = self._combo_row("AMP dtype", ["auto", "bf16", "fp16"], "auto")
        self.yolo_backbone_end_layer = self._spin_row("YOLO backbone end layer", 10, 1)
        self.n_animals = self._spin_row("Animals", 2, 1)
        self.num_keypoints = self._spin_row("Keypoints (0 = auto)", 0, 0)
        self.overwrite = self._checkbox_row("Overwrite existing cache entries", False)
        self.skip_invalid_samples = self._checkbox_row("Skip invalid samples", False)
        self.amp = self._checkbox_row("AMP", True)
        runner = ProcessRunner(command_builder=self.build_command, on_success=on_success)
        self._wrap("Build or refresh YOLO visual feature cache", runner)

    def build_command(self) -> list[str]:
        script = _script_path("precompute_visual_features_x.py")
        if not script.exists():
            raise ValueError(f"Missing script: {script}")
        splits = [part for part in self.splits.text().replace(",", " ").split() if part]
        if not splits:
            raise ValueError("At least one split is required.")
        cmd = [
            *_python_invocation(),
            str(script),
            "--manifest_path",
            self._require(self.manifest_path, "Manifest path"),
            "--yolo_weights",
            self._require(self.yolo_weights, "YOLO pose weights"),
            "--output_dir",
            self._require(self.output_dir, "Feature cache output"),
            "--splits",
            *splits,
            "--batch",
            str(self.batch.value()),
            "--num_workers",
            str(self.num_workers.value()),
            "--device",
            self.device.text().strip() or "cuda",
            "--cache_dtype",
            self.cache_dtype.currentText(),
            "--yolo_backbone_end_layer",
            str(self.yolo_backbone_end_layer.value()),
            "--n_animals",
            str(self.n_animals.value()),
        ]
        if self.num_keypoints.value() > 0:
            cmd.extend(["--num_keypoints", str(self.num_keypoints.value())])
        if self.overwrite.isChecked():
            cmd.append("--overwrite")
        if self.skip_invalid_samples.isChecked():
            cmd.append("--skip_invalid_samples")
        if self.amp.isChecked():
            cmd.append("--amp")
            cmd.extend(["--amp_dtype", self.amp_dtype.currentText()])
        cmd.extend(self._extra())
        return cmd


class MobileNetFeatureCachePanel(WorkflowPanel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.manifest_path = self._path_row("Sequence manifest", mode="file", file_filter="JSON (*.json);;All files (*.*)")
        self.manifest_path.setToolTip("BehaviorScope sequence_manifest.json built from full-video windows or a compatible cache builder.")
        self.checkpoint = self._path_row("MobileNetV3 pose checkpoint", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)")
        self.checkpoint.setToolTip("MobileNetV3-large pose checkpoint containing backbone.features.* weights.")
        self.output_dir = self._path_row("Feature cache output", mode="dir")
        self.splits = self._line_row("Splits", "train val")
        self.batch = self._spin_row("Batch", 2, 1)
        self.num_workers = self._spin_row("Workers", 2, 0)
        self.resize_size = self._spin_row("Resize size", 640, 32)
        self.n_animals = self._spin_row("Animals", 2, 1)
        self.device = self._line_row("Device", "cuda")
        self.cache_dtype = self._combo_row("Cache dtype", ["float32", "float16"], "float32")
        self.max_samples = self._spin_row("Max samples (0 = all)", 0, 0)
        self.overwrite = self._checkbox_row("Overwrite existing cache entries", False)
        self.amp = self._checkbox_row("AMP", True)
        self.amp_dtype = self._combo_row("AMP dtype", ["auto", "bf16", "fp16"], "auto")
        runner = ProcessRunner(command_builder=self.build_command)
        self._wrap("Build MobileNetV3 pose-backbone visual feature cache", runner)

    def build_command(self) -> list[str]:
        script = _script_path("precompute_mobilenetv3_features_x.py")
        if not script.exists():
            raise ValueError(f"Missing script: {script}")
        splits = [part for part in self.splits.text().replace(",", " ").split() if part]
        if not splits:
            raise ValueError("At least one split is required.")
        cmd = [
            *_python_invocation(),
            str(script),
            "--manifest_path",
            self._require(self.manifest_path, "Sequence manifest"),
            "--checkpoint",
            self._require(self.checkpoint, "MobileNetV3 pose checkpoint"),
            "--output_dir",
            self._require(self.output_dir, "Feature cache output"),
            "--splits",
            *splits,
            "--n_animals",
            str(self.n_animals.value()),
            "--resize_size",
            str(self.resize_size.value()),
            "--batch",
            str(self.batch.value()),
            "--num_workers",
            str(self.num_workers.value()),
            "--device",
            self.device.text().strip() or "cuda",
            "--cache_dtype",
            self.cache_dtype.currentText(),
        ]
        if self.max_samples.value() > 0:
            cmd.extend(["--max_samples", str(self.max_samples.value())])
        if self.overwrite.isChecked():
            cmd.append("--overwrite")
        if self.amp.isChecked():
            cmd.append("--amp")
            cmd.extend(["--amp_dtype", self.amp_dtype.currentText()])
        cmd.extend(self._extra())
        return cmd


