from __future__ import annotations

import shlex
import subprocess
import sys
import re
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QProcess, QProcessEnvironment
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


THIS_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = THIS_DIR.parent


def _script_path(name: str) -> Path:
    return THIS_DIR / name


def _split_extra_args(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return shlex.split(text, posix=False)


def _display_command(cmd: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in cmd])


def _python_invocation() -> list[str]:
    return [sys.executable, "-u"]


class ProcessRunner(QWidget):
    def __init__(
        self,
        *,
        command_builder: Callable[[], list[str] | None],
        graceful_stop: Callable[[], str | None] | None = None,
        on_success: Callable[[list[str]], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.command_builder = command_builder
        self.graceful_stop = graceful_stop
        self.on_success = on_success
        self.process: QProcess | None = None
        self.current_cmd: list[str] | None = None
        self._epoch_total: int | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        button_row = QHBoxLayout()
        self.show_btn = QPushButton("Show command")
        self.show_btn.clicked.connect(self._show_command)
        button_row.addWidget(self.show_btn)
        self.start_btn = QPushButton("Start")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.clicked.connect(self._start)
        button_row.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop gracefully")
        self.stop_btn.clicked.connect(self._stop_gracefully)
        self.stop_btn.setEnabled(False)
        button_row.addWidget(self.stop_btn)
        self.kill_btn = QPushButton("Kill")
        self.kill_btn.clicked.connect(self._kill)
        self.kill_btn.setEnabled(False)
        button_row.addWidget(self.kill_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(190)
        self.log.setObjectName("ProcessLog")
        layout.addWidget(self.log)

        self.status = QLabel("Ready")
        self.status.setObjectName("HintLabel")
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setFormat("Ready")
        layout.addWidget(self.progress)

    def _command_or_none(self) -> list[str] | None:
        try:
            return self.command_builder()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid command", str(exc))
            return None

    def _show_command(self) -> None:
        cmd = self._command_or_none()
        if not cmd:
            return
        self._append_log("\n" + _display_command(cmd) + "\n")

    def _start(self) -> None:
        if self.process is not None and self.process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Process already running", "Wait for the current process to finish first.")
            return
        cmd = self._command_or_none()
        if not cmd:
            return
        program = str(cmd[0])
        args = [str(part) for part in cmd[1:]]
        self.current_cmd = [str(part) for part in cmd]
        self._epoch_total = self._cmd_int_value(self.current_cmd, "--epochs")
        self.log.clear()
        self._append_log(_display_command(cmd) + "\n\n")
        process = QProcess(self)
        process.setWorkingDirectory(str(REPO_ROOT))
        process.setProcessChannelMode(QProcess.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        process.setProcessEnvironment(env)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._error)
        self.process = process
        self._set_running(True)
        self.status.setText("Running")
        self.progress.setRange(0, 0)
        self.progress.setFormat("Running...")
        process.start(program, args)
        if not process.waitForStarted(3000):
            self.status.setText("Failed to start")
            self._set_running(False)
            self.progress.setRange(0, 1)
            self.progress.setValue(0)
            self.progress.setFormat("Failed to start")

    def _read_output(self) -> None:
        if self.process is None:
            return
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        self._append_log(data)
        self._update_progress_from_text(data)

    def _finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        self._append_log(f"\n[process exited with code {exit_code}]\n")
        self.status.setText("Finished" if exit_code == 0 else f"Exited with code {exit_code}")
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if exit_code == 0 else 0)
        self.progress.setFormat("Finished" if exit_code == 0 else "Failed")
        self._set_running(False)
        completed_cmd = self.current_cmd
        self.process = None
        self.current_cmd = None
        if exit_code == 0 and completed_cmd is not None and self.on_success is not None:
            try:
                self.on_success(completed_cmd)
            except Exception as exc:
                self._append_log(f"\n[post-run path update failed: {exc}]\n")

    def _error(self, error: QProcess.ProcessError) -> None:
        self.status.setText(f"Process error: {error.name}")

    def _stop_gracefully(self) -> None:
        if self.process is None or self.process.state() == QProcess.NotRunning:
            return
        message: str | None = None
        if self.graceful_stop is not None:
            message = self.graceful_stop()
        if message:
            self._append_log(f"\n[{message}]\n")
            return
        self.process.terminate()
        self._append_log("\n[terminate requested]\n")

    def _kill(self) -> None:
        if self.process is None or self.process.state() == QProcess.NotRunning:
            return
        self.process.kill()
        self._append_log("\n[kill requested]\n")

    def _append_log(self, text: str) -> None:
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(text)
        self.log.moveCursor(QTextCursor.End)

    def _set_running(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.kill_btn.setEnabled(running)

    def _update_progress_from_text(self, text: str) -> None:
        if not self._epoch_total:
            return
        matches = re.findall(r"\bepoch\s+(\d+)\b", text, flags=re.IGNORECASE)
        if not matches:
            return
        epoch = max(int(value) for value in matches)
        total = max(1, int(self._epoch_total))
        epoch = min(epoch, total)
        self.progress.setRange(0, total)
        self.progress.setValue(epoch)
        self.progress.setFormat(f"Epoch {epoch}/{total}")

    @staticmethod
    def _cmd_int_value(cmd: list[str], flag: str) -> int | None:
        try:
            idx = cmd.index(flag)
        except ValueError:
            return None
        if idx + 1 >= len(cmd):
            return None
        try:
            return int(cmd[idx + 1])
        except ValueError:
            return None


class WorkflowPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.form = QFormLayout()
        self.form.setLabelAlignment(self.form.labelAlignment())
        self.extra_args = QLineEdit()
        self.extra_args.setPlaceholderText("Optional raw CLI flags, e.g. --some_flag value")

    def _path_row(
        self,
        label: str,
        *,
        mode: str,
        file_filter: str = "All files (*.*)",
        save: bool = False,
        default: str = "",
    ) -> QLineEdit:
        edit = QLineEdit(default)
        button = QPushButton("Browse")

        def browse() -> None:
            if mode == "dir":
                path = QFileDialog.getExistingDirectory(self, f"Select {label}", edit.text().strip() or str(REPO_ROOT))
            elif save:
                path, _ = QFileDialog.getSaveFileName(self, f"Select {label}", edit.text().strip() or str(REPO_ROOT), file_filter)
            else:
                path, _ = QFileDialog.getOpenFileName(self, f"Select {label}", edit.text().strip() or str(REPO_ROOT), file_filter)
            if path:
                edit.setText(path)

        button.clicked.connect(browse)
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        self.form.addRow(label, row)
        return edit

    def _spin_row(self, label: str, value: int, minimum: int = 0, maximum: int = 100000) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        self.form.addRow(label, spin)
        return spin

    def _double_row(self, label: str, value: float, minimum: float = 0.0, maximum: float = 100000.0, decimals: int = 6) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSingleStep(0.01)
        self.form.addRow(label, spin)
        return spin

    def _combo_row(self, label: str, values: list[str], current: str) -> QComboBox:
        combo = QComboBox()
        combo.addItems(values)
        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        self.form.addRow(label, combo)
        return combo

    def _line_row(self, label: str, default: str = "") -> QLineEdit:
        edit = QLineEdit(default)
        self.form.addRow(label, edit)
        return edit

    def _checkbox_row(self, label: str, checked: bool = False) -> QCheckBox:
        check = QCheckBox(label)
        check.setChecked(checked)
        self.form.addRow("", check)
        return check

    def _wrap(self, title: str, runner: ProcessRunner) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(14, 14, 14, 14)
        header = QLabel(title)
        header.setObjectName("SectionHeader")
        layout.addWidget(header)
        layout.addLayout(self.form)
        self.form.addRow("Extra CLI flags", self.extra_args)
        layout.addWidget(runner)
        layout.addStretch(1)
        scroll.setWidget(inner)
        outer.addWidget(scroll)

    def _require(self, edit: QLineEdit, label: str) -> str:
        value = edit.text().strip()
        if not value:
            raise ValueError(f"{label} is required.")
        return value

    def _extra(self) -> list[str]:
        return _split_extra_args(self.extra_args.text())


_YOLO_WEIGHTS_TOOLTIP = (
    "YOLO-pose checkpoint (.pt) — used for both keypoint detection and visual feature extraction.\n"
    "\n"
    "• MARS top-down mice (the reference dataset):\n"
    "    mars_yolo_pose/runs/pose/train/weights/best.pt\n"
    "  — or —  download a pre-trained MARS YOLO-pose checkpoint from the project release.\n"
    "\n"
    "• Your own dataset / other species:\n"
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
        self.yolo_weights.setPlaceholderText("Required — see README 'Getting YOLO weights'")
        self.yolo_weights.setToolTip(_YOLO_WEIGHTS_TOOLTIP)
        self.output_root = self._path_row("Output dataset root", mode="dir")
        self.manifest_path = self._path_row("Manifest path", mode="file", file_filter="JSON (*.json);;All files (*.*)", save=True)
        self.clip_metadata_csv = self._path_row("clips_metadata.csv", mode="file", file_filter="CSV (*.csv);;All files (*.*)")
        self.window_size = self._spin_row("Window size", 32, 1)
        self.window_stride = self._spin_row("Window stride", 16, 1)
        self.n_animals = self._spin_row("Animals", 2, 1)
        self.crop_size = self._spin_row("Animal crop size", 224, 16)
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
        self._wrap("Prepare class-folder clips into BehaviorScope-Y windows", runner)

    def build_command(self) -> list[str]:
        script = _script_path("prepare_clips_y.py")
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
        self.n_animals = self._spin_row("Animals", 2, 1)
        self.crop_size = self._spin_row("Animal crop size", 224, 16)
        self.yolo_imgsz = self._spin_row("YOLO image size", 640, 32)
        self.yolo_batch = self._spin_row("YOLO batch", 64, 1)
        self.npz_writers = self._spin_row("NPZ writers", 4, 1)
        self.npz_compresslevel = self._spin_row("NPZ deflate level", 1, 0, 9)
        self.label_min_dominance = self._double_row("Label min dominance", 0.5, 0.0, 1.0, 3)
        self.other_subsample = self._double_row("Other subsample", 0.3, 0.0, 1.0, 3)
        self.device = self._line_row("Device", "cuda:0")
        self.skip_existing = self._checkbox_row("Resume completed videos", True)
        self.validate_manifest = self._checkbox_row("Validate manifest after build", True)
        self.validation_sample_limit = self._spin_row("Validation sample limit (0 = all)", 1000, 0)
        runner = ProcessRunner(command_builder=self.build_command, on_success=on_success)
        self._wrap("Prepare full-video annotations into BehaviorScope-Y windows", runner)

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
            "--n_animals",
            str(self.n_animals.value()),
            "--yolo_imgsz",
            str(self.yolo_imgsz.value()),
            "--yolo_batch",
            str(self.yolo_batch.value()),
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
        self.yolo_backbone_end_layer = self._spin_row("YOLO backbone end layer", 10, 1)
        self.n_animals = self._spin_row("Animals", 2, 1)
        self.num_keypoints = self._spin_row("Keypoints (0 = auto)", 0, 0)
        self.overwrite = self._checkbox_row("Overwrite existing cache entries", False)
        self.skip_invalid_samples = self._checkbox_row("Skip invalid samples", False)
        self.amp = self._checkbox_row("AMP", True)
        runner = ProcessRunner(command_builder=self.build_command, on_success=on_success)
        self._wrap("Build or refresh YOLO visual feature cache", runner)

    def build_command(self) -> list[str]:
        script = _script_path("precompute_visual_features_y.py")
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
        cmd.extend(self._extra())
        return cmd


class TrainPanel(WorkflowPanel):
    def __init__(self, parent: QWidget | None = None, on_success: Callable[[list[str]], None] | None = None) -> None:
        super().__init__(parent)
        self.manifest_path = self._path_row("Manifest path", mode="file", file_filter="JSON (*.json);;All files (*.*)")
        self.yolo_weights = self._path_row("YOLO pose weights", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)")
        self.yolo_weights.setPlaceholderText("Required — see README 'Getting YOLO weights'")
        self.yolo_weights.setToolTip(_YOLO_WEIGHTS_TOOLTIP)
        self.project = self._path_row("Run project folder", mode="dir", default=str(THIS_DIR / "runs"))
        self.name = self._line_row("Run name", "behaviorscope_y_run")
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
        self.train_sampler = self._combo_row("Train sampler", ["random", "weighted"], "random")
        self.n_animals = self._spin_row("Animals", 2, 1)
        self.num_keypoints = self._spin_row("Keypoints (0 = auto)", 0, 0)
        self.hidden_dim = self._spin_row("Hidden dim", 256, 16)
        self.num_lstm_layers = self._spin_row("LSTM layers", 1, 1)
        self.bidirectional_lstm = self._checkbox_row("Bidirectional LSTM", False)
        self.pose_fusion_dim = self._spin_row("Pose fusion dim", 128, 16)
        self.pose_fusion_strategy = self._combo_row("Pose fusion", ["gated_attention", "concat", "cross_modal_transformer"], "gated_attention")
        self.sequence_model = self._combo_row("Sequence model", ["attention", "lstm"], "attention")
        self.use_attention_pool = self._checkbox_row("Attention pool after LSTM", False)
        self.attention_heads = self._spin_row("Attention heads", 4, 1)
        self.positional_encoding = self._combo_row("Positional encoding", ["sinusoidal", "learned", "none"], "sinusoidal")
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
        self.per_class_metrics = self._checkbox_row("Per-class metrics", True)
        self.confusion_matrix = self._checkbox_row("Confusion matrix", True)
        self.disable_threshold_decoder = self._checkbox_row("Disable auto threshold decoder", False)
        self.export_single_model = self._checkbox_row("Export single bundled .pt after training", True)
        self.single_model_path = self._path_row("Bundled .pt output", mode="file", file_filter="PyTorch (*.pt);;All files (*.*)", save=True)
        # Temporal splitter — manuscript-grade GT fitting
        self.temporal_splitter_annot_root = self._path_row(
            "Splitter annot root (optional)",
            mode="dir",
            default="",
        )
        self.temporal_splitter_strict_annot = self._checkbox_row(
            "Strict annot mode (skip videos missing .annot)", False
        )
        self.disable_temporal_splitter = self._checkbox_row("Disable temporal splitter", False)
        runner = ProcessRunner(
            command_builder=self.build_command,
            graceful_stop=self.request_stop_after_epoch,
            on_success=on_success,
        )
        self._wrap("Train BehaviorScope-Y classifier", runner)

    def build_command(self) -> list[str]:
        script = _script_path("train_y.py")
        if not script.exists():
            raise ValueError(f"Missing script: {script}")
        cmd = [
            *_python_invocation(),
            str(script),
            "--manifest_path",
            self._require(self.manifest_path, "Manifest path"),
            "--yolo_weights",
            self._require(self.yolo_weights, "YOLO pose weights"),
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
        if self.per_class_metrics.isChecked():
            cmd.append("--per_class_metrics")
        if self.confusion_matrix.isChecked():
            cmd.extend(["--confusion_matrix", "--confusion_matrix_interval", "1"])
        if self.disable_threshold_decoder.isChecked():
            cmd.append("--disable_threshold_decoder")
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
        self.num_frames = self._spin_row("Window frames (0 = model config)", 0, 0)
        self.window_stride = self._spin_row("Window stride (0 = model config)", 0, 0)
        self.temporal_smoothing_window = self._spin_row("Median smoothing window", 0, 0)
        self.bout_min_duration = self._spin_row("Bout min duration frames", 0, 0)
        self.csv_flush_interval = self._spin_row("CSV flush interval", 50, 1)
        self.smooth_bouts = self._checkbox_row("Smooth bouts for MP4/frame CSV", True)
        self.log_metrics = self._checkbox_row("Log runtime metrics", True)
        self.amp = self._checkbox_row("Inference AMP", True)
        self.export_pose = self._checkbox_row("Export pose CSV (keypoints + bbox per frame)", False)
        self.no_bboxes = self._checkbox_row("Hide boxes in review MP4", False)
        self.no_keypoints = self._checkbox_row("Hide keypoints in review MP4", False)
        runner = ProcessRunner(command_builder=self.build_command, graceful_stop=self.request_stop)
        title = "Batch process a folder with BehaviorScope-Y" if batch_mode else "Run BehaviorScope-Y inference"
        self._wrap(title, runner)

    def build_command(self) -> list[str]:
        script = _script_path("infer_y.py")
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
        if self.output.text().strip() and not self.batch_mode:
            cmd.extend(["--output", self.output.text().strip()])
        if self.output_dir.text().strip():
            cmd.extend(["--output_dir", self.output_dir.text().strip()])
        if self.output_video.text().strip() and not self.batch_mode:
            cmd.extend(["--output_video", self.output_video.text().strip()])
        if self.output_video_dir.text().strip():
            cmd.extend(["--output_video_dir", self.output_video_dir.text().strip()])
        if self.smooth_bouts.isChecked():
            cmd.append("--smooth_bouts")
        if self.log_metrics.isChecked():
            cmd.append("--log_metrics")
        if self.amp.isChecked():
            cmd.append("--amp")
        if self.export_pose.isChecked():
            cmd.append("--export_pose")
        if self.no_bboxes.isChecked():
            cmd.append("--no_bboxes")
        if self.no_keypoints.isChecked():
            cmd.append("--no_keypoints")
        stop_flag = self._stop_flag_path()
        cmd.extend(["--stop_flag_file", str(stop_flag)])
        cmd.extend(self._extra())
        return cmd

    def _stop_flag_path(self) -> Path:
        base = Path(self.output_dir.text().strip() or self.output.text().strip() or REPO_ROOT / ".behaviorscope_y")
        if base.suffix:
            base = base.parent
        base.mkdir(parents=True, exist_ok=True)
        return base / "stop_inference.flag"

    def request_stop(self) -> str:
        flag = self._stop_flag_path()
        flag.write_text("stop inference\n", encoding="utf-8")
        return f"requested graceful inference stop via {flag}"
