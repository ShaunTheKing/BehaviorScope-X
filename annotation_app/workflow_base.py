from __future__ import annotations

import re
import shlex
import subprocess
import sys
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


def script_path(name: str) -> Path:
    return THIS_DIR / name


def split_extra_args(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return shlex.split(text, posix=False)


def display_command(cmd: list[str]) -> str:
    return subprocess.list2cmdline([str(part) for part in cmd])


def python_invocation() -> list[str]:
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
        self.show_btn.setToolTip("Preview the exact command and paths before starting a long run.")
        self.show_btn.clicked.connect(self._show_command)
        button_row.addWidget(self.show_btn)
        self.start_btn = QPushButton("Start stage")
        self.start_btn.setObjectName("PrimaryButton")
        self.start_btn.setToolTip("Run this stage with the current paths and settings. Use Show command first if you are unsure.")
        self.start_btn.clicked.connect(self._start)
        button_row.addWidget(self.start_btn)
        self.stop_btn = QPushButton("Stop after step")
        self.stop_btn.setToolTip("Ask the running process to stop cleanly when the script supports it.")
        self.stop_btn.clicked.connect(self._stop_gracefully)
        self.stop_btn.setEnabled(False)
        button_row.addWidget(self.stop_btn)
        self.kill_btn = QPushButton("Force stop")
        self.kill_btn.setToolTip("Terminate the process immediately. Use only when a clean stop is not possible.")
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
        self._append_log("\n" + display_command(cmd) + "\n")

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
        self._append_log(display_command(cmd) + "\n\n")
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
        self._form_fields: dict[QWidget, QWidget] = {}
        self.extra_args = QLineEdit()
        self.extra_args.setPlaceholderText("Optional advanced CLI flags, for example --seed 42")
        self.extra_args.setToolTip("Advanced command-line options appended exactly as entered. Leave empty for the recommended GUI defaults.")

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
        if mode == "dir":
            edit.setPlaceholderText("Browse or paste a folder path")
            kind = "folder"
        elif save:
            edit.setPlaceholderText("Browse or paste an output file path")
            kind = "output file"
        else:
            edit.setPlaceholderText("Browse or paste a file path")
            kind = "file"
        edit.setToolTip(f"{label}: select or paste the {kind} used by this workflow stage.")
        button = QPushButton("Browse")
        button.setToolTip(f"Open a picker for {label}.")

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
        self._form_fields[edit] = row
        return edit

    def _spin_row(self, label: str, value: int, minimum: int = 0, maximum: int = 100000) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        self.form.addRow(label, spin)
        self._form_fields[spin] = spin
        return spin

    def _double_row(self, label: str, value: float, minimum: float = 0.0, maximum: float = 100000.0, decimals: int = 6) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(decimals)
        spin.setValue(value)
        spin.setSingleStep(0.01)
        self.form.addRow(label, spin)
        self._form_fields[spin] = spin
        return spin

    def _combo_row(self, label: str, values: list[str], current: str) -> QComboBox:
        combo = QComboBox()
        combo.addItems(values)
        idx = combo.findText(current)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        self.form.addRow(label, combo)
        self._form_fields[combo] = combo
        return combo

    def _line_row(self, label: str, default: str = "") -> QLineEdit:
        edit = QLineEdit(default)
        self.form.addRow(label, edit)
        self._form_fields[edit] = edit
        return edit

    def _checkbox_row(self, label: str, checked: bool = False) -> QCheckBox:
        check = QCheckBox(label)
        check.setChecked(checked)
        self.form.addRow("", check)
        self._form_fields[check] = check
        return check

    def _set_field_visible(self, field: QWidget, visible: bool) -> None:
        row_widget = self._form_fields.get(field, field)
        if hasattr(self.form, "setRowVisible"):
            self.form.setRowVisible(row_widget, visible)
        else:
            label = self.form.labelForField(row_widget)
            if label is not None:
                label.setVisible(visible)
            row_widget.setVisible(visible)

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
        return split_extra_args(self.extra_args.text())
