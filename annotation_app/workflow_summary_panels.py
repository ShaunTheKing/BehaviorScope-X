from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel, QWidget

from .workflow_base import ProcessRunner, WorkflowPanel, python_invocation as _python_invocation, script_path as _script_path


class EthogramSummaryPanel(WorkflowPanel):
    def __init__(self, *, workflow_label: str, default_output_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.workflow_label = workflow_label
        self.predictions_csv = self._path_row(
            "Prediction CSV",
            mode="file",
            file_filter="CSV (*.csv);;All files (*.*)",
        )
        self.predictions_csv.setToolTip(
            "Prediction CSV from any temporal BehaviorScope-X model. For frame-exact ethograms, choose a *.smoothed_frames.csv file."
        )
        self.predictions_dir = self._path_row("Prediction folder", mode="dir")
        self.predictions_dir.setToolTip(
            "Optional folder to search recursively for prediction CSVs. Use this after batch inference or held-out evaluation."
        )
        self.pattern = self._line_row("Folder pattern", "*.smoothed_frames.csv")
        self.pattern.setToolTip("Glob pattern used when scanning the prediction folder.")
        self.output_dir = self._path_row(
            "Ethogram output folder",
            mode="dir",
            default=str(default_output_root / "ethograms"),
        )
        self.output_dir.setToolTip("Folder where ethogram_segments.csv, ethogram_behavior_summary.csv, and ethogram_summary.json are written.")
        self.fps = self._double_row("FPS", 30.0, 0.1, 1000.0, 3)
        self.fps.setToolTip("Frames per second used to convert frame indices into seconds.")
        self.background_label = self._line_row("Background label", "other")
        self.background_label.setToolTip("Label treated as background when background rows are excluded.")
        self.exclude_background = self._checkbox_row("Exclude background/other from outputs", False)
        self.exclude_background.setToolTip("Hide the background class in ethogram segments and behavior summaries.")

        runner = ProcessRunner(command_builder=self.build_command)
        self._wrap(f"Create ethograms and bout summaries for {workflow_label}", runner)
        note = QLabel(
            "Ethograms are generated from temporal behavior predictions, so this stage is model-family agnostic. "
            "Use smoothed frame CSVs for exact frame timelines; window CSVs are converted by probability averaging or label voting."
        )
        note.setWordWrap(True)
        note.setObjectName("HintLabel")
        self.layout().itemAt(0).widget().widget().layout().insertWidget(1, note)

    def build_command(self) -> list[str]:
        script = _script_path("make_ethogram_summary.py")
        if not script.exists():
            raise ValueError(f"Missing script: {script}")
        output_dir = self._require(self.output_dir, "Ethogram output folder")
        cmd = [
            *_python_invocation(),
            str(script),
            "--output_dir",
            output_dir,
            "--fps",
            str(self.fps.value()),
            "--background_label",
            self.background_label.text().strip() or "other",
        ]
        csv_path = self.predictions_csv.text().strip()
        if csv_path:
            cmd.extend(["--predictions_csv", csv_path])
        predictions_dir = self.predictions_dir.text().strip()
        if predictions_dir:
            cmd.extend(["--predictions_dir", predictions_dir, "--pattern", self.pattern.text().strip() or "*.smoothed_frames.csv"])
        if not csv_path and not predictions_dir:
            raise ValueError("Select a prediction CSV or prediction folder.")
        if self.exclude_background.isChecked():
            cmd.append("--exclude_background")
        cmd.extend(self._extra())
        return cmd

