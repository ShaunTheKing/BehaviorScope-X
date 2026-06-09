from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QLabel, QWidget

from .workflow_base import REPO_ROOT, ProcessRunner, WorkflowPanel, python_invocation as _python_invocation

class ReleaseStagePanel(WorkflowPanel):
    def __init__(
        self,
        *,
        title: str,
        description: str,
        runner_script: Path,
        stage: str,
        workflow_kind: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.stage = stage
        self.workflow_kind = workflow_kind
        self.description = description
        self.runner_script = self._path_row(
            "Workflow runner",
            mode="file",
            file_filter="Python (*.py);;All files (*.*)",
            default=str(runner_script),
        )
        self.action = self._combo_row("Action", ["plan", "run"], "plan")
        self.action.setToolTip("Use plan to inspect commands without launching long jobs. Use run when paths and settings are correct.")
        self.device = self._line_row("Device override", "")
        self.device.setPlaceholderText("Optional, for example cuda:0 or cpu")
        self.device.setToolTip("Overrides the device configured for this workflow stage.")
        self.skip_completed = self._checkbox_row("Skip completed stages", True)
        self.skip_completed.setToolTip("Reuse existing completion markers and outputs when the stage has already finished.")
        self.runner_script.setToolTip("Python entry point for this model-family workflow.")

        if workflow_kind == "dlc":
            self.config_path = self._path_row(
                "DLC workflow config",
                mode="file",
                file_filter="JSON (*.json);;All files (*.*)",
                default=str(runner_script.parent / "final_config.json"),
            )
            self.config_path.setToolTip("DLC workflow configuration with project paths, checkpoints, cache roots, and evaluation outputs.")
        elif workflow_kind == "mobilenetv3":
            self.suite_root = self._path_row(
                "Suite output root",
                mode="dir",
                default=str(REPO_ROOT / "outputs" / "controlled_comparison_runs" / "release_mobilenetv3_backbone"),
            )
            self.suite_root.setToolTip("Root folder for MobileNetV3 cache, training, evaluation, and summary outputs.")
            self.seeds = self._line_row("Seeds", "42 43 44")
            self.seeds.setToolTip("Space- or comma-separated random seeds for repeated classifier runs.")
            self.heads = self._line_row("Temporal heads", "lstm attention")
            self.heads.setToolTip("Temporal classifier families to train or evaluate.")
            self.model_ids = self._line_row("Model IDs", "full_lstm256")
            self.model_ids.setToolTip("Model identifiers from the controlled-comparison configuration.")
            self.cache_batch = self._spin_row("Feature cache batch", 1, 1)
            self.cache_batch.setToolTip("Batch size for MobileNetV3 feature extraction. Increase only if GPU memory allows.")
            self.cache_num_workers = self._spin_row("Feature cache workers", 2, 0)
            self.cache_num_workers.setToolTip("CPU worker count for data loading during feature-cache construction.")
            self.cache_dtype = self._combo_row("Feature cache dtype", ["float32", "float16"], "float32")
            self.cache_dtype.setToolTip("Numeric dtype used to store cached visual descriptors.")
            self.resize_size = self._spin_row("Resize size", 640, 32)
            self.resize_size.setToolTip("Input resize used before MobileNetV3 pose-backbone feature extraction.")
            self.rebuild_npz = self._checkbox_row("Include train/held-out NPZ rebuilds", stage == "build_npz")
            self.rebuild_npz.setToolTip("Rebuild sequence NPZ caches instead of relying only on existing manifests.")
            self.keep_going = self._checkbox_row("Keep going after failed optional commands", True)
            self.keep_going.setToolTip("Continue through optional summary or evaluation commands when a non-critical command fails.")
            self.resource_sample_interval_s = self._double_row("Resource sample interval s", 30.0, 1.0, 3600.0, 1)
            self.resource_sample_interval_s.setToolTip("Interval for GPU and CPU resource logging during long MobileNetV3 stages.")

        runner = ProcessRunner(command_builder=self.build_command)
        self._wrap(title, runner)
        note = QLabel(description)
        note.setWordWrap(True)
        note.setObjectName("HintLabel")
        self.layout().itemAt(0).widget().widget().layout().insertWidget(1, note)

    def build_command(self) -> list[str]:
        runner = Path(self._require(self.runner_script, "Workflow runner"))
        if not runner.exists():
            raise ValueError(f"Missing workflow runner: {runner}")
        cmd = [
            *_python_invocation(),
            str(runner),
            self.action.currentText(),
            "--stage",
            self.stage,
        ]
        device = self.device.text().strip()
        if device:
            cmd.extend(["--device", device])
        if self.skip_completed.isChecked():
            cmd.append("--skip_completed")
        if self.workflow_kind == "dlc":
            config = self.config_path.text().strip()
            if config:
                cmd.extend(["--config", config])
        elif self.workflow_kind == "mobilenetv3":
            suite_root = self.suite_root.text().strip()
            if suite_root:
                cmd.extend(["--suite_root", suite_root])
            seeds = [part for part in self.seeds.text().replace(",", " ").split() if part]
            if seeds:
                cmd.extend(["--seeds", *seeds])
            heads = [part for part in self.heads.text().replace(",", " ").split() if part]
            if heads:
                cmd.extend(["--heads", *heads])
            model_ids = [part for part in self.model_ids.text().replace(",", " ").split() if part]
            if model_ids:
                cmd.extend(["--model_ids", *model_ids])
            cmd.extend(["--cache_batch", str(self.cache_batch.value())])
            cmd.extend(["--cache_num_workers", str(self.cache_num_workers.value())])
            cmd.extend(["--cache_dtype", self.cache_dtype.currentText()])
            cmd.extend(["--resize_size", str(self.resize_size.value())])
            cmd.extend(["--resource_sample_interval_s", str(self.resource_sample_interval_s.value())])
            if self.rebuild_npz.isChecked():
                cmd.append("--rebuild_npz")
            if self.keep_going.isChecked():
                cmd.append("--keep_going")
        cmd.extend(self._extra())
        return cmd


