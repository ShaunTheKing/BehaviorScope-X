from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app_metadata import APP_NAME
from .models import BehaviorRecord, HotkeyBinding
from .store import PROJECT_METADATA_FIELDS, default_color

class HotkeyDialog(QDialog):
    def __init__(self, bindings: list[HotkeyBinding], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Hotkeys")
        self.resize(540, 420)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(len(bindings), 2, self)
        self.table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        for row, binding in enumerate(bindings):
            action_item = QTableWidgetItem(binding.action.replace("_", " ").title())
            action_item.setData(Qt.UserRole, binding.action)
            self.table.setItem(row, 0, action_item)
            editor = QLineEdit(binding.key_sequence)
            self.table.setCellWidget(row, 1, editor)
        layout.addWidget(self.table)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def bindings(self) -> list[HotkeyBinding]:
        result: list[HotkeyBinding] = []
        for row in range(self.table.rowCount()):
            action_item = self.table.item(row, 0)
            editor = self.table.cellWidget(row, 1)
            if action_item is None or not isinstance(editor, QLineEdit):
                continue
            result.append(
                HotkeyBinding(
                    action=str(action_item.data(Qt.UserRole)),
                    key_sequence=editor.text().strip(),
                )
            )
        return result


class HotkeyMapDialog(QDialog):
    def __init__(
        self,
        bindings: list[HotkeyBinding],
        behaviors: list[BehaviorRecord],
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Hotkey Map")
        self.resize(620, 520)
        layout = QVBoxLayout(self)

        header = QLabel(
            "Quick reference for transport, annotation, review, and behavior selection."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self.table = QTableWidget(0, 2, self)
        self.table.setHorizontalHeaderLabels(["Action", "Shortcut"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.NoSelection)
        self.table.setFocusPolicy(Qt.NoFocus)
        layout.addWidget(self.table, 1)

        rows: list[tuple[str, str]] = []
        for binding in sorted(bindings, key=lambda item: item.action):
            rows.append((binding.action.replace("_", " ").title(), binding.key_sequence))
        for behavior in behaviors:
            if behavior.hotkey:
                rows.append((f"Select behavior: {behavior.name}", behavior.hotkey))

        self.table.setRowCount(len(rows))
        for row, (action, key) in enumerate(rows):
            self.table.setItem(row, 0, QTableWidgetItem(action))
            self.table.setItem(row, 1, QTableWidgetItem(key))

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class BehaviorManagerDialog(QDialog):
    def __init__(self, behaviors: list[BehaviorRecord], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Behaviors")
        self.resize(760, 480)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(len(behaviors), 4, self)
        self.table.setHorizontalHeaderLabels(["Name", "Definition", "Color", "Hotkey"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.AllEditTriggers)
        for row, behavior in enumerate(behaviors):
            name_item = QTableWidgetItem(behavior.name)
            name_item.setData(Qt.UserRole, behavior.id)
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, QTableWidgetItem(behavior.definition))
            color_btn = QPushButton(behavior.color)
            color_btn.setProperty("behaviorColor", behavior.color)
            color_btn.setStyleSheet(f"background:{behavior.color}; color:white; border-radius:6px; padding:4px 10px;")
            color_btn.clicked.connect(lambda _=False, btn=color_btn: self._pick_color(btn))
            self.table.setCellWidget(row, 2, color_btn)
            hotkey_edit = QLineEdit(behavior.hotkey or "")
            self.table.setCellWidget(row, 3, hotkey_edit)
        layout.addWidget(self.table)

        button_row = QHBoxLayout()
        add_btn = QPushButton("Add behavior")
        add_btn.clicked.connect(self._add_row)
        button_row.addWidget(add_btn)
        remove_btn = QPushButton("Remove selected")
        remove_btn.clicked.connect(self._remove_selected_rows)
        button_row.addWidget(remove_btn)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick_color(self, button: QPushButton) -> None:
        current = QColor(button.property("behaviorColor") or "#378ADD")
        color = QColorDialog.getColor(current, self, "Select behavior color")
        if not color.isValid():
            return
        value = color.name().upper()
        button.setProperty("behaviorColor", value)
        button.setText(value)
        button.setStyleSheet(f"background:{value}; color:white; border-radius:6px; padding:4px 10px;")

    def _add_row(self) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        name_item = QTableWidgetItem(f"Behavior {row + 1}")
        name_item.setData(Qt.UserRole, None)
        self.table.setItem(row, 0, name_item)
        self.table.setItem(row, 1, QTableWidgetItem(""))
        color = default_color(row)
        color_btn = QPushButton(color)
        color_btn.setProperty("behaviorColor", color)
        color_btn.setStyleSheet(f"background:{color}; color:white; border-radius:6px; padding:4px 10px;")
        color_btn.clicked.connect(lambda _=False, btn=color_btn: self._pick_color(btn))
        self.table.setCellWidget(row, 2, color_btn)
        self.table.setCellWidget(row, 3, QLineEdit(""))
        self.table.setCurrentCell(row, 0)
        self.table.editItem(name_item)

    def _remove_selected_rows(self) -> None:
        selected = sorted({index.row() for index in self.table.selectionModel().selectedRows()}, reverse=True)
        for row in selected:
            self.table.removeRow(row)

    def rows(self) -> list[dict]:
        out: list[dict] = []
        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            definition_item = self.table.item(row, 1)
            color_btn = self.table.cellWidget(row, 2)
            hotkey_edit = self.table.cellWidget(row, 3)
            if name_item is None or not isinstance(color_btn, QPushButton) or not isinstance(hotkey_edit, QLineEdit):
                continue
            name = name_item.text().strip()
            if not name:
                continue
            out.append(
                {
                    "id": name_item.data(Qt.UserRole),
                    "name": name,
                    "definition": "" if definition_item is None else definition_item.text().strip(),
                    "color": str(color_btn.property("behaviorColor") or default_color(row)),
                    "hotkey": hotkey_edit.text().strip() or None,
                }
            )
        return out


class ProjectMetadataDialog(QDialog):
    def __init__(self, metadata: dict[str, str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Project metadata")
        self.resize(720, 560)
        layout = QVBoxLayout(self)
        self.form = QFormLayout()
        self.editors: dict[str, QWidget] = {}

        for key, label in PROJECT_METADATA_FIELDS:
            if key in {"project_summary", "peer_review_notes"}:
                editor = QPlainTextEdit()
                editor.setPlainText(metadata.get(key, ""))
                editor.setMinimumHeight(92)
            else:
                editor = QLineEdit(metadata.get(key, ""))
            self.editors[key] = editor
            self.form.addRow(label, editor)

        form_card = QWidget()
        form_card.setLayout(self.form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(form_card)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def values(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for key, editor in self.editors.items():
            if isinstance(editor, QPlainTextEdit):
                out[key] = editor.toPlainText().strip()
            elif isinstance(editor, QLineEdit):
                out[key] = editor.text().strip()
        return out


class BundleExistingModelDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Bundle Existing Model")
        self.resize(760, 260)
        layout = QVBoxLayout(self)
        self.form = QFormLayout()
        layout.addLayout(self.form)

        self.classifier_checkpoint = self._path_row(
            "Classifier checkpoint",
            "PyTorch (*.pt);;All files (*.*)",
            save=False,
        )
        self.model_config = self._path_row(
            "Model config.json",
            "JSON (*.json);;All files (*.*)",
            save=False,
        )
        self.yolo_weights = self._path_row(
            "YOLO pose weights",
            "PyTorch (*.pt);;All files (*.*)",
            save=False,
        )
        self.output_path = self._path_row(
            "Bundled output .pt",
            "PyTorch (*.pt);;All files (*.*)",
            save=True,
        )

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=self)
        buttons.button(QDialogButtonBox.Ok).setText("Bundle")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _path_row(self, label: str, file_filter: str, *, save: bool) -> QLineEdit:
        edit = QLineEdit()
        button = QPushButton("Browse")

        def browse() -> None:
            start = edit.text().strip() or str(Path.cwd())
            if save:
                path, _ = QFileDialog.getSaveFileName(self, f"Select {label}", start, file_filter)
            else:
                path, _ = QFileDialog.getOpenFileName(self, f"Select {label}", start, file_filter)
            if path:
                edit.setText(path)

        button.clicked.connect(browse)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(edit, 1)
        row_layout.addWidget(button)
        self.form.addRow(label, row)
        return edit

    def values(self) -> dict[str, str]:
        return {
            "classifier_checkpoint": self.classifier_checkpoint.text().strip(),
            "model_config": self.model_config.text().strip(),
            "yolo_weights": self.yolo_weights.text().strip(),
            "output": self.output_path.text().strip(),
        }


class WorkflowGuideDialog(QDialog):
    def __init__(self, *, show_on_startup: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} Workflow Guide")
        self.resize(760, 620)
        layout = QVBoxLayout(self)
        header = QLabel("End-to-end GUI workflow")
        header.setObjectName("SectionHeader")
        layout.addWidget(header)

        intro = QLabel(
            "For a first run, import licensed videos, define behavior labels, assign splits, and export full-video annotations. "
            "Then move to the model-family tab that matches the pose model you want to use. "
            f"{APP_NAME} follows a pose-model-flexible amortized-pose-vision design: YOLO-pose, MobileNetV3, and DeepLabCut-HRNet are independent validated workflows that produce compatible caches, classifiers, evaluations, and output summaries."
        )
        intro.setWordWrap(True)
        intro.setObjectName("HintLabel")
        layout.addWidget(intro)

        steps = [
            ("1. Import videos", "File > Import videos... or File > Import folder..."),
            ("2. Assign video splits", "Project > Assign selected videos to Train, Validation, Held-out Test, or Exclude."),
            ("3. Annotate and approve spans", "Use Annotate + Clip. Draft while labeling, mark spans Ready for checking, then Approve accepted bouts."),
            ("4. Export full-video annotations", "Project > Export full-video annotations... writes source_manifest.csv and .annot files."),
            ("5. Choose pose workflow", "Select YOLO-pose, MobileNetV3, or DeepLabCut-HRNet according to the checkpoint/project you want to bring."),
            ("6. Build sequence cache", "Use that model family's full-video cache stage to create compatible sliding-window NPZs and a sequence_manifest.json."),
            ("7. Build feature cache", "Extract pose-backbone visual descriptors from the same pose workflow before classifier training."),
            ("8. Train classifier", "Use the model family's training stage to fit the temporal behavior classifier and evaluate validation performance."),
            ("9. Evaluate or infer", "Run held-out evaluation when labels are available, or use inference/batch tools for deployment paths that support bundled inference."),
            ("10. Create ethograms", "Use each model family's Ethograms + Bouts tab to convert temporal prediction CSVs into ethogram timelines and bout summaries."),
            ("11. Inspect outputs", "Use each model family's Outputs tab for caches, logs, confusion matrices, ethograms, bout summaries, and exports."),
        ]
        table = QTableWidget(len(steps), 2, self)
        table.setHorizontalHeaderLabels(["Step", "What to do"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)
        for row, (step, detail) in enumerate(steps):
            table.setItem(row, 0, QTableWidgetItem(step))
            table.setItem(row, 1, QTableWidgetItem(detail))
        table.resizeRowsToContents()
        layout.addWidget(table, 1)

        self.show_on_startup = QCheckBox("Show this guide when opening a project")
        self.show_on_startup.setChecked(bool(show_on_startup))
        layout.addWidget(self.show_on_startup)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

    def should_show_on_startup(self) -> bool:
        return self.show_on_startup.isChecked()


class ScientificWorkflowDialog(QDialog):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle(f"{APP_NAME} Scientific Workflow")
        self.resize(780, 560)
        layout = QVBoxLayout(self)
        header = QLabel("Manuscript-aligned workflow")
        header.setObjectName("SectionHeader")
        layout.addWidget(header)

        intro = QLabel(
            f"{APP_NAME} operationalizes amortized pose vision: a validated pose model is reused as both a "
            "keypoint source and a frozen visual-representation source for full-video computational ethology."
        )
        intro.setWordWrap(True)
        intro.setObjectName("HintLabel")
        layout.addWidget(intro)

        rows = [
            (
                "1. Approved annotation export",
                "Only approved spans are exported for full-video training. The export writes BENTO .annot files, class names, and source_manifest.csv with split assignments.",
            ),
            (
                "2. Pose-model routes",
                "YOLO-pose, MobileNetV3, and DeepLabCut-HRNet have backend-specific pose inference and feature-tap logic but feed compatible cache, classifier, and evaluation structures.",
            ),
            (
                "3. Cached multimodal windows",
                "Sequence caches preserve detections, keypoints, pose-derived geometry, crop metadata, visual descriptors, labels, source videos, and train/validation/held-out split provenance.",
            ),
            (
                "4. Temporal decoding",
                "The downstream classifier is the temporal decoder. Cached visual descriptors and pose-derived social geometry are kept separate enough to support stream ablations and controlled comparisons.",
            ),
            (
                "5. Frame, bout, and ethogram readouts",
                "Outputs are organized around validation metrics, held-out tables, confusion matrices, bout summaries, and ethogram exports so frame accuracy is not treated as the only scientific endpoint.",
            ),
            (
                "6. Reproducible execution",
                "Workflow panels expose Show command before Start stage. The displayed command is the CLI invocation used by the underlying script, making GUI runs reproducible from the terminal.",
            ),
            (
                "7. Deployment boundary",
                "Single-file bundled inference is currently a YOLO-backed deployment path. MobileNetV3 and DeepLabCut-HRNet preserve checkpoint, project, and feature-cache provenance for staged evaluation.",
            ),
        ]
        table = QTableWidget(len(rows), 2, self)
        table.setHorizontalHeaderLabels(["Principle", "GUI implementation"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)
        for row, (principle, detail) in enumerate(rows):
            table.setItem(row, 0, QTableWidgetItem(principle))
            table.setItem(row, 1, QTableWidgetItem(detail))
        table.resizeRowsToContents()
        layout.addWidget(table, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)

