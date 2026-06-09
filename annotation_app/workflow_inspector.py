from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from .workflow_base import REPO_ROOT

class ArtifactInspectorPanel(QWidget):
    def __init__(self, *, title: str, default_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        header = QLabel(title)
        header.setObjectName("SectionHeader")
        layout.addWidget(header)
        hint = QLabel(
            "Inspect cached windows, feature caches, model outputs, evaluation tables, "
            "ethogram CSVs, bout summaries, and keypoint/behavior exports under a workflow root."
        )
        hint.setWordWrap(True)
        hint.setObjectName("HintLabel")
        layout.addWidget(hint)
        form = QFormLayout()
        self.root = QLineEdit(str(default_root))
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(self.root, 1)
        row_layout.addWidget(browse)
        form.addRow("Artifact root", row)
        self.patterns = QLineEdit("*.json *.csv *.txt *.pt *.npz *.mp4")
        form.addRow("File patterns", self.patterns)
        layout.addLayout(form)
        refresh = QPushButton("Refresh inventory")
        refresh.setObjectName("PrimaryButton")
        refresh.clicked.connect(self.refresh)
        layout.addWidget(refresh)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(320)
        self.output.setObjectName("ProcessLog")
        layout.addWidget(self.output, 1)
        self.refresh()

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select artifact root", self.root.text().strip() or str(REPO_ROOT))
        if path:
            self.root.setText(path)
            self.refresh()

    def refresh(self) -> None:
        root = Path(self.root.text().strip() or ".")
        patterns = [part for part in self.patterns.text().split() if part]
        lines: list[str] = []
        if not root.exists():
            self.output.setPlainText(f"Missing artifact root:\n{root}")
            return
        lines.append(f"Root: {root}")
        total_files = 0
        total_bytes = 0
        matches: list[Path] = []
        for pattern in patterns:
            matches.extend(path for path in root.rglob(pattern) if path.is_file())
        for path in sorted(set(matches), key=lambda p: (str(p.parent), p.name)):
            total_files += 1
            try:
                total_bytes += path.stat().st_size
            except OSError:
                pass
        lines.append(f"Matched files: {total_files}")
        lines.append(f"Matched size: {total_bytes / (1024 ** 3):.3f} GiB")
        lines.append("")
        for path in sorted(set(matches), key=lambda p: str(p))[:500]:
            try:
                rel = path.relative_to(root)
            except ValueError:
                rel = path
            lines.append(str(rel))
        if total_files > 500:
            lines.append(f"\n... {total_files - 500} additional files not shown")
        self.output.setPlainText("\n".join(lines))


