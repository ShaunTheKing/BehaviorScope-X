from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication, QMessageBox

from annotation_app.main_window import AnnotationMainWindow
from annotation_app.store import AnnotationStore


def _warn_missing_deps(window: AnnotationMainWindow) -> None:
    """Show non-blocking startup warnings for missing optional system dependencies."""
    missing: list[str] = []
    if shutil.which("ffmpeg") is None:
        missing.append(
            "• ffmpeg — needed by the 'Annotate + Clip' tab to extract video clips.\n"
            "  Install: https://ffmpeg.org/download.html\n"
            "  Windows quick install:  winget install ffmpeg\n"
            "              or:  choco install ffmpeg\n"
            "  After installing, restart BehaviorScope-Y."
        )
    if missing:
        body = (
            "The following system tools were not found on your PATH:\n\n"
            + "\n\n".join(missing)
            + "\n\nThe rest of the app (Train / Inference / Batch) works without them."
        )
        QMessageBox.warning(window, "Missing system dependencies", body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "BehaviorScope-Y annotation workspace for clipping behavior bouts "
            "into class-folder datasets."
        )
    )
    parser.add_argument(
        "--project_db",
        type=Path,
        default=Path.cwd() / ".behaviorscope_y" / "annotation_workspace.sqlite",
        help="Path to the SQLite annotation project database.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(sys.argv)
    app.setApplicationName("BehaviorScope-Y Annotation Workspace")
    store = AnnotationStore(args.project_db)
    app.aboutToQuit.connect(store.close)
    project = store.get_or_create_default_project("BehaviorScope-Y Annotation Project")
    window = AnnotationMainWindow(store, project)
    window.show()
    # Check for missing system dependencies after the event loop is running
    # so the window is fully painted before the dialog appears.
    QTimer.singleShot(0, lambda: _warn_missing_deps(window))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
