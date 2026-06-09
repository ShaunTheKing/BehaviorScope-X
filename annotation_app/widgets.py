from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from .models import BehaviorRecord, VideoRecord
from .store import normalize_video_split


def format_ms(ms: int) -> str:
    total_seconds = max(0, int(round(ms / 1000.0)))
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes}:{seconds:02d}"


def color_swatch_icon(color: str, size: int = 14) -> QIcon:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    painter.drawRoundedRect(1, 1, size - 2, size - 2, 3, 3)
    painter.end()
    return QIcon(pixmap)


class VideoListItem(QWidget):
    def __init__(self, video: VideoRecord, selected: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)
        dot = QLabel()
        dot.setFixedSize(10, 10)
        dot.setStyleSheet(f"background:{self._dot_color(video, selected)}; border-radius:5px;")
        layout.addWidget(dot)

        text_col = QVBoxLayout()
        text_col.setContentsMargins(0, 0, 0, 0)
        name = QLabel(video.filename)
        name.setObjectName("videoName")
        counts = QLabel(f"{video.annotation_count} spans  |  {format_ms(video.duration_ms)}")
        counts.setObjectName("videoMeta")
        text_col.addWidget(name)
        text_col.addWidget(counts)
        layout.addLayout(text_col, 1)

        split = normalize_video_split(getattr(video, "split", "train"))
        split_badge = QLabel(split.upper())
        split_badge.setObjectName("SplitBadge")
        split_badge.setStyleSheet(
            f"background:{self._split_color(split)}; color:#EAF0F6; "
            "border-radius:6px; padding:3px 6px; font-size:10px; font-weight:700;"
        )
        layout.addWidget(split_badge)

    @staticmethod
    def _dot_color(video: VideoRecord, selected: bool) -> str:
        if selected:
            return "#3E8EDE"
        if video.approved_count > 0:
            return "#2EA96B"
        if video.annotation_count > 0:
            return "#D58A2D"
        return "#76889A"

    @staticmethod
    def _split_color(split: str) -> str:
        return {
            "train": "#245C3A",
            "val": "#255A78",
            "test": "#6B4F22",
            "exclude": "#4B5563",
        }.get(split, "#4B5563")


class BehaviorButton(QPushButton):
    def __init__(self, behavior: BehaviorRecord, selected: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.behavior = behavior
        self.setCheckable(True)
        self.setChecked(selected)
        label = behavior.name if not behavior.hotkey else f"{behavior.name} [{behavior.hotkey}]"
        self.setText(label)
        self.setCursor(Qt.PointingHandCursor)
        accent = behavior.color
        self.setStyleSheet(
            f"""
            QPushButton {{
                text-align: left;
                padding: 8px 10px;
                border-radius: 8px;
                border: 1px solid #2A3948;
                background: #111923;
                color: #E7EDF4;
            }}
            QPushButton:checked {{
                border: 2px solid {accent};
                background: #152030;
            }}
            QPushButton:hover {{
                background: #172231;
            }}
            """
        )        

