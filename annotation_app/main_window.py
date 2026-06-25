from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QEvent, QSignalBlocker, QTimer, Qt, QUrl
from PySide6.QtGui import QAction, QKeySequence, QShortcut, QTextOption
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QSlider,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app_metadata import (
    ANNOTATION_WORKSPACE_NAME,
    APP_NAME,
)
from .dialogs import (
    BehaviorManagerDialog,
    BundleExistingModelDialog,
    HotkeyDialog,
    HotkeyMapDialog,
    ProjectMetadataDialog,
    ScientificWorkflowDialog,
    WorkflowGuideDialog,
)
from .extractor import export_full_video_annotations, extract_approved_clips
from .models import (
    AnnotationRecord,
    AnnotationStatus,
    BehaviorRecord,
    HotkeyBinding,
    ProjectRecord,
    VideoRecord,
)
from .state import AnnotationSessionState
from .store import AnnotationStore, normalize_video_split
from .timeline import AnnotationTimeline, BehaviorLaneLabels
from .widgets import BehaviorButton, VideoListItem, color_swatch_icon, format_ms
from .workflow_panels import (
    ArtifactInspectorPanel,
    EthogramSummaryPanel,
    FeatureCachePanel,
    InferencePanel,
    MobileNetFullVideoDatasetPanel,
    MobileNetFeatureCachePanel,
    PrepareDatasetPanel,
    PrepareFullVideoDatasetPanel,
    ReleaseStagePanel,
    TrainPanel,
)


@dataclass
class _VideoWidgetRefs:
    item: QListWidgetItem
    video_id: int


class AnnotationMainWindow(QMainWindow):
    def __init__(self, store: AnnotationStore, project: ProjectRecord):
        super().__init__()
        self.store = store
        self.project = project
        self._child_windows: list[AnnotationMainWindow] = []
        self.session = AnnotationSessionState()
        self.videos: list[VideoRecord] = []
        self.behaviors: list[BehaviorRecord] = []
        self.annotations: list[AnnotationRecord] = []
        self.current_video: VideoRecord | None = None
        self.selected_annotation: AnnotationRecord | None = None
        self._video_item_refs: list[_VideoWidgetRefs] = []
        self._shortcuts: list[QShortcut] = []
        self._pending_seek_ms: int | None = None
        self._scrubbing_slider = False
        self.timeline_zoom = 1.0
        self._store_closed = False

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.player.setAudioOutput(self.audio)
        self.audio.setVolume(0.0)

        self.position_persist_timer = QTimer(self)
        self.position_persist_timer.setInterval(1500)
        self.position_persist_timer.timeout.connect(self._persist_resume_state)
        self.position_persist_timer.start()

        self.setWindowTitle(f"{ANNOTATION_WORKSPACE_NAME} - {self.project.name}")
        self.setAttribute(Qt.WA_DeleteOnClose, True)
        self._apply_initial_window_size()
        self._build_actions()
        self._build_ui()
        self._apply_styles()
        self._connect_player()
        self._reload_all()
        self._restore_resume_state()
        QTimer.singleShot(350, self._show_workflow_guide_on_startup)

    def _apply_initial_window_size(self) -> None:
        screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(1440, 900)
            return
        available = screen.availableGeometry()
        width = max(1180, min(available.width() - 80, 1600))
        height = max(760, min(available.height() - 80, 980))
        self.resize(width, height)

    def _show_workflow_guide_on_startup(self) -> None:
        if self.store.get_setting(self.project.id, "show_workflow_guide_on_startup", "1") != "1":
            return
        self._show_workflow_guide()

    def _show_workflow_guide(self) -> None:
        show = self.store.get_setting(self.project.id, "show_workflow_guide_on_startup", "1") == "1"
        dialog = WorkflowGuideDialog(show_on_startup=show, parent=self)
        dialog.exec()
        self.store.set_setting(
            self.project.id,
            "show_workflow_guide_on_startup",
            "1" if dialog.should_show_on_startup() else "0",
        )

    def _show_scientific_workflow(self) -> None:
        ScientificWorkflowDialog(parent=self).exec()

    def closeEvent(self, event):  # pragma: no cover - UI event
        self.position_persist_timer.stop()
        if not self._store_closed:
            self._persist_resume_state()
            self.player.stop()
            self.store.close()
            self._store_closed = True
        super().closeEvent(event)

    def _build_actions(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        new_project_db = QAction("New project database...", self)
        new_project_db.triggered.connect(self._new_project_database)
        file_menu.addAction(new_project_db)
        open_project_db = QAction("Open project database...", self)
        open_project_db.triggered.connect(self._open_project_database)
        file_menu.addAction(open_project_db)
        export_project_db = QAction("Export project database...", self)
        export_project_db.triggered.connect(self._export_project_database)
        file_menu.addAction(export_project_db)
        file_menu.addSeparator()
        import_files = QAction("Import videos...", self)
        import_files.triggered.connect(self._import_video_files)
        file_menu.addAction(import_files)
        import_folder = QAction("Import folder...", self)
        import_folder.triggered.connect(self._import_video_folder)
        file_menu.addAction(import_folder)
        import_behaviors = QAction("Import labels YAML...", self)
        import_behaviors.triggered.connect(self._import_behavior_yaml)
        file_menu.addAction(import_behaviors)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        edit_menu = menu.addMenu("&Edit")
        project_metadata = QAction("Project metadata...", self)
        project_metadata.triggered.connect(self._edit_project_metadata)
        edit_menu.addAction(project_metadata)
        manage_behaviors = QAction("Manage behaviors...", self)
        manage_behaviors.triggered.connect(self._manage_behaviors)
        edit_menu.addAction(manage_behaviors)
        hotkeys = QAction("Hotkeys...", self)
        hotkeys.triggered.connect(self._edit_hotkeys)
        edit_menu.addAction(hotkeys)
        hotkey_map = QAction("Hotkey map...", self)
        hotkey_map.triggered.connect(self._show_hotkey_map)
        edit_menu.addAction(hotkey_map)

        project_menu = menu.addMenu("&Project")
        approve_all_action = QAction("Approve all spans in current video...", self)
        approve_all_action.triggered.connect(self._approve_all_annotations_in_current_video)
        project_menu.addAction(approve_all_action)
        project_menu.addSeparator()
        assign_train_action = QAction("Assign selected videos to Train", self)
        assign_train_action.triggered.connect(lambda: self._assign_selected_videos_to_split("train"))
        project_menu.addAction(assign_train_action)
        assign_val_action = QAction("Assign selected videos to Validation", self)
        assign_val_action.triggered.connect(lambda: self._assign_selected_videos_to_split("val"))
        project_menu.addAction(assign_val_action)
        assign_test_action = QAction("Assign selected videos to Held-out Test", self)
        assign_test_action.triggered.connect(lambda: self._assign_selected_videos_to_split("test"))
        project_menu.addAction(assign_test_action)
        assign_exclude_action = QAction("Exclude selected videos from export", self)
        assign_exclude_action.triggered.connect(lambda: self._assign_selected_videos_to_split("exclude"))
        project_menu.addAction(assign_exclude_action)
        auto_split_action = QAction("Auto-assign train/validation split...", self)
        auto_split_action.triggered.connect(self._auto_assign_train_val_splits)
        project_menu.addAction(auto_split_action)
        project_menu.addSeparator()
        extract_action = QAction("Extract approved clips...", self)
        extract_action.triggered.connect(self._extract_approved_clips)
        project_menu.addAction(extract_action)
        export_full_video_action = QAction("Export full-video annotations...", self)
        export_full_video_action.triggered.connect(self._export_full_video_annotations)
        project_menu.addAction(export_full_video_action)

        model_tools_menu = menu.addMenu("&Model Tools")
        bundle_model_action = QAction("Bundle existing classifier + YOLO...", self)
        bundle_model_action.triggered.connect(self._bundle_existing_model)
        model_tools_menu.addAction(bundle_model_action)

        help_menu = menu.addMenu("&Help")
        workflow_guide_action = QAction("Workflow Guide...", self)
        workflow_guide_action.triggered.connect(self._show_workflow_guide)
        help_menu.addAction(workflow_guide_action)
        scientific_workflow_action = QAction("Scientific Workflow...", self)
        scientific_workflow_action.triggered.connect(self._show_scientific_workflow)
        help_menu.addAction(scientific_workflow_action)

    def _spawn_annotation_window(self, db_path: Path) -> None:
        store = AnnotationStore(db_path)
        project = store.get_or_create_default_project()
        window = AnnotationMainWindow(store, project)
        self._child_windows.append(window)
        window.destroyed.connect(lambda *_: self._child_windows.remove(window) if window in self._child_windows else None)
        window.show()
        window.raise_()
        window.activateWindow()

    def _new_project_database(self) -> None:
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Create project database",
            str(self.store.db_path.parent / "annotation_workspace.sqlite"),
            "SQLite database (*.sqlite *.db);;All files (*.*)",
        )
        if not target:
            return
        db_path = Path(target)
        if db_path.exists():
            overwrite = QMessageBox.question(
                self,
                "Overwrite database?",
                f"{db_path.name} already exists. Open it instead of overwriting?",
                QMessageBox.Open | QMessageBox.Cancel,
                QMessageBox.Open,
            )
            if overwrite != QMessageBox.Open:
                return
            self._spawn_annotation_window(db_path)
            return
        store = AnnotationStore(db_path)
        store.get_or_create_default_project()
        store.close()
        self._spawn_annotation_window(db_path)

    def _open_project_database(self) -> None:
        target, _ = QFileDialog.getOpenFileName(
            self,
            "Open project database",
            str(self.store.db_path.parent),
            "SQLite database (*.sqlite *.db);;All files (*.*)",
        )
        if not target:
            return
        self._spawn_annotation_window(Path(target))

    def _export_project_database(self) -> None:
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Export project database",
            str(self.store.db_path.with_name(f"{self.store.db_path.stem}_shared_review.sqlite")),
            "SQLite database (*.sqlite *.db);;All files (*.*)",
        )
        if not target:
            return
        exported, manifest_path = self.store.export_project_bundle(self.project.id, Path(target))
        QMessageBox.information(
            self,
            "Database exported",
            f"Project database exported to:\n{exported}\n\nMetadata manifest:\n{manifest_path}",
        )

    def _edit_project_metadata(self) -> None:
        dialog = ProjectMetadataDialog(
            self.store.get_project_metadata(self.project.id),
            self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        self.store.save_project_metadata(self.project.id, dialog.values())
        self.project = self.store.get_or_create_default_project()
        self.setWindowTitle(f"{ANNOTATION_WORKSPACE_NAME} - {self.project.name}")

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(12, 12, 12, 12)
        central_layout.setSpacing(8)

        self.workflow_tabs = QTabWidget()
        central_layout.addWidget(self.workflow_tabs, 1)

        annotation_page = QWidget()
        annotation_idx = self.workflow_tabs.addTab(annotation_page, "Annotate + Clip")
        self.workflow_tabs.setTabToolTip(
            annotation_idx,
            "Import videos, assign splits, label bouts, approve spans, and export full-video annotations.",
        )
        root = QHBoxLayout(annotation_page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(12)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root.addWidget(splitter, 1)

        left = QFrame()
        left.setObjectName("Sidebar")
        left.setMinimumWidth(190)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        self.video_header = QLabel("Videos")
        self.video_header.setObjectName("SidebarHeader")
        left_layout.addWidget(self.video_header)
        self.video_list = QListWidget()
        self.video_list.setObjectName("VideoList")
        self.video_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.video_list.currentRowChanged.connect(self._on_video_row_changed)
        left_layout.addWidget(self.video_list, 1)
        splitter.addWidget(left)

        center = QFrame()
        center.setObjectName("CenterPanel")
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)

        toolbar = QFrame()
        toolbar.setObjectName("TopBar")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(14, 8, 14, 8)
        self.video_title = QLabel("No video loaded")
        self.video_title.setObjectName("TitleLabel")
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("TimeLabel")
        toolbar_layout.addWidget(self.video_title, 1)
        toolbar_layout.addWidget(self.time_label)
        center_layout.addWidget(toolbar)

        player_row = QFrame()
        player_layout = QVBoxLayout(player_row)
        player_layout.setContentsMargins(0, 0, 0, 0)
        player_layout.setSpacing(0)
        self.empty_state = QFrame()
        self.empty_state.setObjectName("EmptyState")
        empty_layout = QVBoxLayout(self.empty_state)
        empty_layout.setContentsMargins(24, 24, 24, 24)
        empty_layout.setSpacing(12)
        empty_layout.addStretch(1)
        empty_title = QLabel("Start with your videos")
        empty_title.setObjectName("EmptyStateTitle")
        empty_title.setAlignment(Qt.AlignCenter)
        empty_layout.addWidget(empty_title)
        empty_hint = QLabel("Import videos, define behavior labels, assign splits, and export annotations for model training.")
        empty_hint.setObjectName("HintLabel")
        empty_hint.setAlignment(Qt.AlignCenter)
        empty_hint.setWordWrap(True)
        empty_layout.addWidget(empty_hint)
        empty_actions = QHBoxLayout()
        empty_actions.addStretch(1)
        import_videos_btn = QPushButton("Import videos")
        import_videos_btn.setObjectName("PrimaryButton")
        import_videos_btn.setToolTip("Select one or more video files to add to this annotation project.")
        import_videos_btn.clicked.connect(self._import_video_files)
        empty_actions.addWidget(import_videos_btn)
        import_folder_btn = QPushButton("Import folder")
        import_folder_btn.setToolTip("Add every supported video file from a folder.")
        import_folder_btn.clicked.connect(self._import_video_folder)
        empty_actions.addWidget(import_folder_btn)
        labels_btn = QPushButton("Behavior labels")
        labels_btn.setToolTip("Add or edit the behavior names, definitions, colors, and hotkeys before annotating.")
        labels_btn.clicked.connect(self._manage_behaviors)
        empty_actions.addWidget(labels_btn)
        guide_btn = QPushButton("Workflow guide")
        guide_btn.setToolTip("Open the end-to-end workflow map.")
        guide_btn.clicked.connect(self._show_workflow_guide)
        empty_actions.addWidget(guide_btn)
        science_btn = QPushButton("Scientific workflow")
        science_btn.setToolTip("Show how the GUI maps to the manuscript workflow, outputs, and provenance.")
        science_btn.clicked.connect(self._show_scientific_workflow)
        empty_actions.addWidget(science_btn)
        empty_actions.addStretch(1)
        empty_layout.addLayout(empty_actions)
        empty_layout.addStretch(1)
        player_layout.addWidget(self.empty_state, 1)
        self.video_widget = QVideoWidget()
        self.video_widget.setMinimumHeight(260)
        self.video_widget.setObjectName("VideoSurface")
        player_layout.addWidget(self.video_widget, 1)
        self.player.setVideoOutput(self.video_widget)
        center_layout.addWidget(player_row, 1)

        controls = QFrame()
        controls.setObjectName("ScrubberBar")
        controls_layout = QHBoxLayout(controls)
        controls_layout.setContentsMargins(12, 6, 12, 6)
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setToolTip("Play or pause the current video. Hotkey: Space.")
        controls_layout.addWidget(self.play_btn)
        self.prev_frame_btn = QPushButton("[-]")
        self.prev_frame_btn.clicked.connect(lambda: self._step_frames(-1))
        self.prev_frame_btn.setToolTip("Step backward by one frame. Hotkey: [")
        controls_layout.addWidget(self.prev_frame_btn)
        self.next_frame_btn = QPushButton("[+]")
        self.next_frame_btn.clicked.connect(lambda: self._step_frames(1))
        self.next_frame_btn.setToolTip("Step forward by one frame. Hotkey: ]")
        controls_layout.addWidget(self.next_frame_btn)
        self.current_time = QLabel("0:00")
        controls_layout.addWidget(self.current_time)
        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setRange(0, 0)
        self.scrubber.sliderPressed.connect(self._begin_slider_scrub)
        self.scrubber.sliderReleased.connect(self._end_slider_scrub)
        self.scrubber.sliderMoved.connect(self._slider_moved)
        controls_layout.addWidget(self.scrubber, 1)
        self.speed_combo = QComboBox()
        self.speed_combo.addItem("1x", 1.0)
        self.speed_combo.addItem("2x", 2.0)
        self.speed_combo.addItem("4x", 4.0)
        self.speed_combo.addItem("6x", 6.0)
        self.speed_combo.currentIndexChanged.connect(self._on_speed_changed)
        self.speed_combo.setToolTip("Playback speed. Hotkeys: Ctrl+1 / Ctrl+2 / Ctrl+4 / Ctrl+6.")
        controls_layout.addWidget(self.speed_combo)
        self.hotkey_map_btn = QPushButton("Hotkey map")
        self.hotkey_map_btn.clicked.connect(self._show_hotkey_map)
        self.hotkey_map_btn.setToolTip("Open a visible hotkey reference panel.")
        controls_layout.addWidget(self.hotkey_map_btn)
        self.timeline_zoom_label = QLabel("Timeline zoom 1.0x")
        controls_layout.addWidget(self.timeline_zoom_label)
        self.timeline_zoom_slider = QSlider(Qt.Horizontal)
        self.timeline_zoom_slider.setRange(1, 100)
        self.timeline_zoom_slider.setValue(10)
        self.timeline_zoom_slider.setMinimumWidth(90)
        self.timeline_zoom_slider.setMaximumWidth(160)
        self.timeline_zoom_slider.valueChanged.connect(self._on_timeline_zoom_changed)
        self.timeline_zoom_slider.setToolTip("Zoom the annotation timeline for frame-precise editing.")
        controls_layout.addWidget(self.timeline_zoom_slider)
        self.total_time = QLabel("0:00")
        controls_layout.addWidget(self.total_time)
        center_layout.addWidget(controls)

        timeline_frame = QFrame()
        timeline_frame.setObjectName("TimelineFrame")
        timeline_layout = QHBoxLayout(timeline_frame)
        timeline_layout.setContentsMargins(0, 0, 0, 0)
        timeline_layout.setSpacing(0)
        self.timeline = AnnotationTimeline()
        self.timeline.set_show_labels(False)
        self.timeline.spanCreated.connect(self._create_annotation)
        self.timeline.spanSelected.connect(self._select_annotation_by_id)
        self.timeline.spanDeleteRequested.connect(self._delete_annotation)
        self.timeline.spanEdited.connect(self._edit_annotation_range)
        self.timeline.playheadScrubbed.connect(self._seek_ms)
        self.timeline.spanZoomRequested.connect(self._zoom_to_span)
        self.timeline_labels = BehaviorLaneLabels()
        timeline_layout.addWidget(self.timeline_labels)
        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidget(self.timeline)
        self.timeline_scroll.setWidgetResizable(False)
        self.timeline_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.timeline_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.timeline_scroll.setFrameShape(QFrame.NoFrame)
        self.timeline_scroll.viewport().installEventFilter(self)
        timeline_layout.addWidget(self.timeline_scroll, 1)
        center_layout.addWidget(timeline_frame)

        footer = QFrame()
        footer.setObjectName("BottomBar")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(14, 8, 14, 8)
        self.span_count_label = QLabel("0 spans")
        self.approved_count_label = QLabel("0 approved")
        self.quick_btn = QPushButton("Quick bookmark")
        self.quick_btn.clicked.connect(self._quick_bookmark)
        self.quick_btn.setToolTip("Create a short bookmark centered on the playhead. Hotkey: Q.")
        self.next_video_btn = QPushButton("Next video")
        self.next_video_btn.clicked.connect(self._next_video)
        self.next_video_btn.setToolTip("Advance to the next video in the queue. Hotkey: N.")
        self.export_full_video_btn = QPushButton("Export full-video annotations")
        self.export_full_video_btn.setObjectName("PrimaryButton")
        self.export_full_video_btn.clicked.connect(self._export_full_video_annotations)
        self.export_full_video_btn.setToolTip(
            "Write source_manifest.csv and per-video .annot files for the full-video training workflow."
        )
        footer_layout.addWidget(self.span_count_label)
        footer_layout.addWidget(self.approved_count_label)
        footer_layout.addStretch(1)
        footer_layout.addWidget(self.quick_btn)
        footer_layout.addWidget(self.next_video_btn)
        footer_layout.addWidget(self.export_full_video_btn)
        center_layout.addWidget(footer)
        splitter.addWidget(center)

        right = QFrame()
        right.setObjectName("Inspector")
        right.setMinimumWidth(260)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        inspector_scroll = QScrollArea()
        inspector_scroll.setWidgetResizable(True)
        inspector_scroll.setFrameShape(QFrame.NoFrame)
        inspector_inner = QWidget()
        inspector_scroll.setWidget(inspector_inner)
        inspector_inner_layout = QVBoxLayout(inspector_inner)
        inspector_inner_layout.setContentsMargins(0, 0, 0, 0)
        inspector_inner_layout.setSpacing(10)

        behaviors_frame = QFrame()
        behaviors_layout = QVBoxLayout(behaviors_frame)
        behaviors_layout.setContentsMargins(0, 0, 0, 0)
        header = QLabel("Behaviors")
        header.setObjectName("SidebarHeader")
        behaviors_layout.addWidget(header)
        self.behavior_container = QWidget()
        self.behavior_layout = QVBoxLayout(self.behavior_container)
        self.behavior_layout.setContentsMargins(8, 8, 8, 8)
        self.behavior_layout.setSpacing(6)
        behaviors_layout.addWidget(self.behavior_container)
        inspector_inner_layout.addWidget(behaviors_frame)

        behavior_definition_card = QFrame()
        behavior_definition_card.setObjectName("InspectorCard")
        behavior_definition_layout = QVBoxLayout(behavior_definition_card)
        behavior_definition_layout.setContentsMargins(12, 12, 12, 12)
        behavior_definition_layout.setSpacing(6)
        behavior_definition_header = QLabel("Behavior definition")
        behavior_definition_header.setObjectName("SectionHeader")
        behavior_definition_layout.addWidget(behavior_definition_header)
        self.behavior_definition_title = QLabel("No behavior selected")
        self.behavior_definition_title.setObjectName("SectionHeader")
        behavior_definition_layout.addWidget(self.behavior_definition_title)
        self.behavior_definition_label = QLabel("Select a behavior to see the shared project definition here.")
        self.behavior_definition_label.setWordWrap(True)
        self.behavior_definition_label.setObjectName("HintLabel")
        self.behavior_definition_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        behavior_definition_layout.addWidget(self.behavior_definition_label)
        inspector_inner_layout.addWidget(behavior_definition_card)

        inspector = QFrame()
        inspector.setObjectName("InspectorCard")
        inspector_layout = QVBoxLayout(inspector)
        inspector_layout.setContentsMargins(12, 12, 12, 12)
        inspector_layout.setSpacing(10)
        self.selected_header = QLabel("Selected span")
        self.selected_header.setObjectName("SectionHeader")
        inspector_layout.addWidget(self.selected_header)

        form = QFormLayout()
        self.behavior_combo = QComboBox()
        self.behavior_combo.setToolTip("Relabel the selected span without redrawing it on the timeline.")
        self.behavior_combo.currentIndexChanged.connect(self._save_annotation_fields)
        form.addRow("Behavior", self.behavior_combo)
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setToolTip("Optional confidence score for this annotation. Use 1.00 when boundaries and behavior are clear.")
        self.confidence_spin.valueChanged.connect(self._save_annotation_fields)
        form.addRow("Confidence", self.confidence_spin)
        self.ambiguous_check = QCheckBox("Ambiguous")
        self.ambiguous_check.setToolTip("Mark this span when the behavior or boundary is uncertain but still worth reviewing.")
        self.ambiguous_check.toggled.connect(self._save_annotation_fields)
        form.addRow("", self.ambiguous_check)
        self.locked_check = QCheckBox("Lock position")
        self.locked_check.setToolTip("Prevent timeline drag, resize, relabel, and delete until unlocked. Hotkey: Ctrl+L.")
        self.locked_check.toggled.connect(self._save_annotation_fields)
        form.addRow("", self.locked_check)
        self.start_label = QLabel("-")
        form.addRow("Start", self.start_label)
        self.end_label = QLabel("-")
        form.addRow("End", self.end_label)
        self.duration_label = QLabel("-")
        form.addRow("Duration", self.duration_label)
        inspector_layout.addLayout(form)

        note_label = QLabel("Notes")
        inspector_layout.addWidget(note_label)
        self.notes_edit = QPlainTextEdit()
        self._enforce_notes_edit_left_to_right()
        self.notes_edit.setPlaceholderText("Notes about uncertainty, quality-control checks, or bout context.")
        self.notes_edit.setToolTip("Free-text notes for this span. Text entry is left-to-right.")
        self.notes_edit.setMinimumHeight(96)
        self.notes_edit.setObjectName("NotesEdit")
        self.notes_edit.textChanged.connect(self._save_annotation_fields)
        inspector_layout.addWidget(self.notes_edit)

        status_header = QLabel("Review state")
        status_header.setObjectName("SectionHeader")
        inspector_layout.addWidget(status_header)
        self.status_badge = QLabel("Draft")
        self.status_badge.setObjectName("StatusBadge")
        inspector_layout.addWidget(self.status_badge)
        self.status_hint = QLabel(
            "Draft spans are working notes. Move a span to Ready when boundaries look correct, then Approve or Reject during quality control."
        )
        self.status_hint.setWordWrap(True)
        self.status_hint.setObjectName("HintLabel")
        inspector_layout.addWidget(self.status_hint)

        action_row = QHBoxLayout()
        self.draft_btn = QPushButton("Draft")
        self.draft_btn.setCheckable(True)
        self.draft_btn.setProperty("statusRole", "draft")
        self.draft_btn.setToolTip("Working annotation that is not ready for quality control.")
        self.draft_btn.clicked.connect(lambda: self._set_selected_status(AnnotationStatus.DRAFT))
        self.ready_btn = QPushButton("Ready")
        self.ready_btn.setCheckable(True)
        self.ready_btn.setProperty("statusRole", "ready")
        self.ready_btn.setToolTip("Boundary and label look correct; mark for review.")
        self.ready_btn.clicked.connect(lambda: self._set_selected_status(AnnotationStatus.READY))
        self.approve_btn = QPushButton("Approve")
        self.approve_btn.setCheckable(True)
        self.approve_btn.setProperty("statusRole", "approved")
        self.approve_btn.setToolTip("Accept this span for export and training.")
        self.approve_btn.clicked.connect(lambda: self._set_selected_status(AnnotationStatus.APPROVED))
        self.reject_btn = QPushButton("Reject")
        self.reject_btn.setCheckable(True)
        self.reject_btn.setProperty("statusRole", "rejected")
        self.reject_btn.setToolTip("Exclude this span from approved training/export outputs.")
        self.reject_btn.clicked.connect(lambda: self._set_selected_status(AnnotationStatus.REJECTED))
        action_row.addWidget(self.draft_btn)
        action_row.addWidget(self.ready_btn)
        action_row.addWidget(self.approve_btn)
        action_row.addWidget(self.reject_btn)
        inspector_layout.addLayout(action_row)

        trim_row = QHBoxLayout()
        set_start = QPushButton("Set start to cursor")
        set_start.clicked.connect(self._set_pending_start_at_cursor)
        set_start.setToolTip("Mark the start frame for the next span. Hotkey: Shift+[.")
        set_end = QPushButton("Set end to cursor")
        set_end.clicked.connect(self._set_pending_end_at_cursor)
        set_end.setToolTip("Mark the end frame for the next span. Hotkey: Shift+].")
        clear_pending = QPushButton("Clear pending")
        clear_pending.clicked.connect(self._clear_pending_boundaries)
        clear_pending.setToolTip("Clear pending start/end markers. Hotkey: Ctrl+[.")
        trim_row.addWidget(set_start)
        trim_row.addWidget(set_end)
        trim_row.addWidget(clear_pending)
        inspector_layout.addLayout(trim_row)
        self.pending_label = QLabel("Pending span: none")
        self.pending_label.setObjectName("HintLabel")
        inspector_layout.addWidget(self.pending_label)
        inspector_inner_layout.addWidget(inspector)

        help_card = QFrame()
        help_card.setObjectName("InspectorCard")
        help_layout = QVBoxLayout(help_card)
        help_layout.setContentsMargins(12, 12, 12, 12)
        help_layout.setSpacing(6)
        help_header = QLabel("Workflow Hints")
        help_header.setObjectName("SectionHeader")
        help_layout.addWidget(help_header)
        for text in (
            "1. Select a behavior from the palette or with its hotkey.",
            "2. Use Shift+[ for start and Shift+] for end, or drag on the timeline track.",
            "3. Drag a selected span onto another behavior row to relabel it without redrawing.",
            "4. Use [ and ] to step frame-by-frame at subtle transitions.",
            "5. Single-click an empty track to seek. Double-click a span to zoom around it.",
            "6. Mouse-wheel over the timeline to zoom at the cursor.",
            "7. Lock a span after the boundary is correct, then mark it Ready or Approve.",
        ):
            label = QLabel(text)
            label.setWordWrap(True)
            label.setObjectName("HintLabel")
            help_layout.addWidget(label)
        inspector_inner_layout.addWidget(help_card)

        span_list_card = QFrame()
        span_list_card.setObjectName("InspectorCard")
        span_list_layout = QVBoxLayout(span_list_card)
        span_list_layout.setContentsMargins(12, 12, 12, 12)
        span_list_layout.setSpacing(8)
        span_list_header = QLabel("Video spans")
        span_list_header.setObjectName("SectionHeader")
        span_list_layout.addWidget(span_list_header)
        self.annotation_table = QTableWidget(0, 4)
        self.annotation_table.setHorizontalHeaderLabels(["Behavior", "Start", "End", "State"])
        self.annotation_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.annotation_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.annotation_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.annotation_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.annotation_table.verticalHeader().setVisible(False)
        self.annotation_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.annotation_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.annotation_table.itemSelectionChanged.connect(self._on_annotation_table_selection_changed)
        span_list_layout.addWidget(self.annotation_table)
        inspector_inner_layout.addWidget(span_list_card, 1)
        inspector_inner_layout.addStretch(1)
        right_layout.addWidget(inspector_scroll, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([230, 980, 330])

        status = QStatusBar(self)
        self.setStatusBar(status)
        self.state_label = QLabel("Ready")
        self.statusBar().addPermanentWidget(self.state_label)

        app_root = Path(__file__).resolve().parents[1]
        analysis_root = app_root / "analysis_workflows"

        self.yolo_page = QWidget()
        yolo_layout = QVBoxLayout(self.yolo_page)
        yolo_layout.setContentsMargins(0, 0, 0, 0)
        self.yolo_tabs = QTabWidget()
        yolo_layout.addWidget(self.yolo_tabs, 1)
        yolo_idx = self.workflow_tabs.addTab(self.yolo_page, color_swatch_icon("#3E8EDE"), "YOLO-pose")
        self.workflow_tabs.setTabToolTip(
            yolo_idx,
            "YOLO-pose cache building, feature extraction, temporal classifier training, inference, ethograms, batch processing, and outputs.",
        )

        self.prepare_dataset_panel = PrepareDatasetPanel(self, on_success=self._on_prepare_dataset_success)
        self.prepare_full_video_panel = PrepareFullVideoDatasetPanel(self, on_success=self._on_prepare_full_video_success)
        self.feature_cache_panel = FeatureCachePanel(self, on_success=self._on_feature_cache_success)
        self.train_panel = TrainPanel(self, on_success=self._on_train_success, workflow_label="YOLO-pose", feature_mode="yolo")
        self.inference_panel = InferencePanel(batch_mode=False, parent=self)
        self.batch_panel = InferencePanel(batch_mode=True, parent=self)
        self.yolo_ethogram_panel = EthogramSummaryPanel(
            workflow_label="YOLO-pose",
            default_output_root=app_root / "outputs" / "yolo_pose",
            parent=self,
        )
        self.yolo_outputs_panel = ArtifactInspectorPanel(
            title="YOLO-pose outputs and summaries",
            default_root=app_root / "outputs",
            parent=self,
        )
        self._add_model_workflow_tab(self.yolo_tabs, self.prepare_dataset_panel, "Legacy Clip Cache", "#6A7A89", "Build windows from class-folder clips for older datasets.")
        self._add_model_workflow_tab(self.yolo_tabs, self.prepare_full_video_panel, "Full-Video Cache", "#2EA96B", "Build sliding-window NPZ caches from full videos and exported annotations.")
        self._add_model_workflow_tab(self.yolo_tabs, self.feature_cache_panel, "Train/Val Feature Cache", "#B9872F", "Precompute frozen YOLO-pose visual descriptors for train/validation windows.")
        self._add_model_workflow_tab(self.yolo_tabs, self.train_panel, "Train + Validation Eval", "#7C6BD6", "Train the temporal behavior classifier and evaluate on validation windows.")
        self._add_model_workflow_tab(self.yolo_tabs, self.inference_panel, "Inference", "#3E8EDE", "Run a trained bundled model on one video and export behavior predictions.")
        self._add_model_workflow_tab(self.yolo_tabs, self.batch_panel, "Batch", "#3E8EDE", "Run a trained bundled model across a folder of videos.")
        self._add_model_workflow_tab(self.yolo_tabs, self.yolo_ethogram_panel, "Ethograms + Bouts", "#C27A34", "Create model-agnostic ethogram timelines and bout summaries from YOLO-pose temporal predictions.")
        self._add_model_workflow_tab(self.yolo_tabs, self.yolo_outputs_panel, "Outputs", "#52606D", "Inspect YOLO-pose caches, logs, metrics, ethograms, and exported files.")

        self.mobilenet_page = QWidget()
        mobilenet_layout = QVBoxLayout(self.mobilenet_page)
        mobilenet_layout.setContentsMargins(0, 0, 0, 0)
        self.mobilenet_tabs = QTabWidget()
        mobilenet_layout.addWidget(self.mobilenet_tabs, 1)
        mobilenet_idx = self.workflow_tabs.addTab(self.mobilenet_page, color_swatch_icon("#1FA37A"), "MobileNetV3")
        self.workflow_tabs.setTabToolTip(
            mobilenet_idx,
            "MobileNetV3 cache building, pose-backbone feature extraction, temporal classifier training, ethograms, held-out evaluation, and summary stages.",
        )
        self.mobilenet_full_video_panel = MobileNetFullVideoDatasetPanel(self, on_success=self._on_prepare_full_video_success)
        self.mobilenet_feature_cache_panel = MobileNetFeatureCachePanel(self)
        self.mobilenet_train_panel = TrainPanel(
            self,
            on_success=self._on_train_success,
            workflow_label="MobileNetV3",
            feature_mode="cached",
        )
        self.mobilenet_ethogram_panel = EthogramSummaryPanel(
            workflow_label="MobileNetV3",
            default_output_root=app_root / "outputs" / "mobilenetv3",
            parent=self,
        )
        self._add_model_workflow_tab(
            self.mobilenet_tabs,
            self.mobilenet_full_video_panel,
            "Full-Video Cache",
            "#2EA96B",
            "Build sliding-window NPZ caches directly from videos using the MobileNetV3 pose checkpoint.",
        )
        self._add_model_workflow_tab(
            self.mobilenet_tabs,
            self.mobilenet_feature_cache_panel,
            "Train/Val Feature Cache",
            "#1FA37A",
            "Extract MobileNetV3-large pose-backbone visual descriptors from MobileNetV3 sequence windows.",
        )
        self._add_model_workflow_tab(
            self.mobilenet_tabs,
            self.mobilenet_train_panel,
            "Train + Validation Eval",
            "#7C6BD6",
            "Train the temporal behavior classifier from MobileNetV3 visual features plus pose-derived streams.",
        )
        mobilenet_runner = analysis_root / "mobilenetv3_backbone" / "run_mobilenetv3_backbone.py"
        for stage, title, description in (
            ("build_npz", "Full-Video + Held-Out Caches", "Build or reuse the MARS train/validation and held-out sequence NPZ caches."),
            ("build_visual_cache", "Train/Val Feature Cache", "Extract MobileNetV3 visual descriptors for train/validation and held-out manifests."),
            ("train_neural", "Train Temporal Classifiers", "Train LSTM and attention temporal classifiers from MobileNetV3 visual features plus pose-derived streams."),
            ("eval_neural", "Held-Out Evaluation", "Evaluate trained MobileNetV3 temporal classifiers on held-out MARS videos."),
            ("train_classical", "Static Baseline Training", "Train RF/XGBoost negative-control baselines from tabularized pose and PCA-compressed visual features."),
            ("eval_classical", "Static Baseline Evaluation", "Evaluate RF/XGBoost baselines on held-out MARS videos."),
            ("summarize", "Suite Summary Tables", "Collect frame, bout, per-video, and confusion-matrix summary tables across MobileNetV3 runs."),
        ):
            self._add_model_workflow_tab(
                self.mobilenet_tabs,
                ReleaseStagePanel(
                    title=title,
                    description=description,
                    runner_script=mobilenet_runner,
                    stage=stage,
                    workflow_kind="mobilenetv3",
                    parent=self,
                ),
                title,
                "#1FA37A",
                description,
            )
        self._add_model_workflow_tab(
            self.mobilenet_tabs,
            self.mobilenet_ethogram_panel,
            "Ethograms + Bouts",
            "#C27A34",
            "Create model-agnostic ethogram timelines and bout summaries from MobileNetV3 temporal predictions.",
        )
        self._add_model_workflow_tab(
            self.mobilenet_tabs,
            ArtifactInspectorPanel(
                title="MobileNetV3 outputs and summaries",
                default_root=app_root,
                parent=self,
            ),
            "Outputs",
            "#52606D",
            "Inspect MobileNetV3 caches, trained models, metrics, static-baseline outputs, and summaries.",
        )

        self.dlc_page = QWidget()
        dlc_layout = QVBoxLayout(self.dlc_page)
        dlc_layout.setContentsMargins(0, 0, 0, 0)
        self.dlc_tabs = QTabWidget()
        dlc_layout.addWidget(self.dlc_tabs, 1)
        dlc_idx = self.workflow_tabs.addTab(self.dlc_page, color_swatch_icon("#8A63D2"), "DeepLabCut-HRNet")
        self.workflow_tabs.setTabToolTip(
            dlc_idx,
            "DeepLabCut SuperAnimal fine-tuning, top-down cache building, HRNet feature extraction, temporal classifier training, ethograms, and held-out evaluation.",
        )
        dlc_runner = analysis_root / "dlc_superanimal_topdown" / "run_dlc_superanimal_topdown.py"
        self.dlc_ethogram_panel = EthogramSummaryPanel(
            workflow_label="DeepLabCut-HRNet",
            default_output_root=app_root / "outputs" / "dlc_superanimal_topdown",
            parent=self,
        )
        for stage, title, description in (
            ("train_dlc_pose_detector", "Pose + Detector Fine-Tuning", "Fine-tune the DLC SuperAnimal pose model and detector with validation-based model selection."),
            ("build_trainval_npz", "Full-Video Train/Val Cache", "Run the DLC top-down detector and pose model to build full-video train/validation sequence windows."),
            ("build_trainval_hrnet_cache", "Train/Val HRNet Feature Cache", "Extract pooled multi-resolution HRNet-W32 visual descriptors from DLC top-down crops."),
            ("train_classifier", "Classifier Train + Validation", f"Train the {APP_NAME} temporal classifier using DLC pose-derived and HRNet visual features."),
            ("prepare_heldout_manifest", "Held-Out Manifest", "Prepare the held-out MP4/.annot manifest used for test-video evaluation."),
            ("build_heldout_npz", "Held-Out Full-Video Cache", "Build held-out DLC top-down sequence windows."),
            ("build_heldout_hrnet_cache", "Held-Out HRNet Feature Cache", "Extract HRNet visual descriptors for held-out DLC top-down windows."),
            ("evaluate_heldout", "Held-Out Evaluation", "Evaluate frame, bout, per-video, confusion-matrix, and ethogram-level outputs."),
        ):
            self._add_model_workflow_tab(
                self.dlc_tabs,
                ReleaseStagePanel(
                    title=title,
                    description=description,
                    runner_script=dlc_runner,
                    stage=stage,
                    workflow_kind="dlc",
                    parent=self,
                ),
                title,
                "#8A63D2",
                description,
            )
        self._add_model_workflow_tab(
            self.dlc_tabs,
            self.dlc_ethogram_panel,
            "Ethograms + Bouts",
            "#C27A34",
            "Create model-agnostic ethogram timelines and bout summaries from DeepLabCut-HRNet temporal predictions.",
        )
        self._add_model_workflow_tab(
            self.dlc_tabs,
            ArtifactInspectorPanel(
                title="DeepLabCut-HRNet outputs and summaries",
                default_root=app_root / "outputs" / "dlc_superanimal_topdown",
                parent=self,
            ),
            "Outputs",
            "#52606D",
            "Inspect DeepLabCut-HRNet caches, trained classifiers, held-out metrics, ethograms, and summaries.",
        )

    def _add_workflow_tab(self, widget: QWidget, title: str, color: str) -> None:
        self.workflow_tabs.addTab(widget, color_swatch_icon(color), title)

    def _add_model_workflow_tab(self, tabs: QTabWidget, widget: QWidget, title: str, color: str, tooltip: str = "") -> None:
        idx = tabs.addTab(widget, color_swatch_icon(color), title)
        if tooltip:
            tabs.setTabToolTip(idx, tooltip)

    def _select_yolo_panel(self, widget: QWidget) -> None:
        if hasattr(self, "yolo_page"):
            self.workflow_tabs.setCurrentWidget(self.yolo_page)
        if hasattr(self, "yolo_tabs"):
            self.yolo_tabs.setCurrentWidget(widget)

    def _cmd_arg(self, command: list[str], flag: str) -> str | None:
        try:
            idx = command.index(flag)
        except ValueError:
            return None
        if idx + 1 >= len(command):
            return None
        return str(command[idx + 1])

    def _selected_video_ids(self) -> list[int]:
        ids: list[int] = []
        for item in self.video_list.selectedItems():
            value = item.data(Qt.UserRole)
            if value is not None:
                ids.append(int(value))
        if not ids and self.current_video is not None:
            ids.append(int(self.current_video.id))
        return list(dict.fromkeys(ids))

    def _assign_selected_videos_to_split(self, split: str) -> None:
        video_ids = self._selected_video_ids()
        if not video_ids:
            QMessageBox.information(self, "No videos selected", "Select one or more videos first.")
            return
        split = normalize_video_split(split)
        self.store.set_video_splits(video_ids, split)
        self._reload_all()
        self.statusBar().showMessage(f"Assigned {len(video_ids)} video(s) to {split}.", 5000)

    def _auto_assign_train_val_splits(self) -> None:
        if len(self.videos) < 2:
            QMessageBox.information(self, "Not enough videos", "Import at least two videos before auto-assigning splits.")
            return
        default_ratio = float(self.store.get_setting(self.project.id, "full_video_val_ratio", "0.20") or "0.20")
        val_ratio, ok = QInputDialog.getDouble(
            self,
            "Auto-assign validation split",
            "Fraction of imported videos assigned to validation:",
            default_ratio,
            0.0,
            0.9,
            2,
        )
        if not ok:
            return
        ordered = sorted(self.videos, key=lambda video: (video.filename.lower(), video.id))
        n_val = int(round(len(ordered) * float(val_ratio)))
        if val_ratio > 0.0:
            n_val = min(max(1, n_val), len(ordered) - 1)
        else:
            n_val = 0
        val_ids = {video.id for video in ordered[:n_val]}
        train_ids = [video.id for video in ordered if video.id not in val_ids]
        self.store.set_video_splits(train_ids, "train")
        self.store.set_video_splits(val_ids, "val")
        self.store.set_setting(self.project.id, "full_video_val_ratio", f"{float(val_ratio):.4f}")
        self._reload_all()
        self.statusBar().showMessage(
            f"Auto-assigned train={len(train_ids)} and val={len(val_ids)} videos.",
            6000,
        )

    def _fill_inference_model_inputs(self, model_path: str) -> None:
        for panel in (self.inference_panel, self.batch_panel):
            panel.model_path.setText(model_path)
            panel.model_config.setText("")
            panel.yolo_weights.setText("")
        model_dir = Path(model_path).resolve().parent
        if not self.inference_panel.output_dir.text().strip():
            self.inference_panel.output_dir.setText(str(model_dir / "inference_outputs"))
        if not self.batch_panel.output_dir.text().strip():
            self.batch_panel.output_dir.setText(str(model_dir / "batch_outputs"))
        self._select_yolo_panel(self.inference_panel)

    def _bundle_existing_model(self) -> None:
        dialog = BundleExistingModelDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        values = dialog.values()
        required = {
            "Classifier checkpoint": values["classifier_checkpoint"],
            "YOLO pose weights": values["yolo_weights"],
            "Bundled output .pt": values["output"],
        }
        for label, path in required.items():
            if not path:
                QMessageBox.warning(self, "Missing path", f"{label} is required.")
                return
        classifier = Path(values["classifier_checkpoint"])
        yolo_weights = Path(values["yolo_weights"])
        model_config = Path(values["model_config"]) if values["model_config"] else None
        output = Path(values["output"])
        for label, path in (("Classifier checkpoint", classifier), ("YOLO pose weights", yolo_weights)):
            if not path.is_file():
                QMessageBox.warning(self, "Path not found", f"{label} was not found:\n{path}")
                return
        if model_config is not None and not model_config.is_file():
            QMessageBox.warning(self, "Path not found", f"Model config was not found:\n{model_config}")
            return
        script = Path(__file__).resolve().parents[1] / "package_single_model_x.py"
        if not script.is_file():
            QMessageBox.critical(self, "Missing bundler", f"Could not find:\n{script}")
            return
        cmd = [
            sys.executable,
            str(script),
            "--classifier_checkpoint",
            str(classifier),
            "--yolo_weights",
            str(yolo_weights),
            "--output",
            str(output),
        ]
        if model_config is not None:
            cmd.extend(["--model_config", str(model_config)])
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).resolve().parents[1]),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
        finally:
            QApplication.restoreOverrideCursor()
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "No process output.").strip()
            QMessageBox.critical(self, "Bundling failed", message[:4000])
            return
        self._fill_inference_model_inputs(str(output.resolve()))
        QMessageBox.information(
            self,
            "Bundled model ready",
            f"Wrote bundled model:\n{output.resolve()}\n\nInference and Batch model paths have been filled.",
        )

    def _on_prepare_dataset_success(self, command: list[str]) -> None:
        output_root = self._cmd_arg(command, "--output_root")
        dataset_root = self._cmd_arg(command, "--dataset_root")
        manifest_path = self._cmd_arg(command, "--manifest_path")
        if not manifest_path:
            base = Path(output_root) if output_root else Path(dataset_root or "") / "behaviorscope_x_processed"
            manifest_path = str((base / "sequence_manifest.json").resolve())
        self._fill_training_inputs(manifest_path, self._cmd_arg(command, "--yolo_weights"))
        self._fill_feature_cache_inputs(manifest_path, self._cmd_arg(command, "--yolo_weights"), output_root)

    def _on_prepare_full_video_success(self, command: list[str]) -> None:
        output_root = self._cmd_arg(command, "--output_root")
        manifest_path = self._cmd_arg(command, "--manifest_path")
        if not manifest_path and output_root:
            manifest_path = str((Path(output_root) / "sequence_manifest.json").resolve())
        self._fill_training_inputs(manifest_path, self._cmd_arg(command, "--yolo_weights"))
        self._fill_feature_cache_inputs(manifest_path, self._cmd_arg(command, "--yolo_weights"), output_root)
        source_manifest = self._cmd_arg(command, "--source_manifest_csv")
        if source_manifest:
            annot_root = Path(source_manifest).resolve().parent / "annotations"
            if annot_root.exists():
                self.train_panel.temporal_splitter_annot_root.setText(str(annot_root))

    def _fill_feature_cache_inputs(self, manifest_path: str | None, yolo_weights: str | None, output_root: str | None = None) -> None:
        if not hasattr(self, "feature_cache_panel"):
            return
        if manifest_path:
            self.feature_cache_panel.manifest_path.setText(manifest_path)
        if yolo_weights:
            self.feature_cache_panel.yolo_weights.setText(yolo_weights)
        cache_root = Path(output_root) / "yolo_feature_cache" if output_root else None
        if cache_root is None and manifest_path:
            cache_root = Path(manifest_path).resolve().parent / "yolo_feature_cache"
        if cache_root is not None:
            self.feature_cache_panel.output_dir.setText(str(cache_root))
            self.train_panel.use_feature_cache.setText(str(cache_root))

    def _on_feature_cache_success(self, command: list[str]) -> None:
        cache_dir = self._cmd_arg(command, "--output_dir")
        manifest_path = self._cmd_arg(command, "--manifest_path")
        yolo_weights = self._cmd_arg(command, "--yolo_weights")
        if cache_dir:
            self.train_panel.use_feature_cache.setText(cache_dir)
            self.train_panel.auto_feature_cache.setChecked(False)
        self._fill_training_inputs(manifest_path, yolo_weights)
        self.statusBar().showMessage("Training inputs were filled from the completed feature-cache run.", 6000)

    def _fill_training_inputs(self, manifest_path: str | None, yolo_weights: str | None) -> None:
        if manifest_path:
            self.train_panel.manifest_path.setText(manifest_path)
        if yolo_weights:
            self.train_panel.yolo_weights.setText(yolo_weights)
        self._select_yolo_panel(self.train_panel)
        self.statusBar().showMessage("Training inputs were filled from the completed preprocessing run.", 6000)

    def _on_train_success(self, command: list[str]) -> None:
        project = Path(self._cmd_arg(command, "--project") or "behavior_lstm_runs")
        name = self._cmd_arg(command, "--name") or "run"
        run_dir = (project / name).resolve()
        model_path = run_dir / "best_model_macro_f1.pt"
        if not model_path.exists():
            model_path = run_dir / "best_model.pt"
        single_model_path = self._cmd_arg(command, "--single_model_path")
        if not single_model_path:
            candidate_single = run_dir / "behaviorscope_x_single_model.pt"
            if candidate_single.exists():
                single_model_path = str(candidate_single)
        config_path = run_dir / "config.json"
        yolo_weights = self._cmd_arg(command, "--yolo_weights")
        for panel in (self.inference_panel, self.batch_panel):
            if single_model_path:
                panel.model_path.setText(single_model_path)
                panel.model_config.clear()
                panel.yolo_weights.clear()
            else:
                panel.model_path.setText(str(model_path))
                if yolo_weights:
                    panel.yolo_weights.setText(yolo_weights)
            if config_path.exists() and not single_model_path:
                panel.model_config.setText(str(config_path))
        self.inference_panel.output_dir.setText(str(run_dir / "inference_outputs"))
        self.batch_panel.output_dir.setText(str(run_dir / "batch_outputs"))
        self._select_yolo_panel(self.inference_panel)
        self.statusBar().showMessage("Inference and batch inputs were filled from the completed training run.", 6000)

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0B1118;
                color: #E8EEF4;
                font-family: "Segoe UI", "SF Pro Text", "Helvetica Neue", Arial, sans-serif;
                font-size: 13px;
            }
            QMenuBar, QMenuBar::item, QMenu {
                background: #0B1118;
                color: #E8EEF4;
            }
            QFrame#Sidebar, QFrame#Inspector, QFrame#CenterPanel {
                border: 1px solid #263645;
                border-radius: 12px;
                background: #0F1721;
            }
            QLabel#SidebarHeader {
                padding: 10px 12px;
                border-bottom: 1px solid #263645;
                color: #8B9EB2;
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.08em;
                text-transform: uppercase;
            }
            QListWidget#VideoList {
                border: none;
                outline: none;
                background: #0F1721;
            }
            QListWidget#VideoList::item {
                border-bottom: 1px solid #1C2935;
                padding: 2px;
            }
            QListWidget#VideoList::item:selected {
                background: #162332;
            }
            QLabel#videoName {
                font-size: 12px;
                font-weight: 600;
                color: #E8EEF4;
            }
            QLabel#videoMeta {
                color: #8B9EB2;
                font-size: 11px;
            }
            QFrame#TopBar, QFrame#BottomBar, QFrame#ScrubberBar {
                border-bottom: 1px solid #263645;
                background: #0F1721;
            }
            QFrame#BottomBar {
                border-top: 1px solid #263645;
                border-bottom: none;
            }
            QLabel#TitleLabel {
                font-size: 14px;
                font-weight: 600;
                color: #E8EEF4;
            }
            QLabel#EmptyStateTitle {
                font-size: 18px;
                font-weight: 700;
                color: #FFFFFF;
            }
            QLabel#TimeLabel {
                color: #8B9EB2;
            }
            QVideoWidget#VideoSurface {
                background: #04070B;
            }
            QFrame#EmptyState {
                background: #071019;
                border: none;
            }
            QFrame#TimelineFrame, QFrame#InspectorCard {
                background: #10161E;
                border-top: 1px solid #263645;
            }
            QTabWidget::pane {
                background: #0B1118;
                border: 1px solid #263645;
                border-radius: 8px;
                top: -1px;
            }
            QTabWidget::tab-bar {
                left: 8px;
            }
            QTabBar::tab {
                background: #101A25;
                color: #C9D4DF;
                border: 1px solid #2A3948;
                border-bottom-color: #263645;
                padding: 8px 14px;
                margin-right: 2px;
                min-height: 20px;
                font-size: 14px;
                font-weight: 600;
            }
            QTabBar::tab:selected {
                background: #1A2A3A;
                color: #FFFFFF;
                border-color: #3E8EDE;
                border-bottom-color: #1A2A3A;
                font-weight: 700;
            }
            QTabBar::tab:hover:!selected {
                background: #162332;
                color: #E8EEF4;
                border-color: #3A5065;
            }
            QTabBar::tab:disabled {
                background: #0F1721;
                color: #667789;
            }
            QLabel#SectionHeader {
                color: #E8EEF4;
                font-weight: 600;
                font-size: 13px;
            }
            QLabel#HintLabel {
                color: #8B9EB2;
                font-size: 11px;
            }
            QPushButton {
                background: #162231;
                border: 1px solid #2D4256;
                border-radius: 8px;
                color: #E8EEF4;
                padding: 7px 12px;
            }
            QPushButton:hover {
                background: #1B2B3D;
            }
            QPushButton#PrimaryButton {
                background: #2E7AC7;
                border-color: #4F94D8;
                color: white;
            }
            QPushButton#PrimaryButton:hover {
                background: #3A86D2;
            }
            QPushButton[statusRole="draft"]:checked {
                background: #3B4754;
                border-color: #75889A;
            }
            QPushButton[statusRole="ready"]:checked {
                background: #1D4E67;
                border-color: #57A9CF;
            }
            QPushButton[statusRole="approved"]:checked {
                background: #1E5B3A;
                border-color: #52C584;
            }
            QPushButton[statusRole="rejected"]:checked {
                background: #6B2727;
                border-color: #E27B7B;
            }
            QComboBox, QLineEdit, QPlainTextEdit, QDoubleSpinBox, QTableWidget {
                background: #0F1721;
                border: 1px solid #2A3948;
                border-radius: 8px;
                padding: 6px;
                color: #E8EEF4;
            }
            QPlainTextEdit#NotesEdit {
                selection-background-color: #2E7AC7;
            }
            QLabel#StatusBadge {
                padding: 6px 10px;
                border-radius: 8px;
                background: #1A2431;
                border: 1px solid #324455;
                color: #E8EEF4;
                font-weight: 600;
            }
            QTableWidget {
                gridline-color: #233242;
            }
            QHeaderView::section {
                background: #111B27;
                color: #8B9EB2;
                border: none;
                padding: 6px;
            }
            QSlider::groove:horizontal {
                height: 4px;
                background: #283949;
                border-radius: 2px;
            }
            QSlider::sub-page:horizontal {
                background: #3E8EDE;
                border-radius: 2px;
            }
            QSlider::handle:horizontal {
                width: 12px;
                margin: -4px 0;
                border-radius: 6px;
                background: #3E8EDE;
                border: 1px solid #0F1721;
            }
            """
        )

    def _connect_player(self) -> None:
        self.player.positionChanged.connect(self._on_player_position_changed)
        self.player.durationChanged.connect(self._on_player_duration_changed)
        self.player.playbackStateChanged.connect(self._on_playback_state_changed)
        self.player.mediaStatusChanged.connect(self._on_media_status_changed)

    def _reload_all(self) -> None:
        self.behaviors = self.store.list_behaviors(self.project.id)
        self.videos = self.store.list_videos(self.project.id)
        self._refresh_behavior_buttons()
        self._refresh_behavior_combo()
        self._refresh_video_list()
        self._rebuild_hotkeys()
        if self.current_video is not None:
            self.annotations = self.store.list_annotations(self.current_video.id)
        else:
            self.annotations = []
        self._refresh_annotation_table()
        self._refresh_timeline()
        self._update_footer()

    def _restore_resume_state(self) -> None:
        if not self.videos:
            self._set_empty_state()
            return
        resume_video = self.store.get_setting(self.project.id, "resume_video_id", "")
        resume_position = int(self.store.get_setting(self.project.id, "resume_position_ms", "0") or "0")
        resume_behavior = self.store.get_setting(self.project.id, "resume_behavior_id", "")
        if resume_behavior:
            try:
                self.session.selected_behavior_id = int(resume_behavior)
            except ValueError:
                self.session.selected_behavior_id = None
        target_id: int | None = None
        if resume_video:
            try:
                target_id = int(resume_video)
            except ValueError:
                target_id = None
        if target_id is None or not any(video.id == target_id for video in self.videos):
            target_id = self.videos[0].id
        self._set_current_video(target_id, resume_position)
        if self.session.selected_behavior_id is None and self.behaviors:
            self.session.selected_behavior_id = self.behaviors[0].id
        self._refresh_behavior_buttons()

    def _set_empty_state(self) -> None:
        self.current_video = None
        self.session.set_video(None)
        self.video_title.setText("No videos imported")
        self.time_label.setText("0:00 / 0:00")
        self.current_time.setText("0:00")
        self.total_time.setText("0:00")
        self.scrubber.setRange(0, 0)
        self.annotations = []
        self.empty_state.show()
        self.video_widget.hide()
        self._refresh_annotation_table()
        self._refresh_timeline()
        self._update_footer()

    def _refresh_video_list(self) -> None:
        self.video_list.clear()
        self._video_item_refs.clear()
        self.video_header.setText(f"Videos  {len(self.videos)}")
        for video in self.videos:
            item = QListWidgetItem()
            item.setSizeHint(VideoListItem(video).sizeHint())
            widget = VideoListItem(video, selected=(self.current_video is not None and video.id == self.current_video.id))
            item.setData(Qt.UserRole, video.id)
            self.video_list.addItem(item)
            self.video_list.setItemWidget(item, widget)
            self._video_item_refs.append(_VideoWidgetRefs(item=item, video_id=video.id))
        if self.current_video is not None:
            for index, ref in enumerate(self._video_item_refs):
                if ref.video_id == self.current_video.id:
                    self.video_list.setCurrentRow(index)
                    break

    def _refresh_behavior_buttons(self) -> None:
        while self.behavior_layout.count():
            child = self.behavior_layout.takeAt(0)
            widget = child.widget()
            if widget is not None:
                widget.deleteLater()
        for behavior in self.behaviors:
            btn = BehaviorButton(
                behavior,
                selected=(behavior.id == self.session.selected_behavior_id),
            )
            if behavior.hotkey:
                tooltip = f"Select {behavior.name}. Hotkey: {behavior.hotkey}."
            else:
                tooltip = f"Select {behavior.name}."
            if behavior.definition:
                tooltip += f"\n\nDefinition: {behavior.definition}"
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda checked=False, behavior_id=behavior.id: self._set_selected_behavior(behavior_id))
            self.behavior_layout.addWidget(btn)
        self.behavior_layout.addStretch(1)
        self._update_behavior_definition_panel()

    def _refresh_behavior_combo(self) -> None:
        blocker = QSignalBlocker(self.behavior_combo)
        self.behavior_combo.clear()
        for behavior in self.behaviors:
            self.behavior_combo.addItem(behavior.name, behavior.id)
        del blocker

    def _refresh_annotations(self) -> None:
        if self.current_video is None:
            self.annotations = []
        else:
            self.annotations = self.store.list_annotations(self.current_video.id)
        if self.selected_annotation is not None:
            self.selected_annotation = next(
                (ann for ann in self.annotations if ann.id == self.selected_annotation.id),
                None,
            )
        self._refresh_annotation_table()
        self._refresh_timeline()
        self._update_inspector()
        self._update_footer()

    def _refresh_annotation_table(self) -> None:
        self.annotation_table.setRowCount(len(self.annotations))
        for row, annotation in enumerate(self.annotations):
            behavior_item = QTableWidgetItem(annotation.behavior_name)
            behavior_item.setData(Qt.UserRole, annotation.id)
            self.annotation_table.setItem(row, 0, behavior_item)
            self.annotation_table.setItem(row, 1, QTableWidgetItem(format_ms(annotation.start_ms)))
            self.annotation_table.setItem(row, 2, QTableWidgetItem(format_ms(annotation.end_ms)))
            flags: list[str] = []
            if annotation.is_locked:
                flags.append("LOCK")
            if annotation.is_ambiguous:
                flags.append("?")
            status_text = annotation.status + (f" {' '.join(flags)}" if flags else "")
            self.annotation_table.setItem(row, 3, QTableWidgetItem(status_text))
        if self.selected_annotation is not None:
            for row in range(self.annotation_table.rowCount()):
                item = self.annotation_table.item(row, 0)
                if item is not None and int(item.data(Qt.UserRole)) == self.selected_annotation.id:
                    self.annotation_table.selectRow(row)
                    break

    def _refresh_timeline(self) -> None:
        duration = self.current_video.duration_ms if self.current_video is not None else 1
        playhead_ms = self.player.position() if self.current_video is not None else 0
        self.timeline_labels.set_data(self.behaviors)
        self.timeline.set_viewport_width_hint(self.timeline_scroll.viewport().width())
        self.timeline.set_data(
            behaviors=self.behaviors,
            annotations=self.annotations,
            duration_ms=duration,
            playhead_ms=playhead_ms,
            selected_annotation_id=self.selected_annotation.id if self.selected_annotation is not None else None,
        )
        self.timeline.set_pending_preview(
            selected_behavior_id=self.session.selected_behavior_id,
            pending_start_ms=self.session.pending_start_ms,
            pending_end_ms=self.session.pending_end_ms,
        )
        self.timeline.set_zoom_factor(self.timeline_zoom)

    def _update_footer(self) -> None:
        self.span_count_label.setText(f"{len(self.annotations)} spans")
        approved = len([ann for ann in self.annotations if ann.status == AnnotationStatus.APPROVED.value])
        self.approved_count_label.setText(f"{approved} approved")
        self.state_label.setText(self.session.ui_state.value.replace("_", " ").title())
        self._update_pending_label()

    def _update_pending_label(self) -> None:
        start_ms = self.session.pending_start_ms
        end_ms = self.session.pending_end_ms
        behavior_name = next(
            (behavior.name for behavior in self.behaviors if behavior.id == self.session.selected_behavior_id),
            None,
        )
        if start_ms is None and end_ms is None:
            self.pending_label.setText("Pending span: none")
            return
        parts: list[str] = [behavior_name] if behavior_name else []
        if start_ms is not None:
            parts.append(f"start {format_ms(start_ms)}")
        if end_ms is not None:
            parts.append(f"end {format_ms(end_ms)}")
        if start_ms is not None and end_ms is None:
            parts.append("move the cursor to preview the clip, then set end")
        elif end_ms is not None and start_ms is None:
            parts.append("move the cursor to preview the clip, then set start")
        self.pending_label.setText("Pending span: " + "  |  ".join(parts))

    def _enforce_notes_edit_left_to_right(self) -> None:
        self.notes_edit.setLayoutDirection(Qt.LeftToRight)
        self.notes_edit.viewport().setLayoutDirection(Qt.LeftToRight)
        if hasattr(self.notes_edit, "setCursorMoveStyle"):
            self.notes_edit.setCursorMoveStyle(Qt.LogicalMoveStyle)
        notes_text_option = self.notes_edit.document().defaultTextOption()
        notes_text_option.setAlignment(Qt.AlignLeft)
        notes_text_option.setTextDirection(Qt.LeftToRight)
        notes_text_option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        self.notes_edit.document().setDefaultTextOption(notes_text_option)

    def _update_inspector(self) -> None:
        ann = self.selected_annotation
        enabled = ann is not None
        for widget in (
            self.behavior_combo,
            self.confidence_spin,
            self.ambiguous_check,
            self.locked_check,
            self.notes_edit,
            self.draft_btn,
            self.ready_btn,
            self.approve_btn,
            self.reject_btn,
        ):
            widget.setEnabled(enabled)
        if ann is None:
            self.start_label.setText("-")
            self.end_label.setText("-")
            self.duration_label.setText("-")
            self.status_badge.setText("No span selected")
            self.status_badge.setStyleSheet("")
            self.status_hint.setText(
                "Draft spans are private working notes. Move a span to Ready when boundaries look correct, then Approve or Reject during review."
            )
            self._sync_status_buttons(None)
            if self.notes_edit.toPlainText():
                blocker = QSignalBlocker(self.notes_edit)
                self.notes_edit.setPlainText("")
                del blocker
            blocker = QSignalBlocker(self.ambiguous_check)
            self.ambiguous_check.setChecked(False)
            del blocker
            blocker = QSignalBlocker(self.locked_check)
            self.locked_check.setChecked(False)
            del blocker
            self._enforce_notes_edit_left_to_right()
            return
        blocker = QSignalBlocker(self.behavior_combo)
        idx = self.behavior_combo.findData(ann.behavior_id)
        self.behavior_combo.setCurrentIndex(idx)
        del blocker
        blocker = QSignalBlocker(self.confidence_spin)
        self.confidence_spin.setValue(ann.confidence)
        del blocker
        blocker = QSignalBlocker(self.ambiguous_check)
        self.ambiguous_check.setChecked(ann.is_ambiguous)
        del blocker
        blocker = QSignalBlocker(self.locked_check)
        self.locked_check.setChecked(ann.is_locked)
        del blocker
        if self.notes_edit.toPlainText() != ann.notes:
            blocker = QSignalBlocker(self.notes_edit)
            self.notes_edit.setPlainText(ann.notes)
            del blocker
        self._enforce_notes_edit_left_to_right()
        self.start_label.setText(f"{ann.start_frame}  ({format_ms(ann.start_ms)})")
        self.end_label.setText(f"{ann.end_frame}  ({format_ms(ann.end_ms)})")
        self.duration_label.setText(f"{ann.duration_frames} frames  |  {format_ms(ann.duration_ms)}")
        self._sync_status_buttons(ann.status)
        self._update_status_display(ann.status)

    def _sync_status_buttons(self, status: str | None) -> None:
        for button, status_name in (
            (self.draft_btn, AnnotationStatus.DRAFT.value),
            (self.ready_btn, AnnotationStatus.READY.value),
            (self.approve_btn, AnnotationStatus.APPROVED.value),
            (self.reject_btn, AnnotationStatus.REJECTED.value),
        ):
            blocker = QSignalBlocker(button)
            button.setChecked(status == status_name)
            del blocker

    def _update_status_display(self, status: str) -> None:
        palette = {
            AnnotationStatus.DRAFT.value: ("Draft", "#3B4754", "#75889A", "Draft spans are working bookmarks. Use this while boundaries or labels are still being refined."),
            AnnotationStatus.READY.value: ("Ready", "#1D4E67", "#57A9CF", "Ready spans look correct and are waiting for quality control or approval."),
            AnnotationStatus.APPROVED.value: ("Approved", "#1E5B3A", "#52C584", "Approved spans are included in full-video annotation export and legacy clip extraction."),
            AnnotationStatus.REJECTED.value: ("Rejected", "#6B2727", "#E27B7B", "Rejected spans are kept for traceability but should not be exported."),
        }
        label, background, border, hint = palette.get(
            status,
            ("Unknown", "#1A2431", "#324455", "This span has an unrecognized review state."),
        )
        self.status_badge.setText(label)
        self.status_badge.setStyleSheet(
            f"background: {background}; border: 1px solid {border}; border-radius: 8px; padding: 6px 10px; color: #E8EEF4; font-weight: 600;"
        )
        self.status_hint.setText(hint)

    def _set_selected_behavior(self, behavior_id: int) -> None:
        self.session.selected_behavior_id = behavior_id
        self._refresh_behavior_buttons()
        self._refresh_timeline()
        self._update_footer()
        self.store.save_resume_state(
            self.project.id,
            video_id=self.current_video.id if self.current_video is not None else None,
            position_ms=self.player.position(),
            behavior_id=behavior_id,
        )

    def _update_behavior_definition_panel(self) -> None:
        behavior = next(
            (item for item in self.behaviors if item.id == self.session.selected_behavior_id),
            None,
        )
        if behavior is None:
            self.behavior_definition_title.setText("No behavior selected")
            self.behavior_definition_label.setText(
                "Select a behavior to see the shared project definition here."
            )
            return
        self.behavior_definition_title.setText(behavior.name)
        self.behavior_definition_label.setText(
            behavior.definition.strip() or "No shared definition has been added for this behavior yet."
        )

    def _set_current_video(self, video_id: int, position_ms: int = 0) -> None:
        video = next((item for item in self.videos if item.id == video_id), None)
        if video is None:
            return
        if self.current_video is not None:
            self.store.update_video_progress(
                self.current_video.id,
                position_ms=self.player.position(),
                status="in_progress",
            )
        self.current_video = video
        self.session.set_video(video)
        self.empty_state.hide()
        self.video_widget.show()
        self.video_title.setText(video.filename)
        self.total_time.setText(format_ms(video.duration_ms))
        self.time_label.setText(f"{format_ms(position_ms)} / {format_ms(video.duration_ms)}")
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(video.path))
        self._pending_seek_ms = max(0, min(position_ms, video.duration_ms))
        self.annotations = self.store.list_annotations(video.id)
        self.selected_annotation = None
        self._refresh_video_list()
        self._refresh_annotation_table()
        self._refresh_timeline()
        self._update_inspector()
        self._update_footer()

    def _on_video_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._video_item_refs):
            return
        video_id = self._video_item_refs[row].video_id
        if self.current_video is not None and self.current_video.id == video_id:
            return
        self._persist_resume_state()
        video = next((item for item in self.videos if item.id == video_id), None)
        self._set_current_video(video_id, video.last_position_ms if video is not None else 0)

    def _select_annotation_by_id(self, annotation_id: int) -> None:
        ann = next((item for item in self.annotations if item.id == annotation_id), None)
        self.selected_annotation = ann
        self.session.select_annotation(ann)
        self._refresh_timeline()
        self._update_inspector()
        self._refresh_annotation_table()
        self._update_footer()

    def _on_annotation_table_selection_changed(self) -> None:
        items = self.annotation_table.selectedItems()
        if not items:
            return
        ann_id = int(items[0].data(Qt.UserRole))
        if self.selected_annotation is None or self.selected_annotation.id != ann_id:
            self._select_annotation_by_id(ann_id)

    def _create_annotation(self, behavior_id: int, start_ms: int, end_ms: int) -> None:
        if self.current_video is None:
            return
        start_ms, end_ms = sorted((int(start_ms), int(end_ms)))
        fps = max(self.current_video.fps, 1e-6)
        start_frame = int(round((start_ms / 1000.0) * fps))
        end_frame = int(round((end_ms / 1000.0) * fps))
        ann_id = self.store.create_annotation(
            self.project.id,
            video_id=self.current_video.id,
            behavior_id=behavior_id,
            start_frame=start_frame,
            end_frame=end_frame,
            start_ms=start_ms,
            end_ms=end_ms,
        )
        self.session.clear_pending_boundaries()
        self._refresh_annotations()
        self._select_annotation_by_id(ann_id)

    def _edit_annotation_range(
        self,
        annotation_id: int,
        behavior_id: int,
        start_ms: int,
        end_ms: int,
    ) -> None:
        if self.current_video is None:
            return
        ann = next((item for item in self.annotations if item.id == annotation_id), None)
        if ann is not None and ann.is_locked:
            self.statusBar().showMessage("Unlock the selected span before moving or resizing it.", 3000)
            self._refresh_timeline()
            return
        fps = max(self.current_video.fps, 1e-6)
        start_frame = int(round((start_ms / 1000.0) * fps))
        end_frame = int(round((end_ms / 1000.0) * fps))
        self.store.update_annotation(
            annotation_id,
            start_ms=start_ms,
            end_ms=end_ms,
            start_frame=start_frame,
            end_frame=end_frame,
            behavior_id=behavior_id,
        )
        self._refresh_annotations()
        self._select_annotation_by_id(annotation_id)

    def _save_annotation_fields(self) -> None:
        ann = self.selected_annotation
        if ann is None:
            return
        behavior_id = self.behavior_combo.currentData()
        confidence = self.confidence_spin.value()
        is_ambiguous = self.ambiguous_check.isChecked()
        is_locked = self.locked_check.isChecked()
        notes = self.notes_edit.toPlainText()
        self.store.update_annotation(
            ann.id,
            behavior_id=int(behavior_id) if behavior_id is not None else ann.behavior_id,
            confidence=confidence,
            is_ambiguous=is_ambiguous,
            is_locked=is_locked,
            notes=notes,
        )
        self._refresh_annotations()

    def _delete_annotation(self, annotation_id: int) -> None:
        ann = next((item for item in self.annotations if item.id == annotation_id), None)
        if ann is not None and ann.is_locked:
            self.statusBar().showMessage("Unlock the selected span before deleting it.", 3000)
            return
        self.store.delete_annotation(annotation_id)
        self.selected_annotation = None
        self.session.select_annotation(None)
        self._refresh_annotations()

    def _set_selected_status(self, status: AnnotationStatus) -> None:
        if self.selected_annotation is None:
            return
        ann_id = self.selected_annotation.id
        self.store.update_annotation(ann_id, status=status)
        self._refresh_annotations()
        self._select_annotation_by_id(ann_id)

    def _approve_all_annotations_in_current_video(self) -> None:
        if self.current_video is None:
            QMessageBox.information(self, "No video loaded", "Load a video before approving spans.")
            return
        if not self.annotations:
            QMessageBox.information(
                self,
                "No spans to approve",
                f"No spans exist yet for {self.current_video.filename}.",
            )
            return
        count = len(self.annotations)
        confirm = QMessageBox.question(
            self,
            "Approve all spans?",
            (
                f"Approve all {count} spans in the current video?\n\n"
                f"Video: {self.current_video.filename}"
            ),
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Yes,
        )
        if confirm != QMessageBox.Yes:
            return
        changed = self.store.set_annotation_status_for_video(
            self.current_video.id,
            AnnotationStatus.APPROVED,
        )
        self._refresh_annotations()
        QMessageBox.information(
            self,
            "Spans approved",
            f"Approved {count if changed == 0 else changed} spans in {self.current_video.filename}.",
        )

    def _set_pending_start_at_cursor(self) -> None:
        if self.current_video is None:
            return
        self.session.select_annotation(None)
        self.selected_annotation = None
        self.session.set_pending_start(self.player.position())
        self._finalize_pending_annotation_if_ready()
        self._refresh_annotation_table()
        self._refresh_timeline()
        self._update_inspector()
        self._update_footer()

    def _set_pending_end_at_cursor(self) -> None:
        if self.current_video is None:
            return
        self.session.select_annotation(None)
        self.selected_annotation = None
        self.session.set_pending_end(self.player.position())
        self._finalize_pending_annotation_if_ready()
        self._refresh_annotation_table()
        self._refresh_timeline()
        self._update_inspector()
        self._update_footer()

    def _clear_pending_boundaries(self) -> None:
        self.session.clear_pending_boundaries()
        self._refresh_timeline()
        self._update_footer()

    def _finalize_pending_annotation_if_ready(self) -> None:
        if not self.session.has_pending_pair():
            return
        behavior_id = self.session.selected_behavior_id
        if behavior_id is None and self.behaviors:
            behavior_id = self.behaviors[0].id
        if behavior_id is None:
            return
        start_ms = int(self.session.pending_start_ms or 0)
        end_ms = int(self.session.pending_end_ms or 0)
        self._create_annotation(behavior_id, start_ms, end_ms)

    def _toggle_play(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def _seek_ms(self, position_ms: int) -> None:
        self.player.setPosition(int(max(0, position_ms)))

    def _on_timeline_zoom_changed(self, value: int) -> None:
        self.timeline_zoom = max(0.25, float(value) / 10.0)
        self.timeline_zoom_label.setText(f"Timeline zoom {self.timeline_zoom:.1f}x")
        self._refresh_timeline()
        self._ensure_playhead_visible()

    def _center_timeline_on_ms(self, time_ms: int) -> None:
        bar = self.timeline_scroll.horizontalScrollBar()
        viewport_width = max(1, self.timeline_scroll.viewport().width())
        center_x = self.timeline.time_to_x(int(time_ms))
        bar.setValue(max(0, center_x - (viewport_width // 2)))

    def _zoom_to_span(self, start_ms: int, end_ms: int) -> None:
        span_ms = max(250, int(end_ms) - int(start_ms))
        viewport_width = max(320, self.timeline_scroll.viewport().width() - 80)
        desired_pixels_per_second = (viewport_width * 0.5) / (span_ms / 1000.0)
        target_zoom = max(1.0, min(10.0, desired_pixels_per_second / self.timeline.base_pixels_per_second))
        target_zoom = max(self.timeline_zoom, target_zoom)
        slider_value = max(
            self.timeline_zoom_slider.minimum(),
            min(self.timeline_zoom_slider.maximum(), int(round(target_zoom * 10.0))),
        )
        self.timeline_zoom_slider.setValue(slider_value)
        self._seek_ms(start_ms)
        center_ms = (int(start_ms) + int(end_ms)) // 2
        QTimer.singleShot(0, lambda: self._center_timeline_on_ms(center_ms))

    def _step_frames(self, delta_frames: int) -> None:
        if self.current_video is None:
            return
        fps = max(self.current_video.fps, 1e-6)
        delta_ms = int(round((delta_frames / fps) * 1000.0))
        self._seek_ms(self.player.position() + delta_ms)

    def _jump_ms(self, delta_ms: int) -> None:
        self._seek_ms(self.player.position() + delta_ms)

    def _set_playback_rate(self, rate: float) -> None:
        self.player.setPlaybackRate(float(rate))
        blocker = QSignalBlocker(self.speed_combo)
        idx = self.speed_combo.findData(float(rate))
        if idx >= 0:
            self.speed_combo.setCurrentIndex(idx)
        del blocker

    def _on_speed_changed(self) -> None:
        rate = self.speed_combo.currentData()
        if rate is None:
            return
        self.player.setPlaybackRate(float(rate))

    def _quick_bookmark(self) -> None:
        if self.current_video is None:
            return
        behavior_id = self.session.selected_behavior_id
        if behavior_id is None and self.behaviors:
            behavior_id = self.behaviors[0].id
        if behavior_id is None:
            QMessageBox.warning(self, "No behaviors", "Add at least one behavior before bookmarking.")
            return
        span_ms = int(self.store.get_setting(self.project.id, "bookmark_span_ms", "2000") or "2000")
        center_ms = self.player.position()
        start_ms = max(0, center_ms - span_ms // 2)
        end_ms = min(self.current_video.duration_ms, start_ms + span_ms)
        self._create_annotation(behavior_id, start_ms, end_ms)

    def _next_video(self) -> None:
        if not self.videos:
            return
        if self.current_video is None:
            self._set_current_video(self.videos[0].id, 0)
            return
        index = next((i for i, video in enumerate(self.videos) if video.id == self.current_video.id), 0)
        index = (index + 1) % len(self.videos)
        self._set_current_video(self.videos[index].id, self.videos[index].last_position_ms)

    def _previous_video(self) -> None:
        if not self.videos:
            return
        if self.current_video is None:
            self._set_current_video(self.videos[0].id, 0)
            return
        index = next((i for i, video in enumerate(self.videos) if video.id == self.current_video.id), 0)
        index = (index - 1) % len(self.videos)
        self._set_current_video(self.videos[index].id, self.videos[index].last_position_ms)

    def _import_video_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import videos",
            str(Path.cwd()),
            "Videos (*.mp4 *.avi *.mov *.mkv)",
        )
        if not paths:
            return
        count = self.store.import_videos(self.project.id, [Path(path) for path in paths])
        self._reload_all()
        if self.current_video is None and self.videos:
            self._set_current_video(self.videos[0].id, 0)
        QMessageBox.information(self, "Import complete", f"Imported {count} new videos.")

    def _import_video_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Import folder", str(Path.cwd()))
        if not folder:
            return
        root = Path(folder)
        paths = [path for path in root.rglob("*") if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}]
        count = self.store.import_videos(self.project.id, paths)
        self._reload_all()
        if self.current_video is None and self.videos:
            self._set_current_video(self.videos[0].id, 0)
        QMessageBox.information(self, "Import complete", f"Imported {count} new videos.")

    def _import_behavior_yaml(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import labels YAML",
            str(Path.cwd()),
            "YAML (*.yaml *.yml)",
        )
        if not path:
            return
        self.store.import_behaviors_from_yaml(self.project.id, Path(path))
        self._reload_all()

    def _manage_behaviors(self) -> None:
        dialog = BehaviorManagerDialog(self.behaviors, self)
        if dialog.exec() != QDialog.Accepted:
            return
        warnings = self.store.sync_behaviors(self.project.id, dialog.rows())
        self._reload_all()
        if (
            self.session.selected_behavior_id is None
            or not any(behavior.id == self.session.selected_behavior_id for behavior in self.behaviors)
        ) and self.behaviors:
            self._set_selected_behavior(self.behaviors[0].id)
        if warnings:
            QMessageBox.information(
                self,
                "Behavior updates applied",
                "\n".join(warnings),
            )

    def _edit_hotkeys(self) -> None:
        dialog = HotkeyDialog(self.store.list_hotkeys(self.project.id), self)
        if dialog.exec() != QDialog.Accepted:
            return
        self.store.save_hotkeys(self.project.id, dialog.bindings())
        self._rebuild_hotkeys()

    def _show_hotkey_map(self) -> None:
        dialog = HotkeyMapDialog(
            self.store.list_hotkeys(self.project.id),
            self.behaviors,
            self,
        )
        dialog.exec()

    def _extract_approved_clips(self) -> None:
        confirm = QMessageBox.question(
            self,
            "Legacy clip export",
            (
                "Extract approved clips creates a clip-folder dataset. For full-video training, "
                "use Project -> Export full-video annotations instead so the NPZ builder "
                "can train on full-video sliding windows.\n\n"
                "Continue with legacy clip export?"
            ),
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if confirm != QMessageBox.Yes:
            return
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose dataset root",
            self.store.get_setting(self.project.id, "dataset_root", str(Path.cwd())),
        )
        if not output_dir:
            return
        self.store.set_setting(self.project.id, "dataset_root", output_dir)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            manifest = extract_approved_clips(
                self.store,
                project_id=self.project.id,
                output_root=Path(output_dir),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Extraction failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        per_class = manifest.get("per_class", {})
        summary = "\n".join(
            f"{name}: {stats['clip_count']} clips"
            for name, stats in per_class.items()
        ) or "No clips exported."
        summary += (
            "\n\nWrote clips.json and clips_metadata.csv. "
            "Use clips_metadata.csv with prepare_clips_x.py for source-grouped splits."
        )
        warnings = manifest.get("warnings", [])
        if warnings:
            summary += "\n\nWarnings:\n" + "\n".join(warnings[:6])
        QMessageBox.information(self, "Extraction complete", summary)
        self._refresh_annotations()

    def _export_full_video_annotations(self) -> None:
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose full-video annotation export folder",
            self.store.get_setting(self.project.id, "full_video_annotation_root", str(Path.cwd())),
        )
        if not output_dir:
            return
        default_ratio = float(self.store.get_setting(self.project.id, "full_video_val_ratio", "0.20") or "0.20")
        split_counts = {
            split: sum(1 for video in self.videos if normalize_video_split(video.split) == split)
            for split in ("train", "val", "test", "exclude")
        }
        if split_counts["val"] == 0:
            proceed = QMessageBox.question(
                self,
                "No validation videos assigned",
                (
                    "No videos are currently assigned to Validation. Training needs a validation split "
                    "for model selection.\n\n"
                    "Continue export anyway?"
                ),
            )
            if proceed != QMessageBox.Yes:
                return
        self.store.set_setting(self.project.id, "full_video_annotation_root", output_dir)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            manifest = export_full_video_annotations(
                self.store,
                project_id=self.project.id,
                output_root=Path(output_dir),
                val_ratio=float(default_ratio),
                recommended_window_frames=int(self.prepare_full_video_panel.window_size.value())
                if hasattr(self, "prepare_full_video_panel")
                else 32,
                recommended_stride_frames=int(self.prepare_full_video_panel.window_stride.value())
                if hasattr(self, "prepare_full_video_panel")
                else 16,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Full-video export failed", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()
        if hasattr(self, "prepare_full_video_panel"):
            export_root = Path(output_dir)
            self.prepare_full_video_panel.source_manifest_csv.setText(
                str(export_root / "source_manifest.csv")
            )
            self.prepare_full_video_panel.class_names_file.setText(
                str(export_root / "class_names.txt")
            )
            default_npz_root = export_root / "behaviorscope_npz"
            if not self.prepare_full_video_panel.output_root.text().strip():
                self.prepare_full_video_panel.output_root.setText(str(default_npz_root))
            self._select_yolo_panel(self.prepare_full_video_panel)
        preflight = manifest.get("preflight", {})
        warnings = list(preflight.get("warnings", []))
        info = list(preflight.get("info", []))
        summary = (
            f"Videos: {manifest.get('video_count', 0)}\n"
            f"Approved spans: {manifest.get('approved_annotation_count', 0)}\n"
            f"Splits: {manifest.get('split_counts', {})}\n"
            f"Excluded videos: {split_counts['exclude']}\n\n"
            "Wrote source_manifest.csv, class_names.txt, full_video_annotations.json, "
            "full_video_annotations.batch.json, preflight_report.json, and one .annot file per video.\n\n"
            "The YOLO-pose > Full-Video Cache tab has been filled with the export paths."
        )
        if warnings:
            summary += "\n\nPreflight warnings:\n" + "\n".join(warnings[:6])
        elif info:
            summary += "\n\nPreflight notes:\n" + "\n".join(info[:6])
        else:
            summary += "\n\nPreflight found no immediate issues."
        QMessageBox.information(self, "Full-video export complete", summary)

    def _rebuild_hotkeys(self) -> None:
        for shortcut in self._shortcuts:
            shortcut.setParent(None)
            shortcut.deleteLater()
        self._shortcuts.clear()
        bindings = {binding.action: binding.key_sequence for binding in self.store.list_hotkeys(self.project.id)}
        action_map = {
            "play_pause": self._toggle_play,
            "next_video": self._next_video,
            "previous_video": self._previous_video,
            "next_frame": lambda: self._step_frames(1),
            "previous_frame": lambda: self._step_frames(-1),
            "jump_forward": lambda: self._jump_ms(1000),
            "jump_backward": lambda: self._jump_ms(-1000),
            "quick_bookmark": self._quick_bookmark,
            "approve_selected": lambda: self._set_selected_status(AnnotationStatus.APPROVED),
            "reject_selected": lambda: self._set_selected_status(AnnotationStatus.REJECTED),
            "delete_selected": lambda: self._delete_annotation(self.selected_annotation.id) if self.selected_annotation is not None else None,
            "toggle_ambiguous": self._toggle_selected_ambiguous,
            "toggle_lock_selected": self._toggle_selected_lock,
            "set_pending_start": self._set_pending_start_at_cursor,
            "set_pending_end": self._set_pending_end_at_cursor,
            "clear_pending_boundaries": self._clear_pending_boundaries,
            "rate_1x": lambda: self._set_playback_rate(1.0),
            "rate_2x": lambda: self._set_playback_rate(2.0),
            "rate_4x": lambda: self._set_playback_rate(4.0),
            "rate_6x": lambda: self._set_playback_rate(6.0),
        }
        for action, callback in action_map.items():
            sequence = bindings.get(action, "")
            if not sequence:
                continue
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        for behavior in self.behaviors:
            if not behavior.hotkey:
                continue
            shortcut = QShortcut(QKeySequence(behavior.hotkey), self)
            shortcut.activated.connect(lambda behavior_id=behavior.id: self._set_selected_behavior(behavior_id))
            self._shortcuts.append(shortcut)

    def _toggle_selected_ambiguous(self) -> None:
        if self.selected_annotation is None:
            return
        ann_id = self.selected_annotation.id
        self.store.update_annotation(
            ann_id,
            is_ambiguous=not self.selected_annotation.is_ambiguous,
        )
        self._refresh_annotations()
        self._select_annotation_by_id(ann_id)

    def _toggle_selected_lock(self) -> None:
        if self.selected_annotation is None:
            return
        ann_id = self.selected_annotation.id
        self.store.update_annotation(
            ann_id,
            is_locked=not self.selected_annotation.is_locked,
        )
        self._refresh_annotations()
        self._select_annotation_by_id(ann_id)

    def _on_player_position_changed(self, position: int) -> None:
        if self.current_video is None:
            return
        if not self._scrubbing_slider:
            blocker = QSignalBlocker(self.scrubber)
            self.scrubber.setValue(int(position))
            del blocker
        self.current_time.setText(format_ms(position))
        self.time_label.setText(f"{format_ms(position)} / {format_ms(self.current_video.duration_ms)}")
        self.session.playhead_ms = position
        self._refresh_timeline()
        self._ensure_playhead_visible()

    def _on_player_duration_changed(self, duration: int) -> None:
        if self.current_video is None:
            return
        blocker = QSignalBlocker(self.scrubber)
        self.scrubber.setRange(0, max(duration, self.current_video.duration_ms))
        del blocker
        self.total_time.setText(format_ms(max(duration, self.current_video.duration_ms)))

    def _on_playback_state_changed(self, state) -> None:
        self.play_btn.setText("Pause" if state == QMediaPlayer.PlayingState else "Play")

    def _on_media_status_changed(self, status) -> None:
        if status == QMediaPlayer.LoadedMedia and self._pending_seek_ms is not None:
            self.player.setPosition(self._pending_seek_ms)
            self._pending_seek_ms = None

    def _begin_slider_scrub(self) -> None:
        self._scrubbing_slider = True

    def _end_slider_scrub(self) -> None:
        self._scrubbing_slider = False
        self._seek_ms(self.scrubber.value())

    def _slider_moved(self, value: int) -> None:
        self.current_time.setText(format_ms(value))
        if self.current_video is not None:
            self.time_label.setText(f"{format_ms(value)} / {format_ms(self.current_video.duration_ms)}")

    def _persist_resume_state(self) -> None:
        if self._store_closed:
            return
        self.store.save_resume_state(
            self.project.id,
            video_id=self.current_video.id if self.current_video is not None else None,
            position_ms=self.player.position(),
            behavior_id=self.session.selected_behavior_id,
        )
        if self.current_video is not None:
            self.store.update_video_progress(
                self.current_video.id,
                position_ms=self.player.position(),
                status="in_progress" if self.annotations else "unseen",
            )

    def resizeEvent(self, event):  # pragma: no cover - UI event
        super().resizeEvent(event)
        self.timeline.set_viewport_width_hint(self.timeline_scroll.viewport().width())
        self.timeline.set_zoom_factor(self.timeline_zoom)
        self._ensure_playhead_visible()

    def eventFilter(self, obj, event):  # pragma: no cover - UI event
        if obj is self.timeline_scroll.viewport() and event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            if delta == 0:
                return False
            viewport_pos = event.position().toPoint()
            content_pos = self.timeline.mapFrom(self.timeline_scroll.viewport(), viewport_pos)
            anchor_ms = self.timeline.time_at_x(content_pos.x())
            steps = max(-4, min(4, int(delta / 120) or (1 if delta > 0 else -1)))
            new_value = max(
                self.timeline_zoom_slider.minimum(),
                min(self.timeline_zoom_slider.maximum(), self.timeline_zoom_slider.value() + steps),
            )
            if new_value == self.timeline_zoom_slider.value():
                return True
            self.timeline_zoom_slider.setValue(new_value)
            QTimer.singleShot(
                0,
                lambda anchor_ms=anchor_ms, viewport_x=viewport_pos.x(): self.timeline_scroll.horizontalScrollBar().setValue(
                    max(0, self.timeline.time_to_x(anchor_ms) - viewport_x)
                ),
            )
            event.accept()
            return True
        return super().eventFilter(obj, event)

    def _ensure_playhead_visible(self) -> None:
        if self.current_video is None:
            return
        bar = self.timeline_scroll.horizontalScrollBar()
        viewport_width = max(1, self.timeline_scroll.viewport().width())
        playhead_x = self.timeline.playhead_x()
        left = bar.value()
        right = left + viewport_width
        margin = max(60, viewport_width // 4)
        if playhead_x < left + margin:
            bar.setValue(max(0, playhead_x - margin))
        elif playhead_x > right - margin:
            bar.setValue(max(0, playhead_x - viewport_width + margin))

