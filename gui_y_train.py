#!/usr/bin/env python
"""Tkinter launcher for BehaviorScope-Y training."""

from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


THIS_DIR = Path(__file__).resolve().parent
TRAIN_Y = THIS_DIR / "train_y.py"


def _shellquote(arg: str) -> str:
    if not arg:
        return '""'
    if any(ch.isspace() for ch in arg) or any(ch in arg for ch in ['"', "&", "(", ")"]):
        return '"' + arg.replace('"', '\\"') + '"'
    return arg


def _split_items(text: str) -> list[str]:
    return [part for part in text.replace(",", " ").split() if part]


class BehaviorScopeYTrainGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("BehaviorScope-Y Trainer")
        self.geometry("1020x860")
        self.minsize(980, 760)

        self.process: subprocess.Popen[str] | None = None
        self.log_queue: queue.Queue[str | None] = queue.Queue()

        self._make_vars()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _make_vars(self) -> None:
        self.var_manifest = tk.StringVar()
        self.var_yolo_weights = tk.StringVar()
        self.var_project = tk.StringVar(value=str(Path.cwd() / "behavior_lstm_runs"))
        self.var_name = tk.StringVar(value="behaviorscope_y_run")
        self.var_resume = tk.BooleanVar(value=False)

        self.var_epochs = tk.IntVar(value=10)
        self.var_patience = tk.IntVar(value=5)
        self.var_batch = tk.IntVar(value=24)
        self.var_num_workers = tk.IntVar(value=2)
        self.var_lr = tk.StringVar(value="5e-5")
        self.var_weight_decay = tk.StringVar(value="5e-4")
        self.var_dropout = tk.StringVar(value="0.6")
        self.var_scheduler = tk.StringVar(value="plateau")
        self.var_log_interval = tk.IntVar(value=200)
        self.var_seed = tk.IntVar(value=0)
        self.var_device = tk.StringVar(value="cuda")
        self.var_amp = tk.BooleanVar(value=True)
        self.var_amp_dtype = tk.StringVar(value="auto")

        self.var_train_backbone = tk.BooleanVar(value=False)
        self.var_backbone_lr = tk.StringVar(value="1e-5")
        self.var_yolo_backbone_end_layer = tk.IntVar(value=10)

        self.var_auto_feature_cache = tk.BooleanVar(value=True)
        self.var_use_feature_cache = tk.StringVar()
        self.var_feature_cache_batch = tk.IntVar(value=0)
        self.var_feature_cache_num_workers = tk.IntVar(value=-1)
        self.var_feature_cache_dtype = tk.StringVar(value="float32")

        self.var_train_splits = tk.StringVar(value="train")
        self.var_val_splits = tk.StringVar(value="val")
        self.var_allow_test_split_training = tk.BooleanVar(value=False)
        self.var_skip_invalid_samples = tk.BooleanVar(value=False)

        self.var_class_weighting = tk.StringVar(value="sqrt_inverse")
        self.var_class_weight_clamp = tk.StringVar(value="1.5")
        self.var_train_sampler = tk.StringVar(value="random")

        self.var_per_class_metrics = tk.BooleanVar(value=True)
        self.var_confusion_matrix = tk.BooleanVar(value=True)
        self.var_confusion_matrix_interval = tk.IntVar(value=1)
        self.var_disable_threshold_decoder = tk.BooleanVar(value=False)
        self.var_background_class = tk.StringVar(value="auto")
        self.var_threshold_grid_min = tk.StringVar(value="0.30")
        self.var_threshold_grid_max = tk.StringVar(value="0.95")
        self.var_threshold_grid_step = tk.StringVar(value="0.05")
        self.var_threshold_fit_rounds = tk.IntVar(value=3)
        self.var_save_decoder_json = tk.BooleanVar(value=False)
        # BehaviorScope-Y v3 temporal bout splitter
        self.var_disable_temporal_splitter = tk.BooleanVar(value=False)
        self.var_temporal_splitter_annot_root = tk.StringVar(value="")
        self.var_temporal_splitter_strict_annot = tk.BooleanVar(value=False)
        self.var_save_temporal_splitter_json = tk.BooleanVar(value=False)

        self.var_n_animals = tk.IntVar(value=2)
        self.var_num_keypoints = tk.IntVar(value=0)
        self.var_hidden_dim = tk.IntVar(value=256)
        self.var_num_lstm_layers = tk.IntVar(value=1)
        self.var_bidirectional_lstm = tk.BooleanVar(value=False)
        self.var_sequence_model = tk.StringVar(value="attention")
        self.var_use_attention_pool = tk.BooleanVar(value=False)
        self.var_attention_heads = tk.IntVar(value=4)
        self.var_positional_encoding = tk.StringVar(value="sinusoidal")
        self.var_pose_fusion_dim = tk.IntVar(value=128)
        self.var_pose_fusion_strategy = tk.StringVar(value="gated_attention")

        self.var_disable_visual_streams = tk.BooleanVar(value=False)
        self.var_disable_group_rgb = tk.BooleanVar(value=False)
        self.var_disable_per_animal_rgb = tk.BooleanVar(value=False)
        self.var_disable_pose_self = tk.BooleanVar(value=False)
        self.var_disable_relations = tk.BooleanVar(value=False)
        self.var_relations_pose_only = tk.BooleanVar(value=False)
        self.var_pose_dropout_p_uniform = tk.StringVar(value="0.0")

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.pack(fill=tk.BOTH, expand=True)

        tabs = ttk.Notebook(root)
        tabs.pack(fill=tk.BOTH, expand=True)

        basics = ttk.Frame(tabs, padding=10)
        model = ttk.Frame(tabs, padding=10)
        cache = ttk.Frame(tabs, padding=10)
        ablation = ttk.Frame(tabs, padding=10)
        tabs.add(basics, text="Basics")
        tabs.add(model, text="Model")
        tabs.add(cache, text="Cache + Validation")
        tabs.add(ablation, text="Ablation")

        self._build_basics_tab(basics)
        self._build_model_tab(model)
        self._build_cache_tab(cache)
        self._build_ablation_tab(ablation)

        buttons = ttk.Frame(root)
        buttons.pack(fill=tk.X, pady=(8, 4))
        ttk.Button(buttons, text="Show command", command=self._show_command).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Start training", command=self._start).pack(side=tk.LEFT, padx=6)
        ttk.Button(buttons, text="Stop after epoch", command=self._stop_after_epoch).pack(side=tk.LEFT)
        ttk.Button(buttons, text="Kill process", command=self._kill_process).pack(side=tk.LEFT, padx=6)

        self.txt_log = tk.Text(root, height=15, wrap=tk.NONE)
        self.txt_log.pack(fill=tk.BOTH, expand=False)
        yscroll = ttk.Scrollbar(self.txt_log, orient=tk.VERTICAL, command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=yscroll.set)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.status = tk.StringVar(value="Ready")
        ttk.Label(root, textvariable=self.status, anchor=tk.W).pack(fill=tk.X, pady=(4, 0))

    def _build_basics_tab(self, frame: ttk.Frame) -> None:
        self._path_row(frame, 0, "Manifest", self.var_manifest, self._pick_manifest)
        self._path_row(frame, 1, "YOLO pose weights", self.var_yolo_weights, self._pick_yolo_weights)
        self._path_row(frame, 2, "Project folder", self.var_project, self._pick_project)
        self._entry_row(frame, 3, "Run name", self.var_name, width=38)
        ttk.Checkbutton(frame, text="Resume from last checkpoint in run folder", variable=self.var_resume).grid(
            row=4, column=1, sticky="w", pady=4
        )

        params = ttk.LabelFrame(frame, text="Training", padding=10)
        params.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        params.columnconfigure(1, weight=1)
        params.columnconfigure(3, weight=1)

        self._entry_row(params, 0, "Epochs", self.var_epochs, 10, column=0)
        self._entry_row(params, 0, "Patience", self.var_patience, 10, column=2)
        self._entry_row(params, 1, "Batch", self.var_batch, 10, column=0)
        self._entry_row(params, 1, "Workers", self.var_num_workers, 10, column=2)
        self._entry_row(params, 2, "LR", self.var_lr, 10, column=0)
        self._entry_row(params, 2, "Weight decay", self.var_weight_decay, 10, column=2)
        self._entry_row(params, 3, "Dropout", self.var_dropout, 10, column=0)
        self._combo_row(params, 3, "Scheduler", self.var_scheduler, ["none", "plateau", "cosine"], column=2)
        self._entry_row(params, 4, "Seed", self.var_seed, 10, column=0)
        self._entry_row(params, 4, "Device", self.var_device, 10, column=2)
        self._entry_row(params, 5, "Log interval", self.var_log_interval, 10, column=0)
        self._combo_row(params, 5, "AMP dtype", self.var_amp_dtype, ["auto", "bf16", "fp16"], column=2)
        ttk.Checkbutton(params, text="AMP", variable=self.var_amp).grid(row=6, column=1, sticky="w", pady=4)

    def _build_model_tab(self, frame: ttk.Frame) -> None:
        arch = ttk.LabelFrame(frame, text="Architecture", padding=10)
        arch.pack(fill=tk.X)
        for col in (1, 3):
            arch.columnconfigure(col, weight=1)

        self._entry_row(arch, 0, "Animals", self.var_n_animals, 10, column=0)
        self._entry_row(arch, 0, "Keypoints (0=auto)", self.var_num_keypoints, 10, column=2)
        self._entry_row(arch, 1, "Hidden dim", self.var_hidden_dim, 10, column=0)
        self._entry_row(arch, 1, "LSTM layers", self.var_num_lstm_layers, 10, column=2)
        self._combo_row(arch, 2, "Sequence model", self.var_sequence_model, ["lstm", "attention"], column=0)
        self._entry_row(arch, 2, "Attention heads", self.var_attention_heads, 10, column=2)
        self._combo_row(arch, 3, "Positional encoding", self.var_positional_encoding, ["none", "learned", "sinusoidal"], column=0)
        self._entry_row(arch, 3, "Pose fusion dim", self.var_pose_fusion_dim, 10, column=2)
        self._combo_row(
            arch,
            4,
            "Pose fusion",
            self.var_pose_fusion_strategy,
            ["concat", "gated_attention", "cross_modal_transformer"],
            column=0,
        )
        ttk.Checkbutton(arch, text="Bidirectional LSTM", variable=self.var_bidirectional_lstm).grid(
            row=5, column=1, sticky="w", pady=4
        )
        ttk.Checkbutton(arch, text="Attention pool", variable=self.var_use_attention_pool).grid(
            row=5, column=3, sticky="w", pady=4
        )

        yolo = ttk.LabelFrame(frame, text="YOLO Feature Extractor", padding=10)
        yolo.pack(fill=tk.X, pady=(12, 0))
        yolo.columnconfigure(1, weight=1)
        yolo.columnconfigure(3, weight=1)
        self._entry_row(yolo, 0, "Backbone end layer", self.var_yolo_backbone_end_layer, 10, column=0)
        self._entry_row(yolo, 0, "Backbone LR", self.var_backbone_lr, 10, column=2)
        ttk.Checkbutton(yolo, text="Train YOLO backbone", variable=self.var_train_backbone).grid(
            row=1, column=1, sticky="w", pady=4
        )

    def _build_cache_tab(self, frame: ttk.Frame) -> None:
        cache = ttk.LabelFrame(frame, text="Feature Cache", padding=10)
        cache.pack(fill=tk.X)
        cache.columnconfigure(1, weight=1)
        ttk.Checkbutton(cache, text="Auto-build/reuse feature cache", variable=self.var_auto_feature_cache).grid(
            row=0, column=1, sticky="w", pady=4
        )
        self._path_row(cache, 1, "Existing cache dir", self.var_use_feature_cache, self._pick_feature_cache)
        self._entry_row(cache, 2, "Cache batch (0=train batch)", self.var_feature_cache_batch, 10, column=0)
        self._entry_row(cache, 2, "Cache workers (-1=train workers)", self.var_feature_cache_num_workers, 10, column=2)
        self._combo_row(cache, 3, "Cache dtype", self.var_feature_cache_dtype, ["float32", "float16"], column=0)

        data = ttk.LabelFrame(frame, text="Splits + Class Balance", padding=10)
        data.pack(fill=tk.X, pady=(12, 0))
        data.columnconfigure(1, weight=1)
        data.columnconfigure(3, weight=1)
        self._entry_row(data, 0, "Train splits", self.var_train_splits, 24, column=0)
        self._entry_row(data, 0, "Val splits", self.var_val_splits, 24, column=2)
        self._combo_row(
            data,
            1,
            "Class weights",
            self.var_class_weighting,
            ["none", "inverse", "sqrt_inverse"],
            column=0,
        )
        self._entry_row(data, 1, "Weight clamp", self.var_class_weight_clamp, 10, column=2)
        self._combo_row(data, 2, "Sampler", self.var_train_sampler, ["random", "weighted"], column=0)
        ttk.Checkbutton(data, text="Allow test split training", variable=self.var_allow_test_split_training).grid(
            row=2, column=3, sticky="w", pady=4
        )
        ttk.Checkbutton(data, text="Skip invalid samples", variable=self.var_skip_invalid_samples).grid(
            row=3, column=1, sticky="w", pady=4
        )

        diag = ttk.LabelFrame(frame, text="Validation Diagnostics + Threshold Decoder", padding=10)
        diag.pack(fill=tk.X, pady=(12, 0))
        diag.columnconfigure(1, weight=1)
        diag.columnconfigure(3, weight=1)
        ttk.Checkbutton(diag, text="Per-class metrics", variable=self.var_per_class_metrics).grid(
            row=0, column=1, sticky="w", pady=4
        )
        ttk.Checkbutton(diag, text="Confusion matrix", variable=self.var_confusion_matrix).grid(
            row=0, column=3, sticky="w", pady=4
        )
        self._entry_row(diag, 1, "Confusion interval", self.var_confusion_matrix_interval, 10, column=0)
        self._entry_row(diag, 1, "Background class", self.var_background_class, 14, column=2)
        self._entry_row(diag, 2, "Threshold min", self.var_threshold_grid_min, 10, column=0)
        self._entry_row(diag, 2, "Threshold max", self.var_threshold_grid_max, 10, column=2)
        self._entry_row(diag, 3, "Threshold step", self.var_threshold_grid_step, 10, column=0)
        self._entry_row(diag, 3, "Fit rounds", self.var_threshold_fit_rounds, 10, column=2)
        ttk.Checkbutton(diag, text="Disable threshold decoder", variable=self.var_disable_threshold_decoder).grid(
            row=4, column=1, sticky="w", pady=4
        )
        ttk.Checkbutton(diag, text="Save decoder JSON sidecar", variable=self.var_save_decoder_json).grid(
            row=4, column=3, sticky="w", pady=4
        )
        # ---- BehaviorScope-Y v3 temporal bout splitter ----
        # Auto-fits a target-class splitter on val data.
        # Annot root is OPTIONAL: if provided, manuscript-grade GT is used
        # (gt_source='annot_files'). If left blank, the splitter falls back
        # to window-label projection (gt_source='window_label_projection'),
        # which is noisier — set the annot root for publication runs.
        ttk.Label(diag, text="Temporal splitter - auto-fit on val (target='investigation')").grid(
            row=5, column=0, columnspan=4, sticky="w", pady=(8, 2)
        )
        ttk.Label(diag, text="Annot root (optional)").grid(row=6, column=0, sticky="w", pady=2)
        ttk.Entry(diag, textvariable=self.var_temporal_splitter_annot_root, width=42).grid(
            row=6, column=1, columnspan=2, sticky="we", pady=2
        )
        ttk.Button(
            diag, text="Browse",
            command=self._pick_temporal_splitter_annot_root,
        ).grid(row=6, column=3, sticky="w", padx=4)
        ttk.Checkbutton(diag, text="Disable temporal splitter", variable=self.var_disable_temporal_splitter).grid(
            row=7, column=1, sticky="w", pady=4
        )
        ttk.Checkbutton(diag, text="Save temporal splitter JSON sidecar", variable=self.var_save_temporal_splitter_json).grid(
            row=7, column=3, sticky="w", pady=4
        )
        ttk.Checkbutton(
            diag, text="Strict annot mode (skip videos missing .annot)",
            variable=self.var_temporal_splitter_strict_annot,
        ).grid(row=8, column=1, columnspan=2, sticky="w", pady=4)

    def _build_ablation_tab(self, frame: ttk.Frame) -> None:
        box = ttk.LabelFrame(frame, text="Ablation / Debug Controls", padding=10)
        box.pack(fill=tk.X)
        checks = [
            ("Disable visual streams", self.var_disable_visual_streams),
            ("Disable group RGB", self.var_disable_group_rgb),
            ("Disable per-animal RGB", self.var_disable_per_animal_rgb),
            ("Disable pose self", self.var_disable_pose_self),
            ("Disable relations", self.var_disable_relations),
            ("Relations pose only", self.var_relations_pose_only),
        ]
        for idx, (label, var) in enumerate(checks):
            ttk.Checkbutton(box, text=label, variable=var).grid(
                row=idx // 2, column=(idx % 2) * 2 + 1, sticky="w", padx=4, pady=4
            )
        self._entry_row(box, 4, "Pose dropout uniform", self.var_pose_dropout_p_uniform, 10, column=0)

    def _path_row(self, parent: ttk.Frame, row: int, label: str, var: tk.Variable, command) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2, sticky="e", padx=(8, 0), pady=4)
        parent.columnconfigure(1, weight=1)

    def _entry_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        var: tk.Variable,
        width: int = 10,
        column: int = 0,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=column + 1, sticky="ew", pady=4)

    def _combo_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        var: tk.Variable,
        values: list[str],
        column: int = 0,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, sticky="w", padx=(0, 8), pady=4)
        ttk.Combobox(parent, textvariable=var, values=values, width=16, state="readonly").grid(
            row=row, column=column + 1, sticky="ew", pady=4
        )

    def _pick_manifest(self) -> None:
        path = filedialog.askopenfilename(title="Select sequence_manifest.json", filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            self.var_manifest.set(path)

    def _pick_yolo_weights(self) -> None:
        path = filedialog.askopenfilename(title="Select YOLO pose weights", filetypes=[("PyTorch", "*.pt"), ("All", "*.*")])
        if path:
            self.var_yolo_weights.set(path)

    def _pick_project(self) -> None:
        path = filedialog.askdirectory(title="Select output project folder")
        if path:
            self.var_project.set(path)

    def _pick_feature_cache(self) -> None:
        path = filedialog.askdirectory(title="Select feature cache folder")
        if path:
            self.var_use_feature_cache.set(path)

    def _pick_temporal_splitter_annot_root(self) -> None:
        path = filedialog.askdirectory(
            title="Select MARS .annot root for temporal splitter fitting"
        )
        if path:
            self.var_temporal_splitter_annot_root.set(path)

    def _validate(self) -> bool:
        if not TRAIN_Y.exists():
            messagebox.showerror("Missing train_y.py", f"Cannot find {TRAIN_Y}")
            return False
        if not self.var_manifest.get().strip():
            messagebox.showerror("Missing manifest", "Select a sequence_manifest.json file.")
            return False
        if not self.var_yolo_weights.get().strip():
            messagebox.showerror("Missing YOLO weights", "Select a YOLO pose .pt file.")
            return False
        if self.var_train_backbone.get() and (self.var_auto_feature_cache.get() or self.var_use_feature_cache.get().strip()):
            messagebox.showerror(
                "Incompatible options",
                "Feature cache stores frozen YOLO features. Disable cache if you want to train the YOLO backbone.",
            )
            return False
        return True

    def _build_command(self) -> list[str]:
        py = sys.executable or "python"
        cmd = [
            py,
            str(TRAIN_Y),
            "--manifest_path",
            self.var_manifest.get().strip(),
            "--yolo_weights",
            self.var_yolo_weights.get().strip(),
            "--epochs",
            str(self.var_epochs.get()),
            "--patience",
            str(self.var_patience.get()),
            "--scheduler",
            self.var_scheduler.get(),
            "--batch",
            str(self.var_batch.get()),
            "--num_workers",
            str(self.var_num_workers.get()),
            "--lr",
            self.var_lr.get(),
            "--weight_decay",
            self.var_weight_decay.get(),
            "--dropout",
            self.var_dropout.get(),
            "--class_weighting",
            self.var_class_weighting.get(),
            "--class_weight_clamp",
            self.var_class_weight_clamp.get(),
            "--train_sampler",
            self.var_train_sampler.get(),
            "--n_animals",
            str(self.var_n_animals.get()),
            "--hidden_dim",
            str(self.var_hidden_dim.get()),
            "--num_lstm_layers",
            str(self.var_num_lstm_layers.get()),
            "--pose_fusion_dim",
            str(self.var_pose_fusion_dim.get()),
            "--pose_fusion_strategy",
            self.var_pose_fusion_strategy.get(),
            "--sequence_model",
            self.var_sequence_model.get(),
            "--attention_heads",
            str(self.var_attention_heads.get()),
            "--positional_encoding",
            self.var_positional_encoding.get(),
            "--yolo_backbone_end_layer",
            str(self.var_yolo_backbone_end_layer.get()),
            "--backbone_lr",
            self.var_backbone_lr.get(),
            "--log_interval",
            str(self.var_log_interval.get()),
            "--seed",
            str(self.var_seed.get()),
            "--device",
            self.var_device.get(),
            "--project",
            self.var_project.get().strip(),
            "--name",
            self.var_name.get().strip(),
            "--amp_dtype",
            self.var_amp_dtype.get(),
            "--background_class",
            self.var_background_class.get().strip() or "auto",
            "--threshold_grid_min",
            self.var_threshold_grid_min.get(),
            "--threshold_grid_max",
            self.var_threshold_grid_max.get(),
            "--threshold_grid_step",
            self.var_threshold_grid_step.get(),
            "--threshold_fit_rounds",
            str(self.var_threshold_fit_rounds.get()),
            "--confusion_matrix_interval",
            str(self.var_confusion_matrix_interval.get()),
        ]

        train_splits = _split_items(self.var_train_splits.get())
        if train_splits:
            cmd.append("--train_splits")
            cmd.extend(train_splits)
        val_splits = _split_items(self.var_val_splits.get())
        if val_splits:
            cmd.append("--val_splits")
            cmd.extend(val_splits)

        if self.var_num_keypoints.get() > 0:
            cmd.extend(["--num_keypoints", str(self.var_num_keypoints.get())])
        if self.var_resume.get():
            cmd.append("--resume")
        if self.var_amp.get():
            cmd.append("--amp")
        if self.var_train_backbone.get():
            cmd.append("--train_backbone")
        if self.var_bidirectional_lstm.get():
            cmd.append("--bidirectional_lstm")
        if self.var_use_attention_pool.get():
            cmd.append("--use_attention_pool")
        if self.var_auto_feature_cache.get():
            cmd.append("--auto_feature_cache")
            cmd.extend(["--feature_cache_batch", str(self.var_feature_cache_batch.get())])
            cmd.extend(["--feature_cache_num_workers", str(self.var_feature_cache_num_workers.get())])
            cmd.extend(["--feature_cache_dtype", self.var_feature_cache_dtype.get()])
        if self.var_use_feature_cache.get().strip():
            cmd.extend(["--use_feature_cache", self.var_use_feature_cache.get().strip()])
        if self.var_allow_test_split_training.get():
            cmd.append("--allow_test_split_training")
        if self.var_skip_invalid_samples.get():
            cmd.append("--skip_invalid_samples")
        if self.var_per_class_metrics.get():
            cmd.append("--per_class_metrics")
        if self.var_confusion_matrix.get():
            cmd.append("--confusion_matrix")
        if self.var_disable_threshold_decoder.get():
            cmd.append("--disable_threshold_decoder")
        # Temporal splitter
        if self.var_disable_temporal_splitter.get():
            cmd.append("--disable_temporal_splitter")
        if self.var_temporal_splitter_annot_root.get().strip():
            cmd.extend([
                "--temporal_splitter_annot_root",
                self.var_temporal_splitter_annot_root.get().strip(),
            ])
        if self.var_temporal_splitter_strict_annot.get():
            cmd.append("--temporal_splitter_strict_annot")
        if self.var_save_temporal_splitter_json.get():
            cmd.append("--save_temporal_splitter_json")
        if self.var_save_decoder_json.get():
            cmd.append("--save_decoder_json")
        if self.var_disable_visual_streams.get():
            cmd.append("--disable_visual_streams")
        if self.var_disable_group_rgb.get():
            cmd.append("--disable_group_rgb")
        if self.var_disable_per_animal_rgb.get():
            cmd.append("--disable_per_animal_rgb")
        if self.var_disable_pose_self.get():
            cmd.append("--disable_pose_self")
        if self.var_disable_relations.get():
            cmd.append("--disable_relations")
        if self.var_relations_pose_only.get():
            cmd.append("--relations_pose_only")
        if float(self.var_pose_dropout_p_uniform.get()) > 0:
            cmd.extend(["--pose_dropout_p_uniform", self.var_pose_dropout_p_uniform.get()])
        return cmd

    def _show_command(self) -> None:
        if not self._validate():
            return
        cmd = " ".join(_shellquote(part) for part in self._build_command())
        self.txt_log.insert(tk.END, "\n" + cmd + "\n")
        self.txt_log.see(tk.END)

    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo("Training already running", "A training process is already running.")
            return
        if not self._validate():
            return

        self.txt_log.delete("1.0", tk.END)
        cmd = self._build_command()
        self.status.set("Starting training...")
        try:
            self.process = subprocess.Popen(
                cmd,
                cwd=str(Path.cwd()),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
            )
        except Exception as exc:
            self.process = None
            messagebox.showerror("Launch failed", str(exc))
            self.status.set("Launch failed")
            return

        threading.Thread(target=self._reader_thread, daemon=True).start()
        self.after(100, self._drain_log_queue)
        self.status.set("Training running")

    def _reader_thread(self) -> None:
        assert self.process is not None
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self.log_queue.put(line)
        rc = self.process.wait()
        self.log_queue.put(f"\n[process exited with code {rc}]\n")
        self.log_queue.put(None)

    def _drain_log_queue(self) -> None:
        while True:
            try:
                item = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self.status.set("Finished")
                return
            self.txt_log.insert(tk.END, item)
            self.txt_log.see(tk.END)
        if self.process is not None and self.process.poll() is None:
            self.after(150, self._drain_log_queue)
        else:
            self.status.set("Finished")

    def _stop_after_epoch(self) -> None:
        project = Path(self.var_project.get().strip())
        name = self.var_name.get().strip()
        if not name:
            messagebox.showerror("Missing run name", "Run name is required.")
            return
        run_dir = project / name
        run_dir.mkdir(parents=True, exist_ok=True)
        flag = run_dir / "stop_training.flag"
        flag.write_text("stop after current epoch\n", encoding="utf-8")
        self.txt_log.insert(tk.END, f"\n[requested stop after epoch via {flag}]\n")
        self.txt_log.see(tk.END)

    def _kill_process(self) -> None:
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.status.set("Terminated")

    def _on_close(self) -> None:
        if self.process is not None and self.process.poll() is None:
            if not messagebox.askyesno("Training is running", "Close the GUI and terminate training?"):
                return
            self.process.terminate()
        self.destroy()


if __name__ == "__main__":
    app = BehaviorScopeYTrainGUI()
    app.mainloop()
