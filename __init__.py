"""
Utilities for training and running the flexible behavior classifier (LSTM,
attention pooling, and SlowFast-style video backbone).

Exposes dataset helpers and the BehaviorSequenceClassifier model so scripts can
import them without reaching into module internals.
"""

from .data_y import (
    MultiAnimalSequenceDataset,
    collate_multi_animal,
    compute_class_weights,
    load_n_manifest,
)
from .model_y import MultiAnimalBehaviorSequenceClassifier, YOLOFrameEncoder

__all__ = [
    "MultiAnimalSequenceDataset",
    "collate_multi_animal",
    "compute_class_weights",
    "load_n_manifest",
    "MultiAnimalBehaviorSequenceClassifier",
    "YOLOFrameEncoder",
]
