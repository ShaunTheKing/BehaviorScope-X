"""BehaviorScope-Y helpers for training and running behavior classifiers."""

from .app_metadata import APP_NAME, APP_SLUG, APP_VERSION
from .data_x import (
    MultiAnimalSequenceDataset,
    collate_multi_animal,
    compute_class_weights,
    load_n_manifest,
)
from .model_x import MultiAnimalBehaviorSequenceClassifier, YOLOFrameEncoder

__all__ = [
    "APP_NAME",
    "APP_SLUG",
    "APP_VERSION",
    "MultiAnimalSequenceDataset",
    "collate_multi_animal",
    "compute_class_weights",
    "load_n_manifest",
    "MultiAnimalBehaviorSequenceClassifier",
    "YOLOFrameEncoder",
]
