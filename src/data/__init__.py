"""Data layer: acquisition, validation, storage, and feature computation."""

from .acquisition import acquire_dataset
from .storage import (
    ContrailSample,
    ContrailSampleStore,
    assign_complexity_groups,
    build_samples_from_dir,
)
from .validation import validate_dataset, validate_sample

__all__ = [
    "acquire_dataset",
    "assign_complexity_groups",
    "build_samples_from_dir",
    "ContrailSample",
    "ContrailSampleStore",
    "validate_dataset",
    "validate_sample",
]
