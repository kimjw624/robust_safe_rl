"""Detector-agnostic building blocks shared by every OOD detector.

Both ``ood_autoencoder`` and the forthcoming ``ood_residual`` detector reuse
these: the residual feature definition, rollout collection (force and mass/MOI
variants) plus standardization, reconstruction/score metrics and ROC-PR curves,
and diagnostic plotting.
"""

from .features import (
    FEATURE_VERSION,
    STEP_FEATURE_DIM,
    FEATURE_NAMES,
    make_step_feature,
    feature_metadata,
)
from .dataset import (
    collect_dataset,
    collect_parameter_dataset,
    standardize,
    apply_standardization,
    NOMINAL_MASS,
    NOMINAL_J,
)
from .metrics import (
    evaluate_errors,
    detection_report,
    binary_curves,
    compute_confusion_matrix,
    threshold_metrics,
    summarize,
)
from .io_utils import get_next_indexed_path, get_checkpoint_by_index

__all__ = [
    "FEATURE_VERSION",
    "STEP_FEATURE_DIM",
    "FEATURE_NAMES",
    "make_step_feature",
    "feature_metadata",
    "collect_dataset",
    "collect_parameter_dataset",
    "standardize",
    "apply_standardization",
    "NOMINAL_MASS",
    "NOMINAL_J",
    "evaluate_errors",
    "detection_report",
    "binary_curves",
    "compute_confusion_matrix",
    "threshold_metrics",
    "summarize",
    "get_next_indexed_path",
    "get_checkpoint_by_index",
]
