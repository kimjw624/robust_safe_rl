"""Out-of-distribution detection layer.

Organized as a detector-agnostic ``shared`` layer plus one subpackage per
detector:

- ``shared``          : residual features, rollout datasets, metrics, plots, io.
- ``ood_autoencoder`` : reconstruction-autoencoder detector (implemented).
- ``ood_residual``    : residual-dynamics one-step-prediction detector (planned)
                        -- predicts the next residual state from a history of
                        states and actions; large prediction error flags OOD.

The ``shared`` names are re-exported here for convenience, so both detectors
and the scripts can ``from robust_safe_rl.ood import ...`` the common pieces.
"""

from .shared import (
    FEATURE_VERSION,
    STEP_FEATURE_DIM,
    FEATURE_NAMES,
    make_step_feature,
    feature_metadata,
    collect_dataset,
    collect_parameter_dataset,
    standardize,
    apply_standardization,
    NOMINAL_MASS,
    NOMINAL_J,
    evaluate_errors,
    detection_report,
    binary_curves,
    compute_confusion_matrix,
    threshold_metrics,
    summarize,
    get_next_indexed_path,
    get_checkpoint_by_index,
)
from .ood_autoencoder import AutoEncoder, regularization_penalty

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
    "AutoEncoder",
    "regularization_penalty",
]
