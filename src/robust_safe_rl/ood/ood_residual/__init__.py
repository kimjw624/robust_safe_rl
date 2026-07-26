"""Detector 2: residual-dynamics OOD detector (placeholder — not yet implemented).

Planned approach: train a model to predict the next residual state from a
history of states and actions. In distribution the one-step prediction is
accurate; out of distribution the true next state and the predicted residual
diverge, and that prediction error is the OOD signal.

This subpackage will reuse ``robust_safe_rl.ood.shared`` (features, datasets,
metrics, plots) so the two detectors remain directly comparable. Modules are
added here step by step in the next stage of the project.
"""
