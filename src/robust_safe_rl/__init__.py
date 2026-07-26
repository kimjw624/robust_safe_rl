"""robust_safe_rl: a residual-RL quadrotor framework with OOD-gated residuals.

Layout:
- ``core`` : quadrotor dynamics, geometric SE(3) controller, trajectory, SO(3) math.
- ``ood``  : out-of-distribution detection (autoencoder over residual features).
- ``scripts`` : runnable entry points (tracking demo, train/test/eval/sweep).

Motivation: a residual RL controller improves on a base geometric controller
under disturbances, but degrades when states leave the training distribution.
This package builds the pieces to detect that OOD condition so the residual can
be gated (disabled / fall back to the base controller) outside its valid domain.
"""

__version__ = "0.1.0"
