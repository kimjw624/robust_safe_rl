"""Reconstruction-autoencoder OOD detector.

An MLP autoencoder is trained on in-distribution history-stacked residual
features. At test time a sample is flagged OOD by two signals: (a) its
reconstruction error exceeding the ID outlier reconstruction error, and (b)
its position in the latent space relative to the ID latent cluster.

Empirically this detector does NOT reliably separate *same-type* out-of-range
force disturbances (strict force OOD); its separation shows up mainly for a
*different* disturbance type (mass / moment-of-inertia OOD). Detecting subtle,
same-type OOD is what motivates the residual-dynamics detector (``ood_residual``).

Submodules:
- ``autoencoder``    : the MLP autoencoder model with save/load.
- ``regularization`` : optional sparse / contractive latent penalties.
"""

from .autoencoder import AutoEncoder
from .regularization import regularization_penalty

__all__ = ["AutoEncoder", "regularization_penalty"]
