"""Optional latent-space regularizers for the autoencoder.

Two penalties are supported on top of the reconstruction loss:

- ``sparse``: an L1 activity penalty on the (linear) latent code, encouraging
  each sample to use only a few latent coordinates.
- ``contractive``: a Hutchinson estimate of ||dz/dx||_F^2, penalizing latent
  sensitivity to input perturbations. Estimated on a subset of each mini-batch
  for speed and divided by the latent dimension to decouple the coefficient
  from the bottleneck width.

The hope is that a tighter / more structured ID latent manifold sharpens the
reconstruction-error gap between in- and out-of-distribution inputs.
"""

import torch


def regularization_penalty(model, xb, z, regularizer, contractive_samples):
    """Return the scalar regularization penalty for the given mini-batch.

    Parameters
    ----------
    model : AutoEncoder used to re-encode a subset for the contractive term.
    xb : input mini-batch (requires_grad for the contractive penalty).
    z : latent codes for ``xb`` (used directly by the sparse penalty).
    regularizer : one of {"none", "sparse", "contractive"}.
    contractive_samples : number of rows to use for the Hutchinson estimate.
    """
    if regularizer == "none":
        return z.new_zeros(())

    if regularizer == "sparse":
        # L1 activity penalty: encourages each sample to use only a small subset
        # of latent coordinates. The latent layer is linear, so this works with
        # signed disturbance representations.
        return z.abs().mean()

    if regularizer == "contractive":
        # Hutchinson estimate of ||dz/dx||_F^2. For speed, estimate it on a
        # subset of each mini-batch. Division by latent dimension makes the
        # coefficient less sensitive to the chosen bottleneck width.
        n = min(int(contractive_samples), xb.shape[0])
        x_sub = xb[:n]
        z_sub = model.encode(x_sub)
        probe = torch.empty_like(z_sub).bernoulli_(0.5).mul_(2.0).sub_(1.0)
        projected = (z_sub * probe).sum()
        grad_x = torch.autograd.grad(
            projected,
            x_sub,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return grad_x.square().sum(dim=1).mean() / z_sub.shape[1]

    raise ValueError(f"Unsupported regularizer: {regularizer}")
