"""Residual dynamics model: predicts residual acceleration from a state history.

Input : a flattened history window of [R(9), v(3), omega(3), u_total(4)] frames
        (H_dyn x 19), NO position.
Output: residual acceleration a_res (6) = [translational(3), angular(3)].

The history lets the network implicitly identify the disturbance parameter k
from how the state has evolved under known controls, so it can predict a sharp,
k-specific residual rather than a washed-out mean (recall the residual is
opposite-signed for k<1 vs k>1, so the unconditional mean partially cancels).

An optional spectral-norm wrapping (spectral_norm=True) constrains each linear
layer's largest singular value, bounding the network's Lipschitz constant. This
tends to improve generalization beyond the training distribution -- useful both
for prediction and for a cleaner out-of-distribution prediction-error signal. It
is OFF by default (plain MSE first); enable it once the baseline is understood.
"""

import torch
import torch.nn as nn


def _linear(in_dim, out_dim, spectral_norm=False):
    layer = nn.Linear(in_dim, out_dim)
    if spectral_norm:
        layer = nn.utils.spectral_norm(layer)
    return layer


class ResidualDynamics(nn.Module):
    def __init__(self, history, frame_dim=19, out_dim=6, hidden=(256, 256),
                 spectral_norm=False):
        super().__init__()
        self.history = history
        self.frame_dim = frame_dim
        self.in_dim = history * frame_dim

        layers = []
        last = self.in_dim
        for h in hidden:
            layers.append(_linear(last, h, spectral_norm))
            layers.append(nn.ReLU())
            last = h
        layers.append(_linear(last, out_dim, spectral_norm))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, history, frame_dim) or (B, history*frame_dim)
        if x.dim() == 3:
            x = x.reshape(x.shape[0], -1)
        return self.net(x)