"""SAC networks: squashed-Gaussian actor and twin Q critics.

The actor outputs a tanh-squashed Gaussian; the normalized action lives in
[-1, 1]^4 and is scaled to physical residual bounds inside the environment. The
mean head starts near zero, but the stochastic policy is intentionally initialized
with a broad standard deviation for exploration.

The critics are twin Q networks with optional LayerNorm between hidden layers
(recommended: it stabilises value learning and curbs overestimation). Critic
outputs are LINEAR -- a Q value is not a probability, so no output activation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


LOG_STD_MIN_DEFAULT = -5.0
LOG_STD_MAX_DEFAULT = 2.0


def _mlp(in_dim, hidden, out_dim, layernorm=False):
    layers = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        if layernorm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers), last


class Actor(nn.Module):
    """Squashed-Gaussian policy. Produces actions in [-1, 1]^action_dim."""

    def __init__(self, obs_dim, action_dim, hidden=(512, 512),
                 log_std_min=LOG_STD_MIN_DEFAULT, log_std_max=LOG_STD_MAX_DEFAULT,
                 layernorm=False, zero_init_mean=True, initial_log_std=0.0):
        super().__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Shared trunk, then separate mean / log-std heads.
        trunk = []
        last = obs_dim
        for h in hidden:
            trunk.append(nn.Linear(last, h))
            if layernorm:
                trunk.append(nn.LayerNorm(h))
            trunk.append(nn.ReLU())
            last = h
        self.trunk = nn.Sequential(*trunk)
        self.mean_head = nn.Linear(last, action_dim)
        self.log_std_head = nn.Linear(last, action_dim)

        if zero_init_mean:
            # Deterministic mean starts near zero residual.
            nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
            nn.init.zeros_(self.mean_head.bias)

        # Make the initial stochastic policy deliberately broad and predictable:
        # log_std = initial_log_std for every observation at initialization.
        # With the default 0.0 this gives std=1 before tanh squashing.
        nn.init.zeros_(self.log_std_head.weight)
        nn.init.constant_(self.log_std_head.bias, float(initial_log_std))

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h)
        # Clamp log-std to a sane range (SpinningUp / Yarats convention).
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, obs):
        """Return (action, log_prob, tanh(mean)).

        action is tanh-squashed; log_prob includes the tanh change-of-variables
        correction. The third output is the deterministic (greedy) action, used
        for evaluation.
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)

        x = normal.rsample()                       # reparameterised sample
        y = torch.tanh(x)

        log_prob = normal.log_prob(x)
        # tanh correction: sum over action dims.
        log_prob -= torch.log(1.0 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return y, log_prob, torch.tanh(mean)


class Critic(nn.Module):
    """Single Q network: (obs, action) -> scalar Q. Linear output."""

    def __init__(self, obs_dim, action_dim, hidden=(512, 512), layernorm=True):
        super().__init__()
        self.net, _ = _mlp(obs_dim + action_dim, hidden, 1, layernorm=layernorm)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class TwinCritic(nn.Module):
    """Two independent Q networks (Q1, Q2) for the clipped-double-Q target."""

    def __init__(self, obs_dim, action_dim, hidden=(512, 512), layernorm=True):
        super().__init__()
        self.q1 = Critic(obs_dim, action_dim, hidden, layernorm)
        self.q2 = Critic(obs_dim, action_dim, hidden, layernorm)

    def forward(self, obs, action):
        return self.q1(obs, action), self.q2(obs, action)