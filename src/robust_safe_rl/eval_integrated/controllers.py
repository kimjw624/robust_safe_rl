"""Controller cases for the integrated evaluation.

All four share the geometric base controller and (except base-only) the trained
residual policy. They differ in whether/when the residual is applied:

  BaseOnly            : u = u_base
  BaseResidual        : u = u_base + residual(obs)          (always)
  BaseResidualDetector: u = u_base + residual(obs) UNTIL the OOD detector latches,
                        then u = u_base for the rest of the episode.

The detector-gated case uses a latched confirmation window: it fires only after
`confirm_window` consecutive OOD detections (avoiding chatter from a noisy
score), and once fired it STAYS fired for the episode (latched). On firing, the
fallback zeroes the residual (fall-back-to-base). A `freeze` mode is available
as a future variant: instead of zeroing, hold the last pre-OOD residual.

The residual policy consumes the SAME observation the env produces (obs_mode
from the trained policy's config). Detectors are fed separately by the eval
loop, which has direct access to the twin states and accelerations.
"""

import numpy as np
import torch

from robust_safe_rl.rl.networks import Actor


class ResidualPolicy:
    """Thin wrapper around a trained SAC actor for greedy inference."""

    def __init__(self, checkpoint, obs_dim, action_dim, hidden, action_scale,
                 device="cpu"):
        self.device = torch.device(device)
        self.actor = Actor(obs_dim, action_dim, hidden=tuple(hidden)).to(self.device)
        sd = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(sd["actor"])
        self.actor.eval()
        self.action_scale = np.asarray(action_scale, dtype=np.float64)

    @torch.no_grad()
    def residual(self, obs):
        """Greedy (deterministic) residual in physical units."""
        o = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        _, _, mean = self.actor.sample(o)
        a = mean.squeeze(0).cpu().numpy()          # tanh-squashed in [-1,1]
        return np.clip(a, -1.0, 1.0) * self.action_scale


class LatchedDetectorGate:
    """Latched OOD gate with a confirmation window.

    fire only after `confirm_window` consecutive is_ood()==True; once fired,
    stays fired for the episode. Records the step index at which it fired.
    """

    def __init__(self, confirm_window=10):
        self.confirm_window = int(confirm_window)
        self.reset()

    def reset(self):
        self._consec = 0
        self.fired = False
        self.fired_step = None

    def update(self, is_ood_now, step_idx):
        if self.fired:
            return True
        self._consec = self._consec + 1 if is_ood_now else 0
        if self._consec >= self.confirm_window:
            self.fired = True
            self.fired_step = step_idx
        return self.fired