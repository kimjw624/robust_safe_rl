"""Unified OOD-detector wrappers for the integrated evaluation.

Two detector types share one interface so the combined controller can treat them
interchangeably:

  * AEDetector           -- autoencoder reconstruction error on the transition-
                            aligned 16-D discrepancy+action feature history.
  * DynamicsDetector     -- residual-dynamics prediction error on the [R,v,omega,
                            u_total] 19-D history (no position).

Both expose:
    reset()                          clear the internal history buffer
    score(...)                       push one step, return a scalar OOD score
    threshold                        the fixed detection threshold (saved p99)
    is_ood(score) -> bool            score > threshold

Each maintains its OWN history buffer with its OWN feature contract, because the
two detectors consume different inputs. The combined controller feeds both the
raw pieces they need and lets each build its own window.
"""

import numpy as np
import torch

from robust_safe_rl.core.so3 import so3_log_vector
from robust_safe_rl.ood.ood_autoencoder.autoencoder import AutoEncoder
from robust_safe_rl.rl.residual_dynamics import ResidualDynamics


# ----------------------------------------------------------------- AE detector
AE_STEP_DIM = 16  # [ex(3), ev(3), eR(3), eomega(3), action(4)]


class AEDetector:
    """Autoencoder reconstruction-error detector.

    Feature per step (matching shared/features.make_step_feature):
        [ nominal_next.x - true_next.x,
          nominal_next.v - true_next.v,
          Log(R_true_next^T @ R_nominal_next),
          nominal_next.omega - true_next.omega,
          commanded_action(f,Mx,My,Mz) ]
    A sample is `history_len` such steps, flattened, standardized by the saved
    (mean, std), then scored by mean-squared reconstruction error.
    """

    name = "autoencoder"

    def __init__(self, checkpoint, device="cpu", threshold_key="id_p99"):
        self.device = torch.device(device)
        model, mean, std, thresholds, metadata = AutoEncoder.load(
            checkpoint, map_location=str(self.device))
        self.model = model.to(self.device).eval()
        self.mean = None if mean is None else mean.to(self.device)
        self.std = None if std is None else std.to(self.device)
        self.history_len = int(metadata.get("history_len",
                               self.model.input_dim // AE_STEP_DIM))
        self.threshold = float(thresholds.get(threshold_key,
                               thresholds.get("id_max", np.inf)))
        self._buf = None
        self.reset()

    def reset(self):
        self._buf = np.zeros((self.history_len, AE_STEP_DIM), dtype=np.float64)
        self._filled = 0

    def _step_feature(self, nom_next, true_next, action):
        ex = nom_next["x"] - true_next["x"]
        ev = nom_next["v"] - true_next["v"]
        eR = so3_log_vector(true_next["R"].T @ nom_next["R"])
        ew = nom_next["omega"] - true_next["omega"]
        return np.concatenate([ex, ev, eR, ew, np.asarray(action).reshape(4)])

    def score(self, nom_next, true_next, action):
        f = self._step_feature(nom_next, true_next, action)
        self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1] = f
        self._filled = min(self._filled + 1, self.history_len)

        x = torch.as_tensor(self._buf.reshape(1, -1), dtype=torch.float32,
                            device=self.device)
        if self.mean is not None:
            x = (x - self.mean) / (self.std + 1e-8)
        with torch.no_grad():
            err, _ = self.model.reconstruction_error(x, reduction="none")
        return float(err.item())

    def warmed_up(self):
        return self._filled >= self.history_len

    def is_ood(self, score):
        return score > self.threshold


# ------------------------------------------------------ residual-dyn detector
DYN_FRAME_DIM = 19  # [R(9), v(3), omega(3), u_total(4)]


class DynamicsDetector:
    """Residual-dynamics prediction-error detector.

    Feature per step: [R(9), v(3), omega(3), u_total(4)] (no position). Target
    residual acceleration a_res is computed from the environment's true and
    nominal accelerations at the same state; the score is the L2 error between
    the model's prediction and that target.
    """

    name = "dynamics"

    def __init__(self, checkpoint, threshold, device="cpu"):
        self.device = torch.device(device)
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model = ResidualDynamics(history=ckpt["history"],
                                      hidden=tuple(ckpt["hidden"]),
                                      spectral_norm=ckpt.get("spectral_norm", False)
                                      ).to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.history_len = int(ckpt["history"])
        self.y_mean = torch.as_tensor(ckpt["y_mean"], device=self.device)
        self.y_std = torch.as_tensor(ckpt["y_std"], device=self.device)
        self.threshold = float(threshold)
        self._buf = None
        self.reset()

    def reset(self):
        self._buf = np.zeros((self.history_len, DYN_FRAME_DIM), dtype=np.float64)
        self._filled = 0

    def score(self, true_state, u_total, a_res_true):
        """Push one step and return prediction error vs the true residual accel.

        true_state : dict with R, v, omega (the state the accel was measured at)
        u_total    : (4,) total control applied
        a_res_true : (6,) measured residual acceleration [trans(3), ang(3)]
        """
        frame = np.concatenate([true_state["R"].reshape(9), true_state["v"],
                                true_state["omega"], np.asarray(u_total).reshape(4)])
        self._buf = np.roll(self._buf, -1, axis=0)
        self._buf[-1] = frame
        self._filled = min(self._filled + 1, self.history_len)

        x = torch.as_tensor(self._buf.reshape(1, -1), dtype=torch.float32,
                            device=self.device)
        with torch.no_grad():
            pred = self.model(x) * self.y_std + self.y_mean   # physical units
        target = torch.as_tensor(np.asarray(a_res_true).reshape(1, 6),
                                 dtype=torch.float32, device=self.device)
        return float(torch.norm(pred - target, dim=1).item())

    def warmed_up(self):
        return self._filled >= self.history_len

    def is_ood(self, score):
        return score > self.threshold