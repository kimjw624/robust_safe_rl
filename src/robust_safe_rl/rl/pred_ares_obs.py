"""Predicted-residual-acceleration observation builder for the residual policy.

This wires the frozen (supervised, MSE-pretrained) residual-dynamics predictor
f_res into the policy's observation. At each step it maintains the predictor's
own [R, v, omega, u_total] history, runs f_res to get the predicted residual
acceleration a_res_pred (6-dim), and exposes a short history of those predictions
as the policy observation.

This is the "a_res-only" test: the policy sees ONLY the sequence of predicted
residual accelerations (optionally paired with the residual actions it took), and
must learn to produce corrective residual wrenches from that alone -- no raw
state, no tracking error. It answers whether the residual-dynamics signal is by
itself sufficient to drive good tracking actions.

Observation layout (obs_mode = "pred_ares"):
    [ a_res_pred[t-H+1], ..., a_res_pred[t] ]                 (H x 6)
  optionally interleaved with the residual action that preceded each:
    [ (a_res_pred, u_res)[t-H+1], ..., (a_res_pred, u_res)[t] ]  (H x 10)

The predictor is FROZEN (eval mode, no grad). Its internal [R,v,omega,u_total]
history is separate from the policy's a_res history and is filled from the true
plant each step.
"""

import numpy as np
import torch

from robust_safe_rl.rl.residual_dynamics import ResidualDynamics


ARES_DIM = 6          # predicted residual acceleration [trans(3), ang(3)]
ACTION_DIM = 4
DYN_FRAME_DIM = 19    # [R(9), v(3), omega(3), u_total(4)] fed to f_res


class PredAresObsBuilder:
    """Observation = short history of predicted a_res (optionally + action)."""

    def __init__(self, cfg, device="cpu"):
        self.cfg = cfg
        # fall back to CPU if cuda is requested but unavailable, so loading a
        # checkpoint never crashes on a CPU-only machine
        if str(device) == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.hist = int(getattr(cfg, "ares_hist", 5))
        self.include_action = bool(getattr(cfg, "ares_include_action", False))

        # load and freeze the pretrained residual-dynamics predictor
        ckpt = torch.load(cfg.dyn_ckpt_path, map_location=self.device, weights_only=False)
        self.model = ResidualDynamics(history=ckpt["history"], hidden=tuple(ckpt["hidden"]),
                                      spectral_norm=ckpt.get("spectral_norm", False)).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.dyn_hist = int(ckpt["history"])
        self.y_mean = torch.as_tensor(ckpt["y_mean"], device=self.device)
        self.y_std = torch.as_tensor(ckpt["y_std"], device=self.device)

        per = ARES_DIM + (ACTION_DIM if self.include_action else 0)
        self.obs_dim = self.hist * per
        self.reset()

    def reset(self):
        # predictor's own input history (19-dim frames)
        self._dyn_buf = np.zeros((self.dyn_hist, DYN_FRAME_DIM), dtype=np.float64)
        # policy's a_res(+action) history
        per = ARES_DIM + (ACTION_DIM if self.include_action else 0)
        self._ares_buf = np.zeros((self.hist, per), dtype=np.float64)

    def _predict_ares(self, true_state, u_total):
        frame = np.concatenate([true_state["R"].reshape(9), true_state["v"],
                                true_state["omega"], np.asarray(u_total).reshape(4)])
        self._dyn_buf = np.roll(self._dyn_buf, -1, axis=0)
        self._dyn_buf[-1] = frame
        x = torch.as_tensor(self._dyn_buf.reshape(1, -1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            pred = self.model(x) * self.y_std + self.y_mean     # physical units
        return pred.squeeze(0).cpu().numpy()

    def push(self, true_state, u_total, u_res):
        """Update with the current step. true_state/u_total are used to run the
        frozen predictor; u_res is the residual action that was applied."""
        a_pred = self._predict_ares(true_state, u_total)
        entry = a_pred if not self.include_action else np.concatenate([a_pred, np.asarray(u_res).reshape(4)])
        self._ares_buf = np.roll(self._ares_buf, -1, axis=0)
        self._ares_buf[-1] = entry

    def get(self):
        return self._ares_buf.reshape(-1).astype(np.float32)