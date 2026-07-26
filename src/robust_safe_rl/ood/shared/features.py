"""Transition-aligned residual features for disturbance / OOD detection.

Each feature vector compares a *true* (possibly disturbed) rollout against a
*nominal* rollout that receives the exact same commanded action over the same
step. Pairing the action at time t with the resulting state discrepancy at t+1
ensures that what the autoencoder sees is the effect of the disturbance, not a
controller/action mismatch.

A single sample stacks ``history_len`` consecutive 16-D step features:

    [ ex(3), ev(3), eR(3), eomega(3), action(f, Mx, My, Mz)(4) ]

with the error sign convention ``nominal_next - true_next`` and the relative
attitude encoded as ``Log(R_true_next^T @ R_nominal_next)``.
"""

import numpy as np

from robust_safe_rl.core.so3 import so3_log_vector

FEATURE_VERSION = 2
STEP_FEATURE_DIM = 16
FEATURE_NAMES = (
    "position_error_next[3]",
    "velocity_error_next[3]",
    "relative_rotation_vector_next[3]",
    "angular_velocity_error_next[3]",
    "commanded_action[f,Mx,My,Mz][4]",
)


def make_step_feature(nominal_next_state, true_next_state, action):
    """Build one transition-aligned 16-D feature vector.

    The same commanded action is applied to both systems over [t, t+dt]. The
    feature pairs that action with the state discrepancy observed at t+dt. This
    prevents controller/action mismatch from being mislabeled as a disturbance.
    """
    action = np.asarray(action, dtype=float).reshape(4)

    ex = nominal_next_state["x"] - true_next_state["x"]
    ev = nominal_next_state["v"] - true_next_state["v"]

    # Rotation taking the true attitude into the nominal attitude.
    relative_R = true_next_state["R"].T @ nominal_next_state["R"]
    eR = so3_log_vector(relative_R)

    eomega = nominal_next_state["omega"] - true_next_state["omega"]

    feature = np.concatenate((ex, ev, eR, eomega, action)).astype(np.float64)
    if feature.shape != (STEP_FEATURE_DIM,):
        raise RuntimeError(f"Unexpected step feature shape: {feature.shape}")
    if not np.all(np.isfinite(feature)):
        raise FloatingPointError("Non-finite value found in a step feature.")
    return feature


def feature_metadata(history_len):
    """Return the metadata block stored alongside every checkpoint.

    Downstream test/eval scripts assert on ``feature_version`` and
    ``history_len`` to guarantee they are running against compatible features.
    """
    history_len = int(history_len)
    return {
        "feature_version": FEATURE_VERSION,
        "history_len": history_len,
        "step_feature_dim": STEP_FEATURE_DIM,
        "input_dim": STEP_FEATURE_DIM * history_len,
        "feature_names": list(FEATURE_NAMES),
        "error_sign": "nominal_minus_true",
        "transition_alignment": "action_t_with_state_error_t_plus_1",
        "action_source": "single_true_state_feedback_command_applied_to_both_models",
        "rotation_error": "Log(R_true_next.T @ R_nominal_next)",
    }
