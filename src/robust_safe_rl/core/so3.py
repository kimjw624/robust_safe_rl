"""SO(3) / rotation math shared across the simulator, controller, and features.

This module centralizes the small lie-group helpers that were previously
duplicated in ``dynamics.py``, ``controller.py``, and ``train_autoencoder.py``.
Keeping a single implementation avoids subtle sign or convention drift between
the plant, the controller, and the feature extractor.
"""

import numpy as np


def hat(w):
    """Map a 3-vector to its skew-symmetric matrix, so that hat(w) @ v == w x v."""
    wx, wy, wz = w
    return np.array([
        [0.0, -wz, wy],
        [wz, 0.0, -wx],
        [-wy, wx, 0.0],
    ])


def vee(S):
    """Inverse of :func:`hat`: extract the 3-vector from a skew-symmetric matrix."""
    return np.array([
        -S[1, 2],
        S[0, 2],
        -S[0, 1],
    ])


def project_to_so3(R):
    """Project a near-rotation matrix onto SO(3) via SVD (orthogonalization).

    Guarantees a proper rotation (det == +1) even if the input has drifted from
    orthonormality due to numerical integration error.
    """
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt

    if np.linalg.det(R) < 0.0:
        U[:, -1] *= -1.0
        R = U @ Vt

    return R


def rotation_error(R, Rd):
    """Geometric attitude error e_R = 0.5 * vee(Rd^T R - R^T Rd)."""
    return 0.5 * vee(Rd.T @ R - R.T @ Rd)


def so3_log(R):
    """Matrix logarithm of R, returned as a skew-symmetric matrix.

    Used by the controller to finite-difference the desired attitude. See
    :func:`so3_log_vector` for the version that returns the rotation vector.
    """
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-6:
        return 0.5 * (R - R.T)

    return (theta / (2.0 * np.sin(theta))) * (R - R.T)


def so3_log_vector(R):
    """Return the rotation vector (axis * angle) whose exponential is R.

    Includes a numerically stable branch for angles near pi, where the naive
    ``theta / (2 sin theta)`` formula becomes singular. This is used by the
    feature extractor to encode the relative attitude between the true and
    nominal models.
    """
    R = np.asarray(R, dtype=float).reshape(3, 3)
    cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(cos_theta))

    if theta < 1e-7:
        return 0.5 * vee(R - R.T)

    if np.pi - theta < 1e-5:
        # Stable axis extraction near pi.
        A = 0.5 * (R + np.eye(3))
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        axis[0] = np.copysign(axis[0], R[2, 1] - R[1, 2])
        axis[1] = np.copysign(axis[1], R[0, 2] - R[2, 0])
        axis[2] = np.copysign(axis[2], R[1, 0] - R[0, 1])
        norm = np.linalg.norm(axis)
        if norm < 1e-8:
            return np.zeros(3)
        return theta * axis / norm

    return theta * vee(R - R.T) / (2.0 * np.sin(theta))


def normalize(v, fallback=None, eps=1e-9):
    """Normalize a vector, returning ``fallback`` if it is near zero."""
    v = np.asarray(v, dtype=float).reshape(-1)
    n = np.linalg.norm(v)

    if n > eps:
        return v / n

    if fallback is None:
        raise ValueError("Cannot normalize near-zero vector")

    return np.asarray(fallback, dtype=float).reshape(v.shape)
