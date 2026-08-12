import numpy as np

from scripts.diagnose_controller_interaction import _opposition_metrics


def test_opposition_metrics_detects_180_degree_pair():
    dt = 0.01
    t = np.arange(0.0, 5.0, dt)
    base = np.column_stack([
        0.20 * np.sin(2*np.pi*3.0*t),
        0.15 * np.sin(2*np.pi*2.0*t),
        0.02 * np.sin(2*np.pi*1.0*t),
    ])
    residual = -base
    out = _opposition_metrics(base, residual, dt, transient_ignore_s=0.0, active_threshold=0.001)
    assert out["aggregate_opposite_sign_fraction_when_active"] > 0.99
    assert out["aggregate_cosine_similarity_when_active"] < -0.99
    mx = out["axes"][0]
    assert abs(mx["dominant_common_frequency_hz"] - 3.0) < 0.05
    assert abs(abs(mx["phase_base_minus_residual_deg"]) - 180.0) < 1.0
