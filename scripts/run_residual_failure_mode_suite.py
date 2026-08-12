"""Comprehensive causal failure-mode suite for the legacy direct-wrench residual SAC.

This is a diagnostic experiment harness, NOT a controller modification.
It keeps the trained checkpoint frozen and asks which mechanisms are necessary,
sufficient, or merely amplifying the high-frequency residual-moment chatter.

The suite is intentionally split into two phases:

1. ``broad``: many clean/middle/oscillatory cases, relatively cheap causal
   interventions.  This tests candidate fixes and rules out major mechanisms.
2. ``spectral``: fewer severe oscillatory cases, dense synthetic sine/square
   sweeps on all moment axes.  This estimates the closed-loop frequency response
   from injected residual moment to body rate and baseline-controller reaction.

Important interpretation rules
------------------------------
* Applied-wrench roughness is NOT sufficient evidence by itself: a deliberately
  high-frequency injected moment trivially makes the applied wrench rough.  The
  main physical-response metrics are body-rate roughness/HF energy, e_omega
  roughness, motor saturation, termination, and position RMSE.
* Exact same-seed playback is a determinism sanity check.  Causal evidence comes
  from spectral decomposition, disturbance removal, motor-headroom changes,
  live output shaping, and controller-feedback removal.
* Offline FFT low/high decomposition is only a diagnostic.  The implementable
  candidate fixes are the causal live filters/holds/slew limits.
* 50 Hz is exactly Nyquist at dt=0.01 s and is excluded from sine probes.  The
  dense sweep stops at 49 Hz.

Prerequisites
-------------
This script reuses helpers from the diagnostic patches already used in this
project:
  - scripts/run_residual_ruleout_experiments.py
  - scripts/run_omega_input_ruleout.py
  - scripts/diagnose_policy_chatter_source.py

Recommended sequence
--------------------
Broad screen across all OOD stage types::

  PYTHONPATH=src python3 -m scripts.run_residual_failure_mode_suite \
    --run_dir runs_residual/residual_sac_curriculum/trial_002 \
    --evaluation_dir runs_residual/residual_sac_curriculum/trial_002/evaluation/baseline_vs_residual_best \
    --checkpoint best --phase broad --device cuda

Then deep spectral testing on the severe cases::

  PYTHONPATH=src python3 -m scripts.run_residual_failure_mode_suite \
    --run_dir runs_residual/residual_sac_curriculum/trial_002 \
    --evaluation_dir runs_residual/residual_sac_curriculum/trial_002/evaluation/baseline_vs_residual_best \
    --checkpoint best --phase spectral --device cuda

Both phases write to ``evaluation_dir/residual_failure_mode_suite`` and can be
rerun safely.  Completed case files are skipped unless ``--overwrite`` is set.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from robust_safe_rl.rl.mixer import F_MAX, M_MAX, MAX_MOTOR_THRUST
from robust_safe_rl.rl.residual_env import ResidualTwinEnv
from scripts.evaluate_residual_report import (
    _disturbance_snapshot,
    _load_policy,
    _load_resolved_config,
    _resolve_checkpoint,
)
from scripts.run_residual_ruleout_experiments import (
    _apply_disturbance_transform,
    _find_stage_cfg,
    _max_feasible_scale,
    _preview_base,
    _verify_sampled_disturbance,
)
from scripts.run_omega_input_ruleout import OmegaObservationModifier

EPS = 1e-12
ACTION_DIM = 4
ACTION_LABELS = ("f", "mx", "my", "mz")
MOMENT_LABELS = ("mx", "my", "mz")
WRENCH_SCALE = np.concatenate(([F_MAX], np.asarray(M_MAX, dtype=float)))


# ---------------------------------------------------------------------------
# Small data utilities

def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for k in row:
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x, default=float("nan")) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def _parse_floats(text: str) -> list[float]:
    out = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("expected at least one comma-separated number")
    return out


def _parse_ints(text: str) -> list[int]:
    out = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not out:
        raise ValueError("expected at least one comma-separated integer")
    return out


def _rms_norm(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    if x.ndim == 1:
        return float(np.sqrt(np.mean(x * x)))
    return float(np.sqrt(np.mean(np.sum(x * x, axis=1))))


def _hf_ratio(signal: np.ndarray, dt: float, cutoff_hz: float) -> float:
    x = np.asarray(signal, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < 4:
        return 0.0
    x = x - np.mean(x, axis=0, keepdims=True)
    p = np.abs(np.fft.rfft(x, axis=0)) ** 2
    f = np.fft.rfftfreq(len(x), d=float(dt))
    valid = f > 0.0
    den = float(np.sum(p[valid]))
    return 0.0 if den <= EPS else float(np.sum(p[f >= float(cutoff_hz)])) / den


def _period2_metrics(x: np.ndarray) -> tuple[float, float, float]:
    """Return lag-1 corr, lag-2 corr, sign-alternation fraction for a 1-D signal."""
    y = np.asarray(x, dtype=float)
    if len(y) < 5 or float(np.std(y)) <= 1e-12:
        return 0.0, 0.0, 0.0
    y = y - np.mean(y)
    def corr(lag: int) -> float:
        a, b = y[:-lag], y[lag:]
        den = float(np.linalg.norm(a) * np.linalg.norm(b))
        return 0.0 if den <= EPS else float(np.dot(a, b) / den)
    alt = float(np.mean((y[:-1] * y[1:]) < 0.0))
    return corr(1), corr(2), alt


def _dominant_axis_frequency(actions: np.ndarray, dt: float, min_hz: float = 1.0) -> dict:
    a = np.asarray(actions, dtype=float)
    if len(a) < 4:
        return {"axis": 0, "frequency_hz": 0.0, "magnitude": 0.0}
    freqs = np.fft.rfftfreq(len(a), d=float(dt))
    valid = (freqs >= float(min_hz)) & (freqs < 0.5 / float(dt) - 1e-9)
    best = (-1.0, 0, 0.0)
    for j in range(3):
        x = a[:, j + 1] - np.mean(a[:, j + 1])
        mag = np.abs(np.fft.rfft(x))
        if not np.any(valid):
            continue
        idxs = np.where(valid)[0]
        k = int(idxs[np.argmax(mag[valid])])
        if float(mag[k]) > best[0]:
            best = (float(mag[k]), j, float(freqs[k]))
    return {"axis": int(best[1]), "frequency_hz": float(best[2]), "magnitude": float(best[0])}


# ---------------------------------------------------------------------------
# Case selection: stage-stratified, and enriched with paired-evaluation fields

def _case_key(row: dict) -> tuple[str, int]:
    return str(row.get("stage", "")), int(round(_f(row.get("seed"), -1)))


def _load_case_pool(evaluation_dir: Path) -> list[dict]:
    analysis = evaluation_dir / "oscillation_dataset_analysis"
    labels_path = analysis / "episode_oscillation_labels.csv"
    selected_path = analysis / "selected_cases.csv"
    rows = _read_csv(labels_path)
    if not rows:
        rows = _read_csv(selected_path)
    if not rows:
        raise FileNotFoundError(
            "need oscillation_dataset_analysis/episode_oscillation_labels.csv or selected_cases.csv"
        )

    # Enrich with exact disturbance-vector columns when available.  This makes
    # _verify_sampled_disturbance stricter without requiring them to exist.
    paired = _read_csv(evaluation_dir / "paired_episode_metrics.csv")
    if paired:
        by_key = {_case_key(r): r for r in paired if _case_key(r)[1] >= 0}
        enriched = []
        for r in rows:
            q = dict(by_key.get(_case_key(r), {}))
            q.update(r)  # preserve label/oscillation_score from mining output
            enriched.append(q)
        rows = enriched
    return rows


def _quantile_pick(rows: list[dict], n: int, score_key: str = "oscillation_score") -> list[dict]:
    if n <= 0 or not rows:
        return []
    rows = sorted(rows, key=lambda r: _f(r.get(score_key), 0.0))
    if len(rows) <= n:
        return rows
    idx = np.linspace(0, len(rows) - 1, n)
    return [rows[int(round(i))] for i in idx]


def _select_broad_cases(pool: list[dict], per_stage: dict[str, int]) -> list[dict]:
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_stage[str(r.get("stage", ""))].append(r)
    out: list[dict] = []
    for stage in sorted(by_stage):
        rr = by_stage[stage]
        for label in ("oscillatory", "middle", "clean"):
            sub = [r for r in rr if r.get("label", r.get("class", "")) == label]
            n = int(per_stage.get(label, 0))
            if not sub or n <= 0:
                continue
            if label == "oscillatory":
                chosen = sorted(sub, key=lambda r: _f(r.get("oscillation_score"), 0.0), reverse=True)[:n]
            elif label == "clean":
                chosen = sorted(sub, key=lambda r: _f(r.get("oscillation_score"), 0.0))[:n]
            else:
                chosen = _quantile_pick(sub, n)
            out.extend(chosen)
    return out


def _select_spectral_cases(pool: list[dict], oscillatory_per_stage: int, clean_per_stage: int) -> list[dict]:
    # Only stages that actually contain an oscillatory class are deep-probed.
    # Matched clean cases from the SAME stage determine whether the plant/disturbance
    # operating point itself is unusually susceptible to high-frequency injection.
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for r in pool:
        by_stage[str(r.get("stage", ""))].append(r)
    out: list[dict] = []
    for stage in sorted(by_stage):
        rr = by_stage[stage]
        bad = [r for r in rr if r.get("label", r.get("class", "")) == "oscillatory"]
        if not bad:
            continue
        clean = [r for r in rr if r.get("label", r.get("class", "")) == "clean"]
        out.extend(sorted(bad, key=lambda r: _f(r.get("oscillation_score"), 0.0), reverse=True)[: int(oscillatory_per_stage)])
        out.extend(sorted(clean, key=lambda r: _f(r.get("oscillation_score"), 0.0))[: int(clean_per_stage)])
    return out


# ---------------------------------------------------------------------------
# Action/output interventions
@dataclass(frozen=True)
class LiveVariant:
    name: str
    action_mode: str = "raw"
    value: float = 1.0
    obs_mode: str = "raw"
    obs_value: float = 1.0


class ActionShaper:
    def __init__(self, mode: str, value: float):
        self.mode = str(mode)
        self.value = float(value)
        self.state: np.ndarray | None = None
        self.held: np.ndarray | None = None
        self.i = 0

    def reset(self):
        self.state = None
        self.held = None
        self.i = 0

    def apply(self, raw: np.ndarray) -> np.ndarray:
        a = np.asarray(raw, dtype=float).copy()
        mode, v = self.mode, self.value
        if mode == "raw":
            out = a
        elif mode == "moment_scale":
            out = a.copy(); out[1:4] *= v
        elif mode == "thrust_only":
            out = a.copy(); out[1:4] = 0.0
        elif mode == "moments_only":
            out = a.copy(); out[0] = 0.0
        elif mode in {"moment_lpf", "thrust_lpf", "all_lpf"}:
            if not (0.0 < v <= 1.0):
                raise ValueError("LPF beta must be in (0,1]")
            if self.state is None:
                self.state = a.copy()
            else:
                if mode == "all_lpf":
                    target = a
                elif mode == "moment_lpf":
                    target = np.r_[self.state[0], a[1:4]]
                else:  # thrust_lpf
                    target = np.r_[a[0], self.state[1:4]]
                self.state = (1.0 - v) * self.state + v * target
                if mode == "moment_lpf":
                    self.state[0] = a[0]
                elif mode == "thrust_lpf":
                    self.state[1:4] = a[1:4]
            out = self.state.copy()
        elif mode == "moment_hold":
            steps = max(1, int(round(v)))
            if self.held is None:
                self.held = a.copy()
            elif self.i % steps == 0:
                self.held[1:4] = a[1:4]
                self.held[0] = a[0]
            else:
                self.held[0] = a[0]  # thrust remains 100-Hz; only moments held
            out = self.held.copy()
        elif mode == "moment_slew":
            max_delta = float(v)
            if max_delta <= 0.0:
                raise ValueError("slew limit must be positive")
            if self.state is None:
                self.state = a.copy()
            else:
                delta = np.clip(a[1:4] - self.state[1:4], -max_delta, max_delta)
                self.state[1:4] += delta
                self.state[0] = a[0]
            out = self.state.copy()
        else:
            raise ValueError(f"unknown action mode {mode!r}")
        self.i += 1
        return np.clip(out, -1.0, 1.0)


def _live_variants() -> list[LiveVariant]:
    return [
        LiveVariant("live_raw"),
        LiveVariant("live_moment_lpf_b0p5", "moment_lpf", 0.5),
        LiveVariant("live_moment_lpf_b0p2", "moment_lpf", 0.2),
        LiveVariant("live_moment_lpf_b0p1", "moment_lpf", 0.1),
        LiveVariant("live_moment_lpf_b0p05", "moment_lpf", 0.05),
        LiveVariant("live_thrust_lpf_b0p2", "thrust_lpf", 0.2),
        LiveVariant("live_all_lpf_b0p2", "all_lpf", 0.2),
        LiveVariant("live_all_lpf_b0p1", "all_lpf", 0.1),
        LiveVariant("live_moment_hold_2", "moment_hold", 2.0),
        LiveVariant("live_moment_hold_4", "moment_hold", 4.0),
        LiveVariant("live_moment_hold_10", "moment_hold", 10.0),
        LiveVariant("live_moment_slew_0p2", "moment_slew", 0.2),
        LiveVariant("live_moment_slew_0p1", "moment_slew", 0.1),
        LiveVariant("live_moment_slew_0p05", "moment_slew", 0.05),
        LiveVariant("live_moment_scale_0p5", "moment_scale", 0.5),
        LiveVariant("live_thrust_only", "thrust_only", 0.0),
        LiveVariant("live_omega_lpf_b0p2", "raw", 1.0, "lpf", 0.2),
        LiveVariant("live_omega_gain_0p2", "raw", 1.0, "gain", 0.2),
    ]


# ---------------------------------------------------------------------------
# Playback spectral manipulation

def _delay_sequence(seq: np.ndarray, steps: int) -> np.ndarray:
    x = np.asarray(seq, dtype=float)
    d = max(0, int(steps))
    out = np.zeros_like(x)
    if d == 0:
        out[:] = x
    elif d < len(x):
        out[d:] = x[:-d]
    return out


def _fft_split_moments(seq: np.ndarray, dt: float, cutoff_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact offline low/high decomposition of moment channels.

    Thrust is set to zero in both returned sequences.  For moments,
    low + high == original up to floating-point error.
    """
    x = np.asarray(seq, dtype=float)
    low = np.zeros_like(x)
    high = np.zeros_like(x)
    if len(x) == 0:
        return low, high
    freq = np.fft.rfftfreq(len(x), d=float(dt))
    keep_low = freq <= float(cutoff_hz)
    for j in range(1, 4):
        X = np.fft.rfft(x[:, j])
        Xl = X.copy(); Xl[~keep_low] = 0.0
        Xh = X.copy(); Xh[keep_low] = 0.0
        low[:, j] = np.fft.irfft(Xl, n=len(x))
        high[:, j] = np.fft.irfft(Xh, n=len(x))
    return low, high


def _playback_variants(recorded: np.ndarray, dt: float, args) -> list[tuple[str, np.ndarray, dict | None]]:
    x = np.asarray(recorded, dtype=float)
    out: list[tuple[str, np.ndarray, dict | None]] = [("playback_exact", x.copy(), None)]

    for ms in (20, 50, 100):
        out.append((f"playback_delay_{ms}ms", _delay_sequence(x, int(round((ms / 1000.0) / dt))), None))

    # Same spectrum/amplitude, opposite moment sign: semantic feedback direction
    # changes but spectral energy does not.
    signflip = x.copy(); signflip[:, 1:4] *= -1.0
    out.append(("playback_moment_signflip", signflip, None))

    thrust = np.zeros_like(x); thrust[:, 0] = x[:, 0]
    out.append(("playback_thrust_only", thrust, None))
    moments = np.zeros_like(x); moments[:, 1:4] = x[:, 1:4]
    out.append(("playback_moments_only", moments, None))
    for j, lab in enumerate(MOMENT_LABELS, start=1):
        q = thrust.copy(); q[:, j] = x[:, j]
        out.append((f"playback_thrust_plus_{lab}", q, None))

    for cutoff in _parse_floats(args.playback_cutoffs_hz):
        low, high = _fft_split_moments(x, dt, cutoff)
        tag = str(float(cutoff)).replace(".", "p")
        q = thrust + low
        out.append((f"playback_thrust_plus_moment_low_{tag}Hz", np.clip(q, -1.0, 1.0), None))
        out.append((f"playback_moment_high_only_{tag}Hz", np.clip(high, -1.0, 1.0), None))

    # Disturbance/actuator necessity and saturation necessity.
    out += [
        ("playback_motor_limit_150pct", x.copy(), {"motor_limit_scale": 1.5}),
        ("playback_motor_limit_200pct", x.copy(), {"motor_limit_scale": 2.0}),
        ("playback_force_removed", x.copy(), {"force_scale": 0.0}),
        ("playback_mass_nominal", x.copy(), {"mass_mismatch_scale": 0.0}),
        ("playback_asymmetry_removed", x.copy(), {"all_asymmetry_scale": 0.0}),
        ("playback_actuator_nominal", x.copy(), {"all_actuator_mismatch_scale": 0.0}),
        ("playback_fully_nominalized", x.copy(), {
            "mass_mismatch_scale": 0.0,
            "force_scale": 0.0,
            "all_actuator_mismatch_scale": 0.0,
        }),
    ]

    # Finite excitation followed by zero residual: self-sustained-limit-cycle test.
    burst = np.zeros_like(x)
    i0 = max(0, int(round(args.burst_start_s / dt)))
    i1 = min(len(x), int(round(args.burst_end_s / dt)))
    if i1 > i0:
        burst[i0:i1] = x[i0:i1]
    out.append(("playback_burst_then_zero", burst, None))
    return out


# ---------------------------------------------------------------------------
# Rollouts and metrics

def _empty_trace() -> dict[str, list]:
    return {k: [] for k in (
        "desired_x", "true_x", "action", "residual", "u_base", "u_total",
        "motor_cmd", "sat", "omega_true", "omega_nom", "eomega"
    )}


def _append_pre_state(trace: dict, env: ResidualTwinEnv, desired: dict):
    st = env.dyn_true.state(); sn = env.dyn_nom.state()
    trace["desired_x"].append(np.asarray(desired["x"], dtype=float).copy())
    trace["true_x"].append(np.asarray(st["x"], dtype=float).copy())
    wt = np.asarray(st["omega"], dtype=float).copy()
    wn = np.asarray(sn["omega"], dtype=float).copy()
    trace["omega_true"].append(wt)
    trace["omega_nom"].append(wn)
    trace["eomega"].append(wn - wt)


def _finalize_trace(trace: dict) -> dict:
    return {k: np.asarray(v, dtype=float) for k, v in trace.items()}


def _metrics(trace: dict, dt: float, episode_steps: int, terminated: bool, args) -> dict:
    n = len(trace["action"])
    start = min(max(int(round(args.transient_ignore_s / dt)), 0), max(n - 1, 0))
    sl = slice(start, None)
    pos_err = trace["true_x"] - trace["desired_x"]
    action = trace["action"][sl]
    residual = trace["residual"][sl]
    base = trace["u_base"][sl]
    applied = trace["u_total"][sl]
    omega = trace["omega_true"][sl]
    eomega = trace["eomega"][sl]
    motor = trace["motor_cmd"][sl]

    def diff(x):
        return np.diff(x, axis=0) if len(x) > 1 else np.empty((0,) + x.shape[1:])

    # Dominant action moment period-2 indicator.
    mode = _dominant_axis_frequency(action, dt, 1.0)
    j = 1 + int(mode["axis"])
    rho1, rho2, alt = _period2_metrics(action[:, j] if len(action) else np.zeros(0))

    # Normalize by the physical nominal rotor limit, not by each trace's own
    # maximum.  Trace-dependent normalization would hide between-variant changes.
    motor_norm = motor / float(MAX_MOTOR_THRUST) if len(motor) else motor

    # Post-burst window is meaningful for burst tests and harmless otherwise.
    p0 = int(round((args.burst_end_s + args.post_burst_delay_s) / dt))
    p1 = min(n, int(round((args.burst_end_s + args.post_burst_window_s) / dt)))
    if p1 - p0 >= 4:
        post_omega = trace["omega_true"][p0:p1]
        post_applied = trace["u_total"][p0:p1] / WRENCH_SCALE[None, :]
        post_rough_omega = _rms_norm(diff(post_omega))
        post_rough_wrench = _rms_norm(diff(post_applied))
        post_sat = float(np.mean(trace["sat"][p0:p1]))
    else:
        post_rough_omega = post_rough_wrench = post_sat = float("nan")

    return {
        "episode_length": n,
        "completion_fraction": n / max(int(episode_steps), 1),
        "terminated": float(bool(terminated)),
        "true_des_pos_rmse_m": _rms_norm(pos_err),
        "sat_fraction": float(np.mean(trace["sat"])) if n else 0.0,
        "action_rms": _rms_norm(action),
        "moment_action_rms": _rms_norm(action[:, 1:4]) if len(action) else 0.0,
        "action_roughness": _rms_norm(diff(action)),
        "action_hf_ratio": _hf_ratio(action, dt, args.hf_cutoff_hz),
        "residual_moment_rms_Nm": _rms_norm(residual[:, 1:4]) if len(residual) else 0.0,
        "residual_moment_roughness_Nm": _rms_norm(diff(residual[:, 1:4])) if len(residual) else 0.0,
        "base_moment_roughness_Nm": _rms_norm(diff(base[:, 1:4])) if len(base) else 0.0,
        "base_moment_hf_ratio": _hf_ratio(base[:, 1:4], dt, args.hf_cutoff_hz) if len(base) else 0.0,
        "wrench_roughness": _rms_norm(diff(applied / WRENCH_SCALE[None, :])) if len(applied) else 0.0,
        "wrench_hf_ratio": _hf_ratio(applied / WRENCH_SCALE[None, :], dt, args.hf_cutoff_hz) if len(applied) else 0.0,
        "omega_rms_rad_s": _rms_norm(omega),
        "omega_roughness": _rms_norm(diff(omega)),
        "omega_hf_ratio": _hf_ratio(omega, dt, args.hf_cutoff_hz),
        "eomega_rms_rad_s": _rms_norm(eomega),
        "eomega_roughness": _rms_norm(diff(eomega)),
        "eomega_hf_ratio": _hf_ratio(eomega, dt, args.hf_cutoff_hz),
        "motor_command_roughness": _rms_norm(diff(motor_norm)) if len(motor) else 0.0,
        "dominant_moment_axis": MOMENT_LABELS[int(mode["axis"])],
        "dominant_moment_frequency_hz": float(mode["frequency_hz"]),
        "moment_lag1_corr": rho1,
        "moment_lag2_corr": rho2,
        "moment_sign_alternation_fraction": alt,
        "post_burst_omega_roughness": post_rough_omega,
        "post_burst_wrench_roughness": post_rough_wrench,
        "post_burst_sat_fraction": post_sat,
    }


def _rollout_live(stage_cfg, seed: int, case: dict, agent, variant: LiveVariant, args):
    cfg = copy.deepcopy(stage_cfg); cfg.residual_interface = "wrench"
    if args.max_steps is not None:
        cfg.episode_steps = min(int(cfg.episode_steps), int(args.max_steps))
    env = ResidualTwinEnv(cfg, seed=int(seed))
    obs = env.reset()
    _verify_sampled_disturbance(case, _disturbance_snapshot(env))
    modifier = OmegaObservationModifier(int(cfg.history), variant.obs_mode, variant.obs_value)
    shaper = ActionShaper(variant.action_mode, variant.value)
    trace = _empty_trace()
    terminated = False

    for _ in range(int(cfg.episode_steps)):
        desired = env.traj.desired(env.t)
        _append_pre_state(trace, env, desired)
        actor_obs = modifier.transform(obs)
        raw = np.asarray(agent.act(actor_obs, deterministic=True), dtype=float)
        action = shaper.apply(raw)
        obs, _, term, trunc, info = env.step(action)
        trace["action"].append(action.copy())
        trace["residual"].append(np.asarray(info["residual"], dtype=float).copy())
        trace["u_base"].append(np.asarray(info["u_base"], dtype=float).copy())
        trace["u_total"].append(np.asarray(info["u_total"], dtype=float).copy())
        trace["motor_cmd"].append(np.asarray(info["motor_cmd"], dtype=float).copy())
        trace["sat"].append(float(bool(info["actuator_saturated"])))
        terminated = bool(term)
        if term or trunc:
            break
    tr = _finalize_trace(trace)
    return _metrics(tr, float(cfg.dt), int(cfg.episode_steps), terminated, args), tr


def _rollout_guard(stage_cfg, seed: int, case: dict, agent, args):
    """Live policy with only infeasible residual magnitude shrunk to avoid saturation."""
    cfg = copy.deepcopy(stage_cfg); cfg.residual_interface = "wrench"
    if args.max_steps is not None:
        cfg.episode_steps = min(int(cfg.episode_steps), int(args.max_steps))
    env = ResidualTwinEnv(cfg, seed=int(seed)); obs = env.reset()
    _verify_sampled_disturbance(case, _disturbance_snapshot(env))
    trace = _empty_trace(); terminated = False
    for _ in range(int(cfg.episode_steps)):
        desired = env.traj.desired(env.t); _append_pre_state(trace, env, desired)
        raw = np.asarray(agent.act(obs, deterministic=True), dtype=float)
        wbase = _preview_base(env)
        lam = _max_feasible_scale(wbase, raw * env.action_scale, env.mixer, margin=args.guard_margin)
        action = np.clip(lam * raw, -1.0, 1.0)
        obs, _, term, trunc, info = env.step(action)
        trace["action"].append(action.copy()); trace["residual"].append(np.asarray(info["residual"], float).copy())
        trace["u_base"].append(np.asarray(info["u_base"], float).copy()); trace["u_total"].append(np.asarray(info["u_total"], float).copy())
        trace["motor_cmd"].append(np.asarray(info["motor_cmd"], float).copy()); trace["sat"].append(float(bool(info["actuator_saturated"])))
        terminated = bool(term)
        if term or trunc: break
    tr = _finalize_trace(trace)
    return _metrics(tr, float(cfg.dt), int(cfg.episode_steps), terminated, args), tr


def _rollout_sequence(stage_cfg, seed: int, case: dict, sequence: np.ndarray, args, transform: dict | None = None):
    cfg = copy.deepcopy(stage_cfg); cfg.residual_interface = "wrench"
    if args.max_steps is not None:
        cfg.episode_steps = min(int(cfg.episode_steps), int(args.max_steps))
    env = ResidualTwinEnv(cfg, seed=int(seed)); obs = env.reset()
    _verify_sampled_disturbance(case, _disturbance_snapshot(env))
    _apply_disturbance_transform(env, transform or {})
    trace = _empty_trace(); terminated = False
    seq = np.asarray(sequence, dtype=float)
    for i in range(int(cfg.episode_steps)):
        desired = env.traj.desired(env.t); _append_pre_state(trace, env, desired)
        action = np.zeros(ACTION_DIM) if i >= len(seq) else np.clip(seq[i], -1.0, 1.0)
        obs, _, term, trunc, info = env.step(action)
        trace["action"].append(action.copy()); trace["residual"].append(np.asarray(info["residual"], float).copy())
        trace["u_base"].append(np.asarray(info["u_base"], float).copy()); trace["u_total"].append(np.asarray(info["u_total"], float).copy())
        trace["motor_cmd"].append(np.asarray(info["motor_cmd"], float).copy()); trace["sat"].append(float(bool(info["actuator_saturated"])))
        terminated = bool(term)
        if term or trunc: break
    tr = _finalize_trace(trace)
    return _metrics(tr, float(cfg.dt), int(cfg.episode_steps), terminated, args), tr


def _rollout_baseline_only(stage_cfg, seed: int, case: dict, args):
    steps = int(stage_cfg.episode_steps if args.max_steps is None else min(stage_cfg.episode_steps, args.max_steps))
    return _rollout_sequence(stage_cfg, seed, case, np.zeros((steps, ACTION_DIM)), args)


def _rollout_prerecorded_baseline(stage_cfg, seed: int, case: dict, baseline_wrench: np.ndarray,
                                  residual_actions: np.ndarray, args):
    """Remove the live TRUE geometric-controller feedback.

    The true plant receives a baseline wrench sequence pre-recorded from the
    baseline-only trajectory plus the fixed residual sequence.  The nominal twin
    remains live, so e_omega can still be measured.  This asks whether reaction
    of the true geometric controller is necessary to amplify the injected chatter.
    """
    cfg = copy.deepcopy(stage_cfg); cfg.residual_interface = "wrench"
    if args.max_steps is not None:
        cfg.episode_steps = min(int(cfg.episode_steps), int(args.max_steps))
    env = ResidualTwinEnv(cfg, seed=int(seed)); env.reset()
    _verify_sampled_disturbance(case, _disturbance_snapshot(env))
    trace = _empty_trace(); terminated = False
    nmax = min(int(cfg.episode_steps), len(baseline_wrench), len(residual_actions))

    for i in range(nmax):
        desired = env.traj.desired(env.t); _append_pre_state(trace, env, desired)
        action = np.clip(np.asarray(residual_actions[i], dtype=float), -1.0, 1.0)
        residual = action * env.action_scale
        ubase = np.asarray(baseline_wrench[i], dtype=float)
        ucmd = ubase + residual
        f_true, M_true, mix = env.mixer.apply(float(ucmd[0]), ucmd[1:4], return_info=True)

        # Keep nominal reference twin live and untouched.
        sn = env.dyn_nom.state()
        f_n, M_n, _ = env.ctrl_nom.compute_control(sn, desired)
        f_nom, M_nom, _ = env.mixer.apply_nominal(f_n, M_n, return_info=True)
        st_next = env.dyn_true.step(f_true, M_true)
        sn_next = env.dyn_nom.step(f_nom, M_nom)
        env.step_idx += 1; env.t = env.step_idx * cfg.dt

        trace["action"].append(action.copy()); trace["residual"].append(residual.copy())
        trace["u_base"].append(ubase.copy()); trace["u_total"].append(np.r_[f_true, np.asarray(M_true, float)].copy())
        trace["motor_cmd"].append(np.asarray(mix["motor_cmd"], float).copy()); trace["sat"].append(float(bool(mix["saturated"])))
        terminated = bool(env._check_terminated(sn_next, st_next))
        if terminated:
            break
    tr = _finalize_trace(trace)
    return _metrics(tr, float(cfg.dt), int(cfg.episode_steps), terminated, args), tr


# ---------------------------------------------------------------------------
# Synthetic frequency response

def _synthetic_action(t: float, axis: int, amp: float, freq: float, kind: str,
                      start_s: float, end_s: float) -> np.ndarray:
    a = np.zeros(ACTION_DIM, dtype=float)
    if not (start_s <= t < end_s):
        return a
    phase = 2.0 * math.pi * float(freq) * (t - start_s)
    if kind == "sine":
        val = float(amp) * math.sin(phase)
    elif kind == "square":
        val = float(amp) * (1.0 if math.sin(phase) >= 0.0 else -1.0)
    else:
        raise ValueError(kind)
    a[1 + int(axis)] = val
    return a


def _rollout_synthetic(stage_cfg, seed: int, case: dict, axis: int, amp: float, freq: float,
                       kind: str, args):
    cfg = copy.deepcopy(stage_cfg); cfg.residual_interface = "wrench"
    if args.max_steps is not None:
        cfg.episode_steps = min(int(cfg.episode_steps), int(args.max_steps))
    env = ResidualTwinEnv(cfg, seed=int(seed)); obs = env.reset()
    _verify_sampled_disturbance(case, _disturbance_snapshot(env))
    trace = _empty_trace(); terminated = False
    for _ in range(int(cfg.episode_steps)):
        desired = env.traj.desired(env.t); _append_pre_state(trace, env, desired)
        action = _synthetic_action(env.t, axis, amp, freq, kind, args.spectral_start_s, args.spectral_end_s)
        obs, _, term, trunc, info = env.step(action)
        trace["action"].append(action.copy()); trace["residual"].append(np.asarray(info["residual"], float).copy())
        trace["u_base"].append(np.asarray(info["u_base"], float).copy()); trace["u_total"].append(np.asarray(info["u_total"], float).copy())
        trace["motor_cmd"].append(np.asarray(info["motor_cmd"], float).copy()); trace["sat"].append(float(bool(info["actuator_saturated"])))
        terminated = bool(term)
        if term or trunc: break
    tr = _finalize_trace(trace)
    metrics = _metrics(tr, float(cfg.dt), int(cfg.episode_steps), terminated, args)
    metrics.update(_frequency_response_metrics(tr, float(cfg.dt), axis, freq, args))
    return metrics, tr


def _complex_amplitude(x: np.ndarray, t: np.ndarray, freq: float) -> complex:
    y = np.asarray(x, dtype=float)
    if len(y) == 0:
        return 0j
    y = y - np.mean(y)
    basis = np.exp(-1j * 2.0 * np.pi * float(freq) * np.asarray(t, dtype=float))
    return (2.0 / len(y)) * np.sum(y * basis)


def _frequency_response_metrics(trace: dict, dt: float, axis: int, freq: float, args) -> dict:
    n = len(trace["action"])
    # Ignore initial part of the excitation to reduce transient contamination.
    t0 = float(args.spectral_start_s + args.spectral_settle_s)
    t1 = float(args.spectral_end_s)
    i0 = max(0, int(math.ceil(t0 / dt)))
    i1 = min(n, int(math.floor(t1 / dt)))
    if i1 - i0 < 8:
        return {
            "frf_residual_amp_Nm": float("nan"), "frf_omega_amp_rad_s": float("nan"),
            "frf_omega_gain_rad_s_per_Nm": float("nan"), "frf_omega_phase_deg": float("nan"),
            "frf_base_moment_gain": float("nan"), "frf_base_moment_phase_deg": float("nan"),
        }
    tt = np.arange(i0, i1) * dt
    j = int(axis)
    u = trace["residual"][i0:i1, 1 + j]
    omega = trace["omega_true"][i0:i1, j]
    base = trace["u_base"][i0:i1, 1 + j]
    U = _complex_amplitude(u, tt, freq)
    Ovec = np.asarray([_complex_amplitude(trace["omega_true"][i0:i1, k], tt, freq) for k in range(3)], dtype=complex)
    Bvec = np.asarray([_complex_amplitude(trace["u_base"][i0:i1, 1 + k], tt, freq) for k in range(3)], dtype=complex)
    O = Ovec[j]; B = Bvec[j]
    den = abs(U)
    if den <= 1e-12:
        gain_o = gain_b = total_o = total_b = cross_frac = float("nan"); phase_o = phase_b = float("nan")
    else:
        gain_o = abs(O) / den; gain_b = abs(B) / den
        total_o = float(np.linalg.norm(np.abs(Ovec))) / den
        total_b = float(np.linalg.norm(np.abs(Bvec))) / den
        other = np.delete(np.abs(Ovec), j)
        cross_frac = float(np.linalg.norm(other) / max(np.linalg.norm(np.abs(Ovec)), EPS))
        phase_o = float(np.degrees(np.angle(O / U)))
        phase_b = float(np.degrees(np.angle(B / U)))
    return {
        "frf_residual_amp_Nm": float(abs(U)),
        "frf_omega_amp_rad_s": float(abs(O)),
        "frf_omega_gain_rad_s_per_Nm": float(gain_o),
        "frf_omega_total_gain_rad_s_per_Nm": float(total_o),
        "frf_omega_cross_axis_fraction": float(cross_frac),
        "frf_omega_phase_deg": phase_o,
        "frf_base_moment_gain": float(gain_b),
        "frf_base_moment_total_gain": float(total_b),
        "frf_base_moment_phase_deg": phase_b,
    }


def _frequency_response_thrust_metrics(trace: dict, dt: float, freq: float, args) -> dict:
    n = len(trace["action"])
    i0 = max(0, int(math.ceil((args.spectral_start_s + args.spectral_settle_s) / dt)))
    i1 = min(n, int(math.floor(args.spectral_end_s / dt)))
    if i1 - i0 < 8:
        return {
            "frf_input_force_amp_N": float("nan"),
            "frf_omega_total_gain_rad_s_per_N": float("nan"),
            "frf_base_moment_total_gain_Nm_per_N": float("nan"),
        }
    tt = np.arange(i0, i1) * dt
    U = _complex_amplitude(trace["residual"][i0:i1, 0], tt, freq)
    Ovec = np.asarray([_complex_amplitude(trace["omega_true"][i0:i1, k], tt, freq) for k in range(3)], dtype=complex)
    Bvec = np.asarray([_complex_amplitude(trace["u_base"][i0:i1, 1 + k], tt, freq) for k in range(3)], dtype=complex)
    den = abs(U)
    return {
        "frf_input_force_amp_N": float(den),
        "frf_omega_total_gain_rad_s_per_N": float(np.linalg.norm(np.abs(Ovec)) / den) if den > 1e-12 else float("nan"),
        "frf_base_moment_total_gain_Nm_per_N": float(np.linalg.norm(np.abs(Bvec)) / den) if den > 1e-12 else float("nan"),
    }


def _rollout_synthetic_thrust(stage_cfg, seed: int, case: dict, amp: float, freq: float, args):
    cfg = copy.deepcopy(stage_cfg); cfg.residual_interface = "wrench"
    if args.max_steps is not None:
        cfg.episode_steps = min(int(cfg.episode_steps), int(args.max_steps))
    env = ResidualTwinEnv(cfg, seed=int(seed)); obs = env.reset()
    _verify_sampled_disturbance(case, _disturbance_snapshot(env))
    trace = _empty_trace(); terminated = False
    for _ in range(int(cfg.episode_steps)):
        desired = env.traj.desired(env.t); _append_pre_state(trace, env, desired)
        action = np.zeros(ACTION_DIM, dtype=float)
        if args.spectral_start_s <= env.t < args.spectral_end_s:
            phase = 2.0 * math.pi * float(freq) * (env.t - args.spectral_start_s)
            action[0] = float(amp) * math.sin(phase)
        obs, _, term, trunc, info = env.step(action)
        trace["action"].append(action.copy()); trace["residual"].append(np.asarray(info["residual"], float).copy())
        trace["u_base"].append(np.asarray(info["u_base"], float).copy()); trace["u_total"].append(np.asarray(info["u_total"], float).copy())
        trace["motor_cmd"].append(np.asarray(info["motor_cmd"], float).copy()); trace["sat"].append(float(bool(info["actuator_saturated"])))
        terminated = bool(term)
        if term or trunc: break
    tr = _finalize_trace(trace)
    metrics = _metrics(tr, float(cfg.dt), int(cfg.episode_steps), terminated, args)
    metrics.update(_frequency_response_thrust_metrics(tr, float(cfg.dt), freq, args))
    return metrics, tr


# ---------------------------------------------------------------------------
# Paired aggregation / conservative evidence tables

def _bootstrap_ci(values: Iterable[float], seed: int = 20260811, reps: int = 4000) -> tuple[float, float]:
    x = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(reps), dtype=float)
    for i in range(int(reps)):
        means[i] = float(np.mean(rng.choice(x, size=len(x), replace=True)))
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def _paired_summary(rows: list[dict], reference_variant: str, class_name: str | None = None) -> list[dict]:
    use = [r for r in rows if class_name is None or r.get("class") == class_name]
    refs = {r["case_id"]: r for r in use if r["variant"] == reference_variant}
    variants = sorted({r["variant"] for r in use})
    out = []
    for variant in variants:
        pairs = []
        for r in use:
            if r["variant"] != variant or r["case_id"] not in refs:
                continue
            q = refs[r["case_id"]]
            pairs.append((r, q))
        if not pairs:
            continue
        rmse_ratio = [p[0]["true_des_pos_rmse_m"] / max(p[1]["true_des_pos_rmse_m"], EPS) for p in pairs]
        omega_ratio = [p[0]["omega_roughness"] / max(p[1]["omega_roughness"], EPS) for p in pairs]
        eomega_ratio = [p[0]["eomega_roughness"] / max(p[1]["eomega_roughness"], EPS) for p in pairs]
        action_ratio = [p[0]["action_roughness"] / max(p[1]["action_roughness"], EPS) for p in pairs]
        sat_delta = [p[0]["sat_fraction"] - p[1]["sat_fraction"] for p in pairs]
        rmse_delta = [p[0]["true_des_pos_rmse_m"] - p[1]["true_des_pos_rmse_m"] for p in pairs]
        omega_delta = [p[0]["omega_roughness"] - p[1]["omega_roughness"] for p in pairs]
        ci_lo, ci_hi = _bootstrap_ci(omega_ratio)
        rm_lo, rm_hi = _bootstrap_ci(rmse_ratio)
        out.append({
            "class": class_name or "all",
            "variant": variant,
            "n": len(pairs),
            "rmse_ratio_mean": float(np.mean(rmse_ratio)),
            "rmse_ratio_median": float(np.median(rmse_ratio)),
            "rmse_ratio_mean_ci95_lo": rm_lo,
            "rmse_ratio_mean_ci95_hi": rm_hi,
            "omega_roughness_ratio_mean": float(np.mean(omega_ratio)),
            "omega_roughness_ratio_median": float(np.median(omega_ratio)),
            "omega_roughness_ratio_mean_ci95_lo": ci_lo,
            "omega_roughness_ratio_mean_ci95_hi": ci_hi,
            "eomega_roughness_ratio_mean": float(np.mean(eomega_ratio)),
            "action_roughness_ratio_mean": float(np.mean(action_ratio)),
            "sat_fraction_mean": float(np.mean([p[0]["sat_fraction"] for p in pairs])),
            "reference_sat_fraction_mean": float(np.mean([p[1]["sat_fraction"] for p in pairs])),
            "termination_fraction": float(np.mean([p[0]["terminated"] for p in pairs])),
            "sat_fraction_delta_mean": float(np.mean(sat_delta)),
            "rmse_delta_m_mean": float(np.mean(rmse_delta)),
            "omega_roughness_delta_mean": float(np.mean(omega_delta)),
            "oscillation_suppressed_fraction": float(np.mean(np.asarray(omega_ratio) <= 0.30)),
            "tracking_preserved_fraction": float(np.mean(np.asarray(rmse_ratio) <= 1.10)),
            "both_suppressed_and_tracking_fraction": float(np.mean((np.asarray(omega_ratio) <= 0.30) & (np.asarray(rmse_ratio) <= 1.10))),
            "no_new_termination_fraction": float(np.mean([p[0]["terminated"] <= p[1]["terminated"] for p in pairs])),
        })
    return out


def _hypothesis_table(summary: list[dict]) -> list[dict]:
    """Create a conservative *evidence* table, not automatic causal proofs.

    A failed intervention is never labeled as proof that a hypothesis is false:
    the intervention may simply be too weak or may leave another causal channel
    untouched.  The table reports suppression strength and explicit guards so the
    final mechanism is decided only after inspecting all interventions together.
    """
    by = {(r["class"], r["variant"]): r for r in summary}
    out = []

    def add(question: str, variant: str, guard: str, meaning_if_strong: str):
        r = by.get(("oscillatory", variant))
        if r is None:
            return
        omega = float(r["omega_roughness_ratio_mean"])
        if omega <= 0.30:
            effect = "strong suppression"
        elif omega <= 0.70:
            effect = "partial suppression"
        else:
            effect = "little/no suppression"
        out.append({
            "question": question,
            "test_variant": variant,
            "observed_effect": effect,
            "omega_roughness_ratio_mean": omega,
            "rmse_ratio_mean": r["rmse_ratio_mean"],
            "sat_fraction_mean": r.get("sat_fraction_mean", float("nan")),
            "reference_sat_fraction_mean": r.get("reference_sat_fraction_mean", float("nan")),
            "suppression_fraction": r["oscillation_suppressed_fraction"],
            "tracking_preserved_fraction": r["tracking_preserved_fraction"],
            "interpretation_guard": guard,
            "meaning_if_strong": meaning_if_strong,
        })

    add(
        "Does causal moment-output low-pass shaping suppress the physical state chatter?",
        "live_moment_lpf_b0p2",
        "Check actual action/residual roughness too; a live actor can adapt around a weak filter.",
        "The harmful path is strongly associated with high-bandwidth residual moments and moment-output shaping is a viable fix candidate.",
    )
    add(
        "Does causal thrust-output low-pass shaping suppress chatter?",
        "live_thrust_lpf_b0p2",
        "Especially important under asymmetric motors, where collective thrust can generate true moments.",
        "Residual thrust bandwidth is part of the rotational failure path.",
    )
    add(
        "Does filtering all residual outputs suppress chatter?",
        "live_all_lpf_b0p2",
        "Compare against moment-only and thrust-only filtering before deciding which channel to modify in training.",
        "Residual-output bandwidth, rather than just one state input, is a practical stabilization lever.",
    )
    add(
        "Does reducing actor e_omega bandwidth suppress chatter?",
        "live_omega_lpf_b0p2",
        "This is a causal input intervention, not yet a recommended estimator design.",
        "Fast e_omega -> policy feedback is an important chatter-generation path.",
    )
    add(
        "Is actuator saturation required to maintain the state chatter?",
        "live_sat_guard",
        "Only interpretable if this variant actually drives saturation close to zero relative to raw.",
        "If chatter remains despite removed saturation, saturation is an amplifier/consequence rather than a necessary trigger.",
    )
    add(
        "Does extra motor headroom suppress state chatter for the SAME fixed residual waveform?",
        "playback_motor_limit_200pct",
        "Fixed-waveform playback removes actor adaptation; compare saturation and omega, not applied-wrench roughness alone.",
        "Large suppression would implicate clipping/allocation nonlinearity as an amplifier.",
    )
    add(
        "Does removing external force reduce susceptibility to the SAME fixed residual waveform?",
        "playback_force_removed",
        "This tests operating-point amplification, not whether force caused the actor to generate the original waveform.",
        "External-force operating point materially amplifies the physical response.",
    )
    add(
        "Does nominalizing actuator/geometry mismatch reduce susceptibility to the SAME waveform?",
        "playback_actuator_nominal",
        "Recorded residual is counterfactual on the new plant; use as susceptibility evidence only.",
        "Actuator/geometry mismatch materially amplifies the physical response.",
    )
    add(
        "Does removing rotor-to-rotor asymmetry reduce susceptibility?",
        "playback_asymmetry_removed",
        "Mean gains are preserved; this isolates asymmetry from global gain error.",
        "Asymmetry is a specific amplifier rather than global actuator gain alone.",
    )
    add(
        "Is live true geometric-controller reaction required for the response?",
        "playback_prerecorded_baseline",
        "VALIDATE prerecorded_baseline_zero_residual first; if that control drifts/terminates, this intervention is confounded.",
        "If valid and strongly suppressed, baseline feedback reaction is an important amplifier. If valid and not suppressed, it is not required.",
    )
    add(
        "Can the recorded low-frequency moment component preserve tracking while removing chatter?",
        "playback_thrust_plus_moment_low_10p0Hz",
        "Offline FFT split is diagnostic/noncausal; use live LPF for implementable validation.",
        "Useful tracking correction is concentrated at low frequency while high-frequency moment content is dispensable.",
    )
    add(
        "Can the high-frequency moment component alone excite body-rate chatter?",
        "playback_moment_high_only_10p0Hz",
        "Interpret omega/e_omega response, not wrench roughness, because the input itself is deliberately rough.",
        "High-frequency residual moments are physically sufficient to excite the bad rotational response.",
    )
    return out


# ---------------------------------------------------------------------------
# Plots

def _plot_broad_case(case_dir: Path, traces: dict[str, dict], dt: float):
    wanted = [
        "live_raw", "live_moment_lpf_b0p2", "live_thrust_lpf_b0p2", "live_all_lpf_b0p2",
        "live_omega_lpf_b0p2", "live_sat_guard", "playback_prerecorded_baseline",
    ]
    names = [n for n in wanted if n in traces]
    if not names:
        return
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=False)
    for name in names:
        tr = traces[name]; t = np.arange(len(tr["action"])) * dt
        axes[0].plot(t, np.linalg.norm(tr["action"][:, 1:4], axis=1), label=name, linewidth=0.8)
        axes[1].plot(t, np.linalg.norm(tr["omega_true"], axis=1), label=name, linewidth=0.8)
        axes[2].plot(t, np.linalg.norm(tr["eomega"], axis=1), label=name, linewidth=0.8)
        axes[3].plot(t, tr["sat"], label=name, linewidth=0.8)
    axes[0].set_ylabel("||a_M||")
    axes[1].set_ylabel("||omega|| rad/s")
    axes[2].set_ylabel("||e_omega|| rad/s")
    axes[3].set_ylabel("saturation")
    axes[3].set_xlabel("time [s]")
    for ax in axes: ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncol=3)
    fig.tight_layout(); fig.savefig(case_dir / "broad_key_traces.png", dpi=160); plt.close(fig)


def _plot_spectral_case(case_dir: Path, rows: list[dict]):
    sine = [r for r in rows if r.get("waveform") == "sine" and np.isfinite(r.get("frf_omega_gain_rad_s_per_Nm", np.nan))]
    if not sine:
        return
    for metric, fname, ylabel in [
        ("frf_omega_gain_rad_s_per_Nm", "frf_omega_gain.png", "|omega| / |M_res| [(rad/s)/Nm]"),
        ("frf_base_moment_gain", "frf_baseline_reaction_gain.png", "|M_base| / |M_res|"),
        ("sat_fraction", "frf_saturation.png", "saturation fraction"),
    ]:
        fig, ax = plt.subplots(figsize=(9, 5))
        for axis in MOMENT_LABELS:
            aa = [r for r in sine if r.get("axis") == axis]
            for amp in sorted({float(r["amplitude"]) for r in aa}):
                ss = sorted([r for r in aa if float(r["amplitude"]) == amp], key=lambda r: float(r["frequency_hz"]))
                if ss:
                    ax.plot([float(r["frequency_hz"]) for r in ss], [float(r[metric]) for r in ss], marker="o", linewidth=0.9, label=f"{axis}, a={amp:g}")
        ax.set_xlabel("frequency [Hz]"); ax.set_ylabel(ylabel); ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=3); fig.tight_layout(); fig.savefig(case_dir / fname, dpi=160); plt.close(fig)


# ---------------------------------------------------------------------------
# Phase runners

def _base_row(case: dict, case_id: str, variant: str, family: str, metrics: dict) -> dict:
    return {
        **metrics,
        "case_id": case_id,
        "class": case.get("label", case.get("class", "")),
        "oscillation_score": _f(case.get("oscillation_score"), float("nan")),
        "stage": case.get("stage", ""),
        "seed": int(round(_f(case.get("seed"), -1))),
        "variant": variant,
        "family": family,
    }


def _run_broad_case(case: dict, cfg, curriculum_path: Path, agent, args, out_dir: Path) -> list[dict]:
    stage = str(case["stage"]); seed = int(round(_f(case["seed"])))
    case_id = f"{stage}_seed_{seed}"
    case_dir = out_dir / "broad" / case_id; case_dir.mkdir(parents=True, exist_ok=True)
    done = case_dir / "broad_metrics.csv"
    if done.is_file() and not args.overwrite:
        print(f"[broad skip] {case_id}")
        rows = _read_csv(done)
        # Return numerics converted where aggregation needs them.
        return [{k: (_f(v) if k not in {"case_id","class","stage","variant","family","dominant_moment_axis"} else v) for k,v in r.items()} for r in rows]

    _, stage_cfg = _find_stage_cfg(cfg.env, curriculum_path, stage)
    dt = float(stage_cfg.dt)
    rows: list[dict] = []; traces: dict[str, dict] = {}

    # Live policy / implementable candidate fix family.
    raw_trace = None
    for v in _live_variants():
        m, tr = _rollout_live(stage_cfg, seed, case, agent, v, args)
        rows.append(_base_row(case, case_id, v.name, "live", m)); traces[v.name] = tr
        if v.name == "live_raw": raw_trace = tr
        print(f"[{case_id}] {v.name:30s} RMSE={m['true_des_pos_rmse_m']:.4f} omegaR={m['omega_roughness']:.4g} sat={m['sat_fraction']:.3f}")

    m, tr = _rollout_guard(stage_cfg, seed, case, agent, args)
    rows.append(_base_row(case, case_id, "live_sat_guard", "live", m)); traces["live_sat_guard"] = tr

    assert raw_trace is not None
    recorded = raw_trace["action"]

    # Baseline-only reference, also supplies an open-loop baseline wrench sequence.
    mb, tb = _rollout_baseline_only(stage_cfg, seed, case, args)
    rows.append(_base_row(case, case_id, "baseline_zero_residual", "baseline", mb)); traces["baseline_zero_residual"] = tb

    # Fixed-action playback family: actor completely disconnected.
    for name, seq, transform in _playback_variants(recorded, dt, args):
        m, tr = _rollout_sequence(stage_cfg, seed, case, seq, args, transform=transform)
        rows.append(_base_row(case, case_id, name, "playback", m))
        # Keep only important traces to control disk size.
        if name in {
            "playback_exact", "playback_thrust_only", "playback_moments_only",
            "playback_thrust_plus_moment_low_10p0Hz", "playback_thrust_plus_moment_low_20p0Hz",
            "playback_moment_high_only_10p0Hz", "playback_motor_limit_200pct",
            "playback_actuator_nominal", "playback_asymmetry_removed", "playback_burst_then_zero",
        }:
            traces[name] = tr

    # Remove live true-controller reaction while retaining baseline-only wrench.
    zero_seq = np.zeros_like(recorded)
    m0, tr0 = _rollout_prerecorded_baseline(stage_cfg, seed, case, tb["u_base"], zero_seq, args)
    rows.append(_base_row(case, case_id, "prerecorded_baseline_zero_residual", "controller_ruleout_control", m0))
    traces["prerecorded_baseline_zero_residual"] = tr0

    m, tr = _rollout_prerecorded_baseline(stage_cfg, seed, case, tb["u_base"], recorded, args)
    rows.append(_base_row(case, case_id, "playback_prerecorded_baseline", "controller_ruleout", m))
    traces["playback_prerecorded_baseline"] = tr

    # Ratios versus raw live policy.
    ref = next(r for r in rows if r["variant"] == "live_raw")
    for r in rows:
        for metric in ("true_des_pos_rmse_m", "omega_roughness", "eomega_roughness", "action_roughness", "wrench_roughness"):
            r[f"{metric}_ratio_vs_live_raw"] = float(r[metric]) / max(float(ref[metric]), EPS)
        r["sat_fraction_delta_vs_live_raw"] = float(r["sat_fraction"]) - float(ref["sat_fraction"])

    _write_csv(done, rows)
    np.save(case_dir / "live_raw_actions.npy", recorded)
    (case_dir / "case.json").write_text(json.dumps(case, indent=2), encoding="utf-8")
    _plot_broad_case(case_dir, traces, dt)
    return rows


def _run_spectral_case(case: dict, cfg, curriculum_path: Path, args, out_dir: Path) -> list[dict]:
    stage = str(case["stage"]); seed = int(round(_f(case["seed"])))
    case_id = f"{stage}_seed_{seed}"
    case_dir = out_dir / "spectral" / case_id; case_dir.mkdir(parents=True, exist_ok=True)
    done = case_dir / "spectral_metrics.csv"
    if done.is_file() and not args.overwrite:
        print(f"[spectral skip] {case_id}")
        rows = _read_csv(done)
        return [{k: (_f(v) if k not in {"case_id","class","stage","variant","family","waveform","axis"} else v) for k,v in r.items()} for r in rows]

    _, stage_cfg = _find_stage_cfg(cfg.env, curriculum_path, stage)
    rows: list[dict] = []
    freqs = _parse_floats(args.spectral_frequencies_hz)
    amps = _parse_floats(args.spectral_amplitudes)
    if any(f >= 0.5 / float(stage_cfg.dt) for f in freqs):
        raise ValueError("spectral frequency must be strictly below Nyquist")

    # Dense sine sweep on all axes / amplitudes.
    for axis in range(3):
        for amp in amps:
            for freq in freqs:
                m, _ = _rollout_synthetic(stage_cfg, seed, case, axis, amp, freq, "sine", args)
                name = f"sine_{MOMENT_LABELS[axis]}_{freq:g}Hz_a{amp:g}"
                r = _base_row(case, case_id, name, "synthetic_sine", m)
                r.update({"waveform": "sine", "axis": MOMENT_LABELS[axis], "amplitude": amp, "frequency_hz": freq})
                rows.append(r)

    # Square probes are concentrated where period-2 / clipping behavior matters.
    for axis in range(3):
        for amp in _parse_floats(args.square_amplitudes):
            for freq in _parse_floats(args.square_frequencies_hz):
                m, _ = _rollout_synthetic(stage_cfg, seed, case, axis, amp, freq, "square", args)
                name = f"square_{MOMENT_LABELS[axis]}_{freq:g}Hz_a{amp:g}"
                r = _base_row(case, case_id, name, "synthetic_square", m)
                r.update({"waveform": "square", "axis": MOMENT_LABELS[axis], "amplitude": amp, "frequency_hz": freq})
                rows.append(r)

    # Collective-thrust excitation is essential for the asymmetric-motor hypothesis:
    # with unequal rotor effectiveness, a nominally collective residual can create
    # unintended true moments.  In symmetric/global cases this response should be small.
    for amp in _parse_floats(args.thrust_spectral_amplitudes):
        for freq in _parse_floats(args.thrust_spectral_frequencies_hz):
            m, _ = _rollout_synthetic_thrust(stage_cfg, seed, case, amp, freq, args)
            name = f"sine_thrust_{freq:g}Hz_a{amp:g}"
            r = _base_row(case, case_id, name, "synthetic_thrust", m)
            r.update({"waveform": "sine_thrust", "axis": "thrust", "amplitude": amp, "frequency_hz": freq})
            rows.append(r)

    _write_csv(done, rows); _plot_spectral_case(case_dir, rows)
    return rows


def _write_aggregate_outputs(out_dir: Path, broad_rows: list[dict], spectral_rows: list[dict]):
    if broad_rows:
        _write_csv(out_dir / "all_broad_results.csv", broad_rows)
        summary = []
        classes = sorted({r.get("class", "") for r in broad_rows})
        for cls in classes:
            summary.extend(_paired_summary(broad_rows, "live_raw", cls))
        _write_csv(out_dir / "broad_paired_summary.csv", summary)
        _write_csv(out_dir / "hypothesis_evidence.csv", _hypothesis_table(summary))

        # Stage-specific paired summaries are crucial because global/symmetric and
        # asymmetric-actuator failures need not share the same mechanism.
        by_stage_summary = []
        for stage in sorted({r.get("stage", "") for r in broad_rows}):
            stage_rows = [r for r in broad_rows if r.get("stage") == stage]
            for cls in sorted({r.get("class", "") for r in stage_rows}):
                ss = _paired_summary(stage_rows, "live_raw", cls)
                for q in ss: q["stage"] = stage
                by_stage_summary.extend(ss)
        _write_csv(out_dir / "broad_paired_summary_by_stage.csv", by_stage_summary)
        hyp_stage = []
        for stage in sorted({r.get("stage", "") for r in broad_rows}):
            ss = [r for r in by_stage_summary if r.get("stage") == stage]
            hh = _hypothesis_table(ss)
            for q in hh: q["stage"] = stage
            hyp_stage.extend(hh)
        _write_csv(out_dir / "hypothesis_evidence_by_stage.csv", hyp_stage)

        # Period-2 / near-Nyquist fingerprint of the original live policy.
        raw = [r for r in broad_rows if r.get("variant") == "live_raw"]
        period_rows = []
        for stage in sorted({r.get("stage", "") for r in raw}):
            for cls in sorted({r.get("class", "") for r in raw if r.get("stage") == stage}):
                rr = [r for r in raw if r.get("stage") == stage and r.get("class") == cls]
                if rr:
                    period_rows.append({
                        "stage": stage, "class": cls, "n": len(rr),
                        "dominant_frequency_mean_hz": float(np.mean([r["dominant_moment_frequency_hz"] for r in rr])),
                        "lag1_corr_mean": float(np.mean([r["moment_lag1_corr"] for r in rr])),
                        "lag2_corr_mean": float(np.mean([r["moment_lag2_corr"] for r in rr])),
                        "sign_alternation_fraction_mean": float(np.mean([r["moment_sign_alternation_fraction"] for r in rr])),
                    })
        _write_csv(out_dir / "period2_signature_by_stage.csv", period_rows)

        # Candidate fix table intentionally limited to causal live variants.
        live_summary = [r for r in summary if r["variant"].startswith("live_") and r["variant"] != "live_raw"]
        _write_csv(out_dir / "candidate_fix_evidence.csv", live_summary)

    if spectral_rows:
        _write_csv(out_dir / "all_spectral_results.csv", spectral_rows)
        # Frequency-response aggregate by axis/frequency/amplitude.
        def aggregate_spectral(include_stage: bool):
            grouped: dict[tuple, list[dict]] = defaultdict(list)
            for r in spectral_rows:
                prefix = (r.get("class"), r.get("stage")) if include_stage else (r.get("class"),)
                grouped[prefix + (r.get("waveform"), r.get("axis"), r.get("amplitude"), r.get("frequency_hz"))].append(r)
            agg = []
            for key, rr in sorted(grouped.items(), key=lambda kv: str(kv[0])):
                if include_stage:
                    cls, stage, wave, axis, amp, freq = key
                else:
                    cls, wave, axis, amp, freq = key; stage = "all"
                def mean(k):
                    x = [_f(q.get(k)) for q in rr]; x = [v for v in x if np.isfinite(v)]
                    return float(np.mean(x)) if x else float("nan")
                agg.append({
                    "class": cls, "stage": stage, "waveform": wave, "axis": axis, "amplitude": amp, "frequency_hz": freq, "n": len(rr),
                    "omega_axis_gain_mean": mean("frf_omega_gain_rad_s_per_Nm"),
                    "omega_total_gain_mean": mean("frf_omega_total_gain_rad_s_per_Nm"),
                    "omega_cross_axis_fraction_mean": mean("frf_omega_cross_axis_fraction"),
                    "omega_phase_mean_deg": mean("frf_omega_phase_deg"),
                    "baseline_reaction_gain_mean": mean("frf_base_moment_gain"),
                    "baseline_reaction_total_gain_mean": mean("frf_base_moment_total_gain"),
                    "baseline_reaction_phase_mean_deg": mean("frf_base_moment_phase_deg"),
                    "thrust_to_omega_total_gain_mean": mean("frf_omega_total_gain_rad_s_per_N"),
                    "thrust_to_baseline_moment_gain_mean": mean("frf_base_moment_total_gain_Nm_per_N"),
                    "omega_roughness_mean": mean("omega_roughness"),
                    "sat_fraction_mean": mean("sat_fraction"),
                    "position_rmse_mean_m": mean("true_des_pos_rmse_m"),
                    "termination_fraction": mean("terminated"),
                })
            return agg
        _write_csv(out_dir / "spectral_frequency_response_summary.csv", aggregate_spectral(False))
        _write_csv(out_dir / "spectral_frequency_response_by_stage.csv", aggregate_spectral(True))


def _write_readme(out_dir: Path, broad_cases: list[dict], spectral_cases: list[dict], args):
    text = f"""# Direct-wrench residual failure-mode suite

This suite deliberately separates *actor chatter generation* from *physical response*.

Broad cases selected: {len(broad_cases)}
Spectral matched severe/clean cases selected: {len(spectral_cases)}

Primary files after both phases:
- `broad_paired_summary.csv`: paired intervention effects by clean/middle/oscillatory class.
- `candidate_fix_evidence.csv`: implementable live interventions only.
- `hypothesis_evidence.csv`: conservative rule-out table.
- `spectral_frequency_response_summary.csv`: injected residual moment -> body-rate and baseline-controller response.

Interpret physical oscillation primarily with `omega_roughness`, `omega_hf_ratio`, `eomega_roughness`, saturation, termination, and position RMSE.  Do not use applied-wrench roughness alone as proof, because the diagnostic intentionally injects rough wrench commands.

The dense sine sweep excludes 50 Hz because dt={args.assumed_dt_for_note:g} s gives a 50-Hz Nyquist frequency; exact 50-Hz sampled sine is degenerate.
"""
    (out_dir / "README.md").write_text(text, encoding="utf-8")


def run(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    evaluation_dir = Path(args.evaluation_dir).expanduser().resolve()
    out_name = "residual_failure_mode_suite" if args.max_steps is None else f"residual_failure_mode_suite_smoke_{int(args.max_steps)}"
    out_dir = evaluation_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_resolved_config(run_dir); cfg.env.residual_interface = "wrench"
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU"); device = "cpu"
    cfg.device = device; cfg.env.device = device

    curriculum_path = Path(args.eval_curriculum).expanduser().resolve() if args.eval_curriculum else evaluation_dir / "evaluation_curriculum.toml"
    if not curriculum_path.is_file(): curriculum_path = run_dir / "curriculum.toml"
    if not curriculum_path.is_file(): raise FileNotFoundError("could not locate evaluation or run curriculum")

    pool = _load_case_pool(evaluation_dir)
    broad_cases = _select_broad_cases(pool, {
        "oscillatory": args.broad_oscillatory_per_stage,
        "middle": args.broad_middle_per_stage,
        "clean": args.broad_clean_per_stage,
    })
    spectral_cases = _select_spectral_cases(pool, args.spectral_oscillatory_per_stage, args.spectral_clean_per_stage)

    # Load checkpoint once.  A stage cfg is only needed to construct matching dimensions.
    seed_case = (broad_cases or spectral_cases)[0]
    _, first_cfg = _find_stage_cfg(cfg.env, curriculum_path, seed_case["stage"])
    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint)
    agent, _ = _load_policy(cfg, first_cfg, checkpoint, device)

    broad_rows: list[dict] = []
    spectral_rows: list[dict] = []
    if args.phase in {"broad", "all"}:
        print(f"Running BROAD phase on {len(broad_cases)} stage-stratified cases")
        for case in broad_cases:
            broad_rows.extend(_run_broad_case(case, cfg, curriculum_path, agent, args, out_dir))
    else:
        broad_rows = [
            {k: (_f(v) if k not in {"case_id","class","stage","variant","family","dominant_moment_axis"} else v) for k,v in r.items()}
            for r in _read_csv(out_dir / "all_broad_results.csv")
        ]

    if args.phase in {"spectral", "all"}:
        print(f"Running SPECTRAL phase on {len(spectral_cases)} matched oscillatory/clean cases")
        for case in spectral_cases:
            spectral_rows.extend(_run_spectral_case(case, cfg, curriculum_path, args, out_dir))
    else:
        spectral_rows = [
            {k: (_f(v) if k not in {"case_id","class","stage","variant","family","waveform","axis"} else v) for k,v in r.items()}
            for r in _read_csv(out_dir / "all_spectral_results.csv")
        ]

    _write_aggregate_outputs(out_dir, broad_rows, spectral_rows)
    args.assumed_dt_for_note = float(first_cfg.dt)
    _write_readme(out_dir, broad_cases, spectral_cases, args)
    (out_dir / "metadata.json").write_text(json.dumps({
        "checkpoint": str(checkpoint), "phase": args.phase,
        "broad_case_count": len(broad_cases), "spectral_case_count": len(spectral_cases),
        "broad_oscillatory_per_stage": args.broad_oscillatory_per_stage,
        "broad_middle_per_stage": args.broad_middle_per_stage,
        "broad_clean_per_stage": args.broad_clean_per_stage,
        "spectral_oscillatory_per_stage": args.spectral_oscillatory_per_stage,
        "spectral_clean_per_stage": args.spectral_clean_per_stage,
        "spectral_frequencies_hz": _parse_floats(args.spectral_frequencies_hz),
        "spectral_amplitudes": _parse_floats(args.spectral_amplitudes),
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--evaluation_dir", required=True)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--eval_curriculum", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--phase", choices=("broad", "spectral", "all"), default="broad")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--max_steps", type=int, default=None)

    # Stage-stratified case counts.  With the 600-case OOD evaluation this yields
    # ~30-36 broad cases and 6 severe spectral cases by default.
    p.add_argument("--broad_oscillatory_per_stage", type=int, default=3)
    p.add_argument("--broad_middle_per_stage", type=int, default=2)
    p.add_argument("--broad_clean_per_stage", type=int, default=3)
    p.add_argument("--spectral_oscillatory_per_stage", type=int, default=2)
    p.add_argument("--spectral_clean_per_stage", type=int, default=1)

    p.add_argument("--transient_ignore_s", type=float, default=0.2)
    p.add_argument("--hf_cutoff_hz", type=float, default=10.0)
    p.add_argument("--guard_margin", type=float, default=0.0)
    p.add_argument("--playback_cutoffs_hz", default="5,10,20,30")

    # Burst/post-burst for recorded sequence persistence check.
    p.add_argument("--burst_start_s", type=float, default=1.0)
    p.add_argument("--burst_end_s", type=float, default=2.0)
    p.add_argument("--post_burst_delay_s", type=float, default=0.2)
    p.add_argument("--post_burst_window_s", type=float, default=2.0)

    # Dense FRF window: 5 seconds gives >=5 cycles even at 1 Hz.
    p.add_argument("--spectral_start_s", type=float, default=1.0)
    p.add_argument("--spectral_end_s", type=float, default=6.0)
    p.add_argument("--spectral_settle_s", type=float, default=0.5)
    p.add_argument("--spectral_frequencies_hz", default="1,2,5,10,15,20,25,30,35,40,45,48,49")
    p.add_argument("--spectral_amplitudes", default="0.1,0.25,0.5,0.75,1.0")
    p.add_argument("--square_frequencies_hz", default="20,40,48,49")
    p.add_argument("--square_amplitudes", default="0.25,0.5,1.0")
    p.add_argument("--thrust_spectral_frequencies_hz", default="1,10,20,40,49")
    p.add_argument("--thrust_spectral_amplitudes", default="0.25,0.5,1.0")
    return p


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
