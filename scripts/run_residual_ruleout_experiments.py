"""Causal rule-out battery for legacy direct-wrench residual oscillations.

The goal is not to optimize a controller.  It is to answer questions such as:

* Is direct residual moment authority necessary for the oscillation?
* Is actuator saturation necessary, or merely correlated with the oscillation?
* Can a bounded synthetic "fake residual" deliberately create the same mode?
* Does removing external force or rotor-to-rotor asymmetry remove the mode?
* Does a high-frequency residual burst leave a self-sustained oscillation after
  the burst ends, or does the plant simply follow the injected chatter?
* Which event happens first: residual growth, motor saturation, attitude error,
  or high-frequency applied-wrench activity?

It replays exact stage/seed cases selected by ``analyze_oscillation_dataset``.
All experiments use ``residual_interface='wrench'`` and the same nominal SE(3)
controller.  No retraining is performed.

Examples
--------
First mine cases from the old paired evaluation::

  PYTHONPATH=src python3 -m scripts.analyze_oscillation_dataset \\
    --evaluation_dir runs_residual/residual_sac_curriculum/trial_002/evaluation/baseline_vs_residual_best

Then run a small causal battery::

  PYTHONPATH=src python3 -m scripts.run_residual_ruleout_experiments \\
    --run_dir runs_residual/residual_sac_curriculum/trial_002 \\
    --evaluation_dir runs_residual/residual_sac_curriculum/trial_002/evaluation/baseline_vs_residual_best \\
    --checkpoint best --battery core --max_cases_per_class 2 --device cuda

Use ``--battery full`` after the core battery identifies promising mechanisms.
"""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from robust_safe_rl.core.so3 import rotation_error
from robust_safe_rl.rl.curriculum import env_config_for_stage, load_curriculum
from robust_safe_rl.rl.mixer import F_MAX, M_MAX, MAX_MOTOR_THRUST, NOMINAL_ARM
from robust_safe_rl.rl.residual_env import ResidualTwinEnv
from scripts.evaluate_residual_report import (
    _disturbance_snapshot,
    _load_policy,
    _load_resolved_config,
    _resolve_checkpoint,
)

EPS = 1e-12
WRENCH_SCALE = np.concatenate(([F_MAX], np.asarray(M_MAX, dtype=float)))
MOMENT_LABELS = ("Mx", "My", "Mz")


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def _jsonify(x: Any):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonify(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonify(v) for v in x]
    return x


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
    X = np.fft.rfft(x, axis=0)
    p = np.abs(X) ** 2
    f = np.fft.rfftfreq(len(x), d=dt)
    valid = f > 0.0
    total = float(np.sum(p[valid]))
    if total <= EPS:
        return 0.0
    return float(np.sum(p[f >= cutoff_hz])) / total


def _rotation_error_norms(R_ref: np.ndarray, R_true: np.ndarray) -> np.ndarray:
    return np.asarray([np.linalg.norm(rotation_error(a, b)) for a, b in zip(R_ref, R_true)], dtype=float)


def _find_stage_cfg(base_env_cfg, curriculum_path: Path, stage_query: str):
    curriculum = load_curriculum(curriculum_path)
    for i, stage in enumerate(curriculum.stages):
        if stage.name == stage_query or str(i + 1) == str(stage_query):
            cfg = env_config_for_stage(base_env_cfg, stage)
            cfg.residual_interface = "wrench"
            return stage.name, cfg
    raise ValueError(f"stage {stage_query!r} not found in {curriculum_path}")


def _load_cases(path: Path, max_per_class: int, classes: set[str]) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no cases in {path}")
    out: list[dict] = []
    counts: dict[str, int] = {}
    for row in rows:
        cls = row.get("class", "oscillatory")
        if cls not in classes:
            continue
        if counts.get(cls, 0) >= max_per_class:
            continue
        out.append(row)
        counts[cls] = counts.get(cls, 0) + 1
    return out


def _verify_sampled_disturbance(case: dict, sampled: dict, atol: float = 1e-8):
    """Fail loudly if current code no longer reproduces the selected old case."""
    scalar_pairs = [("k", sampled["k"])]
    for key, actual in scalar_pairs:
        if key in case and case[key] not in (None, ""):
            if not np.isclose(float(case[key]), float(actual), rtol=0.0, atol=atol):
                raise RuntimeError(f"seed/config mismatch for {key}: CSV={case[key]}, rerun={actual}")

    vector_specs = [
        ("external_force", ["force_x_N", "force_y_N", "force_z_N"]),
        ("motor_coeff_scale", [f"motor_coeff_{i}" for i in range(4)]),
        ("moment_coeff_scale", [f"moment_coeff_{i}" for i in range(4)]),
        ("arm_length_scale", [f"arm_length_{i}" for i in range(4)]),
    ]
    for snap_key, csv_keys in vector_specs:
        if all(k in case and case[k] not in (None, "") for k in csv_keys):
            expected = np.asarray([float(case[k]) for k in csv_keys], dtype=float)
            actual = np.asarray(sampled[snap_key], dtype=float)
            if not np.allclose(expected, actual, rtol=0.0, atol=atol):
                raise RuntimeError(f"seed/config mismatch for {snap_key}: CSV={expected}, rerun={actual}")


def _preserve_mean_asymmetry(values: np.ndarray, scale: float) -> np.ndarray:
    """scale=0 removes rotor-to-rotor asymmetry while preserving mean gain."""
    v = np.asarray(values, dtype=float)
    m = float(np.mean(v))
    return m + float(scale) * (v - m)


def _toward_nominal(values: np.ndarray, scale: float) -> np.ndarray:
    """scale=0 -> nominal ones, scale=1 -> original mismatch."""
    v = np.asarray(values, dtype=float)
    return 1.0 + float(scale) * (v - 1.0)


def _apply_disturbance_transform(env: ResidualTwinEnv, transform: dict):
    """Change one sampled episode realization without resampling anything."""
    # Mass/MOI deviation from nominal.
    mass_scale = float(transform.get("mass_mismatch_scale", 1.0))
    env.k = 1.0 + mass_scale * (float(env.k) - 1.0)
    env.dyn_true.set_inertial_scale(env.k)

    force_scale = float(transform.get("force_scale", 1.0))
    env.external_force = np.asarray(env.external_force, dtype=float) * force_scale
    env.dyn_true.external_force = env.external_force.copy()
    env.dyn_true.last_external_force = env.external_force.copy()

    motor = np.asarray(env.motor_coeff_scale, dtype=float)
    moment = np.asarray(env.moment_coeff_scale, dtype=float)
    arm = np.asarray(env.arm_length_scale, dtype=float)

    common_asym = transform.get("all_asymmetry_scale", None)
    motor_asym = float(transform.get("motor_asymmetry_scale", 1.0 if common_asym is None else common_asym))
    moment_asym = float(transform.get("moment_asymmetry_scale", 1.0 if common_asym is None else common_asym))
    arm_asym = float(transform.get("arm_asymmetry_scale", 1.0 if common_asym is None else common_asym))
    motor = _preserve_mean_asymmetry(motor, motor_asym)
    moment = _preserve_mean_asymmetry(moment, moment_asym)
    arm = _preserve_mean_asymmetry(arm, arm_asym)

    mismatch = transform.get("all_actuator_mismatch_scale", None)
    motor_mismatch = float(transform.get("motor_mismatch_scale", 1.0 if mismatch is None else mismatch))
    moment_mismatch = float(transform.get("moment_mismatch_scale", 1.0 if mismatch is None else mismatch))
    arm_mismatch = float(transform.get("arm_mismatch_scale", 1.0 if mismatch is None else mismatch))
    motor = _toward_nominal(motor, motor_mismatch)
    moment = _toward_nominal(moment, moment_mismatch)
    arm = _toward_nominal(arm, arm_mismatch)

    env.motor_coeff_scale = motor
    env.moment_coeff_scale = moment
    env.arm_length_scale = arm
    env.mixer.set_true(
        kf_scale=motor,
        moment_scale=moment,
        arm_x=NOMINAL_ARM * arm,
    )

    limit_scale = float(transform.get("motor_limit_scale", 1.0))
    env.mixer.set_motor_thrust_limit(MAX_MOTOR_THRUST * limit_scale)


def _motor_violation(motor_cmd: np.ndarray, cap: float, margin: float = 0.0) -> float:
    lo = margin * cap
    hi = (1.0 - margin) * cap
    t = np.asarray(motor_cmd, dtype=float)
    return float(np.sum(np.maximum(lo - t, 0.0) + np.maximum(t - hi, 0.0)) / max(cap, EPS))


def _max_feasible_scale(w_base: np.ndarray, residual_wrench: np.ndarray, mixer, margin: float = 0.0) -> float:
    """Largest lambda in [0,1] with rotor commands inside the requested margin.

    This preserves the learned residual direction while shrinking only its
    magnitude.  It is therefore a useful saturation rule-out: if the oscillation
    disappears when only infeasible residual magnitude is removed, clipping is
    strongly implicated.
    """
    t0 = mixer.B_nom_inv @ np.asarray(w_base, dtype=float)
    td = mixer.B_nom_inv @ np.asarray(residual_wrench, dtype=float)
    cap = float(mixer.max_motor_thrust)
    lo, hi = margin * cap, (1.0 - margin) * cap

    lower, upper = 0.0, 1.0
    feasible_interval = True
    for b, d in zip(t0, td):
        if abs(d) <= EPS:
            if b < lo - 1e-12 or b > hi + 1e-12:
                feasible_interval = False
                break
            continue
        a = (lo - b) / d
        c = (hi - b) / d
        l_i, u_i = min(a, c), max(a, c)
        lower = max(lower, l_i)
        upper = min(upper, u_i)
        if lower > upper:
            feasible_interval = False
            break
    if feasible_interval and upper >= 0.0 and lower <= 1.0:
        return float(np.clip(upper, 0.0, 1.0))

    # Baseline itself can already be infeasible. In that case choose the scale
    # that minimizes violation rather than pretending a feasible lambda exists.
    candidates = np.linspace(0.0, 1.0, 101)
    v = [_motor_violation(t0 + lam * td, cap, margin=margin) for lam in candidates]
    return float(candidates[int(np.argmin(v))])


def _candidate_sat_push(action_dim: int, amplitude: float) -> list[np.ndarray]:
    if action_dim != 4:
        raise ValueError("synthetic saturation-push is defined for 4-D wrench action")
    a = float(amplitude)
    candidates = [a * np.asarray(s, dtype=float) for s in itertools.product((-1.0, 1.0), repeat=4)]
    for i in range(4):
        for sign in (-1.0, 1.0):
            x = np.zeros(4); x[i] = sign * a; candidates.append(x)
    return candidates


def _select_saturation_push_action(w_base: np.ndarray, action_scale: np.ndarray, mixer, amplitude: float) -> np.ndarray:
    """Choose a bounded fake residual that maximizes nominal rotor-limit violation."""
    cap = float(mixer.max_motor_thrust)
    best_action = np.zeros(4)
    best_key = (-float("inf"), -float("inf"))
    for a in _candidate_sat_push(4, amplitude):
        wrench = np.asarray(w_base, dtype=float) + a * np.asarray(action_scale, dtype=float)
        motor = mixer.B_nom_inv @ wrench
        violation = _motor_violation(motor, cap, margin=0.0)
        n_out = int(np.sum((motor < 0.0) | (motor > cap)))
        key = (violation, n_out)
        if key > best_key:
            best_key = key
            best_action = a
    return best_action


def _rolling_rms(x: np.ndarray, window: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("_rolling_rms expects 1-D input")
    if len(x) == 0:
        return x.copy()
    w = max(int(window), 1)
    sq = x * x
    c = np.concatenate(([0.0], np.cumsum(sq)))
    out = np.empty(len(x), dtype=float)
    for i in range(len(x)):
        j = max(0, i - w + 1)
        out[i] = math.sqrt(max((c[i + 1] - c[j]) / (i - j + 1), 0.0))
    return out


def _persistent_onset(signal: np.ndarray, threshold: float, persist: int, dt: float, offset_steps: int = 0) -> float:
    x = np.asarray(signal, dtype=float)
    mask = x >= float(threshold)
    p = max(int(persist), 1)
    if len(mask) < p:
        return float("nan")
    run = np.convolve(mask.astype(int), np.ones(p, dtype=int), mode="valid")
    idx = np.where(run >= p)[0]
    if len(idx) == 0:
        return float("nan")
    return float((int(idx[0]) + offset_steps) * dt)


def _first_true_time(mask: np.ndarray, dt: float, start: int = 0) -> float:
    m = np.asarray(mask, dtype=bool)
    start = min(max(int(start), 0), len(m))
    idx = np.where(m[start:])[0]
    return float((start + idx[0]) * dt) if len(idx) else float("nan")


@dataclass(frozen=True)
class Variant:
    name: str
    action_mode: str
    action_param: float = 0.0
    frequency_hz: float = 0.0
    moment_axis: int = 0
    delay_s: float = 0.0
    transform: dict | None = None


def _preview_base(env: ResidualTwinEnv) -> np.ndarray:
    desired = env.traj.desired(env.t)
    st = env.dyn_true.state()
    f, M, _ = env.ctrl_true.compute_control(st, desired, update_history=False)
    return np.array([f, M[0], M[1], M[2]], dtype=float)


def _choose_action(variant: Variant, env: ResidualTwinEnv, obs: np.ndarray, agent,
                   w_base: np.ndarray, burst_start: float, burst_end: float,
                   guard_margin: float, state: dict) -> np.ndarray:
    mode = variant.action_mode
    t = float(env.t)
    if mode == "zero":
        return np.zeros(4, dtype=float)

    if mode.startswith("policy"):
        if mode == "policy_then_zero" and t >= float(variant.action_param):
            return np.zeros(4, dtype=float)
        if mode == "policy_then_freeze" and t >= float(variant.action_param):
            held = state.get("held_policy_action")
            return np.zeros(4, dtype=float) if held is None else np.asarray(held, dtype=float).copy()
        a = np.asarray(agent.act(obs, deterministic=True), dtype=float)
        if mode == "policy_then_freeze":
            state["held_policy_action"] = a.copy()
            return a
        if mode == "policy_then_zero":
            return a
        if mode == "policy":
            return a
        if mode == "policy_guard":
            residual = a * env.action_scale
            lam = _max_feasible_scale(w_base, residual, env.mixer, margin=guard_margin)
            return np.clip(lam * a, -1.0, 1.0)
        if mode == "policy_moment_scale":
            out = a.copy(); out[1:4] *= float(variant.action_param); return out
        if mode == "policy_moments_only":
            out = a.copy(); out[0] = 0.0; return out
        raise ValueError(f"unknown policy mode {mode}")

    if mode in {"oppose_base", "reinforce_base", "delayed_oppose_base", "delayed_reinforce_base"}:
        source = w_base
        if mode.startswith("delayed_"):
            delay_steps = max(1, int(round(float(variant.delay_s) / float(env.cfg.dt))))
            hist = state.get("base_history", [])
            source = np.asarray(hist[-delay_steps], dtype=float) if len(hist) >= delay_steps else np.zeros(4)
        sign = -1.0 if mode == "oppose_base" else 1.0
        if mode == "delayed_oppose_base":
            sign = -1.0
        elif mode == "delayed_reinforce_base":
            sign = 1.0
        physical = np.zeros(4, dtype=float)
        physical[1:4] = sign * float(variant.action_param) * source[1:4]
        return np.clip(physical / np.asarray(env.action_scale, dtype=float), -1.0, 1.0)

    active = burst_start <= t < burst_end
    if not active:
        return np.zeros(4, dtype=float)

    if mode == "sat_push_burst":
        return _select_saturation_push_action(w_base, env.action_scale, env.mixer, variant.action_param)

    a = np.zeros(4, dtype=float)
    idx = 1 + int(variant.moment_axis)
    amp = float(variant.action_param)
    if mode == "constant_burst":
        a[idx] = amp
    elif mode == "sine_burst":
        a[idx] = amp * math.sin(2.0 * math.pi * float(variant.frequency_hz) * t)
    elif mode == "square_burst":
        s = math.sin(2.0 * math.pi * float(variant.frequency_hz) * t)
        a[idx] = amp * (1.0 if s >= 0.0 else -1.0)
    else:
        raise ValueError(f"unknown action mode {mode}")
    return a


def _rollout(stage_cfg, seed: int, case: dict, agent, variant: Variant, args):
    env_cfg = copy.deepcopy(stage_cfg)
    env_cfg.residual_interface = "wrench"
    if args.max_steps is not None:
        env_cfg.episode_steps = min(int(env_cfg.episode_steps), int(args.max_steps))
    env = ResidualTwinEnv(env_cfg, seed=int(seed))
    obs = env.reset()
    sampled = _disturbance_snapshot(env)
    _verify_sampled_disturbance(case, sampled)
    _apply_disturbance_transform(env, variant.transform or {})
    used_disturbance = _disturbance_snapshot(env)

    desired_x = []
    true_x = []; nom_x = []
    true_R = []; nom_R = []
    true_omega = []; nom_omega = []
    actions = []; residuals = []; base = []; cmd = []; applied = []
    motor_cmd = []; motor_sat = []; sat = []

    terminated = truncated = False
    control_state: dict[str, Any] = {"base_history": []}
    for _ in range(int(env_cfg.episode_steps)):
        d = env.traj.desired(env.t)
        st = env.dyn_true.state(); sn = env.dyn_nom.state()
        desired_x.append(np.asarray(d["x"], dtype=float).copy())
        true_x.append(st["x"].copy()); nom_x.append(sn["x"].copy())
        true_R.append(st["R"].copy()); nom_R.append(sn["R"].copy())
        true_omega.append(st["omega"].copy()); nom_omega.append(sn["omega"].copy())

        w_base_preview = _preview_base(env)
        a = _choose_action(
            variant, env, obs, agent, w_base_preview,
            args.burst_start_s, args.burst_end_s, args.guard_margin, control_state,
        )
        control_state["base_history"].append(w_base_preview.copy())
        obs, _reward, term, trunc, info = env.step(a)

        actions.append(np.asarray(a, dtype=float).copy())
        residuals.append(np.asarray(info["residual"], dtype=float).copy())
        base.append(np.asarray(info.get("u_base", w_base_preview), dtype=float).copy())
        cmd.append(np.asarray(info["u_cmd"], dtype=float).copy())
        applied.append(np.asarray(info["u_total"], dtype=float).copy())
        motor_cmd.append(np.asarray(info["motor_cmd"], dtype=float).copy())
        motor_sat.append(np.asarray(info["motor_sat"], dtype=float).copy())
        sat.append(float(bool(info["actuator_saturated"])))

        terminated, truncated = bool(term), bool(trunc)
        if term or trunc:
            break

    # Final state for RMSE alignment is not needed: each command-step state is
    # aligned with desired at the same time before the transition.
    arr = lambda x: np.asarray(x, dtype=float)
    trace = {
        "desired_x": arr(desired_x), "true_x": arr(true_x), "nom_x": arr(nom_x),
        "true_R": arr(true_R), "nom_R": arr(nom_R),
        "true_omega": arr(true_omega), "nom_omega": arr(nom_omega),
        "action": arr(actions), "residual": arr(residuals), "base": arr(base),
        "cmd": arr(cmd), "applied": arr(applied), "motor_cmd": arr(motor_cmd),
        "motor_sat": arr(motor_sat), "sat": arr(sat),
    }
    n = len(trace["applied"])
    dt = float(env_cfg.dt)
    start = min(max(int(round(args.transient_ignore_s / dt)), 0), max(n - 1, 0))

    applied_norm = trace["applied"] / WRENCH_SCALE[None, :]
    action = trace["action"]
    d_w = np.diff(applied_norm, axis=0) if n > 1 else np.empty((0, 4))
    d_a = np.diff(action, axis=0) if len(action) > 1 else np.empty((0, 4))

    pos_err = trace["true_x"] - trace["desired_x"]
    nom_pos_err = trace["true_x"] - trace["nom_x"]
    att_err = _rotation_error_norms(trace["nom_R"], trace["true_R"])
    omega_err = np.linalg.norm(trace["true_omega"] - trace["nom_omega"], axis=1)
    base_m_norm = np.linalg.norm(trace["base"][:, 1:4] / np.asarray(M_MAX)[None, :], axis=1)
    residual_m_action_norm = np.linalg.norm(action[:, 1:4], axis=1)
    cap = float(env.mixer.max_motor_thrust)
    motor_util = np.max(np.abs(trace["motor_cmd"]) / max(cap, EPS), axis=1)

    # Local chatter detector: rolling RMS of sample-to-sample normalized wrench
    # changes.  This detects onset rather than averaging over the whole episode.
    d_w_norm = np.linalg.norm(d_w, axis=1) if len(d_w) else np.empty(0)
    rolling = _rolling_rms(d_w_norm, max(1, int(round(args.onset_window_s / dt))))
    persist = max(1, int(round(args.onset_persist_s / dt)))
    event_start = min(max(int(round(args.event_ignore_s / dt)), 0), max(n - 1, 0))
    # d_w[k] corresponds to the transition ending at approximately step k+1.
    roll_start = min(event_start, len(rolling))
    t_osc = _persistent_onset(
        rolling[roll_start:], args.onset_wrench_diff_rms, persist, dt,
        offset_steps=1 + roll_start,
    )
    t_sat = _first_true_time(trace["sat"] > 0.5, dt, start=event_start)
    t_motor = _first_true_time(motor_util >= args.motor_util_event_threshold, dt, start=event_start)
    t_residual = _first_true_time(residual_m_action_norm >= args.residual_action_event_threshold, dt, start=event_start)
    t_att = _first_true_time(att_err >= args.attitude_event_threshold_rad, dt, start=event_start)
    t_omega = _first_true_time(omega_err >= args.omega_event_threshold_rad_s, dt, start=event_start)
    t_base_m = _first_true_time(base_m_norm >= args.baseline_moment_event_threshold, dt, start=event_start)

    post0 = int(round((args.burst_end_s + args.post_burst_delay_s) / dt))
    post1 = min(n, int(round((args.burst_end_s + args.post_burst_window_s) / dt)))
    if post1 - post0 >= 3:
        post_d = np.diff(applied_norm[post0:post1], axis=0)
        post_rough = _rms_norm(post_d)
        post_hf = _hf_ratio(applied_norm[post0:post1], dt, args.hf_cutoff_hz)
    else:
        post_rough = float("nan"); post_hf = float("nan")

    tail0 = min(n, max(0, int(round(args.tail_start_s / dt))))
    if n - tail0 >= 4:
        tail_d = np.diff(applied_norm[tail0:], axis=0)
        tail_rough = _rms_norm(tail_d)
        tail_hf = _hf_ratio(applied_norm[tail0:], dt, args.hf_cutoff_hz)
    else:
        tail_rough = float("nan"); tail_hf = float("nan")

    metrics = {
        "variant": variant.name,
        "action_mode": variant.action_mode,
        "episode_length": n,
        "completion_fraction": n / max(int(env_cfg.episode_steps), 1),
        "terminated": float(terminated),
        "true_des_pos_rmse_m": _rms_norm(pos_err),
        "true_nom_pos_rmse_m": _rms_norm(nom_pos_err),
        "true_nom_att_rmse_rad": float(np.sqrt(np.mean(att_err ** 2))) if len(att_err) else 0.0,
        "sat_fraction": float(np.mean(trace["sat"])) if n else 0.0,
        "motor_util_peak": float(np.max(motor_util)) if n else 0.0,
        "wrench_roughness": _rms_norm(d_w[start:]),
        "wrench_hf_ratio": _hf_ratio(applied_norm[start:], dt, args.hf_cutoff_hz),
        "action_roughness": _rms_norm(d_a[start:]),
        "action_hf_ratio": _hf_ratio(action[start:], dt, args.hf_cutoff_hz),
        "residual_rms": _rms_norm(trace["residual"][start:]),
        "residual_moment_action_rms": _rms_norm(action[start:, 1:4]),
        "oscillation_onset_s": t_osc,
        "first_saturation_s": t_sat,
        "first_motor_util_event_s": t_motor,
        "first_residual_moment_event_s": t_residual,
        "first_attitude_error_event_s": t_att,
        "first_omega_error_event_s": t_omega,
        "first_baseline_moment_event_s": t_base_m,
        "post_burst_wrench_roughness": post_rough,
        "post_burst_hf_ratio": post_hf,
        "tail_wrench_roughness": tail_rough,
        "tail_wrench_hf_ratio": tail_hf,
        "motor_limit_scale": cap / MAX_MOTOR_THRUST,
        "k": used_disturbance["k"],
        "force_norm_N": float(np.linalg.norm(used_disturbance["external_force"])),
        "motor_asym_rms": float(np.std(used_disturbance["motor_coeff_scale"])),
        "moment_asym_rms": float(np.std(used_disturbance["moment_coeff_scale"])),
        "arm_asym_rms": float(np.std(used_disturbance["arm_length_scale"])),
    }
    metrics["oscillatory"] = float(
        metrics["wrench_roughness"] >= args.osc_roughness_threshold
        and metrics["wrench_hf_ratio"] >= args.osc_hf_threshold
    )
    event_keys = (
        "first_saturation_s", "first_motor_util_event_s", "first_residual_moment_event_s",
        "first_attitude_error_event_s", "first_omega_error_event_s", "first_baseline_moment_event_s",
    )
    for key in event_keys:
        value = metrics[key]
        out_key = key[:-2] + "_minus_osc_s" if key.endswith("_s") else key + "_minus_osc_s"
        metrics[out_key] = (
            value - t_osc if np.isfinite(value) and np.isfinite(t_osc) else float("nan")
        )
    return metrics, trace, used_disturbance


def _variants_for_battery(battery: str) -> list[Variant]:
    core = [
        Variant("baseline_zero_residual", "zero"),
        Variant("policy_original", "policy"),
        Variant("policy_zero_after_1s", "policy_then_zero", action_param=1.0),
        Variant("policy_freeze_after_1s", "policy_then_freeze", action_param=1.0),
        Variant("policy_no_sat_guard", "policy_guard"),
        Variant("policy_thrust_only", "policy_moment_scale", action_param=0.0),
        Variant("policy_moment_50pct", "policy_moment_scale", action_param=0.5),
        Variant("policy_motor_limit_150pct", "policy", transform={"motor_limit_scale": 1.5}),
        Variant("policy_force_removed", "policy", transform={"force_scale": 0.0}),
        Variant("policy_all_asymmetry_removed", "policy", transform={"all_asymmetry_scale": 0.0}),
        Variant("policy_actuator_nominal", "policy", transform={"all_actuator_mismatch_scale": 0.0}),
        Variant("policy_mass_nominal", "policy", transform={"mass_mismatch_scale": 0.0}),
        Variant("fake_sat_push_100pct_burst", "sat_push_burst", action_param=1.0),
        Variant("fake_oppose_base_50pct", "oppose_base", action_param=0.5),
        Variant("fake_delayed_oppose_base_50pct_100ms", "delayed_oppose_base", action_param=0.5, delay_s=0.10),
        Variant("fake_reinforce_base_50pct", "reinforce_base", action_param=0.5),
        Variant("fake_constant_moment_50pct_burst", "constant_burst", action_param=0.5, moment_axis=0),
        Variant("fake_square_49Hz_50pct_burst", "square_burst", action_param=0.5, frequency_hz=49.0, moment_axis=0),
    ]
    if battery == "core":
        return core
    full = list(core)
    for lam in (0.25, 0.75):
        full.append(Variant(f"policy_moment_{int(100*lam)}pct", "policy_moment_scale", action_param=lam))
    for scale in (0.75, 1.25, 2.0):
        full.append(Variant(f"policy_motor_limit_{int(100*scale)}pct", "policy", transform={"motor_limit_scale": scale}))
    for scale in (0.25, 0.5, 0.75):
        full.append(Variant(f"policy_force_{int(100*scale)}pct", "policy", transform={"force_scale": scale}))
        full.append(Variant(f"policy_all_asym_{int(100*scale)}pct", "policy", transform={"all_asymmetry_scale": scale}))
        full.append(Variant(f"policy_actuator_mismatch_{int(100*scale)}pct", "policy", transform={"all_actuator_mismatch_scale": scale}))
    for gain in (0.25, 1.0):
        full.append(Variant(f"fake_oppose_base_{int(100*gain)}pct", "oppose_base", action_param=gain))
        full.append(Variant(f"fake_reinforce_base_{int(100*gain)}pct", "reinforce_base", action_param=gain))
    for amp in (0.25, 0.75):
        full.append(Variant(f"fake_sat_push_{int(100*amp)}pct_burst", "sat_push_burst", action_param=amp))
    for freq in (1.0, 5.0, 10.0, 25.0, 40.0):
        full.append(Variant(f"fake_sine_{int(freq)}Hz_50pct_burst", "sine_burst", action_param=0.5, frequency_hz=freq, moment_axis=0))
    for delay in (0.02, 0.05, 0.20):
        full.append(Variant(
            f"fake_delayed_oppose_base_50pct_{int(1000*delay)}ms",
            "delayed_oppose_base", action_param=0.5, delay_s=delay,
        ))
    return full


def _ratio(a: float, b: float) -> float:
    return float(a / max(abs(b), EPS))


def _build_hypothesis_rows(case_id: str, class_label: str, metrics: list[dict]) -> list[dict]:
    by = {m["variant"]: m for m in metrics}
    if "policy_original" not in by:
        return []
    p = by["policy_original"]
    out = []

    def add(hypothesis: str, variant: str, interpretation: str):
        if variant not in by:
            return
        m = by[variant]
        rough_ratio = _ratio(m["wrench_roughness"], p["wrench_roughness"])
        sat_delta = m["sat_fraction"] - p["sat_fraction"]
        if rough_ratio <= 0.30:
            evidence = "strong_reduction"
        elif rough_ratio <= 0.60:
            evidence = "moderate_reduction"
        elif rough_ratio >= 1.25:
            evidence = "worse"
        else:
            evidence = "little_change"
        out.append({
            "case_id": case_id, "class": class_label, "hypothesis": hypothesis,
            "comparison_variant": variant,
            "policy_roughness": p["wrench_roughness"], "comparison_roughness": m["wrench_roughness"],
            "roughness_ratio_vs_policy": rough_ratio,
            "policy_sat_fraction": p["sat_fraction"], "comparison_sat_fraction": m["sat_fraction"],
            "sat_fraction_delta": sat_delta,
            "comparison_oscillatory": m["oscillatory"],
            "evidence": evidence, "interpretation": interpretation,
        })

    add("direct residual moments are necessary/amplifying", "policy_thrust_only",
        "If roughness collapses with moment residuals removed, the direct moment channel is strongly implicated.")
    add("actuator clipping/saturation is necessary/amplifying", "policy_no_sat_guard",
        "This keeps the learned action direction but shrinks it only when needed for rotor feasibility.")
    add("actuator clipping/headroom is a trigger", "policy_motor_limit_150pct",
        "If extra motor headroom removes the mode, the clipping boundary is causal rather than merely correlated.")
    add("external force is a trigger", "policy_force_removed",
        "Same episode seed, but the sampled constant external force is set to zero.")
    add("rotor-to-rotor asymmetry is a trigger", "policy_all_asymmetry_removed",
        "Rotor means are preserved while per-rotor deviations are collapsed to the mean.")
    add("actuator/geometry mismatch is a trigger", "policy_actuator_nominal",
        "Motor, moment, and arm scales are moved to nominal while mass and force are retained.")
    add("mass/MOI mismatch is a trigger", "policy_mass_nominal",
        "True mass/MOI are set to nominal while the other sampled factors are retained.")

    for variant, hypothesis, interpretation in (
        ("policy_zero_after_1s", "ongoing residual feedback is necessary for a sustained mode",
         "The learned residual is set exactly to zero after 1 s. Compare tail roughness, not only whole-episode roughness."),
        ("policy_freeze_after_1s", "time-varying residual feedback is necessary for a sustained mode",
         "The learned residual is frozen after 1 s, preserving a static bias but removing further feedback variation."),
    ):
        if variant in by:
            m = by[variant]
            policy_tail = p.get("tail_wrench_roughness", float("nan"))
            comp_tail = m.get("tail_wrench_roughness", float("nan"))
            tail_ratio = _ratio(comp_tail, policy_tail) if np.isfinite(policy_tail) and np.isfinite(comp_tail) else float("nan")
            out.append({
                "case_id": case_id, "class": class_label, "hypothesis": hypothesis,
                "comparison_variant": variant,
                "policy_roughness": policy_tail, "comparison_roughness": comp_tail,
                "roughness_ratio_vs_policy": tail_ratio,
                "policy_sat_fraction": p["sat_fraction"], "comparison_sat_fraction": m["sat_fraction"],
                "sat_fraction_delta": m["sat_fraction"] - p["sat_fraction"],
                "comparison_oscillatory": m["oscillatory"],
                "evidence": "strong_reduction" if np.isfinite(tail_ratio) and tail_ratio <= 0.30 else (
                    "moderate_reduction" if np.isfinite(tail_ratio) and tail_ratio <= 0.60 else "little_change"
                ),
                "interpretation": interpretation,
            })

    # Sufficiency tests are most informative in cases that were originally clean.
    if class_label == "clean":
        for v, h in (
            ("fake_sat_push_100pct_burst", "strong bounded residual-induced saturation can be sufficient"),
            ("fake_oppose_base_50pct", "controller opposition alone can be sufficient"),
            ("fake_delayed_oppose_base_50pct_100ms", "delayed controller opposition can be sufficient"),
            ("fake_reinforce_base_50pct", "same-sign moment amplification can be sufficient"),
            ("fake_square_49Hz_50pct_burst", "high-frequency residual chatter can be sufficient"),
        ):
            if v in by:
                m = by[v]
                out.append({
                    "case_id": case_id, "class": class_label, "hypothesis": h,
                    "comparison_variant": v,
                    "policy_roughness": p["wrench_roughness"], "comparison_roughness": m["wrench_roughness"],
                    "roughness_ratio_vs_policy": _ratio(m["wrench_roughness"], p["wrench_roughness"]),
                    "policy_sat_fraction": p["sat_fraction"], "comparison_sat_fraction": m["sat_fraction"],
                    "sat_fraction_delta": m["sat_fraction"] - p["sat_fraction"],
                    "comparison_oscillatory": m["oscillatory"],
                    "evidence": "induced" if m["oscillatory"] > 0.5 else "not_induced",
                    "interpretation": "A synthetic controller-independent residual is used; no RL policy action is involved in this variant.",
                })
    return out


def _plot_case(out: Path, metrics: list[dict], traces: dict[str, dict], dt: float):
    labels = [m["variant"] for m in metrics]
    rough = [m["wrench_roughness"] for m in metrics]
    sat = [m["sat_fraction"] for m in metrics]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(labels)); width = 0.42
    ax.bar(x - width/2, rough, width, label="wrench roughness")
    ax.bar(x + width/2, sat, width, label="saturation fraction")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    ax.legend(); ax.grid(True, axis="y", alpha=0.25); fig.tight_layout()
    fig.savefig(out / "variant_control_quality.png", dpi=160); plt.close(fig)

    key_variants = [v for v in (
        "baseline_zero_residual", "policy_original", "policy_no_sat_guard",
        "fake_sat_push_100pct_burst", "fake_square_49Hz_50pct_burst",
    ) if v in traces]
    for name in key_variants:
        tr = traces[name]
        t = np.arange(len(tr["applied"])) * dt
        applied_norm = tr["applied"] / WRENCH_SCALE[None, :]
        fig, ax = plt.subplots(figsize=(10, 5))
        for j, lab in enumerate(("f", "Mx", "My", "Mz")):
            ax.plot(t, applied_norm[:, j], label=lab)
        ax.set_xlabel("time [s]"); ax.set_ylabel("normalized applied wrench")
        ax.set_title(name); ax.legend(ncol=4); ax.grid(True, alpha=0.25); fig.tight_layout()
        fig.savefig(out / f"trace_{name}.png", dpi=160); plt.close(fig)

    # Event order relative to detected oscillation onset for the learned policy.
    if "policy_original" in {m["variant"] for m in metrics}:
        p = next(m for m in metrics if m["variant"] == "policy_original")
        events = [
            ("residual moment", p["first_residual_moment_event_minus_osc_s"]),
            ("motor util", p["first_motor_util_event_minus_osc_s"]),
            ("saturation", p["first_saturation_minus_osc_s"]),
            ("attitude error", p["first_attitude_error_event_minus_osc_s"]),
            ("omega error", p["first_omega_error_event_minus_osc_s"]),
            ("baseline moment", p["first_baseline_moment_event_minus_osc_s"]),
        ]
        events = [(n, v) for n, v in events if np.isfinite(v)]
        if events:
            fig, ax = plt.subplots(figsize=(8, 4))
            y = np.arange(len(events))
            ax.scatter([v for _, v in events], y, s=50)
            ax.axvline(0.0, linestyle="--", linewidth=1)
            ax.set_yticks(y); ax.set_yticklabels([n for n, _ in events])
            ax.set_xlabel("event time - oscillation onset [s]")
            ax.grid(True, axis="x", alpha=0.25); fig.tight_layout()
            fig.savefig(out / "policy_event_order.png", dpi=160); plt.close(fig)


def run(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    evaluation_dir = Path(args.evaluation_dir).expanduser().resolve()
    cases_csv = Path(args.cases_csv).expanduser().resolve() if args.cases_csv else evaluation_dir / "oscillation_dataset_analysis" / "selected_cases.csv"
    if not cases_csv.is_file():
        raise FileNotFoundError(
            f"selected case file not found: {cases_csv}\n"
            "Run scripts.analyze_oscillation_dataset first or pass --cases_csv."
        )

    classes = {x.strip() for x in args.classes.split(",") if x.strip()}
    cases = _load_cases(cases_csv, args.max_cases_per_class, classes)
    if not cases:
        raise ValueError("no cases selected after class/count filtering")

    cfg = _load_resolved_config(run_dir)
    cfg.env.residual_interface = "wrench"
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU")
        device = "cpu"
    cfg.device = device; cfg.env.device = device

    curriculum_path = Path(args.eval_curriculum).expanduser().resolve() if args.eval_curriculum else evaluation_dir / "evaluation_curriculum.toml"
    if not curriculum_path.is_file():
        curriculum_path = run_dir / "curriculum.toml"
    if not curriculum_path.is_file():
        raise FileNotFoundError("could not find evaluation curriculum or run curriculum")

    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint)
    # A policy can be loaded with any wrench-stage cfg because observation/action
    # dimensions are unchanged across disturbance stages.
    first_stage_name, first_stage_cfg = _find_stage_cfg(cfg.env, curriculum_path, cases[0]["stage"])
    agent, _payload = _load_policy(cfg, first_stage_cfg, checkpoint, device)

    out = Path(args.output).expanduser().resolve() if args.output else evaluation_dir / "residual_ruleout_experiments"
    out.mkdir(parents=True, exist_ok=True)
    variants = _variants_for_battery(args.battery)
    if args.variants:
        requested = {x.strip() for x in args.variants.split(",") if x.strip()}
        available = {v.name for v in variants}
        missing = sorted(requested - available)
        if missing:
            raise ValueError(f"unknown --variants entries for {args.battery} battery: {missing}")
        variants = [v for v in variants if v.name in requested]
        if not variants:
            raise ValueError("--variants selected no experiments")

    all_metrics: list[dict] = []
    all_events: list[dict] = []
    all_hypotheses: list[dict] = []

    for ci, case in enumerate(cases, start=1):
        stage_name, stage_cfg = _find_stage_cfg(cfg.env, curriculum_path, case["stage"])
        seed = int(float(case["seed"]))
        cls = case.get("class", "unknown")
        case_id = f"{stage_name}_seed_{seed}"
        case_out = out / case_id
        case_out.mkdir(parents=True, exist_ok=True)
        print(f"[{ci}/{len(cases)}] {case_id} ({cls})")

        case_metrics: list[dict] = []
        traces: dict[str, dict] = {}
        original_disturbance = None
        for vi, variant in enumerate(variants, start=1):
            metrics, trace, used_disturbance = _rollout(stage_cfg, seed, case, agent, variant, args)
            if original_disturbance is None and variant.name == "baseline_zero_residual":
                original_disturbance = used_disturbance
            row = {
                "case_id": case_id, "class": cls, "stage": stage_name, "seed": seed,
                "source_oscillation_score": case.get("oscillation_score", ""),
                **metrics,
            }
            case_metrics.append(row); all_metrics.append(row); traces[variant.name] = trace
            all_events.append({
                "case_id": case_id, "class": cls, "variant": variant.name,
                "oscillation_onset_s": metrics["oscillation_onset_s"],
                "residual_minus_osc_s": metrics["first_residual_moment_event_minus_osc_s"],
                "motor_util_minus_osc_s": metrics["first_motor_util_event_minus_osc_s"],
                "saturation_minus_osc_s": metrics["first_saturation_minus_osc_s"],
                "attitude_minus_osc_s": metrics["first_attitude_error_event_minus_osc_s"],
                "omega_minus_osc_s": metrics["first_omega_error_event_minus_osc_s"],
                "baseline_moment_minus_osc_s": metrics["first_baseline_moment_event_minus_osc_s"],
            })
            print(
                f"  {vi:02d}/{len(variants):02d} {variant.name:<34} "
                f"rough={metrics['wrench_roughness']:.4f} HF={metrics['wrench_hf_ratio']:.3f} "
                f"sat={100*metrics['sat_fraction']:.1f}% RMSE={metrics['true_des_pos_rmse_m']:.3f} "
                f"term={int(metrics['terminated'])}"
            )

        _write_csv(case_out / "experiment_summary.csv", case_metrics)
        _plot_case(case_out, case_metrics, traces, stage_cfg.dt)
        all_hypotheses.extend(_build_hypothesis_rows(case_id, cls, case_metrics))
        (case_out / "case_metadata.json").write_text(json.dumps(_jsonify({
            "case": case,
            "stage": stage_name,
            "seed": seed,
            "checkpoint": checkpoint,
            "curriculum": curriculum_path,
            "sampled_disturbance": original_disturbance,
            "battery": args.battery,
            "burst_window_s": [args.burst_start_s, args.burst_end_s],
        }), indent=2), encoding="utf-8")

    _write_csv(out / "all_experiment_summary.csv", all_metrics)
    _write_csv(out / "event_timeline.csv", all_events)
    _write_csv(out / "hypothesis_evidence.csv", all_hypotheses)

    # Compact report with the most interpretable causal comparisons.
    lines = [
        "# Direct-wrench residual rule-out experiments",
        "",
        f"Cases: **{len(cases)}**, battery: **{args.battery}**, checkpoint: `{checkpoint.name}`.",
        "",
        "The synthetic residual variants do not use the RL policy. `fake_sat_push_*` chooses a bounded action inside the same normalized [-1,1]^4 residual action space that maximizes rotor-limit violation during the burst. This directly tests whether saturation plus closed-loop feedback can reproduce the bad mode.",
        "",
        "`policy_no_sat_guard` is the complementary necessity test: it keeps the learned residual direction but scales it down only when required for rotor feasibility.",
        "",
        "## Rule-out table",
        "",
        "| case | hypothesis | comparison | roughness ratio vs policy | saturation delta | result |",
        "|---|---|---|---:|---:|---|",
    ]
    for r in all_hypotheses:
        lines.append(
            f"| {r['case_id']} | {r['hypothesis']} | {r['comparison_variant']} | "
            f"{r['roughness_ratio_vs_policy']:.3f} | {r['sat_fraction_delta']:+.3f} | {r['evidence']} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "Use repeated results across several oscillatory and clean seeds, not a single case. A mechanism is especially convincing when both directions agree: removing it suppresses an existing oscillation, and introducing it into a clean case creates a similar mode.",
        "",
        "For burst tests, inspect `post_burst_wrench_roughness`. If the residual becomes zero after the burst but wrench chatter remains high, the feedback/allocation loop has entered a self-sustained mode. If chatter stops immediately, the applied-wrench HF energy is mostly tracking the injected residual itself.",
        "",
        "Event columns are reported relative to detected oscillation onset. Negative values mean the event happened before the oscillation detector fired. If saturation repeatedly precedes onset while attitude/baseline-moment growth follows it, that supports saturation as a trigger; the reverse ordering supports a controller-state error as the upstream trigger.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "run_metadata.json").write_text(json.dumps(_jsonify({
        "run_dir": run_dir, "evaluation_dir": evaluation_dir, "cases_csv": cases_csv,
        "checkpoint": checkpoint, "curriculum": curriculum_path, "battery": args.battery,
        "variants": [v.__dict__ for v in variants],
        "thresholds": {
            "osc_roughness": args.osc_roughness_threshold,
            "osc_hf": args.osc_hf_threshold,
            "onset_wrench_diff_rms": args.onset_wrench_diff_rms,
        },
    }), indent=2), encoding="utf-8")
    print(f"\nRule-out results written to: {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--evaluation_dir", required=True)
    p.add_argument("--cases_csv", default=None)
    p.add_argument("--eval_curriculum", default=None)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--battery", default="core", choices=("core", "full"))
    p.add_argument("--classes", default="oscillatory,clean")
    p.add_argument("--max_cases_per_class", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=None,
                   help="optional rollout-step cap for a quick smoke test; omit for full 1000-step episodes")
    p.add_argument("--variants", default=None,
                   help="optional comma-separated exact variant names from the selected battery")
    p.add_argument("--output", default=None)

    p.add_argument("--hf_cutoff_hz", type=float, default=5.0)
    p.add_argument("--transient_ignore_s", type=float, default=0.5)
    p.add_argument("--osc_roughness_threshold", type=float, default=0.05)
    p.add_argument("--osc_hf_threshold", type=float, default=0.20)

    p.add_argument("--burst_start_s", type=float, default=1.0)
    p.add_argument("--burst_end_s", type=float, default=2.0)
    p.add_argument("--post_burst_delay_s", type=float, default=0.2)
    p.add_argument("--post_burst_window_s", type=float, default=2.0)
    p.add_argument("--tail_start_s", type=float, default=1.2,
                   help="tail window start used by policy-zero/freeze persistence tests")
    p.add_argument("--guard_margin", type=float, default=0.0)

    p.add_argument("--onset_window_s", type=float, default=0.20)
    p.add_argument("--onset_persist_s", type=float, default=0.10)
    p.add_argument("--onset_wrench_diff_rms", type=float, default=0.05)
    p.add_argument("--event_ignore_s", type=float, default=0.10,
                   help="ignore only the first controller-startup transient when ordering onset events")
    p.add_argument("--motor_util_event_threshold", type=float, default=0.98)
    p.add_argument("--residual_action_event_threshold", type=float, default=0.25)
    p.add_argument("--attitude_event_threshold_rad", type=float, default=0.10)
    p.add_argument("--omega_event_threshold_rad_s", type=float, default=1.0)
    p.add_argument("--baseline_moment_event_threshold", type=float, default=0.50)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
