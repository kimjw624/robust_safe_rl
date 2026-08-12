"""Diagnose SE(3)-vs-residual controller interaction without retraining.

This script implements three small causal tests:

1. Residual-moment authority sweep
   Re-run the *same* disturbance realization while scaling only the physical
   residual moments by lambda in {0, 0.25, 0.5, 1}. Residual thrust is left at
   its trained authority. If oscillation grows strongly with lambda, the direct
   moment residual is implicated rather than generic SAC exploration noise.

2. Controller-opposition / frequency test
   For every sweep rollout, measure whether the SE(3) requested moments and the
   learned residual moments are simultaneously active, opposite in sign, and
   concentrated at the same frequency with ~180 deg phase separation.

3. Oracle outer-force injection test
   Apply a known constant external force and compare the ordinary SE(3)
   controller against the same controller with the *known disturbance force*
   added to its translational force vector A before desired attitude Rd is
   constructed. This tests whether the upstream injection location is effective
   before training a new RL policy there.

The script does not modify training code or checkpoints.

Example: automatically select the roughest residual episode from an existing
paired evaluation and run all diagnostics::

    PYTHONPATH=src python3 -m scripts.diagnose_controller_interaction \\
      --run_dir runs_residual/residual_sac_curriculum/trial_002 \\
      --evaluation_dir runs_residual/residual_sac_curriculum/trial_002/evaluation/baseline_vs_residual_best \\
      --checkpoint best \\
      --select worst_roughness \\
      --device cuda

You can also specify an exact case::

    PYTHONPATH=src python3 -m scripts.diagnose_controller_interaction \\
      --run_dir runs_residual/residual_sac_curriculum/trial_002 \\
      --eval_curriculum configs/residual_sac_ood_eval.toml \\
      --stage ood_all_high_global --seed 20660877 \\
      --checkpoint best
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from robust_safe_rl.core.so3 import rotation_error
from robust_safe_rl.rl.curriculum import env_config_for_stage, load_curriculum
from robust_safe_rl.rl.mixer import F_MAX, M_MAX, MAX_MOTOR_THRUST
from robust_safe_rl.rl.residual_env import ResidualTwinEnv

# Reuse the already-tested run/checkpoint loading and reward helper from the
# paired evaluator. Keeping one implementation avoids subtle checkpoint/API
# mismatches between evaluation scripts.
from scripts.evaluate_residual_report import (
    _disturbance_snapshot,
    _load_policy,
    _load_resolved_config,
    _resolve_checkpoint,
    _state_reward_from_states,
)

EPS = 1e-12
WRENCH_SCALE = np.concatenate(([F_MAX], np.asarray(M_MAX, dtype=float)))
MOMENT_LABELS = ("Mx", "My", "Mz")


def _jsonify(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    return value


def _write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonify(obj), indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields, seen = [], set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _rms_norm(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    if x.ndim == 1:
        return float(np.sqrt(np.mean(x * x)))
    return float(np.sqrt(np.mean(np.sum(x * x, axis=1))))


def _high_frequency_ratio(signal: np.ndarray, dt: float, cutoff_hz: float) -> float:
    x = np.asarray(signal, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < 4:
        return 0.0
    x = x - np.mean(x, axis=0, keepdims=True)
    X = np.fft.rfft(x, axis=0)
    power = np.abs(X) ** 2
    freq = np.fft.rfftfreq(len(x), d=dt)
    valid = freq > 0.0
    total = float(np.sum(power[valid]))
    if total <= EPS:
        return 0.0
    return float(np.sum(power[freq >= cutoff_hz])) / total


def _rotation_error_norms(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    return np.asarray([np.linalg.norm(rotation_error(a, b)) for a, b in zip(Ra, Rb)], dtype=float)


def _tilt_deg(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    c = np.clip(R[:, 2, 2], -1.0, 1.0)
    return np.rad2deg(np.arccos(c))


def _parse_floats(text: str, expected: int | None = None) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if expected is not None and len(vals) != expected:
        raise ValueError(f"expected {expected} comma-separated values, got {len(vals)}")
    return vals


def _find_stage_cfg(base_env_cfg, curriculum_path: Path, stage_query: str):
    curriculum = load_curriculum(curriculum_path)
    for i, stage in enumerate(curriculum.stages):
        if stage.name == stage_query or str(i + 1) == str(stage_query):
            return stage.name, env_config_for_stage(base_env_cfg, stage)
    names = [s.name for s in curriculum.stages]
    raise ValueError(f"stage {stage_query!r} not found in {curriculum_path}; choices: {names}")


def _select_case(evaluation_dir: Path, criterion: str) -> dict:
    csv_path = evaluation_dir / "paired_episode_metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"paired evaluation CSV not found: {csv_path}")
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows in {csv_path}")

    if criterion == "worst_roughness":
        key = "residual_applied_wrench_roughness"
        score = lambda r: float(r[key])
    elif criterion == "worst_hf":
        key = "residual_applied_wrench_hf_ratio"
        score = lambda r: float(r[key])
    elif criterion == "worst_rmse_delta":
        key = "delta_true_des_pos_rmse"
        score = lambda r: float(r[key])
    elif criterion == "worst_residual_rmse":
        key = "residual_true_des_pos_rmse"
        score = lambda r: float(r[key])
    elif criterion == "residual_failure":
        failed = [r for r in rows if float(r.get("residual_terminated", 0.0)) > 0.5]
        if failed:
            rows = failed
        key = "delta_true_des_pos_rmse"
        score = lambda r: float(r[key])
    else:
        raise ValueError(f"unknown --select criterion: {criterion}")

    row = max(rows, key=score)
    return {
        "stage": row["stage"],
        "case": int(float(row.get("case", 0))),
        "seed": int(float(row["seed"])),
        "selection_metric": key,
        "selection_value": float(row[key]),
    }


def _load_case(args, cfg):
    evaluation_dir = Path(args.evaluation_dir).expanduser().resolve() if args.evaluation_dir else None
    selected = None
    if evaluation_dir is not None:
        selected = _select_case(evaluation_dir, args.select)
        stage = selected["stage"] if args.stage is None else args.stage
        seed = selected["seed"] if args.seed is None else int(args.seed)
        curriculum_path = (
            Path(args.eval_curriculum).expanduser().resolve()
            if args.eval_curriculum
            else evaluation_dir / "evaluation_curriculum.toml"
        )
    else:
        if args.stage is None or args.seed is None:
            raise ValueError("provide --evaluation_dir, or provide both --stage and --seed")
        stage = args.stage
        seed = int(args.seed)
        curriculum_path = (
            Path(args.eval_curriculum).expanduser().resolve()
            if args.eval_curriculum
            else Path(args.run_dir).expanduser().resolve() / "curriculum.toml"
        )

    if not curriculum_path.is_file():
        raise FileNotFoundError(f"evaluation curriculum not found: {curriculum_path}")
    stage_name, stage_cfg = _find_stage_cfg(cfg.env, curriculum_path, stage)
    return stage_name, stage_cfg, seed, curriculum_path, selected


def _opposition_metrics(base_m: np.ndarray, residual_m: np.ndarray, dt: float,
                        transient_ignore_s: float, active_threshold: float = 0.02,
                        min_freq_hz: float = 0.2) -> dict:
    """Quantify direct moment opposition and common-frequency phase.

    Moments are normalized by the actuator envelope before thresholding. The
    active threshold is therefore a fraction of maximum nominal moment authority.
    """
    b = np.asarray(base_m, dtype=float) / np.asarray(M_MAX, dtype=float)[None, :]
    r = np.asarray(residual_m, dtype=float) / np.asarray(M_MAX, dtype=float)[None, :]
    n = min(len(b), len(r))
    start = int(round(float(transient_ignore_s) / float(dt)))
    start = min(max(start, 0), max(n - 1, 0))
    b, r = b[start:n], r[start:n]

    rows = []
    for j, label in enumerate(MOMENT_LABELS):
        bj, rj = b[:, j], r[:, j]
        active = (np.abs(bj) >= active_threshold) & (np.abs(rj) >= active_threshold)
        if np.any(active):
            opp = float(np.mean((bj[active] * rj[active]) < 0.0))
            ba, ra = bj[active], rj[active]
            denom = float(np.linalg.norm(ba) * np.linalg.norm(ra))
            cosine = float(np.dot(ba, ra) / denom) if denom > EPS else float("nan")
        else:
            opp, cosine = float("nan"), float("nan")

        # Demeaned spectral cross-phase at the frequency where both signals have
        # the largest geometric-mean power. This is deliberately simple and
        # interpretable rather than a full spectral identification routine.
        bd = bj - np.mean(bj) if len(bj) else bj
        rd = rj - np.mean(rj) if len(rj) else rj
        if len(bd) >= 8:
            B = np.fft.rfft(bd)
            R = np.fft.rfft(rd)
            freq = np.fft.rfftfreq(len(bd), d=dt)
            valid = freq >= min_freq_hz
            common = np.sqrt((np.abs(B) ** 2) * (np.abs(R) ** 2))
            common[~valid] = 0.0
            idx = int(np.argmax(common))
            if common[idx] > EPS:
                phase = float(np.rad2deg(np.angle(B[idx] * np.conj(R[idx]))))
                dominant = float(freq[idx])
                b_amp = float(2.0 * np.abs(B[idx]) / len(bd))
                r_amp = float(2.0 * np.abs(R[idx]) / len(rd))
            else:
                phase, dominant, b_amp, r_amp = (float("nan"),) * 4
        else:
            phase, dominant, b_amp, r_amp = (float("nan"),) * 4

        rows.append({
            "axis": label,
            "active_fraction": float(np.mean(active)) if len(active) else 0.0,
            "opposite_sign_fraction_when_active": opp,
            "cosine_similarity_when_active": cosine,
            "dominant_common_frequency_hz": dominant,
            "phase_base_minus_residual_deg": phase,
            "baseline_normalized_amplitude_at_common_freq": b_amp,
            "residual_normalized_amplitude_at_common_freq": r_amp,
        })

    # Aggregate direct opposition across all moment axes/time samples.
    active_all = (np.abs(b) >= active_threshold) & (np.abs(r) >= active_threshold)
    if np.any(active_all):
        ba, ra = b[active_all], r[active_all]
        denom = float(np.linalg.norm(ba) * np.linalg.norm(ra))
        aggregate_cos = float(np.dot(ba, ra) / denom) if denom > EPS else float("nan")
        aggregate_opp = float(np.mean((ba * ra) < 0.0))
    else:
        aggregate_cos = aggregate_opp = float("nan")

    return {
        "axes": rows,
        "aggregate_opposite_sign_fraction_when_active": aggregate_opp,
        "aggregate_cosine_similarity_when_active": aggregate_cos,
    }


def _rollout_residual_scaled(env_cfg, seed: int, agent, moment_scale: float,
                             hf_cutoff_hz: float, transient_ignore_s: float):
    """Run deterministic SAC with only residual moment authority multiplied by lambda."""
    env = ResidualTwinEnv(copy.deepcopy(env_cfg), seed=int(seed))
    env.action_scale = np.asarray(env.action_scale, dtype=float).copy()
    env.action_scale[1:4] *= float(moment_scale)
    obs = env.reset()
    disturbance = _disturbance_snapshot(env)

    times = [0.0]
    true_x, nom_x, desired_x = [], [], []
    true_R, nom_R = [], []
    action, residual, base, cmd, applied = [], [], [], [], []
    sat = []
    state_rewards = []

    def record_state():
        d = env.traj.desired(env.t)
        st, sn = env.dyn_true.state(), env.dyn_nom.state()
        desired_x.append(np.asarray(d["x"], dtype=float))
        true_x.append(st["x"])
        nom_x.append(sn["x"])
        true_R.append(st["R"])
        nom_R.append(sn["R"])

    record_state()
    terminated = truncated = False
    term_reason = ""
    for _ in range(int(env_cfg.episode_steps)):
        a = np.asarray(agent.act(obs, deterministic=True), dtype=np.float32)
        obs, _reward, term, trunc, info = env.step(a)

        res = np.asarray(info["residual"], dtype=float)
        ucmd = np.asarray(info["u_cmd"], dtype=float)
        ubase = ucmd - res
        utotal = np.asarray(info["u_total"], dtype=float)

        action.append(a.copy())
        residual.append(res)
        base.append(ubase)
        cmd.append(ucmd)
        applied.append(utotal)
        sat.append(float(bool(info.get("actuator_saturated", False))))
        state_rewards.append(_state_reward_from_states(env.cfg, env.dyn_nom.state(), env.dyn_true.state()))
        disturbance = _disturbance_snapshot(env, info)
        times.append(float(env.t))
        record_state()

        if term or trunc:
            terminated, truncated = bool(term), bool(trunc)
            if term:
                st, sn = env.dyn_true.state(), env.dyn_nom.state()
                reasons = []
                if np.linalg.norm(sn["x"] - st["x"]) > env.cfg.term_pos_error:
                    reasons.append("model_ref_position")
                if st["R"][2, 2] < env._tilt_cos_thresh:
                    reasons.append("tilt")
                term_reason = "+".join(reasons) if reasons else "other"
            break

    arr = lambda x: np.asarray(x, dtype=float)
    trace = {
        "time": arr(times),
        "true_x": arr(true_x),
        "nom_x": arr(nom_x),
        "desired_x": arr(desired_x),
        "true_R": arr(true_R),
        "nom_R": arr(nom_R),
        "action": arr(action),
        "residual": arr(residual),
        "base": arr(base),
        "cmd": arr(cmd),
        "applied": arr(applied),
        "sat": arr(sat),
        "state_reward": arr(state_rewards),
        "disturbance": disturbance,
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": term_reason,
    }

    n = len(trace["applied"])
    post = slice(1, n + 1)
    pos_des = trace["true_x"][post] - trace["desired_x"][post]
    pos_nom = trace["true_x"][post] - trace["nom_x"][post]
    att_nom = _rotation_error_norms(trace["nom_R"][post], trace["true_R"][post])

    start = int(round(float(transient_ignore_s) / float(env_cfg.dt)))
    start = min(max(start, 0), max(n - 1, 0))
    applied_norm = trace["applied"][start:] / WRENCH_SCALE[None, :]
    residual_m_norm = trace["residual"][start:, 1:4] / np.asarray(M_MAX)[None, :]
    d_applied = np.diff(applied_norm, axis=0) if len(applied_norm) > 1 else np.empty((0, 4))
    d_res_m = np.diff(residual_m_norm, axis=0) if len(residual_m_norm) > 1 else np.empty((0, 3))

    metrics = {
        "moment_scale": float(moment_scale),
        "episode_length": int(n),
        "completion_fraction": float(n / max(int(env_cfg.episode_steps), 1)),
        "terminated": float(terminated),
        "termination_reason": term_reason,
        "true_des_pos_rmse": _rms_norm(pos_des),
        "true_nom_pos_rmse": _rms_norm(pos_nom),
        "true_nom_att_rmse_rad": float(np.sqrt(np.mean(att_nom ** 2))) if len(att_nom) else 0.0,
        "model_ref_tracking_score": float(np.sum(trace["state_reward"]) / max(int(env_cfg.episode_steps), 1)),
        "actuator_sat_fraction": float(np.mean(trace["sat"])) if len(trace["sat"]) else 0.0,
        "applied_wrench_roughness": _rms_norm(d_applied),
        "applied_wrench_hf_ratio": _high_frequency_ratio(applied_norm, env_cfg.dt, hf_cutoff_hz),
        "residual_moment_roughness": _rms_norm(d_res_m),
        "residual_moment_hf_ratio": _high_frequency_ratio(residual_m_norm, env_cfg.dt, hf_cutoff_hz),
    }
    interaction = _opposition_metrics(
        trace["base"][:, 1:4], trace["residual"][:, 1:4], env_cfg.dt,
        transient_ignore_s=transient_ignore_s,
    )
    metrics["opposite_sign_fraction_when_active"] = interaction["aggregate_opposite_sign_fraction_when_active"]
    metrics["cosine_similarity_when_active"] = interaction["aggregate_cosine_similarity_when_active"]
    return metrics, trace, interaction


def _manual_outer_force_rollout(base_env_cfg, fixed_force: np.ndarray, oracle: bool,
                                hf_cutoff_hz: float, transient_ignore_s: float):
    """Pure-force experiment with nominal inertia/actuators and known force correction.

    In NED dynamics v_dot = g e3 - A/m + F_ext/m when R e3 tracks A.
    Therefore the exact disturbance-canceling correction to the controller's
    thrust vector A is +F_ext. We realize A_new = A + F_ext without changing
    Controller by passing a_modified = a_des - F_ext/m_nom.
    """
    env_cfg = copy.deepcopy(base_env_cfg)
    env_cfg.disturbances = ("none",)
    env_cfg.k_min = env_cfg.k_max = 1.0
    env_cfg.per_motor_params = False
    env = ResidualTwinEnv(env_cfg, seed=12345)
    env.reset()

    force = np.asarray(fixed_force, dtype=float).reshape(3)
    env.external_force = force.copy()
    env.dyn_true.external_force = force.copy()
    env.dyn_true.last_external_force = force.copy()

    times = [0.0]
    true_x, nom_x, desired_x = [], [], []
    true_R, nom_R, Rd_true = [], [], []
    base, applied = [], []
    sat = []
    state_reward = []

    def record_state(Rd=None):
        d = env.traj.desired(env.t)
        st, sn = env.dyn_true.state(), env.dyn_nom.state()
        desired_x.append(np.asarray(d["x"], dtype=float))
        true_x.append(st["x"])
        nom_x.append(sn["x"])
        true_R.append(st["R"])
        nom_R.append(sn["R"])
        if Rd is None:
            Rd_true.append(st["R"].copy())
        else:
            Rd_true.append(np.asarray(Rd, dtype=float).copy())

    record_state()
    terminated = truncated = False
    for _ in range(int(env_cfg.episode_steps)):
        desired = env.traj.desired(env.t)
        st, sn = env.dyn_true.state(), env.dyn_nom.state()

        desired_true = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in desired.items()}
        if oracle:
            # A_new = A + F_ext, and A contains -m*a_des.
            desired_true["a"] = np.asarray(desired["a"], dtype=float) - force / float(env.ctrl_true.mass)

        f_t, M_t, ctrl_info = env.ctrl_true.compute_control(st, desired_true)
        f_n, M_n, _ = env.ctrl_nom.compute_control(sn, desired)

        f_true, M_true, mix_true = env.mixer.apply(f_t, M_t, return_info=True)
        f_nom, M_nom, _ = env.mixer.apply_nominal(f_n, M_n, return_info=True)
        env.dyn_true.step(f_true, M_true)
        env.dyn_nom.step(f_nom, M_nom)
        env.step_idx += 1
        env.t = env.step_idx * env_cfg.dt

        base.append(np.array([f_t, *M_t], dtype=float))
        applied.append(np.array([f_true, *M_true], dtype=float))
        sat.append(float(bool(mix_true["saturated"])))
        state_reward.append(_state_reward_from_states(env.cfg, env.dyn_nom.state(), env.dyn_true.state()))
        times.append(float(env.t))
        record_state(ctrl_info["Rd"])

        term = env._check_terminated(env.dyn_nom.state(), env.dyn_true.state())
        trunc = env.step_idx >= env_cfg.episode_steps
        if term or trunc:
            terminated, truncated = bool(term), bool(trunc)
            break

    arr = lambda x: np.asarray(x, dtype=float)
    trace = {
        "time": arr(times),
        "true_x": arr(true_x),
        "nom_x": arr(nom_x),
        "desired_x": arr(desired_x),
        "true_R": arr(true_R),
        "nom_R": arr(nom_R),
        "Rd_true": arr(Rd_true),
        "base": arr(base),
        "applied": arr(applied),
        "sat": arr(sat),
        "state_reward": arr(state_reward),
        "fixed_external_force": force.copy(),
        "terminated": terminated,
        "truncated": truncated,
    }

    n = len(trace["applied"])
    post = slice(1, n + 1)
    pos_des = trace["true_x"][post] - trace["desired_x"][post]
    pos_nom = trace["true_x"][post] - trace["nom_x"][post]
    att_nom = _rotation_error_norms(trace["nom_R"][post], trace["true_R"][post])
    att_to_command = _rotation_error_norms(trace["Rd_true"][post], trace["true_R"][post])

    start = int(round(float(transient_ignore_s) / float(env_cfg.dt)))
    start = min(max(start, 0), max(n - 1, 0))
    applied_norm = trace["applied"][start:] / WRENCH_SCALE[None, :]
    d_applied = np.diff(applied_norm, axis=0) if len(applied_norm) > 1 else np.empty((0, 4))

    metrics = {
        "controller": "outer_force_oracle" if oracle else "baseline",
        "episode_length": int(n),
        "terminated": float(terminated),
        "true_des_pos_rmse": _rms_norm(pos_des),
        "true_nom_pos_rmse": _rms_norm(pos_nom),
        "true_nom_att_rmse_rad": float(np.sqrt(np.mean(att_nom ** 2))) if len(att_nom) else 0.0,
        "true_to_commanded_att_rmse_rad": float(np.sqrt(np.mean(att_to_command ** 2))) if len(att_to_command) else 0.0,
        "model_ref_tracking_score": float(np.sum(trace["state_reward"]) / max(int(env_cfg.episode_steps), 1)),
        "actuator_sat_fraction": float(np.mean(trace["sat"])) if len(trace["sat"]) else 0.0,
        "applied_wrench_roughness": _rms_norm(d_applied),
        "applied_wrench_hf_ratio": _high_frequency_ratio(applied_norm, env_cfg.dt, hf_cutoff_hz),
        "mean_true_tilt_deg": float(np.mean(_tilt_deg(trace["true_R"][post]))) if n else 0.0,
        "mean_commanded_tilt_deg": float(np.mean(_tilt_deg(trace["Rd_true"][post]))) if n else 0.0,
    }
    return metrics, trace


def _plot_moment_scaling(out: Path, results: list[tuple[dict, dict, dict]]):
    lambdas = [r[0]["moment_scale"] for r in results]
    rmse = [r[0]["true_des_pos_rmse"] for r in results]
    rough = [r[0]["applied_wrench_roughness"] for r in results]
    hf = [r[0]["applied_wrench_hf_ratio"] for r in results]
    sat = [r[0]["actuator_sat_fraction"] for r in results]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(lambdas, rmse, marker="o")
    ax.set_xlabel("Residual moment scale lambda")
    ax.set_ylabel("True -> desired position RMSE [m]")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "moment_scale_position_rmse.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(lambdas, rough, marker="o", label="applied wrench roughness")
    ax.plot(lambdas, hf, marker="s", label="applied wrench HF ratio")
    ax.plot(lambdas, sat, marker="^", label="actuator saturation fraction")
    ax.set_xlabel("Residual moment scale lambda")
    ax.set_ylabel("Dimensionless diagnostic")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "moment_scale_control_quality.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for metrics, trace, _ in results:
        n = len(trace["applied"])
        err = np.linalg.norm(trace["true_x"][1:n+1] - trace["nom_x"][1:n+1], axis=1)
        ax.plot(trace["time"][1:n+1], err, label=f"lambda={metrics['moment_scale']:g}")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("True -> nominal position error [m]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "moment_scale_model_reference_error.png", dpi=180)
    plt.close(fig)

    # Detailed moment traces for full residual authority.
    idx = int(np.argmin([abs(x - 1.0) for x in lambdas]))
    metrics, trace, _ = results[idx]
    t = trace["time"][:len(trace["applied"])]
    for j, label in enumerate(MOMENT_LABELS):
        fig, ax = plt.subplots(figsize=(9.0, 4.6))
        ax.plot(t, trace["base"][:, j+1], label="SE(3) requested")
        ax.plot(t, trace["residual"][:, j+1], label="residual")
        ax.plot(t, trace["cmd"][:, j+1], label="combined command", alpha=0.8)
        ax.plot(t, trace["applied"][:, j+1], label="actual applied", alpha=0.8)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"{label} [N m]")
        ax.set_title(f"{label}: direct SE(3) / residual interaction (lambda={metrics['moment_scale']:g})")
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(out / f"interaction_{label.lower()}_time.png", dpi=180)
        plt.close(fig)


def _plot_interaction_spectra(out: Path, trace: dict, dt: float, transient_ignore_s: float):
    start = int(round(transient_ignore_s / dt))
    b = trace["base"][start:, 1:4] / np.asarray(M_MAX)[None, :]
    r = trace["residual"][start:, 1:4] / np.asarray(M_MAX)[None, :]
    if len(b) < 8:
        return
    freq = np.fft.rfftfreq(len(b), d=dt)
    for j, label in enumerate(MOMENT_LABELS):
        B = np.abs(np.fft.rfft(b[:, j] - np.mean(b[:, j])))
        R = np.abs(np.fft.rfft(r[:, j] - np.mean(r[:, j])))
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        ax.plot(freq, B, label="SE(3)")
        ax.plot(freq, R, label="residual")
        ax.set_xlim(0.0, min(20.0, 0.5 / dt))
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Normalized FFT magnitude")
        ax.set_title(f"{label}: baseline vs residual spectrum")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"interaction_{label.lower()}_spectrum.png", dpi=180)
        plt.close(fig)


def _plot_oracle(out: Path, base_trace: dict, oracle_trace: dict):
    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    for trace, label in ((base_trace, "baseline"), (oracle_trace, "outer-force oracle")):
        n = len(trace["applied"])
        err = np.linalg.norm(trace["true_x"][1:n+1] - trace["desired_x"][1:n+1], axis=1)
        ax.plot(trace["time"][1:n+1], err, label=label)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("True -> desired position error [m]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "oracle_position_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    for trace, label in ((base_trace, "baseline true tilt"), (oracle_trace, "oracle true tilt")):
        n = len(trace["applied"])
        ax.plot(trace["time"][1:n+1], _tilt_deg(trace["true_R"][1:n+1]), label=label)
    n = len(oracle_trace["applied"])
    ax.plot(oracle_trace["time"][1:n+1], _tilt_deg(oracle_trace["Rd_true"][1:n+1]),
            label="oracle commanded tilt", linestyle="--")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Tilt from upright [deg]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "oracle_tilt.png", dpi=180)
    plt.close(fig)

    for j, label in enumerate(MOMENT_LABELS):
        fig, ax = plt.subplots(figsize=(8.5, 4.6))
        for trace, name in ((base_trace, "baseline"), (oracle_trace, "outer-force oracle")):
            n = len(trace["applied"])
            ax.plot(trace["time"][:n], trace["applied"][:, j+1], label=name)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel(f"Applied {label} [N m]")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"oracle_{label.lower()}_applied.png", dpi=180)
        plt.close(fig)


def _report(out: Path, case_meta: dict, sweep_rows: list[dict], opposition_rows: list[dict],
            oracle_rows: list[dict], fixed_force: np.ndarray):
    full = min(sweep_rows, key=lambda r: abs(r["moment_scale"] - 1.0))
    zero = min(sweep_rows, key=lambda r: abs(r["moment_scale"] - 0.0))
    b_oracle = next(r for r in oracle_rows if r["controller"] == "baseline")
    o_oracle = next(r for r in oracle_rows if r["controller"] == "outer_force_oracle")

    lines = [
        "# SE(3) / residual interaction diagnostics",
        "",
        "## Selected disturbance case",
        "",
        f"- Stage: `{case_meta['stage']}`",
        f"- Seed: `{case_meta['seed']}`",
    ]
    if case_meta.get("selected"):
        s = case_meta["selected"]
        lines += [f"- Auto-selection: `{s['selection_metric']}` = {s['selection_value']:.6g}"]
    lines += [
        "",
        "## 1. Residual-moment scaling test",
        "",
        "Only the *physical residual moment authority* was scaled; residual thrust remained unchanged.",
        "The policy was deterministic in every run and the disturbance seed was identical.",
        "",
        "| lambda | pos RMSE [m] | wrench roughness | HF ratio | saturation | opposite-sign active |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sweep_rows:
        lines.append(
            f"| {r['moment_scale']:.2f} | {r['true_des_pos_rmse']:.6f} | {r['applied_wrench_roughness']:.6f} "
            f"| {r['applied_wrench_hf_ratio']:.4f} | {r['actuator_sat_fraction']:.3f} "
            f"| {r['opposite_sign_fraction_when_active']:.3f} |"
        )

    lines += [
        "",
        "A strong monotonic increase in roughness/HF energy/saturation as lambda increases is causal evidence that the direct moment residual participates in the oscillation. It does not by itself prove whether the root cause is policy dynamics, SE(3) interaction, or saturation.",
        "",
        "## 2. Baseline-residual opposition test (lambda = 1)",
        "",
        "| axis | active fraction | opposite sign when active | cosine similarity | common frequency [Hz] | phase [deg] |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in opposition_rows:
        lines.append(
            f"| {r['axis']} | {r['active_fraction']:.3f} | {r['opposite_sign_fraction_when_active']:.3f} "
            f"| {r['cosine_similarity_when_active']:.3f} | {r['dominant_common_frequency_hz']:.3f} "
            f"| {r['phase_base_minus_residual_deg']:.1f} |"
        )
    lines += [
        "",
        "The strongest controller-fighting signature is: both commands active, a high opposite-sign fraction, negative cosine similarity, and a shared narrow-band frequency with phase near +/-180 deg.",
        "",
        "## 3. Oracle outer-force injection test",
        "",
        f"Isolated constant force: `{np.array2string(np.asarray(fixed_force), precision=3)}` N. Mass/inertia and actuators were nominal so this test isolates the injection location.",
        "",
        "| controller | pos RMSE [m] | true->nom attitude RMSE [rad] | true->command attitude RMSE [rad] | wrench roughness | HF ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in oracle_rows:
        lines.append(
            f"| {r['controller']} | {r['true_des_pos_rmse']:.6f} | {r['true_nom_att_rmse_rad']:.6f} "
            f"| {r['true_to_commanded_att_rmse_rad']:.6f} | {r['applied_wrench_roughness']:.6f} "
            f"| {r['applied_wrench_hf_ratio']:.4f} |"
        )

    oracle_improvement = 100.0 * (b_oracle["true_des_pos_rmse"] - o_oracle["true_des_pos_rmse"]) / max(b_oracle["true_des_pos_rmse"], EPS)
    rough_change = o_oracle["applied_wrench_roughness"] / max(b_oracle["applied_wrench_roughness"], EPS)
    lines += [
        "",
        f"Oracle position-RMSE improvement: **{oracle_improvement:.2f}%**.",
        f"Oracle/base applied-wrench roughness ratio: **{rough_change:.3f}x**.",
        "",
        "If the oracle produces low position error while the true attitude follows the *modified commanded attitude* smoothly, then the outer-force injection point is mechanically effective and worth testing with a learned correction. It is not evidence that SAC will automatically learn the correct outer-force mapping.",
        "",
        "## Interpretation guardrails",
        "",
        "- `lambda=0` still retains the learned residual thrust channel; it isolates direct residual moments, not the entire policy.",
        "- A phase near 180 deg is meaningful only when both moment signals have nontrivial amplitude at the reported common frequency.",
        "- Saturation can itself create oscillation. If saturation rises sharply with lambda, inspect motor limits before attributing everything to feedback-loop conflict.",
        "- The oracle uses exact disturbance knowledge and is only an injection-point feasibility test, not a deployable controller.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = _load_resolved_config(run_dir)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU")
        device = "cpu"
    cfg.device = device
    cfg.env.device = device

    stage_name, stage_cfg, seed, curriculum_path, selected = _load_case(args, cfg)
    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint)
    agent, _payload = _load_policy(cfg, stage_cfg, checkpoint, device)

    if args.output:
        out = Path(args.output).expanduser().resolve()
    else:
        out = run_dir / "evaluation" / f"interaction_diagnostics_{stage_name}_seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)

    lambdas = _parse_floats(args.moment_scales)
    if not lambdas:
        raise ValueError("--moment_scales must contain at least one value")
    if any(x < 0.0 for x in lambdas):
        raise ValueError("moment scales must be nonnegative")

    print(f"Selected case: stage={stage_name}, seed={seed}")
    print(f"Checkpoint: {checkpoint}")
    print("Running residual-moment scaling sweep...")
    sweep_results = []
    reference_disturbance = None
    for lam in lambdas:
        metrics, trace, interaction = _rollout_residual_scaled(
            stage_cfg, seed, agent, lam,
            hf_cutoff_hz=args.hf_cutoff_hz,
            transient_ignore_s=args.transient_ignore_s,
        )
        if reference_disturbance is None:
            reference_disturbance = trace["disturbance"]
        else:
            # Same seed/stage should produce exactly the same episode parameters.
            for key in ("external_force", "motor_coeff_scale", "moment_coeff_scale", "arm_length_scale"):
                if not np.allclose(reference_disturbance[key], trace["disturbance"][key], equal_nan=True):
                    raise RuntimeError(f"disturbance mismatch across lambda sweep for {key}")
            if not np.isclose(reference_disturbance["k"], trace["disturbance"]["k"], equal_nan=True):
                raise RuntimeError("mass/MOI mismatch across lambda sweep")
        sweep_results.append((metrics, trace, interaction))
        print(
            f"  lambda={lam:>4.2f}  RMSE={metrics['true_des_pos_rmse']:.5f} m  "
            f"rough={metrics['applied_wrench_roughness']:.4f}  HF={metrics['applied_wrench_hf_ratio']:.3f}  "
            f"sat={100*metrics['actuator_sat_fraction']:.1f}%  term={int(metrics['terminated'])}"
        )

    full_idx = int(np.argmin([abs(r[0]["moment_scale"] - 1.0) for r in sweep_results]))
    full_metrics, full_trace, full_interaction = sweep_results[full_idx]
    print("\nOpposition metrics at lambda closest to 1:")
    for row in full_interaction["axes"]:
        print(
            f"  {row['axis']}: active={row['active_fraction']:.2f}, "
            f"opposite={row['opposite_sign_fraction_when_active']:.2f}, "
            f"cos={row['cosine_similarity_when_active']:.2f}, "
            f"f={row['dominant_common_frequency_hz']:.2f} Hz, "
            f"phase={row['phase_base_minus_residual_deg']:.1f} deg"
        )

    print("\nRunning isolated oracle outer-force test...")
    fixed_force = np.asarray(_parse_floats(args.oracle_force, expected=3), dtype=float)
    oracle_rows, oracle_traces = [], []
    for oracle in (False, True):
        metrics, trace = _manual_outer_force_rollout(
            cfg.env, fixed_force, oracle,
            hf_cutoff_hz=args.hf_cutoff_hz,
            transient_ignore_s=args.transient_ignore_s,
        )
        oracle_rows.append(metrics)
        oracle_traces.append(trace)
        print(
            f"  {metrics['controller']:<18} RMSE={metrics['true_des_pos_rmse']:.5f} m  "
            f"rough={metrics['applied_wrench_roughness']:.4f}  HF={metrics['applied_wrench_hf_ratio']:.3f}"
        )

    sweep_rows = [r[0] for r in sweep_results]
    opposition_rows = full_interaction["axes"]
    _write_csv(out / "moment_scaling_summary.csv", sweep_rows)
    _write_csv(out / "opposition_metrics_lambda1.csv", opposition_rows)
    _write_csv(out / "oracle_outer_force_summary.csv", oracle_rows)
    _write_json(out / "case_metadata.json", {
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "curriculum": curriculum_path,
        "stage": stage_name,
        "seed": seed,
        "selected": selected,
        "disturbance": reference_disturbance,
        "moment_scales": lambdas,
        "oracle_force_NED_N": fixed_force,
        "hf_cutoff_hz": args.hf_cutoff_hz,
        "transient_ignore_s": args.transient_ignore_s,
        "note": "Oracle A correction is +F_ext in this NED implementation because dynamics contain -A/m + F_ext/m.",
    })

    _plot_moment_scaling(out, sweep_results)
    _plot_interaction_spectra(out, full_trace, stage_cfg.dt, args.transient_ignore_s)
    _plot_oracle(out, oracle_traces[0], oracle_traces[1])
    _report(
        out,
        {"stage": stage_name, "seed": seed, "selected": selected},
        sweep_rows,
        opposition_rows,
        oracle_rows,
        fixed_force,
    )

    print(f"\nDiagnostics written to: {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True, help="training trial directory")
    p.add_argument("--checkpoint", default="best", help="best, last, step number, or .pt path")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"))

    case = p.add_argument_group("problematic-case selection")
    case.add_argument("--evaluation_dir", default=None,
                      help="existing paired evaluation directory; enables automatic case selection")
    case.add_argument("--select", default="worst_roughness",
                      choices=("worst_roughness", "worst_hf", "worst_rmse_delta", "worst_residual_rmse", "residual_failure"))
    case.add_argument("--eval_curriculum", default=None,
                      help="curriculum TOML; defaults to evaluation_dir/evaluation_curriculum.toml or run_dir/curriculum.toml")
    case.add_argument("--stage", default=None, help="exact stage name or 1-based stage index; overrides auto-selected stage")
    case.add_argument("--seed", type=int, default=None, help="exact episode seed; overrides auto-selected seed")

    diag = p.add_argument_group("diagnostic settings")
    diag.add_argument("--moment_scales", default="0,0.25,0.5,1.0",
                      help="physical residual moment multipliers; residual thrust is unchanged")
    diag.add_argument("--hf_cutoff_hz", type=float, default=5.0)
    diag.add_argument("--transient_ignore_s", type=float, default=0.5)
    diag.add_argument("--oracle_force", default="3,0,0",
                      help="fixed NED external force [Fx,Fy,Fz] N for isolated oracle test")
    diag.add_argument("--output", default=None, help="output directory")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
