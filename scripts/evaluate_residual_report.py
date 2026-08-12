"""Paired baseline-vs-residual evaluation for a trained residual SAC policy.

The evaluator replays *exactly the same episode randomization* twice:
  1) baseline geometric controller only (residual action = 0),
  2) baseline + deterministic residual SAC policy.

It produces report-ready CSV/JSON/Markdown summaries and plots for tracking,
model-reference error, control effort/smoothness, actuator usage, and representative
trajectories. A curriculum TOML defines the disturbance distributions. By default
it uses the curriculum saved inside the training run, but a different TOML can be
passed for OOD/generalization testing without changing code.

Example
-------
PYTHONPATH=src python3 -m scripts.evaluate_residual_report \
    --run_dir runs_residual/residual_sac_curriculum/trial_002 \
    --checkpoint best \
    --episodes_per_stage 50 \
    --device cuda
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from robust_safe_rl.core.so3 import rotation_error
from robust_safe_rl.rl.config import Config
from robust_safe_rl.rl.curriculum import env_config_for_stage, load_curriculum
from robust_safe_rl.rl.mixer import F_MAX, M_MAX, MAX_MOTOR_THRUST
from robust_safe_rl.rl.residual_env import ResidualTwinEnv
from robust_safe_rl.rl.sac import SAC


EPS = 1e-12
WRENCH_SCALE = np.concatenate(([F_MAX], np.asarray(M_MAX, dtype=float)))
WRENCH_LABELS = ("Collective thrust [N]", "Mx [N m]", "My [N m]", "Mz [N m]")
ACTION_LABELS = ("thrust residual", "roll residual", "pitch residual", "yaw residual")


def _jsonify(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
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
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _load_resolved_config(run_dir: Path) -> Config:
    raw = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    cfg = Config()

    def apply(obj, values):
        for key, value in values.items():
            if not hasattr(obj, key):
                # Forward compatibility: evaluation should still work if a run
                # contains logging-only keys absent from the local code version.
                continue
            current = getattr(obj, key)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(obj, key, value)

    apply(cfg.env, raw.get("env", {}))
    apply(cfg.sac, raw.get("sac", {}))
    apply(cfg.net, raw.get("net", {}))
    top = {k: v for k, v in raw.items() if k not in {"env", "sac", "net"}}
    apply(cfg, top)
    return cfg


def _resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    candidate = Path(checkpoint).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    ckpt_dir = run_dir / "checkpoints"
    key = checkpoint.strip().lower()
    if key in {"best", "last", "interrupted"}:
        path = ckpt_dir / f"{key}.pt"
    elif key.isdigit():
        path = ckpt_dir / f"step_{int(key):09d}.pt"
    elif key.startswith("step_"):
        path = ckpt_dir / (key if key.endswith(".pt") else key + ".pt")
    else:
        path = ckpt_dir / checkpoint
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


def _load_policy(cfg: Config, env_cfg, checkpoint: Path, device: str):
    probe = ResidualTwinEnv(copy.deepcopy(env_cfg), seed=0)
    agent = SAC(probe.obs_dim, probe.action_dim, cfg.sac, cfg.net, device=device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if "agent" in payload and isinstance(payload["agent"], dict):
        payload = payload["agent"]
    if "actor" not in payload:
        raise KeyError("checkpoint does not contain an 'actor' state_dict")
    agent.actor.load_state_dict(payload["actor"])
    agent.actor.eval()
    return agent, payload


def _vec(value, n: int, fill=np.nan) -> np.ndarray:
    """Best-effort fixed-size float vector for compatibility with older envs."""
    if value is None:
        return np.full(n, fill, dtype=float)
    try:
        a = np.asarray(value, dtype=float).reshape(-1)
    except Exception:
        return np.full(n, fill, dtype=float)
    if a.size == 1:
        return np.full(n, float(a[0]), dtype=float)
    if a.size != n:
        return np.full(n, fill, dtype=float)
    return a.copy()


def _mixer_scales(env: ResidualTwinEnv):
    """Infer true kf/moment/arm scales directly from B_true/B_nom when possible."""
    mixer = getattr(env, "mixer", None)
    if mixer is None or not hasattr(mixer, "B_true") or not hasattr(mixer, "B_nom"):
        nan4 = np.full(4, np.nan, dtype=float)
        return nan4.copy(), nan4.copy(), nan4.copy(), "unavailable"
    try:
        Bt = np.asarray(mixer.B_true, dtype=float).reshape(4, 4)
        Bn = np.asarray(mixer.B_nom, dtype=float).reshape(4, 4)
        kf = Bt[0] / Bn[0]
        # B rows are [collective, roll, pitch, yaw].  Divide out kf first;
        # the remaining row ratios are geometry or momentConstant scales.
        arm_y = (Bt[1] / Bn[1]) / kf
        arm_x = (Bt[2] / Bn[2]) / kf
        arm = 0.5 * (arm_x + arm_y)
        moment = (Bt[3] / Bn[3]) / kf
        return kf.copy(), moment.copy(), arm.copy(), "mixer matrices"
    except Exception:
        nan4 = np.full(4, np.nan, dtype=float)
        return nan4.copy(), nan4.copy(), nan4.copy(), "unavailable"


def _disturbance_snapshot(env: ResidualTwinEnv, info: dict | None = None) -> dict:
    """Read the *actual* episode randomization across env code versions.

    Some local versions stored disturbances on ``env`` while others only kept
    the external force inside ``dyn_true`` or the actuator mismatch in the mixer
    matrix.  Evaluation must not depend on one storage convention.
    """
    info = {} if info is None else info
    sources = {}

    if "k" in info:
        k = float(info["k"])
        sources["k"] = "step info"
    elif hasattr(env, "k"):
        k = float(env.k)
        sources["k"] = "env.k"
    else:
        mass = float(getattr(getattr(env, "dyn_true", None), "mass", np.nan))
        mass_nom = float(getattr(getattr(env, "cfg", None), "mass_nom", np.nan))
        k = mass / mass_nom if np.isfinite(mass) and np.isfinite(mass_nom) and mass_nom != 0 else np.nan
        sources["k"] = "dyn_true.mass / mass_nom" if np.isfinite(k) else "unavailable"

    if "external_force" in info:
        force = _vec(info["external_force"], 3)
        sources["external_force"] = "step info"
    elif hasattr(env, "external_force"):
        force = _vec(getattr(env, "external_force"), 3)
        sources["external_force"] = "env.external_force"
    elif hasattr(getattr(env, "dyn_true", None), "external_force"):
        force = _vec(env.dyn_true.external_force, 3)
        sources["external_force"] = "dyn_true.external_force"
    else:
        force = np.full(3, np.nan, dtype=float)
        sources["external_force"] = "unavailable"

    mix_kf, mix_moment, mix_arm, mix_source = _mixer_scales(env)
    values = {}
    for key, mix_value in (
        ("motor_coeff_scale", mix_kf),
        ("moment_coeff_scale", mix_moment),
        ("arm_length_scale", mix_arm),
    ):
        if key in info:
            values[key] = _vec(info[key], 4)
            sources[key] = "step info"
        elif hasattr(env, key):
            values[key] = _vec(getattr(env, key), 4)
            sources[key] = f"env.{key}"
        else:
            values[key] = mix_value
            sources[key] = mix_source

    return {
        "k": k,
        "external_force": force,
        "motor_coeff_scale": values["motor_coeff_scale"],
        "moment_coeff_scale": values["moment_coeff_scale"],
        "arm_length_scale": values["arm_length_scale"],
        "_sources": sources,
    }


def _assert_same_disturbance(a: dict, b: dict):
    if not np.isclose(a["k"], b["k"], rtol=0.0, atol=1e-14, equal_nan=True):
        raise RuntimeError("paired environments sampled different mass/MOI scales")
    for key in ("external_force", "motor_coeff_scale", "moment_coeff_scale", "arm_length_scale"):
        if not np.allclose(a[key], b[key], rtol=0.0, atol=1e-14, equal_nan=True):
            raise RuntimeError(f"paired environments sampled different {key}")


def _state_reward_from_states(cfg, sn: dict, st: dict) -> float:
    """Recompute the model-reference state reward independently of env info."""
    ep = (np.asarray(sn["x"]) - np.asarray(st["x"])) / float(cfg.obs_pos_scale)
    ev = (np.asarray(sn["v"]) - np.asarray(st["v"])) / float(cfg.obs_vel_scale)
    eR = rotation_error(sn["R"], st["R"]) / float(cfg.obs_att_scale)
    ew = (np.asarray(sn["omega"]) - np.asarray(st["omega"])) / float(cfg.obs_omega_scale)
    raw = (
        float(cfg.w_pos) * np.exp(-float(np.dot(ep, ep)) / float(cfg.tau_pos) ** 2)
        + float(cfg.w_vel) * np.exp(-float(np.dot(ev, ev)) / float(cfg.tau_vel) ** 2)
        + float(cfg.w_att) * np.exp(-float(np.dot(eR, eR)) / float(cfg.tau_att) ** 2)
        + float(cfg.w_omega) * np.exp(-float(np.dot(ew, ew)) / float(cfg.tau_omega) ** 2)
    )
    return float(raw / float(cfg.reward_norm))


def _preflight_stage(stage_name: str, env_cfg, seed: int):
    """Verify that requested disturbances are actually active before a long run."""
    snapshots = []
    for offset in range(2):
        env = ResidualTwinEnv(copy.deepcopy(env_cfg), seed=int(seed + offset))
        env.reset()
        snapshots.append(_disturbance_snapshot(env))

    enabled = set(getattr(env_cfg, "disturbances", ("massmoi",)))
    if enabled == {"none"}:
        enabled = set()
    issues = []

    ks = np.asarray([d["k"] for d in snapshots], dtype=float)
    if "massmoi" in enabled and float(env_cfg.k_max) - float(env_cfg.k_min) > 1e-12:
        if np.all(np.isfinite(ks)) and np.allclose(ks, 1.0, atol=1e-14, rtol=0.0):
            issues.append("massmoi requested but sampled k stayed exactly 1")

    forces = np.asarray([d["external_force"] for d in snapshots], dtype=float)
    if "force" in enabled and float(getattr(env_cfg, "external_force_max", 0.0)) > 0.0:
        if np.all(np.isfinite(forces)) and np.allclose(forces, 0.0, atol=1e-14, rtol=0.0):
            issues.append("force requested but dyn_true external force stayed exactly zero")
        elif not np.all(np.isfinite(forces)):
            issues.append("force requested but evaluator cannot introspect the applied external force")

    for disturbance, key, lo_name, hi_name in (
        ("motor_coeff", "motor_coeff_scale", "motor_coeff_min", "motor_coeff_max"),
        ("moment_coeff", "moment_coeff_scale", "moment_coeff_min", "moment_coeff_max"),
        ("arm_length", "arm_length_scale", "arm_length_min", "arm_length_max"),
    ):
        if disturbance not in enabled:
            continue
        lo = float(getattr(env_cfg, lo_name))
        hi = float(getattr(env_cfg, hi_name))
        vals = np.asarray([d[key] for d in snapshots], dtype=float)
        if not np.all(np.isfinite(vals)):
            issues.append(f"{disturbance} requested but evaluator cannot introspect the true mixer parameters")
        elif hi - lo > 1e-12 and np.allclose(vals, 1.0, atol=1e-14, rtol=0.0):
            issues.append(f"{disturbance} requested but true mixer remained nominal")

    if issues:
        details = "; ".join(issues)
        raise RuntimeError(
            f"preflight failed for stage {stage_name!r}: {details}. "
            "This usually means src/robust_safe_rl/rl/residual_env.py is an older version "
            "that does not implement the configured disturbance family. Do not trust an "
            "OOD or curriculum comparison until this is resolved."
        )

    d = snapshots[0]
    def fmt(a):
        a = np.asarray(a, dtype=float)
        return "unknown" if not np.all(np.isfinite(a)) else np.array2string(a, precision=3, suppress_small=True)
    print(
        f"[preflight] {stage_name}: k={d['k']:.3f}  force={fmt(d['external_force'])}  "
        f"motor={fmt(d['motor_coeff_scale'])}  moment={fmt(d['moment_coeff_scale'])}  arm={fmt(d['arm_length_scale'])}"
    )


def _termination_reason(env: ResidualTwinEnv) -> str:
    sn = env.dyn_nom.state()
    st = env.dyn_true.state()
    reasons = []
    if np.linalg.norm(sn["x"] - st["x"]) > env.cfg.term_pos_error:
        reasons.append("model_ref_position")
    if st["R"][2, 2] < env._tilt_cos_thresh:
        reasons.append("tilt")
    return "+".join(reasons) if reasons else "other"


def _high_frequency_ratio(signal: np.ndarray, dt: float, cutoff_hz: float) -> float:
    x = np.asarray(signal, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < 4:
        return 0.0
    x = x - np.mean(x, axis=0, keepdims=True)
    spectrum = np.fft.rfft(x, axis=0)
    power = np.abs(spectrum) ** 2
    freq = np.fft.rfftfreq(len(x), d=dt)
    valid = freq > 0.0
    total = float(np.sum(power[valid]))
    if total <= EPS:
        return 0.0
    high = float(np.sum(power[freq >= cutoff_hz]))
    return high / total


def _rms_norm(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    if x.ndim == 1:
        return float(np.sqrt(np.mean(x * x)))
    return float(np.sqrt(np.mean(np.sum(x * x, axis=1))))


def _rmse_vec(a: np.ndarray, b: np.ndarray) -> float:
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return _rms_norm(d)


def _finite_mean(x, default=float("nan")) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else float(default)


def _finite_max(x, default=float("nan")) -> float:
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.max(a)) if a.size else float(default)


def _rotation_error_norms(R_ref: np.ndarray, R_true: np.ndarray) -> np.ndarray:
    return np.asarray([
        np.linalg.norm(rotation_error(R_ref[i], R_true[i]))
        for i in range(len(R_ref))
    ], dtype=float)


def _rollout(env_cfg, seed: int, agent=None, hf_cutoff_hz: float = 5.0, transient_ignore_s: float = 0.5):
    """Run one controller on one frozen episode randomization.

    If ``agent`` is None, residual action is identically zero (baseline only).
    State arrays include the initial state at t=0. Control/reward arrays contain
    one entry per executed transition.
    """
    env = ResidualTwinEnv(copy.deepcopy(env_cfg), seed=int(seed))
    obs = env.reset()
    disturbance = _disturbance_snapshot(env)

    times = [0.0]
    desired_x, desired_v = [], []
    true_x, true_v, true_R, true_omega = [], [], [], []
    nom_x, nom_v, nom_R, nom_omega = [], [], [], []

    def record_state():
        d = env.traj.desired(env.t)
        st = env.dyn_true.state()
        sn = env.dyn_nom.state()
        desired_x.append(np.asarray(d["x"], dtype=float))
        desired_v.append(np.asarray(d["v"], dtype=float))
        true_x.append(st["x"])
        true_v.append(st["v"])
        true_R.append(st["R"])
        true_omega.append(st["omega"])
        nom_x.append(sn["x"])
        nom_v.append(sn["v"])
        nom_R.append(sn["R"])
        nom_omega.append(sn["omega"])

    record_state()

    control_time = []
    actions = []
    residual_wrench = []
    base_requested = []
    command_wrench = []
    applied_wrench = []
    motor_cmd = []
    motor_sat = []
    state_reward = []
    full_reward = []
    effort_penalty = []
    smooth_penalty = []
    saturation = []

    terminated = False
    truncated = False
    term_reason = ""

    horizon = int(env_cfg.episode_steps)
    for _ in range(horizon):
        if agent is None:
            action = np.zeros(env.action_dim, dtype=np.float32)
        else:
            action = np.asarray(agent.act(obs, deterministic=True), dtype=np.float32)

        t_before = env.t
        obs, reward, term, trunc, info = env.step(action)

        control_time.append(float(t_before))
        actions.append(action.copy())

        # Physical residual wrench.  Newer envs expose it explicitly; older
        # direct-wrench envs can reconstruct it from normalized action scale.
        if "residual" in info:
            res = _vec(info.get("residual"), 4, fill=0.0)
        elif hasattr(env, "action_scale"):
            res = np.asarray(action, dtype=float) * _vec(env.action_scale, 4, fill=0.0)
        else:
            res = np.zeros(4, dtype=float)

        total = _vec(info.get("u_total"), 4)
        if not np.all(np.isfinite(total)):
            raise RuntimeError(
                "ResidualTwinEnv.step() did not expose info['u_total']; the evaluator "
                "needs the actual applied [f, Mx, My, Mz] wrench for control plots."
            )

        # New mixer-aware envs expose the pre-allocation combined command as
        # u_cmd.  In older direct-wrench envs, command == applied wrench.
        if "u_cmd" in info:
            cmd = _vec(info.get("u_cmd"), 4)
        else:
            cmd = total.copy()
        base = _vec(info.get("u_base"), 4)
        if not np.all(np.isfinite(base)):
            base = cmd - res

        # Motor-level logging was added after the original environment.  Keep
        # evaluation useful on older runs: mark unavailable motor channels NaN
        # rather than crashing or pretending they were measured.
        mc = _vec(info.get("motor_cmd"), 4)
        ms = _vec(info.get("motor_sat"), 4)
        sat_value = info.get("actuator_saturated", None)
        if (not np.all(np.isfinite(mc)) or not np.all(np.isfinite(ms))) and hasattr(getattr(env, "mixer", None), "allocate"):
            try:
                _, mc2, ms2, sat2 = env.mixer.allocate(cmd[0], cmd[1:4])
                mc, ms = _vec(mc2, 4), _vec(ms2, 4)
                if sat_value is None:
                    sat_value = bool(sat2)
            except Exception:
                pass

        # Recompute the model-reference state reward directly from the post-step
        # states.  This keeps the score correct even if an older env did not put
        # reward_state / shaping terms in info.
        sn_now = env.dyn_nom.state()
        st_now = env.dyn_true.state()
        sr = _state_reward_from_states(env.cfg, sn_now, st_now)
        previous_action = actions[-2] if len(actions) > 1 else np.zeros_like(action)
        effort = float(getattr(env.cfg, "w_action_effort", 0.0)) * float(np.linalg.norm(action))
        smooth = float(getattr(env.cfg, "w_action_smooth", 0.0)) * float(np.linalg.norm(action - previous_action))

        residual_wrench.append(res)
        base_requested.append(base)
        command_wrench.append(cmd)
        applied_wrench.append(total)
        motor_cmd.append(mc)
        motor_sat.append(ms)
        state_reward.append(sr)
        full_reward.append(float(reward))
        effort_penalty.append(float(info.get("reward_effort_penalty", effort)))
        smooth_penalty.append(float(info.get("reward_smooth_penalty", smooth)))
        saturation.append(float(bool(sat_value)) if sat_value is not None else np.nan)

        # Step info, when present, is the most authoritative disturbance record.
        disturbance = _disturbance_snapshot(env, info)

        times.append(float(env.t))
        record_state()

        if term or trunc:
            terminated = bool(term)
            truncated = bool(trunc)
            if terminated:
                term_reason = _termination_reason(env)
            break

    arr = lambda x: np.asarray(x, dtype=float)
    trace = {
        "time": arr(times),
        "control_time": arr(control_time),
        "desired_x": arr(desired_x),
        "desired_v": arr(desired_v),
        "true_x": arr(true_x),
        "true_v": arr(true_v),
        "true_R": arr(true_R),
        "true_omega": arr(true_omega),
        "nom_x": arr(nom_x),
        "nom_v": arr(nom_v),
        "nom_R": arr(nom_R),
        "nom_omega": arr(nom_omega),
        "action": arr(actions),
        "residual_wrench": arr(residual_wrench),
        "base_requested": arr(base_requested),
        "command_wrench": arr(command_wrench),
        "applied_wrench": arr(applied_wrench),
        "motor_cmd": arr(motor_cmd),
        "motor_sat": arr(motor_sat),
        "state_reward": arr(state_reward),
        "full_reward": arr(full_reward),
        "effort_penalty": arr(effort_penalty),
        "smooth_penalty": arr(smooth_penalty),
        "saturation": arr(saturation),
        "disturbance": disturbance,
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": term_reason,
    }

    # Metrics use post-transition states (index 1:), matching training evaluation.
    n = len(trace["full_reward"])
    post = slice(1, n + 1)
    p_td = trace["true_x"][post] - trace["desired_x"][post]
    p_tn = trace["true_x"][post] - trace["nom_x"][post]
    p_nd = trace["nom_x"][post] - trace["desired_x"][post]
    v_td = trace["true_v"][post] - trace["desired_v"][post]
    v_tn = trace["true_v"][post] - trace["nom_v"][post]
    w_tn = trace["true_omega"][post] - trace["nom_omega"][post]
    att_tn = _rotation_error_norms(trace["nom_R"][post], trace["true_R"][post])

    u_norm_all = trace["applied_wrench"] / WRENCH_SCALE[None, :]
    a_all = trace["action"]
    # Ignore the controller/finite-difference initialization transient when
    # quantifying chatter. Tracking RMSE and reward still use the full episode.
    steady = trace["control_time"] >= float(transient_ignore_s)
    if not np.any(steady):
        steady = np.ones(len(trace["control_time"]), dtype=bool)
    u_norm = u_norm_all[steady]
    a = a_all[steady]
    du_norm = np.diff(u_norm, axis=0) if len(u_norm) > 1 else np.empty((0, 4))
    da = np.diff(a, axis=0) if len(a) > 1 else np.empty((0, 4))

    metrics = {
        "episode_length": int(n),
        "completion_fraction": float(n / max(horizon, 1)),
        "terminated": float(terminated),
        "truncated": float(truncated),
        "full_return": float(np.sum(trace["full_reward"])),
        # Fixed-horizon state/model-reference score: missing steps after a
        # failure contribute zero, making early termination automatically poor.
        "model_ref_tracking_score": float(np.sum(trace["state_reward"]) / max(horizon, 1)),
        "true_des_pos_rmse": _rms_norm(p_td),
        "true_nom_pos_rmse": _rms_norm(p_tn),
        "nom_des_pos_rmse": _rms_norm(p_nd),
        "true_des_vel_rmse": _rms_norm(v_td),
        "true_nom_vel_rmse": _rms_norm(v_tn),
        "true_nom_att_rmse": float(np.sqrt(np.mean(att_tn ** 2))) if len(att_tn) else 0.0,
        "true_nom_omega_rmse": _rms_norm(w_tn),
        "true_des_pos_max": float(np.max(np.linalg.norm(p_td, axis=1))) if len(p_td) else 0.0,
        "true_nom_pos_max": float(np.max(np.linalg.norm(p_tn, axis=1))) if len(p_tn) else 0.0,
        "actuator_sat_fraction": _finite_mean(trace["saturation"]) if n else 0.0,
        "normalized_action_rms": _rms_norm(a),
        "normalized_action_peak": float(np.max(np.abs(a))) if a.size else 0.0,
        "normalized_action_roughness": _rms_norm(da),
        "applied_wrench_normalized_rms": _rms_norm(u_norm),
        "applied_wrench_roughness": _rms_norm(du_norm),
        "applied_wrench_hf_ratio": _high_frequency_ratio(u_norm, env_cfg.dt, hf_cutoff_hz),
        "action_hf_ratio": _high_frequency_ratio(a, env_cfg.dt, hf_cutoff_hz),
        "mean_effort_penalty": float(np.mean(trace["effort_penalty"])) if n else 0.0,
        "mean_smooth_penalty": float(np.mean(trace["smooth_penalty"])) if n else 0.0,
        "motor_utilization_mean": _finite_mean(trace["motor_sat"] / MAX_MOTOR_THRUST) if n else float("nan"),
        "motor_utilization_peak": _finite_max(trace["motor_sat"] / MAX_MOTOR_THRUST) if n else float("nan"),
        "termination_reason": term_reason,
    }
    channel_names = ("f", "mx", "my", "mz")
    for j, name in enumerate(channel_names):
        applied = trace["applied_wrench"][:, j]
        resid = trace["residual_wrench"][:, j]
        metrics[f"applied_{name}_rms"] = float(np.sqrt(np.mean(applied ** 2))) if len(applied) else 0.0
        metrics[f"applied_{name}_peak_abs"] = float(np.max(np.abs(applied))) if len(applied) else 0.0
        metrics[f"residual_{name}_rms"] = float(np.sqrt(np.mean(resid ** 2))) if len(resid) else 0.0
        metrics[f"action_{name}_rms"] = float(np.sqrt(np.mean(trace["action"][:, j] ** 2))) if n else 0.0
    trace["analysis_transient_ignore_s"] = float(transient_ignore_s)
    return metrics, trace


def _disturbance_columns(d: dict) -> dict:
    row = {"k": float(d["k"])}
    for j, name in enumerate(("x", "y", "z")):
        row[f"force_{name}_N"] = float(d["external_force"][j])
    row["force_norm_N"] = float(np.linalg.norm(d["external_force"]))
    for prefix, key in (
        ("motor_coeff", "motor_coeff_scale"),
        ("moment_coeff", "moment_coeff_scale"),
        ("arm_length", "arm_length_scale"),
    ):
        a = np.asarray(d[key], dtype=float)
        for j in range(4):
            row[f"{prefix}_{j}"] = float(a[j])
        row[f"{prefix}_mean"] = float(np.mean(a))
        row[f"{prefix}_min"] = float(np.min(a))
        row[f"{prefix}_max"] = float(np.max(a))
    return row


def _paired_row(stage: str, case: int, seed: int, disturbance: dict,
                baseline: dict, residual: dict) -> dict:
    row = {"stage": stage, "case": int(case), "seed": int(seed)}
    row.update(_disturbance_columns(disturbance))
    metric_keys = [k for k, v in baseline.items() if isinstance(v, (int, float, np.number))]
    for key in metric_keys:
        b = float(baseline[key])
        r = float(residual[key])
        row[f"baseline_{key}"] = b
        row[f"residual_{key}"] = r
        row[f"delta_{key}"] = r - b
        if key in {
            "true_des_pos_rmse", "true_nom_pos_rmse", "true_des_vel_rmse",
            "true_nom_vel_rmse", "true_nom_att_rmse", "true_nom_omega_rmse",
            "true_des_pos_max", "true_nom_pos_max", "actuator_sat_fraction",
            "applied_wrench_roughness", "applied_wrench_hf_ratio",
        }:
            row[f"improvement_pct_{key}"] = 100.0 * (b - r) / max(abs(b), EPS)
    row["residual_pos_rmse_win"] = float(
        residual["true_des_pos_rmse"] < baseline["true_des_pos_rmse"]
    )
    row["baseline_termination_reason"] = baseline.get("termination_reason", "")
    row["residual_termination_reason"] = residual.get("termination_reason", "")
    return row


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator,
                       n_boot: int = 2000) -> tuple[float, float]:
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan")
    if len(x) == 1:
        return float(x[0]), float(x[0])
    idx = rng.integers(0, len(x), size=(int(n_boot), len(x)))
    means = np.mean(x[idx], axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _aggregate(rows: list[dict], label: str, bootstrap_seed: int = 0) -> dict:
    rng = np.random.default_rng(bootstrap_seed)
    out = {"stage": label, "episodes": len(rows)}
    metrics = (
        "true_des_pos_rmse", "true_nom_pos_rmse", "nom_des_pos_rmse",
        "true_des_vel_rmse", "true_nom_vel_rmse", "true_nom_att_rmse",
        "true_nom_omega_rmse", "true_des_pos_max", "true_nom_pos_max",
        "model_ref_tracking_score", "full_return", "completion_fraction", "terminated",
        "actuator_sat_fraction", "normalized_action_rms", "normalized_action_roughness",
        "applied_wrench_normalized_rms", "applied_wrench_roughness",
        "applied_wrench_hf_ratio", "action_hf_ratio",
        "motor_utilization_mean", "motor_utilization_peak",
        "applied_f_rms", "applied_mx_rms", "applied_my_rms", "applied_mz_rms",
        "residual_f_rms", "residual_mx_rms", "residual_my_rms", "residual_mz_rms",
    )
    for metric in metrics:
        b_all = np.asarray([r[f"baseline_{metric}"] for r in rows], dtype=float)
        p_all = np.asarray([r[f"residual_{metric}"] for r in rows], dtype=float)
        mask = np.isfinite(b_all) & np.isfinite(p_all)
        b = b_all[mask]
        p = p_all[mask]
        d = p - b
        if len(b) == 0:
            for key in (
                f"baseline_{metric}_mean", f"baseline_{metric}_std",
                f"residual_{metric}_mean", f"residual_{metric}_std",
                f"paired_delta_{metric}_mean", f"paired_delta_{metric}_std",
                f"paired_delta_{metric}_ci95_low", f"paired_delta_{metric}_ci95_high",
            ):
                out[key] = float("nan")
            continue
        out[f"baseline_{metric}_mean"] = float(np.mean(b))
        out[f"baseline_{metric}_std"] = float(np.std(b, ddof=1)) if len(b) > 1 else 0.0
        out[f"residual_{metric}_mean"] = float(np.mean(p))
        out[f"residual_{metric}_std"] = float(np.std(p, ddof=1)) if len(p) > 1 else 0.0
        out[f"paired_delta_{metric}_mean"] = float(np.mean(d))
        out[f"paired_delta_{metric}_std"] = float(np.std(d, ddof=1)) if len(d) > 1 else 0.0
        lo, hi = _bootstrap_mean_ci(d, rng)
        out[f"paired_delta_{metric}_ci95_low"] = lo
        out[f"paired_delta_{metric}_ci95_high"] = hi

    rmse_b = np.asarray([r["baseline_true_des_pos_rmse"] for r in rows], dtype=float)
    rmse_r = np.asarray([r["residual_true_des_pos_rmse"] for r in rows], dtype=float)
    out["mean_position_rmse_improvement_pct"] = float(
        100.0 * (np.mean(rmse_b) - np.mean(rmse_r)) / max(abs(np.mean(rmse_b)), EPS)
    )
    out["position_rmse_win_rate"] = float(np.mean(rmse_r < rmse_b))
    diff = rmse_r - rmse_b
    sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
    out["paired_position_rmse_effect_size_dz"] = float(np.mean(diff) / sd) if sd > EPS else 0.0
    return out


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def _savefig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_aggregate(summary_rows: list[dict], output: Path):
    stages = [r["stage"] for r in summary_rows if r["stage"] != "ALL"]
    rows = [r for r in summary_rows if r["stage"] != "ALL"]
    if not rows:
        return
    x = np.arange(len(rows))
    width = 0.36

    def barplot(metric, ylabel, filename):
        bmean = [r[f"baseline_{metric}_mean"] for r in rows]
        bstd = [r[f"baseline_{metric}_std"] for r in rows]
        rmean = [r[f"residual_{metric}_mean"] for r in rows]
        rstd = [r[f"residual_{metric}_std"] for r in rows]
        plt.figure(figsize=(max(9, 1.5 * len(rows)), 5.4))
        plt.bar(x - width / 2, bmean, width, yerr=bstd, capsize=3, label="baseline")
        plt.bar(x + width / 2, rmean, width, yerr=rstd, capsize=3, label="baseline + residual")
        plt.xticks(x, stages, rotation=25, ha="right")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(axis="y", alpha=0.25)
        _savefig(output / filename)

    barplot("true_des_pos_rmse", "True -> desired position RMSE [m]", "position_rmse_by_stage.png")
    barplot("true_nom_pos_rmse", "True -> nominal position RMSE [m]", "model_reference_rmse_by_stage.png")
    barplot("model_ref_tracking_score", "Fixed-horizon model-reference tracking score", "tracking_score_by_stage.png")
    barplot("applied_wrench_roughness", "Applied wrench roughness [normalized RMS step change]", "control_roughness_by_stage.png")
    barplot("applied_wrench_hf_ratio", "Applied wrench high-frequency energy ratio", "control_high_frequency_ratio_by_stage.png")


def _plot_paired_scatter(rows: list[dict], output: Path):
    b = np.asarray([r["baseline_true_des_pos_rmse"] for r in rows])
    q = np.asarray([r["residual_true_des_pos_rmse"] for r in rows])
    if len(b) == 0:
        return
    hi = float(max(np.max(b), np.max(q), 1e-6))
    plt.figure(figsize=(6, 6))
    plt.scatter(b, q, alpha=0.7)
    plt.plot([0, hi], [0, hi], linestyle="--", label="equal performance")
    plt.xlabel("Baseline true -> desired RMSE [m]")
    plt.ylabel("Baseline + residual true -> desired RMSE [m]")
    plt.legend()
    plt.grid(alpha=0.25)
    _savefig(output / "paired_position_rmse_scatter.png")

    improvement = 100.0 * (b - q) / np.maximum(np.abs(b), EPS)
    plt.figure(figsize=(7, 4.8))
    plt.hist(improvement, bins=min(30, max(8, int(np.sqrt(len(improvement))))))
    plt.axvline(0.0, linestyle="--")
    plt.xlabel("Position RMSE improvement [%] (positive = residual better)")
    plt.ylabel("Episode count")
    plt.grid(axis="y", alpha=0.25)
    _savefig(output / "position_rmse_improvement_distribution.png")


def _plot_representative(stage: str, tag: str, baseline: dict, residual: dict, out: Path):
    case_dir = out / "representative_cases" / _safe_name(stage) / tag
    case_dir.mkdir(parents=True, exist_ok=True)

    # 3D path.
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(residual["desired_x"][:, 0], residual["desired_x"][:, 1], residual["desired_x"][:, 2], label="desired")
    ax.plot(baseline["true_x"][:, 0], baseline["true_x"][:, 1], baseline["true_x"][:, 2], label="baseline")
    ax.plot(residual["true_x"][:, 0], residual["true_x"][:, 1], residual["true_x"][:, 2], label="baseline + residual")
    ax.plot(residual["nom_x"][:, 0], residual["nom_x"][:, 1], residual["nom_x"][:, 2], label="nominal reference", alpha=0.75)
    ax.set_xlabel("North [m]")
    ax.set_ylabel("East [m]")
    ax.set_zlabel("Down [m]")
    ax.invert_zaxis()
    ax.legend()
    plt.tight_layout()
    plt.savefig(case_dir / "trajectory_3d.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Position states vs time.
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ("North x [m]", "East y [m]", "Down z [m]")
    for j, ax in enumerate(axes):
        ax.plot(residual["time"], residual["desired_x"][:, j], label="desired")
        ax.plot(baseline["time"], baseline["true_x"][:, j], label="baseline")
        ax.plot(residual["time"], residual["true_x"][:, j], label="baseline + residual")
        ax.plot(residual["time"], residual["nom_x"][:, j], label="nominal", alpha=0.75)
        ax.set_ylabel(labels[j])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(ncol=4)
    _savefig(case_dir / "position_time.png")

    # Desired tracking and model-reference position error.
    plt.figure(figsize=(10, 5.5))
    b_des = np.linalg.norm(baseline["true_x"] - baseline["desired_x"], axis=1)
    r_des = np.linalg.norm(residual["true_x"] - residual["desired_x"], axis=1)
    b_nom = np.linalg.norm(baseline["true_x"] - baseline["nom_x"], axis=1)
    r_nom = np.linalg.norm(residual["true_x"] - residual["nom_x"], axis=1)
    plt.plot(baseline["time"], b_des, label="baseline: true -> desired")
    plt.plot(residual["time"], r_des, label="residual: true -> desired")
    plt.plot(baseline["time"], b_nom, linestyle="--", label="baseline: true -> nominal")
    plt.plot(residual["time"], r_nom, linestyle="--", label="residual: true -> nominal")
    plt.xlabel("Time [s]")
    plt.ylabel("Position error norm [m]")
    plt.legend()
    plt.grid(alpha=0.25)
    _savefig(case_dir / "position_error_norms.png")

    # All model-reference state-group errors.
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for trace, prefix in ((baseline, "baseline"), (residual, "baseline + residual")):
        t = trace["time"]
        ep = np.linalg.norm(trace["true_x"] - trace["nom_x"], axis=1)
        ev = np.linalg.norm(trace["true_v"] - trace["nom_v"], axis=1)
        eR = _rotation_error_norms(trace["nom_R"], trace["true_R"])
        ew = np.linalg.norm(trace["true_omega"] - trace["nom_omega"], axis=1)
        axes[0].plot(t, ep, label=prefix)
        axes[1].plot(t, ev, label=prefix)
        axes[2].plot(t, eR, label=prefix)
        axes[3].plot(t, ew, label=prefix)
    axes[0].set_ylabel("Position [m]")
    axes[1].set_ylabel("Velocity [m/s]")
    axes[2].set_ylabel("Attitude error [rad]")
    axes[3].set_ylabel("Angular rate [rad/s]")
    axes[3].set_xlabel("Time [s]")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend()
    _savefig(case_dir / "model_reference_state_errors.png")

    # Actual applied wrench comparison.
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(baseline["control_time"], baseline["applied_wrench"][:, j], label="baseline actual")
        ax.plot(residual["control_time"], residual["applied_wrench"][:, j], label="residual actual")
        ax.set_ylabel(WRENCH_LABELS[j])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time [s]")
    axes[0].legend()
    _savefig(case_dir / "applied_wrench_comparison.png")

    # Residual controller decomposition: baseline request + residual = command -> actual.
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
    for j, ax in enumerate(axes):
        t = residual["control_time"]
        ax.plot(t, residual["base_requested"][:, j], label="base request")
        ax.plot(t, residual["residual_wrench"][:, j], label="residual correction")
        ax.plot(t, residual["command_wrench"][:, j], label="combined command")
        ax.plot(t, residual["applied_wrench"][:, j], linestyle="--", label="actual wrench")
        ax.set_ylabel(WRENCH_LABELS[j])
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time [s]")
    axes[0].legend(ncol=4)
    _savefig(case_dir / "residual_wrench_decomposition.png")

    # Normalized SAC action.
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(residual["control_time"], residual["action"][:, j])
        ax.set_ylabel(ACTION_LABELS[j])
        ax.set_ylim(-1.05, 1.05)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time [s]")
    _savefig(case_dir / "normalized_residual_action.png")

    # Rotor command utilization, after saturation.  Older direct-wrench envs
    # did not expose motor-level data; show that explicitly instead of crashing.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    motor_available = (
        baseline["motor_sat"].ndim == 2 and residual["motor_sat"].ndim == 2
        and np.any(np.isfinite(baseline["motor_sat"]))
        and np.any(np.isfinite(residual["motor_sat"]))
    )
    if motor_available:
        for rotor in range(4):
            axes[0].plot(baseline["control_time"], baseline["motor_sat"][:, rotor] / MAX_MOTOR_THRUST, label=f"motor {rotor}")
            axes[1].plot(residual["control_time"], residual["motor_sat"][:, rotor] / MAX_MOTOR_THRUST, label=f"motor {rotor}")
        axes[0].legend(ncol=4)
    else:
        for ax in axes:
            ax.text(0.5, 0.5, "motor-level data unavailable in this env version", ha="center", va="center", transform=ax.transAxes)
    axes[0].set_title("Baseline motor utilization")
    axes[1].set_title("Baseline + residual motor utilization")
    for ax in axes:
        ax.set_ylabel("fraction of max")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time [s]")
    _savefig(case_dir / "motor_utilization.png")

    # Model-reference state reward.
    plt.figure(figsize=(10, 4.8))
    plt.plot(baseline["control_time"], baseline["state_reward"], label="baseline")
    plt.plot(residual["control_time"], residual["state_reward"], label="baseline + residual")
    plt.xlabel("Time [s]")
    plt.ylabel("State/model-reference reward")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.grid(alpha=0.25)
    _savefig(case_dir / "tracking_reward.png")

    # Applied-wrench spectrum on normalized channels, useful for spotting chatter.
    plt.figure(figsize=(9, 5))
    for trace, label in ((baseline, "baseline"), (residual, "baseline + residual")):
        mask = trace["control_time"] >= float(trace.get("analysis_transient_ignore_s", 0.0))
        x = (trace["applied_wrench"] / WRENCH_SCALE[None, :])[mask]
        t_spec = trace["control_time"][mask]
        if len(x) < 4:
            continue
        x = x - np.mean(x, axis=0, keepdims=True)
        spec = np.abs(np.fft.rfft(x, axis=0)) ** 2
        psd = np.sum(spec, axis=1)
        freq = np.fft.rfftfreq(len(x), d=float(np.median(np.diff(t_spec))) if len(x) > 1 else 0.01)
        if len(psd) > 1 and np.sum(psd[1:]) > EPS:
            psd = psd / np.sum(psd[1:])
        plt.semilogy(freq[1:], psd[1:] + EPS, label=label)
    plt.xlim(0.0, 20.0)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Normalized wrench spectral energy")
    plt.legend()
    plt.grid(alpha=0.25)
    _savefig(case_dir / "applied_wrench_spectrum.png")


def _make_markdown_report(path: Path, checkpoint: Path, summaries: list[dict],
                          all_rows: list[dict], episodes_per_stage: int,
                          hf_cutoff_hz: float, transient_ignore_s: float):
    overall = next(r for r in summaries if r["stage"] == "ALL")
    b_rmse = overall["baseline_true_des_pos_rmse_mean"]
    r_rmse = overall["residual_true_des_pos_rmse_mean"]
    b_score = overall["baseline_model_ref_tracking_score_mean"]
    r_score = overall["residual_model_ref_tracking_score_mean"]
    b_model = overall["baseline_true_nom_pos_rmse_mean"]
    r_model = overall["residual_true_nom_pos_rmse_mean"]
    b_term = overall["baseline_terminated_mean"]
    r_term = overall["residual_terminated_mean"]
    win = overall["position_rmse_win_rate"]
    ci_lo = overall["paired_delta_true_des_pos_rmse_ci95_low"]
    ci_hi = overall["paired_delta_true_des_pos_rmse_ci95_high"]

    lines = [
        "# Residual SAC paired evaluation",
        "",
        f"- Checkpoint: `{checkpoint}`",
        f"- Paired episodes: **{len(all_rows)}** ({episodes_per_stage} per evaluated stage)",
        "- Every baseline/residual pair uses the **same seed and exactly the same sampled disturbances/uncertainties**.",
        "- Residual policy is evaluated deterministically (`tanh(mean)`).",
        f"- High-frequency control metric uses a **{hf_cutoff_hz:g} Hz** cutoff on actuator-envelope-normalized applied wrench, after ignoring the first **{transient_ignore_s:g} s** controller-initialization transient.",
        "",
        "## Overall comparison",
        "",
        f"- True -> desired position RMSE: baseline **{b_rmse:.6f} m**, residual **{r_rmse:.6f} m** "
        f"({overall['mean_position_rmse_improvement_pct']:.2f}% improvement in the mean).",
        f"- Paired RMSE difference (residual - baseline) 95% bootstrap CI: **[{ci_lo:.6g}, {ci_hi:.6g}] m**.",
        f"- Residual has lower position RMSE in **{100.0 * win:.1f}%** of paired episodes.",
        f"- True -> nominal RMSE: baseline **{b_model:.6f} m**, residual **{r_model:.6f} m**.",
        f"- Fixed-horizon model-reference tracking score: baseline **{b_score:.5f}**, residual **{r_score:.5f}**.",
        f"- Termination rate: baseline **{100*b_term:.2f}%**, residual **{100*r_term:.2f}%**.",
        "",
        "## Per-stage summary",
        "",
        "| Stage | Baseline RMSE mean±std [m] | Residual RMSE mean±std [m] | Mean improvement | Residual win rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in summaries:
        if r["stage"] == "ALL":
            continue
        lines.append(
            f"| {r['stage']} | {r['baseline_true_des_pos_rmse_mean']:.6f} ± {r['baseline_true_des_pos_rmse_std']:.6f} "
            f"| {r['residual_true_des_pos_rmse_mean']:.6f} ± {r['residual_true_des_pos_rmse_std']:.6f} "
            f"| {r['mean_position_rmse_improvement_pct']:.2f}% | {100*r['position_rmse_win_rate']:.1f}% |"
        )

    lines += [
        "",
        "## How to interpret the control plots",
        "",
        "- `applied_wrench_comparison.png`: directly checks whether the residual controller introduces visible force/moment chatter.",
        "- `residual_wrench_decomposition.png`: separates the baseline request, learned correction, combined command, and true applied wrench.",
        "- `normalized_residual_action.png`: shows whether SAC is spending a large fraction of its ±residual authority or rapidly switching sign.",
        "- `motor_utilization.png`: checks rotor saturation/headroom rather than only component-wise wrench limits.",
        "- `applied_wrench_spectrum.png`: highlights high-frequency control energy that may be hard to notice in time traces.",
        "",
        "## Recommended follow-up evaluations",
        "",
        "1. **OOD ranges:** evaluate with a separate curriculum TOML using uncertainty ranges outside training (for example 0.6–1.4) and time-varying disturbances when implemented.",
        "2. **Asymmetric motor degradation:** test per-motor coefficient/arm perturbations even if training used global symmetric scales.",
        "3. **Trajectory generalization:** repeat on hover, circle, different figure-eight amplitudes/speeds, and trajectory periods not seen in training.",
        "4. **Authority ablation:** compare residual authority values such as 5%, 10%, 20%, and 30% to quantify robustness gained per unit control authority.",
        "5. **History ablation:** compare the 10-step history against shorter history/no history to show whether temporal context materially helps.",
        "6. **Curriculum ablation:** compare curriculum+rehearsal, curriculum without rehearsal, and full-domain training from the start.",
        "7. **Noise/delay:** add sensor noise, state-estimation error, latency, and motor lag before making real-system claims.",
        "8. **Worst-case search:** after random Monte Carlo evaluation, explicitly optimize/search disturbance parameters for cases that maximize tracking error.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    cfg = _load_resolved_config(run_dir)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU")
        device = "cpu"
    cfg.device = device
    cfg.env.device = device

    curriculum_path = Path(args.eval_curriculum).expanduser().resolve() if args.eval_curriculum else run_dir / "curriculum.toml"
    if not curriculum_path.is_file():
        # Non-curriculum fallback: evaluate the base environment as one stage.
        curriculum = None
        stage_names = ["fixed_distribution"]
        stage_cfgs = [copy.deepcopy(cfg.env)]
    else:
        curriculum = load_curriculum(curriculum_path)
        stage_names = [s.name for s in curriculum.stages]
        stage_cfgs = [env_config_for_stage(cfg.env, s) for s in curriculum.stages]

    if args.stages:
        requested = {x.strip() for x in args.stages.split(",") if x.strip()}
        keep = [i for i, name in enumerate(stage_names) if name in requested or str(i + 1) in requested]
        if not keep:
            raise ValueError(f"--stages matched none of: {stage_names}")
        stage_names = [stage_names[i] for i in keep]
        stage_cfgs = [stage_cfgs[i] for i in keep]

    # Fail fast if the local environment is stale and silently ignores a
    # disturbance family requested by the training/evaluation TOML.
    print("Running disturbance preflight...")
    for i, (stage_name, stage_cfg) in enumerate(zip(stage_names, stage_cfgs)):
        _preflight_stage(stage_name, stage_cfg, int(args.seed + 10_000_000 + 100 * i))

    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint)
    agent, ckpt_payload = _load_policy(cfg, stage_cfgs[0], checkpoint, device)

    if args.output:
        output = Path(args.output).expanduser().resolve()
    else:
        output = run_dir / "evaluation" / f"baseline_vs_residual_{checkpoint.stem}"
    output.mkdir(parents=True, exist_ok=True)

    metadata = {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_training": _jsonify(ckpt_payload.get("_training", {})),
        "evaluation_curriculum": str(curriculum_path) if curriculum_path.is_file() else None,
        "stages": stage_names,
        "episodes_per_stage": int(args.episodes_per_stage),
        "base_seed": int(args.seed),
        "device": device,
        "hf_cutoff_hz": float(args.hf_cutoff_hz),
        "transient_ignore_s": float(args.transient_ignore_s),
        "paired_randomization": True,
    }
    _write_json(output / "evaluation_config.json", metadata)
    # Preserve the resolved training configuration beside the report.
    (output / "training_config.json").write_text((run_dir / "config.json").read_text(encoding="utf-8"), encoding="utf-8")
    if curriculum_path.is_file():
        (output / "evaluation_curriculum.toml").write_text(curriculum_path.read_text(encoding="utf-8"), encoding="utf-8")

    paired_rows = []
    long_rows = []
    representative = {}

    total_cases = len(stage_names) * int(args.episodes_per_stage)
    done_cases = 0
    for si, (stage_name, stage_cfg) in enumerate(zip(stage_names, stage_cfgs)):
        stage_records = []
        for case in range(int(args.episodes_per_stage)):
            seed = int(args.seed + 100_000 * si + case)
            b_metrics, b_trace = _rollout(stage_cfg, seed, agent=None, hf_cutoff_hz=args.hf_cutoff_hz, transient_ignore_s=args.transient_ignore_s)
            r_metrics, r_trace = _rollout(stage_cfg, seed, agent=agent, hf_cutoff_hz=args.hf_cutoff_hz, transient_ignore_s=args.transient_ignore_s)
            _assert_same_disturbance(b_trace["disturbance"], r_trace["disturbance"])

            pair = _paired_row(stage_name, case, seed, b_trace["disturbance"], b_metrics, r_metrics)
            paired_rows.append(pair)
            stage_records.append((pair, b_trace, r_trace))

            dist_cols = _disturbance_columns(b_trace["disturbance"])
            for controller, metrics in (("baseline", b_metrics), ("baseline_plus_residual", r_metrics)):
                row = {"stage": stage_name, "case": case, "seed": seed, "controller": controller, **dist_cols}
                row.update(metrics)
                long_rows.append(row)

            done_cases += 1
            if done_cases == 1 or done_cases % max(1, args.progress_every) == 0 or done_cases == total_cases:
                print(
                    f"[{done_cases:>4}/{total_cases}] {stage_name} case {case:03d}  "
                    f"RMSE base={b_metrics['true_des_pos_rmse']:.5f} m  "
                    f"res={r_metrics['true_des_pos_rmse']:.5f} m"
                )

        # Keep traces only for report plots: median baseline difficulty plus,
        # optionally, additional hardest-baseline cases.
        stage_records.sort(key=lambda item: item[0]["baseline_true_des_pos_rmse"])
        nplot = min(max(int(args.plot_cases_per_stage), 0), len(stage_records))
        chosen = []
        if nplot > 0:
            chosen.append(("median", stage_records[len(stage_records) // 2]))
        if nplot > 1:
            chosen.append(("hardest_baseline", stage_records[-1]))
        if nplot > 2:
            chosen.append(("easiest_baseline", stage_records[0]))
        for tag, (pair, b_trace, r_trace) in chosen[:nplot]:
            representative[(stage_name, tag)] = (pair, b_trace, r_trace)

    _write_csv(output / "episode_metrics.csv", long_rows)
    _write_csv(output / "paired_episode_metrics.csv", paired_rows)

    summaries = []
    for si, stage in enumerate(stage_names):
        rows = [r for r in paired_rows if r["stage"] == stage]
        summaries.append(_aggregate(rows, stage, bootstrap_seed=args.seed + si))
    summaries.append(_aggregate(paired_rows, "ALL", bootstrap_seed=args.seed + 9999))
    _write_csv(output / "summary_by_stage.csv", summaries)
    _write_json(output / "summary.json", {"by_stage": summaries[:-1], "overall": summaries[-1]})

    _plot_aggregate(summaries, output)
    _plot_paired_scatter(paired_rows, output)
    for (stage, tag), (pair, b_trace, r_trace) in representative.items():
        case_dir = output / "representative_cases" / _safe_name(stage) / tag
        _write_json(case_dir / "case_info.json", pair)
        _plot_representative(stage, tag, b_trace, r_trace, output)

    _make_markdown_report(
        output / "REPORT.md",
        checkpoint,
        summaries,
        paired_rows,
        int(args.episodes_per_stage),
        float(args.hf_cutoff_hz),
        float(args.transient_ignore_s),
    )

    overall = summaries[-1]
    print("\n=== Overall paired evaluation ===")
    print(
        f"true->desired position RMSE: "
        f"baseline {overall['baseline_true_des_pos_rmse_mean']:.6f} +/- {overall['baseline_true_des_pos_rmse_std']:.6f} m  |  "
        f"residual {overall['residual_true_des_pos_rmse_mean']:.6f} +/- {overall['residual_true_des_pos_rmse_std']:.6f} m"
    )
    print(f"mean RMSE improvement: {overall['mean_position_rmse_improvement_pct']:.2f}%")
    print(f"residual win rate: {100.0 * overall['position_rmse_win_rate']:.1f}%")
    print(
        f"model-reference tracking score: "
        f"baseline {overall['baseline_model_ref_tracking_score_mean']:.5f}  |  "
        f"residual {overall['residual_model_ref_tracking_score_mean']:.5f}"
    )
    print(
        f"termination rate: baseline {100.0 * overall['baseline_terminated_mean']:.2f}%  |  "
        f"residual {100.0 * overall['residual_terminated_mean']:.2f}%"
    )
    print(f"report written to: {output}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True, help="training trial directory containing config.json and checkpoints/")
    p.add_argument("--checkpoint", default="best", help="best, last, interrupted, step number, or checkpoint path")
    p.add_argument("--episodes_per_stage", type=int, default=50, help="paired Monte Carlo episodes per stage")
    p.add_argument("--seed", type=int, default=20260810, help="base seed for unseen paired evaluation cases")
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--eval_curriculum", default=None, help="optional TOML; defaults to <run_dir>/curriculum.toml")
    p.add_argument("--stages", default=None, help="optional comma-separated stage names or 1-based stage numbers")
    p.add_argument("--output", default=None, help="output directory; default is inside the run's evaluation/ folder")
    p.add_argument("--plot_cases_per_stage", type=int, default=1, help="0..3 representative paired cases plotted per stage")
    p.add_argument("--hf_cutoff_hz", type=float, default=5.0, help="cutoff for high-frequency normalized-wrench energy ratio")
    p.add_argument("--transient_ignore_s", type=float, default=0.5, help="initial seconds ignored only for control-chatter metrics")
    p.add_argument("--progress_every", type=int, default=10, help="print progress every N paired cases")
    args = p.parse_args()
    if args.episodes_per_stage < 1:
        p.error("--episodes_per_stage must be >= 1")
    if args.plot_cases_per_stage < 0:
        p.error("--plot_cases_per_stage must be >= 0")
    if args.hf_cutoff_hz <= 0:
        p.error("--hf_cutoff_hz must be > 0")
    if args.transient_ignore_s < 0:
        p.error("--transient_ignore_s must be >= 0")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
