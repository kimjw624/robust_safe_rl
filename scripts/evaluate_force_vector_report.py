"""Paired evaluation for the 3-D force-vector (delta_A) residual SAC policy.

Each Monte Carlo case is replayed twice with exactly the same sampled plant
uncertainties/disturbances:

  1) geometric baseline only (delta_A = 0),
  2) geometric baseline + deterministic SAC delta_A residual.

The primary metric is true-to-desired position tracking.  The report also
compares fixed-horizon tracking reward, termination, actuator saturation,
actual-wrench roughness/high-frequency energy, motor utilization, and the
learned delta_A signal itself.  Nominal-to-desired RMSE is deliberately omitted
from the main report.

Example
-------
PYTHONPATH=src python3 -m scripts.evaluate_force_vector_report \
    --run_dir runs_residual/residual_sac_force_vector/trial_001 \
    --checkpoint best \
    --episodes_per_stage 100 \
    --device cuda
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from robust_safe_rl.rl.config import Config
from robust_safe_rl.rl.curriculum import env_config_for_stage, load_curriculum
from robust_safe_rl.rl.mixer import F_MAX, M_MAX, MAX_MOTOR_THRUST
from robust_safe_rl.rl.residual_env import ResidualTwinEnv
from robust_safe_rl.rl.sac import SAC


EPS = 1e-12
WRENCH_SCALE = np.concatenate(([F_MAX], np.asarray(M_MAX, dtype=float)))
WRENCH_LABELS = ("Collective thrust [N]", "Mx [N m]", "My [N m]", "Mz [N m]")
A_LABELS = ("delta A_x [N]", "delta A_y [N]", "delta A_z [N]")


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
    fields, seen = [], set()
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
                continue
            current = getattr(obj, key)
            if isinstance(current, tuple) and isinstance(value, list):
                value = tuple(value)
            setattr(obj, key, value)

    apply(cfg.env, raw.get("env", {}))
    apply(cfg.sac, raw.get("sac", {}))
    apply(cfg.net, raw.get("net", {}))
    apply(cfg, {k: v for k, v in raw.items() if k not in {"env", "sac", "net"}})
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
    if probe.residual_interface != "force_vector":
        raise ValueError(
            f"run is configured for residual_interface={probe.residual_interface!r}; "
            "this evaluator is for 'force_vector' runs"
        )
    agent = SAC(probe.obs_dim, probe.action_dim, cfg.sac, cfg.net, device=device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if "agent" in payload and isinstance(payload["agent"], dict):
        payload = payload["agent"]
    if "actor" not in payload:
        raise KeyError("checkpoint does not contain an 'actor' state_dict")
    agent.actor.load_state_dict(payload["actor"])
    agent.actor.eval()
    return agent, payload


def _vec(value, n: int, fill=np.nan):
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


def _disturbance_snapshot(env: ResidualTwinEnv, info=None):
    info = {} if info is None else info
    return {
        "k": float(info.get("k", getattr(env, "k", np.nan))),
        "external_force": _vec(info.get("external_force", getattr(env, "external_force", None)), 3),
        "motor_coeff_scale": _vec(info.get("motor_coeff_scale", getattr(env, "motor_coeff_scale", None)), 4),
        "moment_coeff_scale": _vec(info.get("moment_coeff_scale", getattr(env, "moment_coeff_scale", None)), 4),
        "arm_length_scale": _vec(info.get("arm_length_scale", getattr(env, "arm_length_scale", None)), 4),
    }


def _assert_same_disturbance(a: dict, b: dict):
    if not np.isclose(a["k"], b["k"], rtol=0.0, atol=1e-14, equal_nan=True):
        raise RuntimeError("paired environments sampled different mass/MOI scales")
    for key in ("external_force", "motor_coeff_scale", "moment_coeff_scale", "arm_length_scale"):
        if not np.allclose(a[key], b[key], rtol=0.0, atol=1e-14, equal_nan=True):
            raise RuntimeError(f"paired environments sampled different {key}")


def _preflight_stage(stage_name: str, env_cfg, seed: int):
    snapshots = []
    for offset in range(2):
        env = ResidualTwinEnv(copy.deepcopy(env_cfg), seed=seed + offset)
        if env.residual_interface != "force_vector":
            raise RuntimeError("preflight expected residual_interface='force_vector'")
        env.reset()
        snapshots.append(_disturbance_snapshot(env))

    enabled = set(getattr(env_cfg, "disturbances", ("massmoi",)))
    if enabled == {"none"}:
        enabled = set()
    issues = []

    ks = np.asarray([d["k"] for d in snapshots], dtype=float)
    if "massmoi" in enabled and float(env_cfg.k_max) - float(env_cfg.k_min) > 1e-12:
        if np.allclose(ks, 1.0, atol=1e-14, rtol=0.0):
            issues.append("massmoi requested but k stayed exactly 1")

    forces = np.asarray([d["external_force"] for d in snapshots], dtype=float)
    if "force" in enabled and float(env_cfg.external_force_max) > 0.0:
        if np.allclose(forces, 0.0, atol=1e-14, rtol=0.0):
            issues.append("force requested but external force stayed zero")

    for disturbance, key, lo_name, hi_name in (
        ("motor_coeff", "motor_coeff_scale", "motor_coeff_min", "motor_coeff_max"),
        ("moment_coeff", "moment_coeff_scale", "moment_coeff_min", "moment_coeff_max"),
        ("arm_length", "arm_length_scale", "arm_length_min", "arm_length_max"),
    ):
        if disturbance not in enabled:
            continue
        lo, hi = float(getattr(env_cfg, lo_name)), float(getattr(env_cfg, hi_name))
        vals = np.asarray([d[key] for d in snapshots], dtype=float)
        if hi - lo > 1e-12 and np.allclose(vals, 1.0, atol=1e-14, rtol=0.0):
            issues.append(f"{disturbance} requested but true mixer stayed nominal")

    if issues:
        raise RuntimeError(f"preflight failed for {stage_name!r}: {'; '.join(issues)}")

    d = snapshots[0]
    fmt = lambda a: np.array2string(np.asarray(a), precision=3, suppress_small=True)
    print(
        f"[preflight] {stage_name}: k={d['k']:.3f}  force={fmt(d['external_force'])}  "
        f"motor={fmt(d['motor_coeff_scale'])}  moment={fmt(d['moment_coeff_scale'])}  "
        f"arm={fmt(d['arm_length_scale'])}"
    )


def _rms_norm(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return 0.0
    if x.ndim == 1:
        return float(np.sqrt(np.mean(x * x)))
    return float(np.sqrt(np.mean(np.sum(x * x, axis=1))))


def _high_frequency_ratio(signal, dt: float, cutoff_hz: float):
    x = np.asarray(signal, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if len(x) < 4:
        return 0.0
    x = x - np.mean(x, axis=0, keepdims=True)
    spec = np.fft.rfft(x, axis=0)
    power = np.abs(spec) ** 2
    freq = np.fft.rfftfreq(len(x), d=dt)
    valid = freq > 0
    total = float(np.sum(power[valid]))
    if total <= EPS:
        return 0.0
    return float(np.sum(power[freq >= cutoff_hz]) / total)


def _termination_reason(env: ResidualTwinEnv):
    sn, st = env.dyn_nom.state(), env.dyn_true.state()
    reasons = []
    if np.linalg.norm(sn["x"] - st["x"]) > env.cfg.term_pos_error:
        reasons.append("model_ref_position")
    if st["R"][2, 2] < env._tilt_cos_thresh:
        reasons.append("tilt")
    return "+".join(reasons) if reasons else "other"


def _rollout(env_cfg, seed: int, agent, hf_cutoff_hz: float, transient_ignore_s: float):
    env = ResidualTwinEnv(copy.deepcopy(env_cfg), seed=int(seed))
    if env.residual_interface != "force_vector":
        raise ValueError("force-vector evaluator received a non-force-vector EnvConfig")
    obs = env.reset()
    disturbance = _disturbance_snapshot(env)

    time = [0.0]
    control_time = []
    desired_x, true_x, true_v, true_R, true_omega = [], [], [], [], []
    action, delta_A = [], []
    A_base, A_cmd = [], []
    u_cmd, u_total = [], []
    motor_sat, saturation = [], []
    state_reward, full_reward = [], []

    def record_state():
        d = env.traj.desired(env.t)
        st = env.dyn_true.state()
        desired_x.append(np.asarray(d["x"], dtype=float))
        true_x.append(st["x"])
        true_v.append(st["v"])
        true_R.append(st["R"])
        true_omega.append(st["omega"])

    record_state()
    terminated = truncated = False
    term_reason = ""
    horizon = int(env_cfg.episode_steps)

    for _ in range(horizon):
        a = np.zeros(env.action_dim, dtype=np.float32) if agent is None else np.asarray(
            agent.act(obs, deterministic=True), dtype=np.float32
        )
        t0 = env.t
        obs, reward, term, trunc, info = env.step(a)

        control_time.append(t0)
        action.append(a.copy())
        delta_A.append(_vec(info.get("residual_force_vector", info.get("residual")), 3, fill=0.0))
        A_base.append(_vec(info.get("A_base"), 3, fill=np.nan))
        A_cmd.append(_vec(info.get("A_cmd"), 3, fill=np.nan))
        u_cmd.append(_vec(info.get("u_cmd"), 4))
        u_total.append(_vec(info.get("u_total"), 4))
        motor_sat.append(_vec(info.get("motor_sat"), 4))
        saturation.append(float(bool(info.get("actuator_saturated", False))))
        state_reward.append(float(info.get("reward_state", 0.0)))
        full_reward.append(float(reward))
        disturbance = _disturbance_snapshot(env, info)

        time.append(env.t)
        record_state()
        if term or trunc:
            terminated, truncated = bool(term), bool(trunc)
            if terminated:
                term_reason = _termination_reason(env)
            break

    arr = lambda x: np.asarray(x, dtype=float)
    tr = {
        "time": arr(time),
        "control_time": arr(control_time),
        "desired_x": arr(desired_x),
        "true_x": arr(true_x),
        "true_v": arr(true_v),
        "true_R": arr(true_R),
        "true_omega": arr(true_omega),
        "action": arr(action),
        "delta_A": arr(delta_A),
        "A_base": arr(A_base),
        "A_cmd": arr(A_cmd),
        "u_cmd": arr(u_cmd),
        "u_total": arr(u_total),
        "motor_sat": arr(motor_sat),
        "saturation": arr(saturation),
        "state_reward": arr(state_reward),
        "full_reward": arr(full_reward),
        "disturbance": disturbance,
        "terminated": terminated,
        "truncated": truncated,
        "termination_reason": term_reason,
    }

    n = len(tr["full_reward"])
    post = slice(1, n + 1)
    p_err = tr["true_x"][post] - tr["desired_x"][post]
    desired_v = np.asarray([env.traj.desired(float(t))["v"] for t in tr["time"][1:n + 1]])
    v_err = tr["true_v"][post] - desired_v

    steady = tr["control_time"] >= float(transient_ignore_s)
    if not np.any(steady):
        steady = np.ones(n, dtype=bool)
    u_norm = tr["u_total"][steady] / WRENCH_SCALE[None, :]
    a_steady = tr["action"][steady]
    dA_steady = tr["delta_A"][steady]
    du = np.diff(u_norm, axis=0) if len(u_norm) > 1 else np.empty((0, 4))
    da = np.diff(a_steady, axis=0) if len(a_steady) > 1 else np.empty((0, 3))
    ddA = np.diff(dA_steady, axis=0) if len(dA_steady) > 1 else np.empty((0, 3))

    tilt_deg = np.degrees(np.arccos(np.clip(tr["true_R"][post, 2, 2], -1.0, 1.0))) if n else np.zeros(0)
    motor_util = tr["motor_sat"] / MAX_MOTOR_THRUST if n else np.empty((0, 4))

    metrics = {
        "episode_length": int(n),
        "completion_fraction": float(n / max(horizon, 1)),
        "terminated": float(terminated),
        "truncated": float(truncated),
        "termination_reason": term_reason,
        "return": float(np.sum(tr["full_reward"])),
        "tracking_score": float(np.sum(tr["state_reward"]) / max(horizon, 1)),
        "true_des_pos_rmse": _rms_norm(p_err),
        "true_des_vel_rmse": _rms_norm(v_err),
        "true_des_pos_max": float(np.max(np.linalg.norm(p_err, axis=1))) if n else 0.0,
        "actuator_sat_fraction": float(np.mean(tr["saturation"])) if n else 0.0,
        "motor_utilization_mean": float(np.nanmean(motor_util)) if motor_util.size else 0.0,
        "motor_utilization_peak": float(np.nanmax(motor_util)) if motor_util.size else 0.0,
        "max_tilt_deg": float(np.max(tilt_deg)) if len(tilt_deg) else 0.0,
        "omega_rms": _rms_norm(tr["true_omega"][post]),
        "normalized_action_rms": _rms_norm(a_steady),
        "normalized_action_peak": float(np.max(np.abs(a_steady))) if a_steady.size else 0.0,
        "normalized_action_roughness": _rms_norm(da),
        "action_hf_ratio": _high_frequency_ratio(a_steady, env_cfg.dt, hf_cutoff_hz),
        "delta_A_rms_N": _rms_norm(dA_steady),
        "delta_A_peak_N": float(np.max(np.linalg.norm(dA_steady, axis=1))) if len(dA_steady) else 0.0,
        "delta_A_roughness_N": _rms_norm(ddA),
        "delta_A_hf_ratio": _high_frequency_ratio(dA_steady, env_cfg.dt, hf_cutoff_hz),
        "applied_wrench_roughness": _rms_norm(du),
        "applied_wrench_hf_ratio": _high_frequency_ratio(u_norm, env_cfg.dt, hf_cutoff_hz),
    }
    for j, name in enumerate(("x", "y", "z")):
        vals = tr["delta_A"][:, j]
        metrics[f"delta_A_{name}_mean_N"] = float(np.mean(vals)) if len(vals) else 0.0
        metrics[f"delta_A_{name}_rms_N"] = float(np.sqrt(np.mean(vals ** 2))) if len(vals) else 0.0
    return metrics, tr


def _disturbance_columns(d):
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


def _paired_row(stage, case, seed, disturbance, baseline, residual):
    row = {"stage": stage, "case": int(case), "seed": int(seed), **_disturbance_columns(disturbance)}
    for key, bval in baseline.items():
        if not isinstance(bval, (int, float, np.number)):
            continue
        b, r = float(bval), float(residual[key])
        row[f"baseline_{key}"] = b
        row[f"residual_{key}"] = r
        row[f"delta_{key}"] = r - b
    b, r = baseline["true_des_pos_rmse"], residual["true_des_pos_rmse"]
    row["position_rmse_improvement_pct"] = 100.0 * (b - r) / max(abs(b), EPS)
    row["residual_pos_rmse_win"] = float(r < b)
    row["baseline_termination_reason"] = baseline.get("termination_reason", "")
    row["residual_termination_reason"] = residual.get("termination_reason", "")
    return row


def _bootstrap_mean_ci(values, rng, n_boot=2000):
    x = np.asarray(values, dtype=float)
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(x[0]), float(x[0])
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = np.mean(x[idx], axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _aggregate(rows, label, seed):
    rng = np.random.default_rng(seed)
    out = {"stage": label, "episodes": len(rows)}
    metrics = (
        "true_des_pos_rmse", "true_des_vel_rmse", "true_des_pos_max", "tracking_score",
        "return", "completion_fraction", "terminated", "actuator_sat_fraction",
        "motor_utilization_mean", "motor_utilization_peak", "max_tilt_deg", "omega_rms",
        "normalized_action_rms", "normalized_action_roughness", "action_hf_ratio",
        "delta_A_rms_N", "delta_A_peak_N", "delta_A_roughness_N", "delta_A_hf_ratio",
        "applied_wrench_roughness", "applied_wrench_hf_ratio",
    )
    for metric in metrics:
        b = np.asarray([r[f"baseline_{metric}"] for r in rows], dtype=float)
        q = np.asarray([r[f"residual_{metric}"] for r in rows], dtype=float)
        mask = np.isfinite(b) & np.isfinite(q)
        b, q = b[mask], q[mask]
        d = q - b
        if len(b) == 0:
            continue
        out[f"baseline_{metric}_mean"] = float(np.mean(b))
        out[f"baseline_{metric}_std"] = float(np.std(b, ddof=1)) if len(b) > 1 else 0.0
        out[f"residual_{metric}_mean"] = float(np.mean(q))
        out[f"residual_{metric}_std"] = float(np.std(q, ddof=1)) if len(q) > 1 else 0.0
        out[f"paired_delta_{metric}_mean"] = float(np.mean(d))
        lo, hi = _bootstrap_mean_ci(d, rng)
        out[f"paired_delta_{metric}_ci95_low"] = lo
        out[f"paired_delta_{metric}_ci95_high"] = hi

    rb = np.asarray([r["baseline_true_des_pos_rmse"] for r in rows], dtype=float)
    rr = np.asarray([r["residual_true_des_pos_rmse"] for r in rows], dtype=float)
    out["mean_position_rmse_improvement_pct"] = 100.0 * (np.mean(rb) - np.mean(rr)) / max(abs(np.mean(rb)), EPS)
    out["position_rmse_win_rate"] = float(np.mean(rr < rb))
    out["baseline_true_des_pos_rmse_p95"] = float(np.percentile(rb, 95))
    out["residual_true_des_pos_rmse_p95"] = float(np.percentile(rr, 95))
    out["baseline_true_des_pos_rmse_worst"] = float(np.max(rb))
    out["residual_true_des_pos_rmse_worst"] = float(np.max(rr))
    return out


def _savefig(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def _plot_aggregate(summaries, output):
    rows = [r for r in summaries if r["stage"] != "ALL"]
    if not rows:
        return
    names = [r["stage"] for r in rows]
    x = np.arange(len(rows))
    width = 0.36

    def bars(metric, ylabel, filename):
        bm = [r[f"baseline_{metric}_mean"] for r in rows]
        bs = [r[f"baseline_{metric}_std"] for r in rows]
        rm = [r[f"residual_{metric}_mean"] for r in rows]
        rs = [r[f"residual_{metric}_std"] for r in rows]
        plt.figure(figsize=(max(8, 1.6 * len(rows)), 4.8))
        plt.bar(x - width / 2, bm, width, yerr=bs, capsize=3, label="baseline")
        plt.bar(x + width / 2, rm, width, yerr=rs, capsize=3, label="baseline + delta_A")
        plt.xticks(x, names, rotation=30, ha="right")
        plt.ylabel(ylabel)
        plt.legend()
        _savefig(output / filename)

    bars("true_des_pos_rmse", "True -> desired position RMSE [m]", "position_rmse_by_stage.png")
    bars("tracking_score", "Fixed-horizon tracking score", "tracking_score_by_stage.png")
    bars("actuator_sat_fraction", "Actuator saturation fraction", "saturation_by_stage.png")
    bars("applied_wrench_roughness", "Applied-wrench roughness", "wrench_roughness_by_stage.png")
    bars("applied_wrench_hf_ratio", "Applied-wrench HF energy ratio", "wrench_hf_ratio_by_stage.png")


def _plot_paired(rows, output):
    b = np.asarray([r["baseline_true_des_pos_rmse"] for r in rows])
    r = np.asarray([r["residual_true_des_pos_rmse"] for r in rows])
    lim = max(float(np.max(b)), float(np.max(r)), 1e-6)
    plt.figure(figsize=(5.4, 5.2))
    plt.scatter(b, r, alpha=0.65)
    plt.plot([0, lim], [0, lim], "--", linewidth=1)
    plt.xlabel("Baseline RMSE [m]")
    plt.ylabel("Baseline + delta_A RMSE [m]")
    plt.title("Paired position RMSE")
    _savefig(output / "paired_position_rmse_scatter.png")

    imp = np.asarray([r["position_rmse_improvement_pct"] for r in rows])
    plt.figure(figsize=(6.4, 4.2))
    plt.hist(imp, bins=min(30, max(8, int(np.sqrt(len(imp))))))
    plt.axvline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Position RMSE improvement [%]")
    plt.ylabel("Paired cases")
    _savefig(output / "position_rmse_improvement_distribution.png")


def _plot_case(stage, tag, b, r, output):
    d = output / "representative_cases" / stage.replace("/", "_") / tag
    d.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(b["desired_x"][:, 0], b["desired_x"][:, 1], b["desired_x"][:, 2], label="desired")
    ax.plot(b["true_x"][:, 0], b["true_x"][:, 1], b["true_x"][:, 2], label="baseline")
    ax.plot(r["true_x"][:, 0], r["true_x"][:, 1], r["true_x"][:, 2], label="baseline + delta_A")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.legend()
    _savefig(d / "trajectory_3d.png")

    for j, name in enumerate(("x", "y", "z")):
        plt.figure(figsize=(8, 3.6))
        plt.plot(b["time"], b["desired_x"][:, j], label="desired")
        plt.plot(b["time"], b["true_x"][:, j], label="baseline")
        plt.plot(r["time"], r["true_x"][:, j], label="baseline + delta_A")
        plt.xlabel("time [s]"); plt.ylabel(f"{name} [m]"); plt.legend()
        _savefig(d / f"position_{name}.png")

    be = np.linalg.norm(b["true_x"] - b["desired_x"], axis=1)
    re = np.linalg.norm(r["true_x"] - r["desired_x"], axis=1)
    plt.figure(figsize=(8, 3.8))
    plt.plot(b["time"], be, label="baseline")
    plt.plot(r["time"], re, label="baseline + delta_A")
    plt.xlabel("time [s]"); plt.ylabel("position error norm [m]"); plt.legend()
    _savefig(d / "position_error_norm.png")

    plt.figure(figsize=(8, 4.2))
    for j, label in enumerate(A_LABELS):
        plt.plot(r["control_time"], r["delta_A"][:, j], label=label)
    plt.xlabel("time [s]"); plt.ylabel("delta_A [N]"); plt.legend(ncol=3)
    _savefig(d / "residual_force_vector.png")

    for j, label in enumerate(WRENCH_LABELS):
        plt.figure(figsize=(8, 3.6))
        plt.plot(b["control_time"], b["u_total"][:, j], label="baseline actual")
        plt.plot(r["control_time"], r["u_total"][:, j], label="delta_A actual")
        plt.plot(r["control_time"], r["u_cmd"][:, j], linestyle="--", label="delta_A commanded")
        plt.xlabel("time [s]"); plt.ylabel(label); plt.legend()
        _savefig(d / f"applied_wrench_{j}.png")

    plt.figure(figsize=(8, 4.2))
    for j in range(4):
        plt.plot(r["control_time"], r["motor_sat"][:, j] / MAX_MOTOR_THRUST, label=f"motor {j+1}")
    plt.xlabel("time [s]"); plt.ylabel("motor utilization"); plt.ylim(bottom=0); plt.legend(ncol=2)
    _savefig(d / "motor_utilization_residual.png")


def _make_report(path, checkpoint, summaries, episodes_per_stage, hf_cutoff):
    overall = summaries[-1]
    text = f"""# Force-vector residual SAC paired evaluation\n\nCheckpoint: `{checkpoint}`\n\nEach baseline and residual episode used the same sampled disturbance realization.\nThe learned action is a 3-D force correction `delta_A` inserted before the geometric controller creates desired attitude and moments.\n\n## Overall\n\n- Cases: {overall['episodes']} ({episodes_per_stage} per stage)\n- Baseline position RMSE: {overall['baseline_true_des_pos_rmse_mean']:.6f} +/- {overall['baseline_true_des_pos_rmse_std']:.6f} m\n- Baseline + delta_A position RMSE: {overall['residual_true_des_pos_rmse_mean']:.6f} +/- {overall['residual_true_des_pos_rmse_std']:.6f} m\n- Mean RMSE improvement: {overall['mean_position_rmse_improvement_pct']:.2f}%\n- Residual win rate: {100.0 * overall['position_rmse_win_rate']:.1f}%\n- Residual RMSE p95 / worst: {overall['residual_true_des_pos_rmse_p95']:.6f} / {overall['residual_true_des_pos_rmse_worst']:.6f} m\n- Baseline RMSE p95 / worst: {overall['baseline_true_des_pos_rmse_p95']:.6f} / {overall['baseline_true_des_pos_rmse_worst']:.6f} m\n- Tracking score: baseline {overall['baseline_tracking_score_mean']:.5f}, residual {overall['residual_tracking_score_mean']:.5f}\n- Termination rate: baseline {100.0 * overall['baseline_terminated_mean']:.2f}%, residual {100.0 * overall['residual_terminated_mean']:.2f}%\n- Actuator saturation: baseline {100.0 * overall['baseline_actuator_sat_fraction_mean']:.2f}%, residual {100.0 * overall['residual_actuator_sat_fraction_mean']:.2f}%\n- Applied-wrench roughness: baseline {overall['baseline_applied_wrench_roughness_mean']:.6f}, residual {overall['residual_applied_wrench_roughness_mean']:.6f}\n- Applied-wrench HF ratio (>={hf_cutoff:g} Hz): baseline {overall['baseline_applied_wrench_hf_ratio_mean']:.6f}, residual {overall['residual_applied_wrench_hf_ratio_mean']:.6f}\n- Mean residual force-vector RMS: {overall['residual_delta_A_rms_N_mean']:.6f} N\n\nThe primary effectiveness criterion is true-to-desired position tracking together with stability/control-quality metrics. Nominal-to-desired RMSE is intentionally not used as a main comparison metric.\n"""
    path.write_text(text, encoding="utf-8")


def evaluate(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    cfg = _load_resolved_config(run_dir)
    if str(cfg.env.residual_interface) != "force_vector":
        raise ValueError(
            f"run config says residual_interface={cfg.env.residual_interface!r}; expected 'force_vector'"
        )
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"

    curriculum_path = Path(args.eval_curriculum).expanduser() if args.eval_curriculum else run_dir / "curriculum.toml"
    if curriculum_path.is_file():
        curriculum = load_curriculum(curriculum_path)
        stage_names = [s.name for s in curriculum.stages]
        stage_cfgs = [env_config_for_stage(cfg.env, s) for s in curriculum.stages]
    else:
        stage_names = ["fixed_distribution"]
        stage_cfgs = [copy.deepcopy(cfg.env)]

    if args.stages:
        wanted = {x.strip() for x in args.stages.split(",") if x.strip()}
        idx = [i for i, n in enumerate(stage_names) if n in wanted or str(i + 1) in wanted]
        if not idx:
            raise ValueError(f"--stages matched none of {stage_names}")
        stage_names = [stage_names[i] for i in idx]
        stage_cfgs = [stage_cfgs[i] for i in idx]

    print("Running disturbance preflight...")
    for i, (name, ecfg) in enumerate(zip(stage_names, stage_cfgs)):
        _preflight_stage(name, ecfg, int(args.seed + 10_000_000 + 100 * i))

    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint)
    agent, payload = _load_policy(cfg, stage_cfgs[0], checkpoint, device)

    output = Path(args.output).expanduser().resolve() if args.output else run_dir / "evaluation" / f"baseline_vs_force_vector_{checkpoint.stem}"
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "evaluation_config.json", {
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_training": payload.get("_training", {}),
        "evaluation_curriculum": str(curriculum_path) if curriculum_path.is_file() else None,
        "episodes_per_stage": args.episodes_per_stage,
        "seed": args.seed,
        "device": device,
        "residual_interface": "force_vector",
        "force_vector_limit_N": float(cfg.env.force_vector_limit_N),
        "hf_cutoff_hz": args.hf_cutoff_hz,
        "transient_ignore_s": args.transient_ignore_s,
        "paired_randomization": True,
    })

    pairs, long_rows, reps = [], [], {}
    total = len(stage_names) * args.episodes_per_stage
    done = 0
    for si, (stage, ecfg) in enumerate(zip(stage_names, stage_cfgs)):
        records = []
        for case in range(args.episodes_per_stage):
            seed = int(args.seed + 100_000 * si + case)
            bm, bt = _rollout(ecfg, seed, None, args.hf_cutoff_hz, args.transient_ignore_s)
            rm, rt = _rollout(ecfg, seed, agent, args.hf_cutoff_hz, args.transient_ignore_s)
            _assert_same_disturbance(bt["disturbance"], rt["disturbance"])
            row = _paired_row(stage, case, seed, bt["disturbance"], bm, rm)
            pairs.append(row)
            records.append((row, bt, rt))
            for name, metrics in (("baseline", bm), ("baseline_plus_delta_A", rm)):
                long_rows.append({"stage": stage, "case": case, "seed": seed, "controller": name, **_disturbance_columns(bt["disturbance"]), **metrics})
            done += 1
            if done == 1 or done % max(1, args.progress_every) == 0 or done == total:
                print(f"[{done:>4}/{total}] {stage} case {case:03d}  RMSE base={bm['true_des_pos_rmse']:.5f} m  A-res={rm['true_des_pos_rmse']:.5f} m")

        records.sort(key=lambda z: z[0]["baseline_true_des_pos_rmse"])
        if args.plot_cases_per_stage > 0 and records:
            reps[(stage, "median")] = records[len(records) // 2]
        if args.plot_cases_per_stage > 1 and records:
            reps[(stage, "hardest_baseline")] = records[-1]

    _write_csv(output / "paired_episode_metrics.csv", pairs)
    _write_csv(output / "episode_metrics.csv", long_rows)
    summaries = [_aggregate([r for r in pairs if r["stage"] == s], s, args.seed + i) for i, s in enumerate(stage_names)]
    summaries.append(_aggregate(pairs, "ALL", args.seed + 9999))
    _write_csv(output / "summary_by_stage.csv", summaries)
    _write_json(output / "summary.json", {"by_stage": summaries[:-1], "overall": summaries[-1]})

    _plot_aggregate(summaries, output)
    _plot_paired(pairs, output)
    for (stage, tag), (row, bt, rt) in reps.items():
        case_dir = output / "representative_cases" / stage.replace("/", "_") / tag
        _write_json(case_dir / "case_info.json", row)
        _plot_case(stage, tag, bt, rt, output)

    _make_report(output / "REPORT.md", checkpoint, summaries, args.episodes_per_stage, args.hf_cutoff_hz)

    o = summaries[-1]
    print("\n=== Overall paired force-vector evaluation ===")
    print(
        f"true->desired position RMSE: baseline {o['baseline_true_des_pos_rmse_mean']:.6f} +/- {o['baseline_true_des_pos_rmse_std']:.6f} m  |  "
        f"A-residual {o['residual_true_des_pos_rmse_mean']:.6f} +/- {o['residual_true_des_pos_rmse_std']:.6f} m"
    )
    print(f"mean RMSE improvement: {o['mean_position_rmse_improvement_pct']:.2f}%")
    print(f"A-residual win rate: {100.0 * o['position_rmse_win_rate']:.1f}%")
    print(f"termination rate: baseline {100.0 * o['baseline_terminated_mean']:.2f}%  |  A-residual {100.0 * o['residual_terminated_mean']:.2f}%")
    print(f"actuator saturation: baseline {100.0 * o['baseline_actuator_sat_fraction_mean']:.2f}%  |  A-residual {100.0 * o['residual_actuator_sat_fraction_mean']:.2f}%")
    print(f"report written to: {output}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--episodes_per_stage", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260810)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--eval_curriculum", default=None)
    p.add_argument("--stages", default=None, help="comma-separated stage names or 1-based indices")
    p.add_argument("--output", default=None)
    p.add_argument("--plot_cases_per_stage", type=int, default=1, choices=(0, 1, 2))
    p.add_argument("--hf_cutoff_hz", type=float, default=5.0)
    p.add_argument("--transient_ignore_s", type=float, default=0.5)
    p.add_argument("--progress_every", type=int, default=10)
    args = p.parse_args()
    if args.episodes_per_stage < 1:
        p.error("--episodes_per_stage must be >= 1")
    if args.hf_cutoff_hz <= 0:
        p.error("--hf_cutoff_hz must be > 0")
    if args.transient_ignore_s < 0:
        p.error("--transient_ignore_s must be >= 0")
    return args


if __name__ == "__main__":
    evaluate(parse_args())
