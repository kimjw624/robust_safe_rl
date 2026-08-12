"""Rule out whether legacy direct-wrench residual chatter is merely an aggressive waveform.

This diagnostic complements ``diagnose_policy_chatter_source`` and
``run_omega_input_ruleout``.  It asks two concrete questions:

1. If the trained policy is removed and we inject synthetic residual moments,
   which frequency/amplitude combinations reproduce the oscillatory/saturated
   mode?
2. If we record the policy residual waveform once, then replay modified copies
   of that waveform OPEN LOOP (policy disconnected), is the exact timing/phase
   important or is its spectral content/amplitude alone enough?

Important caveat
----------------
``playback_exact`` is a *determinism/sanity control*, not by itself a causal
proof.  In a deterministic simulation, replaying the exact residual sequence
from the exact same initial condition can reproduce the original trajectory.
The informative comparisons are delayed/scaled/filtered playback, playback on
an otherwise nominal plant, and finite synthetic bursts followed by zero
residual.

Recommended first run::

  PYTHONPATH=src python3 -m scripts.run_residual_waveform_ruleout \
    --run_dir runs_residual/residual_sac_curriculum/trial_002 \
    --evaluation_dir runs_residual/residual_sac_curriculum/trial_002/evaluation/baseline_vs_residual_best \
    --checkpoint best --classes oscillatory,clean --max_cases_per_class 2 \
    --battery core --device cuda

A larger sweep can then be run with ``--battery full``.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from robust_safe_rl.rl.mixer import F_MAX, M_MAX
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
    _preview_base,
    _select_saturation_push_action,
    _verify_sampled_disturbance,
)

EPS = 1e-12
ACTION_DIM = 4
WRENCH_SCALE = np.concatenate(([F_MAX], np.asarray(M_MAX, dtype=float)))
ACTION_LABELS = ("f", "mx", "my", "mz")
MOMENT_LABELS = ("mx", "my", "mz")


@dataclass(frozen=True)
class PlaybackVariant:
    name: str
    sequence: np.ndarray
    transform: dict | None = None


@dataclass(frozen=True)
class SyntheticVariant:
    name: str
    kind: str
    axis: int = 0
    amplitude: float = 0.0
    frequency_hz: float = 0.0


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
        w.writeheader()
        w.writerows(rows)


def _load_cases(path: Path, classes: set[str], max_per_class: int) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: list[dict] = []
    counts: dict[str, int] = {}
    for row in rows:
        cls = row.get("class", row.get("label", ""))
        if cls not in classes:
            continue
        if counts.get(cls, 0) >= max_per_class:
            continue
        out.append(row)
        counts[cls] = counts.get(cls, 0) + 1
    if not out:
        raise ValueError(f"no selected cases found in {path}")
    return out


def _parse_floats(text: str) -> list[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("expected at least one comma-separated number")
    return vals


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
    f = np.fft.rfftfreq(len(x), d=dt)
    valid = f > 0.0
    total = float(np.sum(p[valid]))
    if total <= EPS:
        return 0.0
    return float(np.sum(p[f >= cutoff_hz])) / total


def _delay_sequence(seq: np.ndarray, steps: int) -> np.ndarray:
    """Delay a normalized action sequence by zero-padding, never circularly."""
    x = np.asarray(seq, dtype=float)
    d = max(int(steps), 0)
    out = np.zeros_like(x)
    if d == 0:
        out[:] = x
    elif d < len(x):
        out[d:] = x[:-d]
    return out


def _lowpass_sequence(seq: np.ndarray, beta: float) -> np.ndarray:
    """Causal first-order LPF initialized from the first recorded action."""
    x = np.asarray(seq, dtype=float)
    if len(x) == 0:
        return x.copy()
    b = float(beta)
    if not (0.0 < b <= 1.0):
        raise ValueError("beta must be in (0, 1]")
    out = np.empty_like(x)
    state = x[0].copy()
    out[0] = state
    for i in range(1, len(x)):
        state = (1.0 - b) * state + b * x[i]
        out[i] = state
    return out


def _window_sequence(seq: np.ndarray, dt: float, start_s: float, end_s: float) -> np.ndarray:
    x = np.asarray(seq, dtype=float)
    out = np.zeros_like(x)
    i0 = max(0, int(round(float(start_s) / float(dt))))
    i1 = min(len(x), int(round(float(end_s) / float(dt))))
    if i1 > i0:
        out[i0:i1] = x[i0:i1]
    return out


def _dominant_moment_mode(actions: np.ndarray, dt: float, min_hz: float = 1.0) -> dict:
    """Return the moment axis/frequency carrying the largest non-DC FFT peak."""
    a = np.asarray(actions, dtype=float)
    if len(a) < 4:
        return {"axis": 0, "frequency_hz": 0.0, "rms": 0.0, "peak": 0.0}
    best = (-np.inf, 0, 0.0)
    freqs = np.fft.rfftfreq(len(a), d=float(dt))
    mask = freqs >= float(min_hz)
    for axis in range(3):
        x = a[:, axis + 1] - np.mean(a[:, axis + 1])
        mag = np.abs(np.fft.rfft(x))
        if not np.any(mask):
            continue
        idxs = np.where(mask)[0]
        k = int(idxs[np.argmax(mag[mask])])
        key = float(mag[k])
        if key > best[0]:
            best = (key, axis, float(freqs[k]))
    axis = int(best[1])
    x = a[:, axis + 1]
    return {
        "axis": axis,
        "frequency_hz": float(best[2]),
        "rms": float(np.sqrt(np.mean(x * x))) if len(x) else 0.0,
        "peak": float(np.max(np.abs(x))) if len(x) else 0.0,
    }


def _action_provider_from_sequence(sequence: np.ndarray) -> Callable:
    seq = np.asarray(sequence, dtype=float)
    state = {"i": 0}

    def provider(env, obs, w_base):
        i = state["i"]
        state["i"] += 1
        if i >= len(seq):
            return np.zeros(ACTION_DIM, dtype=float)
        return np.clip(seq[i], -1.0, 1.0)

    return provider


def _synthetic_provider(variant: SyntheticVariant, args) -> Callable:
    def provider(env, obs, w_base):
        t = float(env.t)
        if not (args.burst_start_s <= t < args.burst_end_s):
            return np.zeros(ACTION_DIM, dtype=float)
        if variant.kind == "sat_push":
            return _select_saturation_push_action(
                w_base,
                env.action_scale,
                env.mixer,
                float(variant.amplitude),
            )
        out = np.zeros(ACTION_DIM, dtype=float)
        idx = 1 + int(variant.axis)
        phase = 2.0 * math.pi * float(variant.frequency_hz) * (t - args.burst_start_s)
        if variant.kind == "sine":
            out[idx] = float(variant.amplitude) * math.sin(phase)
        elif variant.kind == "square":
            out[idx] = float(variant.amplitude) * (1.0 if math.sin(phase) >= 0.0 else -1.0)
        elif variant.kind == "constant":
            out[idx] = float(variant.amplitude)
        else:
            raise ValueError(f"unknown synthetic kind {variant.kind!r}")
        return np.clip(out, -1.0, 1.0)

    return provider


def _rollout(
    stage_cfg,
    seed: int,
    case: dict,
    action_provider: Callable,
    args,
    *,
    transform: dict | None = None,
    save_observations: bool = False,
):
    cfg = copy.deepcopy(stage_cfg)
    cfg.residual_interface = "wrench"
    if args.max_steps is not None:
        cfg.episode_steps = min(int(cfg.episode_steps), int(args.max_steps))
    env = ResidualTwinEnv(cfg, seed=int(seed))
    obs = env.reset()
    _verify_sampled_disturbance(case, _disturbance_snapshot(env))
    _apply_disturbance_transform(env, transform or {})

    desired_x = []
    true_x = []
    actions = []
    residual = []
    base = []
    applied = []
    motor_cmd = []
    sat = []
    omega_true = []
    observations = []

    terminated = truncated = False
    for _ in range(int(cfg.episode_steps)):
        desired = env.traj.desired(env.t)
        st = env.dyn_true.state()
        desired_x.append(np.asarray(desired["x"], dtype=float).copy())
        true_x.append(np.asarray(st["x"], dtype=float).copy())
        omega_true.append(np.asarray(st["omega"], dtype=float).copy())
        if save_observations:
            observations.append(np.asarray(obs, dtype=np.float32).copy())

        w_base_preview = _preview_base(env)
        a = np.asarray(action_provider(env, obs, w_base_preview), dtype=float)
        a = np.clip(a, -1.0, 1.0)
        obs, _reward, term, trunc, info = env.step(a)

        actions.append(a.copy())
        residual.append(np.asarray(info["residual"], dtype=float).copy())
        base.append(np.asarray(info.get("u_base", w_base_preview), dtype=float).copy())
        applied.append(np.asarray(info["u_total"], dtype=float).copy())
        motor_cmd.append(np.asarray(info["motor_cmd"], dtype=float).copy())
        sat.append(float(bool(info["actuator_saturated"])))
        terminated, truncated = bool(term), bool(trunc)
        if term or trunc:
            break

    arr = lambda x: np.asarray(x, dtype=float)
    trace = {
        "desired_x": arr(desired_x),
        "true_x": arr(true_x),
        "action": arr(actions),
        "residual": arr(residual),
        "base": arr(base),
        "applied": arr(applied),
        "motor_cmd": arr(motor_cmd),
        "sat": arr(sat),
        "omega_true": arr(omega_true),
        "observations": np.asarray(observations, dtype=np.float32),
    }
    metrics = _metrics(trace, float(cfg.dt), int(cfg.episode_steps), terminated, args)
    return metrics, trace


def _metrics(trace: dict, dt: float, episode_steps: int, terminated: bool, args) -> dict:
    n = len(trace["action"])
    start = min(max(int(round(args.transient_ignore_s / dt)), 0), max(n - 1, 0))
    pos_err = trace["true_x"] - trace["desired_x"]
    action_tail = trace["action"][start:]
    applied_norm = trace["applied"] / WRENCH_SCALE[None, :]
    applied_tail = applied_norm[start:]
    omega_tail = trace["omega_true"][start:]
    d_action = np.diff(action_tail, axis=0) if len(action_tail) > 1 else np.empty((0, 4))
    d_applied = np.diff(applied_tail, axis=0) if len(applied_tail) > 1 else np.empty((0, 4))
    d_omega = np.diff(omega_tail, axis=0) if len(omega_tail) > 1 else np.empty((0, 3))

    post0 = max(0, int(round((args.burst_end_s + args.post_burst_delay_s) / dt)))
    post1 = min(n, int(round((args.burst_end_s + args.post_burst_window_s) / dt)))
    if post1 - post0 >= 4:
        post_applied = applied_norm[post0:post1]
        post_d = np.diff(post_applied, axis=0)
        post_rough = _rms_norm(post_d)
        post_hf = _hf_ratio(post_applied, dt, args.hf_cutoff_hz)
        post_sat = float(np.mean(trace["sat"][post0:post1]))
        post_omega_rough = _rms_norm(np.diff(trace["omega_true"][post0:post1], axis=0))
    else:
        post_rough = post_hf = post_sat = post_omega_rough = float("nan")

    return {
        "episode_length": n,
        "completion_fraction": n / max(episode_steps, 1),
        "terminated": float(terminated),
        "true_des_pos_rmse_m": _rms_norm(pos_err),
        "sat_fraction": float(np.mean(trace["sat"])) if n else 0.0,
        "action_rms": _rms_norm(action_tail),
        "moment_action_rms": _rms_norm(action_tail[:, 1:4]) if len(action_tail) else 0.0,
        "action_roughness": _rms_norm(d_action),
        "action_hf_ratio": _hf_ratio(action_tail, dt, args.hf_cutoff_hz),
        "wrench_roughness": _rms_norm(d_applied),
        "wrench_hf_ratio": _hf_ratio(applied_tail, dt, args.hf_cutoff_hz),
        "omega_roughness": _rms_norm(d_omega),
        "omega_hf_ratio": _hf_ratio(omega_tail, dt, args.hf_cutoff_hz),
        "post_burst_wrench_roughness": post_rough,
        "post_burst_wrench_hf_ratio": post_hf,
        "post_burst_sat_fraction": post_sat,
        "post_burst_omega_roughness": post_omega_rough,
    }


def _policy_recording(stage_cfg, seed: int, case: dict, agent, args):
    def provider(env, obs, w_base):
        return np.asarray(agent.act(obs, deterministic=True), dtype=float)
    return _rollout(stage_cfg, seed, case, provider, args, save_observations=True)


def _build_playback_variants(recorded: np.ndarray, dt: float, args) -> list[PlaybackVariant]:
    out = [
        PlaybackVariant("playback_exact", recorded.copy()),
        PlaybackVariant("playback_scale_0p5", np.clip(0.5 * recorded, -1.0, 1.0)),
        PlaybackVariant("playback_scale_0p2", np.clip(0.2 * recorded, -1.0, 1.0)),
        PlaybackVariant("playback_lpf_beta_0p2", _lowpass_sequence(recorded, 0.2)),
        PlaybackVariant(
            "playback_burst_then_zero",
            _window_sequence(recorded, dt, args.burst_start_s, args.burst_end_s),
        ),
        PlaybackVariant(
            "playback_exact_nominal_plant",
            recorded.copy(),
            transform={
                "mass_mismatch_scale": 0.0,
                "force_scale": 0.0,
                "all_actuator_mismatch_scale": 0.0,
            },
        ),
    ]
    for delay_s in _parse_floats(args.playback_delays_s):
        steps = max(0, int(round(delay_s / dt)))
        label_ms = int(round(delay_s * 1000.0))
        out.append(PlaybackVariant(f"playback_delay_{label_ms}ms", _delay_sequence(recorded, steps)))
    return out


def _build_synthetic_variants(mode: dict, args) -> list[SyntheticVariant]:
    axis = int(mode["axis"])
    fdom = float(mode["frequency_hz"])
    rms = float(mode["rms"])
    sine_amp = min(1.0, math.sqrt(2.0) * rms)
    square_amp = min(1.0, rms)
    variants = [
        SyntheticVariant("synthetic_sat_push_burst", "sat_push", amplitude=1.0),
        SyntheticVariant("synthetic_constant_match_peak_burst", "constant", axis=axis, amplitude=min(1.0, float(mode["peak"]))),
    ]
    if fdom > 0.0:
        variants += [
            SyntheticVariant("synthetic_match_rms_sine_burst", "sine", axis=axis, amplitude=sine_amp, frequency_hz=fdom),
            SyntheticVariant("synthetic_match_rms_square_burst", "square", axis=axis, amplitude=square_amp, frequency_hz=fdom),
        ]

    if args.battery == "core":
        freqs = _parse_floats(args.core_frequencies_hz)
        amps = _parse_floats(args.core_amplitudes)
    else:
        freqs = _parse_floats(args.full_frequencies_hz)
        amps = _parse_floats(args.full_amplitudes)
    waveforms = [x.strip() for x in args.waveforms.split(",") if x.strip()]
    for waveform in waveforms:
        if waveform not in {"sine", "square"}:
            raise ValueError("waveforms must contain only sine/square")
        for f in freqs:
            for amp in amps:
                ftag = str(float(f)).replace(".", "p")
                atag = str(float(amp)).replace(".", "p")
                variants.append(SyntheticVariant(
                    f"sweep_{waveform}_{ftag}Hz_a{atag}", waveform, axis=axis,
                    amplitude=float(amp), frequency_hz=float(f),
                ))
    return variants


def _plot_case(case_dir: Path, rows: list[dict], traces: dict[str, dict], mode: dict, dt: float):
    selected = [
        "policy_live", "playback_exact", "playback_delay_50ms",
        "playback_lpf_beta_0p2", "playback_exact_nominal_plant",
        "synthetic_match_rms_sine_burst", "synthetic_sat_push_burst",
    ]
    names = [n for n in selected if n in traces]
    if not names:
        return
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    axis = int(mode["axis"])
    for name in names:
        tr = traces[name]
        t = np.arange(len(tr["action"])) * dt
        axes[0].plot(t, tr["action"][:, axis + 1], label=name, linewidth=0.9)
        axes[1].plot(t, np.linalg.norm(tr["omega_true"], axis=1), label=name, linewidth=0.9)
        axes[2].plot(t, tr["sat"], label=name, linewidth=0.9)
    axes[0].set_ylabel(f"residual a_{MOMENT_LABELS[axis]}")
    axes[1].set_ylabel("||omega_true|| [rad/s]")
    axes[2].set_ylabel("saturation")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(case_dir / "waveform_ruleout_traces.png", dpi=160)
    plt.close(fig)

    sweep = [r for r in rows if r.get("family") == "synthetic_sweep"]
    for metric, fname in [
        ("wrench_roughness", "sweep_wrench_roughness.png"),
        ("sat_fraction", "sweep_saturation.png"),
        ("post_burst_wrench_roughness", "sweep_post_burst_roughness.png"),
    ]:
        if not sweep:
            continue
        fig, ax = plt.subplots(figsize=(8, 5))
        for waveform in sorted({r["waveform"] for r in sweep}):
            ss = [r for r in sweep if r["waveform"] == waveform]
            for amp in sorted({float(r["amplitude"]) for r in ss}):
                aa = sorted([r for r in ss if float(r["amplitude"]) == amp], key=lambda r: float(r["frequency_hz"]))
                ax.plot([float(r["frequency_hz"]) for r in aa], [float(r[metric]) for r in aa], marker="o", label=f"{waveform} a={amp:g}")
        ax.set_xlabel("frequency [Hz]")
        ax.set_ylabel(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(case_dir / fname, dpi=160)
        plt.close(fig)


def _write_trace_csv(path: Path, traces: dict[str, dict], names: list[str], dt: float):
    names = [n for n in names if n in traces]
    if not names:
        return
    n = max(len(traces[n]["action"]) for n in names)
    rows = []
    for i in range(n):
        row = {"step": i, "time_s": i * dt}
        for name in names:
            tr = traces[name]
            if i >= len(tr["action"]):
                continue
            for j, lab in enumerate(ACTION_LABELS):
                row[f"{name}_action_{lab}"] = float(tr["action"][i, j])
            row[f"{name}_sat"] = float(tr["sat"][i])
            row[f"{name}_omega_norm"] = float(np.linalg.norm(tr["omega_true"][i]))
        rows.append(row)
    _write_csv(path, rows)


def _report(all_rows: list[dict]) -> str:
    lines = [
        "# Residual waveform rule-out",
        "",
        "`playback_exact` is only a deterministic replay sanity control. The discriminating tests are delayed/scaled/filtered playback, nominal-plant playback, and finite synthetic bursts.",
        "",
    ]
    for case_id in sorted({r["case_id"] for r in all_rows}):
        rows = [r for r in all_rows if r["case_id"] == case_id]
        cls = rows[0]["class"]
        live = next(r for r in rows if r["variant"] == "policy_live")
        lines += [f"## {case_id} ({cls})", ""]
        lines.append(
            f"Live policy: RMSE={live['true_des_pos_rmse_m']:.4f} m, "
            f"wrench roughness={live['wrench_roughness']:.4g}, "
            f"HF={live['wrench_hf_ratio']:.3f}, sat={live['sat_fraction']:.3f}."
        )
        for name in [
            "playback_delay_10ms", "playback_delay_50ms", "playback_delay_100ms",
            "playback_scale_0p2", "playback_lpf_beta_0p2",
            "playback_exact_nominal_plant", "playback_burst_then_zero",
            "synthetic_match_rms_sine_burst", "synthetic_match_rms_square_burst",
            "synthetic_sat_push_burst",
        ]:
            rr = next((r for r in rows if r["variant"] == name), None)
            if rr is None:
                continue
            lines.append(
                f"- `{name}`: wrench rough x{rr['wrench_roughness_ratio_vs_live']:.3f}, "
                f"RMSE x{rr['pos_rmse_ratio_vs_live']:.3f}, sat={rr['sat_fraction']:.3f}, "
                f"post-burst rough={rr['post_burst_wrench_roughness']:.4g}"
            )
        sweep = [r for r in rows if r.get("family") == "synthetic_sweep"]
        if sweep:
            worst = max(sweep, key=lambda r: float(r["wrench_roughness"]))
            lines.append(
                f"- strongest synthetic sweep response: `{worst['variant']}` -> "
                f"rough={worst['wrench_roughness']:.4g}, sat={worst['sat_fraction']:.3f}, "
                f"post-burst rough={worst['post_burst_wrench_roughness']:.4g}."
            )
        lines.append("")

    lines += [
        "## Interpretation",
        "",
        "- If modest playback delays (10-100 ms) strongly suppress the mode while exact playback reproduces it, timing/phase relative to the closed-loop state is important; amplitude/spectrum alone is insufficient.",
        "- If synthetic sine/square bursts at comparable amplitude reproduce the same wrench roughness/saturation, high-frequency residual moment injection itself is sufficient to drive the bad mode.",
        "- If synthetic bursts are violent only while the burst is active and `post_burst_wrench_roughness` collapses afterward, the plant/controller does not have a self-sustained limit cycle; continuing residual excitation is required.",
        "- If `synthetic_sat_push_burst` creates saturation but post-burst motion is smooth, saturation alone is not sufficient to sustain the original chatter.",
        "- If exact recorded playback is benign on the nominalized plant but severe under the original disturbance, the disturbance/controller/actuator operating point is necessary in addition to the residual waveform.",
        "",
    ]
    return "\n".join(lines)


def run(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    evaluation_dir = Path(args.evaluation_dir).expanduser().resolve()
    cases_csv = (
        Path(args.cases_csv).expanduser().resolve()
        if args.cases_csv
        else evaluation_dir / "oscillation_dataset_analysis" / "selected_cases.csv"
    )
    classes = {x.strip() for x in args.classes.split(",") if x.strip()}
    cases = _load_cases(cases_csv, classes, int(args.max_cases_per_class))

    cfg = _load_resolved_config(run_dir)
    cfg.env.residual_interface = "wrench"
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU")
        device = "cpu"
    cfg.device = device
    cfg.env.device = device

    curriculum_path = (
        Path(args.eval_curriculum).expanduser().resolve()
        if args.eval_curriculum
        else evaluation_dir / "evaluation_curriculum.toml"
    )
    if not curriculum_path.is_file():
        curriculum_path = run_dir / "curriculum.toml"
    if not curriculum_path.is_file():
        raise FileNotFoundError("could not find evaluation curriculum or run curriculum")

    _, first_stage_cfg = _find_stage_cfg(cfg.env, curriculum_path, cases[0]["stage"])
    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint)
    agent, _ = _load_policy(cfg, first_stage_cfg, checkpoint, device)

    out_dir = evaluation_dir / "residual_waveform_ruleout"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []

    for case in cases:
        stage = case["stage"]
        seed = int(float(case["seed"]))
        cls = case.get("class", case.get("label", ""))
        case_id = f"{stage}_seed_{seed}"
        _, stage_cfg = _find_stage_cfg(cfg.env, curriculum_path, stage)
        dt = float(stage_cfg.dt)
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        live_metrics, live_trace = _policy_recording(stage_cfg, seed, case, agent, args)
        mode = _dominant_moment_mode(live_trace["action"], dt, args.dominant_min_hz)
        traces: dict[str, dict] = {"policy_live": live_trace}
        rows: list[dict] = []
        live_row = {
            **live_metrics,
            "case_id": case_id, "class": cls, "stage": stage, "seed": seed,
            "variant": "policy_live", "family": "live_policy",
            "dominant_axis": MOMENT_LABELS[int(mode["axis"])],
            "dominant_frequency_hz": mode["frequency_hz"],
            "live_axis_action_rms": mode["rms"],
            "live_axis_action_peak": mode["peak"],
        }
        rows.append(live_row)

        print(
            f"\n[{case_id}] live: axis={MOMENT_LABELS[int(mode['axis'])]} "
            f"fdom={mode['frequency_hz']:.2f}Hz rms={mode['rms']:.3f} "
            f"rough={live_metrics['wrench_roughness']:.4g} sat={live_metrics['sat_fraction']:.3f}"
        )

        # Playback family: policy is disconnected.  The action waveform is fixed.
        for pv in _build_playback_variants(live_trace["action"], dt, args):
            metrics, trace = _rollout(
                stage_cfg, seed, case, _action_provider_from_sequence(pv.sequence), args,
                transform=pv.transform,
            )
            traces[pv.name] = trace
            row = {
                **metrics,
                "case_id": case_id, "class": cls, "stage": stage, "seed": seed,
                "variant": pv.name, "family": "playback",
                "dominant_axis": MOMENT_LABELS[int(mode["axis"])],
                "dominant_frequency_hz": mode["frequency_hz"],
            }
            rows.append(row)
            print(
                f"  {pv.name:30s} rough={metrics['wrench_roughness']:.4g} "
                f"HF={metrics['wrench_hf_ratio']:.3f} sat={metrics['sat_fraction']:.3f} "
                f"RMSE={metrics['true_des_pos_rmse_m']:.4f}"
            )

        # Synthetic family: no policy and no recorded waveform.
        for sv in _build_synthetic_variants(mode, args):
            metrics, trace = _rollout(
                stage_cfg, seed, case, _synthetic_provider(sv, args), args,
            )
            keep_trace = (
                sv.name in {"synthetic_match_rms_sine_burst", "synthetic_match_rms_square_burst", "synthetic_sat_push_burst"}
            )
            if keep_trace:
                traces[sv.name] = trace
            family = "synthetic_sweep" if sv.name.startswith("sweep_") else "synthetic_probe"
            row = {
                **metrics,
                "case_id": case_id, "class": cls, "stage": stage, "seed": seed,
                "variant": sv.name, "family": family,
                "waveform": sv.kind,
                "axis": MOMENT_LABELS[int(sv.axis)] if sv.kind != "sat_push" else "adaptive",
                "amplitude": sv.amplitude,
                "frequency_hz": sv.frequency_hz,
                "dominant_axis": MOMENT_LABELS[int(mode["axis"])],
                "dominant_frequency_hz": mode["frequency_hz"],
            }
            rows.append(row)

        # Ratios vs live policy after all variants exist.
        for row in rows:
            row["wrench_roughness_ratio_vs_live"] = row["wrench_roughness"] / max(live_metrics["wrench_roughness"], EPS)
            row["action_roughness_ratio_vs_live"] = row["action_roughness"] / max(live_metrics["action_roughness"], EPS)
            row["pos_rmse_ratio_vs_live"] = row["true_des_pos_rmse_m"] / max(live_metrics["true_des_pos_rmse_m"], EPS)
            all_rows.append(row)

        _write_csv(case_dir / "metrics.csv", rows)
        _write_csv(case_dir / "synthetic_sweep.csv", [r for r in rows if r["family"] == "synthetic_sweep"])
        np.save(case_dir / "recorded_policy_actions.npy", live_trace["action"])
        _write_trace_csv(
            case_dir / "selected_traces.csv", traces,
            [
                "policy_live", "playback_exact", "playback_delay_10ms", "playback_delay_50ms",
                "playback_delay_100ms", "playback_lpf_beta_0p2", "playback_exact_nominal_plant",
                "playback_burst_then_zero", "synthetic_match_rms_sine_burst",
                "synthetic_match_rms_square_burst", "synthetic_sat_push_burst",
            ],
            dt,
        )
        _plot_case(case_dir, rows, traces, mode, dt)
        (case_dir / "dominant_mode.json").write_text(json.dumps(mode, indent=2), encoding="utf-8")

    _write_csv(out_dir / "all_results.csv", all_rows)
    _write_csv(out_dir / "all_synthetic_sweep.csv", [r for r in all_rows if r.get("family") == "synthetic_sweep"])
    (out_dir / "REPORT.md").write_text(_report(all_rows), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "cases_csv": str(cases_csv),
        "battery": args.battery,
        "classes": sorted(classes),
        "max_cases_per_class": int(args.max_cases_per_class),
        "burst_start_s": float(args.burst_start_s),
        "burst_end_s": float(args.burst_end_s),
        "important_note": "Exact same-seed playback is a sanity control; delayed/scaled/filtered/nominal playback and finite synthetic bursts are causal discriminators.",
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--evaluation_dir", required=True)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--cases_csv", default=None)
    p.add_argument("--classes", default="oscillatory,clean")
    p.add_argument("--max_cases_per_class", type=int, default=2)
    p.add_argument("--eval_curriculum", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--battery", choices=("core", "full"), default="core")
    p.add_argument("--max_steps", type=int, default=None)

    p.add_argument("--burst_start_s", type=float, default=1.0)
    p.add_argument("--burst_end_s", type=float, default=2.0)
    p.add_argument("--post_burst_delay_s", type=float, default=0.2)
    p.add_argument("--post_burst_window_s", type=float, default=2.0)
    p.add_argument("--transient_ignore_s", type=float, default=0.2)
    p.add_argument("--hf_cutoff_hz", type=float, default=10.0)
    p.add_argument("--dominant_min_hz", type=float, default=1.0)

    p.add_argument("--playback_delays_s", default="0.01,0.02,0.05,0.10")
    p.add_argument("--waveforms", default="sine,square")
    p.add_argument("--core_frequencies_hz", default="1,5,10,20,30,40,49")
    p.add_argument("--core_amplitudes", default="0.25,0.5,1.0")
    p.add_argument("--full_frequencies_hz", default="1,2,5,10,15,20,25,30,35,40,45,49")
    p.add_argument("--full_amplitudes", default="0.1,0.25,0.5,0.75,1.0")
    return p


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
