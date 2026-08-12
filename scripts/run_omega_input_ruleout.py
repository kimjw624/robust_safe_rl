"""Live closed-loop rule-out experiment for the legacy direct-wrench SAC policy.

Purpose
-------
The previous offline ablation showed that removing the angular-rate discrepancy
(e_omega) from the actor observation almost eliminates action chatter in the
selected oscillatory episodes.  This script tests whether that relationship is
*causal in the live closed loop*.

Only the observation seen by the frozen SAC actor is modified.  The true plant,
nominal model, geometric controller, mixer, actuator limits, disturbance, reward,
and trained policy weights are unchanged.

The default battery separates two hypotheses:

1. High-frequency e_omega is the trigger:
   preserve DC/low-frequency content but low-pass only the actor's e_omega input.
2. Excessive actor gain with respect to e_omega is the trigger:
   scale e_omega amplitude without filtering it.

It also includes a live zero-e_omega ablation as a hard necessity test.

For history observations, the same modification is applied consistently to all
10 e_omega blocks.  For LPF variants, a causal filtered e_omega history is
maintained across timesteps; the geometric controller always receives the raw
state and is never filtered.

Example
-------
PYTHONPATH=src python3 -m scripts.run_omega_input_ruleout \
  --run_dir runs_residual/residual_sac_curriculum/trial_002 \
  --evaluation_dir runs_residual/residual_sac_curriculum/trial_002/evaluation/baseline_vs_residual_best \
  --checkpoint best \
  --classes oscillatory,clean \
  --max_cases_per_class 3 \
  --device cuda
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

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
    _find_stage_cfg,
    _verify_sampled_disturbance,
)
from scripts.diagnose_policy_chatter_source import _history_slices

EPS = 1e-12
ERROR_DIM = 12
ACTION_DIM = 4
OMEGA_LOCAL = slice(9, 12)
WRENCH_SCALE = np.concatenate(([F_MAX], np.asarray(M_MAX, dtype=float)))


@dataclass(frozen=True)
class Variant:
    name: str
    mode: str
    value: float = 1.0


def _default_variants() -> list[Variant]:
    return [
        Variant("raw", "raw", 1.0),
        Variant("omega_lpf_beta_0p5", "lpf", 0.5),
        Variant("omega_lpf_beta_0p2", "lpf", 0.2),
        Variant("omega_lpf_beta_0p1", "lpf", 0.1),
        Variant("omega_lpf_beta_0p05", "lpf", 0.05),
        Variant("omega_gain_0p5", "gain", 0.5),
        Variant("omega_gain_0p2", "gain", 0.2),
        Variant("omega_zero", "gain", 0.0),
    ]


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


def _omega_blocks(obs: np.ndarray, history: int) -> list[np.ndarray]:
    errors, _, dim = _history_slices(history, ACTION_DIM)
    x = np.asarray(obs, dtype=np.float32)
    if len(x) != dim:
        raise ValueError(f"unexpected obs dim {len(x)}; expected {dim}")
    return [x[sl][OMEGA_LOCAL].copy() for sl in errors]


def _set_omega_blocks(obs: np.ndarray, history: int, values: Iterable[np.ndarray]) -> np.ndarray:
    errors, _, dim = _history_slices(history, ACTION_DIM)
    x = np.asarray(obs, dtype=np.float32).copy()
    if len(x) != dim:
        raise ValueError(f"unexpected obs dim {len(x)}; expected {dim}")
    values = list(values)
    if len(values) != len(errors):
        raise ValueError(f"need {len(errors)} omega blocks, got {len(values)}")
    for sl, value in zip(errors, values):
        block = x[sl].copy()
        block[OMEGA_LOCAL] = np.asarray(value, dtype=np.float32)
        x[sl] = block
    return x


class OmegaObservationModifier:
    """Modify only e_omega in a 156-D legacy history observation.

    LPF state is initialized from the first *current* measurement so that the
    experiment does not create an artificial episode-start transient.  The
    zero-padded pre-episode history remains zero.  Thereafter the modifier keeps
    a causal queue of filtered e_omega values and writes that queue into all
    history error blocks.
    """

    def __init__(self, history: int, mode: str, value: float):
        self.history = int(history)
        self.mode = str(mode)
        self.value = float(value)
        self._filtered: deque[np.ndarray] | None = None
        self._state: np.ndarray | None = None

    def reset(self):
        self._filtered = None
        self._state = None

    def transform(self, obs: np.ndarray) -> np.ndarray:
        raw = _omega_blocks(obs, self.history)
        if self.mode == "raw":
            return np.asarray(obs, dtype=np.float32).copy()
        if self.mode == "gain":
            return _set_omega_blocks(obs, self.history, [self.value * x for x in raw])
        if self.mode != "lpf":
            raise ValueError(f"unknown omega observation mode {self.mode!r}")
        beta = self.value
        if not (0.0 < beta <= 1.0):
            raise ValueError("LPF beta must be in (0,1]")

        current = np.asarray(raw[-1], dtype=float)
        if self._filtered is None:
            # Preserve the env's zero padding, but avoid a fake measurement
            # startup transient by initializing the filter state to current.
            self._state = current.copy()
            self._filtered = deque(
                [np.asarray(x, dtype=float).copy() for x in raw[:-1]] + [current.copy()],
                maxlen=self.history,
            )
        else:
            assert self._state is not None
            self._state = (1.0 - beta) * self._state + beta * current
            self._filtered.append(self._state.copy())
        return _set_omega_blocks(obs, self.history, list(self._filtered))


def _current_error_slice(history: int) -> slice:
    errors, _, _ = _history_slices(history, ACTION_DIM)
    return errors[-1]


def _group_channels() -> dict[str, tuple[int, int, int]]:
    return {
        "pos": (0, 1, 2),
        "vel": (3, 4, 5),
        "att": (6, 7, 8),
        "omega": (9, 10, 11),
    }


def _finite_difference_sensitivity(
    agent,
    observations: np.ndarray,
    history: int,
    epsilon: float,
    stride: int,
) -> list[dict]:
    """Local actor sensitivity per *normalized* observation unit.

    This is diagnostic only.  It directly answers whether a low reward weight
    also resulted in low policy sensitivity (it need not).
    """
    error_slices, _, _ = _history_slices(history, ACTION_DIM)
    current = error_slices[-1]
    rows = []
    for k in range(0, len(observations), max(int(stride), 1)):
        obs = np.asarray(observations[k], dtype=np.float32)
        for scope in ("current", "all_history"):
            for group, channels in _group_channels().items():
                total_sq = 0.0
                moment_sq = 0.0
                for channel in channels:
                    plus = obs.copy(); minus = obs.copy()
                    target_slices = [current] if scope == "current" else error_slices
                    for sl in target_slices:
                        plus[sl.start + channel] += epsilon
                        minus[sl.start + channel] -= epsilon
                    ap = np.asarray(agent.act(plus, deterministic=True), dtype=float)
                    am = np.asarray(agent.act(minus, deterministic=True), dtype=float)
                    deriv = (ap - am) / (2.0 * epsilon)
                    total_sq += float(np.dot(deriv, deriv))
                    moment_sq += float(np.dot(deriv[1:4], deriv[1:4]))
                rows.append({
                    "step": k,
                    "scope": scope,
                    "group": group,
                    "action_sensitivity": float(np.sqrt(total_sq)),
                    "moment_action_sensitivity": float(np.sqrt(moment_sq)),
                })
    return rows


def _rollout(stage_cfg, seed: int, case: dict, agent, variant: Variant, args):
    env_cfg = copy.deepcopy(stage_cfg)
    env_cfg.residual_interface = "wrench"
    if args.max_steps is not None:
        env_cfg.episode_steps = min(int(env_cfg.episode_steps), int(args.max_steps))

    env = ResidualTwinEnv(env_cfg, seed=int(seed))
    obs = env.reset()
    _verify_sampled_disturbance(case, _disturbance_snapshot(env))
    history = int(env_cfg.history)
    modifier = OmegaObservationModifier(history, variant.mode, variant.value)

    desired_x = []
    true_x = []
    actions = []
    applied = []
    sat = []
    raw_omega = []
    actor_omega = []
    raw_observations = []

    terminated = truncated = False
    for _ in range(int(env_cfg.episode_steps)):
        d = env.traj.desired(env.t)
        st = env.dyn_true.state()
        desired_x.append(np.asarray(d["x"], dtype=float).copy())
        true_x.append(np.asarray(st["x"], dtype=float).copy())

        actor_obs = modifier.transform(obs)
        raw_observations.append(np.asarray(obs, dtype=np.float32).copy())
        raw_omega.append(_omega_blocks(obs, history)[-1])
        actor_omega.append(_omega_blocks(actor_obs, history)[-1])
        a = np.asarray(agent.act(actor_obs, deterministic=True), dtype=float)
        obs, _reward, term, trunc, info = env.step(a)

        actions.append(a.copy())
        applied.append(np.asarray(info["u_total"], dtype=float).copy())
        sat.append(float(bool(info["actuator_saturated"])))
        terminated, truncated = bool(term), bool(trunc)
        if term or trunc:
            break

    arr = lambda x: np.asarray(x, dtype=float)
    trace = {
        "desired_x": arr(desired_x),
        "true_x": arr(true_x),
        "action": arr(actions),
        "applied": arr(applied),
        "sat": arr(sat),
        "raw_omega_obs": arr(raw_omega),
        "actor_omega_obs": arr(actor_omega),
        "raw_observations": np.asarray(raw_observations, dtype=np.float32),
    }
    n = len(trace["action"])
    dt = float(env_cfg.dt)
    start = min(max(int(round(args.transient_ignore_s / dt)), 0), max(n - 1, 0))
    action_tail = trace["action"][start:]
    applied_norm = trace["applied"] / WRENCH_SCALE[None, :]
    applied_tail = applied_norm[start:]
    pos_err = trace["true_x"] - trace["desired_x"]

    d_action = np.diff(action_tail, axis=0) if len(action_tail) > 1 else np.empty((0, ACTION_DIM))
    d_applied = np.diff(applied_tail, axis=0) if len(applied_tail) > 1 else np.empty((0, ACTION_DIM))
    d_raw_omega = np.diff(trace["raw_omega_obs"][start:], axis=0) if n - start > 1 else np.empty((0, 3))
    d_actor_omega = np.diff(trace["actor_omega_obs"][start:], axis=0) if n - start > 1 else np.empty((0, 3))

    metrics = {
        "variant": variant.name,
        "omega_mode": variant.mode,
        "omega_value": variant.value,
        "episode_length": n,
        "completion_fraction": n / max(int(env_cfg.episode_steps), 1),
        "terminated": float(terminated),
        "true_des_pos_rmse_m": _rms_norm(pos_err),
        "sat_fraction": float(np.mean(trace["sat"])) if n else 0.0,
        "action_roughness": _rms_norm(d_action),
        "action_hf_ratio": _hf_ratio(action_tail, dt, args.hf_cutoff_hz),
        "moment_action_rms": _rms_norm(action_tail[:, 1:4]) if len(action_tail) else 0.0,
        "wrench_roughness": _rms_norm(d_applied),
        "wrench_hf_ratio": _hf_ratio(applied_tail, dt, args.hf_cutoff_hz),
        "raw_omega_obs_roughness": _rms_norm(d_raw_omega),
        "actor_omega_obs_roughness": _rms_norm(d_actor_omega),
        "raw_omega_obs_hf_ratio": _hf_ratio(trace["raw_omega_obs"][start:], dt, args.hf_cutoff_hz),
        "actor_omega_obs_hf_ratio": _hf_ratio(trace["actor_omega_obs"][start:], dt, args.hf_cutoff_hz),
    }
    return metrics, trace


def _plot_case(case_dir: Path, traces: dict[str, dict], dt: float):
    names = [name for name in traces if name in {
        "raw", "omega_lpf_beta_0p2", "omega_lpf_beta_0p1", "omega_gain_0p2", "omega_zero"
    }]
    if not names:
        names = list(traces)[:5]

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=False)
    for name in names:
        tr = traces[name]
        t = np.arange(len(tr["action"])) * dt
        axes[0].plot(t, np.linalg.norm(tr["action"][:, 1:4], axis=1), label=name, linewidth=1.0)
        axes[1].plot(t, np.linalg.norm(tr["raw_omega_obs"], axis=1), label=name, linewidth=1.0)
        axes[2].plot(t, np.linalg.norm(tr["actor_omega_obs"], axis=1), label=name, linewidth=1.0)
    axes[0].set_ylabel("||moment action||")
    axes[1].set_ylabel("||raw e_omega obs||")
    axes[2].set_ylabel("||actor e_omega obs||")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(case_dir / "omega_input_ruleout.png", dpi=160)
    plt.close(fig)


def _aggregate(rows: list[dict]) -> list[dict]:
    out = []
    classes = sorted({r["class"] for r in rows})
    variants = sorted({r["variant"] for r in rows})
    for cls in classes:
        for variant in variants:
            ss = [r for r in rows if r["class"] == cls and r["variant"] == variant]
            if not ss:
                continue
            out.append({
                "class": cls,
                "variant": variant,
                "n": len(ss),
                "pos_rmse_mean_m": float(np.mean([r["true_des_pos_rmse_m"] for r in ss])),
                "action_roughness_mean": float(np.mean([r["action_roughness"] for r in ss])),
                "wrench_roughness_mean": float(np.mean([r["wrench_roughness"] for r in ss])),
                "wrench_hf_ratio_mean": float(np.mean([r["wrench_hf_ratio"] for r in ss])),
                "sat_fraction_mean": float(np.mean([r["sat_fraction"] for r in ss])),
                "action_roughness_ratio_vs_raw_mean": float(np.mean([r["action_roughness_ratio_vs_raw"] for r in ss])),
                "wrench_roughness_ratio_vs_raw_mean": float(np.mean([r["wrench_roughness_ratio_vs_raw"] for r in ss])),
                "pos_rmse_ratio_vs_raw_mean": float(np.mean([r["pos_rmse_ratio_vs_raw"] for r in ss])),
            })
    return out


def _sensitivity_aggregate(rows: list[dict]) -> list[dict]:
    out = []
    for cls in sorted({r["class"] for r in rows}):
        for scope in ("current", "all_history"):
            for group in _group_channels():
                ss = [r for r in rows if r["class"] == cls and r["scope"] == scope and r["group"] == group]
                if not ss:
                    continue
                out.append({
                    "class": cls,
                    "scope": scope,
                    "group": group,
                    "n_samples": len(ss),
                    "action_sensitivity_mean": float(np.mean([r["action_sensitivity"] for r in ss])),
                    "moment_action_sensitivity_mean": float(np.mean([r["moment_action_sensitivity"] for r in ss])),
                })
    return out


def _report(aggregate: list[dict], sensitivity: list[dict]) -> str:
    lines = [
        "# Live e_omega actor-input rule-out",
        "",
        "Only the SAC actor observation is modified. The geometric controller and plant receive raw state.",
        "",
    ]
    for cls in sorted({r["class"] for r in aggregate}):
        lines += [f"## {cls}", ""]
        ss = [r for r in aggregate if r["class"] == cls]
        ss.sort(key=lambda r: r["action_roughness_ratio_vs_raw_mean"])
        for r in ss:
            lines.append(
                f"- `{r['variant']}`: action rough x{r['action_roughness_ratio_vs_raw_mean']:.3f}, "
                f"wrench rough x{r['wrench_roughness_ratio_vs_raw_mean']:.3f}, "
                f"position RMSE x{r['pos_rmse_ratio_vs_raw_mean']:.3f}, sat={r['sat_fraction_mean']:.3f}"
            )
        lines.append("")

    lines += ["## Interpretation guide", ""]
    lines += [
        "- LPF helps strongly but gain scaling does not: high-frequency e_omega content is specifically causal.",
        "- Gain scaling and LPF both help: excessive closed-loop policy gain through e_omega is more likely than frequency alone.",
        "- Only omega_zero helps: e_omega is necessary, but the relationship is nonlinear / threshold-like.",
        "- None of the live variants help despite the offline ablation: e_omega was a marker of another closed-loop mechanism rather than the cause.",
        "- A useful intervention must also keep position RMSE and termination acceptable; reducing chatter by making the policy blind is not automatically a controller fix.",
        "",
        "## Local actor sensitivity (per normalized observation unit)",
        "",
    ]
    for cls in sorted({r["class"] for r in sensitivity}):
        lines.append(f"### {cls}")
        for scope in ("current", "all_history"):
            rows = [r for r in sensitivity if r["class"] == cls and r["scope"] == scope]
            rows.sort(key=lambda r: -r["moment_action_sensitivity_mean"])
            if rows:
                ordered = ", ".join(f"{r['group']}={r['moment_action_sensitivity_mean']:.3g}" for r in rows)
                lines.append(f"- {scope}: {ordered}")
        lines.append("")
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
    if cfg.env.obs_mode != "history":
        raise ValueError("this diagnostic requires obs_mode='history'")

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
    variants = _default_variants()

    out_dir = evaluation_dir / "omega_input_ruleout"
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    sens_rows: list[dict] = []

    for case in cases:
        stage = case["stage"]
        seed = int(float(case["seed"]))
        cls = case.get("class", case.get("label", ""))
        case_id = f"{stage}_seed_{seed}"
        _, stage_cfg = _find_stage_cfg(cfg.env, curriculum_path, stage)
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        case_rows = []
        traces: dict[str, dict] = {}
        raw_trace = None
        raw_metrics = None
        for variant in variants:
            metrics, trace = _rollout(stage_cfg, seed, case, agent, variant, args)
            traces[variant.name] = trace
            metrics.update({"case_id": case_id, "class": cls, "stage": stage, "seed": seed})
            if variant.name == "raw":
                raw_metrics = metrics.copy()
                raw_trace = trace
            case_rows.append(metrics)
            print(
                f"[{case_id}] {variant.name:22s} "
                f"RMSE={metrics['true_des_pos_rmse_m']:.4f} "
                f"a_rough={metrics['action_roughness']:.4f} "
                f"w_rough={metrics['wrench_roughness']:.4f} "
                f"HF={metrics['wrench_hf_ratio']:.3f} sat={metrics['sat_fraction']:.3f}"
            )

        assert raw_metrics is not None and raw_trace is not None
        for m in case_rows:
            m["action_roughness_ratio_vs_raw"] = m["action_roughness"] / max(raw_metrics["action_roughness"], EPS)
            m["wrench_roughness_ratio_vs_raw"] = m["wrench_roughness"] / max(raw_metrics["wrench_roughness"], EPS)
            m["pos_rmse_ratio_vs_raw"] = m["true_des_pos_rmse_m"] / max(raw_metrics["true_des_pos_rmse_m"], EPS)
            all_rows.append(m)

        sensitivity = _finite_difference_sensitivity(
            agent,
            raw_trace["raw_observations"],
            int(stage_cfg.history),
            float(args.sensitivity_epsilon),
            int(args.sensitivity_stride),
        )
        for s in sensitivity:
            s.update({"case_id": case_id, "class": cls, "stage": stage, "seed": seed})
            sens_rows.append(s)

        _write_csv(case_dir / "metrics.csv", case_rows)
        _write_csv(case_dir / "policy_input_sensitivity.csv", sensitivity)
        _plot_case(case_dir, traces, float(stage_cfg.dt))

        # Compact traces for raw plus the most informative interventions.
        trace_rows = []
        keep = ["raw", "omega_lpf_beta_0p2", "omega_lpf_beta_0p1", "omega_gain_0p2", "omega_zero"]
        n = max(len(traces[k]["action"]) for k in keep)
        for i in range(n):
            row = {"step": i, "time_s": i * float(stage_cfg.dt)}
            for name in keep:
                tr = traces[name]
                if i >= len(tr["action"]):
                    continue
                for j, lab in enumerate(("f", "mx", "my", "mz")):
                    row[f"{name}_action_{lab}"] = float(tr["action"][i, j])
                for j, lab in enumerate(("x", "y", "z")):
                    row[f"{name}_raw_eomega_{lab}"] = float(tr["raw_omega_obs"][i, j])
                    row[f"{name}_actor_eomega_{lab}"] = float(tr["actor_omega_obs"][i, j])
            trace_rows.append(row)
        _write_csv(case_dir / "traces.csv", trace_rows)

    aggregate = _aggregate(all_rows)
    sens_agg = _sensitivity_aggregate(sens_rows)
    _write_csv(out_dir / "all_results.csv", all_rows)
    _write_csv(out_dir / "aggregate_by_variant_and_class.csv", aggregate)
    _write_csv(out_dir / "all_policy_input_sensitivity.csv", sens_rows)
    _write_csv(out_dir / "aggregate_policy_input_sensitivity.csv", sens_agg)
    (out_dir / "REPORT.md").write_text(_report(aggregate, sens_agg), encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps({
        "checkpoint": str(checkpoint),
        "cases_csv": str(cases_csv),
        "classes": sorted(classes),
        "max_cases_per_class": int(args.max_cases_per_class),
        "hf_cutoff_hz": float(args.hf_cutoff_hz),
        "sensitivity_epsilon_normalized": float(args.sensitivity_epsilon),
        "sensitivity_stride": int(args.sensitivity_stride),
        "variants": [v.__dict__ for v in variants],
        "important_note": "Only the SAC actor observation e_omega is modified; the geometric controller uses raw state.",
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--evaluation_dir", required=True)
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--cases_csv", default=None)
    p.add_argument("--classes", default="oscillatory,clean")
    p.add_argument("--max_cases_per_class", type=int, default=3)
    p.add_argument("--eval_curriculum", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--transient_ignore_s", type=float, default=0.2)
    p.add_argument("--hf_cutoff_hz", type=float, default=10.0)
    p.add_argument("--sensitivity_epsilon", type=float, default=0.02,
                   help="finite-difference perturbation in normalized observation units")
    p.add_argument("--sensitivity_stride", type=int, default=20,
                   help="evaluate actor input sensitivity every N closed-loop steps")
    return p


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
