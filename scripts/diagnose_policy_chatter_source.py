"""Isolate whether legacy direct-wrench SAC chatter is policy-internal or closed-loop-induced.

This script is intentionally narrower than ``run_residual_ruleout_experiments``.
It answers the specific question:

    "Does the deterministic residual policy itself generate high-frequency
    actions, or does the chatter require the live plant/controller feedback?"

For each selected clean/oscillatory stage+seed, it first re-runs the original
deterministic direct-wrench policy and records the exact 156-D actor observations
and actions.  It then evaluates the same frozen actor OFFLINE, with no plant,
mixer, or SE(3) controller in the loop.

Offline counterfactuals
-----------------------
recorded_obs
    Actor evaluated on the exact recorded observations.  Sanity check; should
    reproduce the recorded deterministic action almost exactly.

zero_action_history
    Keep every recorded 12-D error-history block exactly as observed, but set
    all nine past-action blocks to zero.  If chatter remains, the error-state
    trajectory alone is sufficient to drive the actor chatter.

self_action_history
    Keep the recorded error-history blocks fixed, but replace past-action blocks
    with the actor's own generated actions.  The physical plant is disconnected.
    This tests actor + action-history recurrence on the recorded state trajectory.

self_no_pos / self_no_vel / self_no_att / self_no_omega
    Same as self_action_history, but one error family is zeroed in every history
    frame.  Large roughness reduction identifies actor input channels that are
    necessary for the chatter.

frozen_error_zero_seed / frozen_error_recorded_seed
    Freeze ALL physical error inputs at one snapshot near the highest action
    roughness.  Only the past-action history is allowed to evolve.  If a
    persistent oscillation survives here, the actor + its action-history is
    sufficient by itself; no plant/controller feedback is required.

Interpretation
--------------
The strongest "policy itself" evidence is persistent high-frequency action in
``frozen_error_*``.  If those are smooth while the original closed loop chatters,
the deterministic MLP is not spontaneously oscillating: changing physical-state
inputs produced by the closed loop are necessary.

Example
-------
PYTHONPATH=src python3 -m scripts.diagnose_policy_chatter_source \
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
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from robust_safe_rl.rl.residual_env import ResidualTwinEnv
from scripts.evaluate_residual_report import (
    _load_policy,
    _load_resolved_config,
    _resolve_checkpoint,
)
from scripts.run_residual_ruleout_experiments import (
    _find_stage_cfg,
    _verify_sampled_disturbance,
    _disturbance_snapshot,
)

ERROR_DIM = 12
WRENCH_ACTION_DIM = 4
EPS = 1e-12


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
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _load_cases(path: Path, classes: set[str], max_per_class: int) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
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


def _history_slices(history: int, action_dim: int = WRENCH_ACTION_DIM):
    """Return ordered error/action slices for causal history observations."""
    if history < 1:
        raise ValueError("history must be >= 1")
    errors = []
    actions = []
    p = 0
    for _ in range(history - 1):
        errors.append(slice(p, p + ERROR_DIM))
        p += ERROR_DIM
        actions.append(slice(p, p + action_dim))
        p += action_dim
    errors.append(slice(p, p + ERROR_DIM))
    p += ERROR_DIM
    return errors, actions, p


def _replace_action_history(obs: np.ndarray, history_actions: np.ndarray, history: int) -> np.ndarray:
    out = np.asarray(obs, dtype=np.float32).copy()
    _, action_slices, dim = _history_slices(history, history_actions.shape[1] if history_actions.size else WRENCH_ACTION_DIM)
    if len(out) != dim:
        raise ValueError(f"unexpected observation dimension {len(out)}; expected {dim}")
    if history_actions.shape != (history - 1, WRENCH_ACTION_DIM):
        raise ValueError(
            f"history_actions shape {history_actions.shape}; expected {(history - 1, WRENCH_ACTION_DIM)}"
        )
    for sl, a in zip(action_slices, history_actions):
        out[sl] = a
    return out


def _extract_action_history(obs: np.ndarray, history: int) -> np.ndarray:
    _, action_slices, dim = _history_slices(history)
    x = np.asarray(obs, dtype=np.float32)
    if len(x) != dim:
        raise ValueError(f"unexpected observation dimension {len(x)}; expected {dim}")
    if not action_slices:
        return np.zeros((0, WRENCH_ACTION_DIM), dtype=np.float32)
    return np.stack([x[sl] for sl in action_slices], axis=0)


def _ablate_error_channels(obs: np.ndarray, history: int, channels: Iterable[int]) -> np.ndarray:
    out = np.asarray(obs, dtype=np.float32).copy()
    error_slices, _, dim = _history_slices(history)
    if len(out) != dim:
        raise ValueError(f"unexpected observation dimension {len(out)}; expected {dim}")
    channels = np.asarray(list(channels), dtype=int)
    for sl in error_slices:
        block = out[sl].copy()
        block[channels] = 0.0
        out[sl] = block
    return out


def _freeze_error_blocks(obs: np.ndarray, history: int) -> np.ndarray:
    """Replace all historical error blocks by the current/latest error block."""
    out = np.asarray(obs, dtype=np.float32).copy()
    error_slices, _, dim = _history_slices(history)
    if len(out) != dim:
        raise ValueError(f"unexpected observation dimension {len(out)}; expected {dim}")
    current = out[error_slices[-1]].copy()
    for sl in error_slices:
        out[sl] = current
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
    X = np.fft.rfft(x, axis=0)
    p = np.abs(X) ** 2
    f = np.fft.rfftfreq(len(x), d=dt)
    valid = f > 0.0
    total = float(np.sum(p[valid]))
    if total <= EPS:
        return 0.0
    return float(np.sum(p[f >= cutoff_hz])) / total


def _action_metrics(actions: np.ndarray, dt: float, cutoff_hz: float, warmup_steps: int = 20) -> dict:
    a = np.asarray(actions, dtype=float)
    start = min(max(int(warmup_steps), 1), max(len(a) - 1, 1))
    tail = a[start:]
    da = np.diff(tail, axis=0) if len(tail) > 1 else np.zeros((0, WRENCH_ACTION_DIM))
    return {
        "action_rms": _rms_norm(tail),
        "moment_action_rms": _rms_norm(tail[:, 1:4]) if len(tail) else 0.0,
        "action_peak_abs": float(np.max(np.abs(tail))) if len(tail) else 0.0,
        "action_roughness": _rms_norm(da),
        "action_hf_ratio": _hf_ratio(tail, dt, cutoff_hz),
    }


def _policy_actions(agent, observations: np.ndarray) -> np.ndarray:
    return np.asarray(
        [np.asarray(agent.act(obs, deterministic=True), dtype=float) for obs in observations],
        dtype=float,
    )


def _replay_with_action_history(
    agent,
    observations: np.ndarray,
    history: int,
    mode: str,
    channel_ablation: Iterable[int] = (),
) -> np.ndarray:
    """Actor-only replay. Recorded error blocks stay fixed; action history is counterfactual."""
    generated: list[np.ndarray] = []
    out = []
    for obs0 in observations:
        obs = _ablate_error_channels(obs0, history, channel_ablation) if channel_ablation else np.asarray(obs0, dtype=np.float32).copy()
        if mode == "zero":
            hist = np.zeros((history - 1, WRENCH_ACTION_DIM), dtype=np.float32)
        elif mode == "self":
            recent = generated[-(history - 1):]
            pad = [np.zeros(WRENCH_ACTION_DIM, dtype=float) for _ in range((history - 1) - len(recent))]
            hist = np.asarray(pad + recent, dtype=np.float32)
        else:
            raise ValueError(mode)
        obs = _replace_action_history(obs, hist, history)
        a = np.asarray(agent.act(obs, deterministic=True), dtype=float)
        out.append(a)
        generated.append(a)
    return np.asarray(out, dtype=float)


def _frozen_error_autonomous(
    agent,
    snapshot_obs: np.ndarray,
    history: int,
    steps: int,
    seed_mode: str,
) -> np.ndarray:
    """No plant, no changing errors. Only the actor's past-action history evolves."""
    base = _freeze_error_blocks(snapshot_obs, history)
    if seed_mode == "recorded":
        hist = _extract_action_history(snapshot_obs, history).astype(float)
    elif seed_mode == "zero":
        hist = np.zeros((history - 1, WRENCH_ACTION_DIM), dtype=float)
    else:
        raise ValueError(seed_mode)

    actions = []
    for _ in range(int(steps)):
        obs = _replace_action_history(base, hist.astype(np.float32), history)
        a = np.asarray(agent.act(obs, deterministic=True), dtype=float)
        actions.append(a)
        if history > 1:
            hist = np.vstack([hist[1:], a[None, :]])
    return np.asarray(actions, dtype=float)


def _rough_snapshot_index(actions: np.ndarray, window: int = 20) -> int:
    if len(actions) < 3:
        return 0
    d = np.linalg.norm(np.diff(actions, axis=0), axis=1)
    w = max(2, min(int(window), len(d)))
    kernel = np.ones(w) / w
    roll = np.convolve(d, kernel, mode="valid")
    return int(np.argmax(roll) + w)


def _run_closed_loop(env_cfg, seed: int, case: dict, agent, max_steps: int | None):
    cfg = copy.deepcopy(env_cfg)
    cfg.residual_interface = "wrench"
    if max_steps is not None:
        cfg.episode_steps = min(int(cfg.episode_steps), int(max_steps))
    env = ResidualTwinEnv(cfg, seed=int(seed))
    obs = env.reset()
    _verify_sampled_disturbance(case, _disturbance_snapshot(env))

    observations = []
    actions = []
    sat = []
    for _ in range(int(cfg.episode_steps)):
        observations.append(np.asarray(obs, dtype=np.float32).copy())
        a = np.asarray(agent.act(obs, deterministic=True), dtype=float)
        actions.append(a)
        obs, _, terminated, truncated, info = env.step(a)
        sat.append(float(bool(info.get("actuator_saturated", False))))
        if terminated or truncated:
            break
    return {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=float),
        "sat": np.asarray(sat, dtype=float),
        "dt": float(cfg.dt),
    }


def _case_report(case_id: str, cls: str, metrics: list[dict]) -> str:
    by = {m["variant"]: m for m in metrics}
    original = by["closed_loop_original"]
    frozen = by.get("frozen_error_recorded_seed")
    zero_hist = by.get("zero_action_history")
    self_hist = by.get("self_action_history")

    lines = [f"## {case_id} ({cls})", ""]
    lines.append(
        f"Closed-loop action roughness={original['action_roughness']:.6g}, "
        f"HF={original['action_hf_ratio']:.3f}, sat={original.get('sat_fraction', float('nan')):.3f}."
    )
    if frozen:
        ratio = frozen["action_roughness"] / max(original["action_roughness"], EPS)
        if ratio >= 0.5 and frozen["action_hf_ratio"] >= 0.25:
            msg = "Strong evidence that actor + past-action history can sustain chatter without the plant/controller."
        elif ratio <= 0.2:
            msg = "Actor/history alone is NOT sufficient; live changing state inputs are required."
        else:
            msg = "Actor/history contributes, but does not fully reproduce the closed-loop chatter."
        lines.append(
            f"Frozen-error actor-only roughness ratio={ratio:.3f}, HF={frozen['action_hf_ratio']:.3f}: **{msg}**"
        )
    if zero_hist:
        ratio = zero_hist["action_roughness"] / max(original["action_roughness"], EPS)
        lines.append(
            f"Recorded errors with zero past-action history: roughness ratio={ratio:.3f}. "
            + ("Past-action history is not necessary." if ratio >= 0.7 else
               "Past-action history materially contributes." if ratio <= 0.3 else
               "Past-action history has a moderate effect.")
        )
    if self_hist:
        lines.append(
            f"Recorded errors + self-generated action history: roughness ratio="
            f"{self_hist['action_roughness'] / max(original['action_roughness'], EPS):.3f}."
        )

    ablations = []
    for name in ("self_no_pos", "self_no_vel", "self_no_att", "self_no_omega"):
        if name in by and self_hist:
            ablations.append((name, by[name]["action_roughness"] / max(self_hist["action_roughness"], EPS)))
    if ablations:
        ablations.sort(key=lambda x: x[1])
        name, ratio = ablations[0]
        lines.append(f"Strongest input ablation: `{name}` -> self-history roughness ratio={ratio:.3f}.")
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
        raise ValueError("policy chatter isolation currently requires obs_mode='history'")

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

    first_stage_name, first_stage_cfg = _find_stage_cfg(cfg.env, curriculum_path, cases[0]["stage"])
    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint)
    agent, _ = _load_policy(cfg, first_stage_cfg, checkpoint, device)

    out_dir = evaluation_dir / "policy_chatter_causality"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    report_parts = [
        "# Residual policy chatter causality",
        "",
        "This report separates **actor-output chatter** from the live plant/SE(3)/mixer feedback.",
        "The decisive test is `frozen_error_*`: all physical error inputs are frozen and only past-action history evolves.",
        "",
    ]

    for case in cases:
        stage = case["stage"]
        seed = int(float(case["seed"]))
        cls = case.get("class", case.get("label", ""))
        case_id = f"{stage}_seed_{seed}"
        _, stage_cfg = _find_stage_cfg(cfg.env, curriculum_path, stage)
        trace = _run_closed_loop(stage_cfg, seed, case, agent, args.max_steps)
        obs = trace["observations"]
        a_orig = trace["actions"]
        dt = trace["dt"]
        if len(obs) < 4:
            print(f"[skip] {case_id}: too short")
            continue
        history = int(stage_cfg.history)
        _, _, expected_dim = _history_slices(history)
        if obs.shape[1] != expected_dim:
            raise ValueError(
                f"{case_id}: observation dim {obs.shape[1]} does not match wrench history layout {expected_dim}"
            )

        variants: dict[str, np.ndarray] = {}
        variants["closed_loop_original"] = a_orig
        variants["recorded_obs"] = _policy_actions(agent, obs)
        variants["zero_action_history"] = _replay_with_action_history(agent, obs, history, "zero")
        variants["self_action_history"] = _replay_with_action_history(agent, obs, history, "self")
        variants["self_no_pos"] = _replay_with_action_history(agent, obs, history, "self", range(0, 3))
        variants["self_no_vel"] = _replay_with_action_history(agent, obs, history, "self", range(3, 6))
        variants["self_no_att"] = _replay_with_action_history(agent, obs, history, "self", range(6, 9))
        variants["self_no_omega"] = _replay_with_action_history(agent, obs, history, "self", range(9, 12))

        idx = _rough_snapshot_index(a_orig, args.snapshot_window_steps)
        snapshot = obs[min(idx, len(obs) - 1)]
        frozen_steps = int(args.frozen_steps)
        variants["frozen_error_zero_seed"] = _frozen_error_autonomous(
            agent, snapshot, history, frozen_steps, "zero"
        )
        variants["frozen_error_recorded_seed"] = _frozen_error_autonomous(
            agent, snapshot, history, frozen_steps, "recorded"
        )

        rows = []
        orig = None
        for name, arr in variants.items():
            m = {
                "case_id": case_id,
                "class": cls,
                "stage": stage,
                "seed": seed,
                "variant": name,
                "steps": len(arr),
                **_action_metrics(arr, dt, args.hf_cutoff_hz, args.metric_warmup_steps),
            }
            if name == "closed_loop_original":
                m["sat_fraction"] = float(np.mean(trace["sat"])) if len(trace["sat"]) else 0.0
                orig = m
            rows.append(m)

        if orig is None:
            continue
        for m in rows:
            m["roughness_ratio_vs_closed_loop"] = (
                m["action_roughness"] / max(orig["action_roughness"], EPS)
            )
            m["hf_ratio_delta_vs_closed_loop"] = m["action_hf_ratio"] - orig["action_hf_ratio"]
            all_rows.append(m)

        # Save one compact trace CSV per case.
        case_dir = out_dir / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        n = max(len(x) for x in variants.values())
        trace_rows = []
        for i in range(n):
            row = {"step": i, "time_s": i * dt}
            for name, arr in variants.items():
                if i < len(arr):
                    for j, lab in enumerate(("f", "mx", "my", "mz")):
                        row[f"{name}_{lab}"] = float(arr[i, j])
            trace_rows.append(row)
        _write_csv(case_dir / "actor_action_traces.csv", trace_rows)
        _write_csv(case_dir / "metrics.csv", rows)

        # Plot moment actions; this makes period-2 / near-Nyquist chatter obvious.
        fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=False)
        for ax, channel, lab in zip(axes, (1, 2, 3), ("Mx action", "My action", "Mz action")):
            for name in (
                "closed_loop_original",
                "zero_action_history",
                "self_action_history",
                "frozen_error_recorded_seed",
            ):
                arr = variants[name]
                t = np.arange(len(arr)) * dt
                ax.plot(t, arr[:, channel], label=name, linewidth=1.0)
            ax.set_ylabel(lab)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("time [s]")
        axes[0].legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(case_dir / "policy_chatter_isolation.png", dpi=160)
        plt.close(fig)

        report_parts.append(_case_report(case_id, cls, rows))
        print(
            f"[{case_id}] original rough={orig['action_roughness']:.4g}, "
            f"frozen={next(m for m in rows if m['variant']=='frozen_error_recorded_seed')['action_roughness']:.4g}"
        )

    _write_csv(out_dir / "all_policy_chatter_metrics.csv", all_rows)

    # Aggregate by class + variant.
    agg_rows = []
    for cls in sorted({r["class"] for r in all_rows}):
        for variant in sorted({r["variant"] for r in all_rows}):
            ss = [r for r in all_rows if r["class"] == cls and r["variant"] == variant]
            if not ss:
                continue
            agg_rows.append({
                "class": cls,
                "variant": variant,
                "n": len(ss),
                "action_roughness_mean": float(np.mean([r["action_roughness"] for r in ss])),
                "action_hf_ratio_mean": float(np.mean([r["action_hf_ratio"] for r in ss])),
                "roughness_ratio_vs_closed_loop_mean": float(np.mean([r["roughness_ratio_vs_closed_loop"] for r in ss])),
            })
    _write_csv(out_dir / "aggregate_by_class.csv", agg_rows)

    (out_dir / "REPORT.md").write_text("\n".join(report_parts), encoding="utf-8")
    metadata = {
        "checkpoint": str(checkpoint),
        "cases_csv": str(cases_csv),
        "classes": sorted(classes),
        "max_cases_per_class": int(args.max_cases_per_class),
        "hf_cutoff_hz": float(args.hf_cutoff_hz),
        "frozen_steps": int(args.frozen_steps),
        "snapshot_window_steps": int(args.snapshot_window_steps),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
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
    p.add_argument("--hf_cutoff_hz", type=float, default=10.0)
    p.add_argument("--metric_warmup_steps", type=int, default=20)
    p.add_argument("--frozen_steps", type=int, default=400)
    p.add_argument("--snapshot_window_steps", type=int, default=20)
    return p


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
