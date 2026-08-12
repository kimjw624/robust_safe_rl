"""Final in-distribution check before retraining the direct-wrench residual SAC.

Purpose
-------
This script asks one narrow question on the ORIGINAL six-stage ID curriculum:

    Does low-pass filtering ONLY the residual collective-thrust channel reduce
    body-rate/control oscillation and motor saturation without giving up
    position tracking?

The trained checkpoint is frozen.  Every controller variant is evaluated on the
same episode seed, so mass/MOI, force, and (global) actuator/geometry draws are
paired exactly.  No OOD curriculum is used unless the user explicitly passes a
nonstandard file, and by default the script refuses stage names that do not
match the six training stages.

Variants
--------
* baseline_zero_residual
* live_raw
* live_thrust_lpf_b0p5
* live_thrust_lpf_b0p2
* live_thrust_lpf_b0p1
* live_moment_lpf_b0p2       (channel-specificity control)
* live_all_lpf_b0p2          (all-action smoothing control)

The LPF is applied in normalized SAC action space.  For thrust-only filtering,
moments remain raw at 100 Hz:

    a_f,applied[t] = (1-beta) a_f,applied[t-1] + beta a_f,raw[t]
    a_M,applied[t] = a_M,raw[t]

This is a diagnostic intervention only; it does NOT modify training code.

Recommended final run (300 paired ID cases, 2100 rollouts total)::

  PYTHONPATH=src python3 -m scripts.verify_id_thrust_filter \
    --run_dir runs_residual/residual_sac_curriculum/trial_002 \
    --curriculum configs/residual_sac_curriculum.toml \
    --checkpoint best \
    --episodes_per_stage 50 \
    --device cuda
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from robust_safe_rl.rl.curriculum import env_config_for_stage, load_curriculum
from robust_safe_rl.rl.mixer import F_MAX
from robust_safe_rl.rl.residual_env import ResidualTwinEnv
from scripts.evaluate_residual_report import (
    _disturbance_snapshot,
    _load_policy,
    _load_resolved_config,
    _resolve_checkpoint,
)
from scripts.run_residual_failure_mode_suite import (
    ACTION_DIM,
    ActionShaper,
    LiveVariant,
    _append_pre_state,
    _empty_trace,
    _finalize_trace,
    _hf_ratio,
    _metrics,
    _rms_norm,
)

EPS = 1e-12
EXPECTED_ID_STAGES = (
    "01_massmoi_mild",
    "02_massmoi_full",
    "03_add_force_mild",
    "04_force_full",
    "05_add_actuator_geometry_mild",
    "06_all_full",
)

VARIANTS = (
    LiveVariant("baseline_zero_residual", "raw", 1.0),
    LiveVariant("live_raw", "raw", 1.0),
    LiveVariant("live_thrust_lpf_b0p5", "thrust_lpf", 0.5),
    LiveVariant("live_thrust_lpf_b0p2", "thrust_lpf", 0.2),
    LiveVariant("live_thrust_lpf_b0p1", "thrust_lpf", 0.1),
    LiveVariant("live_moment_lpf_b0p2", "moment_lpf", 0.2),
    LiveVariant("live_all_lpf_b0p2", "all_lpf", 0.2),
)

THRUST_VARIANTS = (
    "live_thrust_lpf_b0p5",
    "live_thrust_lpf_b0p2",
    "live_thrust_lpf_b0p1",
)


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


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    return x


def _validate_id_curriculum(curriculum, allow_nonstandard: bool):
    names = tuple(stage.name for stage in curriculum.stages)
    if not allow_nonstandard and names != EXPECTED_ID_STAGES:
        raise ValueError(
            "This verification is ID-only. Expected exactly the six training stages:\n"
            f"  {EXPECTED_ID_STAGES}\n"
            f"but curriculum contains:\n  {names}\n"
            "Pass the normal configs/residual_sac_curriculum.toml, or use "
            "--allow_nonstandard_curriculum only if this difference is intentional."
        )
    for stage in curriculum.stages:
        if not allow_nonstandard and stage.name.lower().startswith("ood"):
            raise ValueError(f"OOD stage detected in ID verification: {stage.name}")


def _metrics_args(args):
    # _metrics also contains generic post-burst fields; make them inert here.
    return SimpleNamespace(
        transient_ignore_s=float(args.transient_ignore_s),
        hf_cutoff_hz=float(args.hf_cutoff_hz),
        burst_end_s=1e9,
        post_burst_delay_s=0.2,
        post_burst_window_s=1.0,
    )


def _extra_action_metrics(raw_actions: np.ndarray, applied_actions: np.ndarray,
                          trace: dict, dt: float, args) -> dict:
    n = min(len(raw_actions), len(applied_actions), len(trace["u_total"]))
    start = min(max(int(round(float(args.transient_ignore_s) / dt)), 0), max(n - 1, 0))
    raw = np.asarray(raw_actions[:n], dtype=float)[start:]
    applied = np.asarray(applied_actions[:n], dtype=float)[start:]
    residual = np.asarray(trace["residual"][:n], dtype=float)[start:]
    ubase = np.asarray(trace["u_base"][:n], dtype=float)[start:]
    utotal = np.asarray(trace["u_total"][:n], dtype=float)[start:]

    def diff(x):
        return np.diff(x, axis=0) if len(x) > 1 else np.empty((0,) + x.shape[1:])

    def rough_1d(x):
        x = np.asarray(x, dtype=float)
        return float(np.sqrt(np.mean(np.diff(x) ** 2))) if len(x) > 1 else 0.0

    return {
        "raw_action_roughness": _rms_norm(diff(raw)),
        "raw_action_hf_ratio": _hf_ratio(raw, dt, args.hf_cutoff_hz),
        "raw_thrust_action_roughness": rough_1d(raw[:, 0]) if len(raw) else 0.0,
        "raw_thrust_action_hf_ratio": _hf_ratio(raw[:, 0], dt, args.hf_cutoff_hz) if len(raw) else 0.0,
        "raw_moment_action_roughness": _rms_norm(diff(raw[:, 1:4])) if len(raw) else 0.0,
        "applied_thrust_action_roughness": rough_1d(applied[:, 0]) if len(applied) else 0.0,
        "applied_thrust_action_hf_ratio": _hf_ratio(applied[:, 0], dt, args.hf_cutoff_hz) if len(applied) else 0.0,
        "residual_thrust_rms_N": float(np.sqrt(np.mean(residual[:, 0] ** 2))) if len(residual) else 0.0,
        "residual_thrust_roughness_N": rough_1d(residual[:, 0]) if len(residual) else 0.0,
        "residual_thrust_hf_ratio": _hf_ratio(residual[:, 0], dt, args.hf_cutoff_hz) if len(residual) else 0.0,
        "base_collective_thrust_roughness_N": rough_1d(ubase[:, 0]) if len(ubase) else 0.0,
        "applied_collective_thrust_roughness_N": rough_1d(utotal[:, 0]) if len(utotal) else 0.0,
        "applied_collective_thrust_hf_ratio": _hf_ratio(utotal[:, 0], dt, args.hf_cutoff_hz) if len(utotal) else 0.0,
        "mean_applied_collective_thrust_N": float(np.mean(utotal[:, 0])) if len(utotal) else 0.0,
        "collective_thrust_scale_N": float(F_MAX),
    }


def _rollout(stage_cfg, seed: int, agent, variant: LiveVariant, args):
    cfg = copy.deepcopy(stage_cfg)
    cfg.residual_interface = "wrench"
    if args.max_steps is not None:
        cfg.episode_steps = min(int(cfg.episode_steps), int(args.max_steps))

    env = ResidualTwinEnv(cfg, seed=int(seed))
    obs = env.reset()
    disturbance = _disturbance_snapshot(env)
    shaper = ActionShaper(variant.action_mode, variant.value)
    trace = _empty_trace()
    raw_actions: list[np.ndarray] = []
    applied_actions: list[np.ndarray] = []
    terminated = False

    for _ in range(int(cfg.episode_steps)):
        desired = env.traj.desired(env.t)
        _append_pre_state(trace, env, desired)

        if variant.name == "baseline_zero_residual":
            raw = np.zeros(ACTION_DIM, dtype=float)
            action = raw.copy()
        else:
            raw = np.asarray(agent.act(obs, deterministic=True), dtype=float)
            action = shaper.apply(raw)

        obs, _, term, trunc, info = env.step(action)
        raw_actions.append(raw.copy())
        applied_actions.append(action.copy())
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
    metrics = _metrics(
        tr,
        float(cfg.dt),
        int(cfg.episode_steps),
        terminated,
        _metrics_args(args),
    )
    metrics.update(_extra_action_metrics(
        np.asarray(raw_actions, dtype=float),
        np.asarray(applied_actions, dtype=float),
        tr,
        float(cfg.dt),
        args,
    ))
    return metrics, disturbance


def _disturbance_columns(d: dict) -> dict:
    out = {
        "k": float(d["k"]),
        "force_norm_N": float(np.linalg.norm(d["external_force"])),
        "force_x_N": float(d["external_force"][0]),
        "force_y_N": float(d["external_force"][1]),
        "force_z_N": float(d["external_force"][2]),
    }
    for prefix, key in (
        ("motor_coeff", "motor_coeff_scale"),
        ("moment_coeff", "moment_coeff_scale"),
        ("arm_length", "arm_length_scale"),
    ):
        vals = np.asarray(d[key], dtype=float)
        out[f"{prefix}_mean"] = float(np.mean(vals))
        out[f"{prefix}_spread"] = float(np.max(vals) - np.min(vals))
        for i, v in enumerate(vals):
            out[f"{prefix}_{i}"] = float(v)
    return out


def _assert_same_disturbance(a: dict, b: dict):
    if not np.isclose(float(a["k"]), float(b["k"]), rtol=0.0, atol=1e-14):
        raise RuntimeError("paired variants sampled different k")
    for key in ("external_force", "motor_coeff_scale", "moment_coeff_scale", "arm_length_scale"):
        if not np.allclose(np.asarray(a[key]), np.asarray(b[key]), rtol=0.0, atol=1e-14):
            raise RuntimeError(f"paired variants sampled different {key}")


def _case_key(row: dict) -> tuple[str, int]:
    return str(row["stage"]), int(row["case"])


def _paired_ratio(rows: list[dict], variant: str, metric: str, subset: set[tuple[str, int]] | None = None):
    raw = {_case_key(r): r for r in rows if r["variant"] == "live_raw"}
    vv = {_case_key(r): r for r in rows if r["variant"] == variant}
    keys = sorted(set(raw) & set(vv))
    if subset is not None:
        keys = [k for k in keys if k in subset]
    ratios = []
    diffs = []
    for k in keys:
        a = float(raw[k][metric]); b = float(vv[k][metric])
        ratios.append(b / max(abs(a), EPS))
        diffs.append(b - a)
    return np.asarray(ratios, dtype=float), np.asarray(diffs, dtype=float)


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = 2000):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(x))
    if len(x) == 1:
        return mean, mean, mean
    idx = rng.integers(0, len(x), size=(int(n_boot), len(x)))
    means = np.mean(x[idx], axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return mean, float(lo), float(hi)


def _high_oscillation_subset(rows: list[dict], fraction: float) -> set[tuple[str, int]]:
    raw = [r for r in rows if r["variant"] == "live_raw"]
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for r in raw:
        by_stage[str(r["stage"])].append(r)
    selected: set[tuple[str, int]] = set()
    for stage, rr in by_stage.items():
        rr = sorted(rr, key=lambda r: float(r["omega_roughness"]), reverse=True)
        n = max(1, int(np.ceil(float(fraction) * len(rr))))
        selected.update(_case_key(r) for r in rr[:n])
    return selected


def _summaries(rows: list[dict], subset: set[tuple[str, int]] | None,
               subset_name: str, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    out = []
    variants = [v.name for v in VARIANTS]
    for variant in variants:
        rr = [r for r in rows if r["variant"] == variant]
        if subset is not None:
            rr = [r for r in rr if _case_key(r) in subset]
        if not rr:
            continue
        row = {
            "subset": subset_name,
            "variant": variant,
            "n": len(rr),
            "pos_rmse_mean_m": float(np.mean([r["true_des_pos_rmse_m"] for r in rr])),
            "pos_rmse_median_m": float(np.median([r["true_des_pos_rmse_m"] for r in rr])),
            "termination_rate": float(np.mean([r["terminated"] for r in rr])),
            "sat_fraction_mean": float(np.mean([r["sat_fraction"] for r in rr])),
            "omega_roughness_mean": float(np.mean([r["omega_roughness"] for r in rr])),
            "omega_roughness_median": float(np.median([r["omega_roughness"] for r in rr])),
            "wrench_roughness_mean": float(np.mean([r["wrench_roughness"] for r in rr])),
            "residual_thrust_roughness_N_mean": float(np.mean([r["residual_thrust_roughness_N"] for r in rr])),
            "applied_collective_thrust_roughness_N_mean": float(np.mean([r["applied_collective_thrust_roughness_N"] for r in rr])),
            "raw_thrust_action_roughness_mean": float(np.mean([r["raw_thrust_action_roughness"] for r in rr])),
        }
        if variant != "live_raw" and variant != "baseline_zero_residual":
            for metric, short in (
                ("true_des_pos_rmse_m", "rmse"),
                ("omega_roughness", "omegaR"),
                ("wrench_roughness", "wrenchR"),
                ("sat_fraction", "sat"),
            ):
                ratios, diffs = _paired_ratio(rows, variant, metric, subset)
                if len(ratios):
                    row[f"{short}_ratio_mean_vs_raw"] = float(np.mean(ratios))
                    row[f"{short}_ratio_median_vs_raw"] = float(np.median(ratios))
                    dm, dlo, dhi = _bootstrap_mean_ci(diffs, rng)
                    row[f"{short}_delta_mean_vs_raw"] = dm
                    row[f"{short}_delta_ci95_lo"] = dlo
                    row[f"{short}_delta_ci95_hi"] = dhi
        out.append(row)
    return out


def _stage_summaries(rows: list[dict]) -> list[dict]:
    out = []
    for stage in EXPECTED_ID_STAGES:
        rr_stage = [r for r in rows if r["stage"] == stage]
        if not rr_stage:
            continue
        for variant in [v.name for v in VARIANTS]:
            rr = [r for r in rr_stage if r["variant"] == variant]
            if not rr:
                continue
            row = {
                "stage": stage,
                "variant": variant,
                "n": len(rr),
                "pos_rmse_mean_m": float(np.mean([r["true_des_pos_rmse_m"] for r in rr])),
                "termination_rate": float(np.mean([r["terminated"] for r in rr])),
                "sat_fraction_mean": float(np.mean([r["sat_fraction"] for r in rr])),
                "omega_roughness_mean": float(np.mean([r["omega_roughness"] for r in rr])),
                "wrench_roughness_mean": float(np.mean([r["wrench_roughness"] for r in rr])),
                "residual_thrust_roughness_N_mean": float(np.mean([r["residual_thrust_roughness_N"] for r in rr])),
            }
            if variant not in {"live_raw", "baseline_zero_residual"}:
                subset = {_case_key(r) for r in rr_stage}
                for metric, short in (("true_des_pos_rmse_m", "rmse"), ("omega_roughness", "omegaR")):
                    ratios, _ = _paired_ratio(rows, variant, metric, subset)
                    row[f"{short}_ratio_median_vs_raw"] = float(np.median(ratios)) if len(ratios) else float("nan")
            out.append(row)
    return out


def _plot_stage_metric(stage_rows: list[dict], out: Path, metric: str, ylabel: str,
                       variants: tuple[str, ...]):
    stages = [s for s in EXPECTED_ID_STAGES if any(r["stage"] == s for r in stage_rows)]
    if not stages:
        return
    x = np.arange(len(stages), dtype=float)
    width = 0.8 / max(len(variants), 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    for j, variant in enumerate(variants):
        vals = []
        for stage in stages:
            match = [r for r in stage_rows if r["stage"] == stage and r["variant"] == variant]
            vals.append(float(match[0][metric]) if match else np.nan)
        ax.bar(x + (j - (len(variants)-1)/2) * width, vals, width=width, label=variant)
    ax.set_xticks(x)
    ax.set_xticklabels(stages, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def _decision_text(summary_rows: list[dict], high_rows: list[dict]) -> str:
    all_by = {r["variant"]: r for r in summary_rows}
    high_by = {r["variant"]: r for r in high_rows}
    lines = [
        "# ID thrust-filter verification decision",
        "",
        "This is a diagnostic gate before retraining.  A PASS means the intervention",
        "worked on the six-stage ID evaluation with the frozen old checkpoint; it does",
        "not itself prove the retrained policy will retain the benefit.",
        "",
        "## Candidate gate",
        "",
        "For each thrust-only beta, require:",
        "- high-oscillation subset median omega-roughness ratio <= 0.50 vs raw;",
        "- high-oscillation mean saturation does not increase;",
        "- all-ID mean position RMSE ratio <= 1.05 vs raw;",
        "- all-ID termination rate increases by no more than 1 percentage point.",
        "",
    ]
    passing = []
    raw_all = all_by.get("live_raw", {})
    raw_high = high_by.get("live_raw", {})
    for v in THRUST_VARIANTS:
        a = all_by.get(v, {}); h = high_by.get(v, {})
        if not a or not h or not raw_all or not raw_high:
            continue
        omega_ratio = float(h.get("omegaR_ratio_median_vs_raw", np.inf))
        rmse_ratio = float(a.get("rmse_ratio_mean_vs_raw", np.inf))
        sat_delta = float(h.get("sat_fraction_mean", np.inf)) - float(raw_high.get("sat_fraction_mean", 0.0))
        term_delta = float(a.get("termination_rate", np.inf)) - float(raw_all.get("termination_rate", 0.0))
        passed = omega_ratio <= 0.50 and sat_delta <= 0.0 and rmse_ratio <= 1.05 and term_delta <= 0.01
        lines.append(
            f"- `{v}`: {'**PASS**' if passed else '**FAIL**'}; "
            f"high omegaR median ratio={omega_ratio:.3f}, "
            f"high sat delta={sat_delta:+.3f}, all-ID RMSE ratio={rmse_ratio:.3f}, "
            f"termination delta={term_delta:+.3f}."
        )
        if passed:
            passing.append((omega_ratio, abs(rmse_ratio - 1.0), v))
    lines.append("")
    if passing:
        passing.sort()
        best = passing[0][2]
        lines.append(f"Suggested retraining candidate from this gate: **`{best}`**.")
    else:
        lines.append("No thrust-only filter passed the predeclared gate; do not retrain yet.")
    lines.extend([
        "",
        "## Controls",
        "",
        "Compare `live_moment_lpf_b0p2` and `live_all_lpf_b0p2` with the thrust-only",
        "variant.  If thrust-only is uniquely effective, that strengthens the conclusion",
        "that collective-thrust bandwidth is the useful intervention rather than generic",
        "action smoothing.",
    ])
    return "\n".join(lines) + "\n"


def run(args):
    run_dir = Path(args.run_dir).expanduser().resolve()
    curriculum_path = Path(args.curriculum).expanduser().resolve()
    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir else run_dir / "evaluation" / "id_thrust_filter_verification"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    curriculum = load_curriculum(curriculum_path)
    _validate_id_curriculum(curriculum, bool(args.allow_nonstandard_curriculum))
    cfg = _load_resolved_config(run_dir)
    cfg.env.residual_interface = "wrench"

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; using CPU")
        device = "cpu"
    cfg.device = device
    cfg.env.device = device

    stage_cfgs = [env_config_for_stage(cfg.env, s) for s in curriculum.stages]
    checkpoint = _resolve_checkpoint(run_dir, args.checkpoint)
    agent, _ = _load_policy(cfg, stage_cfgs[0], checkpoint, device)

    print("ID-only verification")
    print(f"  curriculum: {curriculum_path}")
    print(f"  checkpoint: {checkpoint}")
    print(f"  stages: {', '.join(s.name for s in curriculum.stages)}")
    print(f"  episodes/stage: {args.episodes_per_stage}")
    print(f"  variants/case: {len(VARIANTS)}")
    print(f"  total rollouts: {len(curriculum.stages) * args.episodes_per_stage * len(VARIANTS)}")
    print("")

    all_rows: list[dict] = []
    for si, (stage, stage_cfg) in enumerate(zip(curriculum.stages, stage_cfgs)):
        print(f"[stage {si+1}/{len(stage_cfgs)}] {stage.name} disturbances={stage.env_overrides.get('disturbances')}")
        for case in range(int(args.episodes_per_stage)):
            seed = int(args.seed + 100_000 * si + case)
            ref_dist = None
            for variant in VARIANTS:
                metrics, dist = _rollout(stage_cfg, seed, agent, variant, args)
                if ref_dist is None:
                    ref_dist = dist
                else:
                    _assert_same_disturbance(ref_dist, dist)
                row = {
                    "stage": stage.name,
                    "stage_index": si + 1,
                    "case": case,
                    "seed": seed,
                    "variant": variant.name,
                    **_disturbance_columns(dist),
                    **metrics,
                }
                all_rows.append(row)
            if case == 0 and ref_dist is not None:
                dc = _disturbance_columns(ref_dist)
                print(
                    f"  preflight seed={seed}: k={dc['k']:.3f}, |Fext|={dc['force_norm_N']:.3f} N, "
                    f"motor_mean={dc['motor_coeff_mean']:.3f}, arm_mean={dc['arm_length_mean']:.3f}"
                )
            if (case + 1) % max(1, int(args.progress_every)) == 0 or case + 1 == int(args.episodes_per_stage):
                raw = all_rows[-(len(VARIANTS)):]  # current case variants
                rr = next(r for r in raw if r["variant"] == "live_raw")
                tf = next(r for r in raw if r["variant"] == "live_thrust_lpf_b0p2")
                print(
                    f"  case {case+1:3d}/{args.episodes_per_stage}: "
                    f"raw RMSE={rr['true_des_pos_rmse_m']:.4f} omegaR={rr['omega_roughness']:.4g} sat={rr['sat_fraction']:.3f} | "
                    f"thrust-b0.2 RMSE={tf['true_des_pos_rmse_m']:.4f} omegaR={tf['omega_roughness']:.4g} sat={tf['sat_fraction']:.3f}"
                )

    _write_csv(out_dir / "all_episode_metrics.csv", all_rows)
    high_subset = _high_oscillation_subset(all_rows, float(args.high_oscillation_fraction))
    all_summary = _summaries(all_rows, None, "all_id", int(args.seed) + 11)
    high_summary = _summaries(all_rows, high_subset, "raw_high_oscillation", int(args.seed) + 22)
    stage_summary = _stage_summaries(all_rows)
    _write_csv(out_dir / "summary_all_id.csv", all_summary)
    _write_csv(out_dir / "summary_raw_high_oscillation.csv", high_summary)
    _write_csv(out_dir / "summary_by_stage.csv", stage_summary)

    selected_rows = []
    for stage, case in sorted(high_subset):
        rr = next(r for r in all_rows if r["stage"] == stage and r["case"] == case and r["variant"] == "live_raw")
        selected_rows.append({
            "stage": stage, "case": case, "seed": rr["seed"],
            "raw_omega_roughness": rr["omega_roughness"],
            "raw_sat_fraction": rr["sat_fraction"],
            "raw_pos_rmse_m": rr["true_des_pos_rmse_m"],
        })
    _write_csv(out_dir / "raw_high_oscillation_cases.csv", selected_rows)

    key_variants = (
        "baseline_zero_residual", "live_raw", "live_thrust_lpf_b0p2",
        "live_moment_lpf_b0p2", "live_all_lpf_b0p2",
    )
    _plot_stage_metric(stage_summary, out_dir / "id_position_rmse_by_stage.png",
                       "pos_rmse_mean_m", "true -> desired position RMSE [m]", key_variants)
    _plot_stage_metric(stage_summary, out_dir / "id_saturation_by_stage.png",
                       "sat_fraction_mean", "actuator saturation fraction", key_variants)
    _plot_stage_metric(stage_summary, out_dir / "id_omega_roughness_by_stage.png",
                       "omega_roughness_mean", "body-rate roughness", key_variants)
    _plot_stage_metric(stage_summary, out_dir / "id_residual_thrust_roughness_by_stage.png",
                       "residual_thrust_roughness_N_mean", "residual collective-thrust roughness [N/sample]", key_variants)

    (out_dir / "DECISION.md").write_text(_decision_text(all_summary, high_summary), encoding="utf-8")
    metadata = {
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "curriculum": curriculum_path,
        "stage_names": [s.name for s in curriculum.stages],
        "episodes_per_stage": int(args.episodes_per_stage),
        "seed": int(args.seed),
        "variants": [v.name for v in VARIANTS],
        "high_oscillation_fraction_per_stage": float(args.high_oscillation_fraction),
        "hf_cutoff_hz": float(args.hf_cutoff_hz),
        "transient_ignore_s": float(args.transient_ignore_s),
        "max_steps": args.max_steps,
    }
    (out_dir / "metadata.json").write_text(json.dumps(_jsonable(metadata), indent=2), encoding="utf-8")

    print(f"\nWrote: {out_dir}")
    print(f"Decision: {out_dir / 'DECISION.md'}")


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--curriculum", default="configs/residual_sac_curriculum.toml")
    p.add_argument("--checkpoint", default="best")
    p.add_argument("--episodes_per_stage", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260811)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--max_steps", type=int, default=None)
    p.add_argument("--hf_cutoff_hz", type=float, default=10.0)
    p.add_argument("--transient_ignore_s", type=float, default=0.5)
    p.add_argument("--high_oscillation_fraction", type=float, default=0.20)
    p.add_argument("--progress_every", type=int, default=5)
    p.add_argument("--allow_nonstandard_curriculum", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    if args.episodes_per_stage < 1:
        raise SystemExit("--episodes_per_stage must be >= 1")
    if not (0.0 < args.high_oscillation_fraction <= 1.0):
        raise SystemExit("--high_oscillation_fraction must be in (0,1]")
    run(args)


if __name__ == "__main__":
    main()
