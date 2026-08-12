"""Mine residual-wrench evaluation episodes for oscillation triggers.

This script treats an existing paired evaluation as a dataset.  It deliberately
separates two kinds of variables:

* exogenous episode factors: mass/MOI, external force, actuator/geometry scales,
* closed-loop symptoms: saturation, motor utilization, residual-action activity.

The separation matters: a symptom can be an excellent predictor of oscillation
without being the root cause.  The companion script
``run_residual_ruleout_experiments.py`` performs causal manipulations on the
seeds selected here.

The oscillation score uses only control-output behavior, not saturation:

    score = 0.5 * percentile(log10(wrench roughness))
          + 0.5 * percentile(high-frequency wrench ratio)

By default the top/bottom 20% are labeled ``oscillatory`` / ``clean``; the
middle 60% are left unlabeled for the simple threshold-rule analysis.

Example
-------
PYTHONPATH=src python3 -m scripts.analyze_oscillation_dataset \\
  --evaluation_dir runs_residual/residual_sac_curriculum/trial_002/evaluation/baseline_vs_residual_best
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-12

EXOGENOUS_BASE = (
    "k_abs_dev",
    "force_norm_N",
    "force_abs_x_N",
    "force_abs_y_N",
    "force_abs_z_N",
    "motor_coeff_mean",
    "motor_coeff_spread",
    "motor_coeff_asym_rms",
    "motor_coeff_mismatch_rms",
    "moment_coeff_mean",
    "moment_coeff_spread",
    "moment_coeff_asym_rms",
    "moment_coeff_mismatch_rms",
    "arm_length_mean",
    "arm_length_spread",
    "arm_length_asym_rms",
    "arm_length_mismatch_rms",
)

SYMPTOM_FEATURES = (
    "residual_actuator_sat_fraction",
    "residual_motor_utilization_mean",
    "residual_motor_utilization_peak",
    "residual_normalized_action_rms",
    "residual_normalized_action_peak",
    "residual_normalized_action_roughness",
    "residual_residual_f_rms",
    "residual_residual_mx_rms",
    "residual_residual_my_rms",
    "residual_residual_mz_rms",
    "residual_true_des_pos_rmse",
    "residual_terminated",
)


def _float(row: dict, key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


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


def _rank_percentile(values: np.ndarray) -> np.ndarray:
    """Average-rank percentile in [0, 1], robust to ties."""
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n <= 1:
        return np.zeros(n, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i + 1
        while j < n and x[order[j]] == x[order[i]]:
            j += 1
        avg = 0.5 * (i + j - 1)
        ranks[order[i:j]] = avg
        i = j
    return ranks / float(n - 1)


def _scale_features(row: dict, prefix: str) -> dict[str, float]:
    vals = np.asarray([_float(row, f"{prefix}_{i}") for i in range(4)], dtype=float)
    if not np.all(np.isfinite(vals)):
        mean = _float(row, f"{prefix}_mean")
        lo = _float(row, f"{prefix}_min")
        hi = _float(row, f"{prefix}_max")
        return {
            f"{prefix}_mean": mean,
            f"{prefix}_spread": hi - lo,
            f"{prefix}_asym_rms": float("nan"),
            f"{prefix}_mismatch_rms": float("nan"),
        }
    mean = float(np.mean(vals))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_spread": float(np.max(vals) - np.min(vals)),
        f"{prefix}_asym_rms": float(np.sqrt(np.mean((vals - mean) ** 2))),
        f"{prefix}_mismatch_rms": float(np.sqrt(np.mean((vals - 1.0) ** 2))),
    }


def engineer_features(row: dict) -> dict[str, float]:
    out = {
        "k_abs_dev": abs(_float(row, "k") - 1.0),
        "force_norm_N": _float(row, "force_norm_N"),
        "force_abs_x_N": abs(_float(row, "force_x_N")),
        "force_abs_y_N": abs(_float(row, "force_y_N")),
        "force_abs_z_N": abs(_float(row, "force_z_N")),
    }
    for prefix in ("motor_coeff", "moment_coeff", "arm_length"):
        out.update(_scale_features(row, prefix))
    for key in SYMPTOM_FEATURES:
        out[key] = _float(row, key)
    return out


def oscillation_scores(rows: list[dict]) -> np.ndarray:
    rough = np.asarray([_float(r, "residual_applied_wrench_roughness", 0.0) for r in rows])
    hf = np.asarray([_float(r, "residual_applied_wrench_hf_ratio", 0.0) for r in rows])
    rough_rank = _rank_percentile(np.log10(np.maximum(rough, 0.0) + EPS))
    hf_rank = _rank_percentile(np.maximum(hf, 0.0))
    return 0.5 * rough_rank + 0.5 * hf_rank


def _mean_std(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    return float(np.mean(x)), float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def _standardized_mean_difference(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va = np.var(a, ddof=1); vb = np.var(b, ddof=1)
    pooled = math.sqrt(max(0.5 * (va + vb), EPS))
    return float((np.mean(a) - np.mean(b)) / pooled)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.sum(m) < 3:
        return float("nan")
    xr = _rank_percentile(x[m]); yr = _rank_percentile(y[m])
    if np.std(xr) <= EPS or np.std(yr) <= EPS:
        return float("nan")
    return float(np.corrcoef(xr, yr)[0, 1])


def _best_threshold_rule(x: np.ndarray, y: np.ndarray) -> dict:
    """Find an interpretable one-feature threshold maximizing balanced accuracy."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=int)
    valid = np.isfinite(x) & ((y == 0) | (y == 1))
    x, y = x[valid], y[valid]
    if len(x) < 8 or len(np.unique(y)) < 2:
        return {"threshold": float("nan"), "direction": "", "balanced_accuracy": float("nan"),
                "tpr": float("nan"), "tnr": float("nan")}

    ux = np.unique(x)
    if len(ux) > 200:
        thresholds = np.unique(np.quantile(x, np.linspace(0.01, 0.99, 199)))
    elif len(ux) > 1:
        thresholds = 0.5 * (ux[:-1] + ux[1:])
    else:
        thresholds = ux

    best = None
    for thr in thresholds:
        for direction in (">=", "<="):
            pred = x >= thr if direction == ">=" else x <= thr
            pos = y == 1; neg = y == 0
            tpr = float(np.mean(pred[pos])) if np.any(pos) else 0.0
            tnr = float(np.mean(~pred[neg])) if np.any(neg) else 0.0
            ba = 0.5 * (tpr + tnr)
            cand = (ba, tpr + tnr, -abs(float(thr)), float(thr), direction, tpr, tnr)
            if best is None or cand[:3] > best[:3]:
                best = cand
    assert best is not None
    return {
        "threshold": best[3], "direction": best[4], "balanced_accuracy": best[0],
        "tpr": best[5], "tnr": best[6],
    }


def _within_stage_rank(values: np.ndarray, stages: list[str]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan)
    for stage in sorted(set(stages)):
        idx = np.asarray([s == stage for s in stages], dtype=bool)
        v = values[idx]
        finite = np.isfinite(v)
        if np.sum(finite) <= 1:
            out[idx] = 0.5
            continue
        tmp = np.full(len(v), np.nan)
        tmp[finite] = _rank_percentile(v[finite])
        out[idx] = tmp
    return out


def _plot_scatter(out: Path, x: np.ndarray, score: np.ndarray, xlabel: str, name: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(x, score, s=16, alpha=0.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("oscillation score")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / name, dpi=160)
    plt.close(fig)


def run(args):
    eval_dir = Path(args.evaluation_dir).expanduser().resolve()
    csv_path = eval_dir / "paired_episode_metrics.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"missing paired evaluation CSV: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"no rows in {csv_path}")

    out = Path(args.output).expanduser().resolve() if args.output else eval_dir / "oscillation_dataset_analysis"
    out.mkdir(parents=True, exist_ok=True)

    scores = oscillation_scores(rows)
    clean_thr = float(np.quantile(scores, args.clean_quantile))
    severe_thr = float(np.quantile(scores, args.severe_quantile))
    if not 0.0 <= args.clean_quantile < args.severe_quantile <= 1.0:
        raise ValueError("require 0 <= clean_quantile < severe_quantile <= 1")

    features = [engineer_features(r) for r in rows]
    stages = [r["stage"] for r in rows]
    labels = np.full(len(rows), -1, dtype=int)
    # A short non-oscillatory crash is not a useful "clean" control.  Require a
    # near-complete, non-terminated residual rollout for the clean class.
    clean_eligible = np.asarray([
        _float(r, "residual_terminated", 0.0) < 0.5
        and _float(r, "residual_completion_fraction", 1.0) >= args.clean_min_completion
        for r in rows
    ], dtype=bool)
    labels[(scores <= clean_thr) & clean_eligible] = 0
    labels[scores >= severe_thr] = 1

    episode_rows = []
    for i, (r, feat) in enumerate(zip(rows, features)):
        label = "oscillatory" if labels[i] == 1 else ("clean" if labels[i] == 0 else "middle")
        episode_rows.append({
            "stage": r["stage"], "case": r.get("case", ""), "seed": r["seed"],
            "oscillation_score": float(scores[i]), "label": label,
            "wrench_roughness": _float(r, "residual_applied_wrench_roughness"),
            "wrench_hf_ratio": _float(r, "residual_applied_wrench_hf_ratio"),
            "sat_fraction": _float(r, "residual_actuator_sat_fraction"),
            "position_rmse_m": _float(r, "residual_true_des_pos_rmse"),
            "terminated": _float(r, "residual_terminated"),
            **feat,
        })
    _write_csv(out / "episode_oscillation_labels.csv", episode_rows)

    comparison_rows = []
    rule_rows = []
    stage_rule_rows = []
    for group, names in (("exogenous", EXOGENOUS_BASE), ("closed_loop_symptom", SYMPTOM_FEATURES)):
        for name in names:
            x = np.asarray([f.get(name, np.nan) for f in features], dtype=float)
            clean = x[labels == 0]; severe = x[labels == 1]
            cm, cs = _mean_std(clean); sm, ss = _mean_std(severe)
            comparison_rows.append({
                "group": group, "feature": name,
                "clean_mean": cm, "clean_std": cs,
                "oscillatory_mean": sm, "oscillatory_std": ss,
                "standardized_mean_difference_osc_minus_clean": _standardized_mean_difference(severe, clean),
                "spearman_vs_oscillation_score": _spearman(x, scores),
            })
            rule = _best_threshold_rule(x, labels)
            rule_rows.append({"group": group, "feature": name, **rule})
            xr = _within_stage_rank(x, stages)
            stage_rule_rows.append({"group": group, "feature": name, **_best_threshold_rule(xr, labels)})

    _write_csv(out / "feature_group_comparison.csv", comparison_rows)
    _write_csv(out / "feature_threshold_rules.csv", rule_rows)
    _write_csv(out / "feature_threshold_rules_within_stage_rank.csv", stage_rule_rows)

    stage_rows = []
    for stage in sorted(set(stages)):
        idx = np.asarray([s == stage for s in stages], dtype=bool)
        stage_rows.append({
            "stage": stage,
            "episodes": int(np.sum(idx)),
            "clean_fraction": float(np.mean(labels[idx] == 0)),
            "oscillatory_fraction": float(np.mean(labels[idx] == 1)),
            "score_mean": float(np.mean(scores[idx])),
            "roughness_mean": float(np.mean([_float(rows[i], "residual_applied_wrench_roughness") for i in np.where(idx)[0]])),
            "hf_ratio_mean": float(np.mean([_float(rows[i], "residual_applied_wrench_hf_ratio") for i in np.where(idx)[0]])),
            "sat_fraction_mean": float(np.mean([_float(rows[i], "residual_actuator_sat_fraction") for i in np.where(idx)[0]])),
        })
    _write_csv(out / "stage_prevalence.csv", stage_rows)

    # Cases for the causal script: retain the exact disturbance columns from the
    # original row so a rerun can verify that seed/config reproduction is exact.
    selected = []
    n = max(int(args.cases_per_class_per_stage), 0)
    for stage in sorted(set(stages)):
        idx = [i for i, s in enumerate(stages) if s == stage]
        clean_idx = sorted([i for i in idx if labels[i] == 0], key=lambda i: scores[i])[:n]
        severe_idx = sorted([i for i in idx if labels[i] == 1], key=lambda i: scores[i], reverse=True)[:n]
        for label, chosen in (("clean", clean_idx), ("oscillatory", severe_idx)):
            for i in chosen:
                base = dict(rows[i])
                base["class"] = label
                base["oscillation_score"] = float(scores[i])
                selected.append(base)
    _write_csv(out / "selected_cases.csv", selected)

    # A few direct visual checks. Saturation is intentionally plotted as a
    # symptom, not included in the oscillation score itself.
    sat = np.asarray([_float(r, "residual_actuator_sat_fraction") for r in rows])
    force = np.asarray([f["force_norm_N"] for f in features])
    asym = np.asarray([f["motor_coeff_asym_rms"] for f in features])
    _plot_scatter(out, sat, scores, "actuator saturation fraction", "score_vs_saturation.png")
    _plot_scatter(out, force, scores, "external force norm [N]", "score_vs_force.png")
    _plot_scatter(out, asym, scores, "motor coefficient asymmetry RMS", "score_vs_motor_asymmetry.png")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([r["stage"] for r in stage_rows], [r["oscillatory_fraction"] for r in stage_rows])
    ax.set_ylabel("fraction labeled oscillatory")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "oscillatory_fraction_by_stage.png", dpi=160)
    plt.close(fig)

    exo_sorted = sorted(
        [r for r in comparison_rows if r["group"] == "exogenous"],
        key=lambda r: abs(r["spearman_vs_oscillation_score"]) if np.isfinite(r["spearman_vs_oscillation_score"]) else -1,
        reverse=True,
    )
    sym_sorted = sorted(
        [r for r in comparison_rows if r["group"] == "closed_loop_symptom"],
        key=lambda r: abs(r["spearman_vs_oscillation_score"]) if np.isfinite(r["spearman_vs_oscillation_score"]) else -1,
        reverse=True,
    )
    lines = [
        "# Oscillation dataset analysis",
        "",
        f"Episodes: **{len(rows)}**",
        f"Clean threshold (score q={args.clean_quantile:.2f}): **{clean_thr:.4f}**",
        f"Oscillatory threshold (score q={args.severe_quantile:.2f}): **{severe_thr:.4f}**",
        "",
        "The oscillation score uses only applied-wrench roughness and HF ratio. Saturation is deliberately excluded so it can be tested as a candidate mechanism rather than baked into the label.",
        f"Clean controls additionally require residual completion >= {args.clean_min_completion:.2f} and no termination; short quiet crashes are not labeled clean.",
        "",
        "## Strongest exogenous associations",
        "",
        "| feature | Spearman rho | standardized difference |",
        "|---|---:|---:|",
    ]
    for r in exo_sorted[:8]:
        lines.append(f"| {r['feature']} | {r['spearman_vs_oscillation_score']:.3f} | {r['standardized_mean_difference_osc_minus_clean']:.3f} |")
    lines += [
        "",
        "## Strongest closed-loop symptom associations",
        "",
        "| feature | Spearman rho | standardized difference |",
        "|---|---:|---:|",
    ]
    for r in sym_sorted[:8]:
        lines.append(f"| {r['feature']} | {r['spearman_vs_oscillation_score']:.3f} | {r['standardized_mean_difference_osc_minus_clean']:.3f} |")
    lines += [
        "",
        "## How to use this output",
        "",
        "`selected_cases.csv` is the input to `scripts.run_residual_ruleout_experiments`. The causal script replays those exact stage/seed combinations and changes one mechanism at a time: residual moment authority, saturation guard/headroom, disturbance force, actuator asymmetry, and synthetic residual bursts.",
        "",
        "A high association here does **not** establish causality. In particular saturation, motor utilization, and action roughness are closed-loop symptoms. Use the controlled replay battery to rule hypotheses in or out.",
    ]
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "analysis_metadata.json").write_text(json.dumps({
        "source_csv": str(csv_path),
        "episodes": len(rows),
        "clean_quantile": args.clean_quantile,
        "severe_quantile": args.severe_quantile,
        "clean_min_completion": args.clean_min_completion,
        "clean_score_threshold": clean_thr,
        "severe_score_threshold": severe_thr,
    }, indent=2), encoding="utf-8")

    print(f"Analyzed {len(rows)} episodes")
    print(f"  clean <= {clean_thr:.4f}, oscillatory >= {severe_thr:.4f}")
    print(f"  selected causal cases: {len(selected)}")
    print(f"Output: {out}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--evaluation_dir", required=True, help="paired evaluation directory containing paired_episode_metrics.csv")
    p.add_argument("--clean_quantile", type=float, default=0.20)
    p.add_argument("--severe_quantile", type=float, default=0.80)
    p.add_argument("--clean_min_completion", type=float, default=0.95)
    p.add_argument("--cases_per_class_per_stage", type=int, default=2)
    p.add_argument("--output", default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
