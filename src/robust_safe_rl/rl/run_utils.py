"""Run-directory and checkpoint naming utilities.

Layout produced:

    <runs_root>/<run_name>/trial_<N>/
        <run_name>_trial_<N>.pt          final checkpoint
        <run_name>_trial_<N>_step<S>.pt  periodic checkpoints
        log.json                         training / eval curve
        config.json                      full config dump
        eval/                            created by the evaluation script
            <k-tagged plots and logs>

trial_<N> auto-increments: re-running the same run_name creates trial_2,
trial_3, ... so nothing is ever overwritten.
"""

import json
import os
from dataclasses import asdict, is_dataclass


def next_trial_index(runs_root, run_name):
    """Return the smallest trial index whose folder does not yet exist (>=1)."""
    base = os.path.join(runs_root, run_name)
    n = 1
    while os.path.exists(os.path.join(base, f"trial_{n}")):
        n += 1
    return n


def resolve_run_dir(cfg, create=True):
    """Resolve (and optionally create) the trial directory for this run.

    Uses cfg.trial if given, else auto-increments. Returns (run_dir, trial_index).
    """
    trial = cfg.trial if cfg.trial is not None else next_trial_index(cfg.runs_root, cfg.run_name)
    run_dir = os.path.join(cfg.runs_root, cfg.run_name, f"trial_{trial}")
    if create:
        os.makedirs(run_dir, exist_ok=True)
    return run_dir, trial


def resolve_run_dir_args(runs_root, run_name, trial=None, create=True):
    """Same as resolve_run_dir but from plain strings (for scripts that don't
    carry a Config object, e.g. the residual-dynamics trainer). Auto-increments
    the trial index when trial is None. Returns (run_dir, trial_index)."""
    t = trial if trial is not None else next_trial_index(runs_root, run_name)
    run_dir = os.path.join(runs_root, run_name, f"trial_{t}")
    if create:
        os.makedirs(run_dir, exist_ok=True)
    return run_dir, t


def checkpoint_name(run_name, trial, step=None):
    """Checkpoint filename: final if step is None, else step-tagged."""
    if step is None:
        return f"{run_name}_trial_{trial}.pt"
    return f"{run_name}_trial_{trial}_step{step}.pt"


def dump_config(cfg, run_dir):
    """Write the full nested config to config.json for reproducibility."""
    def to_plain(x):
        if is_dataclass(x):
            return {k: to_plain(v) for k, v in asdict(x).items()}
        if isinstance(x, (tuple, list)):
            return [to_plain(v) for v in x]
        return x
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(to_plain(cfg), f, indent=2)