"""Run-directory, logging, and checkpoint naming utilities for residual SAC."""

import json
import os
from dataclasses import asdict, is_dataclass


def next_trial_index(runs_root, run_name):
    """Return the smallest unused positive run index."""
    base = os.path.join(runs_root, run_name)
    n = 1
    while os.path.exists(os.path.join(base, f"trial_{n:03d}")):
        n += 1
    return n


def resolve_run_dir(cfg, create=True):
    """Resolve ``<runs_root>/<run_name>/trial_XXX`` and optionally create it."""
    trial = cfg.trial if cfg.trial is not None else next_trial_index(cfg.runs_root, cfg.run_name)
    run_dir = os.path.join(cfg.runs_root, cfg.run_name, f"trial_{int(trial):03d}")
    if create:
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
        os.makedirs(os.path.join(run_dir, "evaluation"), exist_ok=True)
    return run_dir, int(trial)


def resolve_run_dir_args(runs_root, run_name, trial=None, create=True):
    """Plain-argument variant used by auxiliary scripts."""
    t = int(trial) if trial is not None else next_trial_index(runs_root, run_name)
    run_dir = os.path.join(runs_root, run_name, f"trial_{t:03d}")
    if create:
        os.makedirs(run_dir, exist_ok=True)
    return run_dir, t


def checkpoint_name(kind, step=None):
    """Return a stable checkpoint filename.

    ``kind`` is normally ``best``, ``last``, or ``interrupted``. Periodic
    checkpoints use ``kind='step'`` plus a step number.
    """
    if kind == "step":
        if step is None:
            raise ValueError("step checkpoint requires a step number")
        return f"step_{int(step):09d}.pt"
    return f"{kind}.pt"


def dump_config(cfg, run_dir):
    """Write the fully resolved nested configuration to ``config.json``."""
    def to_plain(x):
        if is_dataclass(x):
            return {k: to_plain(v) for k, v in asdict(x).items()}
        if isinstance(x, (tuple, list)):
            return [to_plain(v) for v in x]
        return x

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(to_plain(cfg), f, indent=2)
