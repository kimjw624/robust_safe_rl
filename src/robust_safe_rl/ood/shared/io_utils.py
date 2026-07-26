"""Small filesystem helpers shared by the runnable scripts.

Checkpoints are written with a zero-padded numeric prefix (e.g. ``001_ae_...``)
so that repeated training runs auto-increment instead of overwriting. These
helpers centralize the "next index" and "load by index" logic that was
previously copy-pasted across the train/test/eval scripts.
"""

from pathlib import Path


def get_next_indexed_path(directory, suffix):
    """Return (path, index) for the next free ``NNN_<suffix>`` in ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    used = []
    for path in directory.glob(f"*_{suffix}"):
        prefix = path.name.split("_", 1)[0]
        if prefix.isdigit():
            used.append(int(prefix))
    index = 1 if not used else max(used) + 1
    return directory / f"{index:03d}_{suffix}", index


def get_checkpoint_by_index(run_dir, index, suffix):
    """Return the path to ``NNN_<suffix>`` for a given index, or raise."""
    path = Path(run_dir) / f"{index:03d}_{suffix}"
    if not path.exists():
        raise FileNotFoundError(f"Could not find checkpoint: {path}")
    return path
