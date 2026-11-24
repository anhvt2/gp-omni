"""Batch-sequential parallel constrained BO on a 2-D toy problem.

    python examples/batch_seq_bo.py

Same problem as ``async_bo.py`` (known constraint ``x0 + x1 <= 2.5``, unknown crash region),
but every iteration launches a whole batch in parallel and waits for all of it to finish.
Settings come from ``config/batch_seq_bo.json``.
"""

import tempfile
import time
from pathlib import Path

import numpy as np

import gp_omni  # noqa: F401
from gp_omni.train.batchbo import batch_bo, batch_bo_config

CONFIG = Path(__file__).parents[1] / "config" / "batch_seq_bo.json"


def simulation(x):
    time.sleep(np.random.default_rng().uniform(0.05, 0.3))  # variable run time
    crashed = np.linalg.norm(x - [1.0, 1.0]) < 0.35  # unknown constraint
    return -((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2) / 100, not crashed


def known_constraint(x):
    return float(x[0] + x[1] <= 2.5)


def main(max_iter: int | None = None, seed: int | None = None) -> dict:
    """Arguments override the settings file."""
    overrides = {k: v for k, v in {"max_iter": max_iter, "seed": seed}.items() if v is not None}
    config = batch_bo_config.from_json(CONFIG, known_constraint=known_constraint, **overrides)
    with tempfile.TemporaryDirectory() as root:
        x, y = batch_bo(root, config, evaluate=simulation).run()
    print(f"best feasible x = {np.round(x, 3)}, y = {y:.4f}")
    return {"x": x, "y": y}


if __name__ == "__main__":
    main()
