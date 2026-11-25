"""Asynchronous parallel constrained GP-Hedge BO on a 2-D toy problem.

    python examples/async_bo.py

Maximises ``-rosenbrock`` with a known constraint (``x0 + x1 <= 2.5``, analytic) and an
unknown one (the "simulation" crashes inside a disc, discovered only by running it). Jobs
take random time on a thread pool, so proposals are made while others are still running.
"""

import tempfile
import time
from pathlib import Path

import numpy as np

import gp_omni  # noqa: F401
from gp_omni.opt.parallel.asyncbo import async_bo, async_bo_config

CONFIG = Path(__file__).parents[1] / "config" / "async_bo.json"


def simulation(x):
    time.sleep(np.random.default_rng().uniform(0.05, 0.3))  # variable run time
    crashed = np.linalg.norm(x - [1.0, 1.0]) < 0.35  # unknown constraint
    return -((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2) / 100, not crashed


def known_constraint(x):
    return float(x[0] + x[1] <= 2.5)


def main(max_iter: int | None = None, seed: int | None = None) -> dict:
    """Settings come from ``config/async_bo.json``; arguments override the file."""
    overrides = {k: v for k, v in {"max_iter": max_iter, "seed": seed}.items() if v is not None}
    config = async_bo_config.from_json(CONFIG, known_constraint=known_constraint, **overrides)
    with tempfile.TemporaryDirectory() as root:
        x, y = async_bo(root, config, evaluate=simulation).run()
    print(f"best feasible x = {np.round(x, 3)}, y = {y:.4f}")
    return {"x": x, "y": y}


if __name__ == "__main__":
    main()
