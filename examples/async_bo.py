"""Asynchronous parallel constrained GP-Hedge BO on a 2-D toy problem.

    python examples/async_bo.py

Maximises ``-rosenbrock`` with a known constraint (``x0 + x1 <= 2.5``, analytic) and an
unknown one (the "simulation" crashes inside a disc, discovered only by running it). Jobs
take random time on a thread pool, so proposals are made while others are still running.
"""

import tempfile
import time

import numpy as np

import gp_omni  # noqa: F401
from gp_omni.asyncbo import AsyncBO, AsyncBOConfig


def simulation(x):
    time.sleep(np.random.default_rng().uniform(0.05, 0.3))  # variable run time
    crashed = np.linalg.norm(x - [1.0, 1.0]) < 0.35  # unknown constraint
    return -((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2) / 100, not crashed


def known_constraint(x):
    return float(x[0] + x[1] <= 2.5)


def main(max_iter: int = 20, seed: int = 0) -> dict:
    cfg = AsyncBOConfig(
        model_name="rosen",
        x_lb=[-1.0, -1.0],
        x_ub=[2.0, 2.0],
        exploit_size=2,
        explore_size=1,
        explore_clf_size=1,
        max_iter=max_iter,
        num_init_point=4,
        check_time=0.01,
        gp_max_iter=60,
        cma_max_evals=60,
        seed=seed,
        known_constraint=known_constraint,
    )
    with tempfile.TemporaryDirectory() as root:
        x, y = AsyncBO(root, cfg, evaluate=simulation).run()
    print(f"best feasible x = {np.round(x, 3)}, y = {y:.4f}")
    return {"x": x, "y": y}


if __name__ == "__main__":
    main()
