"""Accuracy of ``GPR`` against the true value of each test function, per modelling context.

Metrics are computed on held-out Sobol points against the noise-free truth:
``nrmse = rmse / std(truth)`` (0 is perfect, 1 is as bad as predicting the mean) and ``r2``.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import torch
from torch import Tensor

import gp_omni  # noqa: F401  (sets float64)
from gp_omni import GPR

from .common import (
    BenchmarkFunction,
    MultiFidelityFunction,
    MultiObjectiveFunction,
    sobol_design,
)
from .noisy_regression import NoisyProblem

TEST_SEED = 12345


def error_metrics(pred: Tensor, truth: Tensor) -> dict[str, float]:
    err = pred - truth
    rmse = err.pow(2).mean().sqrt().item()
    sd = truth.std().item()
    return {
        "rmse": rmse,
        "nrmse": rmse / sd,
        "r2": 1 - err.pow(2).sum().item() / ((truth - truth.mean()) ** 2).sum().item(),
        "max_abs_err": err.abs().max().item(),
    }


def _timed(fit: Callable[[], GPR]) -> tuple[GPR, float]:
    t0 = time.perf_counter()
    model = fit()
    return model, time.perf_counter() - t0


def single_fidelity_accuracy(
    fn: BenchmarkFunction,
    n_train: int,
    n_test: int = 1000,
    seed: int = 0,
    max_iter: int = 150,
    **gpr_kwargs,
) -> dict[str, float]:
    """Noise-free single-output regression; ``gpr_kwargs`` select e.g. ``additive=2``."""
    x = sobol_design(fn.bounds, n_train, seed)
    xt = sobol_design(fn.bounds, n_test, TEST_SEED)
    model, secs = _timed(lambda: GPR(**gpr_kwargs).fit(x, fn(x), max_iter=max_iter))
    return {
        **error_metrics(model.predict(xt)["fmu"], fn(xt)),
        "n_train": n_train,
        "fit_seconds": secs,
    }


def multi_fidelity_accuracy(
    fn: MultiFidelityFunction,
    n_per_level: tuple[int, ...],
    n_test: int = 1000,
    seed: int = 0,
    max_iter: int = 150,
) -> dict[str, float]:
    """Nested design (each level is a prefix of the cheaper one's Sobol points).

    Compares the multi-fidelity ``GPR`` with a high-fidelity-only ``GPR`` trained on the same
    high-fidelity points, both scored against the true high-fidelity function.
    """
    assert len(n_per_level) == fn.num_fidelities
    pool = sobol_design(fn.bounds, n_per_level[0], seed)
    xs = [pool[:n] for n in n_per_level]
    x = torch.cat(xs)
    fidelity = torch.cat([torch.full((len(a),), k) for k, a in enumerate(xs)])
    y = torch.cat([fn(a, k) for k, a in enumerate(xs)])
    xt = sobol_design(fn.bounds, n_test, TEST_SEED)
    truth = fn(xt)

    mf, secs = _timed(
        lambda: GPR(num_fidelities=fn.num_fidelities).fit(x, y, fidelity=fidelity, max_iter=max_iter)
    )
    hi_only = GPR().fit(xs[-1], fn(xs[-1]), max_iter=max_iter)
    mf_m = error_metrics(mf.predict(xt)["fmu"], truth)
    hi_m = error_metrics(hi_only.predict(xt)["fmu"], truth)
    return {
        "mf_nrmse": mf_m["nrmse"],
        "mf_r2": mf_m["r2"],
        "hi_only_nrmse": hi_m["nrmse"],
        "hi_only_r2": hi_m["r2"],
        "gain": hi_m["nrmse"] / mf_m["nrmse"],
        "fit_seconds": secs,
    }


def multi_objective_accuracy(
    fn: MultiObjectiveFunction,
    n_train: int,
    n_test: int = 1000,
    seed: int = 0,
    max_iter: int = 150,
    sparse_last: int | None = None,
) -> dict[str, list[float] | float]:
    """Joint multi-output ``GPR`` (correlated objectives) vs. one independent ``GPR`` each.

    With ``sparse_last`` the last objective is observed at only the first ``sparse_last`` design
    points (a data-poor objective), the case multi-output modelling is meant to help.
    """
    m = fn.num_objectives
    counts = [n_train] * (m - 1) + [sparse_last or n_train]
    x = sobol_design(fn.bounds, n_train, seed)
    y = fn(x)
    xt = sobol_design(fn.bounds, n_test, TEST_SEED)
    truth = fn(xt)

    xs = torch.cat([x[:n] for n in counts])
    ys = torch.cat([y[:n, k] for k, n in enumerate(counts)])
    output = torch.cat([torch.full((n,), k) for k, n in enumerate(counts)])
    joint, secs = _timed(lambda: GPR(num_outputs=m).fit(xs, ys, output=output, max_iter=max_iter))
    pred = joint.predict(xt)["fmu"]
    joint_nrmse = [error_metrics(pred[:, k], truth[:, k])["nrmse"] for k in range(m)]
    indep_nrmse = []
    for k, n in enumerate(counts):
        single = GPR().fit(x[:n], y[:n, k], max_iter=max_iter)
        indep_nrmse.append(error_metrics(single.predict(xt)["fmu"], truth[:, k])["nrmse"])
    return {
        "joint_nrmse": joint_nrmse,
        "independent_nrmse": indep_nrmse,
        "joint_mean_nrmse": sum(joint_nrmse) / m,
        "independent_mean_nrmse": sum(indep_nrmse) / m,
        "fit_seconds": secs,
    }


def noisy_accuracy(problem: NoisyProblem, seed: int = 0, max_iter: int = 150) -> dict[str, float]:
    """Fit on noisy samples, score against the noise-free truth on a dense grid."""
    x, y = problem.sample(seed)
    xt = problem.grid()
    model, secs = _timed(lambda: GPR().fit(x, y, max_iter=max_iter))
    return {
        **error_metrics(model.predict(xt)["fmu"], problem.truth(xt[:, 0])),
        "n_train": problem.num_train,
        "fit_seconds": secs,
    }
