"""Run real, file-backed BO campaigns against known scalar global minima.

From the repository root::

    PYTHONPATH=tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        python -m bo_functions.campaign --seeds 0 1 2

Inputs use the full benchmark box, scaled by its width for the GP. The public BO
loop maximises, so the evaluator returns -f(x). No optimum locations or truth
samples are supplied to the optimiser. Only completed evaluations are scored.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from time import monotonic

import numpy as np
import torch

import gp_omni  # noqa: F401 (float64)
from gp_omni.opt.parallel import jobs
from gp_omni.opt.parallel.batchbo import batch_bo, batch_bo_config

from .common import BenchmarkFunction
from .multi_fidelity import forrester
from .single_objective import fixed_dim_functions, free_dim_function, scalable_function


@dataclass(frozen=True)
class CampaignCase:
    function: BenchmarkFunction
    n_init: int
    n_evals: int
    tolerance: float  # absolute simple regret, in original objective units
    y_scale: float = 1.0
    stress: bool = False

    @property
    def name(self) -> str:
        return f"{self.function.name}-{self.function.dim}d"


FIXED = fixed_dim_functions()
FORRESTER = forrester()
CASES = (
    CampaignCase(
        BenchmarkFunction(
            "forrester",
            1,
            FORRESTER.lower,
            FORRESTER.upper,
            FORRESTER.levels[-1],
            FORRESTER.global_min,
        ),
        4,
        20,
        0.01,
        10,
    ),
    CampaignCase(free_dim_function("sphere", 2), 6, 50, 0.05, 25),
    CampaignCase(free_dim_function("sum_squares", 2), 6, 40, 0.05, 50),
    CampaignCase(free_dim_function("sum_squares", 3), 8, 60, 0.1, 100, stress=True),
    CampaignCase(free_dim_function("rotated_hyper_ellipsoid", 2), 6, 60, 5.0, 10_000),
    CampaignCase(free_dim_function("trid", 2), 6, 30, 0.05, 50),
    CampaignCase(free_dim_function("sum_pow", 2), 6, 30, 0.01),
    CampaignCase(scalable_function("squared_sum", 3), 8, 35, 0.01, 9),
    CampaignCase(FIXED["three_hump_camel"], 8, 40, 0.05, 1000),
    CampaignCase(FIXED["six_hump_camel"], 8, 40, 0.05, 50),
    CampaignCase(FIXED["hartmann3"], 8, 40, 0.1, 4),
    CampaignCase(FIXED["hartmann6"], 12, 70, 0.3, 4, stress=True),
    CampaignCase(free_dim_function("rosenbrock", 2), 8, 60, 1.0, 100_000, stress=True),
    CampaignCase(free_dim_function("ackley", 2), 8, 60, 0.5, 20, stress=True),
    CampaignCase(FIXED["eggholder"], 10, 60, 10.0, 1000, stress=True),
)
# Exploratory profile spanning every dimension from 3 through 9. Targets are
# fixed before running; misses remain in the report rather than becoming guards.
INTERMEDIATE_CASES = (
    CampaignCase(FIXED["hartmann3"], 10, 100, 0.1, 4),
    CampaignCase(FIXED["hartmann4"], 12, 120, 0.1, 4),
    CampaignCase(free_dim_function("ackley", 5), 15, 160, 0.5, 20),
    CampaignCase(free_dim_function("rosenbrock", 5), 15, 160, 1.0, 100_000),
    CampaignCase(FIXED["hartmann6"], 18, 180, 0.1, 4),
    CampaignCase(free_dim_function("sphere", 7), 21, 140, 0.1, 100),
    CampaignCase(free_dim_function("sum_squares", 8), 24, 300, 0.1, 500),
    CampaignCase(free_dim_function("trid", 9), 27, 300, 0.1, 50_000),
    CampaignCase(free_dim_function("sum_pow", 9), 27, 160, 0.01, 3),
)
SEEDS = (0, 1, 2)


def run_campaign(case: CampaignCase, root: Path, seed: int = 0) -> dict:
    """Use an empty job directory; return truth, regret history and budget diagnostics."""
    fn = case.function
    if fn.global_min is None:
        raise ValueError(f"{fn.name} has no known global minimum")
    if root.exists() and any(root.iterdir()):
        raise ValueError("campaign root must be empty to avoid resuming old evaluations")
    bounds = fn.bounds.numpy()
    width = bounds[1] - bounds[0]
    config = batch_bo_config(
        model_name="campaign",
        x_lb=bounds[0] / width,
        x_ub=bounds[1] / width,
        x_scale=width,
        y_scale=case.y_scale,
        exploit_size=1,
        explore_size=0,
        explore_classifier_size=0,
        acquisition="EI",
        num_init_point=case.n_init,
        max_iter=case.n_evals,
        gp_max_iter=75,
        cma_max_evals=150,
        check_time=0.0001,
        seed=seed,
    )

    def evaluate(x):
        return -fn(torch.as_tensor(x).reshape(1, -1)).item(), True

    started = monotonic()
    # Preserve the caller's Torch RNG; seed any stochastic GP fitting too.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        loop = batch_bo(root, config, evaluate=evaluate)
        try:
            best_x, best_y = loop.run()
        finally:
            loop.pool.shutdown(wait=True)
    x, y, feasible, complete = jobs.update_db(root)
    truth = fn(torch.as_tensor(x)).numpy()
    best_values = np.minimum.accumulate(truth)
    best_value = fn(torch.as_tensor(best_x).reshape(1, -1)).item()
    random_x = bounds[0] + np.random.default_rng(seed).random((case.n_evals, fn.dim)) * width
    random_truth = fn(torch.as_tensor(random_x)).numpy()
    return {
        "case": case.name,
        "dimension": fn.dim,
        "bounds": bounds.tolist(),
        "y_scale": case.y_scale,
        "acquisition": config.acquisition,
        "gp_max_iter": config.gp_max_iter,
        "cma_max_evals": config.cma_max_evals,
        "requested_evals": case.n_evals,
        "random_regret": float(random_truth.min() - fn.global_min),
        "random_regret_history": (np.minimum.accumulate(random_truth) - fn.global_min).tolist(),
        "random_initial_matches": bool(np.allclose(random_x[: case.n_init], x[: case.n_init])),
        "seed": seed,
        "n_init": case.n_init,
        "n_evals": len(y),
        "all_complete": bool((complete == 1).all()),
        "all_feasible": bool((feasible == 1).all()),
        "in_bounds": bool(((x >= bounds[0]) & (x <= bounds[1])).all()),
        "finite": bool(np.isfinite(x).all() and np.isfinite(truth).all()),
        "stored_values_match": bool(np.allclose(-y, truth, atol=1e-5, rtol=1e-7)),
        "returned_value_matches": bool(np.isclose(-best_y, best_value, atol=1e-5, rtol=1e-7)),
        "returned_is_best": bool(np.isclose(best_value, truth.min(), atol=1e-5, rtol=1e-7)),
        "best_x": best_x.tolist(),
        "best_value": best_value,
        "global_min": fn.global_min,
        "initial_regret": float(truth[: case.n_init].min() - fn.global_min),
        "regret": best_value - fn.global_min,
        "regret_history": (best_values - fn.global_min).tolist(),
        "evaluation_values": truth.tolist(),
        "tolerance": case.tolerance,
        "converged": bool(best_value - fn.global_min <= case.tolerance),
        "seconds": monotonic() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--suite", choices=("standard", "intermediate"), default="standard")
    parser.add_argument(
        "--cases", nargs="+", choices=sorted({c.name for c in (*CASES, *INTERMEDIATE_CASES)})
    )
    parser.add_argument("--include-stress", action="store_true")
    parser.add_argument("--max-evals", type=int, help="override total budget (at most 1000)")
    parser.add_argument(
        "--strict", action="store_true", help="exit nonzero if any target is missed"
    )
    parser.add_argument(
        "--output", type=Path, help="write full results and regret histories as JSON"
    )
    args = parser.parse_args()
    suite = INTERMEDIATE_CASES if args.suite == "intermediate" else CASES
    if args.cases and set(args.cases) - {c.name for c in suite}:
        parser.error("selected cases must belong to the selected suite")
    selected = (
        [c for c in suite if c.name in args.cases]
        if args.cases
        else [c for c in suite if args.include_stress or not c.stress]
    )
    if args.max_evals is not None:
        if not 1 <= args.max_evals <= 1000 or any(args.max_evals <= c.n_init for c in selected):
            parser.error("--max-evals must exceed initialization and be at most 1000")
        selected = [replace(c, n_evals=args.max_evals) for c in selected]
    results = []
    print("case seed evals best_value global_min random_best seconds", flush=True)
    for case in selected:
        for seed in args.seeds:
            with tempfile.TemporaryDirectory() as directory:
                result = run_campaign(case, Path(directory), seed)
            results.append(result)
            print(
                f"{case.name} {seed} {result['n_evals']} "
                f"{result['best_value']:.6g} {result['global_min']:.6g} "
                f"{result['random_regret'] + result['global_min']:.6g} "
                f"{result['seconds']:.1f}",
                flush=True,
            )
            if args.output:
                args.output.write_text(json.dumps(results, indent=2) + "\n")

    if args.strict and any(not r["converged"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
