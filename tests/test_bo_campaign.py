"""End-to-end convergence through the public BO loop, evaluator and job database.

Run with OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 for small-GP
performance. Harder landscapes are available via ``python -m bo_functions.campaign``;
see that module's --include-stress option. Tolerances are in original objective units.
"""

import numpy as np
import pytest
from bo_functions.campaign import CASES, SEEDS, run_campaign


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("case", [c for c in CASES if not c.stress], ids=lambda c: c.name)
def test_bo_campaign_reaches_known_global_minimum(case, seed, tmp_path):
    result = run_campaign(case, tmp_path, seed)
    diagnostic = (
        f"{case.name}, seed={seed}, evaluations={result['n_evals']}, "
        f"initial regret={result['initial_regret']:.6g}, "
        f"final regret={result['regret']:.6g}, tolerance={case.tolerance:g}, "
        f"best x={result['best_x']}"
    )
    assert result["n_evals"] == case.n_evals, diagnostic
    for invariant in (
        "all_complete",
        "all_feasible",
        "in_bounds",
        "finite",
        "stored_values_match",
        "returned_value_matches",
        "returned_is_best",
    ):
        assert result[invariant], f"{invariant}: {diagnostic}"
    history = np.asarray(result["regret_history"])
    assert len(history) == case.n_evals
    # Published minima can be rounded (e.g. six-hump camel).
    assert history.min() >= -2e-4 * max(1, abs(result["global_min"])), diagnostic
    assert (np.diff(history) <= 0).all(), diagnostic
    assert result["initial_regret"] > case.tolerance, diagnostic
    assert result["regret"] < result["initial_regret"], diagnostic
    assert result["converged"], diagnostic
