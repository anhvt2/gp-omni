"""Generate the public campaign report and convergence figure (requires matplotlib).

PYTHONPATH=tests python -m bo_functions.campaign_report
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import numpy as np
import torch

from .campaign import INTERMEDIATE_CASES
from .campaign_metrics import average_regret, cumulative_regret
from .plotting import plot_campaign_curves

TESTS = Path(__file__).resolve().parents[1]
RESULTS = TESTS / "results"


def build() -> None:
    runs = {
        (r["case"], r["seed"]): r for r in json.loads((RESULTS / "bo_campaigns.json").read_text())
    }
    assert len(runs) == 2 * len(INTERMEDIATE_CASES), "expected two seeds per problem"
    for case in INTERMEDIATE_CASES:
        for seed in (0, 1):
            r = runs[case.name, seed]
            for flag in (
                "all_complete",
                "all_feasible",
                "in_bounds",
                "finite",
                "stored_values_match",
                "returned_value_matches",
                "returned_is_best",
                "random_initial_matches",
            ):
                assert r[flag], (case.name, seed, flag)
            history = np.asarray(r["regret_history"]) + r["global_min"]
            assert len(history) == r["n_evals"] == r["requested_evals"]
            assert (np.diff(history) <= 0).all()
            assert np.isclose(history[-1], r["best_value"])
            evaluations = np.asarray(r["evaluation_values"])
            assert evaluations.shape == history.shape
            assert r["global_min"] == case.function.global_min
            assert r["n_init"] == case.function.dim, "rerun campaigns with d initial points"
            assert np.isfinite(evaluations).all()
            assert np.allclose(np.minimum.accumulate(evaluations), history)
            r["objective_history"] = history.tolist()
            r["cumulative_regret_history"] = cumulative_regret(
                evaluations, r["global_min"]
            ).tolist()
            r["average_regret_history"] = average_regret(evaluations, r["global_min"]).tolist()
            changes = np.flatnonzero(np.diff(history) < 0)
            r["last_improvement_eval"] = int(changes[-1] + 2) if len(changes) else 1
            bounds = np.asarray(r["bounds"])
            random_x = bounds[0] + np.random.default_rng(seed).random(
                (r["n_evals"], case.function.dim)
            ) * (bounds[1] - bounds[0])
            random_y = case.function(torch.as_tensor(random_x)).numpy()
            r["random_objective_history"] = np.minimum.accumulate(random_y).tolist()
            r["random_cumulative_regret_history"] = cumulative_regret(
                random_y, r["global_min"]
            ).tolist()
            r["random_average_regret_history"] = average_regret(random_y, r["global_min"]).tolist()
            assert np.isclose(min(random_y) - r["global_min"], r["random_regret"])
            assert np.allclose(
                r["cumulative_regret_history"][: r["n_init"]],
                r["random_cumulative_regret_history"][: r["n_init"]],
                rtol=1e-7,
                atol=1e-5,
            )
    ordered = [runs[case.name, seed] for case in INTERMEDIATE_CASES for seed in (0, 1)]
    (RESULTS / "bo_campaigns.json").write_text(json.dumps(ordered, indent=2) + "\n")

    labels = {
        case.name: {
            "hartmann3": "Hartmann",
            "hartmann4": "Hartmann",
            "hartmann6": "Hartmann",
            "sum_squares": "Weighted quadratic",
            "sum_pow": "Sum of powers",
        }.get(case.function.name, case.function.name.title())
        for case in INTERMEDIATE_CASES
    }
    plot_campaign_curves(
        ordered,
        RESULTS / "bo_convergence",
        labels=labels,
        symlog_cases=("rosenbrock-5d", "trid-9d"),
        show_initialization=True,
    )
    table = []
    for case in INTERMEDIATE_CASES:
        records = [runs[case.name, seed] for seed in (0, 1)]
        n = records[0]["n_evals"]
        final = statistics.median(r["best_value"] for r in records)
        random = statistics.median(r["random_regret"] + r["global_min"] for r in records)
        last = " / ".join(str(r["last_improvement_eval"]) for r in records)
        table.append(
            f"| {labels[case.name]} {case.function.dim}D | {n} | {case.function.global_min:.6g} | "
            f"{final:.6g} | {random:.6g} | {last} |"
        )

    report = """# Bayesian optimization: convergence in 3–9 dimensions

Nine problems (3–9 dimensions), two seeds per problem. Each row pairs best evaluated
objective convergence with average regret against evaluation count; lower is better.

![Best objective and average regret versus evaluations](results/bo_convergence.png)

Left: best objective. Right: average regret, **R(T)/T = (1/T) Σᵢ₌₁ᵀ [f(xᵢ) − f*]**, including
initialization and all subsequent evaluations. It averages individual evaluation gaps for T ≥ 1.
Blue: EI; gray: matched random search. Lines are medians; shading is the two-run range.
Black dotted lines mark known minima; purple vertical lines mark completion of initialization.
Rosenbrock and Trid objective axes use symmetric logs;
average regret axes are linear. Curves stop at their actual budgets.

| Problem | Evals/run | Global minimum | BO final | Random final | Last BO improvement, seeds 0 / 1 |
|---|---|---|---|---|---|
""" + "\n".join(table)
    report += """

Final values are medians. Last-improvement counts include initialization; flat tails
indicate no further incumbent improvement.

Full benchmark domains; exact GPR (ARD RBF); EI; one point per `batch_bo` batch. Seeds 0 and 1 share
uniform initial designs of exactly d points with random search (d is the input dimension).
GP fitting: 75 optimizer iterations;
acquisition search: 150 evaluations per CMA-ES restart, three restarts.

[Plotting utility](bo_functions/plotting.py) · [Raw results and histories](results/bo_campaigns.json) · [Vector figure](results/bo_convergence.svg)

Reproduce from the repository root:

```sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=tests
uv run --no-sync python -m bo_functions.campaign --suite intermediate --seeds 0 1 \\
  --output tests/results/bo_campaigns.json
uv run --no-sync --with matplotlib python -m bo_functions.campaign_report
```

Use `--max-evals` for budgets up to 1,000 evaluations per run. Recorded 2026-11-25 on CPU;
numerical libraries can affect optimizer trajectories.
"""
    better = sum(r["regret"] < r["random_regret"] for r in ordered)
    report = report.replace(
        "![Best objective and average regret versus evaluations]",
        f"EI finishes below random search in {better}/{len(ordered)} runs. Several curves flatten "
        "above the global minimum.\n\n![Best objective and average regret versus evaluations]",
    )
    (TESTS / "bo_campaigns.md").write_text(report)
    print(f"Wrote report and curves for {len(ordered)} campaigns.")


if __name__ == "__main__":
    build()
