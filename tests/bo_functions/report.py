"""Print (or write) the accuracy table: ``python -m bo_functions.report [out.md]`` from ``tests/``.

Report medians over five training designs against the noise-free truth. Deterministic
functions use 1000 held-out Sobol points; noisy problems use a 200-point grid.
Set OMP_NUM_THREADS=1, MKL_NUM_THREADS=1, and OPENBLAS_NUM_THREADS=1 when regenerating
the checked-in report. Optimizer results can vary with numerical libraries and thread settings.
"""

from __future__ import annotations

import statistics
import sys
import warnings

from . import accuracy as acc
from . import cases as C

REPORT_SEEDS = (0, 1, 2, 3, 4)

BENCHMARK_CONVENTIONS = """## Benchmark definitions and conventions

The suite includes scalar analytical functions, modified ZDT objectives, multi-fidelity
engineering functions, ZDT and scaled DTLZ objectives, and noisy regression targets.
Bounds, dimensions, fidelity ordering, and noise models are defined with each benchmark.
The following conventions matter when comparing results with other implementations.

| Family | Definition used here |
|---|---|
| Modified ZDT1/2/3 | Three distinct g scalings: 1+9*(sum(tail)/(d-1))^2, 1+(9*sum(tail))^2, and 1+9*sum(tail)^2. ZDT1/2 use [-1,1]^d; ZDT3 uses [0,1]^d. |
| DTLZ2–6 scale | All objectives have a 0.5 multiplier relative to the usual scale. DTLZ2/3 fronts have radius 0.5. |
| DTLZ4 powers | Alpha applies to cosine factors only, with default alpha=100. This differs from standard DTLZ4; no spherical front is assumed. |
| Eggholder | Uses square roots inside both sine arguments, giving approximately -959.6407 at (512, 404.2319). |
| Hartmann4/6 | Sum over four or six input coordinates. Hartmann4 uses the first four columns of the six-dimensional parameter tables and the (1.1-sum)/0.839 rescale. |
| DTLZ1 tail | The distance function g uses the final k=d-M+1 variables. |
| DTLZ5/6 angles | Angles are in radians: theta_1=pi*x_1/2; later angles=pi*(1+2*g*x_i)/(4*(1+g)). Objectives retain the 0.5 scale. |
| Hall–Petch noise | Default variance is 2.2e10*d^3 with d in micrometres, giving sd about 9.71e8 at d=350. This extreme noise case checks finiteness only. The separate hall_petch_noise_in_metres variant converts d to metres in the noise term, giving sd about 0.971. |
| Welded beam | Material-dependent costs and five constraints use this suite's J and buckling-load factors, which differ from the classic welded-beam constraints. The classic feasible optimum is not asserted for this variant. |
| Scalar defaults | Perm uses beta=10; Michalewicz uses m=10. Scalable functions accept an explicit input dimension. |

These experiments measure surrogate regression against analytical truth. Each table states
the training budget and dimension; the additive cases use d=20. Accuracy labels describe
these particular data budgets and designs, and do not establish BO convergence or guarantee
that combining fidelities or outputs improves predictions.
"""


def med(values) -> float:
    return statistics.median(values)


def fmt(x: float) -> str:
    return f"{x:.2g}" if x < 0.1 else f"{x:.2f}"


def verdict(nrmse: float) -> str:
    return "accurate" if nrmse < 0.1 else "rough" if nrmse < 0.5 else "poor" if nrmse < 0.9 else "fails"


def table(headers, rows) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def single_section(context: str, with_plain: bool) -> str:
    rows = []
    for c in (c for c in C.SINGLE_CASES if c.context == context):
        fn = c.function()
        runs = [acc.single_fidelity_accuracy(fn, c.n_train, seed=s, **dict(c.gpr)) for s in REPORT_SEEDS]
        nrmse, r2 = med([r["nrmse"] for r in runs]), med([r["r2"] for r in runs])
        row = [c.fn_name, c.dim, c.n_train, fmt(nrmse), fmt(r2) if r2 > 0 else f"{r2:.2f}"]
        if with_plain:
            plain = med([acc.single_fidelity_accuracy(fn, c.n_train, seed=s)["nrmse"] for s in REPORT_SEEDS])
            row += [fmt(plain), verdict(nrmse)]
        else:
            row += [verdict(nrmse)]
        rows.append(row)
    head = ["function", "d", "n_train", "nrmse", "r2"] + (["plain ARD nrmse"] if with_plain else []) + ["verdict"]
    return table(head, rows)


def multi_fidelity_section() -> str:
    rows = []
    for c in C.MULTI_FIDELITY_CASES:
        fn = c.function()
        runs = [acc.multi_fidelity_accuracy(fn, c.n_per_level, seed=s) for s in REPORT_SEEDS]
        mf, hi = med([r["mf_nrmse"] for r in runs]), med([r["hi_only_nrmse"] for r in runs])
        rows.append([c.fn_name, fn.dim, "/".join(map(str, c.n_per_level)), fmt(mf), fmt(hi),
                     f"{hi / mf:.1f}x", verdict(mf)])
    return table(["function", "d", "n per level (low..high)", "multi-fidelity nrmse",
                  "high-fidelity-only nrmse", "gain", "verdict"], rows)


def multi_objective_section(cases, sparse: bool) -> str:
    rows = []
    for c in cases:
        fn = c.function()
        runs = [acc.multi_objective_accuracy(fn, c.n_train, seed=s, sparse_last=c.sparse_last)
                for s in REPORT_SEEDS]
        m = fn.num_objectives
        joint = [med([r["joint_nrmse"][k] for r in runs]) for k in range(m)]
        indep = [med([r["independent_nrmse"][k] for r in runs]) for k in range(m)]
        if sparse:
            rows.append([c.fn_name, fn.dim, m, f"{c.n_train} (last: {c.sparse_last})",
                         fmt(joint[-1]), fmt(indep[-1])])
        else:
            rows.append([c.fn_name, fn.dim, m, c.n_train, " / ".join(fmt(v) for v in joint),
                         " / ".join(fmt(v) for v in indep), verdict(sum(joint) / m)])
    if sparse:
        return table(["function", "d", "M", "n_train", "joint nrmse (data-poor objective)",
                      "independent nrmse (data-poor objective)"], rows)
    return table(["function", "d", "M", "n_train", "joint multi-output nrmse per objective",
                  "independent nrmse per objective", "verdict (joint mean)"], rows)


def noisy_section() -> str:
    rows = []
    for c in C.NOISY_CASES:
        p = c.problem()
        runs = [acc.noisy_accuracy(p, seed=s) for s in REPORT_SEEDS]
        nrmse = med([r["nrmse"] for r in runs])
        rows.append([c.problem_name, p.num_train, fmt(nrmse), fmt(max(r["nrmse"] for r in runs)), verdict(nrmse)])
    return table(["problem", "n_train", "nrmse (median)", "nrmse (worst of 5)", "verdict"], rows)


def build() -> str:
    parts = [
        "# GPR accuracy on analytical benchmarks\n",
        (
            "Generated by `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "
            "python -m bo_functions.report` (from `tests/`). Every number is the "
            f"median over {len(REPORT_SEEDS)} training designs (seeds {REPORT_SEEDS}) of "
            "**NRMSE = RMSE / std(truth)** of the posterior mean against the *true noise-free* "
            "function (0 = perfect, ~1 = no better than predicting the mean). Deterministic "
            "cases use scrambled-Sobol training designs and 1000 held-out Sobol points "
            "(seed 12345). Noisy cases use uniform-random training locations and Gaussian noise, "
            "scored on a 200-point grid including endpoints. Fits use at most 150 iterations. "
            "Optimizer results can vary with numerical libraries and thread settings. "
            "Verdict: accurate < 0.1 <= rough < 0.5 <= poor < 0.9 <= fails.\n"
        ),
        BENCHMARK_CONVENTIONS,
        "## Single-fidelity, single-objective (plain GPR)\n", single_section(C._PLAIN, False),
        "\n## Additive GPR in 20-D (order-2 kernel, 120 samples)\n",
        single_section(C._ADD, True),
        "\n## Multi-fidelity (AR1 `GPR(num_fidelities=L)`; scored at the highest fidelity)\n",
        multi_fidelity_section(),
        "\n## Multi-objective, equal data per objective (`GPR(num_outputs=M)` vs one GPR per objective)\n",
        multi_objective_section(C.MULTI_OBJECTIVE_CASES, False),
        "\n## Multi-objective, last objective data-poor\n",
        multi_objective_section(C.MULTI_OBJECTIVE_SPARSE_CASES, True),
        "\n## Noisy regression (ordinary GPR, scored against the noise-free truth)\n", noisy_section(),
    ]
    return "\n".join(parts) + "\n"


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    text = build()
    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as fh:
            fh.write(text)
    print(text)
