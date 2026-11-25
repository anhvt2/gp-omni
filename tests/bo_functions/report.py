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

MANUSCRIPT_AUDIT = """## Manuscript verification

Sources were read only from `roglocal:/home/anhvt89/Templates/journal/`:
`asyncParBO/asyncPar_1Feb22.tex`, `scalable3BO/scalable3BO_28May21.tex`,
`sMfConsBO/finalSubmission/sMfConsBO_7Mar20.tex`,
`sMoConsBO/asme_final_submission/sMoConsBO_25Aug21.tex`, and
`monotonicGP/asme-final-submission/monotonicGP_25Aug22.tex`.
Version check on 2026-11-24: recursively enumerated `.tex` files in these five manuscript
folders and compared submission/archive copies with the latest dates embedded in filenames
(rather than filesystem modification times). The benchmark equations agree after ignoring
whitespace and explicit multiplication marks; async BO only reorders Perm. Relevant benchmark
domains and material/noise parameters are also unchanged.

| Manuscript | Submitted/archive copy checked | Latest dated source checked |
|---|---|---|
| Async/parallel BO | `submitted-3/asyncPar_23Apr21.tex` (also arxiv copy) | `asyncPar_1Feb22.tex` |
| Scalable BO | No submission folder found; `arxiv/scalable3BO_30Mar21.tex` checked | `scalable3BO_28May21.tex` |
| Multi-fidelity BO | `finalSubmission/sMfConsBO_7Mar20.tex` | `sMfConsBO_26May20.tex` |
| Multi-objective BO | `asme_final_submission/sMoConsBO_25Aug21.tex` | Root and arxiv `sMoConsBO_25Aug21.tex` |
| Monotonic GP | `asme-final-submission/monotonicGP_25Aug22.tex` | Root and arxiv `monotonicGP_25Aug22.tex` |

The earlier multi-fidelity audit used the final-submission copy without checking the later
May revision. That later revision is now checked: its benchmark equations, domains, and
material constants match. The discrepancies documented below persist in the checked
submission and latest copies; no numerical implementation or accuracy result changes.

The manuscript is the primary definition. A difference from a standard benchmark is not
by itself an error. This audit concerns the implemented functions, not all claims in the papers.

| Definition | Verification and implementation |
|---|---|
| Modified ZDT1/2/3 | Preserve the three printed g scalings. They are different, well-defined variants. ZDT1/2 use [-1,1]^d; ZDT3 uses its subsection's explicit [0,1]^d, despite the preceding centered-domain discussion. |
| DTLZ2–6 scale | Preserve the printed 0.5 multiplier. A common positive scale is valid and preserves dominance. DTLZ2/3 fronts have radius 0.5, not 1. |
| DTLZ4 powers | Preserve alpha on cosine factors only, exactly as printed. This is a valid function differing from standard DTLZ4; a spherical front is not asserted. Alpha=100 is an assumed parameter, unspecified in the displayed manuscript definition. |
| Eggholder | Confirmed inconsistency: without square roots, the printed formula gives +615.7061 at the stated minimizer, versus the stated -959.6407. Retain square roots, giving -959.64066, consistent with the manuscript's cited [SFU definition](https://www.sfu.ca/~ssurjano/egg.html). |
| Hartmann4/6 | Printed reversed inner limits (j=4..3 and j=6..3) produce empty sums and constant functions. Retain j=1..4 and j=1..6. The six-coordinate minimizer also resolves the printed 4-D domain for Hartmann6. See the cited [Hartmann4](https://www.sfu.ca/~ssurjano/hart4.html) and [Hartmann6](https://www.sfu.ca/~ssurjano/hart6.html) definitions. |
| DTLZ1 tail | The introductory convention explicitly says g uses k=d-M+1 tail variables, contradicting the displayed sum over all d. Follow that explicit convention and the cited [DTLZ authors' implementation](https://sop.tik.ee.ethz.ch/pisa/variators/dtlz/index.html). For d=4,M=3,x=(.25,.5,.5,.5), the printed all-variable g is 206.25 instead of the tail-optimum value 0. |
| DTLZ5/6 angles | The display leaves theta_1 undefined and multiplies already radian-valued theta by pi/2 again. Resolve these using the cited DTLZ angle convention: theta_1=pi*x_1/2; later angles=pi*(1+2*g*x_i)/(4*(1+g)). Preserve the manuscript's 0.5 objective scale. This is an explicit interpretation, not a literal transcription. |
| Hall–Petch noise | The printed variance is 2.2e10*d^3 with d numerically in micrometres in the truth equation. Follow it literally by default: sd at 350 micrometres is about 9.71e8. The alternative metre conversion gives sd about 0.971 and is reported separately as hall_petch_noise_in_metres; the text does not establish that conversion, so it is not silently substituted. The literal noise overwhelms the truth; its test checks finiteness, with no accuracy claim. |
| Other functions | ZDT1/2/3/4/6, Forrester, Currin, borehole levels, welded-beam costs and constraints, logistic truth/noise, and the remaining single-objective formulas follow the displayed definitions. Welded-beam constraints retain the printed J and buckling-load factors, even where the classic benchmark differs. Perm beta=10 and Michalewicz m=10 are assumed defaults. |

These are surrogate-regression experiments rather than reproductions of BO runs. Training
budgets and most free dimensions are reduced for routine testing; the dimensions in each
table are authoritative. The additive cases use d=20, not the scalable paper's D=10,000.
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
        "# GPR accuracy on the manuscript test functions\n",
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
        MANUSCRIPT_AUDIT,
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
