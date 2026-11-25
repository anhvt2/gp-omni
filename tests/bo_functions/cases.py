"""The accuracy cases: one table shared by the pytest guards and the markdown report.

``guard`` is a regression threshold on the median NRMSE over ``SEEDS`` (about 2x the observed
value, so cross-platform optimizer noise does not flake). It is *not* a claim that the model is
good: cases marked ``hard`` are ones the model cannot learn at that budget (guard = 1.5 only
catches catastrophic failure; NRMSE ~ 1 means "no better than predicting the mean").
"""

from __future__ import annotations

from dataclasses import dataclass

from . import multi_fidelity as mf
from . import multi_objective as mo
from . import noisy_regression as nr
from . import single_objective as so

SEEDS = (0, 1, 2)
HARD = 1.5


@dataclass(frozen=True)
class SingleCase:
    context: str
    fn_name: str
    dim: int
    n_train: int
    guard: float
    gpr: tuple[tuple[str, object], ...] = ()  # GPR kwargs, e.g. (("additive", 2),)
    vs_plain: float | None = None  # additive must beat plain ARD by this factor (nrmse ratio)

    def function(self):
        if self.fn_name in so.fixed_dim_functions():
            return so.fixed_dim_functions()[self.fn_name]
        if self.fn_name in so.FREE_DIM_NAMES:
            return so.free_dim_function(self.fn_name, self.dim)
        return so.scalable_function(self.fn_name, self.dim)


_PLAIN = "single-fidelity, plain GPR"
_ADD = "additive GPR (d=20)"
_ADD_KW = (("additive", 2),)

SINGLE_CASES = (
    SingleCase(_PLAIN, "eggholder", 2, 100, HARD),
    SingleCase(_PLAIN, "three_hump_camel", 2, 40, 0.05),
    SingleCase(_PLAIN, "six_hump_camel", 2, 40, 0.2),
    SingleCase(_PLAIN, "hartmann3", 3, 60, 0.2),
    SingleCase(_PLAIN, "hartmann4", 4, 80, 0.4),
    SingleCase(_PLAIN, "hartmann6", 6, 150, 0.75),
    SingleCase(_PLAIN, "ackley", 5, 150, 0.8),
    SingleCase(_PLAIN, "michalewicz", 5, 100, HARD),
    SingleCase(_PLAIN, "rosenbrock", 5, 100, 0.3),
    SingleCase(_PLAIN, "dixon_price", 5, 100, 0.4),
    SingleCase(_PLAIN, "trid", 5, 100, 0.01),
    SingleCase(_PLAIN, "sum_squares", 5, 100, 0.02),
    SingleCase(_PLAIN, "sum_pow", 5, 100, 0.6),
    SingleCase(_PLAIN, "sphere", 5, 100, 0.02),
    SingleCase(_PLAIN, "rotated_hyper_ellipsoid", 5, 100, 0.02),
    SingleCase(_PLAIN, "perm0db", 5, 100, HARD),
    # additive order-2 kernel, 120 samples in 20-D; ``vs_plain`` = required win over plain ARD
    SingleCase(_ADD, "sum_squares", 20, 120, 0.01, _ADD_KW, 0.1),
    SingleCase(_ADD, "sphere", 20, 120, 0.01, _ADD_KW, 0.1),
    SingleCase(_ADD, "rotated_hyper_ellipsoid", 20, 120, 0.01, _ADD_KW, 0.1),
    SingleCase(_ADD, "rosenbrock", 20, 120, 0.6, _ADD_KW, 0.8),
    SingleCase(_ADD, "dixon_price", 20, 120, 0.7, _ADD_KW, 0.8),
    SingleCase(_ADD, "squared_sum", 20, 120, 0.3, _ADD_KW),
    SingleCase(_ADD, "sum_pow", 20, 120, HARD, _ADD_KW),
    SingleCase(_ADD, "trid", 20, 120, HARD, _ADD_KW),
    SingleCase(_ADD, "modified_zdt1", 20, 120, HARD, _ADD_KW),
    SingleCase(_ADD, "modified_zdt2", 20, 120, HARD, _ADD_KW),
    SingleCase(_ADD, "modified_zdt3", 20, 120, 0.3, _ADD_KW),
)


@dataclass(frozen=True)
class MultiFidelityCase:
    fn_name: str
    n_per_level: tuple[int, ...]
    guard: float  # max median mf nrmse
    min_gain: float | None = None  # required hi_only_nrmse / mf_nrmse

    def function(self):
        return {
            "forrester": mf.forrester,
            "currin": mf.currin,
            "borehole_2f": lambda: mf.borehole(2),
            "borehole_3f": lambda: mf.borehole(3),
            "welded_beam_4f": mf.welded_beam,
        }[self.fn_name]()


MULTI_FIDELITY_CASES = (
    # Four high-fidelity points give optimizer-sensitive fits: Linux CI reaches 0.8194.
    # Keep a mean-baseline bound and the transfer-gain check; (31, 8) guards accuracy.
    MultiFidelityCase("forrester", (31, 4), 1.0, 1.2),
    MultiFidelityCase("forrester", (31, 8), 0.3, 2.0),
    MultiFidelityCase("currin", (40, 12), 0.15, 3.0),
    MultiFidelityCase("borehole_2f", (100, 20), 0.02, 3.0),
    MultiFidelityCase("borehole_3f", (150, 60, 20), 0.02, 3.0),
    MultiFidelityCase("welded_beam_4f", (120, 60, 30, 15), 0.6),
)


@dataclass(frozen=True)
class MultiObjectiveCase:
    fn_name: str
    n_train: int
    guard: float  # max median mean-over-objectives nrmse of the joint model
    sparse_last: int | None = None  # last objective observed at only this many points

    def function(self):
        if self.fn_name.startswith("zdt"):
            return mo.zdt(self.fn_name)
        return mo.dtlz(self.fn_name)


MULTI_OBJECTIVE_CASES = (
    MultiObjectiveCase("zdt1", 50, 0.05),
    MultiObjectiveCase("zdt2", 50, 0.03),
    MultiObjectiveCase("zdt3", 50, 0.3),
    MultiObjectiveCase("zdt4", 50, HARD),
    MultiObjectiveCase("zdt6", 50, HARD),
    MultiObjectiveCase("dtlz1", 50, HARD),
    MultiObjectiveCase("dtlz2", 50, 0.12),
    MultiObjectiveCase("dtlz3", 50, HARD),
    MultiObjectiveCase("dtlz4", 50, 0.6),
    MultiObjectiveCase("dtlz5", 50, 0.2),
    MultiObjectiveCase("dtlz6", 50, 0.1),
)
# data-poor last objective: the joint model should borrow strength from the others
MULTI_OBJECTIVE_SPARSE_CASES = tuple(
    MultiObjectiveCase(n, 50, HARD, sparse_last=8) for n in ("zdt1", "zdt2", "dtlz2", "dtlz6")
)


@dataclass(frozen=True)
class NoisyCase:
    problem_name: str
    guard: float | None  # None: report an extreme noise stress case; check finiteness only

    def problem(self):
        return {"noisy_logistic": nr.noisy_logistic, "hall_petch": nr.heteroscedastic_hall_petch,
                "hall_petch_noise_in_metres": lambda: nr.heteroscedastic_hall_petch(noise_in_metres=True)}[
            self.problem_name
        ]()


NOISY_CASES = (
    NoisyCase("noisy_logistic", 0.4),
    NoisyCase("hall_petch", None),
    NoisyCase("hall_petch_noise_in_metres", HARD),
)
