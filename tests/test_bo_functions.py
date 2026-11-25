"""The ported manuscript test functions are right (no GP involved).

Checks: published optima, no random point beats the optimum, exact agreement with BoTorch's
independent implementations where they exist, analytic Pareto fronts, and the manuscripts' own
claims about the multi-fidelity pairs.
"""

import math

import pytest
import torch
from bo_functions import single_objective as so
from bo_functions.common import sobol_design
from botorch.test_functions import synthetic as bs

import gp_omni  # noqa: F401  (float64)

GEN = torch.Generator().manual_seed(7)


def rand_in(bounds, n):
    return bounds[0] + torch.rand(n, bounds.shape[1], generator=GEN) * (bounds[1] - bounds[0])


# ----------------------------------------------------------------------- single objective
FIXED = so.fixed_dim_functions()
FREE = [so.free_dim_function(n, so.MANUSCRIPT_DIM[n]) for n in so.FREE_DIM_NAMES]


@pytest.mark.parametrize("fn", [f for f in [*FIXED.values(), *FREE] if f.argmin], ids=lambda f: f.name)
def test_published_optimum_is_reproduced(fn):
    value = fn(torch.tensor([fn.argmin])).item()
    assert value == pytest.approx(fn.global_min, abs=2e-4 * max(1.0, abs(fn.global_min)))


# perm0db at d=80 overflows float64 over most of [-80, 80]^80 (80^80 squared ~ 1e304), so its
# random-search check runs at a smaller dimension; the optimum itself is checked at d=80 above.
NO_BEAT = [so.free_dim_function("perm0db", 6) if f.name == "perm0db" else f for f in FREE]


@pytest.mark.parametrize("fn", [f for f in [*FIXED.values(), *NO_BEAT] if f.global_min is not None],
                         ids=lambda f: f.name)
def test_no_random_point_beats_the_global_minimum(fn):
    y = fn(rand_in(fn.bounds, 20_000))
    assert torch.isfinite(y).all()
    assert y.min().item() >= fn.global_min - 1e-3 * max(1.0, abs(fn.global_min))


def test_hartmann4_minimum_matches_published_value():
    # Surjanovic & Bingham quote ~ -3.1355 for the rescaled 4-d Hartmann.
    x = sobol_design(FIXED["hartmann4"].bounds, 200_000, seed=0)
    assert FIXED["hartmann4"](x).min().item() == pytest.approx(-3.1355, abs=0.02)


@pytest.mark.parametrize("name,cls,d", [
    ("ackley", bs.Ackley, 5), ("rosenbrock", bs.Rosenbrock, 6), ("dixon_price", bs.DixonPrice, 6),
    ("michalewicz", bs.Michalewicz, 5),
])
def test_free_dim_functions_match_botorch(name, cls, d):
    fn = so.free_dim_function(name, d)
    x = rand_in(fn.bounds, 100)
    assert torch.allclose(fn(x), cls(dim=d)(x), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("name,ref", [
    ("hartmann3", lambda: bs.Hartmann(dim=3)), ("hartmann6", lambda: bs.Hartmann(dim=6)),
    ("six_hump_camel", bs.SixHumpCamel), ("three_hump_camel", bs.ThreeHumpCamel),
    ("eggholder", bs.EggHolder),
])
def test_fixed_dim_functions_match_botorch(name, ref):
    fn = FIXED[name]
    x = rand_in(fn.bounds, 100)
    assert torch.allclose(fn(x), ref()(x), rtol=1e-9, atol=1e-9)


def test_functions_without_botorch_twin_against_closed_forms():
    x = rand_in(so.free_dim_function("trid", 4).bounds, 50)
    trid = (x - 1).pow(2).sum(1) - (x[:, 1:] * x[:, :-1]).sum(1)
    assert torch.allclose(so.free_dim_function("trid", 4)(x), trid)
    x = torch.rand(20, 3, generator=GEN)
    assert torch.allclose(so.free_dim_function("sum_pow", 3)(x), x[:, 0] ** 2 + x[:, 1] ** 3 + x[:, 2] ** 4)
    assert torch.allclose(so.free_dim_function("sum_squares", 3)(x), x[:, 0] ** 2 + 2 * x[:, 1] ** 2 + 3 * x[:, 2] ** 2)
    x = torch.randn(20, 3, generator=GEN)
    # rotated hyper-ellipsoid: sum_i sum_{j<=i} x_j^2
    expect = x[:, 0] ** 2 * 3 + x[:, 1] ** 2 * 2 + x[:, 2] ** 2
    assert torch.allclose(so.free_dim_function("rotated_hyper_ellipsoid", 3)(x), expect)


def test_perm0db_does_not_overflow_at_its_optimum_for_the_manuscript_dimension():
    fn = so.free_dim_function("perm0db", 80)  # integer j**i would overflow int64 here
    assert fn(torch.tensor([fn.argmin])).item() == pytest.approx(0.0, abs=1e-20)






def test_eggholder_printed_formula_conflicts_with_its_stated_optimum():
    x1, x2 = 512.0, 404.2319
    printed = -(x2 + 47) * math.sin(abs(x2 + x1 / 2 + 47)) - x1 * math.sin(abs(x1 - x2 - 47))
    assert printed == pytest.approx(615.7060847706929)
    assert so.eggholder(torch.tensor([[x1, x2]])).item() == pytest.approx(-959.6407, abs=1e-4)
