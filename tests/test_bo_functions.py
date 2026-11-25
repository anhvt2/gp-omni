"""The ported manuscript test functions are right (no GP involved).

Checks: published optima, no random point beats the optimum, exact agreement with BoTorch's
independent implementations where they exist, analytic Pareto fronts, and the manuscripts' own
claims about the multi-fidelity pairs.
"""

import math

import pytest
import torch
from bo_functions import multi_fidelity as mf
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


def test_scalable_functions_have_two_active_directions():
    d = 15
    for name in ("modified_zdt1", "modified_zdt2", "modified_zdt3"):
        fn = so.scalable_function(name, d)
        x = rand_in(fn.bounds, 5)
        permuted = x.clone()
        permuted[:, 1:] = x[:, 1:][:, torch.randperm(d - 1, generator=GEN)]
        assert torch.allclose(fn(x), fn(permuted)), name  # depends on x_1 and sum(x_2..) only
    assert so.scalable_function("squared_sum", d)(torch.ones(1, d)).item() == pytest.approx(d**2)
    assert so.scalable_function("modified_zdt1", d)(torch.zeros(1, d)).item() == pytest.approx(1.0)


@pytest.mark.parametrize("name,lo,value", [
    ("modified_zdt1", -1.0, 2.3486121811340026),
    ("modified_zdt2", -1.0, 81.9969512195122),
    ("modified_zdt3", 0.0, 8.16886116991581),
])
def test_modified_zdt_preserves_each_printed_domain_and_g_scaling(name, lo, value):
    fn = so.scalable_function(name, 3)
    assert fn.lower == (lo,) * 3 and fn.upper == (1.0,) * 3
    assert fn(torch.tensor([[0.5, 0.25, 0.75]])).item() == pytest.approx(value)


def test_eggholder_printed_formula_conflicts_with_its_stated_optimum():
    x1, x2 = 512.0, 404.2319
    printed = -(x2 + 47) * math.sin(abs(x2 + x1 / 2 + 47)) - x1 * math.sin(abs(x1 - x2 - 47))
    assert printed == pytest.approx(615.7060847706929)
    assert so.eggholder(torch.tensor([[x1, x2]])).item() == pytest.approx(-959.6407, abs=1e-4)


# --------------------------------------------------------------------------- multi-fidelity
def test_forrester_matches_published_minimum_and_relation():
    fn = mf.forrester()
    t = torch.linspace(0, 1, 100_001).reshape(-1, 1)
    assert fn(t).min().item() == pytest.approx(-6.02074, abs=1e-4)
    assert fn(torch.tensor([[0.757249]])).item() == pytest.approx(-6.02074, abs=1e-4)
    # f_L = 0.5 f_H + 10 (x - 0.5) - 5, by construction
    assert torch.allclose(fn(t, 0), 0.5 * fn(t, 1) + 10 * (t[:, 0] - 0.5) - 5)


def test_currin_maximum_and_boundary():
    fn = mf.currin()
    assert fn(torch.tensor([[0.2167, 0.0087]])).item() == pytest.approx(13.7987, abs=1e-3)
    grid = torch.cartesian_prod(torch.linspace(0, 1, 401), torch.linspace(0, 1, 401))
    assert fn(grid).max().item() == pytest.approx(13.7987, abs=5e-3)
    edge = torch.tensor([[0.5, 0.0], [0.0, 0.0], [1.0, 0.0]])
    assert torch.isfinite(fn(edge)).all() and torch.isfinite(fn(edge, 0)).all()  # x_2 = 0: 1/(2 x_2)
    x = rand_in(fn.bounds, 2000)
    assert torch.corrcoef(torch.stack([fn(x, 0), fn(x, 1)]))[0, 1] > 0.99


def test_borehole_low_fidelities_bracket_the_high_fidelity():
    """The manuscript claims f_L1 <= f_H <= f_L2 for the three-fidelity borehole."""
    fn = mf.borehole(3)
    x = rand_in(fn.bounds, 20_000)
    low2, low1, high = (fn(x, k) for k in range(3))
    assert (low1 <= high).all() and (high <= low2).all()


def test_borehole_levels_are_positive_flow_rates_and_correlated():
    fn = mf.borehole(2)
    x = rand_in(fn.bounds, 5000)
    lo, hi = fn(x, 0), fn(x, 1)
    assert (hi > 0).all() and (lo > 0).all()
    assert torch.corrcoef(torch.stack([lo, hi]))[0, 1] > 0.95


def test_welded_beam_cost_matches_the_classic_optimum():
    # Ragsdell & Phillips optimum: cost 1.7249 at (h, l, t, b) = (0.2057, 3.4705, 9.0366, 0.2057)
    x = torch.tensor([[0.20573, 3.47049, 9.03662, 0.20573]])
    assert mf.weld_cost(*x[0], m=0, w=0).item() == pytest.approx(1.7249, abs=1e-3)
    assert mf.welded_beam()(x).item() == pytest.approx(1.7245, abs=1e-3)  # f_H is steel, w=0
    g = mf.weld_constraints(x, m=0, w=0)[0]
    assert g[1].abs().item() < 0.1  # bending stress active (sigma = sigma_d)
    assert g[2].abs().item() < 1e-9  # b - h = 0 active
    assert g[4].item() > 0  # deflection satisfied


def test_welded_beam_levels_rank_by_material_cost():
    fn = mf.welded_beam()
    x = rand_in(fn.bounds, 500)
    assert fn.num_fidelities == 4
    steel, cast, alu, brass = fn(x, 3), fn(x, 2), fn(x, 1), fn(x, 0)
    assert (cast < steel).all() and (steel < alu).all() and (alu < brass).all()


def test_sobol_design_is_in_bounds_and_seeded():
    b = mf.borehole(2).bounds
    x = sobol_design(b, 33, seed=1)
    assert x.shape == (33, 8) and (x >= b[0]).all() and (x <= b[1]).all()
    assert torch.equal(x, sobol_design(b, 33, seed=1)) and not torch.equal(x, sobol_design(b, 33, seed=2))
