"""Validate analytical benchmarks independently of GP fitting.

Check known optima, sampled lower bounds, agreement with BoTorch where applicable,
analytic Pareto fronts, and explicit multi-fidelity relationships.
"""

import math

import pytest
import torch
from bo_functions import multi_fidelity as mf
from bo_functions import multi_objective as mo
from bo_functions import noisy_regression as nr
from bo_functions import single_objective as so
from bo_functions.common import sobol_design
from botorch.test_functions import multi_objective as bm
from botorch.test_functions import synthetic as bs

import gp_omni  # noqa: F401  (float64)

GEN = torch.Generator().manual_seed(7)


def rand_in(bounds, n):
    return bounds[0] + torch.rand(n, bounds.shape[1], generator=GEN) * (bounds[1] - bounds[0])


# ----------------------------------------------------------------------- single objective
FIXED = so.fixed_dim_functions()
FREE = [so.free_dim_function(n, so.REFERENCE_DIM[n]) for n in so.FREE_DIM_NAMES]


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


def test_perm0db_does_not_overflow_at_its_optimum_in_eighty_dimensions():
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
def test_modified_zdt_preserves_each_variant_domain_and_g_scaling(name, lo, value):
    fn = so.scalable_function(name, 3)
    assert fn.lower == (lo,) * 3 and fn.upper == (1.0,) * 3
    assert fn(torch.tensor([[0.5, 0.25, 0.75]])).item() == pytest.approx(value)


def test_eggholder_requires_square_roots_to_reproduce_its_optimum():
    x1, x2 = 512.0, 404.2319
    without_roots = -(x2 + 47) * math.sin(abs(x2 + x1 / 2 + 47)) - x1 * math.sin(abs(x1 - x2 - 47))
    assert without_roots == pytest.approx(615.7060847706929)
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
    """The three-fidelity borehole satisfies f_L1 <= f_H <= f_L2."""
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


# ------------------------------------------------------------------------ multi-objective
def test_zdt1_zdt2_match_botorch_everywhere():
    for name, cls in (("zdt1", bm.ZDT1), ("zdt2", bm.ZDT2)):
        x = torch.rand(100, 3, generator=GEN)
        assert torch.allclose(mo.zdt(name)(x), cls(dim=3)(x)), name


def test_zdt3_matches_botorch_on_the_pareto_set_only():
    """BoTorch's ZDT3 returns h instead of g*h; the two coincide where g = 1."""
    x = torch.cat([torch.rand(100, 1, generator=GEN), torch.zeros(100, 2)], 1)
    assert torch.allclose(mo.zdt("zdt3")(x), bm.ZDT3(dim=3)(x))
    x = torch.rand(100, 3, generator=GEN)
    assert not torch.allclose(mo.zdt("zdt3")(x), bm.ZDT3(dim=3)(x))


@pytest.mark.parametrize("name", ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6"])
def test_zdt_pareto_set_lies_on_the_analytic_front(name):
    fn = mo.zdt(name)
    f1 = torch.rand(200, 1, generator=GEN)
    x = torch.cat([f1, torch.zeros(200, fn.dim - 1)], 1)  # g = 1 on the Pareto set
    f = fn(x)
    front = {
        "zdt1": lambda a: 1 - a.sqrt(),
        "zdt2": lambda a: 1 - a**2,
        "zdt3": lambda a: 1 - a.sqrt() - a * torch.sin(10 * math.pi * a),
        "zdt4": lambda a: 1 - a.sqrt(),
        "zdt6": lambda a: 1 - a**2,
    }[name]
    assert torch.allclose(f[:, 1], front(f[:, 0]), atol=1e-10)


@pytest.mark.parametrize("name", ["zdt1", "zdt2", "zdt4", "zdt6"])
def test_zdt_front_is_never_dominated_by_random_points(name):
    fn = mo.zdt(name, dim=3)
    front = fn.pareto_front(400)
    f = fn(rand_in(fn.bounds, 5000))
    # no sampled point may strictly dominate a front point
    dom = (f[:, None, :] <= front[None] + 1e-9).all(-1) & (f[:, None, :] < front[None] - 1e-9).any(-1)
    assert not dom.any()


def test_zdt3_front_segments_are_nondominated_and_on_the_curve():
    fn = mo.zdt("zdt3")
    front = fn.pareto_front(500)
    assert torch.allclose(front[:, 1], 1 - front[:, 0].sqrt() - front[:, 0] * torch.sin(10 * math.pi * front[:, 0]))
    f = fn(rand_in(fn.bounds, 5000))
    dom = (f[:, None, :] <= front[None] + 1e-9).all(-1) & (f[:, None, :] < front[None] - 1e-9).any(-1)
    assert not dom.any()


@pytest.mark.parametrize("n", [0, 1, 3, 6, 11, 501])
def test_zdt3_front_returns_the_requested_number_of_points(n):
    assert mo.zdt("zdt3").pareto_front(n).shape == (n, 2)


@pytest.mark.parametrize("name,cls", [("dtlz1", bm.DTLZ1), ("dtlz2", bm.DTLZ2), ("dtlz3", bm.DTLZ3),
                                       ("dtlz5", bm.DTLZ5)])
def test_dtlz_matches_botorch_everywhere(name, cls):
    x = torch.rand(100, 4, generator=GEN)
    scale = 1.0 if name == "dtlz1" else 0.5  # scaled DTLZ variant
    assert torch.allclose(mo.dtlz(name)(x), scale * cls(dim=4, num_objectives=3)(x),
                          atol=1e-9, rtol=1e-9)


def test_dtlz4_variant_applies_alpha_only_to_cosine_factors():
    """This variant applies alpha to cosine factors while leaving sine factors unpowered."""
    x = torch.rand(100, 4, generator=GEN)
    assert torch.allclose(mo.dtlz4(x, 3, alpha=1.0), mo.dtlz2(x, 3))
    assert not torch.allclose(mo.dtlz4(x, 3), mo.dtlz2(x, 3))
    a, b = x[:, 0], x[:, 1]
    radius = 0.5 * (1 + ((x[:, 2:] - 0.5) ** 2).sum(1))
    ca, cb = torch.cos(a**100 * math.pi / 2), torch.cos(b**100 * math.pi / 2)
    expect = radius[:, None] * torch.stack(
        [ca * cb, ca * torch.sin(b * math.pi / 2), torch.sin(a * math.pi / 2)], 1
    )
    assert torch.allclose(mo.dtlz4(x, 3), expect)
    assert mo.dtlz("dtlz4").pareto_front is None


def _pareto_x(n, m, d, tail):
    return torch.cat([torch.rand(n, m - 1, generator=GEN), torch.full((n, d - m + 1), tail)], 1)


@pytest.mark.parametrize("m,d", [(3, 4), (3, 8), (2, 6), (4, 7)])
def test_dtlz_pareto_sets_lie_on_their_analytic_fronts(m, d):
    half = _pareto_x(200, m, d, 0.5)
    assert torch.allclose(mo.dtlz1(half, m).sum(1), torch.full((200,), 0.5))  # simplex
    for fn in (mo.dtlz2, mo.dtlz3):
        assert torch.allclose(fn(half, m).pow(2).sum(1), torch.full((200,), 0.25)), fn.__name__
    assert torch.allclose(mo.dtlz5(half, m).pow(2).sum(1), torch.full((200,), 0.25))
    assert torch.allclose(mo.dtlz6(_pareto_x(200, m, d, 0.0), m).pow(2).sum(1), torch.full((200,), 0.25))
    assert mo.dtlz("dtlz2", m, d)(half).shape == (200, m)


def test_dtlz5_front_is_a_curve():
    """On the Pareto set (g = 0) theta_2.. are all pi/4, so the objectives depend only on x_1:
    changing x_2 does not move the point (the 3-objective front collapses to a curve)."""
    x = _pareto_x(100, 3, 6, 0.5)
    x2 = x.clone()
    x2[:, 1] = torch.rand(100, generator=GEN)
    assert torch.allclose(mo.dtlz5(x, 3), mo.dtlz5(x2, 3), atol=1e-9)
    x6, x6b = _pareto_x(100, 3, 6, 0.0), None
    x6b = x6.clone()
    x6b[:, 1] = torch.rand(100, generator=GEN)
    assert torch.allclose(mo.dtlz6(x6, 3), mo.dtlz6(x6b, 3), atol=1e-9)


def test_multi_objective_default_shapes():
    fn = mo.dtlz("dtlz2")
    assert (fn.dim, fn.num_objectives) == (4, 3)
    assert mo.zdt("zdt1").dim == 3 and mo.zdt("zdt1").num_objectives == 2
    assert len(mo.all_multi_objective()) == 11


def test_zdt4_domain_and_multimodality():
    fn = mo.zdt("zdt4")
    assert fn.lower == (0.0, -5.0, -5.0) and fn.upper == (1.0, 5.0, 5.0)
    f = fn(rand_in(fn.bounds, 2000))
    assert (f[:, 1] > 0).all()  # g >= 1 everywhere; the global front needs x_2.. = 0 exactly


def test_pareto_front_generators_are_on_the_analytic_surfaces():
    assert torch.allclose(mo.dtlz("dtlz1").pareto_front(100).sum(1), torch.full((100,), 0.5))
    for name in ("dtlz2", "dtlz3"):
        assert torch.allclose(mo.dtlz(name).pareto_front(100).pow(2).sum(1), torch.full((100,), 0.25))
    z6 = mo.zdt("zdt6").pareto_front(50)
    assert z6[:, 0].min().item() == pytest.approx(0.2807753191)


# ------------------------------------------------------------------------- noisy regression
def test_noisy_problems_truth_and_noise_models():
    log, hp = nr.noisy_logistic(), nr.heteroscedastic_hall_petch()
    assert log.truth(torch.tensor([0.0])).item() == pytest.approx(0.5)
    assert (hp.truth(torch.tensor([15.0, 350.0])).diff() < 0).all()  # monotone decreasing in d
    assert hp.truth(torch.tensor([350.0])).item() == pytest.approx(17.4, abs=0.1)
    sd = hp.noise_sd(torch.tensor([15.0, 350.0]))
    assert (sd.diff() > 0).all() and sd[1].item() == pytest.approx(math.sqrt(2.2e10 * 350**3))
    assert nr.hall_petch_noise_sd_metres(torch.tensor([350.0])).item() == pytest.approx(0.97, abs=0.05)
    for p in (log, hp):
        x, y = p.sample(seed=3)
        assert x.shape == (p.num_train, 1) and y.shape == (p.num_train,)
        assert x.min() >= p.lower and x.max() <= p.upper
        assert torch.equal(p.sample(seed=3)[1], y)  # deterministic given the seed


def test_sobol_design_is_in_bounds_and_seeded():
    b = mf.borehole(2).bounds
    x = sobol_design(b, 33, seed=1)
    assert x.shape == (33, 8) and (x >= b[0]).all() and (x <= b[1]).all()
    assert torch.equal(x, sobol_design(b, 33, seed=1)) and not torch.equal(x, sobol_design(b, 33, seed=2))
