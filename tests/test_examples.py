"""Smoke tests: every example runs and reaches a sensible error on its toy problem."""

import ast
import importlib.util
import itertools
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parents[1] / "examples"


def run(name: str) -> dict:
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main()


LIMITS = {
    "exact_gpr": {"rmse": 0.05},
    "multioutput": {"icm_rmse": 0.05, "lmc_rmse": 0.05},
    "sparse": {"sgpr_rmse": 0.05, "kiss_rmse": 0.05, "svgp_rmse": 0.1},
}


@pytest.mark.parametrize("name", LIMITS)
def test_example_error(name):
    res = run(name)
    for key, limit in LIMITS[name].items():
        assert res[key] < limit, (key, res[key])


def test_classification_example():
    assert run("classification")["accuracy"] > 0.9


@pytest.mark.parametrize("name", ["multifidelity_botorch", "cokriging"])
def test_multifidelity_beats_high_fidelity_only(name):
    res = run(name)
    assert res["rmse"] < 0.7 * res["hi_only_rmse"], res


def test_cokriging_example_is_accurate():
    assert run("cokriging")["rmse"] < 0.5


def test_dag_cokriging_example():
    res = run("dag_cokriging")
    assert res["rmse"] < 0.08 and res["parent_rmse"] < 0.05, res
    assert res["min_variance"] >= 0 and res["posterior_shape"] == (4, 1)


def test_additive_example_beats_plain_ard():
    res = run("additive")
    assert res["additive_nrmse"] < 0.1
    assert res["additive_nrmse"] < 0.3 * res["plain_ard_nrmse"]


def test_sparse_vfe_example():
    res = run("sparse_vfe")
    assert res["vfe_rmse"] < 0.05 and res["fitc_rmse"] < 0.05
    assert abs(res["vfe_noise_sd"] - 0.1) < 0.02  # VFE recovers the true noise level
    assert res["probit_accuracy"] > 0.9


def test_gpr_multioutput_example():
    res = run("gpr_multioutput")
    assert res["joint_nrmse"] < 0.5 * res["alone_nrmse"], res


def test_gpr_multifidelity_example():
    res = run("gpr_multifidelity")
    assert res["mf_rmse"] < 0.6 * res["hi_only_rmse"], res


def test_gpr_sparse_example():
    res = run("gpr_sparse")
    assert res["vfe_rmse"] < 0.02 and res["fitc_rmse"] < 0.02
    assert abs(res["vfe_noise_sd"] - 0.1) < 0.02


def test_gpr_additive_example():
    res = run("gpr_additive")
    assert res["additive_nrmse"] < 0.1
    assert res["additive_nrmse"] < 0.3 * res["plain_ard_nrmse"], res


def test_gpr_combined_example_default_runs_the_progressive_configs():
    res = run("gpr_combined")
    assert set(res) == {"mo+mf", "mo+mf+sparse", "mo+mf+sparse+additive"}
    assert all(v < 0.1 for v in res.values()), res


def _load(name):
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_constrained_multiobjective_pareto_filter():
    import torch

    mod = _load("constrained_multiobjective")
    # A dominated point, a tied optimum, and two competing extremes.
    y = torch.tensor([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    assert mod.nondominated(y).tolist() == [False, True, True, True, True]


def test_constrained_multiobjective_example():
    import torch

    mod = _load("constrained_multiobjective")
    result = mod.main()
    assert result["low_fidelity_designs"] == 25
    assert result["initial_high_fidelity_designs"] == 9
    assert 1 < result["parallel_batch_size"] <= 4
    assert len(result["batch_feasible"]) == result["parallel_batch_size"]
    x = torch.tensor(result["pareto_x"]).reshape(-1, 1)
    y = torch.tensor(result["pareto_objectives"])
    measured, feasible = mod.evaluate_high(x)
    assert feasible.all()
    assert torch.allclose(y, measured)
    assert mod.nondominated(y).all()


@pytest.mark.parametrize("mo,mf,sparse,additive", list(itertools.product([0, 1], repeat=4)))
def test_gpr_combined_example_any_subset_of_features(mo, mf, sparse, additive):
    """Each of the 16 subsets runs end to end through the example and is accurate."""
    mod = _load("gpr_combined")
    chosen = {f for f, on in zip(mod.FEATURES, (mo, mf, sparse, additive), strict=True) if on}
    res = mod.main(chosen)
    assert len(res) == 1 and next(iter(res.values())) < 0.1, (chosen, res)


def test_gpr_combined_example_command_line_subset():
    """``python examples/gpr_combined.py mo additive`` (high-dimensional multi-objective)."""
    out = subprocess.run(
        [sys.executable, "-W", "ignore", str(EXAMPLES / "gpr_combined.py"), "mo", "additive"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    res = ast.literal_eval(out.strip().splitlines()[-1])
    assert list(res) == ["mo+additive"] and res["mo+additive"] < 0.1, res
    bad = subprocess.run(
        [sys.executable, "-W", "ignore", str(EXAMPLES / "gpr_combined.py"), "bogus"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert bad.returncode != 0 and "unknown features" in bad.stderr


@pytest.mark.parametrize(
    "features",
    [
        [],
        ["mo"],
        ["mf"],
        ["sparse"],
        ["additive"],
        ["mo", "additive"],
        ["mo", "mf", "sparse", "additive"],
    ],
)
def test_gpr_gpc_constraints_example_twins_share_any_features(features):
    mod = _load("gpr_gpc_constraints")
    ((name, res),) = mod.main(set(features)).items()
    assert name == ("+".join(f for f in mod.FEATURES if f in features) or "none")
    assert res["objective_nrmse"] < 0.1 and res["constraint_accuracy"] > 0.9, res


def test_gpr_gpc_constraints_example_command_line():
    run_cmd = lambda *a: subprocess.run(
        [sys.executable, "-W", "ignore", str(EXAMPLES / "gpr_gpc_constraints.py"), *a],
        capture_output=True,
        text=True,
        check=False,
    )
    ok = run_cmd("mo", "additive")
    assert ok.returncode == 0
    res = ast.literal_eval(ok.stdout.strip().splitlines()[-1])
    assert list(res) == ["mo+additive"] and res["mo+additive"]["constraint_accuracy"] > 0.9
    bad = run_cmd("bogus")
    assert bad.returncode != 0 and "unknown features" in bad.stderr
