"""Both Bayesian optimization loops drive a CFD case, with a stand-in for the solver."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples/cfd"))

from foam.campaign import optimize, settings
from foam.case import at_least, cfd_case, variable

CASE = cfd_case(
    "stand_in", "internal", "analytic stand-in", "structured", "simpleFoam",
    (variable("a", 0.0, 1.0, 0.5), variable("b", 1.0, 2.0, 1.5)),
    "loss", (at_least("capacity", 0.2),), build=None,
)  # fmt: skip


def solve(case, design, directory, options, fail_above=2.0):
    if design["a"] > fail_above:
        return {"case": case.name, "design": design, "passed": False, "error": "mesh failed"}
    return {
        "case": case.name,
        "design": design,
        "passed": True,
        "metrics": {"loss": (design["a"] - 0.3) ** 2 + design["b"], "capacity": design["b"]},
    }


@pytest.mark.parametrize("loop", ["batch", "async"])
def test_loop_returns_the_best_feasible_design(tmp_path, loop):
    options = settings(initial=4, batches=2, batch_size=2, loop=loop)
    summary = optimize(CASE, tmp_path / "run", options, evaluate_design=solve)
    assert summary["evaluations"] == 8
    assert summary["failed"] == 0
    best = summary["best"]["metrics"]
    assert best["capacity"] >= 1.5 * 0.8  # the limit set by the reference design
    assert best["loss"] <= summary["reference"]["metrics"]["loss"]
    assert json.loads((tmp_path / "run/summary.json").read_text())["case"] == "stand_in"


def test_failed_designs_are_counted_and_the_campaign_continues(tmp_path):
    options = settings(initial=4, batches=1, batch_size=2)

    def sometimes(*arguments):
        return solve(*arguments, fail_above=0.55)

    summary = optimize(CASE, tmp_path / "run", options, evaluate_design=sometimes)
    assert summary["evaluations"] == 6
    assert 0 < summary["failed"] < 6


def test_failed_reference_stops_the_campaign(tmp_path):
    def broken(case, design, directory, options):
        return {"passed": False, "error": "no mesh"}

    with pytest.raises(RuntimeError, match="reference design failed"):
        optimize(CASE, tmp_path / "run", settings(), evaluate_design=broken)


@pytest.mark.parametrize(
    "bad", [{"initial": 1}, {"loop": "other"}, {"level": "fine"}, {"cores": 0}]
)
def test_settings_are_validated(bad):
    with pytest.raises(ValueError):
        settings(**bad)
