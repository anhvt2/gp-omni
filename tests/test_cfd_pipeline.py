"""The CFD pipeline records every failure; stand-in programs replace OpenFOAM."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "examples/cfd"))

from foam import measure, pipeline
from foam.case import SMOKE, at_least, at_most, cfd_case, run_plan, variable
from foam.dictionaries import render
from foam.report import write_status

GOOD_MESH = """
Mesh stats
    cells:            1200
    Mesh non-orthogonality Max: 41.5 average: 8
    Max skewness = 1.2 OK.

Mesh OK.
"""


@pytest.fixture
def tools(tmp_path, monkeypatch):
    """A directory of stand-in programs placed first on the search path."""
    directory = tmp_path / "bin"
    directory.mkdir()
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ['PATH']}")

    def install(name: str, output: str = "", status: int = 0):
        script = directory / name
        script.write_text(f"#!/bin/sh\ncat <<'EOF'\n{output}\nEOF\nexit {status}\n")
        script.chmod(0o755)

    install("checkMesh", GOOD_MESH)
    install("simpleFoam", "Time = 200\n\nEnd\n")
    return install


def fake_case(metrics=None, mesh=(), solve=(("simpleFoam",),)):
    metrics = {"loss": 2.0, "size": 1.0} if metrics is None else metrics

    def build(directory, design, level):
        (directory / "200").mkdir()
        (directory / "0").mkdir()
        return run_plan(
            mesh=[list(c) for c in mesh],
            solve=[list(c) for c in solve],
            measure=lambda _: dict(metrics),
        )

    return cfd_case(
        "fake", "internal", "stand-in", "structured", "simpleFoam",
        (variable("width", 0.0, 1.0, 0.5),), "loss", (at_most("size"),), build,
    )  # fmt: skip


def run(case, tmp_path, **options):
    options.setdefault("minimum_free_gb", 0.0)
    return pipeline.run_design(
        case, case.reference_design(), SMOKE, tmp_path / "design", **options
    )


def test_passing_run_writes_a_record_and_removes_fields(tools, tmp_path):
    record = run(fake_case(), tmp_path)
    assert record["passed"], record
    assert record["mesh"]["cells"] == 1200
    assert json.loads((tmp_path / "design/record.json").read_text()) == record
    assert not (tmp_path / "design/200").exists()
    assert (tmp_path / "design/0").exists()
    assert (tmp_path / "design/log.simpleFoam").is_file()


def test_mesh_only_preserves_mesh_and_requires_a_separate_verify_directory(tools, tmp_path):
    record = run(fake_case(), tmp_path, mesh_only=True)
    assert record["passed"]
    assert record["stage"] == "meshed"
    assert "metrics" not in record
    directory = tmp_path / "design"
    assert (directory / "200").is_dir()
    assert not (directory / "log.simpleFoam").exists()
    with pytest.raises(RuntimeError, match="mesh-only run"):
        pipeline.clear_unfinished(directory, "fake", SMOKE.name)
    assert (directory / "log.checkMesh").is_file()


@pytest.mark.parametrize(
    ("install", "stage", "message"),
    [
        (("checkMesh", GOOD_MESH + "***Zero or negative cell volume detected."), "check_mesh",
         "negative-volume"),
        (("checkMesh", GOOD_MESH.replace("41.5", "86.2")), "check_mesh", "non-orthogonality"),
        (("simpleFoam", "FOAM FATAL ERROR: boom", 1), "solve", "boom"),
        (("simpleFoam", "Ux, Initial residual = nan, Final\n\nEnd\n"), "solve", "non-finite"),
        (("simpleFoam", "Time = 3\n"), "solve", "did not reach the end"),
    ],
)  # fmt: skip
def test_failures_name_their_stage(tools, tmp_path, install, stage, message):
    tools(*install)
    record = run(fake_case(), tmp_path)
    assert not record["passed"]
    assert record["stage"] == stage
    assert message in record["error"]


@pytest.mark.parametrize(
    ("metrics", "message"),
    [
        ({"loss": float("nan"), "size": 1.0}, "non-finite"),
        ({"loss": 1.0}, "missing metrics"),
        ({"loss": 1.0, "size": 1.0, "flow_imbalance": 0.5}, "flow imbalance"),
    ],
)
def test_bad_measurements_fail_the_design(tools, tmp_path, metrics, message):
    record = run(fake_case(metrics), tmp_path)
    assert record["stage"] == "measure"
    assert message in record["error"]


def test_missing_program_fails_the_mesh_stage(tools, tmp_path):
    record = run(fake_case(mesh=(("no_such_mesher",),)), tmp_path)
    assert record["stage"] == "mesh"
    assert "container" in record["error"]


def test_existing_directory_is_refused(tools, tmp_path):
    (tmp_path / "design").mkdir()
    with pytest.raises(FileExistsError):
        run(fake_case(), tmp_path)


def test_disk_guard_stops_before_creating_anything(tools, tmp_path):
    with pytest.raises(RuntimeError, match="GB free"):
        run(fake_case(), tmp_path, minimum_free_gb=1e9)
    assert not (tmp_path / "design").exists()


def test_design_outside_bounds_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="must lie in"):
        pipeline.run_design(fake_case(), {"width": 2.0}, SMOKE, tmp_path / "design")


def test_limits_are_relative_to_the_reference():
    assert at_least("a", 0.1).bound(10.0) == pytest.approx(9.0)
    assert at_most("a", 0.1).bound(10.0) == pytest.approx(11.0)
    assert at_least("a", 0.1).bound(-10.0) == pytest.approx(-11.0)
    case = fake_case()
    assert case.feasible({"size": 1.05}, {"size": 1.0})
    assert not case.feasible({"size": 1.2}, {"size": 1.0})


def test_measurement_files_are_parsed(tmp_path):
    for patch, value in (("inlet", -2.0), ("outlet", 1.9)):
        directory = tmp_path / f"postProcessing/flux_{patch}/0"
        directory.mkdir(parents=True)
        (directory / "surfaceFieldValue.dat").write_text(f"# Time sum(phi)\n5 0\n10 {value}\n")
    assert measure.last_flux(tmp_path, "outlet") == 1.9
    assert measure.mean_surface(tmp_path, "flux_outlet") == pytest.approx(1.9)
    assert measure.imbalance(tmp_path, "inlet", "outlet") == pytest.approx(0.05)
    directory = tmp_path / "postProcessing/body/0"
    directory.mkdir(parents=True)
    (directory / "force.dat").write_text("10 (1 2 3) 4 5 6 7 8 9\n")
    np.testing.assert_allclose(measure.last_force(tmp_path, "body"), [1, 2, 3])
    with pytest.raises(ValueError, match="no volFieldValue.dat"):
        measure.last_volume(tmp_path, "absent")


def test_dictionary_rendering():
    text = render({"a": {"b": True, "c": (1, 0.5, 0)}, "d": "bounded Gauss upwind", "e": None})
    assert "b true;" in text
    assert "c (1 0.5 0);" in text
    assert "d bounded Gauss upwind;" in text
    assert "e;" in text


def test_status_table_reports_a_failure(tmp_path):
    directory = tmp_path / "fake"
    directory.mkdir()
    record = {"case": "fake", "passed": False, "stage": "solve", "error": "a|b"}
    (directory / "record.json").write_text(json.dumps(record))
    assert write_status(tmp_path, {"fake": fake_case()}) == (0, 1)
    table = (tmp_path / "status.md").read_text()
    assert "FAIL solve" in table
    assert "a/b" in table


def test_verify_only_clears_its_own_unfinished_runs(tools, tmp_path):
    directory = tmp_path / "design"
    assert not pipeline.clear_unfinished(directory, "fake", "smoke")

    run(fake_case(), tmp_path)
    assert pipeline.clear_unfinished(directory, "fake", "smoke")
    with pytest.raises(RuntimeError, match="holds a smoke run"):
        pipeline.clear_unfinished(directory, "fake", "standard")

    record = json.loads((directory / "record.json").read_text()) | {"passed": False}
    (directory / "record.json").write_text(json.dumps(record))
    assert not pipeline.clear_unfinished(directory, "fake", "smoke")
    assert not directory.exists()


@pytest.mark.parametrize("content", [None, '{"case": "other", "level": "smoke"}'])
def test_verify_keeps_folders_it_did_not_create(tmp_path, content):
    directory = tmp_path / "pipe"
    directory.mkdir()
    (directory / "notes.txt").write_text("mine")
    if content:
        (directory / "record.json").write_text(content)
    with pytest.raises(RuntimeError, match="not a run of pipe"):
        pipeline.clear_unfinished(directory, "pipe", "smoke")
    assert (directory / "notes.txt").read_text() == "mine"
