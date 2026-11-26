"""The optimization campaign shared by the 3D design cases in ``examples/fem``."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("skfem")
pytest.importorskip("gmsh")

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/fem"
sys.path.insert(0, str(EXAMPLES))

from cold_plate import PROBLEM as PLATE
from common import campaign
from lattice_beam import PROBLEM as LATTICE

SETTINGS = campaign.settings(initial=3, batches=1, batch_size=2, seed=0, mesh_scale=1.8)


def fake_metrics(temperature, drop=1.0):
    return {"temperature_rise_k": temperature, "pressure_drop_kpa": drop, "mass_kg": 0.18}


def test_worker_process_returns_the_metrics_of_a_design():
    metrics = campaign.evaluate_in_subprocess(PLATE, PLATE.reference_design(), 1.8)
    assert metrics["convected_w"] == pytest.approx(90, rel=1e-4)


def test_worker_process_reports_why_a_design_failed():
    design = {**PLATE.reference_design(), "thickness_mm": 99.0}
    with pytest.raises(RuntimeError, match="thickness_mm"):
        campaign.evaluate_in_subprocess(PLATE, design, 1.8)


def test_worker_process_is_stopped_at_the_time_limit():
    with pytest.raises(RuntimeError, match="time limit"):
        campaign.evaluate_in_subprocess(PLATE, PLATE.reference_design(), 1.8, 0.01)


def test_campaign_records_every_design_and_reports_the_best_feasible(tmp_path):
    summary = campaign.run_campaign(PLATE, tmp_path / "run", SETTINGS)

    assert summary["evaluations"] == 3 + 2
    assert summary["failed"] == 0
    records = [json.loads(p.read_text()) for p in (tmp_path / "run").glob("design_*.json")]
    assert len(records) == 5
    feasible = [r["metrics"] for r in records if r["feasible"]]
    if summary["reference"]["feasible"]:
        feasible.append(summary["reference"]["metrics"])
    best = min(metrics["temperature_rise_k"] for metrics in feasible)
    assert summary["best"]["metrics"]["temperature_rise_k"] == best
    assert summary["feasible"] == sum(r["feasible"] for r in records)
    assert json.loads((tmp_path / "run/summary.json").read_text()) == summary


def test_campaign_refuses_to_overwrite_an_existing_directory(tmp_path):
    with pytest.raises(FileExistsError):
        campaign.run_campaign(PLATE, tmp_path, SETTINGS)


def test_failed_designs_are_infeasible_and_counted(tmp_path, monkeypatch):
    calls = []

    def flaky(problem, design, mesh_scale, timeout_s):
        calls.append(design)
        if len(calls) == 3:
            raise RuntimeError("meshing failed")
        return fake_metrics(20.0 - len(calls))

    monkeypatch.setattr(campaign, "evaluate_in_subprocess", flaky)
    summary = campaign.run_campaign(PLATE, tmp_path / "run", SETTINGS)

    assert summary["failed"] == 1
    assert summary["feasible"] == 4
    errors = [
        record["error"]
        for path in (tmp_path / "run").glob("design_*.json")
        if "error" in (record := json.loads(path.read_text()))
    ]
    assert errors == ["meshing failed"]


def test_campaign_fails_loudly_when_every_design_fails(tmp_path, monkeypatch):
    calls = []

    def broken(problem, design, mesh_scale, timeout_s):
        calls.append(design)
        if len(calls) == 1:  # the reference succeeds
            return fake_metrics(20.0)
        raise RuntimeError("meshing failed")

    monkeypatch.setattr(campaign, "evaluate_in_subprocess", broken)
    with pytest.raises(RuntimeError, match="every candidate design failed"):
        campaign.run_campaign(PLATE, tmp_path / "run", SETTINGS)


def test_infeasible_campaign_reports_no_best_design(tmp_path, monkeypatch):
    monkeypatch.setattr(
        campaign, "evaluate_in_subprocess", lambda *args: fake_metrics(20.0, drop=99.0)
    )
    summary = campaign.run_campaign(PLATE, tmp_path / "run", SETTINGS)
    assert summary["best"] is None


def test_command_line_solves_the_reference_design():
    output = subprocess.run(
        [sys.executable, EXAMPLES / "cold_plate.py", "baseline", "--mesh-scale", "1.8"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "temperature_rise_k" in output


def test_lattice_design_that_once_stalled_the_boolean_union_solves():
    design = {
        "strut_diameter_mm": 1.931421571782407,
        "core_height_mm": 11.321399218330997,
        "bottom_face_mm": 1.6483583112379652,
        "top_face_mm": 1.4749302548311864,
    }
    metrics = campaign.evaluate_in_subprocess(LATTICE, design, 1.5, timeout_s=120)
    assert metrics["deflection_mm"] > 0
