"""Verify the self-contained CFD assets and fresh-case preparation."""

import gzip
import importlib.util
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).parents[1] / "examples/cfd/centrifugal_pump.py"
spec = importlib.util.spec_from_file_location("centrifugal_pump", SCRIPT)
pump = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pump)


def test_bundled_case_integrity_and_mesh():
    manifest = pump.verify_case()
    assert len(manifest) == 25
    for name in ("points", "faces", "owner", "neighbour", "boundary", "cellZones"):
        payload = gzip.decompress(
            (pump.BUNDLED_CASE / "constant/polyMesh" / f"{name}.gz").read_bytes()
        )
        assert b"FoamFile" in payload
    assert b"rotor" in gzip.decompress(
        (pump.BUNDLED_CASE / "constant/polyMesh/cellZones.gz").read_bytes()
    )


def test_case_tampering_is_detected(tmp_path):
    import shutil

    template = tmp_path / "template"
    shutil.copytree(pump.BUNDLED_CASE, template)
    with patch.object(pump, "BUNDLED_CASE", template):
        (template / "0/U").write_text("modified")
        with pytest.raises(ValueError, match="checksum mismatch"):
            pump.verify_case()


def test_simulation_uses_fresh_bundled_inputs(tmp_path):
    case = tmp_path / "design"
    case.mkdir()
    result = {"pressure_rise_pa": 1e6, "shaft_power_w": 5e4}
    with (
        patch.object(pump, "run_command") as command,
        patch.object(pump, "measure", return_value=result),
    ):
        measured = pump.simulate(case, 1.02, 0.95, 800, 60)
    assert measured["speed_rad_s"] == 314 * 0.95
    assert (case / "constant/polyMesh/points.gz").is_file()
    assert "endTime 800;" in (case / "system/controlDict").read_text()
    assert command.call_args_list[-1].args[-1] == "simpleFoam"
    assert (case / "measurements.json").is_file()
    assert (case / "0/U").read_bytes() == (pump.BUNDLED_CASE / "0/U").read_bytes()
