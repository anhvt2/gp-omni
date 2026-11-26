"""Regression checks for the CI pass-rate gate."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ci_tests.py"
spec = importlib.util.spec_from_file_location("ci_tests", SCRIPT)
ci_tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci_tests)


@pytest.mark.parametrize(
    "passed,failed,skipped,errors,status,minimum,expected",
    [
        (4, 1, 0, 0, 1, 80, 0),  # Inclusive boundary.
        (3, 1, 0, 0, 1, 80, 1),
        (3, 1, 0, 0, 1, 75, 0),
        (4, 1, 20, 0, 1, 80, 0),  # Skips do not dilute the rate.
        (1, 1, 20, 0, 1, 80, 1),  # Skips cannot inflate it either.
        (100, 0, 0, 1, 1, 80, 1),
        (0, 0, 5, 0, 0, 80, 1),
        (0, 0, 0, 0, 0, 80, 1),
        (5, 0, 0, 0, 0, 80, 0),
        *[(5, 0, 0, 0, status, 80, status) for status in (2, 3, 4, 5)],
    ],
)
def test_gate(tmp_path, monkeypatch, passed, failed, skipped, errors, status, minimum, expected):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    report = tmp_path / "results.xml"
    cases = (
        "<testcase/>" * passed
        + "<testcase><failure/></testcase>" * failed
        + "<testcase><skipped/></testcase>" * skipped
        + "<testcase><error/></testcase>" * errors
    )
    report.write_text(f"<testsuites><testsuite>{cases}</testsuite></testsuites>")
    assert ci_tests.check_results(report, status, minimum) == expected


@pytest.mark.parametrize("contents", [None, "broken XML"])
def test_invalid_report_fails(tmp_path, contents):
    report = tmp_path / "results.xml"
    if contents is not None:
        report.write_text(contents)
    assert ci_tests.check_results(report, 1, 80) == 1


def test_runner_with_real_pytest(tmp_path, monkeypatch):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    suite = tmp_path / "test_sample.py"
    suite.write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('value', range(5))\n"
        "def test_sample(value):\n"
        "    assert value < 4\n"
    )
    for minimum, expected in [(80, 0), (81, 1)]:
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--min-pass-rate", str(minimum),
             "--report", str(tmp_path / "results.xml"), str(suite), "-q"],
            capture_output=True, text=True, check=False,
        )
        assert result.returncode == expected, result.stdout + result.stderr
        assert "4/5 (80.00%)" in result.stdout


@pytest.mark.parametrize(
    "problem,expected",
    [
        ("def test_error(missing_fixture): pass\n", 1),
        ("raise RuntimeError('collection failed')\n", 2),
    ],
)
def test_runner_preserves_errors(tmp_path, monkeypatch, problem, expected):
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    (tmp_path / "test_good.py").write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('value', range(10))\n"
        "def test_pass(value): pass\n"
    )
    (tmp_path / "test_problem.py").write_text(problem)
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--min-pass-rate", "80",
         "--report", str(tmp_path / "results.xml"), str(tmp_path), "-q"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == expected, result.stdout + result.stderr
