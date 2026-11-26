"""Run pytest with a minimum pass rate while preserving runner/setup errors."""

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def check_results(report: Path, pytest_status: int, minimum: float) -> int:
    if pytest_status not in (0, 1):
        print(f"pytest did not complete normally (exit {pytest_status}).")
        return pytest_status
    try:
        cases = list(ET.parse(report).getroot().iter("testcase"))
    except (OSError, ET.ParseError) as exc:
        print(f"Cannot read pytest results: {exc}")
        return 1

    errors = sum(case.find("error") is not None for case in cases)
    skipped = sum(case.find("skipped") is not None for case in cases)
    failed = sum(case.find("failure") is not None for case in cases)
    executed = len(cases) - skipped
    passed = executed - failed - errors
    rate = 100 * passed / executed if executed else 0
    summary = (
        f"Test pass rate: {passed}/{executed} ({rate:.2f}%); required: {minimum:g}%. "
        f"Failures: {failed}; errors: {errors}; skipped: {skipped}."
    )
    print(summary)
    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a") as stream:
            stream.write(summary + "\n")
    return 0 if executed and not errors and rate >= minimum else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-pass-rate", type=float, default=80)
    parser.add_argument("--report", type=Path, default=Path("test-results/pytest.xml"))
    args, pytest_args = parser.parse_known_args()
    if not 0 < args.min_pass_rate <= 100:
        parser.error("--min-pass-rate must be greater than 0 and at most 100")

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.unlink(missing_ok=True)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", *pytest_args, f"--junitxml={args.report}"],
        check=False,
    )
    return check_results(args.report, result.returncode, args.min_pass_rate)


if __name__ == "__main__":
    sys.exit(main())
