"""Status table of a verify run."""

import json
from collections.abc import Mapping
from pathlib import Path


def write_status(output: Path, cases: Mapping) -> tuple[int, int]:
    """Write ``status.json`` and ``status.md`` from the records; return passed and total."""
    records = {
        path.parent.name: json.loads(path.read_text())
        for path in Path(output).glob("*/record.json")
    }
    ordered = [records[n] for n in cases if n in records]
    (Path(output) / "status.json").write_text(json.dumps(ordered, indent=2) + "\n")
    lines = [
        "| Case | Family | Mesher | Solver | Result | Cells | Non-orth. | Time s | Objective "
        + "| Notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for record in ordered:
        case = cases[record["case"]]
        mesh = record.get("mesh", {})
        metrics = record.get("metrics", {})
        value = metrics.get(case.objective)
        objective = "" if value is None else f"{case.objective} {value:.4g}"
        notes = "; ".join(record.get("flags", []))
        if not record["passed"]:
            notes = record.get("error", "")[:160].replace("|", "/")
        angle = mesh.get("non_orthogonality")
        lines.append(
            f"| {case.name} | {case.family} | {case.mesher} | {case.solver} "
            f"| {'pass' if record['passed'] else 'FAIL ' + record['stage']} "
            f"| {mesh.get('cells') or ''} | {'' if angle is None else f'{angle:.0f}'} "
            f"| {record.get('wall_s', '')} | {objective} | {notes} |"
        )
    passed = sum(r["passed"] for r in ordered)
    lines.append(f"\n{passed} of {len(ordered)} recorded cases passed.")
    (Path(output) / "status.md").write_text("\n".join(lines) + "\n")
    return passed, len(ordered)
