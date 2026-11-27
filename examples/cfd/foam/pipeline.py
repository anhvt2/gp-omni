"""Run one design end to end: build, mesh, check, solve, measure and record."""

import json
import math
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

from .case import cfd_case, mesh_level
from .dictionaries import write_dictionary

SOLVERS = (
    "simpleFoam", "pimpleFoam", "rhoSimpleFoam", "rhoPimpleFoam", "rhoCentralFoam",
    "interFoam", "buoyantSimpleFoam",
)  # fmt: skip
NON_ORTHOGONALITY_LIMIT = 85.0
IMBALANCE_LIMIT = 0.2
MINIMUM_FREE_GB = 30.0


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def require_disk(path: Path, minimum_gb: float) -> None:
    available = free_gb(path)
    if available < minimum_gb:
        raise RuntimeError(
            f"only {available:.1f} GB free at {path}; at least {minimum_gb:g} GB is required"
        )


def parse_check_mesh(text: str) -> dict:
    def number(pattern: str, kind=float):
        found = re.search(pattern, text)
        return kind(found.group(1)) if found else None

    failed = number(r"Failed (\d+) mesh checks", int)
    return {
        "cells": number(r"\n\s*cells:\s+(\d+)", int),
        "non_orthogonality": number(r"non-orthogonality Max: ([\d.eE+-]+)"),
        "skewness": number(r"Max skewness = ([\d.eE+-]+)"),
        "failed_checks": failed or 0,
        "negative_volume": bool(
            re.search(r"negative cell volume|negative volume cells", text, re.IGNORECASE)
        ),
    }


def gate_mesh(report: dict) -> None:
    if not report["cells"]:
        raise ValueError("checkMesh reported no cells")
    if report["negative_volume"]:
        raise ValueError("the mesh has negative-volume cells")
    angle = report["non_orthogonality"]
    if angle is not None and angle >= NON_ORTHOGONALITY_LIMIT:
        raise ValueError(f"non-orthogonality {angle:.1f} is not below {NON_ORTHOGONALITY_LIMIT:g}")


def check_solver_log(text: str, solver: str) -> None:
    if re.search(r"(Initial residual = |sum local = |Final residual = )-?(nan|inf)\b", text):
        raise ValueError(f"{solver} produced non-finite residuals")
    if not re.search(r"^End\s*$", text, re.MULTILINE):
        raise ValueError(f"{solver} did not reach the end of the run")


class _runner:
    def __init__(self, directory: Path, timeout_s: float):
        self.directory = directory
        self.deadline = time.monotonic() + timeout_s
        self.counts: dict[str, int] = {}

    def __call__(self, command: list[str], name: str | None = None) -> str:
        name = name or Path(command[0]).name
        self.counts[name] = self.counts.get(name, 0) + 1
        suffix = "" if self.counts[name] == 1 else f".{self.counts[name]}"
        log_path = self.directory / f"log.{name}{suffix}"
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError(f"time limit reached before {name}")
        with log_path.open("w") as log:
            try:
                done = subprocess.run(
                    command, cwd=self.directory, stdout=log, stderr=subprocess.STDOUT,
                    check=False, timeout=remaining,
                )  # fmt: skip
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"{name} exceeded the time limit; see {log_path.name}") from None
            except FileNotFoundError:
                raise RuntimeError(
                    f"{command[0]} not found. Run inside the OpenFOAM container."
                ) from None
        text = log_path.read_text(errors="replace")
        if done.returncode != 0:
            raise RuntimeError(
                f"{name} failed (exit {done.returncode}); {log_path.name}: {_tail(text)}"
            )
        return text


def _tail(text: str) -> str:
    fatal = text.find("FOAM FATAL")
    excerpt = text[fatal : fatal + 600] if fatal >= 0 else text[-600:]
    return " ".join(excerpt.split())


def run_design(
    case: cfd_case,
    design: dict[str, float],
    level: mesh_level,
    directory: Path,
    timeout_s: float = 1800.0,
    cores: int = 1,
    keep_fields: bool = False,
    minimum_free_gb: float = MINIMUM_FREE_GB,
) -> dict:
    """Run ``design`` in the new ``directory`` and return its record.

    A failed stage gives a record with ``passed`` false instead of an exception. An
    existing directory or a nearly full disk raise, since nothing can be run safely.
    """
    case.check_design(design)
    if cores < 1:
        raise ValueError("cores must be positive")
    directory = Path(directory).resolve()
    directory.parent.mkdir(parents=True, exist_ok=True)
    require_disk(directory.parent, minimum_free_gb)
    directory.mkdir()
    started = time.monotonic()
    record = {
        "case": case.name,
        "design": design,
        "level": level.name,
        "solver": case.solver,
        "mesher": case.mesher,
        "openfoam_version": os.environ.get("WM_PROJECT_VERSION", "unknown"),
        "cores": cores,
        "passed": False,
        "stage": "build",
    }
    _write_record(directory, record)  # marks the directory as a run, even if interrupted
    run = _runner(directory, timeout_s)
    try:
        plan = case.build(directory, design, level)
        record["stage"] = "mesh"
        for command in plan.mesh:
            run(command)
        record["stage"] = "check_mesh"
        record["mesh"] = parse_check_mesh(run(["checkMesh"]))
        gate_mesh(record["mesh"])
        record["stage"] = "solve"
        _solve(run, directory, plan.solve, cores)
        record["stage"] = "measure"
        metrics = {k: float(v) for k, v in plan.measure(directory).items()}
        record["metrics"] = metrics
        needed = {case.objective, *(c.metric for c in case.constraints)}
        absent = needed - set(metrics)
        if absent:
            raise ValueError(f"missing metrics {sorted(absent)}")
        broken = sorted(k for k, v in metrics.items() if not math.isfinite(v))
        if broken:
            raise ValueError(f"non-finite metrics {broken}")
        record["flags"] = case.flags(metrics)
        imbalance = metrics.get("flow_imbalance")
        if imbalance is not None and imbalance > IMBALANCE_LIMIT:
            raise ValueError(f"flow imbalance {imbalance:.2f} exceeds {IMBALANCE_LIMIT:g}")
        record["stage"] = "done"
        record["passed"] = True
    except Exception as error:  # noqa: BLE001 (every failure is recorded with its stage)
        record["error"] = f"{type(error).__name__}: {error}"
    record["wall_s"] = round(time.monotonic() - started, 1)
    _write_record(directory, record)
    if record["passed"] and not keep_fields:
        remove_fields(directory)
    return record


def _write_record(directory: Path, record: dict) -> None:
    partial = directory / "record.json.partial"
    partial.write_text(json.dumps(record, indent=2) + "\n")
    partial.replace(directory / "record.json")


def clear_unfinished(directory: Path, case_name: str, level_name: str) -> bool:
    """Prepare ``directory`` for ``verify``; return true if it already holds a passing run.

    Only an earlier run of the same case and mesh level is ever deleted.
    """
    if not directory.exists():
        return False
    path = directory / "record.json"
    record = json.loads(path.read_text()) if path.is_file() else {}
    if record.get("case") != case_name:
        raise RuntimeError(f"{directory} is not a run of {case_name}; choose another --output")
    if record.get("level") != level_name:
        raise RuntimeError(
            f"{directory} holds a {record.get('level')} run; use another --output for {level_name}"
        )
    if record["passed"]:
        return True
    shutil.rmtree(directory)
    return False


def _solve(run: _runner, directory: Path, commands: list[list[str]], cores: int) -> None:
    for command in commands:
        solver = command[0]
        if solver not in SOLVERS:
            run(command)
            continue
        if cores == 1:
            check_solver_log(run(command), solver)
            continue
        write_dictionary(
            directory / "system/decomposeParDict",
            {"numberOfSubdomains": cores, "method": "scotch"},
        )
        run(["decomposePar", "-force"])
        parallel = ["mpirun", "-np", str(cores), "--oversubscribe", *command, "-parallel"]
        check_solver_log(run(parallel, name=solver), solver)
        run(["reconstructPar", "-latestTime"])


def remove_fields(directory: Path) -> None:
    """Delete meshes and field data, keeping dictionaries, logs, measurements and records."""
    for entry in directory.iterdir():
        if not entry.is_dir():
            continue
        later_time = _is_number(entry.name) and float(entry.name) > 0
        if later_time or entry.name.startswith("processor") or entry.name == "VTK":
            shutil.rmtree(entry)
    for name in ("constant/polyMesh", "constant/triSurface", "constant/extendedFeatureEdgeMesh"):
        shutil.rmtree(directory / name, ignore_errors=True)


def _is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True
