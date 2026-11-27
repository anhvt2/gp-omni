"""Constrained Bayesian optimization of one CFD case with GP-Omni.

Every design runs in its own process through the command line, so a crashed mesher only
loses that design. Constraint limits come from the reference design, which is run first.
"""

import json
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path
from threading import Lock

import numpy as np

from .case import LEVELS, cfd_case

COMMAND = Path(__file__).resolve().parents[1] / "run.py"


@dataclass(frozen=True)
class settings:
    """``initial`` random designs, then ``batches * batch_size`` proposed designs."""

    initial: int = 8
    batches: int = 4
    batch_size: int = 2
    loop: str = "batch"  # "batch" or "async"
    seed: int = 0
    level: str = "smoke"
    cores: int = 1
    timeout_s: float = 1800.0

    def __post_init__(self):
        if self.initial < 2 or self.batches < 1 or self.batch_size < 1:
            raise ValueError("need initial >= 2, batches >= 1 and batch_size >= 1")
        if self.loop not in ("batch", "async"):
            raise ValueError("loop must be 'batch' or 'async'")
        if self.level not in LEVELS:
            raise ValueError(f"level must be one of {sorted(LEVELS)}")
        if self.cores < 1 or not self.timeout_s > 0:
            raise ValueError("cores and timeout_s must be positive")


def evaluate_in_subprocess(
    case: cfd_case, design: dict, directory: Path, options: settings
) -> dict:
    """Run one design with the command line and return its record."""
    command = [
        sys.executable, str(COMMAND), "run", case.name,
        "--design", json.dumps(design), "--output", str(directory),
        "--level", options.level, "--cores", str(options.cores),
        "--timeout-s", str(options.timeout_s),
    ]  # fmt: skip
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    record = Path(directory) / "record.json"
    if not record.is_file():
        last = (done.stderr.strip().splitlines() or ["no error output"])[-1]
        raise RuntimeError(f"{case.name} produced no record: {last}")
    return json.loads(record.read_text())


def optimize(
    case: cfd_case,
    output: Path,
    options: settings,
    evaluate_design: Callable[[cfd_case, dict, Path, settings], dict] = evaluate_in_subprocess,
) -> dict:
    """Optimize ``case`` into the new directory ``output`` and return its summary."""
    from gp_omni.opt.parallel.asyncbo import async_bo, async_bo_config
    from gp_omni.opt.parallel.batchbo import batch_bo, batch_bo_config

    output = Path(output).resolve()
    output.mkdir(parents=True)
    reference = evaluate_design(case, case.reference_design(), output / "reference", options)
    if not reference["passed"]:
        raise RuntimeError(f"the reference design failed: {reference.get('error')}")
    base = reference["metrics"]
    scale = abs(base[case.objective]) or 1.0
    sign = 1.0 if case.maximize else -1.0
    numbers, lock, records = count(1), Lock(), []

    def evaluate(x: np.ndarray) -> tuple[float, bool]:
        with lock:
            directory = output / f"design_{next(numbers):04d}"
        record = evaluate_design(case, case.design_from_unit(x), directory, options)
        if record["passed"]:
            record["feasible"] = case.feasible(record["metrics"], base)
        with lock:
            records.append(record)
        if not record["passed"]:
            raise RuntimeError(record.get("error", "design failed"))  # recorded as infeasible
        return sign * record["metrics"][case.objective] / scale, record["feasible"]

    # one slot of each batch learns where designs are infeasible
    boundary = 1 if options.batch_size > 1 else 0
    dimension = len(case.variables)
    common = {
        "model_name": case.name,
        "x_lb": np.zeros(dimension),
        "x_ub": np.ones(dimension),
        "num_init_point": options.initial,
        "exploit_size": options.batch_size - boundary,
        "explore_size": 0,
        "explore_classifier_size": boundary,
        "check_time": 0.002,
        "wait_time": options.timeout_s * (options.initial + 1) / 3600,
        "seed": options.seed,
    }
    if options.loop == "batch":
        config = batch_bo_config(max_iter=options.initial + options.batches, **common)
        loop = batch_bo(output / "bo", config, evaluate=evaluate)
    else:
        total = options.initial + options.batches * options.batch_size
        config = async_bo_config(max_iter=total, **common)
        loop = async_bo(output / "bo", config, evaluate=evaluate)
    loop.run()

    solved = [r for r in records if r["passed"]]
    if not solved:
        raise RuntimeError(f"every candidate design failed; see {output}")
    feasible = [r for r in solved if r["feasible"]]
    eligible = [reference, *feasible]  # the reference satisfies its own limits
    pick = max if case.maximize else min
    best = pick(eligible, key=lambda r: r["metrics"][case.objective])
    summary = {
        "case": case.name,
        "objective": case.objective,
        "maximize": case.maximize,
        "constraints": [asdict(c) | {"bound": c.bound(base[c.metric])} for c in case.constraints],
        "settings": asdict(options),
        "reference": reference,
        "best": best,
        "evaluations": len(records),
        "feasible": len(feasible),
        "failed": len(records) - len(solved),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
