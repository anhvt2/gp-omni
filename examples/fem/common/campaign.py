"""Constrained batch Bayesian optimization of one design problem.

Every design is solved in its own process: Gmsh keeps global state, so it cannot mesh in
the optimizer's threads, and a crashed or stalled mesh only loses that one design.
"""

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from itertools import count
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock

import numpy as np

from .problem import design_problem

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


@dataclass(frozen=True)
class settings:
    """``initial`` random designs followed by ``batches`` batches of ``batch_size``."""

    initial: int = 12
    batches: int = 8
    batch_size: int = 4
    seed: int = 0
    mesh_scale: float = 1.0
    timeout_s: float = 600.0

    def __post_init__(self):
        if self.initial < 2 or self.batches < 1 or self.batch_size < 1:
            raise ValueError("need initial >= 2, batches >= 1 and batch_size >= 1")
        if not self.mesh_scale > 0 or not self.timeout_s > 0:
            raise ValueError("mesh_scale and timeout_s must be positive")


def evaluate_in_subprocess(
    problem: design_problem, design: dict[str, float], mesh_scale: float, timeout_s: float = 600.0
) -> dict[str, float]:
    """Metrics of one design, solved by its case script in a single-threaded process."""
    environment = os.environ | dict.fromkeys(THREAD_VARIABLES, "1")
    with TemporaryDirectory() as directory:
        request, result = Path(directory, "request.json"), Path(directory, "result.json")
        request.write_text(json.dumps({"design": design, "mesh_scale": mesh_scale}))
        try:
            done = subprocess.run(
                [sys.executable, str(problem.source), "worker", str(request), str(result)],
                check=False,  # the exit status is reported below
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{problem.name} exceeded the {timeout_s:g} s time limit") from None
        if done.returncode != 0:
            last = (done.stderr.strip().splitlines() or ["no error output"])[-1]
            raise RuntimeError(f"{problem.name} failed (exit {done.returncode}): {last}")
        return json.loads(result.read_text())


def run_campaign(problem: design_problem, output: Path, options: settings) -> dict:
    """Optimize ``problem`` into the new directory ``output`` and return its summary.

    The reference design sets the objective scale and stays eligible as the best design.
    """
    from gp_omni.opt.parallel.batchbo import batch_bo, batch_bo_config

    output = Path(output)
    output.mkdir(parents=True)
    reference = _record(
        problem,
        problem.reference_design(),
        evaluate_in_subprocess(
            problem, problem.reference_design(), options.mesh_scale, options.timeout_s
        ),
    )
    scale = reference["metrics"][problem.objective]
    numbers, lock = count(1), Lock()

    def evaluate(x: np.ndarray) -> tuple[float, bool]:
        with lock:
            path = output / f"design_{next(numbers):04d}.json"
        design = problem.design_from_unit(x)
        try:
            metrics = evaluate_in_subprocess(problem, design, options.mesh_scale, options.timeout_s)
        except Exception as error:
            path.write_text(json.dumps({"design": design, "error": str(error)}, indent=2))
            raise  # the optimizer records the design as infeasible
        record = _record(problem, design, metrics)
        path.write_text(json.dumps(record, indent=2))
        return -metrics[problem.objective] / scale, record["feasible"]

    # one slot of each batch learns the feasibility boundary
    boundary = 1 if options.batch_size > 1 else 0
    dimension = len(problem.variables)
    config = batch_bo_config(
        model_name=problem.name,
        x_lb=np.zeros(dimension),
        x_ub=np.ones(dimension),
        num_init_point=options.initial,
        max_iter=options.initial + options.batches,
        exploit_size=options.batch_size - boundary,
        explore_size=0,
        explore_classifier_size=boundary,
        check_time=0.002,
        wait_time=options.timeout_s * (options.initial + 1) / 3600,
        seed=options.seed,
    )
    batch_bo(output / "bo", config, evaluate=evaluate).run()

    records = [json.loads(p.read_text()) for p in sorted(output.glob("design_*.json"))]
    solved = [r for r in records if "error" not in r]
    if not solved:
        raise RuntimeError(f"every candidate design failed; see {output}")
    feasible = [r for r in solved if r["feasible"]]
    eligible = feasible + ([reference] if reference["feasible"] else [])
    best = min(eligible, key=lambda r: r["metrics"][problem.objective], default=None)
    summary = {
        "problem": problem.name,
        "objective": problem.objective,
        "constraints": [asdict(c) for c in problem.constraints],
        "settings": asdict(options),
        "reference": reference,
        "best": best,
        "evaluations": len(records),
        "feasible": len(feasible),
        "failed": len(records) - len(solved),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _record(problem: design_problem, design: dict, metrics: dict) -> dict:
    return {"design": design, "metrics": metrics, "feasible": problem.is_feasible(metrics)}
