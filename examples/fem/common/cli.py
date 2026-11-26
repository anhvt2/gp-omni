"""The command line shared by every case script."""

import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

from . import campaign
from .problem import design_problem


def describe(problem: design_problem) -> str:
    limits = ", ".join(f"{c.metric} {c.sense} {c.limit:g}" for c in problem.constraints)
    bounds = "\n".join(
        f"    {v.name:28s} [{v.lower:g}, {v.upper:g}]  reference {v.reference:g}"
        for v in problem.variables
    )
    return (
        f"{problem.name}: {problem.title}\n"
        f"  minimize {problem.objective} subject to {limits}\n{bounds}"
    )


def report(problem: design_problem, record: dict | None) -> str:
    if record is None:
        return "  no feasible design found"
    design = ", ".join(f"{k} {v:.3g}" for k, v in record["design"].items())
    checks = [f"{problem.objective} {record['metrics'][problem.objective]:.4g}"]
    checks += [
        f"{c.metric} {record['metrics'][c.metric]:.4g} ({c.sense} {c.limit:g})"
        for c in problem.constraints
        if c.metric != problem.objective
    ]
    state = "feasible" if record["feasible"] else "infeasible"
    return f"  {state}: {', '.join(checks)}\n  {design}"


def solve_request(problem: design_problem, request: Path, result: Path) -> None:
    """The worker side of :func:`campaign.evaluate_in_subprocess`."""
    job = json.loads(request.read_text())
    metrics = problem.analyze(job["design"], job["mesh_scale"])
    result.write_text(json.dumps(metrics))


def main(problem: design_problem) -> None:
    if sys.argv[1:2] == ["worker"]:
        solve_request(problem, Path(sys.argv[2]), Path(sys.argv[3]))
        return
    defaults = campaign.settings()
    parser = ArgumentParser(description=f"{problem.title}: parametric 3D FEM optimization.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("describe", help="show the variables, bounds and limits")
    baseline = commands.add_parser("baseline", help="solve the reference design")
    optimize = commands.add_parser("optimize", help="run an optimization campaign")
    for command in (baseline, optimize):
        command.add_argument(
            "--mesh-scale",
            type=float,
            default=defaults.mesh_scale,
            help="element size factor; below 1 refines the mesh",
        )
    optimize.add_argument("output", type=Path, help="new directory for the campaign")
    optimize.add_argument("--initial", type=int, default=defaults.initial)
    optimize.add_argument("--batches", type=int, default=defaults.batches)
    optimize.add_argument("--batch-size", type=int, default=defaults.batch_size)
    optimize.add_argument("--seed", type=int, default=defaults.seed)
    optimize.add_argument(
        "--timeout-s", type=float, default=defaults.timeout_s, help="time limit per design"
    )
    args = parser.parse_args()

    if args.command == "describe":
        print(describe(problem))
        return
    logging.basicConfig(level=logging.WARNING)
    if args.command == "baseline":
        design = problem.reference_design()
        metrics = campaign.evaluate_in_subprocess(problem, design, args.mesh_scale)
        print("\n".join(f"{name} {value:.5g}" for name, value in metrics.items()))
        return
    options = campaign.settings(
        initial=args.initial,
        batches=args.batches,
        batch_size=args.batch_size,
        seed=args.seed,
        mesh_scale=args.mesh_scale,
        timeout_s=args.timeout_s,
    )
    summary = campaign.run_campaign(problem, args.output, options)
    print(
        f"{summary['evaluations']} designs, {summary['feasible']} feasible, "
        f"{summary['failed']} failed"
    )
    print(f"reference\n{report(problem, summary['reference'])}")
    print(f"best\n{report(problem, summary['best'])}")
