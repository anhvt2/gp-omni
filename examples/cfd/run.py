"""Command line of the OpenFOAM design cases. Run inside the OpenFOAM container.

    run.py list
    run.py describe axial_fan
    run.py run axial_fan --output /runs/axial_fan
    run.py verify --output /runs/verify
    run.py optimize axial_fan --output /runs/axial_fan_bo --loop batch
"""

import json
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path

from cases import CASES
from foam.campaign import optimize, settings
from foam.case import LEVELS
from foam.pipeline import MINIMUM_FREE_GB, clear_unfinished, require_disk, run_design
from foam.report import write_status


def describe(name: str) -> str:
    case = CASES[name]
    goal = "maximize" if case.maximize else "minimize"
    lines = [
        f"{case.name}: {case.title}",
        f"family {case.family}, mesher {case.mesher}, solver {case.solver}",
        f"{goal} {case.objective}",
    ]
    lines += [
        f"variable {v.name} in [{v.lower}, {v.upper}], reference {v.reference}"
        for v in case.variables
    ]
    lines += [
        f"limit {c.metric} {c.sense} reference with {c.margin:.0%} margin" for c in case.constraints
    ]
    if case.simplification:
        lines.append(f"simplification: {case.simplification}")
    return "\n".join(lines)


def verify(names, output: Path, level: str, cores: int, timeout_s: float, keep: bool) -> bool:
    """Run each reference design once, skipping cases that already passed in ``output``."""
    output.mkdir(parents=True, exist_ok=True)
    for number, name in enumerate(names, 1):
        directory = output / name
        record = directory / "record.json"
        if clear_unfinished(directory, name, level):
            print(f"[{number}/{len(names)}] {name}: passed earlier", flush=True)
            continue
        require_disk(output, MINIMUM_FREE_GB)
        command = [
            sys.executable, __file__, "run", name, "--output", str(directory),
            "--level", level, "--cores", str(cores), "--timeout-s", str(timeout_s),
        ]  # fmt: skip
        if keep:
            command.append("--keep-fields")
        done = subprocess.run(command, capture_output=True, text=True, check=False)
        if record.is_file():
            result = json.loads(record.read_text())
        else:
            result = {
                "case": name, "level": level, "passed": False, "stage": "start",
                "error": (done.stderr.strip().splitlines() or ["no output"])[-1],
            }  # fmt: skip
            directory.mkdir(exist_ok=True)
            record.write_text(json.dumps(result, indent=2) + "\n")
        state = "passed" if result["passed"] else f"FAILED at {result['stage']}"
        detail = "" if result["passed"] else f": {result.get('error', '')[:300]}"
        print(
            f"[{number}/{len(names)}] {name}: {state} ({result.get('wall_s', 0)} s){detail}",
            flush=True,
        )
        write_status(output, CASES)
    passed, total = write_status(output, CASES)
    print(f"{passed} of {total} recorded cases passed. See {output / 'status.md'}")
    return passed == total


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list the cases")
    show = commands.add_parser("describe", help="variables, objective and limits of a case")
    show.add_argument("case", choices=sorted(CASES))
    single = commands.add_parser("run", help="run one design")
    single.add_argument("case", choices=sorted(CASES))
    single.add_argument("--design", type=json.loads, help="JSON object; default is the reference")
    every = commands.add_parser("verify", help="run reference designs and write a status table")
    every.add_argument("--cases", nargs="+", choices=sorted(CASES), help="default: all")
    every.add_argument("--family", choices=sorted({c.family for c in CASES.values()}))
    bo = commands.add_parser("optimize", help="Bayesian optimization of one case")
    bo.add_argument("case", choices=sorted(CASES))
    bo.add_argument("--initial", type=int, default=settings.initial)
    bo.add_argument("--batches", type=int, default=settings.batches)
    bo.add_argument("--batch-size", type=int, default=settings.batch_size)
    bo.add_argument("--loop", choices=("batch", "async"), default=settings.loop)
    bo.add_argument("--seed", type=int, default=settings.seed)
    for sub in (single, every, bo):
        sub.add_argument("--output", type=Path, required=True, help="output directory")
        sub.add_argument("--level", choices=sorted(LEVELS), default="smoke")
        sub.add_argument("--cores", type=int, default=1, help="MPI processes per design")
        sub.add_argument("--timeout-s", type=float, default=1800.0, help="limit per design")
    for sub in (single, every):
        sub.add_argument("--keep-fields", action="store_true", help="keep mesh and field data")
    args = parser.parse_args()

    if args.command == "list":
        for case in CASES.values():
            print(f"{case.name:36s} {case.family:18s} {case.solver:18s} {case.mesher}")
        print(f"{len(CASES)} cases")
    elif args.command == "describe":
        print(describe(args.case))
    elif args.command == "run":
        case = CASES[args.case]
        record = run_design(
            case, args.design or case.reference_design(), LEVELS[args.level], args.output,
            args.timeout_s, args.cores, args.keep_fields,
        )  # fmt: skip
        print(json.dumps(record, indent=2))
        sys.exit(0 if record["passed"] else 1)
    elif args.command == "verify":
        names = args.cases or [
            n for n, c in CASES.items() if args.family in (None, c.family)
        ]
        ok = verify(
            names, args.output, args.level, args.cores, args.timeout_s, args.keep_fields
        )
        sys.exit(0 if ok else 1)
    else:
        options = settings(
            args.initial, args.batches, args.batch_size, args.loop, args.seed,
            args.level, args.cores, args.timeout_s,
        )  # fmt: skip
        summary = optimize(CASES[args.case], args.output, options)
        print(json.dumps(summary["best"], indent=2))


if __name__ == "__main__":
    main()
