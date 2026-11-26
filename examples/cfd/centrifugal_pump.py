"""Size a centrifugal pump and select its speed using steady OpenFOAM simulations."""

import json
import logging
import shutil
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from tempfile import mkdtemp

import numpy as np

FLOW_M3_S = 0.0416666667
DENSITY_KG_M3 = 1_000
REFERENCE_SPEED_RAD_S = 314.0
MEASUREMENT_INTERVAL = 10
TAIL_SAMPLES = 10
FLOW_TOLERANCE = 0.03


def surface_monitor(name: str, patch: str, operation: str, field: str) -> str:
    return f"""
    {name}
    {{
        type surfaceFieldValue;
        libs (fieldFunctionObjects);
        regionType patch;
        name {patch};
        operation {operation};
        fields ({field});
        writeFields false;
        writeControl timeStep;
        writeInterval {MEASUREMENT_INTERVAL};
    }}
"""


def control_dict(iterations: int) -> str:
    monitors = "".join(
        surface_monitor(name, patch, operation, field)
        for name, patch, operation, field in (
            ("inletPressure", "Inlet", "areaAverage", "p"),
            ("outletPressure", "Outlet", "areaAverage", "p"),
            ("inletFlow", "Inlet", "sum", "phi"),
            ("outletFlow", "Outlet", "sum", "phi"),
        )
    )
    return f"""
FoamFile
{{
    version 2.0;
    format ascii;
    class dictionary;
    object controlDict;
}}
application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {iterations};
deltaT 1;
writeControl timeStep;
writeInterval {iterations};
writeFormat binary;
runTimeModifiable false;

functions
{{
{monitors}
    impellerForces
    {{
        type forces;
        libs (forces);
        patches (Wall_2);
        rho rhoInf;
        rhoInf {DENSITY_KG_M3};
        CofR (0 0 0);
        writeControl timeStep;
        writeInterval {MEASUREMENT_INTERVAL};
    }}
}}
"""


def run_command(case: Path, timeout: float, *command: str) -> None:
    log_path = case / f"log.{command[0]}"
    with log_path.open("a") as log:
        log.write(f"$ {' '.join(command)}\n")
        log.flush()
        try:
            subprocess.run(
                command, cwd=case, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=timeout
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(f"{command[0]} failed. See {log_path}") from exc


def read_monitor(case: Path, name: str, filename: str, iterations: int) -> np.ndarray:
    directory = case / "postProcessing" / name
    paths = list(directory.glob(f"*/{filename}"))
    if len(paths) != 1:
        raise ValueError(f"Expected one {filename} file in {directory}, found {len(paths)}")
    path = paths[0]
    rows = np.loadtxt(path, ndmin=2)
    tail = rows[-TAIL_SAMPLES:]
    last_sample = iterations // MEASUREMENT_INTERVAL * MEASUREMENT_INTERVAL
    expected = last_sample - MEASUREMENT_INTERVAL * np.arange(TAIL_SAMPLES - 1, -1, -1)
    if (
        tail.shape[0] != TAIL_SAMPLES
        or tail.shape[1] < 2
        or not np.array_equal(tail[:, 0], expected)
    ):
        raise ValueError(f"Incomplete monitor history in {path}")
    if not np.isfinite(tail).all():
        raise ValueError(f"Nonfinite monitor values in {path}")
    return tail[:, 1]


def measure(case: Path, speed: float, iterations: int) -> dict[str, float]:
    inlet_p, outlet_p, inlet_q, outlet_q = (
        read_monitor(case, name, "surfaceFieldValue.dat", iterations)
        for name in ("inletPressure", "outletPressure", "inletFlow", "outletFlow")
    )
    moment_x = read_monitor(case, "impellerForces", "moment.dat", iterations)
    # The rotor spins about negative x. The fluid moment about positive x resists it.
    pressure_rise = DENSITY_KG_M3 * (outlet_p - inlet_p)
    shaft_power = speed * moment_x
    if not (
        np.allclose(inlet_q, -FLOW_M3_S, rtol=FLOW_TOLERANCE, atol=0)
        and np.allclose(outlet_q, FLOW_M3_S, rtol=FLOW_TOLERANCE, atol=0)
    ):
        raise ValueError(f"Flow is outside {FLOW_TOLERANCE:.0%} of the prescribed duty in {case}")
    for values in (pressure_rise, shaft_power):
        if np.ptp(values) > 0.03 * abs(values.mean()):
            raise ValueError(f"Pressure rise or shaft power is still changing in {case}")
    if shaft_power.mean() <= 0:
        raise ValueError(f"Nonpositive shaft power in {case}")
    return {
        "pressure_rise_pa": float(pressure_rise.mean()),
        "shaft_power_w": float(shaft_power.mean()),
        "outlet_flow_m3_s": float(outlet_q.mean()),
    }


def simulate(
    upstream: Path,
    case: Path,
    scale: float,
    speed_ratio: float,
    iterations: int,
    timeout: float,
) -> dict[str, float]:
    for folder in ("0", "constant", "system"):
        shutil.copytree(upstream / "volute" / folder, case / folder)
    shutil.copytree(upstream / "transient_case/constant/polyMesh", case / "constant/polyMesh")
    (case / "system/controlDict").write_text(control_dict(iterations))
    speed = REFERENCE_SPEED_RAD_S * speed_ratio
    design = {"scale": scale, "speed_rad_s": speed}
    (case / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    run_command(
        case,
        timeout,
        "foamDictionary",
        "constant/MRFProperties",
        "-entry",
        "MRF1/omega",
        "-set",
        f"{-speed:.10g}",
    )
    run_command(case, timeout, "transformPoints", "-scale", f"({scale} {scale} {scale})")
    run_command(case, timeout, "simpleFoam")
    result = design | measure(case, speed, iterations)
    (case / "measurements.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("upstream", type=Path, help="checkout of FreeCAD_OpenFOAM_Volute_Pump")
    parser.add_argument("output", type=Path, help="new directory for cases, logs, and BO data")
    parser.add_argument("--baseline-only", action="store_true", help="solve the reference case")
    parser.add_argument("--iterations", type=int, default=3000, help="SIMPLE iterations per case")
    parser.add_argument(
        "--batches", type=int, default=3, help="BO batches after the initial designs"
    )
    parser.add_argument(
        "--timeout-hours", type=float, default=4, help="wall time limit per OpenFOAM command"
    )
    args = parser.parse_args()
    if args.iterations < MEASUREMENT_INTERVAL * TAIL_SAMPLES:
        parser.error("at least 100 SIMPLE iterations are required")
    if args.batches < 1:
        parser.error("--batches must be positive")
    if not np.isfinite(args.timeout_hours) or args.timeout_hours <= 0:
        parser.error("--timeout-hours must be positive and finite")
    for command in ("foamDictionary", "transformPoints", "simpleFoam"):
        if shutil.which(command) is None:
            parser.error(f"{command} is missing. Source the OpenFOAM environment first.")
    for required in (
        "volute/0/U",
        "volute/system/fvSolution",
        "volute/constant/MRFProperties",
        "transient_case/constant/polyMesh/cellZones",
    ):
        if not (args.upstream / required).is_file():
            parser.error(f"upstream case is missing {required}")

    logging.basicConfig(level=logging.WARNING)
    args.output.mkdir(parents=True)
    baseline_dir = args.output / "baseline"
    baseline_dir.mkdir()
    timeout = args.timeout_hours * 3600
    baseline = simulate(args.upstream, baseline_dir, 1.0, 1.0, args.iterations, timeout)
    minimum_pressure = 0.9 * baseline["pressure_rise_pa"]
    if minimum_pressure <= 0:
        raise ValueError(f"Reference case produced no positive pressure rise. See {baseline_dir}")
    print(f"Reference design: {json.dumps(baseline)}", flush=True)
    if args.baseline_only:
        return

    from gp_omni.opt.parallel.batchbo import batch_bo, batch_bo_config

    def evaluate(x: np.ndarray) -> tuple[float, bool]:
        case = Path(mkdtemp(prefix="design-", dir=args.output))
        result = simulate(args.upstream, case, *x, args.iterations, timeout)
        return -result["shaft_power_w"], result["pressure_rise_pa"] >= minimum_pressure

    initial_designs = 4
    config = batch_bo_config(
        model_name="pump",
        x_lb=np.array([0.9, 0.85]),
        x_ub=np.array([1.1, 1.15]),
        num_init_point=initial_designs,
        max_iter=initial_designs + args.batches,
        exploit_size=2,
        explore_size=0,
        explore_classifier_size=0,
        check_time=0.02,
        wait_time=args.timeout_hours * 3 + 0.1,
        seed=0,
    )
    batch_bo(args.output / "bo", config, evaluate=evaluate).run()
    completed = [
        json.loads(path.read_text()) for path in args.output.glob("design-*/measurements.json")
    ]
    candidates = [baseline]
    candidates.extend(
        result for result in completed if result["pressure_rise_pa"] >= minimum_pressure
    )
    best = min(candidates, key=lambda result: result["shaft_power_w"])
    print(f"Required static pressure rise {minimum_pressure:.1f} Pa")
    print(f"Candidates with valid solver measurements {len(completed)}")
    print(f"Best feasible design: {json.dumps(best)}")


if __name__ == "__main__":
    main()
