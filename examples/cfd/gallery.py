"""Prepare and visualize one OpenFOAM reference case from the CFD index."""

import fcntl
import gzip
import json
import os
import re
import subprocess
import sys
from argparse import ArgumentParser
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

CFD = Path(__file__).resolve().parent
REPOSITORY = CFD.parents[1]
FOAM = CFD / "container/foam.sh"
POST = CFD.parent / "postprocessing"
FIELD_LABELS = {
    ("U", "0 1 -1 0 0 0 0"): "Speed |U| (m/s)",
    ("p", "0 2 -2 0 0 0 0"): "Kinematic pressure (m²/s²)",
    ("p", "1 -1 -2 0 0 0 0"): "Static pressure (Pa)",
    ("T", "0 0 0 1 0 0 0"): "Temperature (K)",
    ("alpha.water", "0 0 0 0 0 0 0"): "Water volume fraction",
    ("egr", "0 0 0 0 0 0 0"): "EGR passive scalar",
    ("rho", "1 -3 0 0 0 0 0"): "Density (kg/m³)",
}
TIME = re.compile(r"\d+(?:\.\d*)?(?:[eE][+-]?\d+)?")


def read_json(path):
    return json.loads(path.read_text())


def container_path(path, runs):
    return (Path("/runs") / path.relative_to(runs)).as_posix()


def run(command, log=None):
    if log is None:
        subprocess.run(command, cwd=REPOSITORY, check=True)
        return
    with log.open("w") as output:
        try:
            subprocess.run(
                command, cwd=REPOSITORY, stdout=output, stderr=subprocess.STDOUT, check=True
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"{command[0]} failed; see {log}") from error


def openfoam_case(case_id, base, runs, stage):
    path = base / stage
    mesh = stage == "mesh"
    if not path.exists():
        command = [
            FOAM,
            "python",
            CFD.relative_to(REPOSITORY) / "run.py",
            "mesh" if mesh else "run",
            case_id,
            "--level",
            "smoke",
            "--output",
            container_path(path, runs),
        ]
        if not mesh:
            command.append("--keep-fields")
        run(command, base / f"{stage}.log")
    marker = path / "record.json"
    if not marker.is_file():
        raise ValueError(f"{path} has no run record; choose a fresh output directory")
    record = read_json(marker)
    if (
        not record.get("passed")
        or record.get("stage") != ("meshed" if mesh else "done")
        or record.get("case") != case_id
        or record.get("level") != "smoke"
    ):
        raise ValueError(f"{marker} is not a passing {stage} record for {case_id} at smoke level")
    if not (path / "constant/polyMesh").is_dir():
        raise ValueError(f"{path} has no retained mesh; choose a fresh output directory")
    return record


def saved_time(case):
    times = [
        path.name
        for path in case.iterdir()
        if path.is_dir() and TIME.fullmatch(path.name) and Decimal(path.name) > 0
    ]
    if not times:
        raise ValueError(f"{case} has no retained solved fields; choose a fresh output directory")
    return max(times, key=Decimal)


def field_label(case, time, field):
    path = case / time / field
    opener = open
    if not path.is_file():
        path = path.with_name(f"{path.name}.gz")
        opener = gzip.open
    with opener(path, "rb") as source:
        header = source.read(8192).decode("ascii", errors="replace")
    dimensions = re.search(r"\bdimensions\s*\[([^]]+)\]", header)
    key = (field, " ".join(dimensions[1].split()) if dimensions else "")
    if key not in FIELD_LABELS:
        raise ValueError(f"unexpected dimensions for {path}: {key[1]!r}")
    return FIELD_LABELS[key]


def render_view(base, runs, stage, time, record, view, pvpython):
    case = base / stage
    output = base / "figures" / stage
    field = view.get("field")
    recipe = {"time": time, "view": view, "record": record}
    if output.exists():
        metadata = read_json(output / "figure.json")
        if metadata.get("run") != recipe or any(
            not (output / name).is_file() for name in ("preview.png", "scene.pvsm", "scene.png")
        ):
            raise ValueError(f"{output} is stale; move it aside and retry")
        return metadata

    label = field_label(case, time, field) if field else None
    source = export_volume(case, runs, stage, time, field)

    output.parent.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=f".{stage}-", dir=output.parent) as temporary:
        draft = Path(temporary) / stage
        command = [
            pvpython,
            "--force-offscreen-rendering",
            POST / "render.py",
            source,
            draft,
            "--length-unit",
            "m",
        ]
        if field:
            command += ["--field", field, "--label", label]
        for option in ("slice", "frame", "view", "up", "line", "range"):
            if option in view:
                command += [f"--{option}", *(str(value) for value in view[option])]
        if "association" in view:
            command += ["--association", view["association"]]
        run(command, base / f"{stage}-render.log")
        run(
            [sys.executable, POST / "publication.py", draft, "--preview"],
            base / f"{stage}-publication.log",
        )
        metadata_path = draft / "figure.json"
        metadata = read_json(metadata_path) | {"run": recipe}
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        draft.rename(output)
    return metadata


def export_volume(case, runs, stage, time, field):
    vtk = case / "galleryVTK"
    receipt = vtk / ".export.json"
    selection = {"time": time, "field": field}
    files = list(vtk.rglob("internal.vtu"))
    if len(files) == 1 and receipt.is_file() and read_json(receipt) == selection:
        return files[0]
    command = [
        FOAM,
        "foamToVTK",
        "-case",
        container_path(case, runs),
        "-time",
        time,
        "-name",
        vtk.name,
        "-overwrite",
        "-no-boundary",
        "-no-lagrangian",
    ]
    command += ["-fields", f"({field})"] if field else ["-no-fields"]
    run(command, case.parent / f"{stage}-vtk.log")
    files = list(vtk.rglob("internal.vtu"))
    if len(files) != 1:
        raise ValueError(f"expected one exported internal mesh in {vtk}")
    receipt.write_text(json.dumps(selection) + "\n")
    return files[0]


def plot_probes(case, output, label):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    files = list((case / "postProcessing/pressure_probes").glob("*/p"))
    if not files:
        raise ValueError(f"{case} has no pressure probe history")
    samples = np.concatenate([np.loadtxt(path, ndmin=2) for path in sorted(files)])
    if samples.shape[1] < 2:
        raise ValueError(f"{case} pressure probe output has no probe columns")
    samples = samples[np.argsort(samples[:, 0], kind="stable")]
    seconds, first = np.unique(samples[:, 0], return_index=True)
    fig, (history, window) = plt.subplots(2, figsize=(7.08, 5.2), sharex=True, layout="constrained")
    start = len(seconds) // 3  # the fluctuation objective ignores the first third
    for index, values in enumerate(samples[first, 1:].T, 1):
        valid = np.isfinite(values) & (np.abs(values) < 1e100)
        if valid.any():
            values = np.where(valid, values, np.nan)
            history.plot(seconds, values, label=f"Probe {index}")
            late = values[start:]
            window.plot(seconds[start:], late - np.nanmean(late), label=f"Probe {index}")
    if not history.lines:
        raise ValueError(f"{case} pressure probe history has no valid samples")
    history.set(title="Full history", ylabel=label)
    window.set(
        title="Final two thirds, mean removed",
        xlabel="Simulation time (s)",
        ylabel=f"Δ {label[0].lower()}{label[1:]}",
    )
    for axis in (history, window):
        axis.grid(alpha=0.3)
    history.legend()
    with TemporaryDirectory(prefix=".probes-", dir=output.parent) as temporary:
        draft = Path(temporary) / output.name
        fig.savefig(draft, dpi=155)
        draft.replace(output)
    plt.close(fig)


def prepare_gallery(case_id, base, runs, recipe, pvpython):
    mesh = openfoam_case(case_id, base, runs, "mesh")
    reference = openfoam_case(case_id, base, runs, "reference")
    if mesh["design"] != reference["design"] or mesh["mesh"] != reference["mesh"]:
        raise ValueError("mesh and reference records use different designs or mesh checks")
    render_view(base, runs, "mesh", "0", mesh, recipe["mesh"], pvpython)
    time = saved_time(base / "reference")
    render_view(base, runs, "reference", time, reference, recipe["reference"], pvpython)
    if recipe.get("pressure_probes"):
        probe_plot = base / "figures/pressure_probes.png"
        if not probe_plot.is_file():
            label = field_label(base / "reference", time, "p")
            plot_probes(base / "reference", probe_plot, label)
    return time


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("case", help="registered CFD case ID")
    parser.add_argument("--output", type=Path, help="case directory under CFD_RUNS (default: runs)")
    parser.add_argument("--pvpython", default="pvpython", help="ParaView Python executable")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.case):
        parser.error("invalid case ID")
    config = CFD / "cases" / args.case / "view.json"
    if not config.is_file():
        parser.error(f"no visualization recipe for {args.case}")
    recipe = read_json(config)
    runs = Path(os.environ.get("CFD_RUNS", REPOSITORY / "runs")).resolve()
    os.environ["CFD_RUNS"] = str(runs)
    base = (args.output or runs / "cfd_gallery" / args.case).resolve()
    if base == runs or not base.is_relative_to(runs):
        parser.error(f"--output must be inside {runs}")

    try:
        base.mkdir(parents=True, exist_ok=True)
        with (base / ".gallery.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"another gallery run is using {base}") from error
            time = prepare_gallery(args.case, base, runs, recipe, args.pvpython)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"{args.case}: {error}\n")

    print(f"{args.case} · {time} · {base / 'figures'}")


if __name__ == "__main__":
    main()
