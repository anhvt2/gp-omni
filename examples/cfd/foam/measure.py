"""Function objects written into ``controlDict`` and the readers of their output.

Each builder returns ``{name: settings}``; each reader takes the case directory and the same
arguments. Values come from the last sample, which is enough for smoke-level runs.
"""

from pathlib import Path

import numpy as np

INTERVAL = 5
_COMMON = {"writeControl": "timeStep", "writeInterval": INTERVAL}
_FIELDS = "(fieldFunctionObjects)"


def surface_value(name: str, patch: str, operation: str, field: str, weight: str | None = None):
    settings = {
        "type": "surfaceFieldValue",
        "libs": _FIELDS,
        "regionType": "patch",
        "name": patch,
        "operation": operation,
        "fields": (field,),
        "writeFields": False,
        "log": False,
        **_COMMON,
    }
    if weight:
        settings["weightField"] = weight
    return {name: settings}


def flux(*patches: str) -> dict:
    result = {}
    for patch in patches:
        result |= surface_value(f"flux_{patch}", patch, "sum", "phi")
    return result


def average(field: str, *patches: str) -> dict:
    result = {}
    for patch in patches:
        result |= surface_value(f"{field}_{patch}", patch, "areaAverage", field)
    return result


def total_pressure(*patches: str) -> dict:
    """Flux-weighted total pressure per density of incompressible flow."""
    result = {
        "total_pressure": {
            "type": "pressure",
            "libs": _FIELDS,
            "mode": "total",
            "rho": "rhoInf",
            "rhoInf": 1,
            "result": "total_p",
            "writeControl": "none",
            "log": False,
        }
    }
    for patch in patches:
        result |= surface_value(
            f"total_p_{patch}", patch, "weightedAverage", "total_p", weight="phi"
        )
    return result


def compressible_total_pressure(*patches: str) -> dict:
    """Flux-weighted ``p + rho |U|^2 / 2`` of compressible flow."""
    result = {
        "total_pressure": {
            "type": "pressure",
            "libs": _FIELDS,
            "mode": "total",
            "result": "total_p",
            "writeControl": "none",
            "log": False,
        }
    }
    for patch in patches:
        result |= surface_value(
            f"total_p_{patch}", patch, "weightedAverage", "total_p", weight="phi"
        )
    return result


def forces(
    name: str, patches, density: float | None, centre=(0, 0, 0), reference_pressure: float = 0.0
) -> dict:
    """Force and moment of the fluid on ``patches``; ``density`` is ``None`` if compressible."""
    settings = {
        "type": "forces",
        "libs": "(forces)",
        "patches": tuple(patches),
        "CofR": centre,
        "log": False,
        **_COMMON,
    }
    if density is not None:
        settings |= {"rho": "rhoInf", "rhoInf": density}
    if reference_pressure:
        settings["pRef"] = reference_pressure
    return {name: settings}


def probes(name: str, points, field: str = "p") -> dict:
    return {
        name: {
            "type": "probes",
            "libs": "(sampling)",
            "fields": (field,),
            "probeLocations": tuple(tuple(p) for p in points),
            "writeControl": "timeStep",
            "writeInterval": 1,
        }
    }


def volume_value(name: str, operation: str, field: str, zone: str | None = None) -> dict:
    settings = {
        "type": "volFieldValue",
        "libs": _FIELDS,
        "operation": operation,
        "fields": (field,),
        "writeFields": False,
        "log": False,
        "regionType": "cellZone" if zone else "all",
        **_COMMON,
    }
    if zone:
        settings["name"] = zone
    return {name: settings}


def scalar_transport(field: str) -> dict:
    return {
        f"transport_{field}": {
            "type": "scalarTransport",
            "libs": "(solverFunctionObjects)",
            "field": field,
            "schemesField": field,
            "nCorr": 1,
            "writeControl": "writeTime",
            "log": False,
        }
    }


def read_table(case: Path, name: str, filename: str) -> np.ndarray:
    """Numeric rows of the latest ``postProcessing/<name>/*/<filename>``."""
    root = Path(case) / "postProcessing" / name
    paths = sorted(root.glob(f"*/{filename}"), key=lambda p: float(p.parent.name))
    if not paths:
        raise ValueError(f"no {filename} below {root}")
    rows = [
        [float(v) for v in line.replace("(", " ").replace(")", " ").split()]
        for line in paths[-1].read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not rows:
        raise ValueError(f"{paths[-1]} holds no samples")
    return np.array(rows)


def last_surface(case: Path, name: str) -> float:
    return float(read_table(case, name, "surfaceFieldValue.dat")[-1, 1])


def mean_surface(case: Path, name: str) -> float:
    """Average over the second half of the samples, for oscillating transients."""
    rows = read_table(case, name, "surfaceFieldValue.dat")
    return float(rows[len(rows) // 2 :, 1].mean())


def last_volume(case: Path, name: str) -> float:
    return float(read_table(case, name, "volFieldValue.dat")[-1, 1])


def last_flux(case: Path, patch: str) -> float:
    return last_surface(case, f"flux_{patch}")


def imbalance(case: Path, *patches: str) -> float:
    """Net flux relative to the largest patch flux; zero for a conservative solution."""
    values = np.array([last_flux(case, p) for p in patches])
    largest = np.abs(values).max()
    return float(abs(values.sum()) / largest) if largest > 0 else float("inf")


def last_force(case: Path, name: str) -> np.ndarray:
    return read_table(case, name, "force.dat")[-1, 1:4]


def last_moment(case: Path, name: str) -> np.ndarray:
    return read_table(case, name, "moment.dat")[-1, 1:4]


def probe_fluctuation(case: Path, name: str, field: str = "p") -> float:
    """Largest standard deviation over the probes, ignoring the first third of the run."""
    rows = read_table(case, name, field)
    return float(rows[len(rows) // 3 :, 1:].std(axis=0).max())
