"""Reverse-flow cyclone."""

import re
from pathlib import Path

from foam import measure as post
from foam.boundaries import flow_inlet, pressure_outlet, wall
from foam.cad import fluid_surfaces, fuse, name_caps, tube
from foam.case import at_most, run_plan, variable
from foam.dictionaries import write_dictionary

from ..internal_cad import AIR_DENSITY, _case, _snappy_fluid, _steady, _total


def cyclone_separator(case, design, level):
    """A reverse-flow cyclone: converged gas flow, then one-way coupled dust tracking."""
    body = 0.1
    tip, finder = design["cone_tip_radius_m"], design["finder_radius_m"]
    width = design["inlet_width_m"]
    height = 0.09

    def build(occ):
        barrel = tube(occ, (0, 0, 0.0), (0, 0, 0.2), body)
        cone = tube(occ, (0, 0, -0.25), (0, 0, 0.0), tip, body)
        duct = (3, occ.addBox(-0.18, body - width, 0.2 - height - 0.005, 0.18, width, height))
        parts = fuse(occ, [barrel, cone, duct, tube(occ, (0, 0, 0.2), (0, 0, 0.3), finder)])
        # the vortex finder wall is a thin sleeve reaching down into the barrel
        outer = tube(occ, (0, 0, 0.09), (0, 0, 0.2), finder + 0.006)
        inner = tube(occ, (0, 0, 0.09), (0, 0, 0.2), finder)
        sleeve, _ = occ.cut([outer], [inner])
        fluid, _ = occ.cut(parts, sleeve)
        return fluid

    caps = {
        "inlet": (-0.18, body - width / 2, 0.2 - height / 2 - 0.005),
        "outlet": (0, 0, 0.3),
        "dust": (0, 0, -0.25),
    }
    surfaces, _ = fluid_surfaces(build, name_caps(caps), 0.01)
    rate = 15.0 * width * height
    commands, patches = _snappy_fluid(
        case, level, surfaces,
        {"inlet": flow_inlet(rate), "outlet": pressure_outlet(), "walls": wall(), "dust": wall()},
        0.012, (0.0013, 0.0017, -0.0711), (1, 1),
    )  # fmt: skip
    functions = post.flux("inlet", "outlet") | post.total_pressure("inlet", "outlet")
    _steady(case, patches, 15.0, width, level, functions)
    _write_particles(case, level)

    def measure(directory):
        logs = (directory / "log.icoUncoupledKinematicParcelFoam").read_text()
        injected, captured = _particle_fate(logs)
        return {
            "flow_imbalance": post.imbalance(directory, "inlet", "outlet"),
            "capture_fraction": captured / injected,
            "pressure_drop_pa": AIR_DENSITY
            * (_total(directory, "inlet") - _total(directory, "outlet")),
        }

    return run_plan(
        mesh=commands,
        solve=[
            ["simpleFoam"],
            ["cp", "system/controlDict.particles", "system/controlDict"],
            ["icoUncoupledKinematicParcelFoam"],
        ],
        measure=measure,
    )


def _write_particles(case: Path, level) -> None:
    """Steady tracking through the converged flow: one step follows every parcel."""
    start = level.iterations
    write_dictionary(
        case / "system/controlDict.particles",
        {
            "application": "icoUncoupledKinematicParcelFoam",
            "startFrom": "startTime", "startTime": start, "stopAt": "endTime",
            "endTime": start + 1, "deltaT": 1, "writeControl": "timeStep",
            "writeInterval": 1, "writeFormat": "binary", "timePrecision": 10,
            "runTimeModifiable": False,
        },
    )  # fmt: skip
    write_dictionary(
        case / "constant/g",
        {"dimensions": "[0 1 -2 0 0 0 0]", "value": (0, 0, -9.81)},
        "uniformDimensionedVectorField",
    )
    write_dictionary(
        case / "constant/transportProperties",
        {"transportModel": "Newtonian", "nu": 1.5e-5, "rhoInf": AIR_DENSITY},
    )
    write_dictionary(
        case / "constant/kinematicCloudProperties",
        {
            "solution": {
                "active": True, "coupled": False, "transient": "yes",
                "cellValueSourceCorrection": "off", "maxCo": 0.3, "sourceTerms": {"schemes": {}},
                "calcFrequency": 1, "maxTrackTime": 5.0,
                "interpolationSchemes": {"rho": "cell", "U": "cellPoint", "mu": "cell"},
                "integrationSchemes": {"U": "Euler"},
            },
            "constantProperties": {"rho0": 2000, "youngsModulus": 6e8, "poissonsRatio": 0.35},
            "subModels": {
                "particleForces": {"sphereDrag": None, "gravity": None},
                "injectionModels": {
                    "dust": {
                        "type": "patchInjection", "patch": "inlet", "SOI": start,
                        "duration": 1, "parcelsPerSecond": 2000, "massTotal": 1e-4,
                        "massFlowRate": "constant 0.001",
                        "parcelBasisType": "mass", "U0": (15, 0, 0),
                        "flowRateProfile": "constant 1",
                        "sizeDistribution": {
                            "type": "fixedValue", "fixedValueDistribution": {"value": 2e-5},
                        },
                    }
                },
                "dispersionModel": "none",
                "patchInteractionModel": "localInteraction",
                "localInteractionCoeffs": {
                    "patches": (
                        '( dust { type stick; } "(inlet|outlet)" { type escape; } '
                        '"(walls|outside)" { type rebound; e 0.9; mu 0.1; } )'
                    ),
                },
                "surfaceFilmModel": "none",
                "stochasticCollisionModel": "none",
                "collisionModel": "none",
            },
            "cloudFunctions": {},
        },
    )  # fmt: skip


def _particle_fate(log: str) -> tuple[int, int]:
    """Parcels injected and parcels held on the dust patch, from the last cloud report."""
    added = re.findall(r"parcels added\s*=\s*(\d+)", log)
    held = re.findall(r"patch dust[^\n]*\n(?:\s*-[^\n]*\n)*?\s*-\s*stick\s*=\s*(\d+)", log)
    if not added or not int(added[-1]):
        raise ValueError("no parcels were injected")
    if not held:
        raise ValueError("the cloud report holds no fate for the dust patch")
    return int(added[-1]), int(held[-1])


CASE = _case(
    "cyclone_separator",
    "Reverse-flow cyclone",
    cyclone_separator,
    (
        variable("cone_tip_radius_m", 0.03, 0.05, 0.038),
        variable("finder_radius_m", 0.035, 0.055, 0.045),
        variable("inlet_width_m", 0.03, 0.05, 0.04),
    ),
    "capture_fraction",
    (at_most("pressure_drop_pa", 0.2),),
    maximize=True,
    positive=("pressure_drop_pa",),
    simplification="one-way coupled particles of one size",
)
