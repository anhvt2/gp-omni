"""Branched and shaped internal flows: Gmsh surfaces meshed with ``snappyHexMesh``."""

import re
from math import cos, pi, radians, sin
from pathlib import Path

import numpy as np
from foam import measure as post
from foam.boundaries import flow_inlet, pressure_outlet, wall
from foam.cad import fluid_surfaces, fuse, name_caps, tube
from foam.case import at_least, at_most, cfd_case, run_plan, variable
from foam.dictionaries import write_dictionary
from foam.physics import flow, write_flow_case
from foam.snappy import domain, surface, write_snappy

WATER = {"density": 1000.0, "viscosity": 1e-6}
AIR_DENSITY = 1.2


def _snappy_fluid(case, level, surfaces: dict, patches: dict, cell: float, inside, depth=(1, 2)):
    """Mesh the region enclosed by ``surfaces`` inside a slightly larger box."""
    if set(surfaces) != set(patches):
        raise ValueError(f"surfaces {sorted(surfaces)} do not match {sorted(patches)}")
    everything = np.concatenate([t.reshape(-1, 3) for t in surfaces.values()])
    low, high = everything.min(axis=0), everything.max(axis=0)
    margin = 1.37 * cell
    box = domain(
        tuple(low - margin),
        tuple(high + 1.11 * margin),
        cell,
        {side: "outside" for side in ("x0", "x1", "y0", "y1", "z0", "z1")},
    )
    all_patches = patches | {"outside": wall()}
    commands = write_snappy(
        case,
        box,
        {n: surface(t, depth) for n, t in surfaces.items()},
        all_patches,
        inside,
        level,
    )
    return commands, patches


def _steady(case, patches, speed, length, level, functions, **options):
    write_flow_case(
        case,
        patches,
        flow(
            "simpleFoam",
            velocity=speed,
            length=length,
            iterations=level.iterations,
            second_order=level.resolution > 1,
            functions=functions,
            **options,
        ),
    )


def _total(directory: Path, patch: str) -> float:
    return post.last_surface(directory, f"total_p_{patch}")


def water_valve(case, design, level):
    """A butterfly valve in a 50 mm water line, at 1 litre per second."""
    radius = 0.025
    opening = radians(design["opening_deg"])
    thickness, seat = design["disc_thickness_m"], design["seat_height_m"]
    disc_radius = radius - seat - 0.001

    def build(occ):
        line = [tube(occ, (-0.12, 0, 0), (0.2, 0, 0), radius)]
        # the disc turns about the y axis; at 90 degrees it lies along the flow
        axis = np.array([sin(radians(90) - opening), 0.0, cos(radians(90) - opening)])
        normal = np.array([axis[2], 0.0, -axis[0]])
        start = -0.5 * thickness * normal
        disc = tube(occ, start, start + thickness * normal, disc_radius)
        ring = tube(occ, (-0.034, 0, 0), (-0.026, 0, 0), radius)
        bore = tube(occ, (-0.034, 0, 0), (-0.026, 0, 0), radius - seat)
        seat_ring, _ = occ.cut([ring], [bore])
        fluid, _ = occ.cut(line, [disc, *seat_ring])
        return fluid

    def classify(centre):
        if abs(centre[0] + 0.12) < 1e-6:
            return "inlet"
        if abs(centre[0] - 0.2) < 1e-6:
            return "outlet"
        if np.linalg.norm(centre) < 0.0045:
            return "disc"
        return "walls"

    surfaces, _ = fluid_surfaces(build, classify, 0.004)
    if "disc" not in surfaces:
        raise ValueError("the disc surface was not found")
    rate = 1.0e-3
    commands, patches = _snappy_fluid(
        case, level, surfaces,
        {"inlet": flow_inlet(rate), "outlet": pressure_outlet(), "walls": wall(), "disc": wall()},
        0.006, (-0.0871, 0.0013, 0.0017),
    )  # fmt: skip
    functions = post.flux("inlet", "outlet") | post.total_pressure("inlet", "outlet")
    functions |= post.forces("disc", ["disc"], WATER["density"])
    _steady(case, patches, rate / (pi * radius**2), 2 * radius, level, functions, **WATER)

    def measure(directory):
        return {
            "flow_imbalance": post.imbalance(directory, "inlet", "outlet"),
            "pressure_loss_pa": WATER["density"]
            * (_total(directory, "inlet") - _total(directory, "outlet")),
            "flow_force_n": float(np.linalg.norm(post.last_force(directory, "disc"))),
        }

    return run_plan(mesh=commands, solve=[["simpleFoam"]], measure=measure)


RUNNERS = (0.05, 0.13, 0.21, 0.29)


def intake_manifold_egr(case, design, level):
    """Fresh air and recirculated gas mixing in a plenum feeding four runners."""
    plenum, runner = design["plenum_radius_m"], design["runner_radius_m"]
    port = design["egr_position_m"]

    def build(occ):
        parts = [
            tube(occ, (0.0, 0, 0), (0.34, 0, 0), plenum),
            tube(occ, (-0.14, 0, 0), (0.005, 0, 0), 0.025),
            tube(occ, (port, 0.07, 0), (port, 0, 0), 0.012),
        ]
        parts += [tube(occ, (x, 0, 0), (x, 0, -0.18), runner) for x in RUNNERS]
        return fuse(occ, parts)

    caps = {"inlet": (-0.14, 0, 0), "egr": (port, 0.07, 0)}
    caps |= {f"runner_{n}": (x, 0, -0.18) for n, x in enumerate(RUNNERS, 1)}
    surfaces, _ = fluid_surfaces(build, name_caps(caps), 0.008)
    air, gas = 0.036, 0.004
    outlets = [f"runner_{n}" for n in range(1, 5)]
    commands, patches = _snappy_fluid(
        case, level, surfaces,
        {
            "inlet": flow_inlet(air, egr=0.0), "egr": flow_inlet(gas, egr=1.0),
            **{n: pressure_outlet() for n in outlets}, "walls": wall(),
        },
        0.009, (0.1713, 0.0029, 0.0031), (1, 1),
    )  # fmt: skip
    patches_all = ("inlet", "egr", *outlets)
    functions = post.flux(*patches_all) | post.total_pressure("inlet", *outlets)
    functions |= post.scalar_transport("egr")
    for name in outlets:
        functions |= post.surface_value(
            f"egr_{name}", name, "weightedAverage", "egr", weight="phi"
        )
    _steady(case, patches, 20.0, 0.05, level, functions, scalars=("egr",))

    def measure(directory):
        share = np.array([post.last_surface(directory, f"egr_{n}") for n in outlets])
        flows = np.array([abs(post.last_flux(directory, n)) for n in outlets])
        outlet_total = np.average([_total(directory, n) for n in outlets], weights=flows)
        return {
            "flow_imbalance": post.imbalance(directory, *patches_all),
            "mixture_imbalance": float(np.ptp(share) / max(abs(share.mean()), 1e-9)),
            "flow_imbalance_between_runners": float(np.ptp(flows) / flows.mean()),
            "pressure_loss_pa": AIR_DENSITY * float(_total(directory, "inlet") - outlet_total),
        }

    return run_plan(mesh=commands, solve=[["simpleFoam"]], measure=measure)


def exhaust_manifold(case, design, level):
    """Hot gas from four ports gathering into one collector, a short compressible transient."""
    runner, collector = design["runner_radius_m"], design["collector_radius_m"]
    outlet_x = design["collector_position_m"]
    ports = (0.0, 0.09, 0.18, 0.27)

    def build(occ):
        parts = [tube(occ, (0.0, 0, 0), (0.27, 0, 0), collector)]
        parts += [tube(occ, (x, 0.12, 0), (x, 0, 0), runner) for x in ports]
        parts.append(tube(occ, (outlet_x, 0, 0), (outlet_x, 0, -0.15), collector))
        return fuse(occ, parts)

    inlets = [f"port_{n}" for n in range(1, 5)]
    caps = {n: (x, 0.12, 0) for n, x in zip(inlets, ports)}
    caps["outlet"] = (outlet_x, 0, -0.15)
    surfaces, volume = fluid_surfaces(build, name_caps(caps), 0.008)
    commands, patches = _snappy_fluid(
        case, level, surfaces,
        {
            **{n: flow_inlet(0.008, temperature=800.0) for n in inlets},
            "outlet": pressure_outlet(101325.0), "walls": wall(),
        },
        0.008, (0.0451, 0.0019, 0.0023), (1, 1),
    )  # fmt: skip
    functions = post.flux(*inlets, "outlet")
    functions |= post.compressible_total_pressure(*inlets, "outlet")
    write_flow_case(
        case,
        patches,
        flow(
            "rhoPimpleFoam", velocity=50.0, length=2 * runner, time_step=1e-4,
            time_steps=level.time_steps, temperature=800.0, functions=functions,
        ),
    )  # fmt: skip

    def measure(directory):
        # pressure waves still travel through the manifold, so average over time
        inlet = np.mean([post.mean_surface(directory, f"total_p_{n}") for n in inlets])
        outlet = post.mean_surface(directory, "total_p_outlet")
        return {
            "flow_mismatch": post.imbalance(directory, *inlets, "outlet"),
            "pressure_loss_pa": float(inlet - outlet),
            "volume_m3": volume,
        }

    return run_plan(mesh=commands, solve=[["rhoPimpleFoam"]], measure=measure)


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


def human_upper_airways(case, design, level):
    """An idealized trachea with a narrowing, dividing into two main bronchi."""
    radius = 0.009
    narrow = radius * design["constriction_ratio"]
    position = design["constriction_position"] * 0.1
    angle = radians(design["branch_angle_deg"])
    branch, length = 0.0065, 0.06
    left = (-length * sin(angle), 0.0, -0.12 - length * cos(angle))
    right = (length * sin(angle), 0.0, -0.12 - length * cos(angle))

    def build(occ):
        parts = [
            tube(occ, (0, 0, 0), (0, 0, -position + 0.012), radius),
            tube(occ, (0, 0, -position + 0.012), (0, 0, -position), radius, narrow),
            tube(occ, (0, 0, -position), (0, 0, -position - 0.012), narrow, radius),
            tube(occ, (0, 0, -position - 0.012), (0, 0, -0.12), radius),
            (3, occ.addSphere(0, 0, -0.12, radius)),
            tube(occ, (0, 0, -0.12), left, branch),
            tube(occ, (0, 0, -0.12), right, branch),
        ]
        return fuse(occ, parts)

    caps = {"mouth": (0, 0, 0), "left": left, "right": right}
    surfaces, _ = fluid_surfaces(build, name_caps(caps), 0.002)
    rate = 5.0e-4  # 30 litres per minute
    commands, patches = _snappy_fluid(
        case, level, surfaces,
        {"mouth": flow_inlet(rate), "left": pressure_outlet(), "right": pressure_outlet(),
         "walls": wall()},
        0.0022, (0.0007, 0.0011, -0.0053), (1, 1),
    )  # fmt: skip
    names = ("mouth", "left", "right")
    functions = post.flux(*names) | post.total_pressure(*names)
    _steady(case, patches, rate / (pi * radius**2), 2 * radius, level, functions)

    def measure(directory):
        flows = {n: abs(post.last_flux(directory, n)) for n in names}
        out = np.average(
            [_total(directory, n) for n in names[1:]], weights=[flows["left"], flows["right"]]
        )
        return {
            "flow_imbalance": post.imbalance(directory, *names),
            "pressure_drop_pa": AIR_DENSITY * float(_total(directory, "mouth") - out),
            "left_flow_fraction": flows["left"] / (flows["left"] + flows["right"]),
        }

    return run_plan(mesh=commands, solve=[["simpleFoam"]], measure=measure)


def pelton_distributor(case, design, level):
    """A tapering main pipe feeding three nozzles of a Pelton turbine."""
    angle = radians(design["branch_angle_deg"])
    branch, end = design["branch_radius_m"], design["end_radius_m"]
    main = 0.1
    stations = (0.37, 0.69, 0.97)
    reach = 0.35
    tips = [(x + reach * cos(angle), reach * sin(angle), 0.0) for x in stations]

    def build(occ):
        parts = [
            tube(occ, (0, 0, 0), (0.28, 0, 0), main),
            tube(occ, (0.28, 0, 0), (1.04, 0, 0), main, end),
        ]
        for start, stop in zip(stations, tips):
            middle = tuple(a + 0.75 * (b - a) for a, b in zip((start, 0, 0), stop))
            parts.append(tube(occ, (start, 0, 0), middle, branch))
            parts.append(tube(occ, middle, stop, branch, 0.6 * branch))
        return fuse(occ, parts)

    outlets = ["nozzle_1", "nozzle_2", "nozzle_3"]
    caps = {"inlet": (0, 0, 0), **dict(zip(outlets, tips))}
    surfaces, _ = fluid_surfaces(build, name_caps(caps), 0.02)
    rate = 0.08
    commands, patches = _snappy_fluid(
        case, level, surfaces,
        {"inlet": flow_inlet(rate), **{n: pressure_outlet() for n in outlets}, "walls": wall()},
        0.024, (0.1013, 0.0071, 0.0053), (1, 1),
    )  # fmt: skip
    names = ("inlet", *outlets)
    functions = post.flux(*names) | post.total_pressure(*names)
    _steady(case, patches, rate / (pi * main**2), 2 * main, level, functions, **WATER)

    def measure(directory):
        flows = np.array([abs(post.last_flux(directory, n)) for n in outlets])
        out = np.average([_total(directory, n) for n in outlets], weights=flows)
        return {
            "flow_imbalance": post.imbalance(directory, *names),
            "nozzle_imbalance": float(np.ptp(flows) / flows.mean()),
            "pressure_loss_pa": WATER["density"] * float(_total(directory, "inlet") - out),
        }

    return run_plan(mesh=commands, solve=[["simpleFoam"]], measure=measure)


def _case(name, title, build, variables, objective, constraints, **options):
    solver = options.pop("solver", "simpleFoam")
    return cfd_case(
        name, "internal", title, "snappy", solver, variables, objective, constraints, build,
        **options,
    )  # fmt: skip


CASES = [
    _case(
        "water_valve", "Butterfly valve in a water line", water_valve,
        (
            variable("opening_deg", 35.0, 80.0, 55.0),
            variable("disc_thickness_m", 0.003, 0.007, 0.005),
            variable("seat_height_m", 0.0012, 0.004, 0.0025),
        ),
        "pressure_loss_pa", (at_most("flow_force_n", 0.2),), positive=("pressure_loss_pa",),
    ),
    _case(
        "intake_manifold_egr", "Intake manifold with exhaust gas recirculation",
        intake_manifold_egr,
        (
            variable("plenum_radius_m", 0.032, 0.05, 0.04),
            variable("runner_radius_m", 0.015, 0.022, 0.018),
            variable("egr_position_m", -0.08, -0.025, -0.05),
        ),
        "mixture_imbalance", (at_most("pressure_loss_pa", 0.2),),
        positive=("pressure_loss_pa",), simplification="steady flow, passive scalar mixing",
    ),
    _case(
        "exhaust_manifold", "Four-into-one exhaust manifold", exhaust_manifold,
        (
            variable("runner_radius_m", 0.014, 0.022, 0.018),
            variable("collector_radius_m", 0.022, 0.034, 0.028),
            variable("collector_position_m", 0.06, 0.21, 0.135),
        ),
        "pressure_loss_pa", (at_most("volume_m3"),), solver="rhoPimpleFoam",
        positive=("pressure_loss_pa",),
        simplification="steady port flows, short transient",
    ),
    _case(
        "cyclone_separator", "Reverse-flow cyclone", cyclone_separator,
        (
            variable("cone_tip_radius_m", 0.03, 0.05, 0.038),
            variable("finder_radius_m", 0.035, 0.055, 0.045),
            variable("inlet_width_m", 0.03, 0.05, 0.04),
        ),
        "capture_fraction", (at_most("pressure_drop_pa", 0.2),), maximize=True,
        positive=("pressure_drop_pa",), simplification="one-way coupled particles of one size",
    ),
    _case(
        "human_upper_airways", "Idealized upper airway", human_upper_airways,
        (
            variable("constriction_ratio", 0.55, 0.9, 0.7),
            variable("constriction_position", 0.25, 0.7, 0.45),
            variable("branch_angle_deg", 25.0, 55.0, 38.0),
        ),
        "pressure_drop_pa", (at_least("left_flow_fraction", 0.15),),
        positive=("pressure_drop_pa",), simplification="idealized tubes, steady inhalation",
    ),
    _case(
        "pelton_distributor", "Pelton turbine distributor", pelton_distributor,
        (
            variable("branch_angle_deg", 30.0, 65.0, 45.0),
            variable("branch_radius_m", 0.04, 0.06, 0.05),
            variable("end_radius_m", 0.055, 0.09, 0.07),
        ),
        "nozzle_imbalance", (at_most("pressure_loss_pa", 0.2),),
        positive=("pressure_loss_pa",), simplification="three nozzles, single phase",
    ),
]
