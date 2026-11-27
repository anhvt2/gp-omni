"""Shared set-up and performance measurements of the bladed machines.

Every rotor turns about +z. In the rotating frame the approach flow comes at
``-atan(omega * r / meridional_speed)``, which fixes the sense of the blade angles.
"""

from dataclasses import dataclass
from math import atan, degrees, pi
from pathlib import Path

from foam import measure as post
from foam.boundaries import (
    blockage,
    flow_inlet,
    interface,
    moving_wall,
    partial_interface,
    periodic,
    pressure_outlet,
    slip,
    total_pressure_inlet,
    velocity_inlet,
    wall,
)
from foam.case import mesh_level, run_plan
from foam.passage import build_mesh, machine
from foam.physics import flow, rotation, write_flow_case

GRAVITY = 9.81
CP, GAS_EXPONENT = 1005.0, 0.4 / 1.4


def relative_angle(omega: float, radius: float, speed: float) -> float:
    """Degrees of the rotating-frame approach flow without inlet swirl."""
    return -degrees(atan(omega * radius / speed))


@dataclass
class duty:
    """Operating point. ``kind`` is pump, turbine, propeller or wind."""

    kind: str
    speed: float  # meridional velocity scale, or the flight / wind speed
    length: float
    flow: float | None = None  # full machine, m3/s or kg/s
    density: float = 1.2
    viscosity: float = 1.5e-5
    compressible: bool = False
    inlet_pressure: float = 101325.0  # compressible total pressure
    outlet_pressure: float = 101325.0
    inlet_temperature: float = 300.0
    transient: bool = False
    time_step: float = 0.0
    wall_shear: bool = False
    probes: tuple = ()
    partial_overlap: bool = False


def _conditions(names, design: machine, point: duty) -> dict:
    sliding = point.transient
    solid = moving_wall() if sliding else wall()
    result = {}
    for name in names:
        if name == "inlet":
            if point.kind in ("propeller", "wind"):
                result[name] = velocity_inlet((0, 0, point.speed))
            elif point.compressible and point.flow:
                result[name] = flow_inlet(
                    point.flow / design.sectors, temperature=point.inlet_temperature
                )
            elif point.compressible:
                result[name] = total_pressure_inlet(point.inlet_pressure, point.inlet_temperature)
            else:
                result[name] = flow_inlet(point.flow / design.sectors)
        elif name == "outlet":
            result[name] = pressure_outlet(
                point.outlet_pressure if point.compressible else 0.0
            )
        elif name == "farfield":
            result[name] = slip()
        elif name in ("periodic_a", "periodic_b"):
            result[name] = periodic("periodic_b" if name == "periodic_a" else "periodic_a")
        elif name in ("front_static", "front_rotor", "rear_static", "rear_rotor"):
            side, owner = name.split("_")
            other = "rotor" if owner == "static" else "static"
            result[name] = interface(f"{side}_{other}")
        elif name in ("rotor_couple", "volute_couple"):
            other = "volute_couple" if name == "rotor_couple" else "rotor_couple"
            if point.partial_overlap:
                result[name] = partial_interface(other, name.replace("couple", "block"))
            else:
                result[name] = interface(other)
        elif name == "casing" or (
            name in ("hub", "shroud") and not (sliding and design.spiral)
        ):
            result[name] = wall()
        else:
            result[name] = solid
    if point.partial_overlap:
        for side in ("rotor", "volute"):
            result[f"{side}_block"] = blockage(f"{side}_couple", moving=side == "rotor")
    return result


def build_machine(
    case: Path, design: machine, point: duty, level: mesh_level
) -> run_plan:
    built = build_mesh(design, level).build()
    built.require_positive_volumes()
    patches = _conditions(built.patch_names(), design, point)
    built.write(case, {name: condition.mesh_entry() for name, condition in patches.items()})

    rotors = [row for row in design.rows if row.omega]
    if not rotors:
        raise ValueError("a machine needs a rotating row")
    density = None if point.compressible else point.density
    functions = post.flux("inlet", "outlet")
    if point.compressible:
        functions |= post.compressible_total_pressure("inlet", "outlet")
        for patch in ("inlet", "outlet"):
            functions |= post.surface_value(
                f"T_{patch}", patch, "weightedAverage", "T", weight="phi"
            )
    else:
        functions |= post.total_pressure("inlet", "outlet")
    for row in rotors:
        names = (f"blade_{row.name}", f"hub_{row.name}", "hub_rotor", "shroud_rotor")
        walls = [n for n in dict.fromkeys(names) if n in patches]
        functions |= post.forces(f"load_{row.name}", walls, density)
        functions |= post.forces(f"blade_{row.name}", [f"blade_{row.name}"], density)
    shear_patches = [f"blade_{row.name}" for row in rotors]
    if point.wall_shear:
        functions |= {
            "wall_shear": {
                "type": "wallShearStress",
                "libs": "(fieldFunctionObjects)",
                "patches": tuple(shear_patches),
                "writeControl": "none",
                "log": False,
            },
            "wall_shear_magnitude": {
                "type": "mag",
                "libs": "(fieldFunctionObjects)",
                "field": "wallShearStress",
                "result": "shear",
                "writeControl": "none",
                "log": False,
            },
        }
        for patch in shear_patches:
            functions |= post.surface_value(f"shear_{patch}", patch, "areaAverage", "shear")
    if point.probes:
        functions |= post.probes("pressure_probes", point.probes)

    stationary = tuple(n for n in ("shroud", "hub", "farfield") if n in patches)
    motion = None
    if point.transient:
        motion = {
            "dynamicFvMesh": "dynamicMotionSolverFvMesh",
            "motionSolverLibs": ("fvMotionSolvers",),
            "motionSolver": "solidBody",
            "cellZone": "rotor",
            "solidBodyMotionFunction": "rotatingMotion",
            "rotatingMotionCoeffs": {
                "origin": (0, 0, 0),
                "axis": (0, 0, 1),
                "omega": rotors[0].omega,
            },
        }
    solver = "pimpleFoam" if point.transient else "simpleFoam"
    if point.compressible:
        solver = "rhoSimpleFoam"
    model = flow(
        solver,
        velocity=point.speed,
        length=point.length,
        density=point.density,
        viscosity=point.viscosity,
        iterations=level.iterations,
        time_step=point.time_step,
        time_steps=level.time_steps,
        pressure=point.inlet_pressure,
        temperature=point.inlet_temperature,
        second_order=level.resolution > 1,
        rotations=()
        if point.transient
        else tuple(rotation(f"zone_{r.name}", r.omega, stationary) for r in rotors),
        motion=motion,
        functions=functions,
    )
    write_flow_case(case, patches, model)

    def measure(directory: Path) -> dict[str, float]:
        return _performance(directory, design, point, rotors)

    return run_plan(solve=[[solver]], measure=measure)


def _performance(case: Path, design: machine, point: duty, rotors) -> dict[str, float]:
    sectors = design.sectors
    inflow = abs(post.last_flux(case, "inlet"))
    # a short transient run is not expected to balance, so it is reported but not checked
    balance = "flow_mismatch" if point.transient else "flow_imbalance"
    metrics = {balance: post.imbalance(case, "inlet", "outlet")}
    torque = sum(
        post.last_moment(case, f"load_{r.name}")[2] * r.omega for r in rotors
    )  # power the fluid gives to the rotors
    thrust = -sum(post.last_force(case, f"load_{r.name}")[2] for r in rotors) * sectors
    fluid_to_rotor = float(torque) * sectors
    scale = 1.0 if point.compressible else point.density
    total_in = post.last_surface(case, "total_p_inlet") * scale
    total_out = post.last_surface(case, "total_p_outlet") * scale
    flow_key = "mass_flow_kg_s" if point.compressible else "flow_m3_s"
    metrics[flow_key] = inflow * sectors
    if point.kind == "propeller":
        power = -fluid_to_rotor
        metrics |= {"thrust_n": float(thrust), "shaft_power_w": power}
        metrics["efficiency"] = _ratio(thrust * point.speed, power)
    elif point.kind == "wind":
        radius = max(r for _, r in design.shroud)
        available = 0.5 * point.density * point.speed**3 * pi * (point.length) ** 2
        moment = post.last_moment(case, f"blade_{rotors[0].name}")
        metrics |= {
            "output_power_w": fluid_to_rotor,
            "power_coefficient": fluid_to_rotor / available,
            "thrust_n": float(-thrust),
            "root_moment_nm": float((moment[0] ** 2 + moment[1] ** 2) ** 0.5),
            "farfield_radius_m": float(radius),
        }
    elif point.compressible:
        temperature = post.last_surface(case, "T_inlet")
        mass = inflow * sectors
        ratio = total_out / total_in
        isentropic = mass * CP * temperature * (ratio**GAS_EXPONENT - 1)
        metrics["pressure_ratio"] = ratio
        if point.kind == "pump":
            metrics["shaft_power_w"] = -fluid_to_rotor
            metrics["efficiency"] = _ratio(isentropic, -fluid_to_rotor)
        else:
            metrics["output_power_w"] = fluid_to_rotor
            metrics["efficiency"] = _ratio(fluid_to_rotor, -isentropic)
    elif point.kind == "pump":
        rise = total_out - total_in
        power = -fluid_to_rotor
        metrics |= {
            "pressure_rise_pa": rise,
            "head_m": rise / (point.density * GRAVITY),
            "shaft_power_w": power,
            "efficiency": _ratio(rise * inflow * sectors, power),
        }
    else:
        drop = total_in - total_out
        metrics |= {
            "pressure_drop_pa": drop,
            "output_power_w": fluid_to_rotor,
            "efficiency": _ratio(fluid_to_rotor, drop * inflow * sectors),
        }
    if point.wall_shear:
        values = [post.last_surface(case, f"shear_blade_{r.name}") for r in rotors]
        metrics["wall_shear_pa"] = point.density * max(values)
    if point.probes:
        scale = 1.0 if point.compressible else point.density
        metrics["pressure_fluctuation_pa"] = scale * post.probe_fluctuation(
            case, "pressure_probes"
        )
    return metrics


def _ratio(top: float, bottom: float) -> float:
    """A quotient that stays finite when a short run leaves the denominator near zero."""
    if abs(bottom) < 1e-9:
        return 0.0
    return float(top / bottom)
