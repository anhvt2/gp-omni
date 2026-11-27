"""Complete case files for the five flow models used by the examples.

``flow`` describes the model; :func:`write_flow_case` writes ``0``, ``constant`` and
``system``. All turbulent cases use k-omega SST with wall functions.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .boundaries import boundary
from .dictionaries import value, write_dictionary, write_field

COMPRESSIBLE = ("rhoSimpleFoam", "rhoPimpleFoam", "rhoCentralFoam", "buoyantSimpleFoam")
STEADY = ("simpleFoam", "rhoSimpleFoam", "buoyantSimpleFoam")
COUPLED = ("symmetry", "empty", "wedge", "periodic", "ami", "acmi")


@dataclass(frozen=True)
class rotation:
    """A multiple-reference-frame zone turning about the z axis (rad/s)."""

    zone: str
    omega: float
    stationary: tuple[str, ...] = ()


@dataclass
class flow:
    solver: str
    velocity: float  # velocity scale for turbulence initial values
    length: float  # length scale for turbulence initial values
    density: float = 1.2  # incompressible reference density
    viscosity: float = 1.5e-5  # kinematic, incompressible
    iterations: int = 200
    time_step: float = 0.0
    time_steps: int = 0
    pressure: float = 101325.0  # initial absolute pressure of compressible models
    temperature: float = 293.15
    turbulent: bool = True
    second_order: bool = False
    rotations: tuple[rotation, ...] = ()
    motion: Mapping[str, object] | None = None  # constant/dynamicMeshDict
    libraries: tuple[str, ...] = ()
    initial_velocity: tuple[float, float, float] | None = None  # default: rest or freestream
    functions: Mapping[str, object] = field(default_factory=dict)
    scalars: tuple[str, ...] = ()
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    closed: bool = False  # no boundary fixes the pressure level

    @property
    def compressible(self) -> bool:
        return self.solver in COMPRESSIBLE

    @property
    def steady(self) -> bool:
        return self.solver in STEADY


def write_flow_case(case: Path, patches: Mapping[str, boundary], model: flow) -> None:
    case = Path(case)
    _fields(case / "0", patches, model)
    _constant(case / "constant", model)
    _system(case / "system", model)


def _turbulence_levels(model: flow) -> tuple[float, float]:
    k = 1.5 * (0.05 * model.velocity) ** 2
    omega = k**0.5 / (0.09**0.25 * 0.07 * model.length)
    return k, omega


def _vector(item) -> str:
    return f"uniform {value(item)}"


def _field(case: Path, name: str, kind: str, dimensions: str, initial, patches, rule) -> None:
    entries = {}
    for patch, condition in patches.items():
        if condition.kind in COUPLED:
            entry = {"type": condition.mesh_entry()["type"]}
            if condition.kind in ("periodic", "ami", "acmi"):
                entry["value"] = _vector(initial)
        else:
            kind_name = condition.kind
            if kind_name == "blockage":
                kind_name = "moving_wall" if condition.options["moving"] else "wall"
            entry = rule(kind_name, condition.options, initial)
        entries[patch] = entry
    write_field(case / name, kind, dimensions, initial, entries)


def _fields(directory: Path, patches, model: flow) -> None:
    solver = model.solver
    k0, omega0 = _turbulence_levels(model)
    inlets = ("velocity_inlet", "flow_inlet", "swirl_inlet", "supersonic_inlet",
              "total_pressure_inlet", "wind_profile_inlet")  # fmt: skip
    walls = ("wall", "moving_wall", "rotating_wall")

    def velocity(kind, options, initial):
        if kind == "wall":
            return {"type": "noSlip"}
        if kind == "moving_wall":
            return {"type": "movingWallVelocity", "value": _vector((0, 0, 0))}
        if kind == "rotating_wall":
            return {
                "type": "rotatingWallVelocity",
                "origin": options["origin"],
                "axis": options["axis"],
                "omega": options["omega"],
            }
        if kind == "slip":
            return {"type": "slip"}
        if kind in ("velocity_inlet", "supersonic_inlet"):
            return {"type": "fixedValue", "value": _vector(options["velocity"])}
        if kind == "flow_inlet":
            key = "massFlowRate" if model.compressible else "volumetricFlowRate"
            entry = {"type": "flowRateInletVelocity", key: f"constant {value(options['flow'])}"}
            if model.compressible:
                entry["rhoInlet"] = model.pressure / (287.0 * model.temperature)
            entry["value"] = _vector((0, 0, 0))
            return entry
        if kind == "wind_profile_inlet":
            return {
                "type": "atmBoundaryLayerInletVelocity",
                "flowDir": (1, 0, 0),
                "zDir": (0, 0, 1),
                "Uref": options["speed"],
                "Zref": options["height"],
                "z0": _vector(options["roughness"]),
                "zGround": _vector(0.0),
                "d": _vector(0.0),
                "value": _vector((options["speed"], 0, 0)),
            }
        if kind == "swirl_inlet":
            return {
                "type": "cylindricalInletVelocity",
                "origin": (0, 0, 0),
                "axis": (0, 0, 1),
                "axialVelocity": f"constant {value(options['axial'])}",
                "radialVelocity": f"constant {value(options['radial'])}",
                "rpm": f"constant {value(0.0)}",
                "tangentialVelocity": f"constant {value(options['tangential'])}",
                "value": _vector((0, 0, 0)),
            }
        if kind == "total_pressure_inlet":
            return {"type": "pressureInletOutletVelocity", "value": _vector((0, 0, 0))}
        if kind == "pressure_outlet":
            return {"type": "pressureInletOutletVelocity", "value": _vector(initial)}
        if kind == "freestream":
            return {
                "type": "freestreamVelocity",
                "freestreamValue": _vector(options["velocity"]),
                "value": _vector(options["velocity"]),
            }
        return {"type": "zeroGradient"}

    def pressure_value(options, initial):
        given = options.get("pressure", 0.0)
        return given if given or not model.compressible else initial

    def pressure(kind, options, initial):
        if kind == "pressure_outlet":
            return {"type": "fixedValue", "value": _vector(pressure_value(options, initial))}
        if kind == "supersonic_inlet":
            return {"type": "fixedValue", "value": _vector(options["pressure"])}
        if kind == "total_pressure_inlet":
            entry = {"type": "totalPressure", "p0": _vector(options["pressure"])}
            if model.compressible:
                entry |= {"psi": "thermo:psi", "gamma": 1.4}
            entry["value"] = _vector(initial)
            return entry
        if kind == "freestream":
            return {
                "type": "freestreamPressure",
                "freestreamValue": _vector(pressure_value(options, initial)),
                "value": _vector(initial),
            }
        return {"type": "zeroGradient"}

    def pressure_rgh(kind, options, initial):
        if kind == "pressure_outlet":
            level = pressure_value(options, initial)
            return {"type": "fixedValue", "value": _vector(level)}
        if kind == "total_pressure_inlet":
            return {
                "type": "totalPressure",
                "p0": _vector(options["pressure"]),
                "value": _vector(initial),
            }
        return {"type": "fixedFluxPressure", "value": _vector(initial)}

    def temperature(kind, options, initial):
        given = options.get("temperature")
        if kind == "total_pressure_inlet":
            return {
                "type": "totalTemperature",
                "T0": _vector(given or initial),
                "psi": "thermo:psi",
                "gamma": 1.4,
                "value": _vector(initial),
            }
        if kind in inlets or (kind in walls and given):
            return {"type": "fixedValue", "value": _vector(given or initial)}
        if kind in ("pressure_outlet", "freestream"):
            far = given or initial
            return {"type": "inletOutlet", "inletValue": _vector(far), "value": _vector(far)}
        return {"type": "zeroGradient"}

    def turbulence(wall_function):
        def rule(kind, options, initial):
            if kind in walls:
                return {"type": wall_function, "value": _vector(initial)}
            if kind in inlets:
                return {"type": "fixedValue", "value": _vector(initial)}
            if kind in ("pressure_outlet", "freestream"):
                return {
                    "type": "inletOutlet",
                    "inletValue": _vector(initial),
                    "value": _vector(initial),
                }
            return {"type": "zeroGradient"}

        return rule

    def eddy(wall_function):
        def rule(kind, options, initial):
            kind_name = wall_function if kind in walls else "calculated"
            return {"type": kind_name, "value": _vector(initial)}

        return rule

    def scalar(name):
        def rule(kind, options, initial):
            if kind in inlets:
                return {
                    "type": "fixedValue",
                    "value": _vector(options.get("scalars", {}).get(name, 0.0)),
                }
            if kind in ("pressure_outlet", "freestream"):
                return {"type": "inletOutlet", "inletValue": _vector(0.0), "value": _vector(0.0)}
            return {"type": "zeroGradient"}

        return rule

    start = (0.0, 0.0, 0.0)
    for condition in patches.values():
        if condition.kind in ("freestream", "supersonic_inlet"):
            start = tuple(condition.options["velocity"])
    if model.initial_velocity is not None:
        start = tuple(model.initial_velocity)
    _field(directory, "U", "volVectorField", "[0 1 -1 0 0 0 0]", start, patches, velocity)
    if solver == "interFoam":
        _field(directory, "p_rgh", "volScalarField", "[1 -1 -2 0 0 0 0]", 0.0, patches,
               pressure_rgh)  # fmt: skip
        _field(directory, "alpha.water", "volScalarField", "[0 0 0 0 0 0 0]", 0.0, patches,
               scalar("water"))  # fmt: skip
    elif model.compressible:
        dimensions = "[1 -1 -2 0 0 0 0]"
        if solver == "buoyantSimpleFoam":
            _field(directory, "p_rgh", "volScalarField", dimensions, model.pressure, patches,
                   pressure_rgh)  # fmt: skip
            _field(directory, "p", "volScalarField", dimensions, model.pressure, patches,
                   lambda kind, options, initial: {"type": "calculated",
                                                   "value": _vector(initial)})  # fmt: skip
        else:
            _field(directory, "p", "volScalarField", dimensions, model.pressure, patches, pressure)
        _field(directory, "T", "volScalarField", "[0 0 0 1 0 0 0]", model.temperature, patches,
               temperature)  # fmt: skip
    else:
        _field(directory, "p", "volScalarField", "[0 2 -2 0 0 0 0]", 0.0, patches, pressure)
    if model.turbulent:
        _field(directory, "k", "volScalarField", "[0 2 -2 0 0 0 0]", k0, patches,
               turbulence("kqRWallFunction"))  # fmt: skip
        _field(directory, "omega", "volScalarField", "[0 0 -1 0 0 0 0]", omega0, patches,
               turbulence("omegaWallFunction"))  # fmt: skip
        _field(directory, "nut", "volScalarField", "[0 2 -1 0 0 0 0]", k0 / omega0, patches,
               eddy("nutkWallFunction"))  # fmt: skip
        if model.compressible:
            _field(directory, "alphat", "volScalarField", "[1 -1 -1 0 0 0 0]", 0.0, patches,
                   eddy("compressible::alphatWallFunction"))  # fmt: skip
    for name in model.scalars:
        _field(directory, name, "volScalarField", "[0 0 0 0 0 0 0]", 0.0, patches, scalar(name))


def _constant(directory: Path, model: flow) -> None:
    turbulence = {"simulationType": "laminar"}
    if model.turbulent:
        turbulence = {
            "simulationType": "RAS",
            "RAS": {"RASModel": "kOmegaSST", "turbulence": "on", "printCoeffs": "off"},
        }
    write_dictionary(directory / "turbulenceProperties", turbulence)
    if model.solver == "interFoam":
        write_dictionary(
            directory / "transportProperties",
            {
                "phases": ("water", "air"),
                "water": {"transportModel": "Newtonian", "nu": 1e-6, "rho": 1000},
                "air": {"transportModel": "Newtonian", "nu": 1.5e-5, "rho": 1.2},
                "sigma": 0.07,
            },
        )
    elif model.compressible:
        buoyant = model.solver == "buoyantSimpleFoam"
        central = model.solver == "rhoCentralFoam"
        write_dictionary(
            directory / "thermophysicalProperties",
            {
                "thermoType": {
                    "type": "heRhoThermo" if buoyant else "hePsiThermo",
                    "mixture": "pureMixture",
                    "transport": "const",
                    "thermo": "hConst",
                    "equationOfState": "perfectGas",
                    "specie": "specie",
                    "energy": "sensibleInternalEnergy" if central else "sensibleEnthalpy",
                },
                "mixture": {
                    "specie": {"molWeight": 28.96},
                    "thermodynamics": {"Cp": 1005, "Hf": 0},
                    "transport": {"mu": 1.8e-5, "Pr": 0.71},
                },
            },
        )
    else:
        write_dictionary(
            directory / "transportProperties",
            {"transportModel": "Newtonian", "nu": model.viscosity},
        )
    if model.solver in ("interFoam", "buoyantSimpleFoam"):
        write_dictionary(
            directory / "g",
            {"dimensions": "[0 1 -2 0 0 0 0]", "value": model.gravity},
            "uniformDimensionedVectorField",
        )
    if model.solver == "buoyantSimpleFoam":
        write_dictionary(directory / "radiationProperties", {"radiationModel": "none"})
    if model.rotations:
        write_dictionary(
            directory / "MRFProperties",
            {
                f"MRF_{r.zone}": {
                    "cellZone": r.zone,
                    "active": "yes",
                    "nonRotatingPatches": r.stationary,
                    "origin": (0, 0, 0),
                    "axis": (0, 0, 1),
                    "omega": r.omega,
                }
                for r in model.rotations
            },
        )
    if model.motion:
        write_dictionary(directory / "dynamicMeshDict", model.motion)


def _system(directory: Path, model: flow) -> None:
    steady = model.steady
    end = model.iterations if steady else model.time_steps * model.time_step
    control = {
        "application": model.solver,
        "startFrom": "startTime",
        "startTime": 0,
        "stopAt": "endTime",
        "endTime": end,
        "deltaT": 1 if steady else model.time_step,
        "writeControl": "timeStep",
        "writeInterval": model.iterations if steady else model.time_steps,
        "purgeWrite": 1,
        "writeFormat": "binary",
        "writePrecision": 10,
        "timePrecision": 10,
        "runTimeModifiable": False,
        "functions": dict(model.functions),
    }
    if model.solver == "interFoam":
        control |= {"adjustTimeStep": False, "maxCo": 1, "maxAlphaCo": 1, "maxDeltaT": 1}
    if model.libraries:
        control["libs"] = model.libraries
    write_dictionary(directory / "controlDict", control)
    write_dictionary(directory / "fvSchemes", _schemes(model))
    write_dictionary(directory / "fvSolution", _solution(model))
    if model.solver in ("rhoSimpleFoam", "rhoPimpleFoam", "buoyantSimpleFoam"):
        write_dictionary(
            directory / "fvOptions",
            {
                "limit_temperature": {
                    "type": "limitTemperature",
                    "active": "yes",
                    "selectionMode": "all",
                    "min": 150,
                    "max": 1200,
                }
            },
        )


def _schemes(model: flow) -> dict:
    solver = model.solver
    bounded = "bounded " if model.steady else ""
    momentum = "linearUpwind grad(U)" if model.second_order else "upwind"
    transported = ["k", "omega", "h", "e", "K", "Ekp", *model.scalars]
    div = {"default": "none", "div(phi,U)": f"{bounded}Gauss {momentum}"}
    for name in transported:
        div[f"div(phi,{name})"] = f"{bounded}Gauss upwind"
    div['"div\\(.*dev2.*\\)"'] = "Gauss linear"
    if model.compressible:
        div["div(phid,p)"] = "Gauss upwind"
        div["div(phiMRF,p)"] = "Gauss upwind"
        div["div(phi,(p|rho))"] = f"{bounded}Gauss upwind"
    schemes = {
        "ddtSchemes": {"default": "steadyState" if model.steady else "Euler"},
        "gradSchemes": {"default": "cellLimited Gauss linear 1"},
        "divSchemes": div,
        "laplacianSchemes": {"default": "Gauss linear limited corrected 0.5"},
        "interpolationSchemes": {"default": "linear"},
        "snGradSchemes": {"default": "limited corrected 0.5"},
        "wallDist": {"method": "meshWave"},
    }
    if solver == "interFoam":
        schemes["divSchemes"] = {
            "default": "none",
            "div(rhoPhi,U)": "Gauss upwind",
            "div(phi,alpha)": "Gauss vanLeer",
            "div(phirb,alpha)": "Gauss linear",
            '"div\\(.*dev2.*\\)"': "Gauss linear",
        }
    if solver == "rhoCentralFoam":
        schemes["fluxScheme"] = "Kurganov"
        schemes["divSchemes"] = {"default": "none", "div(tauMC)": "Gauss linear"}
        schemes["interpolationSchemes"] = {
            "default": "linear",
            "reconstruct(rho)": "Minmod",
            "reconstruct(U)": "MinmodV",
            "reconstruct(T)": "Minmod",
        }
    return schemes


def _solution(model: flow) -> dict:
    solver = model.solver
    smooth = {
        "solver": "smoothSolver",
        "smoother": "symGaussSeidel",
        "tolerance": 1e-8,
        "relTol": 0.1,
    }
    final = smooth | {"relTol": 0}
    gamg = {"solver": "GAMG", "smoother": "GaussSeidel", "tolerance": 1e-7, "relTol": 0.05}
    pressure_name = "p_rgh" if solver in ("interFoam", "buoyantSimpleFoam") else "p"
    reference = {}
    if model.closed:
        prefix = "pRef" if pressure_name == "p" else "p_rghRef"
        reference = {f"{prefix}Cell": 0, f"{prefix}Value": 0}
        if model.compressible:
            reference[f"{prefix}Value"] = model.pressure
    scalars = "|".join(["U", "k", "omega", "h", "e", *model.scalars])
    solvers = {
        pressure_name: gamg,
        f"{pressure_name}Final": gamg | {"relTol": 0},
        f'"({scalars})"': smooth,
        f'"({scalars})Final"': final,
    }
    if solver == "simpleFoam":
        return {
            "solvers": solvers,
            "SIMPLE": {"nNonOrthogonalCorrectors": 1, **reference},
            "relaxationFactors": {
                "fields": {"p": 0.3},
                "equations": {"U": 0.5, '"(k|omega)"': 0.4, '".*"': 0.7},
            },
        }
    if solver == "buoyantSimpleFoam":
        # buoyancy-driven rooms need weak momentum and energy relaxation
        return {
            "solvers": solvers,
            "SIMPLE": {"nNonOrthogonalCorrectors": 1, "momentumPredictor": "no", **reference},
            "relaxationFactors": {
                "fields": {"p_rgh": 0.7},
                "equations": {"U": 0.3, "h": 0.3, '"(k|omega)"': 0.5},
            },
        }
    if solver == "rhoSimpleFoam":
        return {
            "solvers": solvers,
            "SIMPLE": {
                "nNonOrthogonalCorrectors": 1,
                "pMinFactor": 0.4,
                "pMaxFactor": 2.5,
                **reference,
            },
            "relaxationFactors": {
                "fields": {"p": 0.2, "rho": 0.05},
                "equations": {"U": 0.3, '"(k|omega)"': 0.3, '"(h|e)"': 0.2},
            },
        }
    if solver == "interFoam":
        solvers['"alpha.water.*"'] = {
            "nAlphaCorr": 2,
            "nAlphaSubCycles": 1,
            "cAlpha": 1,
            **final,
        }
        solvers['"pcorr.*"'] = gamg | {"relTol": 0}
        return {
            "solvers": solvers,
            "PIMPLE": {
                "momentumPredictor": "no",
                "nOuterCorrectors": 1,
                "nCorrectors": 3,
                "nNonOrthogonalCorrectors": 1,
                **reference,
            },
        }
    if solver == "rhoCentralFoam":
        diagonal = {"solver": "diagonal"}
        return {
            "solvers": {
                '"(rho|rhoU|rhoE)"': diagonal,
                '"(U|e|k|omega)"': final,
                '"(U|e|k|omega)Final"': final,
            }
        }
    solvers['"(rho|rhoFinal)"'] = {"solver": "diagonal"}
    solvers['"pcorr.*"'] = gamg | {"relTol": 0}
    solvers['"cellDisplacement.*"'] = gamg
    return {
        "solvers": solvers,
        "PIMPLE": {
            "nOuterCorrectors": 2,
            "nCorrectors": 2,
            "nNonOrthogonalCorrectors": 1,
            "correctPhi": "no",
            **reference,
        },
        "relaxationFactors": {"equations": {'".*"': 1}},
    }
