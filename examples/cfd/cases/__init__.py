"""All design cases by name, in family order."""

from importlib import import_module

_CASE_MODULES = (
    "axial_fan",
    "axial_fan_optimization",
    "axial_fan_acoustic",
    "double_rotor_fan",
    "axial_pump",
    "parametric_axial_blood_pump",
    "axial_turbine",
    "kaplan_turbine",
    "ship_hull_propeller",
    "potsdam_propeller",
    "propeller_dji_9450",
    "propeller_optimization",
    "wind_turbine",
    "axial_compressor",
    "centrifugal_pump",
    "centrifugal_pump_optimization",
    "centrifugal_pump_partial_overlap",
    "fda_pump",
    "centrifugal_fan",
    "francis_turbine",
    "radial_turbine",
    "radial_compressor",
    "hecc_stage",
    "pipe",
    "parametric_pipe",
    "hydraulic_valve",
    "labyrinth_maze",
    "human_comfort",
    "rushton_impeller",
    "mechanical_siren",
    "water_valve",
    "intake_manifold_egr",
    "exhaust_manifold",
    "cyclone_separator",
    "human_upper_airways",
    "pelton_distributor",
    "flat_plate",
    "naca_0012",
    "naca_0012_acoustic",
    "bullet_223",
    "ahmed_body",
    "motorbike_benchmark",
    "train_aerodynamics",
    "spitfire",
    "building_wind_load",
    "dimpled_ball",
    "beam_ball",
    "formula_one",
    "drivaer",
)

CASES = {}
for module_name in _CASE_MODULES:
    case = import_module(f".{module_name}", __package__).CASE
    if case.name != module_name or case.name in CASES:
        raise RuntimeError(f"invalid or duplicate case name: {case.name}")
    CASES[case.name] = case
