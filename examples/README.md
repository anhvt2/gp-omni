# Engineering design examples

GP-Omni couples Bayesian optimization with finite element and flow solvers. Choose an
example to see its reference design and how to reproduce the mesh, solve, or optimization.

## Finite element analysis

[FEM setup and run commands](fem/README.md)

The FEM previews show undeformed reference geometry, except for the heat sink's solved
temperature rise, which uses an assumed convection coefficient, and the cantilever's
deflection profile for a 60 mm root and 45 mm tip design.

| [Drone motor arm](fem/drone_arm.py) | [Gusseted L bracket](fem/l_bracket.py) | [Pin-fin heat sink](postprocessing/README.md#heat-sink-fem) |
| --- | --- | --- |
| ![Hollow tapered motor arm with lightening holes](assets/drone_arm.png) | ![L bracket with its reinforcing gusset](assets/l_bracket.png) | ![Temperature rise in kelvins across the base and fins](assets/heat_sink_temperature.png) |

| [Stiffened panel](fem/stiffened_panel.py) | [Connecting rod](fem/connecting_rod.py) | [Lifting lug](fem/lifting_lug.py) |
| --- | --- | --- |
| ![Plate with three longitudinal ribs and one cross rib](assets/stiffened_panel.png) | ![Connecting rod with pin bores at both ends](assets/connecting_rod.png) | ![Lifting lug with a circular pin hole](assets/lifting_lug.png) |

| [Pressure vessel nozzle](fem/pressure_nozzle.py) | [Flexure hinge](fem/flexure_hinge.py) | [Liquid cold plate](fem/cold_plate.py) |
| --- | --- | --- |
| ![Symmetry-reduced vessel shell and nozzle](assets/pressure_nozzle.png) | ![Flexure hinge with a narrowed section](assets/flexure_hinge.png) | ![Cold plate reference geometry](assets/cold_plate.png) |

| [Lattice beam](fem/lattice_beam.py) | [Tapered aluminum cantilever](fem/README.md#tapered-aluminum-cantilever) |
| --- | --- |
| ![Beam with diagonal lattice struts](assets/lattice_beam.png) | ![Downward deflection along the 600 mm span under a 1 kN tip load](assets/cantilever_deflection.png) |

## Computational fluid dynamics

[OpenFOAM setup and run commands](cfd/README.md) · [Verification and model limits](cfd/verification.md)

The CFD previews show coarse reference runs. Full-annulus previews reconstruct the
solved periodic passage.

### Axial turbomachinery

| [Ducted axial fan](cfd/cases/axial_fan/README.md) | [Axial fan blade count](cfd/cases/axial_fan_optimization/README.md) | [Axial fan acoustics](cfd/cases/axial_fan_acoustic/README.md) |
| --- | --- | --- |
| ![Reconstructed full-annulus field for ducted axial fan](cfd/cases/axial_fan/annulus.png) | ![Flow speed (m/s) at iteration 200 for Axial fan blade count](cfd/cases/axial_fan_optimization/results.png) | ![Pressure time series at the probes for Axial fan acoustics](cfd/cases/axial_fan_acoustic/pressure-probes.png) |

| [Counter-rotating fan](cfd/cases/double_rotor_fan/README.md) | [Axial water pump with stator](cfd/cases/axial_pump/README.md) | [Miniature axial blood pump](cfd/cases/parametric_axial_blood_pump/README.md) |
| --- | --- | --- |
| ![Reconstructed full-annulus field for counter-rotating fan](cfd/cases/double_rotor_fan/annulus.png) | ![Reconstructed full-annulus field for axial water pump with stator](cfd/cases/axial_pump/annulus.png) | ![Reconstructed full-annulus field for miniature axial blood pump](cfd/cases/parametric_axial_blood_pump/annulus.png) |

| [Axial turbine stage](cfd/cases/axial_turbine/README.md) | [Kaplan turbine](cfd/cases/kaplan_turbine/README.md) | [Marine propeller in open water](cfd/cases/ship_hull_propeller/README.md) |
| --- | --- | --- |
| ![Reconstructed full-annulus field for axial turbine stage](cfd/cases/axial_turbine/annulus.png) | ![Reconstructed full-annulus field for the Kaplan turbine](cfd/cases/kaplan_turbine/annulus.png) | ![Reconstructed full-annulus field for marine propeller in open water](cfd/cases/ship_hull_propeller/annulus.png) |

| [Potsdam propeller advance sweep](cfd/cases/potsdam_propeller/README.md) | [Two-blade drone propeller](cfd/cases/propeller_dji_9450/README.md) | [Three-blade propeller sizing](cfd/cases/propeller_optimization/README.md) |
| --- | --- | --- |
| ![Reconstructed full-annulus field for the Potsdam propeller](cfd/cases/potsdam_propeller/annulus.png) | ![Reconstructed full-annulus field for two-blade drone propeller](cfd/cases/propeller_dji_9450/annulus.png) | ![Reconstructed full-annulus field for three-blade propeller sizing](cfd/cases/propeller_optimization/annulus.png) |

| [Three-blade wind turbine](cfd/cases/wind_turbine/README.md) | [Subsonic axial compressor stage](cfd/cases/axial_compressor/README.md) |
| --- | --- |
| ![Reconstructed full-annulus field for three-blade wind turbine](cfd/cases/wind_turbine/annulus.png) | ![Reconstructed full-annulus field for subsonic axial compressor stage](cfd/cases/axial_compressor/annulus.png) |

### Radial and mixed-flow turbomachinery

| [Volute water pump](cfd/cases/centrifugal_pump/README.md) | [Pump impeller and casing](cfd/cases/centrifugal_pump_optimization/README.md) | [Sliding pump with offset casing](cfd/cases/centrifugal_pump_partial_overlap/README.md) |
| --- | --- | --- |
| ![Kinematic pressure (m²/s²) at iteration 200 for Volute water pump](cfd/cases/centrifugal_pump/results.png) | ![Kinematic pressure (m²/s²) at iteration 200 for Pump impeller and casing](cfd/cases/centrifugal_pump_optimization/results.png) | ![Kinematic pressure (m²/s²) at t = 0.015 s for Sliding pump with offset casing](cfd/cases/centrifugal_pump_partial_overlap/results.png) |

| [Small centrifugal blood pump](cfd/cases/fda_pump/README.md) | [Backswept centrifugal fan](cfd/cases/centrifugal_fan/README.md) | [Francis turbine with casing and guide vanes](cfd/cases/francis_turbine/README.md) |
| --- | --- | --- |
| ![Flow speed (m/s) at iteration 200 for Small centrifugal blood pump](cfd/cases/fda_pump/results.png) | ![Flow speed (m/s) at iteration 200 for Backswept centrifugal fan](cfd/cases/centrifugal_fan/results.png) | ![Kinematic pressure (m²/s²) at iteration 200 for Francis turbine with casing and guide vanes](cfd/cases/francis_turbine/results.png) |

| [Hot-gas radial inflow turbine](cfd/cases/radial_turbine/README.md) | [Centrifugal compressor with casing](cfd/cases/radial_compressor/README.md) | [Centrifugal compressor with vaned diffuser](cfd/cases/hecc_stage/README.md) |
| --- | --- | --- |
| ![Static pressure (Pa) at iteration 200 for Hot-gas radial inflow turbine](cfd/cases/radial_turbine/results.png) | ![Static pressure (Pa) at iteration 200 for Centrifugal compressor with casing](cfd/cases/radial_compressor/results.png) | ![Reconstructed full-annulus field for centrifugal compressor with vaned diffuser](cfd/cases/hecc_stage/annulus.png) |

The bundled volute pump also has a [standalone sizing workflow](cfd/cases/centrifugal_pump/README.md#standalone-pump-sizing).

### Internal and process flows

| [180-degree pipe bend](cfd/cases/pipe/README.md) | [Nine-diameter pipe bend](cfd/cases/parametric_pipe/README.md) | [Axisymmetric poppet valve](cfd/cases/hydraulic_valve/README.md) |
| --- | --- | --- |
| ![Flow speed (m/s) at iteration 200 for 180-degree pipe bend](cfd/cases/pipe/results.png) | ![Flow speed (m/s) at iteration 200 for Nine-diameter pipe bend](cfd/cases/parametric_pipe/results.png) | ![Flow speed (m/s) at iteration 200 for Axisymmetric poppet valve](cfd/cases/hydraulic_valve/results.png) |

| [Labyrinth seal filling with water](cfd/cases/labyrinth_maze/README.md) | [Ventilated room](cfd/cases/human_comfort/README.md) | [Rushton turbine in a stirred vessel](cfd/cases/rushton_impeller/README.md) |
| --- | --- | --- |
| ![Water volume fraction at t = 0.0001 s for Labyrinth seal filling with water](cfd/cases/labyrinth_maze/results.png) | ![Temperature (K) at iteration 200 for Ventilated room](cfd/cases/human_comfort/results.png) | ![Flow speed (m/s) at iteration 200 for Rushton turbine in a stirred vessel](cfd/cases/rushton_impeller/results.png) |

| [Sliding-mesh siren](cfd/cases/mechanical_siren/README.md) | [Butterfly valve in a water line](cfd/cases/water_valve/README.md) | [Intake manifold with exhaust gas recirculation](cfd/cases/intake_manifold_egr/README.md) |
| --- | --- | --- |
| ![Pressure time series at the probes for Sliding-mesh siren](cfd/cases/mechanical_siren/pressure-probes.png) | ![Flow speed (m/s) at iteration 200 for Butterfly valve in a water line](cfd/cases/water_valve/results.png) | ![EGR passive scalar at iteration 200 for Intake manifold with exhaust gas recirculation](cfd/cases/intake_manifold_egr/results.png) |

| [Four-into-one exhaust manifold](cfd/cases/exhaust_manifold/README.md) | [Reverse-flow cyclone](cfd/cases/cyclone_separator/README.md) |
| --- | --- |
| ![Static pressure (Pa) at t = 0.005 s for Four-into-one exhaust manifold](cfd/cases/exhaust_manifold/results.png) | ![Flow speed (m/s) at iteration 201 for Reverse-flow cyclone](cfd/cases/cyclone_separator/results.png) |

| [Idealized upper airway](cfd/cases/human_upper_airways/README.md) | [Pelton turbine distributor](cfd/cases/pelton_distributor/README.md) |
| --- | --- |
| ![Flow speed (m/s) at iteration 200 for Idealized upper airway](cfd/cases/human_upper_airways/results.png) | ![Flow speed (m/s) at iteration 200 for Pelton turbine distributor](cfd/cases/pelton_distributor/results.png) |

### External aerodynamics

| [Turbulent flat plate](cfd/cases/flat_plate/README.md) | [NACA 0012 airfoil in two dimensions](cfd/cases/naca_0012/README.md) | [NACA 0012 trailing-edge acoustics](cfd/cases/naca_0012_acoustic/README.md) |
| --- | --- | --- |
| ![Flow speed (m/s) at iteration 200 for Turbulent flat plate](cfd/cases/flat_plate/results.png) | ![Flow speed (m/s) at iteration 200 for NACA 0012 airfoil in two dimensions](cfd/cases/naca_0012/results.png) | ![Pressure time series at the probes for NACA 0012 trailing-edge acoustics](cfd/cases/naca_0012_acoustic/pressure-probes.png) |

| [Supersonic bullet](cfd/cases/bullet_223/README.md) | [Ahmed reference car body, half model](cfd/cases/ahmed_body/README.md) | [Solver run time on the motorbike geometry](cfd/cases/motorbike_benchmark/README.md) |
| --- | --- | --- |
| ![Density (kg/m³) at t = 2e-06 s for Supersonic bullet](cfd/cases/bullet_223/results.png) | ![Flow speed (m/s) at iteration 200 for Ahmed reference car body, half model](cfd/cases/ahmed_body/results.png) | ![Flow speed (m/s) at iteration 200 for Solver run time on the motorbike geometry](cfd/cases/motorbike_benchmark/results.png) |

| [Train car nose shape and yaw](cfd/cases/train_aerodynamics/README.md) | [Fighter airframe lift and drag](cfd/cases/spitfire/README.md) | [Tall building in an atmospheric wind profile](cfd/cases/building_wind_load/README.md) |
| --- | --- | --- |
| ![Flow speed (m/s) at iteration 200 for Train car nose shape and yaw](cfd/cases/train_aerodynamics/results.png) | ![Flow speed (m/s) at iteration 200 for Fighter airframe lift and drag](cfd/cases/spitfire/results.png) | ![Flow speed (m/s) at iteration 200 for Tall building in an atmospheric wind profile](cfd/cases/building_wind_load/results.png) |

| [Spinning dimpled ball](cfd/cases/dimpled_ball/README.md) | [Ball on a cantilever rod](cfd/cases/beam_ball/README.md) |
| --- | --- |
| ![Flow speed (m/s) at t = 0.001 s for Spinning dimpled ball](cfd/cases/dimpled_ball/results.png) | ![Flow speed (m/s) at iteration 200 for Ball on a cantilever rod](cfd/cases/beam_ball/results.png) |

| [Simplified open-wheel racing car](cfd/cases/formula_one/README.md) | [Simplified notchback sedan](cfd/cases/drivaer/README.md) |
| --- | --- |
| ![Flow speed (m/s) at iteration 200 for Simplified open-wheel racing car](cfd/cases/formula_one/results.png) | ![Flow speed (m/s) at iteration 200 for Simplified notchback sedan](cfd/cases/drivaer/results.png) |

## Meshes and figures

[Prepare meshes and export solver fields](preprocessing/README.md) ·
[Render ParaView scenes, sample VTK profiles, and export figures](postprocessing/README.md)

Run commands from the repository root and use a fresh directory under `runs/` for each
mesh or simulation. [Numerical examples](../tests/numerical/) and
[analytical benchmarks](../tests/bo_functions/) live under `tests/`.
