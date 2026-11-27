# Verification results

49 of 49 cases passed the `smoke` checks. Each reference design ran with one process on an AMD Ryzen 7 5800X, Linux x86-64, OpenCFD OpenFOAM v2406, on 2026-11-26.

**A pass means the pipeline works, not that the numbers are right.** Meshes are coarse and runs stop after 200 iterations or 50 time steps. No mesh study or comparison with experiments was done. Values below record what the run produced.

## Axial turbomachinery

| Case | Mesher | Solver | Cells | Time s | Objective | First constraint | Simplification |
|---|---|---|---|---|---|---|---|
| axial_fan | structured | simpleFoam | 12528 | 6 | max efficiency 0.9257 | pressure_rise_pa 235.7 | shrouded blades without tip clearance |
| axial_fan_optimization | structured | simpleFoam | 12528 | 5 | max efficiency 0.9257 | pressure_rise_pa 235.7 | shrouded blades without tip clearance |
| axial_fan_acoustic | structured | pimpleFoam | 87696 | 55 | min pressure_fluctuation_pa 5.381 | pressure_rise_pa 194.8 | probe pressures only, no far-field acoustics |
| double_rotor_fan | structured | simpleFoam | 17712 | 8 | max efficiency 0.947 | pressure_rise_pa 228.6 | equal blade counts, frozen rotor, no tip clearance |
| axial_pump | structured | simpleFoam | 16848 | 7 | max efficiency 0.917 | pressure_rise_pa 2.696e+04 | equal blade counts, frozen rotor, no tip clearance |
| parametric_axial_blood_pump | structured | simpleFoam | 16848 | 8 | min wall_shear_pa 134.5 | pressure_rise_pa 5053 | equal blade counts, frozen rotor, no tip clearance, wall shear in place of a blood damage model |
| axial_turbine | structured | simpleFoam | 17496 | 8 | max efficiency 0.7591 | output_power_w 870.9 | equal blade counts, frozen rotor, no tip clearance, no spiral casing |
| kaplan_turbine | structured | simpleFoam | 17496 | 7 | max efficiency 0.7346 | output_power_w 4406 | equal blade counts, frozen rotor, no tip clearance, axial guide vanes, no spiral or draft tube |
| ship_hull_propeller | structured | simpleFoam | 22320 | 10 | max efficiency 0.7378 | thrust_n 59.7 | axial inflow, slip far field, low tip speed ratio set by the mesher's angle limit, no hull |
| potsdam_propeller | structured | simpleFoam | 22320 | 10 | max efficiency 0.7384 | thrust_n 230 | axial inflow, slip far field, low tip speed ratio set by the mesher's angle limit, own blade shape, no cavitation |
| propeller_dji_9450 | structured | simpleFoam | 22680 | 11 | min shaft_power_w 176.2 | thrust_n 4.881 | axial inflow, slip far field, low tip speed ratio set by the mesher's angle limit, own blade shape, no acoustics |
| propeller_optimization | structured | simpleFoam | 22680 | 11 | min shaft_power_w 162.2 | thrust_n 4.939 | axial inflow, slip far field, low tip speed ratio set by the mesher's angle limit, fixed axial inflow in place of static hover |
| wind_turbine | structured | simpleFoam | 23040 | 10 | max output_power_w 5694 | root_moment_nm 673.1 | axial inflow, slip far field, low tip speed ratio set by the mesher's angle limit, no tower |
| axial_compressor | structured | rhoSimpleFoam | 17064 | 8 | max efficiency 0.8811 | pressure_ratio 1.019 | equal blade counts, frozen rotor, no tip clearance |

## Radial and mixed-flow turbomachinery

| Case | Mesher | Solver | Cells | Time s | Objective | First constraint | Simplification |
|---|---|---|---|---|---|---|---|
| centrifugal_pump | bundled | simpleFoam | 229999 | 188 | min shaft_power_w 5.237e+04 | pressure_rise_pa 6.938e+05 |  |
| centrifugal_pump_optimization | structured | simpleFoam | 87696 | 41 | max efficiency 0.7423 | head_m 7.83 | two-dimensional blades, rectangular spiral casing with a radial slot over half a turn |
| centrifugal_pump_partial_overlap | structured | pimpleFoam | 87696 | 40 | min shaft_power_w 3842 | pressure_rise_pa 6.509e+04 | two-dimensional blades, rectangular spiral casing with a radial slot over half a turn, short transient |
| fda_pump | structured | simpleFoam | 62640 | 30 | min wall_shear_pa 31.71 | pressure_rise_pa 2.385e+04 | two-dimensional blades, rectangular spiral casing with a radial slot over half a turn, own geometry |
| centrifugal_fan | structured | simpleFoam | 125280 | 59 | max efficiency 0.7181 | pressure_rise_pa 310.2 | two-dimensional blades, rectangular spiral casing with a radial slot over half a turn |
| francis_turbine | structured | simpleFoam | 199584 | 103 | max efficiency 0.5371 | output_power_w 1.842e+04 | two-dimensional blades, rectangular spiral casing with a radial slot over half a turn used as the inlet, no draft tube |
| radial_turbine | structured | rhoSimpleFoam | 199584 | 115 | max efficiency 0.5346 | output_power_w 502.6 | two-dimensional blades, rectangular spiral casing with a radial slot over half a turn used as the inlet |
| radial_compressor | structured | rhoSimpleFoam | 137808 | 74 | max efficiency 0.5273 | pressure_ratio 1.117 | two-dimensional blades, rectangular spiral casing with a radial slot over half a turn |
| hecc_stage | structured | rhoSimpleFoam | 15984 | 8 | max efficiency 0.5664 | pressure_ratio 1.102 | representative stage with equal impeller and diffuser blade counts |

## Internal and process flows

| Case | Mesher | Solver | Cells | Time s | Objective | First constraint | Simplification |
|---|---|---|---|---|---|---|---|
| pipe | structured | simpleFoam | 15680 | 8 | min pressure_loss_pa 627.2 | volume_m3 0.01559 | flow only, no structural analysis |
| parametric_pipe | structured | simpleFoam | 15680 | 8 | min pressure_loss_pa 627.2 | volume_m3 0.01559 |  |
| hydraulic_valve | structured | simpleFoam | 5010 | 4 | min pressure_loss_pa 9888 | flow_force_n 0.3502 | axisymmetric, no structural analysis |
| labyrinth_maze | structured | interFoam | 3600 | 2 | min leakage_m3_s 0.002553 | length_m 0.05 | axisymmetric, short transient from an air-filled gap |
| human_comfort | structured | buoyantSimpleFoam | 32208 | 14 | min comfort_deviation_k 0.5856 | draught_speed_m_s 0.4622 | box manikin, no radiation or humidity |
| rushton_impeller | structured | simpleFoam | 65466 | 29 | min shaft_power_w 18.29 | mean_speed_m_s 0.1985 | single phase, flat bottom |
| mechanical_siren | structured | pimpleFoam | 43776 | 26 | max pressure_amplitude_pa 56.17 | pressure_drop_pa 74.32 | probe pressures only, short transient |
| water_valve | snappy | simpleFoam | 26275 | 18 | min pressure_loss_pa 1148 | flow_force_n 2.407 |  |
| intake_manifold_egr | snappy | simpleFoam | 17868 | 14 | min mixture_imbalance 0.3941 | pressure_loss_pa 312.3 | steady flow, passive scalar mixing |
| exhaust_manifold | snappy | rhoPimpleFoam | 14591 | 8 | min pressure_loss_pa 524.5 | volume_m3 0.001384 | steady port flows, short transient |
| cyclone_separator | snappy | simpleFoam | 27813 | 18 | max capture_fraction 0.13 | pressure_drop_pa 270.1 | one-way coupled particles of one size |
| human_upper_airways | snappy | simpleFoam | 21280 | 14 | min pressure_drop_pa 5.863 | left_flow_fraction 0.5127 | idealized tubes, steady inhalation |
| pelton_distributor | snappy | simpleFoam | 11868 | 8 | min nozzle_imbalance 0.01222 | pressure_loss_pa 7085 | three nozzles, single phase |

## External aerodynamics

| Case | Mesher | Solver | Cells | Time s | Objective | First constraint | Simplification |
|---|---|---|---|---|---|---|---|
| flat_plate | structured | simpleFoam | 3800 | 2 | min drag_error 0.02464 | cell_count 3800 | operating variables only, a pipeline check |
| naca_0012 | structured | simpleFoam | 7080 | 2 | max lift_to_drag 15.88 | lift_coefficient 0.47 |  |
| naca_0012_acoustic | structured | pimpleFoam | 7080 | 2 | min pressure_fluctuation_pa 31.06 | lift_coefficient 0.2827 | two-dimensional short transient, probes only |
| bullet_223 | structured | rhoCentralFoam | 11400 | 3 | min drag_coefficient 0.5003 | mass_g 5.003 | axisymmetric short transient, no structural analysis |
| ahmed_body | snappy | simpleFoam | 26386 | 15 | min drag_coefficient 0.7169 | volume_m3 0.1113 | no support stilts, rounding only in side view |
| motorbike_benchmark | snappy | simpleFoam | 54488 | 54 | min execution_time_s 28.51 | drag_coefficient 0.6576 | speed benchmark; processes are set with --cores |
| train_aerodynamics | snappy | simpleFoam | 61342 | 37 | min drag_coefficient 0.2886 | volume_m3 1.153 | single car, stationary ground |
| spitfire | snappy | simpleFoam | 64558 | 43 | max lift_to_drag 5.709 | lift_coefficient 0.4202 | simplified airframe, no propeller |
| building_wind_load | snappy | simpleFoam | 62300 | 33 | min base_moment_coefficient 0.427 | volume_m3 0.05692 | generic tower, steady wind |
| dimpled_ball | snappy | rhoPimpleFoam | 67095 | 22 | min drag_coefficient 0.4206 | lift_coefficient 0.01419 | short transient, spinning surface without moving dimples, no acoustics |
| beam_ball | snappy | simpleFoam | 35066 | 21 | min drag_n 1.804 | deflection_m 0.000222 | one-way estimate from beam theory, no fluid-structure coupling |
| formula_one | snappy | simpleFoam | 69861 | 51 | max downforce_coefficient 0.2808 | drag_coefficient 0.8736 | simplified body, static wheels and ground |
| drivaer | snappy | simpleFoam | 59290 | 40 | min drag_coefficient 0.6029 | volume_m3 7.273 | own geometry, not the published DrivAer model; static wheels and ground |

## Known limits of these runs

- `checkMesh` reported a failed quality check, within the pass conditions, for: formula_one, motorbike_benchmark, spitfire.
- Short transient runs (`pimpleFoam`, `rhoPimpleFoam`, `rhoCentralFoam`, `interFoam`) do not reach developed flow. `labyrinth_maze` only begins to fill with water and `bullet_223` covers 2 microseconds, so their objectives are not meaningful at this level.
- `cyclone_separator`: the captured fraction is very sensitive to the unconverged flow. It was 0.13 with one process and 0.08 with four.
- `centrifugal_pump` stops at 200 iterations, where its pressure rise (about 694 kPa) is far from the value after 800 iterations (about 1041 kPa). The stand-alone driver run to 800 iterations on this machine reproduced the recorded ARM64 value within 0.02%.
- `ahmed_body` drag is well above published measurements, as expected without wall layers on a mesh this coarse.
- `parametric_pipe` and `pipe`, and `axial_fan` and `axial_fan_optimization`, share their reference designs, so their reference results are identical.
- The `standard` mesh level was not run. Transient time steps are not rescaled for it.

Reproduce with `run.py verify --output /runs/verify`, see [README](README.md).
