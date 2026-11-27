# Parametric 3D FEM design optimization

Ten constrained design problems that run GP-Omni's batch Bayesian optimization against a
three-dimensional finite element solver. Each case is one script in this folder with its
own geometry, loads, bounds and limits. Meshing, solvers, the campaign runner and the
command line are shared in [common](common/). Use [mesh preparation](../preprocessing/README.md#3d-fem)
to solve a selected 3D design and export its fields for [ParaView](../postprocessing/README.md#heat-sink-fem).

## Run a case

Run from the repository root. [Gmsh](https://gmsh.info/) (GPL-2.0-or-later) and scikit-fem
are installed for the run and are not project dependencies. On a headless Linux machine
Gmsh also needs the system OpenGL libraries, such as `libglu1-mesa`.

```bash
alias fem="uv run --locked --with 'scikit-fem==12.0.2' --with 'gmsh==4.15.2' python"
fem examples/fem/drone_arm.py describe                  # variables, bounds, limits
fem examples/fem/drone_arm.py baseline                  # solve the reference design
fem examples/fem/drone_arm.py optimize drone-arm-run    # new output directory
```

A default campaign solves the reference design and 44 candidates (12 random, then 8
batches of 4). `--initial`, `--batches`, `--batch-size`, `--seed`, `--timeout-s` and
`--mesh-scale` (below 1 refines the mesh) change it. Each design runs in its own
single-threaded process. The output directory holds one JSON file per design, the
optimizer's database in `bo/`, and `summary.json`.

## Cases

All cases minimize one objective subject to limits.

| Case | Variables | Minimize | Subject to |
| --- | --- | --- | --- |
| [drone_arm](drone_arm.py) | 5 | mass | tip deflection, side impact stress, first frequency |
| [l_bracket](l_bracket.py) | 5 | mass | tip deflection, stress |
| [heat_sink](heat_sink.py) | 4 | temperature rise | mass |
| [stiffened_panel](stiffened_panel.py) | 4 | mass | first frequency, pressure deflection |
| [connecting_rod](connecting_rod.py) | 6 | mass | stress, elongation, lateral deflection |
| [lifting_lug](lifting_lug.py) | 4 | mass | stress, deflection |
| [pressure_nozzle](pressure_nozzle.py) | 4 | mass | junction stress |
| [flexure_hinge](flexure_hinge.py) | 4 | rotational stiffness | stress, axial and out-of-plane stiffness |
| [cold_plate](cold_plate.py) | 5 | temperature rise | pressure drop, mass |
| [lattice_beam](lattice_beam.py) | 4 | mass | tip deflection, stress, first frequency |

## Example results

One campaign per case with seed 0 and default settings, on Linux x86-64. "Best" is the
best feasible design among the reference and the 44 candidates.

| Case | Reference | Best | Change | Feasible candidates |
| --- | --- | --- | --- | --- |
| drone_arm | 0.05971 kg | 0.05665 kg | -5% | 13 of 44 |
| l_bracket | 0.1476 kg | 0.1016 kg | -31% | 19 of 44 |
| heat_sink | 55.91 K | 39.91 K | -29% | 14 of 44 |
| stiffened_panel | 0.3139 kg | 0.3139 kg | 0% | 17 of 44 |
| connecting_rod | 0.2962 kg | 0.2499 kg | -16% | 12 of 44 |
| lifting_lug | 0.7302 kg | 0.7302 kg | 0% | 10 of 44 |
| pressure_nozzle | 36.07 kg | 30.61 kg | -15% | 18 of 44 |
| flexure_hinge | 27.38 N m/rad | 2.817 N m/rad | -90% | 11 of 44 |
| cold_plate | 11.46 K | 10.57 K | -8% | 12 of 44 |
| lattice_beam | 0.01559 kg | 0.01559 kg | 0% | 16 of 44 |

**The optimizer underperforms on several cases.** In five of them plain space-filling
sampling with the same budget is estimated to do better, and three campaigns did not
improve on the reference. Part of this is traced to the library; see the
[bug report](../../bugs/constrained_bo_infeasible_incumbent.md). Do not read these
numbers as the achievable optimum.

## Assumptions and limits

- These are optimization demonstrations, not design qualifications. Validate any design
  for its actual duty.
- Small-deflection linear elasticity and steady conduction. No contact, fatigue, buckling
  or plasticity. Pin loads are cosine tractions.
- The heat sink uses an assumed film coefficient that depends on the pin gap. The cold
  plate uses the Dittus-Boelter and Blasius correlations with a uniform coolant
  temperature.
- `stress_mpa` is a volume-weighted percentile of the von Mises stress (99th or higher,
  sometimes away from supports), not the peak, which depends on the mesh at corners.
- Constraint limits were calibrated for the demonstration from 80 designs per case. They
  are not code allowables.
- Meshes differ slightly between platforms, so results can differ in the third digit.

## Verification

`tests/test_fem_solvers.py` compares the solvers with closed-form tension, Timoshenko
deflection, beam frequencies and conduction. `tests/test_fem_cases.py` checks every case
at its reference and bound corners, and `tests/test_fem_campaign.py` checks the campaign.
Refining every element size to 0.7 of its default changes each reference metric by less
than 2%. This was checked at the reference designs only. The tests need Gmsh and
scikit-fem and are skipped without them, as in the current CI job.

```bash
uv run --locked --group dev --with 'scikit-fem==12.0.2' --with 'gmsh==4.15.2' \
  pytest tests/test_fem_solvers.py tests/test_fem_cases.py tests/test_fem_campaign.py
```

## Reference geometry gallery

These images use the reference dimensions from the current 3D problem definitions.
They show undeformed surfaces. Each view has its own scale.

![FEM reference geometry overview](../assets/fem_reference_geometries.png)

| Case | Reference geometry | Problem definition |
| --- | --- | --- |
| Drone motor arm | [Image](../assets/drone_arm.png) | [Source](drone_arm.py) |
| Gusseted L bracket | [Image](../assets/l_bracket.png) | [Source](l_bracket.py) |
| Pin-fin heat sink | [Image](../assets/heat_sink.png) | [Source](heat_sink.py) |
| Stiffened panel | [Image](../assets/stiffened_panel.png) | [Source](stiffened_panel.py) |
| Connecting rod | [Image](../assets/connecting_rod.png) | [Source](connecting_rod.py) |
| Lifting lug | [Image](../assets/lifting_lug.png) | [Source](lifting_lug.py) |
| Pressure vessel nozzle | [Image](../assets/pressure_nozzle.png) | [Source](pressure_nozzle.py) |
| Flexure hinge | [Image](../assets/flexure_hinge.png) | [Source](flexure_hinge.py) |
| Liquid cold plate | [Image](../assets/cold_plate.png) | [Source](cold_plate.py) |
| Lattice beam | [Image](../assets/lattice_beam.png) | [Source](lattice_beam.py) |

The pressure nozzle image shows the symmetry-reduced domain used by its solver.
Regenerate the images after changes to the reference geometry using the
[rendering instructions](../assets/README.md).

## Tapered aluminum cantilever

Minimize the mass of a 600 mm cantilever with a 45 mm fixed width and a 1 kN tip load.
The optimizer varies the root height and the ratio of tip height to root height, subject
to a 1.5 mm tip deflection limit. The scikit-fem model uses cubic Hermite beam elements,
an ideal clamp, and small elastic deflections.

```bash
uv run --locked --with 'scikit-fem==12.0.2' python examples/fem/cantilever.py runs/cantilever
```

The script reports the best feasible dimensions, mass, and tip deflection. The run
directory holds the optimization observations and feasibility flags.
To view the deflection along the beam, follow the
[cantilever VTK workflow](../postprocessing/README.md#cantilever-fem).
