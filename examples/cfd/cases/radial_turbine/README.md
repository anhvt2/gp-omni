# Hot-gas radial inflow turbine

[CFD case index](../../README.md#radial-and-mixed-flow-turbomachinery) · [Verification results](../../verification.md#radial-and-mixed-flow-turbomachinery) · [Case source](__init__.py) · [View settings](view.json)

A compressible inward-flow turbine receives heated gas through a spiral casing and vane row. The
vane angle and runner exit offset affect delivered power and efficiency.

**Solver** `rhoSimpleFoam` · **Mesh** structured `polyMesh` · **Level** `smoke`

## Mesh and results

| Initial mesh with cell edges | Solved field |
| --- | --- |
| ![Initial radial_turbine mesh section](mesh.png) | ![Static pressure (Pa) at iteration 200](results.png) |

Initial mesh and solved field on the `z = 0.024 m` plane. The results show point data for `p` at
iteration 200. The transverse static-pressure section samples the radial inlet and passages.

`checkMesh` reported 199,584 cells and maximum non-orthogonality of 72.64°. It reported zero
failed quality checks. The measured flow imbalance was 0.10%.

## Boundary conditions

| Patches | Velocity condition | `p` condition |
| --- | --- | --- |
| `blade_rotor`, `blade_vane`, `casing`, `hub`, `hub_rotor`, `shroud` | `noSlip` | `zeroGradient` |
| `inlet` | `flowRateInletVelocity` | `zeroGradient` |
| `outlet` | `pressureInletOutletVelocity` | `fixedValue` |

## Design and reference measurements

| Variable | Unit | Lower | Reference | Upper |
| --- | --- | ---: | ---: | ---: |
| `vane_angle_deg` | ° | 48 | 58 | 66 |
| `exit_offset_deg` | ° | -8 | 2 | 14 |

| Role | Metric | Reference | Campaign bound |
| --- | --- | ---: | ---: |
| **Maximize** | `efficiency` | 0.534629 | — |
| Constraint | `output_power_w` | 502.642 W | ≥ 452.378 W (10% margin) |

This smoke run checks the workflow on a coarse mesh. The saved iteration is a coarse-run
reference. Model limits include two-dimensional blades, rectangular spiral casing with a radial
slot over half a turn used as the inlet.

## Reproduce the views

Set up [OpenFOAM and ParaView](../../README.md#set-up), then run from the repository root.

```bash
python3 examples/cfd/gallery.py radial_turbine
```

The workflow writes to `runs/cfd_gallery/radial_turbine/`. Open `figures/mesh/scene.pvsm`
or `figures/reference/scene.pvsm` in ParaView. Each `figure.json` records the saved time,
field association, color range, camera, and source run. See the
[shared workflow](../../README.md#run-one-case) for individual mesh, solve, and optimization commands.
