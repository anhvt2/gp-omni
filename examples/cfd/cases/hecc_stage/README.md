# Compressor stage with a vaned diffuser

[CFD case index](../../README.md#radial-and-mixed-flow-turbomachinery) · [Verification results](../../verification.md#radial-and-mixed-flow-turbomachinery) · [Case source](__init__.py) · [View settings](view.json)

This representative centrifugal stage couples a backswept impeller to a vaned diffuser. Speed
and diffuser angle change the efficiency and pressure ratio.

**Solver** `rhoSimpleFoam` · **Mesh** structured `polyMesh` · **Level** `smoke`

## Mesh and results

| Initial mesh with cell edges | Solved field |
| --- | --- |
| ![Initial hecc_stage mesh section](mesh.png) | ![Static pressure (Pa) at iteration 200](results.png) |

Initial mesh and solved field on the `z = 0.029 m` plane. The results show point data for `p` at
iteration 200. The pressure cut shows part of one impeller and diffuser sector at the selected
height.

`checkMesh` reported 15,984 cells and maximum non-orthogonality of 72.05°. It reported zero
failed quality checks. The measured flow imbalance was 0.20%.

## Boundary conditions

| Patches | Velocity condition | `p` condition |
| --- | --- | --- |
| `blade_diffuser`, `blade_rotor`, `hub`, `hub_rotor`, `shroud` | `noSlip` | `zeroGradient` |
| `inlet` | `flowRateInletVelocity` | `zeroGradient` |
| `outlet` | `pressureInletOutletVelocity` | `fixedValue` |
| `periodic_a`, `periodic_b` | `cyclic` | `cyclic` |

## Design and reference measurements

| Variable | Unit | Lower | Reference | Upper |
| --- | --- | ---: | ---: | ---: |
| `speed_ratio` | — | 0.8 | 1 | 1.1 |
| `diffuser_angle_deg` | ° | 50 | 60 | 66 |

| Role | Metric | Reference | Campaign bound |
| --- | --- | ---: | ---: |
| **Maximize** | `efficiency` | 0.566428 | — |
| Constraint | `pressure_ratio` | 1.10207 | ≥ 1.09105 (1% margin) |

This smoke run checks the workflow on a coarse mesh. The saved iteration is a coarse-run
reference. Model limits include representative stage with equal impeller and diffuser blade
counts.

## Reproduce the views

Set up [OpenFOAM and ParaView](../../README.md#set-up), then run from the repository root.

```bash
python3 examples/cfd/gallery.py hecc_stage
```

The workflow writes to `runs/cfd_gallery/hecc_stage/`. Open `figures/mesh/scene.pvsm`
or `figures/reference/scene.pvsm` in ParaView. Each `figure.json` records the saved time,
field association, color range, camera, and source run. See the
[shared workflow](../../README.md#run-one-case) for individual mesh, solve, and optimization commands.
