# Ahmed reference car body, half model

[CFD case index](../../README.md#external-aerodynamics) · [Verification results](../../verification.md#external-aerodynamics) · [Case source](__init__.py) · [View settings](view.json)

A half-width Ahmed body sits above a stationary ground plane in a 40 m/s wind tunnel. The rear
slant, diffuser and nose rounding control the shape at a constrained body volume.

**Solver** `simpleFoam` · **Mesh** `snappyHexMesh` · **Level** `smoke`

## Mesh and results

| Initial mesh with cell edges | Solved field |
| --- | --- |
| ![Initial ahmed_body mesh section](mesh.png) | ![Speed \|U\| (m/s) at iteration 200](results.png) |

Initial mesh and solved field on the `y = 0.015 m` plane. The results show point data for `U` at
iteration 200. The near-symmetry-plane view shows the cut cells around the slant and the near
wake. The coarse mesh has no wall layers. The verification report records drag above published
measurements.

`checkMesh` reported 26,386 cells and maximum non-orthogonality of 46.83°. It reported zero
failed quality checks. The measured flow imbalance was 0.00%.

## Boundary conditions

| Patches | Velocity condition | `p` condition |
| --- | --- | --- |
| `inlet` | `fixedValue` | `zeroGradient` |
| `outlet` | `pressureInletOutletVelocity` | `fixedValue` |
| `left` | `symmetry` | `symmetry` |
| `right`, `top` | `slip` | `zeroGradient` |
| `ground`, `body` | `noSlip` | `zeroGradient` |

## Design and reference measurements

| Variable | Unit | Lower | Reference | Upper |
| --- | --- | ---: | ---: | ---: |
| `slant_angle_deg` | ° | 12 | 25 | 35 |
| `slant_length_m` | m | 0.15 | 0.222 | 0.25 |
| `diffuser_angle_deg` | ° | 0 | 2 | 8 |
| `nose_radius_m` | m | 0.06 | 0.1 | 0.12 |

| Role | Metric | Reference | Campaign bound |
| --- | --- | ---: | ---: |
| **Minimize** | `drag_coefficient` | 0.716885 | — |
| Constraint | `volume_m3` | 0.111309 m³ | ≥ 0.10797 m³ (3% margin) |

This smoke run checks the workflow on a coarse mesh. The saved iteration is a coarse-run
reference. Model limits include no support stilts, rounding only in side view.

## Reproduce the views

Set up [OpenFOAM and ParaView](../../README.md#set-up), then run from the repository root.

```bash
python3 examples/cfd/gallery.py ahmed_body
```

The workflow writes to `runs/cfd_gallery/ahmed_body/`. Open `figures/mesh/scene.pvsm`
or `figures/reference/scene.pvsm` in ParaView. Each `figure.json` records the saved time,
field association, color range, camera, and source run. See the
[shared workflow](../../README.md#run-one-case) for individual mesh, solve, and optimization commands.
