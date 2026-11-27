# Train car nose shape and yaw

[CFD case index](../../README.md#external-aerodynamics) · [Verification results](../../verification.md#external-aerodynamics) · [Case source](__init__.py) · [View settings](view.json)

A single rounded train car is simulated above a stationary ground plane at a prescribed inlet
speed. Nose length, shape and yaw affect drag under a volume constraint.

**Solver** `simpleFoam` · **Mesh** `snappyHexMesh` · **Level** `smoke`

## Mesh and results

| Initial mesh with cell edges | Solved field |
| --- | --- |
| ![Initial train_aerodynamics mesh section](mesh.png) | ![Speed \|U\| (m/s) at iteration 200](results.png) |

Initial mesh and solved field on the `y = 0.02 m` plane. The results show point data for `U` at
iteration 200. The near-center longitudinal slice displays the nose, underbody and modeled wake.

`checkMesh` reported 61,342 cells and maximum non-orthogonality of 45.06°. It reported zero
failed quality checks. The measured flow imbalance was 0.00%.

## Boundary conditions

| Patches | Velocity condition | `p` condition |
| --- | --- | --- |
| `inlet` | `fixedValue` | `zeroGradient` |
| `outlet` | `pressureInletOutletVelocity` | `fixedValue` |
| `left`, `right`, `top` | `slip` | `zeroGradient` |
| `ground`, `train` | `noSlip` | `zeroGradient` |

## Design and reference measurements

| Variable | Unit | Lower | Reference | Upper |
| --- | --- | ---: | ---: | ---: |
| `nose_length_m` | m | 0.5 | 0.9 | 1.4 |
| `nose_exponent` | — | 0.35 | 0.55 | 0.8 |
| `yaw_deg` | ° | 0 | 3 | 10 |

| Role | Metric | Reference | Campaign bound |
| --- | --- | ---: | ---: |
| **Minimize** | `drag_coefficient` | 0.288646 | — |
| Constraint | `volume_m3` | 1.15257 m³ | ≥ 1.09494 m³ (5% margin) |

This smoke run checks the workflow on a coarse mesh. The saved iteration is a coarse-run
reference. Model limits include single car, stationary ground.

## Reproduce the views

Set up [OpenFOAM and ParaView](../../README.md#set-up), then run from the repository root.

```bash
python3 examples/cfd/gallery.py train_aerodynamics
```

The workflow writes to `runs/cfd_gallery/train_aerodynamics/`. Open `figures/mesh/scene.pvsm`
or `figures/reference/scene.pvsm` in ParaView. Each `figure.json` records the saved time,
field association, color range, camera, and source run. See the
[shared workflow](../../README.md#run-one-case) for individual mesh, solve, and optimization commands.
