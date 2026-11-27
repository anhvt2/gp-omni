# Bend with nine free diameters

[CFD case index](../../README.md#internal-and-process-flows) · [Verification results](../../verification.md#internal-and-process-flows) · [Case source](__init__.py) · [View settings](view.json)

Nine diameters along a fixed 180° bend vary the cross-section while the straight ends stay at
100 mm. The target is total pressure loss at limited fluid volume.

**Solver** `simpleFoam` · **Mesh** structured `polyMesh` · **Level** `smoke`

## Mesh and results

| Initial mesh with cell edges | Solved field |
| --- | --- |
| ![Initial parametric_pipe mesh section](mesh.png) | ![Speed \|U\| (m/s) at iteration 200](results.png) |

Initial mesh and solved field on the `y = 0 m` plane. The results show point data for `U` at
iteration 200. The reference uses the same constant diameter and curve as the pipe case. Its
slice and radial profile show the common baseline.

`checkMesh` reported 15,680 cells and maximum non-orthogonality of 26.06°. It reported zero
failed quality checks. The measured flow imbalance was 0.00%.

![Speed |U| (m/s) sampled along a line](profile.png)

The profile runs from `(0.19, 0, 0.25)` to `(0.31, 0, 0.25)` m.

## Boundary conditions

| Patches | Velocity condition | `p` condition |
| --- | --- | --- |
| `inlet` | `flowRateInletVelocity` | `zeroGradient` |
| `outlet` | `pressureInletOutletVelocity` | `fixedValue` |
| `walls` | `noSlip` | `zeroGradient` |

## Design and reference measurements

| Variable | Unit | Lower | Reference | Upper |
| --- | --- | ---: | ---: | ---: |
| `diameter_1_m` | m | 0.085 | 0.1 | 0.12 |
| `diameter_2_m` | m | 0.085 | 0.1 | 0.12 |
| `diameter_3_m` | m | 0.085 | 0.1 | 0.12 |
| `diameter_4_m` | m | 0.085 | 0.1 | 0.12 |
| `diameter_5_m` | m | 0.085 | 0.1 | 0.12 |
| `diameter_6_m` | m | 0.085 | 0.1 | 0.12 |
| `diameter_7_m` | m | 0.085 | 0.1 | 0.12 |
| `diameter_8_m` | m | 0.085 | 0.1 | 0.12 |
| `diameter_9_m` | m | 0.085 | 0.1 | 0.12 |

| Role | Metric | Reference | Campaign bound |
| --- | --- | ---: | ---: |
| **Minimize** | `pressure_loss_pa` | 627.181 Pa | — |
| Constraint | `volume_m3` | 0.0155933 m³ | ≤ 0.0163729 m³ (5% margin) |

This smoke run checks the workflow on a coarse mesh. The saved iteration is a coarse-run
reference.

## Reproduce the views

Set up [OpenFOAM and ParaView](../../README.md#set-up), then run from the repository root.

```bash
python3 examples/cfd/gallery.py parametric_pipe
```

The workflow writes to `runs/cfd_gallery/parametric_pipe/`. Open `figures/mesh/scene.pvsm`
or `figures/reference/scene.pvsm` in ParaView. Each `figure.json` records the saved time,
field association, color range, camera, and source run. See the
[shared workflow](../../README.md#run-one-case) for individual mesh, solve, and optimization commands.
