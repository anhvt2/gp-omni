# Subsonic axial compressor stage

[CFD case index](../../README.md#axial-turbomachinery) · [Verification results](../../verification.md#axial-turbomachinery) · [Case source](__init__.py) · [View settings](view.json)

A representative compressible compressor stage has a rotor and stator with matching periodic
sectors. Camber, stator angle and speed affect efficiency and pressure ratio.

**Solver** `rhoSimpleFoam` · **Mesh** structured `polyMesh` · **Level** `smoke`

## Mesh and results

| Initial mesh with cell edges | Solved field |
| --- | --- |
| ![Initial axial_compressor mesh section](mesh.png) | ![Static pressure (Pa) at iteration 200](results.png) |

Initial mesh and solved field on the `x = 0.157 m` plane. The results show point data for `p` at
iteration 200. The static-pressure cut follows the stage. The reported pressure ratio is a bulk
stage metric.

`checkMesh` reported 17,064 cells and maximum non-orthogonality of 59.81°. It reported zero
failed quality checks. The measured flow imbalance was 0.00%.

### Full-annulus field

![Full-annulus solved field for axial_compressor](annulus.png)

The transverse section at `z = 0.062 m` shows 18 rotational copies of the solved
passage. The editable `figures/annulus/scene.pvsm` contains the full 3D reconstruction.
Move the slice or show the merged volume in ParaView to inspect another location.

## Boundary conditions

| Patches | Velocity condition | `p` condition |
| --- | --- | --- |
| `blade_rotor`, `blade_stator`, `hub`, `hub_rotor`, `shroud` | `noSlip` | `zeroGradient` |
| `inlet` | `flowRateInletVelocity` | `zeroGradient` |
| `outlet` | `pressureInletOutletVelocity` | `fixedValue` |
| `periodic_a`, `periodic_b` | `cyclic` | `cyclic` |

## Design and reference measurements

| Variable | Unit | Lower | Reference | Upper |
| --- | --- | ---: | ---: | ---: |
| `camber_deg` | ° | 4 | 8 | 14 |
| `stator_angle_deg` | ° | 20 | 32 | 45 |
| `speed_ratio` | — | 0.8 | 1 | 1.1 |

| Role | Metric | Reference | Campaign bound |
| --- | --- | ---: | ---: |
| **Maximize** | `efficiency` | 0.881114 | — |
| Constraint | `pressure_ratio` | 1.01891 | ≥ 1.00872 (1% margin) |

This smoke run checks the workflow on a coarse mesh. The saved iteration is a coarse-run
reference. Model limits include equal blade counts, frozen rotor, no tip clearance.

## Reproduce the views

Set up [OpenFOAM and ParaView](../../README.md#set-up), then run from the repository root.

```bash
python3 examples/cfd/gallery.py axial_compressor
```

The workflow writes to `runs/cfd_gallery/axial_compressor/`. Open `figures/mesh/scene.pvsm`
or `figures/reference/scene.pvsm` in ParaView. Each `figure.json` records the saved time,
field association, color range, camera, and source run. See the
[shared workflow](../../README.md#run-one-case) for individual mesh, solve, and optimization commands.
