# Axial water pump with stator

[CFD case index](../../README.md#axial-turbomachinery) · [Verification results](../../verification.md#axial-turbomachinery) · [Case source](__init__.py) · [View settings](view.json)

This water pump combines a rotor and stator in a periodic annular passage. The design varies
rotor incidence and camber and stator offset at the prescribed operating point.

**Solver** `simpleFoam` · **Mesh** structured `polyMesh` · **Level** `smoke`

## Mesh and results

| Initial mesh with cell edges | Solved field |
| --- | --- |
| ![Initial axial_pump mesh section](mesh.png) | ![Kinematic pressure (m²/s²) at iteration 200](results.png) |

Initial mesh and solved field on the `x = 0.068 m` plane. The results show point data for `p` at
iteration 200. The pressure cut follows the stage in the axial direction. The objective and
constraint use the patch pressure rise and shaft work.

`checkMesh` reported 16,848 cells and maximum non-orthogonality of 67.02°. It reported zero
failed quality checks. The measured flow imbalance was 0.00%.

### Full-annulus field

![Full-annulus solved field for axial_pump](annulus.png)

The transverse section at `z = 0.064 m` shows 7 rotational copies of the solved
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
| `incidence_deg` | ° | -2 | 2 | 6 |
| `camber_deg` | ° | 5 | 10 | 16 |
| `stator_offset_deg` | ° | -6 | 0 | 6 |

| Role | Metric | Reference | Campaign bound |
| --- | --- | ---: | ---: |
| **Maximize** | `efficiency` | 0.917024 | — |
| Constraint | `pressure_rise_pa` | 26958.8 Pa | ≥ 24262.9 Pa (10% margin) |

This smoke run checks the workflow on a coarse mesh. The saved iteration is a coarse-run
reference. Model limits include equal blade counts, frozen rotor, no tip clearance.

## Reproduce the views

Set up [OpenFOAM and ParaView](../../README.md#set-up), then run from the repository root.

```bash
python3 examples/cfd/gallery.py axial_pump
```

The workflow writes to `runs/cfd_gallery/axial_pump/`. Open `figures/mesh/scene.pvsm`
or `figures/reference/scene.pvsm` in ParaView. Each `figure.json` records the saved time,
field association, color range, camera, and source run. See the
[shared workflow](../../README.md#run-one-case) for individual mesh, solve, and optimization commands.
