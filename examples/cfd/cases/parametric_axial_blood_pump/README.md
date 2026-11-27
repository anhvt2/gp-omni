# Miniature axial blood pump

[CFD case index](../../README.md#axial-turbomachinery) · [Verification results](../../verification.md#axial-turbomachinery) · [Case source](__init__.py) · [View settings](view.json)

A small rotor and stator move a single-phase blood analogue through a periodic annulus. Pitch,
hub ratio and rotor speed change the measured blade-wall shear while maintaining pressure rise.

**Solver** `simpleFoam` · **Mesh** structured `polyMesh` · **Level** `smoke`

## Mesh and results

| Initial mesh with cell edges | Solved field |
| --- | --- |
| ![Initial parametric_axial_blood_pump mesh section](mesh.png) | ![Speed \|U\| (m/s) at iteration 200](results.png) |

Initial mesh and solved field on the `x = 0.0045 m` plane. The results show point data for `U`
at iteration 200. The speed image shows the passage flow. The reported wall-shear quantity is a
case metric derived from the rotor wall fields.

`checkMesh` reported 16,848 cells and maximum non-orthogonality of 70.12°. It reported zero
failed quality checks. The measured flow imbalance was 0.01%.

### Full-annulus field

![Full-annulus solved field for parametric_axial_blood_pump](annulus.png)

The transverse section at `z = 0.0096 m` shows 5 rotational copies of the solved
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
| `pitch_deg` | ° | 0 | 4 | 8 |
| `hub_ratio` | — | 0.4 | 0.5 | 0.62 |
| `speed_rad_s` | rad/s | 500 | 650 | 800 |

| Role | Metric | Reference | Campaign bound |
| --- | --- | ---: | ---: |
| **Minimize** | `wall_shear_pa` | 134.486 Pa | — |
| Constraint | `pressure_rise_pa` | 5053.08 Pa | ≥ 4295.12 Pa (15% margin) |

This smoke run checks the workflow on a coarse mesh. The saved iteration is a coarse-run
reference. Model limits include equal blade counts, frozen rotor, no tip clearance, wall shear
in place of a blood damage model.

## Reproduce the views

Set up [OpenFOAM and ParaView](../../README.md#set-up), then run from the repository root.

```bash
python3 examples/cfd/gallery.py parametric_axial_blood_pump
```

The workflow writes to `runs/cfd_gallery/parametric_axial_blood_pump/`. Open `figures/mesh/scene.pvsm`
or `figures/reference/scene.pvsm` in ParaView. Each `figure.json` records the saved time,
field association, color range, camera, and source run. See the
[shared workflow](../../README.md#run-one-case) for individual mesh, solve, and optimization commands.
