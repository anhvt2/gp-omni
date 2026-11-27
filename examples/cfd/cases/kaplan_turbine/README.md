# Kaplan runner behind guide vanes

[CFD case index](../../README.md#axial-turbomachinery) · [Verification results](../../verification.md#axial-turbomachinery) · [Case source](__init__.py) · [View settings](view.json)

A simplified Kaplan stage uses axial guide vanes and a runner. Guide angle, runner offset and
rotational speed affect extracted power while the design seeks efficiency.

**Solver** `simpleFoam` · **Mesh** structured `polyMesh` · **Level** `smoke`

## Mesh and results

| Initial mesh with cell edges | Solved field |
| --- | --- |
| ![Initial kaplan_turbine mesh section](mesh.png) | ![Speed \|U\| (m/s) at iteration 200](results.png) |

Initial mesh and solved field on the `x = 0.14 m` plane. The results show point data for `U` at
iteration 200. The speed field is sampled along the guide-vane and runner passage. The model
omits the casing and draft tube.

`checkMesh` reported 17,496 cells and maximum non-orthogonality of 60.21°. It reported zero
failed quality checks. The measured flow imbalance was 0.00%.

### Full-annulus field

![Full-annulus solved field for kaplan_turbine](annulus.png)

The transverse section at `z = 0.2275 m` shows 11 rotational copies of the solved
passage. The editable `figures/annulus/scene.pvsm` contains the full 3D reconstruction.
Move the slice or show the merged volume in ParaView to inspect another location.

## Boundary conditions

| Patches | Velocity condition | `p` condition |
| --- | --- | --- |
| `blade_rotor`, `blade_vane`, `hub`, `hub_rotor`, `shroud` | `noSlip` | `zeroGradient` |
| `inlet` | `flowRateInletVelocity` | `zeroGradient` |
| `outlet` | `pressureInletOutletVelocity` | `fixedValue` |
| `periodic_a`, `periodic_b` | `cyclic` | `cyclic` |

## Design and reference measurements

| Variable | Unit | Lower | Reference | Upper |
| --- | --- | ---: | ---: | ---: |
| `guide_angle_deg` | ° | 40 | 52 | 62 |
| `runner_offset_deg` | ° | -6 | 4 | 14 |
| `speed_ratio` | — | 0.75 | 1 | 1.2 |

| Role | Metric | Reference | Campaign bound |
| --- | --- | ---: | ---: |
| **Maximize** | `efficiency` | 0.734638 | — |
| Constraint | `output_power_w` | 4405.76 W | ≥ 3965.18 W (10% margin) |

This smoke run checks the workflow on a coarse mesh. The saved iteration is a coarse-run
reference. Model limits include equal blade counts, frozen rotor, no tip clearance, axial guide
vanes, no spiral or draft tube.

## Reproduce the views

Set up [OpenFOAM and ParaView](../../README.md#set-up), then run from the repository root.

```bash
python3 examples/cfd/gallery.py kaplan_turbine
```

The workflow writes to `runs/cfd_gallery/kaplan_turbine/`. Open `figures/mesh/scene.pvsm`
or `figures/reference/scene.pvsm` in ParaView. Each `figure.json` records the saved time,
field association, color range, camera, and source run. See the
[shared workflow](../../README.md#run-one-case) for individual mesh, solve, and optimization commands.
