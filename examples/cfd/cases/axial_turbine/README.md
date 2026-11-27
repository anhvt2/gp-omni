# Axial turbine stage

[CFD case index](../../README.md#axial-turbomachinery) · [Verification results](../../verification.md#axial-turbomachinery) · [Case source](__init__.py) · [View settings](view.json)

A periodic guide-vane and runner passage represents an axial turbine at a specified flow. Vane
angle, runner exit offset and speed ratio change extracted power and efficiency.

**Solver** `simpleFoam` · **Mesh** structured `polyMesh` · **Level** `smoke`

## Mesh and results

| Initial mesh with cell edges | Solved field |
| --- | --- |
| ![Initial axial_turbine mesh section](mesh.png) | ![Speed \|U\| (m/s) at iteration 200](results.png) |

Initial mesh and solved field on the `x = 0.25 m` plane. The results show point data for `U` at
iteration 200. The cut passes through the vane and runner region. The metric table reports the
shaft power associated with this reference.

`checkMesh` reported 17,496 cells and maximum non-orthogonality of 61.60°. It reported zero
failed quality checks. The measured flow imbalance was 0.00%.

### Full-annulus field

![Full-annulus solved field for axial_turbine](annulus.png)

The transverse section at `z = 0.1625 m` shows 26 rotational copies of the solved
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
| `vane_angle_deg` | ° | 50 | 60 | 68 |
| `exit_offset_deg` | ° | -6 | 4 | 14 |
| `speed_ratio` | — | 0.75 | 1 | 1.2 |

| Role | Metric | Reference | Campaign bound |
| --- | --- | ---: | ---: |
| **Maximize** | `efficiency` | 0.759126 | — |
| Constraint | `output_power_w` | 870.898 W | ≥ 783.808 W (10% margin) |

This smoke run checks the workflow on a coarse mesh. The saved iteration is a coarse-run
reference. Model limits include equal blade counts, frozen rotor, no tip clearance, no spiral
casing.

## Reproduce the views

Set up [OpenFOAM and ParaView](../../README.md#set-up), then run from the repository root.

```bash
python3 examples/cfd/gallery.py axial_turbine
```

The workflow writes to `runs/cfd_gallery/axial_turbine/`. Open `figures/mesh/scene.pvsm`
or `figures/reference/scene.pvsm` in ParaView. Each `figure.json` records the saved time,
field association, color range, camera, and source run. See the
[shared workflow](../../README.md#run-one-case) for individual mesh, solve, and optimization commands.
