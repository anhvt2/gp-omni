# Marine propeller in open water

[CFD case index](../../README.md#axial-turbomachinery) · [Verification results](../../verification.md#axial-turbomachinery) · [Case source](__init__.py) · [View settings](view.json)

An open-water periodic propeller case uses prescribed axial inflow. Pitch, chord and speed alter
efficiency while thrust is constrained.

**Solver** `simpleFoam` · **Mesh** structured `polyMesh` · **Level** `smoke`

## Mesh and results

| Initial mesh with cell edges | Solved field |
| --- | --- |
| ![Initial ship_hull_propeller mesh section](mesh.png) | ![Speed \|U\| (m/s) at iteration 200](results.png) |

Initial mesh and solved field on the `x = 0.12 m` plane. The results show point data for `U` at
iteration 200. The meridional slice samples flow through and behind the representative blade
passage. The domain contains the propeller and a slip far field.

`checkMesh` reported 22,320 cells and maximum non-orthogonality of 64.62°. It reported zero
failed quality checks. The measured flow imbalance was 0.00%.

### Full-annulus field

![Full-annulus solved field for ship_hull_propeller](annulus.png)

The transverse section at `z = 0.135 m` shows 4 rotational copies of the solved
passage. The editable `figures/annulus/scene.pvsm` contains the full 3D reconstruction.
Move the slice or show the merged volume in ParaView to inspect another location.

## Boundary conditions

| Patches | Velocity condition | `p` condition |
| --- | --- | --- |
| `blade_rotor`, `hub`, `hub_rotor` | `noSlip` | `zeroGradient` |
| `farfield` | `slip` | `zeroGradient` |
| `inlet` | `fixedValue` | `zeroGradient` |
| `outlet` | `pressureInletOutletVelocity` | `fixedValue` |
| `periodic_a`, `periodic_b` | `cyclic` | `cyclic` |

## Design and reference measurements

| Variable | Unit | Lower | Reference | Upper |
| --- | --- | ---: | ---: | ---: |
| `pitch_deg` | ° | 2 | 5 | 10 |
| `chord_fraction` | — | 0.1 | 0.14 | 0.18 |
| `speed_ratio` | — | 0.85 | 1 | 1.1 |

| Role | Metric | Reference | Campaign bound |
| --- | --- | ---: | ---: |
| **Maximize** | `efficiency` | 0.737813 | — |
| Constraint | `thrust_n` | 59.7048 N | ≥ 53.7343 N (10% margin) |

This smoke run checks the workflow on a coarse mesh. The saved iteration is a coarse-run
reference. Model limits include axial inflow, slip far field, low tip speed ratio set by the
mesher's angle limit, no hull.

## Reproduce the views

Set up [OpenFOAM and ParaView](../../README.md#set-up), then run from the repository root.

```bash
python3 examples/cfd/gallery.py ship_hull_propeller
```

The workflow writes to `runs/cfd_gallery/ship_hull_propeller/`. Open `figures/mesh/scene.pvsm`
or `figures/reference/scene.pvsm` in ParaView. Each `figure.json` records the saved time,
field association, color range, camera, and source run. See the
[shared workflow](../../README.md#run-one-case) for individual mesh, solve, and optimization commands.
