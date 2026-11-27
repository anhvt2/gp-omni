# OpenFOAM design cases

Forty-nine constrained design problems modelled on the application areas of the
[CFD Support case studies](https://www.cfdsupport.com/case-studies/). Each one builds its
geometry and mesh, solves the flow with OpenFOAM, measures an objective and constraints, and
plugs into GP-Omni's Bayesian optimization.

**These are representative problems, not replicas.** The geometry is generated here, no CFD
Support software or files are used, and results are not comparable with the published reports.

**The verification level is a pipeline check.** A passing `smoke` run shows that a case
meshes, solves and reports numbers. It does not show that the numbers are converged or
accurate. See [verification results](verification.md) for each case, including values that
are flagged as implausible.

## Set up

The cases run in a Docker image: OpenCFD OpenFOAM v2406 plus the repository's locked Python
environment and the `openfoam` dependency group ([foamlib](https://github.com/gerlero/foamlib),
GPL-3.0-only, and [Gmsh](https://gmsh.info/), GPL-2.0-or-later). Both are used only by these
examples. Torch is installed as the locked version's CPU build.

```bash
examples/cfd/container/foam.sh build        # about 4.3 GB
```

`foam.sh` runs any command from the repository root inside the image, as your user. Case
output goes to `runs/` (ignored by git), mounted at `/runs`. Set `CFD_RUNS` to use another
host directory.

## Run

```bash
foam="examples/cfd/container/foam.sh python examples/cfd/run.py"
$foam list
$foam describe axial_fan                      # variables, objective, limits, simplifications
$foam mesh axial_fan --output /runs/axial_fan_mesh
$foam run axial_fan --output /runs/axial_fan  # the reference design
$foam run axial_fan --output /runs/axial_fan_b \
    --design '{"inlet_offset_deg": 1, "camber_deg": 12, "chord_fraction": 0.28}'
$foam verify --output /runs/verify            # all cases; reruns only those not yet passed
$foam verify --family radial_turbo --output /runs/verify_radial
$foam optimize axial_fan --output /runs/axial_fan_bo --loop batch --cores 2
```

Every output directory must be new, except that `verify` resumes in its own directory.

| Option | Meaning |
|---|---|
| `--level smoke` | Default. Coarse mesh, 200 iterations or 50 time steps |
| `--level standard` | Twice the cells per direction, second-order momentum, 2000 iterations or steps. **Not verified** |
| `--cores N` | MPI processes per design |
| `--timeout-s` | Wall time limit per design, 1800 s by default |
| `--keep-fields` | Keep meshes and fields of passing runs |

A run stops before it starts if less than 30 GB is free. Passing runs delete their meshes
and field data and keep dictionaries, logs, measurements and `record.json`.
For a run selected for analysis, use `--keep-fields` and follow the
[VTK and ParaView workflow](../postprocessing/README.md#pipe-bend-cfd).

## Optimization

`optimize` solves the reference design first. Constraint limits are relative to it (for
example, pressure rise at least 90% of the reference), so they follow the mesh level. The
objective is scaled by the reference value. Defaults are 8 random designs and 4 batches of 2.
`--loop batch` uses the batch-sequential loop and `--loop async` the asynchronous one. A
design that fails to mesh or solve is recorded and counts as infeasible. `summary.json`
holds the best feasible design.

Before trusting a campaign, note the open library issue: constrained Bayesian optimization
can take its best-so-far value from an infeasible point, which can stop a campaign from
improving. Multi-objective and multi-fidelity acquisitions are not implemented.

## How a case is checked

| Check | Pass condition |
|---|---|
| Mesh | No negative-volume cells, non-orthogonality below 85. Other `checkMesh` warnings are recorded |
| Solver | Reaches the end, no non-finite residuals |
| Flow balance | Within 20% for steady cases with a measured inlet and outlet. Transient cases report it as `flow_mismatch` |
| Measurements | Objective and constraint values present and finite |

## Layout

| Path | Content |
|---|---|
| [run.py](run.py) | Command line |
| [cases/](cases/) | One module per group of cases |
| [foam/](foam/) | Shared meshing, case files, measurements, pipeline and campaigns |
| [container/](container/) | Image and launcher |
| [centrifugal_pump.py](centrifugal_pump.py) | The stand-alone pump example, also case `centrifugal_pump` |

Three meshing paths are used. **Structured** meshes (blade passages, ducts, two-dimensional
and axisymmetric domains, rooms and vessels) are written directly as `polyMesh` by
[hex_mesh.py](foam/hex_mesh.py), so they can be tested without OpenFOAM. **snappyHexMesh**
meshes complex bodies from generated surfaces, with Gmsh for internal flows. The pump uses
its **bundled** mesh.

To add real geometry to a `snappy` case, replace its generated triangles with those of your
own closed surface.

Rotationally periodic faces use OpenFOAM's matching `cyclic` coupling. A non-conforming
`cyclicAMI` coupling on twisted blade passages made a rotating zone create swirl out of
nothing, overstating fan efficiency by half.

## Centrifugal water pump

The [standalone pump](centrifugal_pump.py) minimizes shaft power at a fixed flow of
0.04167 m³/s while maintaining at least 90% of the reference static pressure rise.
Design variables are uniform mesh scale and impeller speed. The steady `simpleFoam`
model uses a multiple reference frame and `kOmegaSST` turbulence. Shaft power is
computed from the resisting rotor-wall moment and angular speed.

Run the commands in the container as shown. In a shell that has sourced the OpenCFD
OpenFOAM v2406 environment, replace `"$foam" python` with `uv run --locked python` and
use local output directories.

```bash
foam=examples/cfd/container/foam.sh
"$foam" python examples/cfd/centrifugal_pump.py --verify-only
"$foam" python examples/cfd/centrifugal_pump.py /runs/pump_baseline --baseline-only
"$foam" python examples/cfd/centrifugal_pump.py /runs/pump_run
```

The default campaign solves the reference and ten candidates at 3,000 iterations each.
Use `--batches 1` for six candidates. The driver records pressures, patch flows, power
and logs. It checks that both patch flows are within 3% of the prescribed rate and that
pressure rise and shaft power vary by at most 3% over the final ten samples. The model
uses fixed mesh connectivity, one operating point, and steady single-phase flow.
Engineering use requires mesh refinement and validation at the intended duty.
To plot the reference, export `/runs/pump_baseline/baseline` with `foamToVTK` and render the resulting
`internal.vtu`. See the [bundled case](pump_case/README.md) for model inputs,
reference measurements, checksums, and provenance.

## Licenses

Code in this folder is under the repository license. `pump_case/` is GPL-3.0, see its
[README](pump_case/README.md). The motorbike surface is read from the OpenFOAM installation
at run time and is not part of the repository.
