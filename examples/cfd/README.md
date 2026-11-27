# OpenFOAM design cases

Choose a case for its model, boundary conditions, checked mesh, solved field, and
reproduction steps. The 49 representative designs cover application areas from the
[CFD Support case studies](https://www.cfdsupport.com/case-studies/). The cases use
GP-Omni's Bayesian optimization with OpenFOAM objectives and constraints.

The previews use the coarse `smoke` reference runs. The
[verification report](verification.md) explains the checks, known mesh warnings,
and limits on interpreting the results.

## Cases

### Axial turbomachinery

| Case | Example |
| --- | --- |
| [`axial_fan`](cases/axial_fan/README.md) | Ducted axial fan, one blade passage |
| [`axial_fan_optimization`](cases/axial_fan_optimization/README.md) | Axial fan with a free blade count |
| [`axial_fan_acoustic`](cases/axial_fan_acoustic/README.md) | Sliding-mesh axial fan with pressure probes |
| [`double_rotor_fan`](cases/double_rotor_fan/README.md) | Counter-rotating fan |
| [`axial_pump`](cases/axial_pump/README.md) | Axial water pump with stator |
| [`parametric_axial_blood_pump`](cases/parametric_axial_blood_pump/README.md) | Miniature axial blood pump |
| [`axial_turbine`](cases/axial_turbine/README.md) | Axial turbine stage |
| [`kaplan_turbine`](cases/kaplan_turbine/README.md) | Kaplan runner behind guide vanes |
| [`ship_hull_propeller`](cases/ship_hull_propeller/README.md) | Marine propeller in open water |
| [`potsdam_propeller`](cases/potsdam_propeller/README.md) | Five-blade propeller advance sweep |
| [`propeller_dji_9450`](cases/propeller_dji_9450/README.md) | Two-blade drone propeller |
| [`propeller_optimization`](cases/propeller_optimization/README.md) | Three-blade propeller sizing |
| [`wind_turbine`](cases/wind_turbine/README.md) | Three-blade wind turbine |
| [`axial_compressor`](cases/axial_compressor/README.md) | Subsonic axial compressor stage |

### Radial and mixed-flow turbomachinery

| Case | Example |
| --- | --- |
| [`centrifugal_pump`](cases/centrifugal_pump/README.md) | Volute water pump on the bundled reference mesh |
| [`centrifugal_pump_optimization`](cases/centrifugal_pump_optimization/README.md) | Pump impeller and casing shape |
| [`centrifugal_pump_partial_overlap`](cases/centrifugal_pump_partial_overlap/README.md) | Sliding pump with an offset casing |
| [`fda_pump`](cases/fda_pump/README.md) | Small centrifugal blood pump |
| [`centrifugal_fan`](cases/centrifugal_fan/README.md) | Backswept centrifugal fan |
| [`francis_turbine`](cases/francis_turbine/README.md) | Francis turbine with casing and guide vanes |
| [`radial_turbine`](cases/radial_turbine/README.md) | Hot-gas radial inflow turbine |
| [`radial_compressor`](cases/radial_compressor/README.md) | Centrifugal compressor with casing |
| [`hecc_stage`](cases/hecc_stage/README.md) | Compressor stage with a vaned diffuser |

### Internal and process flows

| Case | Example |
| --- | --- |
| [`pipe`](cases/pipe/README.md) | Water pipe with a 180 degree bend |
| [`parametric_pipe`](cases/parametric_pipe/README.md) | Bend with nine free diameters |
| [`hydraulic_valve`](cases/hydraulic_valve/README.md) | Axisymmetric poppet valve |
| [`labyrinth_maze`](cases/labyrinth_maze/README.md) | Labyrinth seal filling with water |
| [`human_comfort`](cases/human_comfort/README.md) | Ventilated room with a seated occupant |
| [`rushton_impeller`](cases/rushton_impeller/README.md) | Rushton turbine in a stirred vessel |
| [`mechanical_siren`](cases/mechanical_siren/README.md) | Sliding-mesh siren |
| [`water_valve`](cases/water_valve/README.md) | Butterfly valve in a water line |
| [`intake_manifold_egr`](cases/intake_manifold_egr/README.md) | Intake manifold with exhaust gas recirculation |
| [`exhaust_manifold`](cases/exhaust_manifold/README.md) | Four-into-one exhaust manifold |
| [`cyclone_separator`](cases/cyclone_separator/README.md) | Reverse-flow cyclone |
| [`human_upper_airways`](cases/human_upper_airways/README.md) | Idealized upper airway |
| [`pelton_distributor`](cases/pelton_distributor/README.md) | Pelton turbine distributor |

### External aerodynamics

| Case | Example |
| --- | --- |
| [`flat_plate`](cases/flat_plate/README.md) | Turbulent flat plate against a drag correlation |
| [`naca_0012`](cases/naca_0012/README.md) | NACA 0012 airfoil in two dimensions |
| [`naca_0012_acoustic`](cases/naca_0012_acoustic/README.md) | Unsteady pressure near the NACA 0012 trailing edge |
| [`bullet_223`](cases/bullet_223/README.md) | Supersonic bullet |
| [`ahmed_body`](cases/ahmed_body/README.md) | Ahmed reference car body, half model |
| [`motorbike_benchmark`](cases/motorbike_benchmark/README.md) | Solver run time on the motorbike geometry |
| [`train_aerodynamics`](cases/train_aerodynamics/README.md) | Train car nose shape and yaw |
| [`spitfire`](cases/spitfire/README.md) | Fighter airframe lift and drag |
| [`building_wind_load`](cases/building_wind_load/README.md) | Tall building in an atmospheric wind profile |
| [`dimpled_ball`](cases/dimpled_ball/README.md) | Spinning dimpled ball |
| [`beam_ball`](cases/beam_ball/README.md) | Ball on a cantilever rod |
| [`formula_one`](cases/formula_one/README.md) | Simplified open-wheel racing car |
| [`drivaer`](cases/drivaer/README.md) | Simplified notchback sedan |

## Set up

Build the image from the repository root. It contains OpenCFD OpenFOAM v2406 and
the locked Python environment with foamlib and Gmsh.

```bash
examples/cfd/container/foam.sh build
```

Install ParaView with `pvpython` and use host Python 3.10 or later with Matplotlib for the
preview layout. Verify that ParaView can render off-screen. Some Linux builds
require an accessible X display even in off-screen mode. The
[post-processing guide](../postprocessing/README.md) describes the environment
used for the figures.

## Run one case

Run commands from the repository root. For example, to build, solve and visualize
the [pipe bend](cases/pipe/README.md), run:

```bash
python3 examples/cfd/gallery.py pipe
```

The gallery command creates or reuses a passing smoke mesh and a solved reference
with retained fields. It exports the selected field and saved time to VTK, then
renders both views using the settings in the case directory. It writes the
previews, editable `scene.pvsm` files and `figure.json` provenance under
`runs/cfd_gallery/<case-id>/figures/`. Open a state in ParaView and point it at the
corresponding `internal.vtu` in `mesh/galleryVTK/` or `reference/galleryVTK/` if
the run directory has moved. Generate full-resolution figures with
`examples/postprocessing/publication.py` and the directory containing the scene.

The 14 examples with a matching rotational `cyclic` pair also include a
full-annulus field, as shown in the [axial fan case](cases/axial_fan/README.md#full-annulus-field).
The gallery reads the pitch from the paired mesh faces and tiles the saved sector
through 360°. Scalars carry over to each sector and vectors rotate with the mesh.
The preview shows a transverse section and `figures/annulus/scene.pvsm` contains
the full 3D periodic reconstruction.

Use `--output` for a fresh gallery directory under `runs/`. Set `CFD_RUNS` to
change the host runs directory, which the launcher mounts at `/runs`. If an
export or render fails, read the named log and rerun the gallery command to
reuse the passing OpenFOAM runs. After changing a view recipe, move the affected
figure directory aside to render it again from the retained run.

The underlying commands work independently:

```bash
foam=examples/cfd/container/foam.sh
"$foam" python examples/cfd/run.py list
"$foam" python examples/cfd/run.py describe pipe
"$foam" python examples/cfd/run.py mesh pipe --output /runs/pipe_mesh
"$foam" python examples/cfd/run.py run pipe --output /runs/pipe --keep-fields
"$foam" python examples/cfd/run.py run pipe --output /runs/pipe_b \
    --design '{"diameter_m": 0.105, "bend_radius_m": 0.24}'
"$foam" python examples/cfd/run.py verify --output /runs/verify
"$foam" python examples/cfd/run.py verify --family radial_turbo --output /runs/verify_radial
```

Use a fresh output directory for each mesh or solve. `verify` resumes in its own
directory, skipping passing cases. The run requires at least 30 GB free.
Passing solves remove meshes and fields by default, retaining dictionaries,
logs, measurements, and `record.json`. Use `--keep-fields` when you plan to
export fields. A mesh-only run retains its mesh.

| Option | Meaning |
| --- | --- |
| `--level smoke` | Default coarse mesh with 200 iterations or up to 50 steps |
| `--level standard` | Finer mesh with second-order momentum and up to 2,000 iterations or steps. Verification covers the smoke level. |
| `--cores N` | MPI processes per design |
| `--timeout-s` | Wall time limit per design, 1,800 s by default |

## Optimization

```bash
foam=examples/cfd/container/foam.sh
"$foam" python examples/cfd/run.py optimize axial_fan --output /runs/axial_fan_bo --loop batch --cores 2
```

The campaign solves the reference first. Constraint bounds and objective scales
are computed from that run at the selected mesh level. Defaults are eight random
designs and four batches of two. Use `--loop async` for asynchronous optimization.
A failed design counts as infeasible. The campaign writes the best feasible
design to `summary.json`.

The constrained optimizer can take its best-so-far value from an infeasible
point, which can prevent improvement. Multi-objective and multi-fidelity
acquisitions are outside the supported workflow.

## Centrifugal water pump

The [dedicated pump case](cases/centrifugal_pump/README.md) includes the gallery
views and the commands for the longer standalone pump sizing workflow.

## Source layout

The [case directories](cases/) hold individual case definitions and view
recipes. The family modules share geometry and model helpers. The [foam
directory](foam/) holds the shared meshing, solver, measurement and optimization
pipeline. There are three mesh paths:

- Structured meshes are written as OpenFOAM `polyMesh` directly.
- `snappyHexMesh` uses generated surfaces, with Gmsh used for the internal CAD cases.
  You can substitute your own closed surface in the corresponding geometry routine.
- The bundled centrifugal pump reads its distributed mesh and case inputs.

Matching `cyclic` boundaries are used for rotationally periodic faces. Earlier
non-conforming `cyclicAMI` coupling could introduce artificial swirl in twisted
rotating passages and inflate the measured efficiency.

Code in this directory follows the repository license. The [bundled
pump](pump_case/README.md) retains GPL-3.0 licensing, and the motorbike surface is
read from the OpenFOAM installation at run time.
