# Mesh preparation

Run these commands from the repository root. Keep meshes and solver output under the
Git-ignored `runs/` directory. Check that the disk has space for the selected cases and
reuse an existing OpenFOAM image when one is available.

## 3D FEM

The ten 3D cases use their existing Gmsh geometry, mesh sizing, and local refinements.
Gmsh on Linux requires the system OpenGL libraries, such as `libglu1-mesa`. For mesh
preparation and solves only, use a small environment with Gmsh, scikit-fem, and meshio.

```bash
mkdir -p runs
uv venv --python 3.12 runs/.venv_engineering
uv pip install --python runs/.venv_engineering/bin/python \
  gmsh==4.15.2 scikit-fem==12.0.2 meshio==5.3.5
runs/.venv_engineering/bin/python -m examples.preprocessing.fem prepare heat_sink runs/heat_sink_mesh
OPENBLAS_NUM_THREADS=1 runs/.venv_engineering/bin/python -m examples.preprocessing.fem solve \
  runs/heat_sink_mesh/mesh.npz runs/heat_sink
```

`prepare` takes a case name, a new output directory, and optional `--mesh-scale` and
`--design` arguments. A mesh scale below 1 refines the mesh. The default scale is 1,
and the default design uses the case's reference dimensions. Supply all design
variables as a JSON object to mesh a candidate. The preparation step writes three files.

| File | Content |
| --- | --- |
| `mesh.npz` | Linear tetrahedral connectivity and coordinates, case name, design, and mesh scale for `solve` |
| `mesh.vtu` | The same mesh in VTK XML with coordinates in mm |
| `mesh.json` | Design and mesh size statistics, including minimum and maximum tetrahedron volumes |

`solve` uses the saved connectivity directly. It writes `solution.vtu` with quadratic
tetrahedra and `record.json` with metrics, feasibility, and exported field names.

## OpenFOAM

Use the [OpenFOAM launcher](../cfd/README.md#set-up) to build the case and run `checkMesh`
without starting a solver. The `mesh` and `run` commands share the case builder and mesh
checks. Each command takes its own new output directory.

```bash
foam=examples/cfd/container/foam.sh
"$foam" python examples/cfd/run.py mesh pipe --output /runs/pipe_mesh
"$foam" foamToVTK -case /runs/pipe_mesh -time 0 -no-boundary -no-fields
```

The mesh-only directory contains the native OpenFOAM case with its initial fields,
`log.checkMesh`, and a `record.json` with stage `meshed`. The optional export produces
`runs/pipe_mesh/VTK/pipe_mesh_0/internal.vtu`, which can be plotted using
[the mesh renderer](../postprocessing/README.md#mesh-plots).
