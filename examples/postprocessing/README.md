# Simulation results and publication figures

Use ParaView and its bundled VTK Python modules to render `.vtu` files and sample
solution fields along a line. A second script uses Matplotlib, NumPy, and Pillow to
write 600 DPI PNG and LZW-compressed TIFF figures with a physical scale bar and
color legend. Sampled profiles are saved as CSV and as vector PDF and SVG.

Run from the repository root. `pvpython` must be able to render off-screen. ParaView
builds that use GLX still need an accessible X display, such as an existing display
or one provided by Xvfb. On Ubuntu 22.04 the packaged ParaView 5.10 worked with
`DISPLAY=:0 PYTHONPATH=/usr/lib/python3/dist-packages`. Use a separate Python interpreter
with Matplotlib installed for `publication.py` if the ParaView interpreter lacks it.
For example, use `uv run --no-project --with matplotlib python` in place of `python3`.
[ParaView scripting and export documentation](https://docs.paraview.org/en/v5.10.0/UsersGuide/savingResults.html)

## Heat sink (FEM)

[Prepare and solve the heat sink](../preprocessing/README.md#3d-fem), then plot the
temperature rise in K on the undeformed solid. The line runs through the base at
`y = 20 mm` and `z = 1.5 mm`.

```bash
pvpython --force-offscreen-rendering examples/postprocessing/render.py \
  runs/heat_sink/solution.vtu runs/heat_sink_figures \
  --field temperature_rise_k --label 'Temperature rise (K)' --length-unit mm \
  --line 1 20 1.5 39 20 1.5
python3 examples/postprocessing/publication.py runs/heat_sink_figures
```

The [renderer](render.py) also accepts structural displacement vectors in mm and
element-average von Mises stress in MPa. Look up the arrays in the solve's `record.json`.
For example, the drone arm exports `thrust_displacement_mm` on points and
`side_impact_von_mises_mpa` on cells. Exported stresses are means over each tetrahedron.
The optimization reports volume-weighted stress percentiles, sometimes over a restricted
region. Frequencies remain scalar metrics in the record.

## Cantilever (FEM)

With the [FEM environment](../preprocessing/README.md#3d-fem), solve a 60 mm root
and 45 mm tip design. The VTU samples the cubic Hermite solution at eight
intervals per element. Its geometry is the undeformed beam axis.

```bash
runs/.venv_engineering/bin/python -m examples.preprocessing.fem cantilever \
  runs/cantilever --root-height-mm 60 --tip-ratio 0.75
pvpython --force-offscreen-rendering examples/postprocessing/render.py \
  runs/cantilever/solution.vtu runs/cantilever_figures \
  --field downward_deflection_mm --label 'Downward deflection (mm)' --length-unit mm \
  --line 0 0 0 600 0 0 --view 0 0 1
python3 examples/postprocessing/publication.py runs/cantilever_figures
```

The profile gives deflection along the span. This design deflects about 1.59 mm
at the tip under the 1 kN load, above the 1.5 mm optimization limit.

## Pipe bend (CFD)

The [pipe bend case guide](../cfd/cases/pipe/README.md) shows the mesh and solved
field and gives the single command that generates both. The [CFD case index](../cfd/README.md)
links to the same workflow for every OpenFOAM example.

Run one case at the smoke level and retain its fields. Export only the final velocity
field. The cross-section passes through the center plane at `y = 0 m` and the sampled
line crosses the pipe radially at the middle of the bend.

```bash
foam=examples/cfd/container/foam.sh
"$foam" python examples/cfd/run.py run pipe --output /runs/pipe --keep-fields
"$foam" foamToVTK -case /runs/pipe -latestTime -no-boundary -fields '(U)'
pvpython --force-offscreen-rendering examples/postprocessing/render.py \
  runs/pipe/VTK/pipe_200/internal.vtu runs/pipe_figures \
  --field U --label 'Speed (m/s)' --length-unit m \
  --slice 0 0 0 0 -1 0 --line 0.19 0 0.25 0.31 0 0.25
python3 examples/postprocessing/publication.py runs/pipe_figures
```

The time in the VTU path follows the selected solver output. For a different time,
case or mesh level, use that export's `internal.vtu`. OpenFOAM can write the same field
on points and cells. The command above uses the exported point values. Pass
`--association CELLS` for piecewise cell values. Check field dimensions in the solver
output before labelling a plot. For example, some incompressible cases store
kinematic pressure in `m²/s²`.

To save disk space, retain raw solver fields only for runs selected for analysis.
Passing CFD runs remove raw fields by default. `-latestTime`,
`-no-boundary`, and `-fields` limit each VTK export. Keep the solver record, logs,
VTU, and figures for the runs you want to reproduce.

## Mesh plots

Omit `--field` to plot the mesh edges. For a cross-section, give the slice origin and
normal in the native coordinate system.

```bash
pvpython --force-offscreen-rendering examples/postprocessing/render.py \
  runs/pipe_mesh/VTK/pipe_mesh_0/internal.vtu runs/pipe_mesh_figures \
  --length-unit m --slice 0 0 0 0 -1 0
python3 examples/postprocessing/publication.py runs/pipe_mesh_figures
```

## Figure settings and outputs

- Field scenes use ParaView's **Fast** preset on its **Warm Gray** background.
  The publication legend uses the same Lab-interpolated colors and numeric range
  as the scene.
  [Kitware's colormap and background guidance](https://www.kitware.com/new-default-colormap-and-background-in-next-paraview-release/)
- A field is coloured by its scalar value or vector magnitude. Use the same
  `--range MIN MAX` across designs for a direct comparison. Colorbar extensions mark
  values outside the specified range. `--view` controls the direction toward the camera
  and `--up` sets its vertical axis. Use `--frame XMIN XMAX YMIN YMAX ZMIN ZMAX`
  to frame a region of interest in native coordinates.
- `--line` accepts two points in the full volume. `--samples` sets the number of
  uniformly spaced positions. The CSV includes coordinates and a validity mask.
  Samples outside the domain appear as gaps in the profile.
- `scene.pvsm` preserves the editable ParaView view. If the VTU moves, point ParaView
  to its new location when loading the state. `scene.png` is the base render.
  `figure.json` records the source, VTK and ParaView versions, units, selected field,
  colour range and stops, background, view, and sampling parameters.
- `field.png` and `field.tiff` default to 180 mm wide at 600 DPI. Set
  `--width-mm 85` on `publication.py` for a single-column figure.
  `profile.pdf`, `profile.svg`, and `profile.csv` are written when a line is requested.
  `publication.py --preview` writes a compact PNG scene and, when sampled, a PNG profile.
- Report the case, design, mesh level or mesh scale, field units, and slice or line
  coordinates with the figure. For engineering conclusions, check resolution,
  convergence and model validity for the intended operating conditions.
