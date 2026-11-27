# Engineering design examples

GP-Omni couples Bayesian optimization with finite element and flow solvers. The examples
below include reference designs, optimization commands, and workflows for preparing meshes
and turning simulation fields into figures.

| Examples | Start here |
| --- | --- |
| 3D finite element analysis | [Ten structural and thermal designs](fem/README.md) |
| Beam sizing | [Tapered aluminum cantilever](fem/README.md#tapered-aluminum-cantilever) |
| Computational fluid dynamics | [49 OpenFOAM design cases](cfd/README.md) |
| Pump sizing | [Centrifugal water pump](cfd/README.md#centrifugal-water-pump) |
| Mesh preparation | [Gmsh and OpenFOAM workflows](preprocessing/README.md) |
| Results and figures | [ParaView scenes, VTK profiles, and publication exports](postprocessing/README.md) |

## From simulation to figure

| Heat sink · Temperature rise above ambient | Pipe bend · Speed on the mid-plane |
| --- | --- |
| ![Temperature rise across a baseplate and pin fins with a 20 mm scale bar](assets/heat_sink_temperature.png) | ![Flow speed in a U-shaped pipe with a 0.2 m scale bar](assets/pipe_bend_speed.png) |
| [Prepare, solve and plot the heat sink](postprocessing/README.md#heat-sink-fem) | [Solve, export and plot the pipe bend](postprocessing/README.md#pipe-bend-cfd) |

The previews use reference designs. The pipe bend is a coarse smoke run. The heat-sink
model uses an assumed convection coefficient. See the case guides for verification and
model limits.

Run commands from the repository root and use a fresh directory under `runs/` for each
mesh or simulation. [Numerical examples](../tests/numerical/) and
[analytical benchmarks](../tests/bo_functions/) live under `tests/`.
