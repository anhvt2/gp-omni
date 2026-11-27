<p align="center">
  <img src="docs/assets/gp_omni_banner.svg" alt="GP-Omni — Gaussian processes. More possibilities. A GP curve and uncertainty band connect observations to a proposed experiment." width="100%">
</p>

<h1 align="center">GP-Omni</h1>
<p align="center"><strong>Make every experiment count.</strong><br>
Composable Gaussian processes for expensive questions and better next experiments.</p>

<p align="center">
  <a href="https://github.com/anhvt2/gp-omni/actions/workflows/ci.yml"><img src="https://github.com/anhvt2/gp-omni/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="docs/getting_started.md#install"><img src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&amp;logoColor=white" alt="Python 3.12+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Apache 2.0 license"></a>
  <a href="docs/README.md"><img src="https://img.shields.io/badge/docs-get%20started-0E8A16?logo=readthedocs&amp;logoColor=white" alt="Documentation"></a>
</p>

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&amp;logoColor=white" alt="Built with PyTorch"></a>
  <a href="https://gpytorch.ai/"><img src="https://img.shields.io/badge/GPyTorch-3B6EA8" alt="Built with GPyTorch"></a>
  <a href="https://botorch.org/"><img src="https://img.shields.io/badge/BoTorch-005A9C" alt="Built with BoTorch"></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/lint-Ruff-D7FF64?logo=ruff&amp;logoColor=white" alt="Lint with Ruff"></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/dependencies-uv-DE5FE9?logo=uv&amp;logoColor=white" alt="Dependencies managed with uv"></a>
</p>

<p align="center">
  <a href="examples/README.md"><strong>Engineering design optimization examples</strong></a><br>
  <a href="examples/fem/README.md"><img src="https://img.shields.io/badge/scikit--fem-FEM-2C5282" alt="scikit-fem for finite element analysis in the FEM examples"></a>
  <a href="examples/fem/common/mesh.py"><img src="https://img.shields.io/badge/Gmsh-CAD%20%26%20meshing-5B4B8A" alt="Gmsh for CAD and meshing in the 3D FEM examples"></a>
  <a href="examples/cfd/README.md"><img src="https://img.shields.io/badge/OpenFOAM-CFD-16736A" alt="OpenFOAM for flow simulation in the CFD examples"></a>
</p>

<p align="center">
  <a href="docs/getting_started.md"><strong>Get started →</strong></a> &nbsp; · &nbsp;
  <a href="docs/README.md">Explore the docs</a> &nbsp; · &nbsp;
  <a href="docs/test_function_accuracy.md">See the benchmarks</a>
</p>

## An expensive evaluation deserves a smarter next step

Simulations take time. Experiments cost money. GP-Omni learns from the observations you
already have, estimates what remains uncertain, and helps choose where to look next.

Build a surrogate for your own workflow, or connect an experiment to the built-in
Bayesian optimization loops. Powered by **PyTorch, GPyTorch, and BoTorch**.

## More ways to model your world

| **Learn across responses** | **Connect levels of fidelity** |
| :--- | :--- |
| Model correlated outputs together, sharing structure across the quantities you care about. | Combine cheap approximations with expensive observations in a multi-fidelity surrogate. |
| **Make room for more data** | **Find structure in many inputs** |
| Use inducing-point approximations when a dense GP becomes too costly. | Model additive effects and low-order interactions across input dimensions. |

**Four capabilities. Combine them your way.** Use correlated outputs, multiple fidelities,
sparse inference, and additive kernels together for regression and feasibility classification.
[Find your model →](docs/use-cases.md)

## From observations to the next experiment

<p align="center">
  <strong>Observe</strong> &nbsp; → &nbsp; <strong>Learn</strong> &nbsp; → &nbsp; <strong>Choose</strong> &nbsp; → &nbsp; <strong>Evaluate</strong> &nbsp; ↻
</p>

Run one evaluation at a time, launch coordinated batches, or keep workers busy as results
arrive. Learn unknown feasibility constraints alongside the objective.
[Run your first optimization →](docs/getting_started.md#plug-in-your-objective)

The built-in loops maximize one scalar objective at one fidelity. Multi-output optimization,
Pareto search, and cost-aware fidelity selection need a custom acquisition and workflow.

## Example gallery

Connect the loop to a finite element or flow solver. Select an example to see its model and
how to reproduce it.

| [Pin-fin heat sink](examples/postprocessing/README.md#heat-sink-fem) | [Pipe bend](examples/cfd/cases/pipe/README.md) | [Ducted axial fan](examples/cfd/cases/axial_fan/README.md) |
| :---: | :---: | :---: |
| ![Temperature rise in kelvins across the base and fins of a heat sink](examples/assets/heat_sink_temperature.png) | ![Flow speed through the mid-plane of a 180-degree pipe bend](examples/assets/pipe_bend_speed.png) | ![Flow speed across a full annulus reconstructed from one fan passage](examples/cfd/cases/axial_fan/annulus.png) |
| FEM · temperature rise | CFD · flow speed | CFD · flow speed |

| [Pump impeller and casing](examples/cfd/cases/centrifugal_pump_optimization/README.md) | [Ahmed reference car body](examples/cfd/cases/ahmed_body/README.md) | [Cyclone separator](examples/cfd/cases/cyclone_separator/README.md) |
| :---: | :---: | :---: |
| ![Kinematic pressure through the pump impeller and spiral casing](examples/cfd/cases/centrifugal_pump_optimization/results.png) | ![Flow speed in a longitudinal section around the car body](examples/cfd/cases/ahmed_body/results.png) | ![Flow speed inside a reverse-flow cyclone](examples/cfd/cases/cyclone_separator/results.png) |
| CFD · pressure | CFD · flow speed | CFD · flow speed |

**[All examples](examples/README.md)** ·
[10 3D finite element designs](examples/fem/README.md) ·
[49 OpenFOAM cases](examples/cfd/README.md)

The CFD images show coarse reference runs. The heat sink uses an assumed convection
coefficient. Each case guide describes its model and limits.

## Start exploring

- **[Get started](docs/getting_started.md)** — installation, runnable code, and configuration.
- **[Choose a model](docs/use-cases.md)** — match your data and evaluation workflow.
- **[Explore hierarchical fidelities](docs/dag-cokriging.md)** — co-kriging over a directed graph.
- **[Run engineering design examples](examples/README.md)** — structural, thermal and flow designs with FEM and CFD solvers.
- **[Check the evidence](docs/test_function_accuracy.md)** — analytical benchmarks and reproduction settings.
- **[Contribute](docs/getting_started.md#contribute)** — bring a use case, report a bug, or improve the toolkit.

---

<p align="center">
  <img src="docs/assets/gp_omni_icon.svg" alt="GP-Omni orbital GP icon" width="40"><br>
  <strong>GP-Omni</strong> · Open source · <a href="LICENSE">Apache 2.0</a><br>
  <sub>Install as <code>gp-omni</code>. Import as <code>gp_omni</code>.</sub>
</p>
