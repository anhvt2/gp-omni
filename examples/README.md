# Engineering design optimization

This example runs GP-Omni's constrained batch Bayesian optimization against a physics solver.
Run the commands from the repository root and give each run a new output directory.
The [numerical scripts](../tests/numerical/) and [analytical benchmarks](../tests/bo_functions/)
live under `tests/`.

## FEM · Tapered aluminum cantilever

Minimize the mass of a 600 mm cantilever with a rectangular cross section and a 1 kN tip load.
The 45 mm width stays fixed. The optimizer varies the root height and the ratio of tip height
to root height. Each evaluation uses scikit-fem with cubic Hermite beam elements to solve
the linear Euler–Bernoulli model and checks a 1.5 mm tip deflection limit.
[scikit-fem beam element example](https://scikit-fem.readthedocs.io/en/stable/listofexamples.html#example-34-euler-bernoulli)

```bash
uv run --locked --with 'scikit-fem==12.0.2' python examples/fem/cantilever.py cantilever-run
```

The script reports the best observed feasible dimensions, mass, and tip deflection. The output
directory contains the BO observations and feasibility flags. The model assumes an ideal clamp
and small elastic deflections and sizes the beam for stiffness.
