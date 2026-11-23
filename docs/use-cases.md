# Setting up a use case

`OmniGP` has four switches. They are independent: turn on any subset, in any combination.

| Feature | Switch | Example |
|---|---|---|
| 1. Correlated multi-objective | `num_outputs=O` | [`examples/omni_multiobjective.py`](../examples/omni_multiobjective.py) |
| 2. Multi-fidelity | `num_fidelities=L` | [`examples/omni_multifidelity.py`](../examples/omni_multifidelity.py) |
| 3. Big data, sparse covariance | `nu=<inducing points>` (`s=0` VFE, `1` FITC) | [`examples/omni_bigdata.py`](../examples/omni_bigdata.py) |
| 4. Additive, high-dimensional | `additive=R` (orders `1..R`) | [`examples/omni_additive.py`](../examples/omni_additive.py) |
| 1+2, 1+2+3, 1+2+3+4 | any mix of the above | [`examples/omni_combined.py`](../examples/omni_combined.py) |

```python
from gp_omni import OmniGP

model = OmniGP(num_outputs=2, num_fidelities=2, nu=40, additive=2)  # all four
model.fit(x, y, output=o, fidelity=f)  # x (n, D), y (n,), o and f: integer index per row
pred = model.predict(x_new)  # highest fidelity; every output when O > 1
pred["fmu"], pred["fs2"]  # latent mean / variance, shape (m,) or (m, O)
pred["ymu"], pred["ys2"]  # with observation noise
```

Run any example with `uv run python examples/omni_multiobjective.py`. Each prints a few
numbers and compares against a baseline.

## Data layout

Always one table: stack every observation as a row, then say which output and fidelity it
belongs to. Rows of different outputs or fidelities may sit at different inputs and need not
be nested.

| Argument | Meaning |
|---|---|
| `x` | inputs, shape `(n, D)` |
| `y` | targets, shape `(n,)` |
| `output` | integer in `0..O-1` per row (omit when `O = 1`) |
| `fidelity` | integer in `0..L-1` per row, `0` cheapest and `L-1` the one you want (omit when `L = 1`) |

Targets are standardised per `(output, fidelity)` before fitting and mapped back afterwards,
so you do not need to rescale them. Scale the inputs to a comparable range (e.g. the unit
cube); the default length scales start at the standard deviation of each input.

## 1. Correlated multi-objective

Use when several objectives share structure and some have little data. Outputs are
coregionalised, `B = W W' + diag(kappa^2)` (`rank=` sets the rank of `W`, default 1), so
the data-rich objectives sharpen the data-poor ones. In
[`omni_multiobjective.py`](../examples/omni_multiobjective.py) an output with only 8
samples reaches 0.03 normalised error jointly, against 0.75 alone.
`predict(x)` returns all outputs as columns; `predict(x, output=k)` returns one.

## 2. Multi-fidelity

Use for cheap low-fidelity data plus a few expensive points. Fidelities are autoregressive,
`f_l = rho_l * f_{l-1} + delta_l`, with one `rho` per output and any number of levels
(not only two). [`omni_multifidelity.py`](../examples/omni_multifidelity.py) uses 30
cheap and 4 expensive points. `predict(x, fidelity=l)` chooses the level, default the
highest.

## 3. Big data with a sparse covariance

Use when `n` is too large for a dense `n x n` matrix. `nu` inducing locations replace the
data in the covariance (cost `O(n nu^2)`); [`omni_bigdata.py`](../examples/omni_bigdata.py)
fits 20,000 points with 40. The inducing locations are learned unless `learn_xu=False`, and
start from randomised k-means centres (`init="kmeans"`, or `"random"`). `s=0` is the
variational free energy bound and recovers the noise level (0.0998 vs 0.1 in the example);
`s=1` is FITC; values in between are sparse power EP. With several tasks, each location is
shared by every `(output, fidelity)`, so `nu` locations give `nu x O x L` inducing inputs.

## 4. Additive GP for high-dimensional regression

Use when the response is a sum of low-order effects of a few inputs. `additive=R` uses
orders `1..R` (`additive=1` purely additive; a list such as `[1, 3]` picks orders). It has
one length scale per input and one amplitude per order.
[`omni_additive.py`](../examples/omni_additive.py) recovers a 20-D additive function from
120 points (0.013 normalised error, against 0.39 for a plain ARD kernel). `R` cannot exceed
`D`, and cost grows with `D * R`.

## Combining

Nothing needs to change in the data layout: pass what the features you turn on need.
[`omni_combined.py`](../examples/omni_combined.py) lets you pick the features on the command
line (`mo`, `mf`, `sparse`, `additive`, any subset), for example a high-dimensional
multi-objective model:

```bash
uv run python examples/omni_combined.py mo additive
uv run python examples/omni_combined.py mf sparse
uv run python examples/omni_combined.py mo mf sparse additive    # or: all
uv run python examples/omni_combined.py                          # 1+2, then +3, then +4
```

It generates data to match the features you pick and prints the error at the highest
fidelity. `tests/test_examples.py` runs all 16 subsets through this script, and
`tests/test_omni.py` runs all 16
combinations (including none) and also checks that each feature earns its keep: each of
multi-objective, multi-fidelity and additive cuts the error to under half of its baseline, sparse
stays close to exact, and a 15,000-point fit with all four on takes seconds.

Why this works: each switch changes one ingredient. Multi-output and multi-fidelity change
the covariance (`gp_omni.sparse.multitask.TaskCov`), additive changes the base kernel
(`covADD`), and sparse changes how that covariance is approximated (gpml `apxSparse`). A
sparse approximation only needs a covariance function, so it applies to any of the others.

## Choosing between the classes

| You want | Use |
|---|---|
| any mix of the four features | `OmniGP` |
| gpml-style single-task sparse GP, with `vfe_xu_opt` swaps | `gp_omni.sparse.SparseGP` |
| two fidelities, noiseless, small design, the ooDACE fitting procedure | `gp_omni.AR1CoKriging` |
| GPyTorch/BoTorch models directly (SVGP, KISS-GP, ...) | `gp_omni.builders`, `examples/sparse.py` |

On the tiny noiseless Forrester problem, `AR1CoKriging` (sequential, profiled fit) is far
more accurate than `OmniGP` (joint marginal likelihood with learned noise): RMSE 0.13
against 1.7, with 3.7 for high-fidelity only. Prefer `AR1CoKriging` in that regime.

## Limits

- Multi-output and multi-fidelity models use Gaussian noise (one level per task).
  Classification (`lik="likErf"` / `"likLogistic"`, Laplace approximation) works with
  `nu` and `additive`, for one task only.
- Several outputs are predicted as marginals; the covariance between outputs at a test
  point is not returned.
- `vfe_xu_opt` inducing-input swaps are only available through `SparseGP`.
- The structured multi-task covariance reproduces `AR1CoKriging` at fixed hyperparameters
  (tested). `AR1CoKriging` itself has not been compared against output from MATLAB ooDACE
  (see `tests/fixtures/generate_cokriging.m`).
- Bayesian optimization (acquisitions, GP-Hedge, async scheduler) is not built yet.
