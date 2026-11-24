# gp-omni

[![CI](https://github.com/anhvt2/gp-omni/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/anhvt2/gp-omni/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](#install)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Lint: Ruff](https://img.shields.io/badge/lint-Ruff-D7FF64?logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)

**Versatile Gaussian processes for Bayesian optimization. A plug-and-play pipeline from observations to the next experiment.**

gp-omni is an open-source Python toolkit for modeling expensive black-box functions and
choosing where to evaluate them next. Combine correlated outputs, multiple fidelities,
sparse approximations, and additive kernels through one GP interface. Connect your
simulation or experiment to a built-in optimization loop, or use the model components
in your own workflow.

The toolkit brings together PyTorch-based GP inference, GPyTorch model builders, and
BoTorch-compatible models. Released under the [Apache 2.0 license](LICENSE).

## One toolkit, many scenarios

| Your problem | What to plug in | Example |
|---|---|---|
| Several objectives or responses share structure | `GPR(num_outputs=O)` for correlated outputs | [Multi-output surrogate](examples/gpr_multioutput.py) |
| Cheap approximations and expensive experiments | `GPR(num_fidelities=L)` for autoregressive fidelity levels | [Multi-fidelity](examples/gpr_multifidelity.py) |
| Too many observations for a dense GP | `GPR(nu=M)` for inducing-point inference: VFE, FITC, or sparse power EP | [Sparse inference](examples/gpr_sparse.py) |
| Many inputs with low-order interactions | `GPR(additive=R)` for interaction orders `1..R` | [Additive modeling](examples/gpr_additive.py) |
| Unknown feasibility or simulation failures | `GPC` for learned feasibility probabilities, with the same four switches | [Objective + constraint models](examples/gpr_gpc_constraints.py) |
| Evaluations run in parallel and finish together | `batch_bo` for batch-sequential constrained BO | [Batch BO](examples/batch_seq_bo.py) |
| Evaluations finish at different times | `async_bo` for asynchronous constrained BO with GP-Hedge | [Async BO](examples/async_bo.py) |

The four surrogate switches are independent: use any subset, including all four together.
The built-in BO loops currently optimize one scalar objective at one fidelity using default
`GPR` / `GPC` models. Multi-output and multi-fidelity surrogates are available for
custom optimization workflows; Pareto search and cost-aware fidelity selection require
your own acquisition and orchestration logic.

## Install

Requires **Python 3.12+**. From a source checkout:

```bash
git clone https://github.com/anhvt2/gp-omni.git
cd gp-omni
uv sync --locked
```

Or install the local package into an existing environment:

```bash
python -m pip install -e .
```

The distribution is named `gp-omni`; import it as `gp_omni`. Importing the package sets
PyTorch's default dtype to `torch.float64` for GP linear algebra.

## Plug in your objective

Provide a Python callable returning `(objective_value, feasible)`, set bounds and a budget,
and run the pipeline:

```mermaid
flowchart LR
    initial["Evaluate initial designs"] --> fit["Fit objective and feasibility GPs"]
    fit --> propose["Propose experiments"]
    propose --> evaluate["Evaluate and collect results"]
    evaluate --> fit
```

This complete example runs two acquisition jobs concurrently and returns the best feasible
observation. Run it with `uv run python your_script.py` after installation.

```python
import tempfile

import numpy as np

from gp_omni.train.asyncbo import async_bo, async_bo_config


def evaluate(x: np.ndarray) -> tuple[float, bool]:
    # Replace this with your simulator, experiment, or model evaluation.
    value = -float(np.sum((x - 0.3) ** 2))
    feasible = bool(x[0] < 0.9)
    return value, feasible


config = async_bo_config(
    model_name="my_experiment",
    x_lb=[0.0, 0.0],
    x_ub=[1.0, 1.0],
    num_init_point=4,
    max_iter=12,
    exploit_size=2,
    explore_size=0,
    explore_classifier_size=0,
    check_time=0.001,  # polling interval in minutes
    gp_max_iter=60,
    cma_max_evals=60,
    seed=0,
)

with tempfile.TemporaryDirectory() as run_dir:
    best_x, best_y = async_bo(run_dir, config, evaluate=evaluate).run()

print("Best feasible observation:", best_x, best_y)
```

Both BO loops **maximize** the objective; negate a loss to minimize it. The example uses
a temporary directory; use a persistent run directory to retain job inputs and results.

Choose the execution pattern that matches your evaluations:

| Mode | Configuration | Behavior |
|---|---|---|
| Sequential | `async_bo` with `exploit_size=1`, both exploration sizes `0` | One evaluation at a time |
| Batch-sequential | `batch_bo` + `batch_bo_config` | Launch a batch, wait for all results, then propose the next batch |
| Asynchronous | `async_bo` + `async_bo_config` | Refill available slots while other evaluations continue |

`async_bo_config.max_iter` counts total evaluations, including initial points.
`batch_bo_config.max_iter` counts iterations: after `num_init_point` initial evaluations,
each remaining iteration launches a whole batch. Batch size is the sum of `exploit_size`,
`explore_size`, and `explore_classifier_size`.

The loops include PI, EI, and UCB acquisitions, CMA-ES acquisition optimization, objective
and constraint exploration slots, and predicted values for pending points to diversify
proposals. Async BO uses GP-Hedge to select acquisitions; batch BO uses a configured
acquisition (EI by default). Known constraint callbacks and learned feasibility probabilities
weight exploitation proposals; initial and exploration points can still be infeasible.

Load reusable settings with `async_bo_config.from_json(...)` or
`batch_bo_config.from_json(...)`; see [async settings](config/async_bo.json) and
[batch settings](config/batch_seq_bo.json). For external executables, supply a
`query_script` instead of `evaluate`; the job-folder protocol is described in
[`async_bo`](src/train/asyncbo/loop.py).

## Build, train, infer

The source follows the three phases of a GP workflow:

| Package | Responsibility |
| --- | --- |
| [`gp_omni.build`](src/build/) | Prepare models, covariance, means, likelihoods, and inducing inputs for the problem |
| [`gp_omni.train`](src/train/) | Optimize evidence or ELBO objectives and orchestrate Bayesian optimization |
| [`gp_omni.infer`](src/infer/) | Compute posterior distributions, predictive moments, and class probabilities |

```python
from gp_omni.build import GPR
from gp_omni.train import fit
from gp_omni.infer import predict

model = GPR(cov="matern_ard", nu=20)
fit(model, x, y, max_iter=100)
prediction = predict(model, x_new)
```

The model methods `fit()` and `predict()` delegate to these phase implementations.
For GPyTorch models, use `train.gpytorch.fit_exact` or `fit_variational` and
`infer.predict`; supply the separately trained likelihood for variational predictions.
Co-kriging constructors prepare data and topology; call `fit()` explicitly to train
and assemble their posterior before predicting.

## Compose your surrogate

Use the same `fit` / `predict` interface as your modeling needs grow. This example combines
two correlated outputs, two fidelities, sparse inference, and additive interactions:

```python
import torch

from gp_omni import GPR, GPC

torch.manual_seed(0)
x = torch.rand(160, 3)
output = torch.arange(160) % 2
fidelity = (torch.arange(160) // 2) % 2
y = torch.sin(6 * x[:, 0]) + 0.3 * output + 0.2 * fidelity
feasible = x[:, 0] + x[:, 1] < 1.2

settings = dict(num_outputs=2, num_fidelities=2, nu=20, additive=2)
objective = GPR(**settings).fit(x, y, output=output, fidelity=fidelity)
constraint = GPC(**settings).fit(x, feasible, output=output, fidelity=fidelity)

x_next = torch.rand(5, 3)
prediction = objective.predict(x_next)
mean, variance = prediction["fmu"], prediction["fs2"]  # shape (5, 2)
probability_feasible = constraint.predict_proba(x_next)  # shape (5, 2)
```

Keep observations in one table: inputs `x` have shape `(n, D)`, targets `y` have shape
`(n,)`, and optional integer `output` and `fidelity` indices identify each row. Outputs and
fidelities may be observed at different inputs. Fidelity `0` is the cheapest; `L-1` is the
highest. Predictions default to every output at the highest fidelity; pass `output=k` or
`fidelity=l` to select one.

Regression targets are standardized per output/fidelity pair and predictions are returned
in the original units. Scale inputs to comparable ranges, such as the unit cube.
`fmu` / `fs2` describe the latent function; `ymu` / `ys2` include observation noise.
`GPC` accepts boolean, `{0, 1}`, or `{-1, +1}` labels and returns marginal feasibility
probabilities for each constraint output.

See the [documentation](docs/README.md) for model choices, inference options,
data layouts, and model limitations. Run combinations directly:

```bash
uv run python examples/gpr_combined.py mo additive
uv run python examples/gpr_combined.py mf sparse
uv run python examples/gpr_combined.py all
uv run python examples/gpr_gpc_constraints.py all
```

Model, training, inference, and BO defaults live in [`config/`](config/README.md).
Explicit arguments override them; use `GPR.from_json(...)` or `GPC.from_json(...)` for
per-run constructor settings. Defaults are read at module import time.

Models are constructed through `gp_omni.build`, fitted through `gp_omni.train`, and
queried through `gp_omni.infer`. Project-owned helper types use snake_case; likelihood
names are `gaussian`, `probit`, and `logistic`. Covariance names include `se_ard`,
`se_iso`, `matern_ard`, and `matern_iso`. Update older configuration files accordingly.
The old `sparse`, `models`, `builders`, and `fit` import paths have been replaced by the
phase packages. Co-kriging now requires an explicit `fit()` call before prediction.

## Work with GPyTorch and BoTorch

For direct GPyTorch / BoTorch workflows, [`gp_omni.build.gpytorch`](src/build/gpytorch.py) provides
exact, additive, SGPR, KISS-GP, multitask ICM/LMC, and variational models. Use
[`gp_omni.train.gpytorch`](src/train/gpytorch.py) for fitting helpers. The exact GP builders, `ar1_cokriging`, and `dag_cokriging`
expose BoTorch-compatible posteriors; `GPR` / `GPC` expose their own prediction
interface.

- [Exact GP](examples/exact_gpr.py) · [Correlated outputs](examples/multioutput.py)
- [SGPR, svgp, and KISS-GP](examples/sparse.py) · [Variational classification](examples/classification.py)
- [AR1 co-kriging](examples/cokriging.py) · [BoTorch multi-fidelity](examples/multifidelity_botorch.py)
- [Hierarchical DAG co-kriging](examples/dag_cokriging.py) · [DAG model guide](docs/dag-cokriging.md)
- [Sparse variational inference (VFE / FITC)](examples/sparse_vfe.py)

## Contribute

Bug reports, new use cases, examples, and pull requests are welcome. To check a local change:

```bash
uv sync --locked
uv run ruff check .
uv run pytest -q
```

The test suite covers all 16 combinations of the four surrogate switches for regression
and classification, plus kernels, sparse inference, co-kriging, and the BO loops.
