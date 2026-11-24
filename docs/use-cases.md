# Choose a model and workflow

## Model switches

`OmniGP` predicts objective values. Combine any of these four options:

| Need | Option | Example |
|---|---|---|
| Correlated responses or objectives | `num_outputs=O` | [Multiple outputs](../examples/omni_multiobjective.py) |
| Cheap and expensive evaluations | `num_fidelities=L` | [Multiple fidelities](../examples/omni_multifidelity.py) |
| Larger datasets | `nu=M` inducing locations | [Sparse GP](../examples/omni_bigdata.py) |
| Low-order effects across many inputs | `additive=R` for orders `1..R` | [Additive GP](../examples/omni_additive.py) |

For sparse inference, `s=0` selects VFE (default), `s=1` selects FITC, and values between
select sparse power EP. Inducing locations are learned unless `fit(..., learn_xu=False)`.
With multiple outputs or fidelities, `M` locations expand to `M * O * L` inducing inputs.

For additive models, `R` must not exceed the input dimension. Use a list such as
`additive=[1, 3]` to select specific interaction orders.

```python
from gp_omni import OmniGP

# x, y, output, and fidelity are your observation tensors; see the table below.
model = OmniGP(num_outputs=2, num_fidelities=2, nu=40, additive=2)
model.fit(x, y, output=output, fidelity=fidelity)
prediction = model.predict(x_new)
mean, variance = prediction["fmu"], prediction["fs2"]
```

## Data and predictions

Store each observation as one row. Different outputs and fidelities can use different inputs.

| Tensor | Shape | Meaning |
|---|---|---|
| `x` | `(n, D)` | Input values; scale to comparable ranges, such as `[0, 1]` |
| `y` | `(n,)` | Observed target values |
| `output` | `(n,)` | Integer output index, `0..O-1`; omit for one output |
| `fidelity` | `(n,)` | Integer fidelity index, `0..L-1`; omit for one fidelity |

Fidelity `0` is cheapest; `L-1` is highest. Targets are standardized per output/fidelity
pair during fitting and predictions return to the original units.

`predict(x_new)` returns every output at the highest fidelity. Pass `output=k` or
`fidelity=l` to select one. Results have shape `(m,)` for one output or `(m, O)` for all
outputs. `fmu` / `fs2` are latent mean / variance; `ymu` / `ys2` include observation noise.
Cross-output predictive covariances are not returned.

## Unknown constraints

`OmniGPC` uses the same four model switches to learn feasibility. Labels can be boolean,
`0/1`, or `-1/+1`; positive labels mean feasible.

```python
from gp_omni import OmniGPC

constraint = OmniGPC().fit(x, feasible)
probability = constraint.predict_proba(x_new)
decision = constraint.predict_feasible(x_new)  # probability >= 0.5
```

Classification uses a probit likelihood by default, or `lik="likLogistic"`, with a Laplace
approximation. With multiple constraints, predictions are marginal probabilities per output.
See the [combined objective and constraint example](../examples/omni_constraints.py).

## Bayesian optimization

Supply `evaluate(x) -> (objective_value, feasible)`. Both built-in loops maximize the
objective; negate a loss to minimize it. The [quickstart](../README.md#plug-in-your-objective)
contains a complete runnable pipeline.

| Execution mode | Use | Budget |
|---|---|---|
| Sequential | `AsyncBO`, one exploitation slot and zero exploration slots | `max_iter` total evaluations |
| Batch-sequential | `BatchSeqBO` | Initial evaluations, then one batch per remaining iteration |
| Asynchronous | `AsyncBO`, multiple slots | `max_iter` total evaluations |

Batch size is `exploit_size + explore_size + explore_classifier_size`. For batch BO,
total evaluations are `num_init_point + (max_iter - num_init_point) * batch_size`.

Async BO selects PI, EI, or UCB through GP-Hedge. Batch BO uses the configured acquisition
(EI by default). Known constraint callbacks and learned feasibility weight exploitation
proposals; initial and exploration points can still be infeasible.

The built-in loops use a scalar, single-fidelity objective with default models. Multi-output
optimization, fidelity selection, and other surrogate configurations need a custom workflow.
Settings and runnable examples: [async BO](../examples/async_bo.py),
[batch BO](../examples/batch_seq_bo.py), [async config](../config/async_bo.json),
[batch config](../config/batch_seq_bo.json).

## Other model choices

| Need | Use |
|---|---|
| Small, noiseless, two-fidelity designs | [`AR1CoKriging`](../examples/cokriging.py) |
| Noiseless fidelities linked by a DAG, including multiple parents | [`DAGCoKriging`](dag-cokriging.md) |
| Direct GPyTorch models: exact, SGPR, SVGP, KISS-GP, or ICM/LMC | [`gp_omni.builders`](../src/builders.py) |
| GPML-style sparse inference and inducing-input swaps | [`SparseGP`](../examples/sparse_gpml.py) |

The exact GP builders, `AR1CoKriging`, and `DAGCoKriging` expose BoTorch-compatible posteriors.
`OmniGP` and `OmniGPC` use the prediction interface above.

## Related Notes

- [Documentation index](README.md) — guide overview.
- [Project README](../README.md) — installation, runnable pipeline, and contributions.
