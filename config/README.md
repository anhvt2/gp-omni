# Configuration

The package reads these JSON defaults when each consuming Python module is imported.
Restart Python after editing defaults. The files are also included in built wheels as
`gp_omni.config`; loading them does not depend on the current working directory.

| File | Settings |
| --- | --- |
| [models.json](models.json) | GPR, GPC, sparse models, AR1/DAG co-kriging, covariance/mean defaults, initial hyperparameters |
| [training.json](training.json) | Fit budgets, learning rates, restarts, optimizer tolerances, inducing-input initialization |
| [inference.json](inference.json) | Laplace iterations, tolerance and line search; sparse approximation and inducing noise |
| [builders.json](builders.json) | GPyTorch builder and additive-kernel defaults |
| [bo.json](bo.json) | Shared BO loop defaults, batch acquisition, CMA-ES search budgets, duplicate tolerance and UCB confidence |
| [async_bo.json](async_bo.json) | Example asynchronous BO run |
| [batch_seq_bo.json](batch_seq_bo.json) | Example batch-sequential BO run |

Sections match the consuming model or operation. `gpr` and `gpc` contain constructor
settings; `gpr_fit` and `gpc_fit` contain `fit()` defaults. The generic sparse model
uses `sparse` and `sparse_fit`. Numerical defaults retain their previous values. Likelihood and covariance selectors
use snake_case names such as `gaussian`, `probit`, `logistic`, and `matern_ard`.

For sparse inference, `inference.json` has a low-level default power `s=1` (FITC). The high-level models explicitly pass their own `s=0` (VFE) default from `models.json`.

Explicit Python arguments override defaults. Load model constructor settings for a run
from a flat JSON object, for example `{"nu": 20, "cov": "matern_ard", "seed": 42}`:

```python
from gp_omni import GPR, GPC

objective = GPR.from_json("my_model.json", degree=5)
constraint = GPC.from_json("my_model.json", lik="logistic")
objective.fit(x, y, max_iter=100)
constraint.fit(x, feasible, max_iter=100)
```

Omitted keys use the corresponding packaged defaults. `sparse_gp.from_json(...)` supports
the same constructor loading pattern and requires `nu` in the file or as an override.
Unknown constructor keys raise `TypeError`.

Load a BO run with `async_bo_config.from_json(...)` or `batch_bo_config.from_json(...)`.
Their files use flat dataclass field names. Python overrides take precedence, including
`known_constraint`, which is a callable and cannot be represented in JSON.

Procedural flags such as `grad`, `diag`, and `observation_noise`, and mathematical
constants defining the algorithms remain in Python.
