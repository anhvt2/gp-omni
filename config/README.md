# Configuration

[models.json](models.json) supplies constructor, fit, and initialization defaults for
GPR, GPC, SparseGP, AR1/DAG co-kriging, covariance and mean functions.

These JSON resources are included in wheels as `gp_omni.config`, so loading works
independently of the current working directory. Defaults are read at module import;
restart Python after editing them. Explicit Python arguments override defaults.

[training.json](training.json) supplies fit budgets, learning rates, restarts, optimizer
tolerances and inducing initialization. [inference.json](inference.json) supplies Laplace
iterations, tolerance and line search, and sparse inducing noise. Low-level sparse
inference defaults to FITC (`s=1`); high-level models explicitly pass VFE (`s=0`).

[builders.json](builders.json) supplies GPyTorch builder and additive-kernel defaults.
