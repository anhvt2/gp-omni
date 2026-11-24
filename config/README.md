# Configuration

[models.json](models.json) supplies constructor, fit, and initialization defaults for
GPR, GPC, SparseGP, AR1/DAG co-kriging, covariance and mean functions.

These JSON resources are included in wheels as `gp_omni.config`, so loading works
independently of the current working directory. Defaults are read at module import;
restart Python after editing them. Explicit Python arguments override defaults.
