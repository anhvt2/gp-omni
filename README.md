# gp-omni

One Python package for the GP features that used to be spread over several toolboxes,
built on **GPyTorch** and **BoTorch** with `uv`.

| Feature | Where |
|---|---|
| **Any combination** of multi-output, multi-fidelity, sparse and additive: one class, four independent switches | `gp_omni.OmniGP`, see [docs/use-cases.md](docs/use-cases.md) |
| Exact GP, gpml-style kernel/mean/likelihood names | `gp_omni.builders.exact_gp`, `gp_omni.compat` |
| Classification (probit, variational) | `gp_omni.builders.SVGP` |
| Correlated multi-output (ICM / LMC) | `gp_omni.builders.MultitaskExactGP` |
| Sparse variational GP, a port of gpml `apxSparse` (FITC / SPEP / VFE, Gaussian and Laplace) | `gp_omni.sparse.SparseGP`, `examples/sparse_gpml.py` |
| Big data via GPyTorch: SGPR, SVGP, KISS-GP | `sgpr`, `SVGP`, `kiss_gp` |
| Additive / high-dimensional (gpml `covADD`) | `gp_omni.kernels.AdditiveKernel`, `additive_gp` |
| Multi-fidelity AR1 co-kriging (port of the owner's MATLAB `@CoKriging`) | `gp_omni.AR1CoKriging` |
| Multi-fidelity via BoTorch | `examples/multifidelity_botorch.py` |

Bayesian optimization (BoTorch acquisitions, GP-Hedge, the async scheduler) is Phase 2 and
not implemented yet. `AR1CoKriging` already exposes a BoTorch `posterior`, so BoTorch
acquisition functions work with it.

```bash
uv sync
uv run pytest
uv run python examples/cokriging.py
```

Importing `gp_omni` sets torch's default dtype to float64.
See `docs/gpml-mapping.md` for MATLAB users, and `tests/fixtures/generate_cokriging.m` for the
optional MATLAB parity check.
