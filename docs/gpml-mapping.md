# gpml → gp-omni → GPyTorch/BoTorch

Covers every gpml function used by `ooDaceAsyncParHedge.GitHub/mainprog*.m` and
`bayesOptSrc-gpml/`, plus the features gp-omni targets.

| gpml | gp-omni | Underlying | Notes |
|---|---|---|---|
| `covSEard` | `gpml_kernel("covSEard", d)` | `ScaleKernel(RBFKernel(ard_num_dims=d))` | gpml `sf` ↔ `sqrt(outputscale)`, `ell` ↔ `lengthscale` |
| `covSEiso` | `gpml_kernel("covSEiso", d)` | `ScaleKernel(RBFKernel())` | |
| `covMaterniso`, `{@covMaterniso, d}` | `gpml_kernel(("covMaterniso", 3), d)` | `ScaleKernel(MaternKernel(nu=d/2))` | gpml `d ∈ {1,3,5}` ↔ `nu = d/2` |
| `covMaternard` | `gpml_kernel(("covMaternard", 5), d)` | `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=d))` | |
| `covRQiso/ard`, `covPERiso/ard`, `covLINiso/ard`, `covPoly` | `gpml_kernel(...)` | `RQKernel`, `PeriodicKernel`, `LinearKernel`, `PolynomialKernel` | LIN is not ARD here |
| `covSum`, `covProd` | `gpml_kernel(("covSum", [...]), d)` | `+`, `*` on kernels | |
| `covADD` (orders `1:R`) | `gp_omni.kernels.AdditiveKernel(d, R)` / `additive_gp` | Newton–Girard recursion | shared length scale like gpml; `ard=True` for per-dim |
| `meanConst` | `gpml_mean("meanConst", d)` | `ConstantMean` | |
| `{@meanSum,{@meanLinear,@meanConst}}` | `gpml_mean("meanSum", d)` | `LinearMean(d, bias=True)` | |
| `likGauss` + `infGaussLik` | `exact_gp(...)` | `ExactGP` + `GaussianLikelihood` | exact inference |
| `likErf` + `infEP` | `SVGP` + `BernoulliLikelihood` + `fit_variational` | variational (probit link) | **No EP/Laplace in GPyTorch.** Predictive probabilities differ slightly from EP |
| `minimize(hyp, @gp, -n, ...)` | `fit_exact(model, num_restarts=k)` | `fit_gpytorch_mll` (L-BFGS-B) | gpml optimises log-hyperparameters; GPyTorch uses softplus |
| `apxSparse` (FITC/VFE/SPEP), `infGaussLik`, `infLaplace`, `hyp.xu`, `hyp.snu`, `vfe_xu_opt` | `gp_omni.sparse` (`SparseGP`, `SparseCov`, `apx`, `inf_gauss_lik`, `inf_laplace`, `vfe_xu_opt`) | own port of gpml v4.2, autograd derivatives | `opt.s` = 1 FITC, 0 VFE, between SPEP. Matches gpml to ~1e-7 (see `tests/test_sparse_gpml.py`). Not ported: `infEP`/`infKL`/`infVB`, priors (`infPrior`), LU mode for negative W, `covPP`, `apxGrid` |
| `apxSparse`, GPyTorch route | `sgpr(...)` (VFE), `SVGP` | `InducingPointKernel`, `VariationalStrategy` | GPyTorch alternatives; no FITC/SPEP |
| `apxGrid` / `infGrid` | `kiss_gp(...)` | `GridInterpolationKernel` | low input dimension only |
| ooDACE `@CoKriging` | `gp_omni.AR1CoKriging` | owner's port | two fidelities, see class docstring |
| GPflow ICM/LMC | `MultitaskExactGP(kind="icm"/"lmc")` | `MultitaskKernel`, `LCMKernel` | |

gpml's `hyp` vectors store `log` of positive hyperparameters. GPyTorch stores unconstrained
raw values behind softplus, so hyperparameters cannot be copied across numerically
without converting.
