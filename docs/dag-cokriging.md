# Hierarchical DAG co-kriging

`dag_cokriging` models noiseless observations at any number of fidelity nodes linked
by a supplied directed acyclic graph. Nodes may have multiple parents, and designs
may differ across nodes. Each node needs nonempty observations in the same input
dimension. The graph is supplied by the user; fitting learns weights, not topology.

```python
from gp_omni import dag_cokriging

# X[j]: (n_j, d), Y[j]: (n_j,) or (n_j, 1), indexed by node j.
# Diamond: 0 feeds 1 and 2, which both feed 3.
edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
model = dag_cokriging(X, Y, edges, regression="regpoly0", num_restarts=4)
model.fit()
mean, variance = model.predict(x_new)  # unique sink: node 3
parent_mean, parent_variance = model.predict(x_new, fidelity=1)
posterior = model.posterior(x_new)  # BoTorch-compatible, with input gradients
B = model.coupling_matrix  # B[parent, child], zero for every absent edge
```

Run the complete example with `uv run python examples/dag_cokriging.py`.
Node identifiers are list indices, not fidelity ranks. Edges and nodes need not be
in topological order. If several nodes are sinks (including disconnected nodes),
supply `target=j` explicitly. Prediction at any node conditions on **all** node data.
A graph with one node and no edges gives ordinary universal kriging.

## Covariance construction

At each node, the process is a weighted sum of its parents plus an independent
Gaussian process discrepancy:

$$f_j(x)=\sum_{i\in\operatorname{parents}(j)}B_{ij}f_i(x)+\delta_j(x).$$

Let $k_q$ be the discrepancy covariance at node $q$, including its variance, and
$A=(I-B^\mathsf{T})^{-1}$. Then

$$K_{ij}(x,x')=\sum_q A_{iq} A_{jq} k_q(x,x').$$

The acyclic graph permits computing $A$ by propagation in topological order.
Each entry sums products along **all** paths from an ancestor to a descendant,
including the identity contribution at each node. Shared ancestors are accounted
for once per discrepancy, including in diamond-shaped graphs.

Missing edges fix entries of **B** to zero. They do not force covariance blocks
to zero: indirect paths and shared ancestors can induce dependence. Completely
disconnected components have zero cross-covariance. Direct weights may be zero
or negative; learned weights default to bounds `(-5, 5)`.

## Fitting and fixed parameters

All nodes share a correlation family (`gauss`, `exp`, `matern32`, or `matern52`)
and regression degree (`none`, `regpoly0`, `regpoly1`, or `regpoly2`). Each node has
independent per-dimension log10 precisions and a positive discrepancy variance;
each edge has one constant weight. Scale inputs to comparable ranges, such as the
unit cube. Outputs are used in their original units.

Sequential parent predictions initialize discrepancy scales. Bounded, multistart
joint maximum likelihood then fits all kernel precisions, edge weights, and
variances against the assembled training covariance. Trend coefficients are
profiled out by generalized least squares; their uncertainty contributes to the
posterior variance. Each node must have a full-rank regression basis with more
observations than basis columns.

`hp_bounds` bounds log10 precisions. `log_variance_bounds` bounds log10 variance
multipliers relative to the initialized per-node scales, defaulting to `(-6, 6)`.
`rho_bounds` bounds every edge weight. Set `num_restarts=0` to optimize only the
initial starting point; use `seed` for repeatable random restarts.

Supply all hyperparameters to skip fitting:

```python
fixed = model.hyperparameters
restored = dag_cokriging(X, Y, edges, regression="regpoly0", hyperparameters=fixed).fit()
```

The dictionary contains `theta` with shape `(L, d)` (also accepts `(L, 1)` for
isotropic kernels), `rho` with shape `(E,)` in the supplied edge order, and
`sigma2` with shape `(L,)`. Variances must be positive and are honored exactly.
Keep graph ordering, correlation family, regression, and target configuration
alongside this dictionary when restoring a model.

## Scope and compatibility

The model uses dense exact inference: O(N³) time and O(N²) memory for N total
observations. Highly correlated parents can make edge weights difficult to
identify; predictive accuracy does not guarantee recovery of physical weights.
Fit uses CPU float64. Registered training data and fitted buffers support moving
the fitted model to another device with `.to(...)`; float64 is recommended.

`predict` returns marginal means and variances of shape `(..., q)`.
`posterior` returns one selected node with mean shape `(..., q, 1)`, full candidate
covariance, and input gradients for BoTorch acquisition optimization. It supports
posterior transforms and `output_indices=[0]`. Observation noise is unsupported;
small relative diagonal jitter is numerical stabilization only. Selecting the
next fidelity or optimizing jointly over fidelities requires a custom workflow.

The existing `ar1_cokriging` retains its two-level interface, sequential fitting,
and MATLAB parity behavior. With the same two-node kernel parameters, edge
weight, and discrepancy variances, the new model gives the same high-fidelity
predictions up to numerical jitter. Joint fitting can produce different fitted
parameters. Lower-node predictions in the DAG model incorporate descendant data;
the legacy low-fidelity prediction uses low-fidelity data alone.
