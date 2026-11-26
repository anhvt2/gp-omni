"""Fit a four-fidelity diamond DAG on nonnested one-dimensional designs.

Run: python tests/numerical/dag_cokriging.py
"""

import torch

from gp_omni import dag_cokriging


def main() -> dict:
    def responses(x):
        root = torch.sin(6 * x)
        left = 0.8 * root + 0.3 * torch.cos(10 * x)
        right = 1.2 * root + x**2
        target = 0.6 * left + 0.5 * right + 0.1 * torch.sin(3 * x)
        return root, left, right, target

    generator = torch.Generator().manual_seed(5)
    X = [torch.rand(n, 1, generator=generator) for n in (25, 15, 13, 7)]
    Y = [responses(x[:, 0])[i] for i, x in enumerate(X)]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    model = dag_cokriging(X, Y, edges, regression="regpoly0", num_restarts=1, seed=4).fit()
    grid = torch.linspace(0, 1, 101)[:, None]
    mean, variance = model.predict(grid)  # unique terminal node 3
    parent_mean, _ = model.predict(grid, fidelity=1)  # conditioned on all node data
    posterior = model.posterior(grid[:4])  # differentiable BoTorch posterior
    return {
        "rmse": float(((mean - responses(grid[:, 0])[3]) ** 2).mean().sqrt()),
        "parent_rmse": float(((parent_mean - responses(grid[:, 0])[1]) ** 2).mean().sqrt()),
        "min_variance": float(variance.min()),
        "posterior_shape": tuple(posterior.mean.shape),
        "edge_weights": model.rho.tolist(),
    }


if __name__ == "__main__":
    print(main())
