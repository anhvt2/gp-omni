"""DAG covariance, joint conditioning, fitting, and acquisition integration."""

import numpy as np
import pytest
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.optim import optimize_acqf

from gp_omni import AR1CoKriging, DAGCoKriging


def data(nodes=5):
    rng = np.random.default_rng(12)
    X = [torch.tensor(rng.uniform(0, 1, (7 + i, 1))) for i in range(nodes)]
    Y = [torch.sin(6 * x[:, 0]) + i * x[:, 0] ** 2 for i, x in enumerate(X)]
    return X, Y


def fixed(nodes, edges):
    return {
        "theta": torch.linspace(-0.4, 0.6, nodes)[:, None],
        "sigma2": torch.linspace(0.5, 1.5, nodes),
        "rho": torch.linspace(0.7, 1.2, len(edges)),
    }


def dense_reference(model, x, node):
    """Independent path expansion and NumPy GP conditioning with Matérn 3/2."""
    n = model.num_nodes
    hp = {k: v.numpy() for k, v in model.hyperparameters.items()}
    B = np.zeros((n, n))
    for (i, j), weight in zip(model.edges, hp["rho"]):
        B[j, i] = weight
    A = sum(np.linalg.matrix_power(B, k) for k in range(n))

    def cov(i, j, a, b):
        result = np.zeros((len(a), len(b)))
        for q in range(n):
            distance = np.sqrt(((a[:, None] - b[None, :]) ** 2 * 10 ** hp["theta"][q]).sum(-1))
            t = np.sqrt(3) * distance
            result += A[i, q] * A[j, q] * hp["sigma2"][q] * (1 + t) * np.exp(-t)
        return result

    def basis(i, x):
        if model.regression == "none":
            return np.zeros((len(x), 0))
        F = np.column_stack([np.ones(len(x)), x])
        return np.concatenate([A[i, q] * F for q in range(n)], axis=1)

    X = [v.numpy() for v in model.X]
    S = np.block([[cov(i, j, a, b) for j, b in enumerate(X)] for i, a in enumerate(X)])
    F = np.concatenate([basis(i, a) for i, a in enumerate(X)])
    y = np.concatenate([v.numpy() for v in model.Y])
    cross = np.concatenate([cov(node, j, x, b) for j, b in enumerate(X)], axis=1)
    SiF = np.linalg.solve(S, F)
    beta = np.linalg.solve(F.T @ SiF, SiF.T @ y) if F.shape[1] else np.empty(0)
    mean = basis(node, x) @ beta + cross @ np.linalg.solve(S, y - F @ beta)
    posterior = cov(node, node, x, x) - cross @ np.linalg.solve(S, cross.T)
    if F.shape[1]:
        U = basis(node, x) - cross @ SiF
        posterior += U @ np.linalg.solve(F.T @ SiF, U.T)
    return A, mean, posterior


@pytest.mark.parametrize("regression", ["none", "regpoly1"])
@pytest.mark.parametrize(
    "edges,target",
    [
        ([(0, 1), (1, 2), (2, 3), (3, 4)], 4),
        ([(3, 1), (3, 2), (1, 0), (2, 0)], 0),  # diamond plus isolated node
        ([], 2),
    ],
)
def test_dense_reference_and_interpolation(regression, edges, target):
    X, Y = data()
    m = DAGCoKriging(X, Y, edges, target, regression=regression, hyperparameters=fixed(5, edges))
    x = np.linspace(0.03, 0.97, 6)[:, None]
    for node in range(5):
        A, mean, cov = dense_reference(m, x, node)
        np.testing.assert_allclose(m.loadings.numpy(), A)
        post = m.posterior(torch.tensor(x), fidelity=node)
        np.testing.assert_allclose(post.mean[:, 0].detach(), mean, atol=2e-6)
        np.testing.assert_allclose(post.distribution.covariance_matrix.detach(), cov, atol=2e-7)
        pred, var = m.predict(X[node], fidelity=node)
        torch.testing.assert_close(pred, Y[node], atol=2e-6, rtol=2e-6)
        assert var.max() < 1e-7


def test_missing_edge_allows_indirect_covariance_and_shared_ancestors():
    X, Y = data(4)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    m = DAGCoKriging(X, Y, edges, hyperparameters=fixed(4, edges))
    A = m.loadings
    assert A[3, 0] == m.rho[0] * m.rho[2] + m.rho[1] * m.rho[3]
    assert m._covariance(1, 2, X[1], X[2], A, m.theta, m.variances).abs().max() > 0
    assert m._covariance(0, 3, X[0], X[3], A, m.theta, m.variances).abs().max() > 0


def test_zero_edge_and_disconnected_nodes_do_not_transfer_data():
    X, Y = data(3)
    edges = [(0, 1)]
    hp = fixed(3, edges)
    hp["rho"].zero_()
    a = DAGCoKriging(X, Y, edges, target=1, hyperparameters=hp)
    b = DAGCoKriging(X, [Y[0] + 100, Y[1], Y[2] - 100], edges, target=1, hyperparameters=hp)
    x = torch.linspace(0, 1, 10)[:, None]
    for va, vb in zip(a.predict(x), b.predict(x)):
        torch.testing.assert_close(va, vb)


@pytest.mark.parametrize("kind", ["none", "regpoly0", "regpoly1"])
def test_two_node_predictions_match_legacy_with_same_variances(kind):
    X, Y = data(2)
    old = AR1CoKriging(
        X, Y, regression=kind, hyperparameters={"theta1": [0.2], "theta2": [0.4], "rho": 1.3}
    )
    hp = {"theta": [[0.2], [0.4]], "rho": [1.3], "sigma2": [old.level1.sigma2, old.level2.sigma2]}
    new = DAGCoKriging(X, Y, [(0, 1)], regression=kind, hyperparameters=hp)
    x = torch.linspace(0.01, 0.99, 20)[:, None]
    for actual, expected in zip(new.predict(x), old.predict(x)):
        torch.testing.assert_close(actual, expected, atol=2e-7, rtol=2e-5)


def test_batches_gradients_and_acquisition_optimization():
    X, Y = data(3)
    edges = [(0, 1), (1, 2)]
    m = DAGCoKriging(X, Y, edges, regression="regpoly1", hyperparameters=fixed(3, edges))
    x = torch.tensor([[[0.11], [0.45]], [[0.67], [0.92]]], requires_grad=True)
    post = m.posterior(x)
    mean, var = m.predict(x)
    torch.testing.assert_close(post.mean[..., 0], mean)
    torch.testing.assert_close(post.variance[..., 0], var, atol=1e-8, rtol=1e-5)
    assert torch.linalg.eigvalsh(post.distribution.covariance_matrix).min() > 0
    assert torch.autograd.gradcheck(lambda z: m.posterior(z).mean, x)
    assert torch.autograd.gradcheck(lambda z: m.posterior(z).variance, x)
    acq = LogExpectedImprovement(m, best_f=Y[-1].max())
    candidate, value = optimize_acqf(
        acq,
        torch.tensor([[0.0], [1.0]]),
        q=1,
        num_restarts=2,
        raw_samples=8,
        options={"maxiter": 10},
    )
    assert candidate.shape == (1, 1) and torch.isfinite(value)


def test_joint_fit_is_finite_and_interpolates():
    X, Y = data(3)
    m = DAGCoKriging(X, Y, [(0, 1), (0, 2), (1, 2)], regression="regpoly0", num_restarts=0)
    assert all(torch.isfinite(t).all() for t in m.hyperparameters.values())
    for i in range(3):
        mean, var = m.predict(X[i], fidelity=i)
        torch.testing.assert_close(mean, Y[i], atol=1e-5, rtol=1e-5)
        assert var.min() >= 0
    restored = DAGCoKriging(
        X, Y, list(m.edges), regression="regpoly0", hyperparameters=m.hyperparameters
    )
    torch.testing.assert_close(restored.predict(X[0])[0], m.predict(X[0])[0])


def test_single_node_and_buffer_conversion():
    X, Y = data(1)
    m = DAGCoKriging(X, Y, [], hyperparameters=fixed(1, []))
    assert m.target == 0
    assert "_X0" in m.state_dict() and "_L" in m.state_dict()
    m.to(dtype=torch.float32)
    assert m.predict(X[0])[0].dtype == torch.float32


@pytest.mark.parametrize(
    "edges,target",
    [
        ([(0, 0)], 0),
        ([(0, 1), (1, 0)], 1),
        ([(0, 2)], 1),
        ([(0, 1), (0, 1)], 1),
        ([(0.0, 1)], 1),
        ([], None),
        ([], -1),
    ],
)
def test_rejects_invalid_graph(edges, target):
    X, Y = data(2)
    with pytest.raises(ValueError):
        DAGCoKriging(X, Y, edges, target=target)


@pytest.mark.parametrize(
    "key,value",
    [
        ("theta", [[1, 2], [3, 4]]),
        ("rho", [1, 2]),
        ("sigma2", [0, 1]),
        ("theta", [[float("nan")], [1]]),
        ("sigma2", [1, float("inf")]),
    ],
)
def test_rejects_invalid_hyperparameters(key, value):
    X, Y = data(2)
    hp = fixed(2, [(0, 1)])
    hp[key] = value
    with pytest.raises(ValueError):
        DAGCoKriging(X, Y, [(0, 1)], hyperparameters=hp)


def test_rejects_invalid_data_and_unsupported_posterior_options():
    X, Y = data(1)
    with pytest.raises(ValueError):
        DAGCoKriging([], [], [])
    with pytest.raises(ValueError):
        DAGCoKriging(X, [Y[0][None]], [])
    with pytest.raises(ValueError):
        DAGCoKriging([torch.ones(3, 1)], [torch.ones(3)], [], regression="regpoly1")
    with pytest.raises(ValueError):
        DAGCoKriging(X, Y, [], hp_bounds=(2, -2))
    m = DAGCoKriging(X, Y, [], hyperparameters=fixed(1, []))
    with pytest.raises(ValueError):
        m.predict(X[0], fidelity=2)
    with pytest.raises(ValueError):
        m.posterior(X[0], output_indices=[1])
    with pytest.raises(NotImplementedError):
        m.posterior(X[0], observation_noise=True)


@pytest.mark.parametrize("name", ["gauss", "exp", "matern32", "matern52"])
def test_multidimensional_kernels_signed_edges_and_transform(name):
    from botorch.acquisition.objective import ScalarizedPosteriorTransform

    rng = torch.Generator().manual_seed(19)
    X = [torch.rand(9, 2, generator=rng) for _ in range(3)]
    Y = [torch.sin(x[:, 0] * 5) + x[:, 1] for x in X]
    hp = {"theta": [[0.3], [0.1], [0.7]], "rho": [-0.6, 1.3], "sigma2": [1, 0.5, 0.2]}
    m = DAGCoKriging(X, Y, [(0, 1), (1, 2)], correlation=name, hyperparameters=hp)
    B = m.coupling_matrix
    assert B[0, 2] == 0 and B[0, 1] == -0.6
    torch.testing.assert_close(m.loadings, torch.linalg.inv(torch.eye(3) - B.T))
    for node in range(3):
        torch.testing.assert_close(m.predict(X[node], fidelity=node)[0], Y[node])
    x = torch.rand(4, 2, generator=rng)
    transform = ScalarizedPosteriorTransform(weights=torch.tensor([2.0]), offset=1.0)
    post = m.posterior(x, posterior_transform=transform, output_indices=[0])
    base = m.posterior(x)
    torch.testing.assert_close(post.mean, base.mean * 2 + 1)
    torch.testing.assert_close(post.variance, base.variance * 4)


def test_joint_likelihood_parameter_gradients():
    X, Y = data(3)
    edges = [(0, 1), (0, 2), (1, 2)]
    m = DAGCoKriging(X, Y, edges, regression="regpoly0", hyperparameters=fixed(3, edges))
    theta = m.theta.clone().requires_grad_()
    rho = m.rho.clone().requires_grad_()
    log_s = m.variances.log().requires_grad_()

    def objective(t, r, s):
        L, _, _, _, residual = m._factor(t, r, s.exp())
        return L.diagonal().log().sum() + 0.5 * residual.square().sum()

    assert torch.autograd.gradcheck(objective, (theta, rho, log_s), atol=2e-4)
