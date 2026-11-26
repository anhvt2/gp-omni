"""Screen a multi-fidelity Pareto front, then evaluate a high-fidelity batch in parallel.

This is one custom surrogate-screening round, not a built-in multi-objective BO loop.
Both objectives are maximized; feasibility is checked again after evaluation.
"""

from concurrent.futures import ThreadPoolExecutor

import torch

from gp_omni import GPC, GPR


def objectives(x, fidelity):
    """Two competing scores, with a biased cheap approximation at fidelity 0."""
    y = torch.cat([1 - (x - 0.2) ** 2, 1 - (x - 0.8) ** 2], dim=1)
    return y if fidelity == 1 else 0.85 * y + 0.08 * torch.sin(6 * x)


def evaluate_high(x):
    """Replace with an expensive simulator returning two scores and feasibility."""
    return objectives(x, 1), x[:, 0] <= 0.65


def nondominated(y):
    """Keep rows for which no other row is at least as good in every objective."""
    dominates = (y[:, None] >= y[None, :]).all(-1) & (y[:, None] > y[None, :]).any(-1)
    return ~dominates.any(dim=0)


def main():
    torch.manual_seed(0)
    x_low = torch.linspace(0, 1, 25).reshape(-1, 1)
    x_high = x_low[::3]  # nine expensive designs, nested within the cheap design
    y_high, feasible_high = evaluate_high(x_high)

    # Long-form observations: one row per (design, objective, fidelity).
    x = torch.cat([x_low, x_high])
    y = torch.cat([objectives(x_low, 0), y_high])
    fidelity = torch.cat([torch.zeros(len(x_low)), torch.ones(len(x_high))]).long()
    surrogate = GPR(num_outputs=2, num_fidelities=2).fit(
        x.repeat_interleave(2, dim=0),
        y.reshape(-1),
        output=torch.arange(2).repeat(len(x)),
        fidelity=fidelity.repeat_interleave(2),
        max_iter=80,
    )
    # Feasibility labels here are measured only at the target (high) fidelity.
    constraint = GPC().fit(x_high, feasible_high, max_iter=60)

    candidates = torch.linspace(0, 1, 201).reshape(-1, 1)
    unseen = torch.cdist(candidates, x_high).min(dim=1).values > 1e-6
    probability = constraint.predict_proba(candidates)
    candidates = candidates[unseen & (probability >= 0.8)]
    if len(candidates) == 0:
        raise RuntimeError("No candidates pass the feasibility screen; collect more labels.")
    # Predictions default to the highest fidelity, returning shape (n_candidates, 2).
    mean = surrogate.predict(candidates)["fmu"]
    front = candidates[nondominated(mean)]
    front = front[front[:, 0].argsort()]
    indices = torch.linspace(0, len(front) - 1, min(4, len(front))).round().long()
    batch = front[indices]

    # Up to four concurrent evaluations, followed by a batch barrier.
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(evaluate_high, batch.split(1)))
    y_batch = torch.cat([result[0] for result in results])
    feasible_batch = torch.cat([result[1] for result in results])

    # Report observed feasible trade-offs, including the original high-fidelity data.
    observed_x = torch.cat([x_high, batch])
    observed_y = torch.cat([y_high, y_batch])
    feasible = torch.cat([feasible_high, feasible_batch])
    keep = nondominated(observed_y[feasible])
    return {
        "low_fidelity_designs": len(x_low),
        "initial_high_fidelity_designs": len(x_high),
        "parallel_batch_size": len(batch),
        "batch_feasible": feasible_batch.tolist(),
        "pareto_x": observed_x[feasible][keep].flatten().tolist(),
        "pareto_objectives": observed_y[feasible][keep].tolist(),
    }


if __name__ == "__main__":
    print(main())
