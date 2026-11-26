"""Use case 1: correlated multi-objective (several outputs that share structure).

Two objectives measured at the same inputs, but the second has only 8 samples. The joint
model borrows strength from the first; a stand-alone GP on the 8 points cannot.

Data layout: stack all rows in one ``x``/``y`` and tell the model which output each row is.
"""

import torch

import gp_omni  # noqa: F401
from gp_omni import GPR


def main() -> dict:
    g = torch.Generator().manual_seed(0)
    base = lambda x: torch.sin(3 * x[:, 0]) + torch.cos(2 * x[:, 1]) + x[:, 2] ** 2
    f = [lambda x: base(x), lambda x: 0.8 * base(x) + 0.2]  # two correlated objectives
    n = [150, 8]  # samples per output
    x = torch.cat([torch.rand(k, 3, generator=g) for k in n])
    output = torch.cat([torch.full((k,), float(o)).long() for o, k in enumerate(n)])
    y = torch.cat(
        [f[o](x[output == o]) + 0.02 * torch.randn(k, generator=g) for o, k in enumerate(n)]
    )

    model = GPR(num_outputs=2).fit(x, y, output=output)  # <- the feature switch
    alone = GPR().fit(x[output == 1], y[output == 1])  # only the 8 samples

    xt = torch.rand(400, 3, generator=g)
    truth = f[1](xt)
    rmse = lambda p: ((p - truth) ** 2).mean().sqrt().item() / truth.std().item()
    return {
        "joint_nrmse": rmse(model.predict(xt, output=1)["fmu"]),
        "alone_nrmse": rmse(alone.predict(xt)["fmu"]),
    }


if __name__ == "__main__":
    print(main())
