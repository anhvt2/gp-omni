"""Size a tapered aluminum cantilever under a tip load using scikit-fem."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from skfem import Basis, BilinearForm, ElementLineHermite, MeshLine, asm, condense, solve

LENGTH_M = 0.6
WIDTH_M = 0.045
TIP_LOAD_N = 1_000
YOUNG_MODULUS_PA = 69e9
DENSITY_KG_M3 = 2_700
DEFLECTION_LIMIT_MM = 1.5
REFERENCE_ROOT_MM = 60
ROOT_HEIGHT_BOUNDS_MM = (45, 90)
TIP_RATIO_BOUNDS = (0.5, 1.0)


def solve_beam(root_height_mm: float, tip_ratio: float, elements: int = 24):
    """Return the mass, Hermite basis, and displacement coefficients."""
    root_height = root_height_mm / 1_000
    tip_height = root_height * tip_ratio
    mesh = MeshLine(np.linspace(0, LENGTH_M, elements + 1)).with_boundaries(
        {"root": lambda x: np.isclose(x[0], 0), "tip": lambda x: np.isclose(x[0], LENGTH_M)}
    )
    basis = Basis(mesh, ElementLineHermite(), intorder=6)

    @BilinearForm
    def bending(u, v, w):
        height = root_height + (tip_height - root_height) * w.x[0] / LENGTH_M
        inertia = WIDTH_M * height**3 / 12
        return YOUNG_MODULUS_PA * inertia * u.hess[0, 0] * v.hess[0, 0]

    tip = basis.get_dofs("tip").nodal["u"].item()
    load = np.zeros(basis.N)
    load[tip] = -TIP_LOAD_N
    displacement = solve(*condense(asm(bending, basis), load, D=basis.get_dofs("root")))
    mass = DENSITY_KG_M3 * LENGTH_M * WIDTH_M * (root_height + tip_height) / 2
    return mass, basis, displacement


def analyze(root_height_mm: float, tip_ratio: float, elements: int = 24) -> tuple[float, float]:
    """Return mass in kg and downward tip deflection in mm."""
    mass, basis, displacement = solve_beam(root_height_mm, tip_ratio, elements)
    tip = basis.get_dofs("tip").nodal["u"].item()
    return mass, -1_000 * displacement[tip]


def evaluate(x: np.ndarray) -> tuple[float, bool]:
    root_ratio, tip_ratio = x
    mass, deflection = analyze(REFERENCE_ROOT_MM * root_ratio, tip_ratio)
    return -mass, deflection <= DEFLECTION_LIMIT_MM


def main() -> None:
    from gp_omni.opt.parallel.batchbo import batch_bo, batch_bo_config

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for the BO campaign")
    args = parser.parse_args()
    args.output.mkdir(parents=True)

    config = batch_bo_config(
        model_name="cantilever",
        x_lb=np.array([ROOT_HEIGHT_BOUNDS_MM[0] / REFERENCE_ROOT_MM, TIP_RATIO_BOUNDS[0]]),
        x_ub=np.array([ROOT_HEIGHT_BOUNDS_MM[1] / REFERENCE_ROOT_MM, TIP_RATIO_BOUNDS[1]]),
        num_init_point=8,
        max_iter=16,
        exploit_size=2,
        explore_size=0,
        explore_classifier_size=1,
        check_time=0.001,
        seed=0,
    )
    x, objective = batch_bo(args.output, config, evaluate=evaluate).run()
    if not np.isfinite(objective):
        raise RuntimeError("The campaign found no feasible design.")
    root_height = REFERENCE_ROOT_MM * x[0]
    mass, deflection = analyze(root_height, x[1])
    print(f"Root height {root_height:.2f} mm, tip height {root_height * x[1]:.2f} mm")
    print(f"Mass {mass:.3f} kg, tip deflection {deflection:.3f} mm")


if __name__ == "__main__":
    main()
