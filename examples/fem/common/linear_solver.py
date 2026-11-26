"""Sparse direct solution of symmetric positive definite finite element systems."""

import numpy as np
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator, eigsh, splu


class factorization:
    """LU factors of a symmetric positive definite matrix, reusable across load cases."""

    def __init__(self, matrix: spmatrix):
        self.matrix = matrix.tocsc()
        self.factors = splu(
            self.matrix,
            permc_spec="MMD_AT_PLUS_A",
            diag_pivot_thresh=0.0,
            options={"SymmetricMode": True},
        )

    def solve(self, load: np.ndarray) -> np.ndarray:
        solution = self.factors.solve(load)
        residual = np.linalg.norm(self.matrix @ solution - load)
        if not residual <= 1e-5 * np.linalg.norm(load):
            raise RuntimeError("inaccurate linear solve; is the model fully constrained?")
        return solution

    def smallest_eigenvalues(self, mass: spmatrix, count: int) -> np.ndarray:
        """The ``count`` smallest eigenvalues of ``matrix x = value mass x``."""
        size = self.matrix.shape[0]
        values = eigsh(
            self.matrix,
            k=count,
            M=mass,
            sigma=0,
            which="LM",
            OPinv=LinearOperator((size, size), self.factors.solve),
            v0=np.random.default_rng(0).random(size),
            return_eigenvectors=False,
        )
        return np.sort(values)
