"""Finite difference discretisation of the 1-D p‑Laplacian."""

import numba
import numpy as np
from scipy.sparse import diags_array, spmatrix

from .base import SpatialDiscretization


@numba.njit(fastmath=True)
def _fast_rhs(t, u, p, dx, h, epsilon):
    N = len(u)
    dudt = np.empty(N)
    # boundary fluxes …
    grad = (u[0] - h) / dx
    flux_in = (grad * grad + epsilon * epsilon) ** ((p - 2) / 2) * grad

    for i in range(N - 1):
        grad = (u[i + 1] - u[i]) / dx
        flux_out = (grad * grad + epsilon * epsilon) ** ((p - 2) / 2) * grad
        dudt[i] = (flux_out - flux_in) / dx
        flux_in = flux_out

    grad = (0.0 - u[N - 1]) / dx
    flux_out = (grad * grad + epsilon * epsilon) ** ((p - 2) / 2) * grad
    dudt[N - 1] = (flux_out - flux_in) / dx
    return dudt


class FDMDiscretization(SpatialDiscretization):
    """Uniform grid, finite‑difference stencil."""

    def __init__(self, p: float, h: float, L: float, Nx: int, epsilon: float):
        self.p = p
        self.h = h
        self.L = L
        self.Nx = Nx
        self.epsilon = epsilon
        self.dx = L / Nx
        self._x_full = np.linspace(0, L, Nx + 1)
        self._sparsity = diags_array(
            [np.ones(Nx - 2), np.ones(Nx - 1), np.ones(Nx - 2)],
            offsets=(-1, 0, 1),
            shape=(Nx - 1, Nx - 1),
            format="csc",
        )

    @property
    def state_size(self) -> int:
        return self.Nx - 1  # interior nodes

    def get_initial_state(self) -> np.ndarray:
        return np.zeros(self.Nx - 1)

    def compute_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        return _fast_rhs(t, state, self.p, self.dx, self.h, self.epsilon)

    @property
    def sparsity_pattern(self) -> spmatrix:
        return self._sparsity

    def get_full_solution(self, state: np.ndarray) -> np.ndarray:
        full = np.empty(self.Nx + 1)
        full[0] = self.h
        full[1:-1] = state
        full[-1] = 0.0
        return full

    def get_node_coordinates(self) -> np.ndarray:
        return self._x_full

    def compute_l2_error(self, state: np.ndarray, ref_state: np.ndarray) -> float:
        return np.sqrt(self.dx * np.sum((state - ref_state) ** 2))
