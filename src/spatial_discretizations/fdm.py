"""Finite difference discretisation of the 1-D p‑Laplacian."""

import numba
import numpy as np
from scipy.sparse import csc_matrix, diags_array, spmatrix

from .base import SpatialDiscretization


@numba.njit(fastmath=True)
def _fast_rhs(t, u, p, dx, h, epsilon):
    N = len(u)
    dudt = np.empty(N)
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


@numba.njit(fastmath=True, inline="always")
def _Gprime(g, p, epsilon):
    """Derivative of the flux (|g|^p term) with respect to gradient g."""
    g2e2 = g * g + epsilon * epsilon
    if abs(p - 2.0) < 1e-14:
        # p=2 => linear diffusion => G'(g) = 1
        return 1.0
    # G(g) = (g^2+eps^2)^((p-2)/2) * g
    # G'(g) = (g^2+eps^2)^((p-2)/2) + (p-2)*g^2*(g^2+eps^2)^((p-4)/2)
    term1 = g2e2 ** ((p - 2.0) / 2.0)
    term2 = (p - 2.0) * g * g * g2e2 ** ((p - 4.0) / 2.0)
    return term1 + term2


@numba.njit(fastmath=True)
def _update_jac_banded_inplace(u, p, dx, h, epsilon, banded):
    N = len(u)
    dx2 = dx * dx

    # Precompute Gprime on the N+1 edges to avoid redundant fractional powers
    Gp = np.empty(N + 1)

    # Left boundary edge
    Gp[0] = _Gprime((u[0] - h) / dx, p, epsilon)

    # Interior edges
    for i in range(1, N):
        Gp[i] = _Gprime((u[i] - u[i - 1]) / dx, p, epsilon)

    # Right boundary edge
    Gp[N] = _Gprime((0.0 - u[N - 1]) / dx, p, epsilon)

    # Assemble banded matrix
    for i in range(N):
        Gp_m = Gp[i]
        Gp_p = Gp[i + 1]

        banded[1, i] = -(Gp_p + Gp_m) / dx2
        if i > 0:
            banded[2, i - 1] = Gp_m / dx2
        if i < N - 1:
            banded[0, i + 1] = Gp_p / dx2


@numba.njit(fastmath=True)
def _pack_csc_from_banded(banded, data, indptr):
    N = banded.shape[1]
    for j in range(N):
        start = indptr[j]
        if j == 0:
            data[start] = banded[1, 0]
            data[start + 1] = banded[2, 0]
        elif j == N - 1:
            data[start] = banded[0, j]
            data[start + 1] = banded[1, j]
        else:
            data[start] = banded[0, j]
            data[start + 1] = banded[1, j]
            data[start + 2] = banded[2, j]


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

        # Preallocate zero-allocation Jacobian structures
        N = self.Nx - 1
        self._banded_jac = np.zeros((3, N))

        # Manually pack a CSC structure for zero-allocation sparse updates
        indptr = np.zeros(N + 1, dtype=np.int32)
        nnz = 3 * N - 2
        indices = np.zeros(nnz, dtype=np.int32)
        data = np.zeros(nnz, dtype=np.float64)

        idx = 0
        for j in range(N):
            indptr[j] = idx
            if j > 0:
                indices[idx] = j - 1
                idx += 1
            indices[idx] = j
            idx += 1
            if j < N - 1:
                indices[idx] = j + 1
                idx += 1
        indptr[N] = idx
        self._jac_csc = csc_matrix((data, indices, indptr), shape=(N, N))

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

    def compute_jac_banded(self, t: float, state: np.ndarray) -> np.ndarray:
        _update_jac_banded_inplace(
            state, self.p, self.dx, self.h, self.epsilon, self._banded_jac
        )
        return self._banded_jac

    def compute_jac_rhs(self, t: float, state: np.ndarray) -> spmatrix:
        self.compute_jac_banded(t, state)
        _pack_csc_from_banded(
            self._banded_jac, self._jac_csc.data, self._jac_csc.indptr
        )
        return self._jac_csc
