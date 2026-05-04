# src/time_integrators/scipy_integrator.py
"""SciPy solve_ivp integration backend with analytical Jacobian support."""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import issparse
from typing import Callable, Optional
from .base import SolverResult, SolverStats
from ..hooks import SolverHook


def _sparse_to_banded(J_sparse, n, lband, uband):
    """Convert a tridiagonal sparse matrix (lband=uband=1) to LSODA banded format.

    Parameters
    ----------
    J_sparse : scipy.sparse matrix (n, n)
    n : int
        Number of ODE variables.
    lband, uband : int
        Lower and upper bandwidths (must be 1 for tridiagonal).

    Returns
    -------
    banded_jac : ndarray of shape (lband+uband+1, n)
        Banded storage in Fortran order (upper band first, then diagonal, then lower band).
        Entries outside the matrix are set to 0.
    """
    if lband != 1 or uband != 1:
        raise NotImplementedError("Only tridiagonal banded Jacobian is supported here.")
    J = J_sparse.tocoo()
    # Allocate banded storage: rows 0..(lband+uband) correspond to
    #   row 0: upper band (j, j+1)
    #   row 1: diagonal (j, j)
    #   row 2: lower band (j, j-1)
    banded = np.zeros((3, n), dtype=float)
    for i, j, v in zip(J.row, J.col, J.data):
        if i == j:
            banded[1, j] = v
        elif i == j - 1:          # super-diagonal
            banded[0, j] = v      # upper band for column j
        elif i == j + 1:          # sub-diagonal
            banded[2, j] = v
    return banded


class ScipyIntegrator:
    def solve(
        self,
        t_eval: np.ndarray,
        y0: np.ndarray,
        rhs: Callable,
        sparsity: Optional = None,
        jac: Optional[Callable] = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        hook: Optional[SolverHook] = None,
        **kwargs,
    ) -> SolverResult:
        method = kwargs.pop("method", "LSODA")
        sparse = kwargs.pop("sparse", True)

        solve_kwargs = dict(method=method, rtol=rtol, atol=atol, t_eval=t_eval)

        # --- Handle Jacobian ---
        if jac is not None:
            # If we have an analytical Jacobian, wrap it to return the right format
            if method == "LSODA" and sparse:
                # LSODA with banded Jacobian: need (lband+uband+1, n) array
                lband, uband = 1, 1
                solve_kwargs["lband"] = lband
                solve_kwargs["uband"] = uband
                n = len(y0)
                jac_banded = lambda t, y: _sparse_to_banded(jac(t, y), n, lband, uband)
                solve_kwargs["jac"] = jac_banded
            else:
                # BDF, Radau (or LSODA dense) – the callable itself returns a
                # sparse matrix, which solve_ivp accepts directly.
                solve_kwargs["jac"] = jac
        else:
            # No analytical Jacobian: use sparsity pattern for finite‑difference
            # approximations (LSODA ignores this flag and uses band storage).
            if method == "LSODA" and sparse:
                solve_kwargs["lband"] = 1
                solve_kwargs["uband"] = 1
            elif sparse and sparsity is not None:
                solve_kwargs["jac_sparsity"] = sparsity

        # Wrap RHS to call the hook
        def rhs_wrapper(t, y):
            dydt = rhs(t, y)
            if hook:
                hook(t, y.copy())
            return dydt

        sol = solve_ivp(rhs_wrapper, (0, max(t_eval)), y0, **solve_kwargs)

        stats = SolverStats(
            success=sol.success,
            message=sol.message,
            nfev=sol.nfev,
            njev=sol.njev,
            nlu=sol.nlu,
        )
        data = {t: sol.y[:, i] for i, t in enumerate(sol.t) if t in t_eval}
        return SolverResult(data, stats)
