"""Thin coordinator – holds PDE model, delegates to backends."""
import numpy as np
from .backends.scipy_backend import ScipySolver
from .backends.sundials_backend import SundialsSolver
from .backends.base import SolverResult, SolverStats
from .physics import fast_p_laplacian_rhs
from .model import PLaplacianModel
from .hooks import SolverHook

class PLaplacianSolver:
    def __init__(self, model: PLaplacianModel):
        self.model = model

    def solve(self, times, method="LSODA", sparse=True, rtol=1e-6, atol=1e-6,
              hook: SolverHook = None, check_propagation=False):
        t_eval = np.sort(times)
        y0 = np.zeros(self.model.Nx - 1)  # interior only

        # select backend
        if method.upper() in ("CVODE", "IDA"):
            backend = SundialsSolver()
        else:
            backend = ScipySolver()

        # pass sparsity pattern if needed
        sparsity = self.model.sparsity if sparse else None

        result = backend.solve(
            t_eval, y0,
            rhs=lambda t, y: fast_p_laplacian_rhs(
                t, y, self.model.p, self.model.dx, self.model.h, self.model.epsilon
            ),
            sparsity=sparsity,
            rtol=rtol, atol=atol,
            method=method, sparse=sparse,
            hook=hook            # forward the observer
        )

        # reconstruct full u(x) with boundaries
        full_data = {}
        for t, u_int in result.data.items():
            full = np.empty(self.model.Nx + 1)
            full[0] = self.model.h
            full[1:-1] = u_int
            full[-1] = 0.0
            full_data[t] = full

        # ----- Propagation sanity checks -----
        if check_propagation:
            if not result.stats.success:
                return full_data, result.stats
            if not full_data:
                return full_data, SolverStats(
                    False,
                    "Solver returned no output data (propagation check skipped)",
                    result.stats.nfev, result.stats.njev, result.stats.nlu
                )
            # Gather all interior values across all time steps
            interiors = np.column_stack(
                [full_data[t][1:-1] for t in sorted(full_data.keys())]
            )
            max_interior = interiors.max()
            min_interior = interiors.min()
            max_abs_interior = np.abs(interiors).max()

            # 1) Interior ever above 1.001 ?
            if max_interior > 1.001:
                return full_data, SolverStats(
                    False,
                    f"Propagation error: interior exceeded 1.0 (max={max_interior:.6f})",
                    result.stats.nfev, result.stats.njev, result.stats.nlu
                )
            # 2) Max absolute value never above 0.001 ?
            if max_abs_interior <= 0.001:
                return full_data, SolverStats(
                    False,
                    "Propagation failure: solution did not propagate (max|interior| ≤ 0.001)",
                    result.stats.nfev, result.stats.njev, result.stats.nlu
                )
            # 3) Interior ever below -0.001 ?
            if min_interior < -0.001:
                return full_data, SolverStats(
                    False,
                    f"Propagation error: negative interior value (min={min_interior:.6f})",
                    result.stats.nfev, result.stats.njev, result.stats.nlu
                )

        return full_data, result.stats
