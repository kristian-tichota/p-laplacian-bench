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

    def solve(self, times, method="LSODA", sparse=True, rtol=1e-6, atol=1e-6, hook: SolverHook = None):
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
            rhs=lambda t, y: fast_p_laplacian_rhs(t, y, self.model.p, self.model.dx, self.model.h, self.model.epsilon),
            sparsity=sparsity,
            rtol=rtol, atol=atol,
            method=method, sparse=sparse,
            hook=hook            # forward the observer
        )

        # reconstruct full u(x) with boundaries
        full_data = {}
        for t, u_int in result.data.items():
            full = np.empty(self.model.Nx+1)
            full[0] = self.model.h
            full[1:-1] = u_int
            full[-1] = 0.0
            full_data[t] = full
        return full_data, result.stats
