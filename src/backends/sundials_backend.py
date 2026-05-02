"""SUNDIALS (CVODE) backend via scikits.odes."""
import numpy as np
from typing import Callable, Optional
from .base import SolverBackend, SolverResult, SolverStats

class SundialsSolver:
    def solve(self, t_eval, y0, rhs, sparsity=None, rtol=1e-6, atol=1e-6, **kwargs):
        try:
            from scikits.odes import ode
        except ImportError as exc:
            raise ImportError("Install scikits.odes for SUNDIALS support") from exc

        method = kwargs.get("method", "cvode").lower()
        options = dict(rtol=rtol, atol=atol,
                       max_steps=kwargs.get("max_steps", 1000000),
                       lband=1, uband=1, linsolver="band")
        solver = ode(method, lambda t, y, ydot: ydot.__setitem__(slice(None), rhs(t, y)), **options)

        t_vec = np.array(t_eval)
        if t_vec[0] != 0.0:
            t_vec = np.insert(t_vec, 0, 0.0)

        sol = solver.solve(t_vec, y0)
        if sol.flag < 0:
            return SolverResult({}, SolverStats(False, f"SUNDIALS failed: flag {sol.flag}", 0, 0, 0))

        info = solver.get_info()
        stats = SolverStats(True, "ok",
                            info.get("NumRhsEvals", 0),
                            info.get("NumJacEvals", 0),
                            info.get("NumLinSolvSetups", 0))
        data = {t: sol.values.y[i, :]
                for i, t in enumerate(sol.values.t) if t in t_eval}
        return SolverResult(data, stats)
