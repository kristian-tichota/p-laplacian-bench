"""SciPy solve_ivp integration backend."""
import numpy as np
from scipy.integrate import solve_ivp
from typing import Callable, Optional
from .base import SolverResult, SolverStats
from ..hooks import SolverHook

class ScipyIntegrator:
    def solve(self, t_eval, y0, rhs, sparsity=None, rtol=1e-5, atol=1e-6, hook: Optional[SolverHook] = None, **kwargs):
        method = kwargs.pop("method", "LSODA")
        sparse = kwargs.pop("sparse", True)

        solve_kwargs = dict(method=method, rtol=rtol, atol=atol, t_eval=t_eval)
        if method == "LSODA" and sparse:
            solve_kwargs["lband"] = 1
            solve_kwargs["uband"] = 1
        elif sparse and sparsity is not None:
            solve_kwargs["jac_sparsity"] = sparsity

        # Wrap RHS to call the hook
        def rhs_wrapper(t, y):
            dydt = rhs(t, y)
            if hook:
                hook(t, y.copy())   # copy to avoid mutation during live plotting
            return dydt

        sol = solve_ivp(rhs_wrapper, (0, max(t_eval)), y0, **solve_kwargs)

        stats = SolverStats(success=sol.success, message=sol.message,
                            nfev=sol.nfev, njev=sol.njev, nlu=sol.nlu)
        data = {t: sol.y[:, i] for i, t in enumerate(sol.t) if t in t_eval}
        return SolverResult(data, stats)
