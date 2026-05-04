"""Thin coordinator – holds PDE model, delegates to backends."""
import numpy as np
from .time_integrators.scipy_integrator import ScipyIntegrator
from .time_integrators.sundials_integrator import SundialsIntegrator
from .time_integrators.base import SolverResult, SolverStats
from .spatial_discretizations.base import SpatialDiscretization
from .hooks import SolverHook
from typing import Optional
from .time_integrators.fenicsx_direct import FEniCSxDirectIntegrator

class PLaplacianSolver:
    def __init__(self, discretization: SpatialDiscretization, config):
        self.disc = discretization
        self.config = config

    def solve(self, times, hook: Optional[SolverHook] = None,
              check_propagation: bool = False):
        t_eval = np.sort(times)
        y0 = self.disc.get_initial_state()

        if self.config.method.upper() == "FENICSX_DIRECT":
            backend = FEniCSxDirectIntegrator(self.disc)
        elif self.config.method.upper() in ("CVODE", "IDA"):
            backend = SundialsIntegrator()
        else:
            backend = ScipyIntegrator()

        jac = None
        #if hasattr(self.disc, "compute_jac_rhs"):
        #   jac = lambda t, y: self.disc.compute_jac_rhs(t, y)

            
        sparsity = self.disc.sparsity_pattern if self.config.sparse else None

        # Wrap RHS so the hook receives the *full* solution
        def rhs_wrapped(t, y):
            dydt = self.disc.compute_rhs(t, y)
            if hook:
                full = self.disc.get_full_solution(y)
                hook(t, full)
            return dydt

        result = backend.solve(
            t_eval, y0, rhs_wrapped, sparsity=sparsity, jac=jac,
            rtol=self.config.rtol, atol=self.config.atol,
            method=self.config.method, sparse=self.config.sparse, dt=self.config.dt
        )

        # Map state vectors to full spatial solutions
        full_data = {t: self.disc.get_full_solution(state)
                     for t, state in result.data.items()}

        if check_propagation:
            if not result.stats.success:
                return full_data, result.stats
            if not full_data:
                return full_data, SolverStats(
                    False, "No output data", 0, 0, 0)
            interiors = np.column_stack(
                [full_data[t][1:-1] for t in sorted(full_data.keys())]
            )
            max_interior = interiors.max()
            min_interior = interiors.min()
            max_abs_interior = np.abs(interiors).max()

            if max_interior > 1.001:
                return full_data, SolverStats(
                    False,
                    f"Propagation error: interior > 1.0 (max={max_interior:.6f})",
                    result.stats.nfev, result.stats.njev, result.stats.nlu)
            if max_abs_interior <= 0.001:
                return full_data, SolverStats(
                    False, "Propagation failure: no interior movement",
                    result.stats.nfev, result.stats.njev, result.stats.nlu)
            if min_interior < -0.001:
                return full_data, SolverStats(
                    False,
                    f"Propagation error: negative value (min={min_interior:.6f})",
                    result.stats.nfev, result.stats.njev, result.stats.nlu)

        return full_data, result.stats
