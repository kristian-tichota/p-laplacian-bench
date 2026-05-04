"""FEniCSx‑native time‑stepper without method of lines (backward Euler,
nonlinear solve at each step)."""

from __future__ import annotations

from typing import Dict, Optional

import dolfinx
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import numpy as np
import ufl
from dolfinx import fem, nls
from mpi4py import MPI

from ..hooks import SolverHook
from ..spatial_discretizations.base import SpatialDiscretization
from .base import SolverResult, SolverStats


class FEniCSxDirectIntegrator:
    """Implicit time integration using FEniCSx’s Newton solver directly."""

    def __init__(self, discretization: SpatialDiscretization):
        # This assumes discretization is a FEniCSxDiscretization
        self.disc = discretization

    def solve(
        self,
        t_eval: np.ndarray,
        y0: np.ndarray,
        rhs=None,  # ignored, only for interface compatibility
        sparsity=None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        method: str = "FENICSX_DIRECT",
        sparse: bool = True,
        hook: Optional[SolverHook] = None,
        **kwargs,
    ) -> SolverResult:
        # Extract parameters from the discretisation
        p = self.disc.p
        epsilon = self.disc.epsilon
        L = self.disc.L
        V = self.disc.function_space
        mesh = self.disc.mesh
        bcs = self.disc.get_dirichlet_bcs()

        # Time stepping settings
        dt_init = kwargs.get("dt", 1e-3)
        dt_min = kwargs.get("dt_min", 1e-12)
        dt_max = kwargs.get("dt_max", 0.1 * max(t_eval))
        newton_maxit = kwargs.get("newton_maxit", 30)

        # Unknown and previous state
        u = fem.Function(V)
        u_old = fem.Function(V)
        u_old.x.array[:] = y0
        u.x.array[:] = y0

        # Define variational problem
        v = ufl.TestFunction(V)
        dt = fem.Constant(mesh, dt_init)
        grad_u = ufl.grad(u)
        D = (ufl.dot(grad_u, grad_u) + epsilon**2) ** ((p - 2) / 2)
        # Backward Euler residual: (u - u_old)/dt * v + D grad(u)·grad(v) dx
        F = ((u - u_old) / dt * v + D * ufl.dot(grad_u, ufl.grad(v))) * ufl.dx

        problem = fem.petsc.NewtonSolverNonlinearProblem(F, u, bcs=bcs)

        solver = nls.petsc.NewtonSolver(MPI.COMM_SELF, problem)
        solver.rtol = rtol
        solver.atol = atol
        solver.max_it = newton_maxit

        t = 0.0
        t_end = max(t_eval)
        t_out_idx = 0
        data: Dict[float, np.ndarray] = {}
        nfe_total = nje_total = nlu_total = 0
        current_dt = dt_init

        while t < t_end:
            # If we would overshoot the next required output, step exactly to it
            next_out_t = t_eval[t_out_idx]
            if t + current_dt >= next_out_t:
                dt_val = next_out_t - t
                if dt_val <= 0:
                    t_out_idx += 1
                    continue
                dt.value = dt_val
                u_old.x.array[:] = u.x.array
                try:
                    niter, converged = solver.solve(u)
                except Exception:
                    converged = False
                    niter = 0
                if not converged:
                    current_dt *= 0.5
                    if current_dt < dt_min:
                        return SolverResult(
                            data, SolverStats(False, f"Time step too small at t={t}")
                        )
                    u.x.array[:] = u_old.x.array
                    continue

                nfe_total += niter
                nje_total += niter
                nlu_total += niter
                t = next_out_t
                data[t] = u.x.array.copy()
                t_out_idx += 1
                if t_out_idx >= len(t_eval):
                    break
                # Adapt dt based on Newton iterations
                if niter <= 3:
                    current_dt = min(current_dt * 2.0, dt_max)
                elif niter >= 8:
                    current_dt = max(current_dt * 0.5, dt_min)
            else:
                dt_val = current_dt
                dt.value = dt_val
                u_old.x.array[:] = u.x.array
                try:
                    niter, converged = solver.solve(u)
                except Exception:
                    converged = False
                    niter = 0
                if not converged:
                    current_dt *= 0.5
                    if current_dt < dt_min:
                        return SolverResult(
                            data, SolverStats(False, f"Time step too small at t={t}")
                        )
                    u.x.array[:] = u_old.x.array
                    continue

                nfe_total += niter
                nje_total += niter
                nlu_total += niter
                t += dt_val
                if hook:
                    hook(t, u.x.array.copy())
                if niter <= 3:
                    current_dt = min(current_dt * 2.0, dt_max)
                elif niter >= 8:
                    current_dt = max(current_dt * 0.5, dt_min)

        stats = SolverStats(True, "ok", nfev=nfe_total, njev=nje_total, nlu=nlu_total)
        return SolverResult(data, stats)
