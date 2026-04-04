import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import diags_array
from .physics import fast_p_laplacian_rhs


class PLaplacianSolver:
    def __init__(self, p, h, L=1.0, Nx=1000, epsilon=1e-6):
        self.p = p
        self.h = h
        self.Nx = Nx
        self.dx = L / Nx
        self.epsilon = epsilon
        self.x = np.linspace(0, L, Nx + 1)

        n_dim = Nx - 1
        diagonals = [
            np.ones(n_dim - 1),
            np.ones(n_dim),
            np.ones(n_dim - 1),
        ]

        self.sparsity = diags_array(
            diagonals,
            offsets=(-1, 0, 1),  # type: ignore
            shape=(n_dim, n_dim),
            format="csc",
            dtype=float,
        )

    def solve(self, times, method="LSODA", sparse=True, rtol=1e-5, atol=1e-6, **kwargs):
        u0_interior = np.zeros(self.Nx - 1)
        t_eval = sorted(times)
        
        is_sundials = isinstance(method, str) and method.upper() in ["CVODE", "IDA"]

        jac_val = None
        if sparse:
            if isinstance(method, str) and (method == "LSODA" or is_sundials):
                # LSODA and SUNDIALS expect bandwidths for banded Jacobians
                kwargs["lband"] = 1
                kwargs["uband"] = 1
            else:
                # SciPy's BDF/Radau or compatible custom classes expect the sparse matrix
                jac_val = self.sparsity

        if is_sundials:
            return self._solve_sundials(t_eval, u0_interior, method, rtol, atol, **kwargs)

        solve_ivp_kwargs = {
            "fun": fast_p_laplacian_rhs,
            "t_span": (0, max(times)),
            "y0": u0_interior,
            "method": method,
            "t_eval": t_eval,
            "args": (self.p, self.dx, self.h, self.epsilon),
            "rtol": rtol,
            "atol": atol,
            **kwargs
        }

        # Dynamically attach jac_sparsity to avoid breaking solvers that reject it natively
        if jac_val is not None:
            solve_ivp_kwargs["jac_sparsity"] = jac_val

        sol = solve_ivp(**solve_ivp_kwargs)

        reconstructed_data = self._reconstruct(sol.t, sol.y)

        stats = {
            "nfev": sol.nfev,
            "njev": sol.njev,
            "nlu": sol.nlu,
            "success": sol.success,
            "message": sol.message,
        }

        return reconstructed_data, stats

    def _solve_sundials(self, t_eval, u0_interior, method, rtol, atol, **kwargs):
        try:
            from scikits.odes import ode
        except ImportError as exc:
            raise ImportError(
                "SUNDIALS solvers require the 'scikits.odes' package to be installed."
            ) from exc

        # SUNDIALS expects the signature f(t, y, ydot) allowing inplace updates
        def rhs_sundials(t, y, ydot):
            ydot[:] = fast_p_laplacian_rhs(t, y, self.p, self.dx, self.h, self.epsilon)

        options = {"rtol": rtol, "atol": atol, **kwargs}
        solver = ode(method.lower(), rhs_sundials, **options)
        sol = solver.solve(t_eval, u0_interior)

        # sol.values.y has shape (len(t_eval), len(u0)) -> transpose for SciPy compatibility
        y_matrix = sol.values.y.T 
        reconstructed_data = self._reconstruct(sol.values.t, y_matrix)

        stats = {
            "success": sol.flag >= 0,
            "message": sol.message,
        }
        return reconstructed_data, stats

    def _reconstruct(self, time_steps, y_matrix):
        results = {}
        for i, t in enumerate(time_steps):
            u_full = np.empty(self.Nx + 1)
            u_full[0] = self.h
            u_full[1:-1] = y_matrix[:, i]
            u_full[-1] = 0.0
            results[t] = u_full
        return results
