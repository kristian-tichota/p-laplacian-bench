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
            if isinstance(method, str) and (method == "LSODA"):
                # LSODA and SUNDIALS expect bandwidths for banded Jacobians
                kwargs["lband"] = 1
                kwargs["uband"] = 1
            elif is_sundials:
                kwargs["lband"] = 1
                kwargs["uband"] = 1
                kwargs["linsolver"] = "band"
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

        # SUNDIALS requires the first element of the time array to be the initial time (0.0)
        t_sundials = np.array(t_eval)
        if t_sundials[0] != 0.0:
            t_sundials = np.insert(t_sundials, 0, 0.0)

        sol = solver.solve(t_sundials, u0_interior)

        if sol.flag < 0:
            error_msg = f"SUNDIALS solver failed with flag {sol.flag}"
            if getattr(sol, "errors", None):
                error_msg += f": {sol.errors}"
            return None, {"success": False, "message": error_msg}

        # sol.values.y has shape (len(t_sundials), len(u0)) -> transpose for SciPy compatibility
        y_matrix = sol.values.y.T 
        t_out = sol.values.t

        # Filter out the t=0.0 step if it wasn't originally requested by the user
        valid_indices = [i for i, t in enumerate(t_out) if t in t_eval]
        y_matrix_filtered = y_matrix[:, valid_indices]
        t_out_filtered = t_out[valid_indices]

        reconstructed_data = self._reconstruct(t_out_filtered, y_matrix_filtered)

        # Retrieve statistics from the solver object, not the solution namedtuple
        info = solver.get_info()

        stats = {
            "success": True,
            "message": getattr(sol, "message", "Integration successful."),
            "nfev": info.get("NumRhsEvals", 0),      # Number of function evaluations
            "njev": info.get("NumJacEvals", 0),      # Number of Jacobian evaluations
            "nlu": info.get("NumLinSolvSetups", 0),  # Number of LU decompositions
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
