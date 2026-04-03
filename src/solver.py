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
        jac_val = None
        if method == "LSODA":
            if sparse:
                kwargs["lband"] = 1
                kwargs["uband"] = 1
            else:
                jac_val = self.sparsity if sparse else None

        sol = solve_ivp(
            fun=fast_p_laplacian_rhs,
            t_span=(0, max(times)),
            y0=u0_interior,
            method=method,
            jac_sparsity=jac_val,
            t_eval=sorted(times),
            args=(self.p, self.dx, self.h, self.epsilon),
            rtol=rtol,
            atol=atol,
            **kwargs,
        )

        reconstructed_data = self._reconstruct(sol.t, sol.y)

        stats = {
            "nfev": sol.nfev,
            "njev": sol.njev,
            "nlu": sol.nlu,
            "success": sol.success,
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
