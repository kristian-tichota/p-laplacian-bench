import numpy as np
from numba import njit


@njit(fastmath=True)
def fast_p_laplacian_rhs(t, u, p, dx, h, epsilon=1e-6):
    N = len(u)
    dudt = np.empty(N)

    # 1. Left boundary flux (h to u[0])
    grad = (u[0] - h) / dx
    flux_in = (grad**2 + epsilon**2) ** ((p - 2) / 2) * grad

    # 2. Interior fluxes
    for i in range(N - 1):
        grad = (u[i + 1] - u[i]) / dx
        flux_out = (grad**2 + epsilon**2) ** ((p - 2) / 2) * grad

        dudt[i] = (flux_out - flux_in) / dx
        flux_in = flux_out  # The out-flux of cell i becomes the in-flux of cell i+1

    # 3. Right boundary flux (u[-1] to 0.0)
    grad = (0.0 - u[N - 1]) / dx
    flux_out = (grad**2 + epsilon**2) ** ((p - 2) / 2) * grad
    dudt[N - 1] = (flux_out - flux_in) / dx

    return dudt
