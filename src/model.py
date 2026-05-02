"""Immutable definition of the p‑Laplacian PDE."""
import numpy as np
from scipy.sparse import diags_array
from dataclasses import dataclass

@dataclass(frozen=True)
class PLaplacianModel:
    p: float = 2.5
    h: float = 1.0
    L: float = 1.0
    Nx: int = 1000
    epsilon: float = 1e-6

    @property
    def dx(self):
        return self.L / self.Nx

    @property
    def x(self):
        return np.linspace(0, self.L, self.Nx + 1)

    @property
    def sparsity(self):
        n_dim = self.Nx - 1
        return diags_array([np.ones(n_dim-1), np.ones(n_dim), np.ones(n_dim-1)],
                           offsets=(-1,0,1), shape=(n_dim,n_dim), format="csc")
