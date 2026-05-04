import numpy as np
from scipy.sparse import spmatrix
import scipy.sparse.linalg as spla
from mpi4py import MPI
import dolfinx
from dolfinx import mesh, fem, la
import ufl

from .base import SpatialDiscretization

class FEniCSxDiscretization(SpatialDiscretization):
    def __init__(self, p: float, h: float, L: float, Nx: int, epsilon: float, mesh_in: dolfinx.mesh.Mesh | None = None):
        self.p = p
        self.h = h
        self.L = L
        self.Nx = Nx          
        self.epsilon = epsilon

        # 1. Mesh Initialization (Supports injection of adapted meshes)
        if mesh_in is None:
            self._mesh = mesh.create_interval(MPI.COMM_WORLD, Nx, [0.0, L])
        else:
            self._mesh = mesh_in
            
        self._V = fem.functionspace(self._mesh, ("Lagrange", 1))

        left_dofs = fem.locate_dofs_geometrical(self._V, lambda x: np.isclose(x[0], 0.0))
        right_dofs = fem.locate_dofs_geometrical(self._V, lambda x: np.isclose(x[0], L))
        self._bcs_dofs = np.concatenate((left_dofs, right_dofs)).astype(np.int32)

        self._dx = ufl.Measure("dx", domain=self._mesh)

        # 2. Native Mass Lumping via UFL
        u_trial = ufl.TrialFunction(self._V)
        v_test = ufl.TestFunction(self._V)
        
        # Use vertex quadrature to force a diagonal mass matrix
        dx_lumped = ufl.Measure("dx", domain=self._mesh, 
                                metadata={"quadrature_rule": "vertex", "quadrature_degree": 1})
        a = u_trial * v_test * dx_lumped
        M_fenics = fem.assemble_matrix(fem.form(a))
        
        # Extract Sparsity Pattern directly from dolfinx structures
        M_scipy = M_fenics.to_scipy()
        self._sparsity = M_scipy.copy()
        self._sparsity.data[:] = 1.0
        
        M_mod = M_scipy.tolil()
        for dof in self._bcs_dofs:
            M_mod[dof, :] = 0.0
            M_mod[:, dof] = 0.0
            M_mod[dof, dof] = 1.0
            
        self._M_inv_diag = 1.0 / M_mod.diagonal()

        # 3. UFL-Based Nonlinear Residual (Handles non-uniform element sizes)
        self._u_func = fem.Function(self._V)
        grad_u = ufl.grad(self._u_func)
        D = (ufl.dot(grad_u, grad_u) + self.epsilon**2) ** ((self.p - 2) / 2)
        self._F_res = D * ufl.dot(grad_u, ufl.grad(v_test)) * self._dx
        self._residual_form = fem.form(self._F_res)

        self._x_full = self._mesh.geometry.x[:, 0].copy()

    @property
    def state_size(self) -> int:
        return self._V.dofmap.index_map.size_global

    def get_initial_state(self) -> np.ndarray:
        if hasattr(self, "_initial_state"):
            return self._initial_state.copy()
            
        y0 = np.zeros(self.state_size, dtype=float)
        y0[0] = self.h       
        y0[-1] = 0.0         
        return y0

    def compute_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        self._u_func.x.array[:] = state.ravel()
        R = fem.assemble_vector(self._residual_form)
        R.scatter_reverse(la.InsertMode.add)
        
        # Access and modify the underlying NumPy array directly
        R.array[self._bcs_dofs] = 0.0
        
        # Apply inverse mass matrix natively as an O(N) vector multiplication
        dydt = -R.array * self._M_inv_diag
        return dydt

    @property
    def sparsity_pattern(self) -> spmatrix:
        return self._sparsity

    def get_full_solution(self, state: np.ndarray) -> np.ndarray:
        return state

    def get_node_coordinates(self) -> np.ndarray:
        return self._x_full

    def compute_l2_error(self, state: np.ndarray, ref_state: np.ndarray) -> float:
        cell_size = self._mesh.h(0, 0)[0]
        return np.sqrt(cell_size * np.sum((state - ref_state) ** 2))

    def refine(self, solution: np.ndarray, fraction: float = 0.2, min_cells: int = 0) -> "FEniCSxDiscretization":
        """Adaptive mesh refinement based on squared gradient indicator."""
        self._u_func.x.array[:] = solution
        grad_u = ufl.grad(self._u_func)
        
        ind_form = fem.form(grad_u**2 * self._dx)
        indicator = fem.assemble_scalar(ind_form).array

        n_cells = len(indicator)
        n_refine = max(min_cells, int(np.ceil(fraction * n_cells)))
        if n_refine == 0:
            return self

        threshold = np.sort(indicator)[-n_refine]
        marked = indicator >= threshold

        new_mesh, _, _ = mesh.refine_plaza(self._mesh, marked)

        u_old = fem.Function(self._V)
        u_old.x.array[:] = solution
        
        V_new = fem.functionspace(new_mesh, ("Lagrange", 1))
        u_new = fem.Function(V_new)
        u_new.interpolate(u_old)
        u_new.x.scatter_forward()

        new_disc = FEniCSxDiscretization(
            p=self.p, h=self.h, L=self.L, Nx=self.Nx, 
            epsilon=self.epsilon, mesh_in=new_mesh
        )
        new_disc._initial_state = u_new.x.array.copy()
        return new_disc
