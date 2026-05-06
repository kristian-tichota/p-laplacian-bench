"""
FEniCSx‑based FEM discretisation of the 1‑D p‑Laplacian.
"""

from __future__ import annotations

import dolfinx
import dolfinx.fem.petsc
import numpy as np
import ufl
from basix.ufl import element
from dolfinx import fem, la, mesh
from mpi4py import MPI
from scipy.sparse import csr_array, diags, lil_array, spmatrix

from .base import SpatialDiscretization


class FEniCSxDiscretization(SpatialDiscretization):
    """
    Finite‑element discretisation for ∂_t u = ∇·( D ∇ u )
    with D = (|∇u|² + ε²)^{(p-2)/2}, constant ε, uniform mesh.
    """

    def __init__(
        self,
        p: float,
        h: float,
        L: float = 1.0,
        Nx: int = 1000,
        epsilon: float = 1e-6,
        degree: int = 1,
        mesh_in: dolfinx.mesh.Mesh | None = None,
    ):
        self.p = p
        self.h = h
        self.L = L
        self.epsilon = epsilon
        self.degree = degree

        # --- Mesh and function space ---
        if mesh_in is None:
            self._mesh = mesh.create_interval(MPI.COMM_WORLD, Nx, [0.0, L])
        else:
            self._mesh = mesh_in
        self.Nx = self._mesh.topology.index_map(self._mesh.topology.dim).size_local

        e = element("Lagrange", self._mesh.basix_cell(), degree, shape=())
        self._V = fem.functionspace(self._mesh, e)
        n_dof = self._V.dofmap.index_map.size_global
        self._sparsity = diags(
            [np.ones(n_dof - 1), np.ones(n_dof), np.ones(n_dof - 1)],
            offsets=[-1, 0, 1],
            format="csc",
        )

        # --- Boundary dofs ---
        self._left_dofs = fem.locate_dofs_geometrical(
            self._V, lambda x: np.isclose(x[0], 0.0)
        )
        self._right_dofs = fem.locate_dofs_geometrical(
            self._V, lambda x: np.isclose(x[0], L)
        )
        self._bcs_dofs = np.concatenate((self._left_dofs, self._right_dofs)).astype(
            np.int32
        )

        # --- Lumped mass matrix (vertex quadrature → diagonal) ---
        u = ufl.TrialFunction(self._V)
        v = ufl.TestFunction(self._V)
        dx_lumped = ufl.dx(
            domain=self._mesh,
            metadata={"quadrature_rule": "vertex", "quadrature_degree": 1},
        )
        M_form = fem.form(u * v * dx_lumped)
        M = fem.assemble_matrix(M_form).to_scipy()
        # Sparsity pattern for finite‑difference fallback
        self._sparsity = csr_array(M).copy()
        self._sparsity.data[:] = 1.0

        # Apply Dirichlet BCs: rows/cols for boundary dofs become identity
        M_mod = M.tolil()
        for dof in self._bcs_dofs:
            M_mod[dof, :] = 0.0
            M_mod[:, dof] = 0.0
            M_mod[dof, dof] = 1.0
        self._M_inv_diag = 1.0 / M_mod.diagonal()  # O(N) inverse

        # --- Nonlinear residual F(u;v) = ∫ D ∇u·∇v dx ---
        self._u_func = fem.Function(self._V)
        grad_u = ufl.grad(self._u_func)
        D = (ufl.dot(grad_u, grad_u) + self.epsilon**2) ** ((self.p - 2) / 2)
        self._F_res = D * ufl.dot(grad_u, ufl.grad(v)) * ufl.dx
        self._residual_form = fem.form(self._F_res)

        # --- Analytical Jacobian: dF/du (optional, not used by default) ---
        du = ufl.TrialFunction(self._V)
        dF = ufl.derivative(self._F_res, self._u_func, du)
        self._jac_res_form = fem.form(dF)

        # Node coordinates
        self._x_full = self._mesh.geometry.x[:, 0].copy()

        self._bcs_cached = self.get_dirichlet_bcs()
        self._J_petsc = dolfinx.fem.petsc.create_matrix(self._jac_res_form)

    # ── SpatialDiscretization interface ──────────────────────────
    @property
    def mesh(self):
        return self._mesh

    @property
    def function_space(self):
        return self._V

    def get_dirichlet_bcs(self) -> list:
        """Return DirichletBC objects for left (value h) and right (value 0)."""
        import dolfinx.fem

        bc_left = dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(self._mesh, self.h), self._left_dofs, self._V
        )
        bc_right = dolfinx.fem.dirichletbc(
            dolfinx.fem.Constant(self._mesh, 0.0), self._right_dofs, self._V
        )
        return [bc_left, bc_right]

    @property
    def state_size(self) -> int:
        return self._V.dofmap.index_map.size_global

    def get_initial_state(self) -> np.ndarray:
        y0 = np.zeros(self.state_size)
        y0[self._left_dofs] = self.h
        y0[self._right_dofs] = 0.0
        return y0

    def compute_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """RHS: ẏ = -M⁻¹ R(y)  (mass lumped, O(N) per step)."""
        self._u_func.x.array[:] = state.ravel()
        R = fem.assemble_vector(self._residual_form)
        R.scatter_reverse(la.InsertMode.add)
        R.array[self._bcs_dofs] = 0.0
        return -R.array * self._M_inv_diag

    @property
    def sparsity_pattern(self) -> spmatrix:
        """Tridiagonal sparsity pattern (for finite‑difference fallback)."""
        return self._sparsity

    def get_full_solution(self, state: np.ndarray) -> np.ndarray:
        return state.ravel()

    def get_node_coordinates(self) -> np.ndarray:
        return self._x_full

    def compute_l2_error(self, state: np.ndarray, ref_state: np.ndarray) -> float:
        cell_size = self._mesh.h(0, 0)[0]
        return np.sqrt(cell_size * np.sum((state.ravel() - ref_state.ravel()) ** 2))

    # ── Optional analytical Jacobian (not used by default) ────────

    def compute_jac_rhs(self, t: float, state: np.ndarray) -> spmatrix:
        """
        Exact Jacobian of `compute_rhs` w.r.t. `state` using native PETSc assembly.
        Returns a sparse matrix (tridiagonal).
        """
        import dolfinx.fem.petsc
        from scipy.sparse import csr_matrix

        self._u_func.x.array[:] = state.ravel()

        # Zero out the preallocated PETSc matrix
        self._J_petsc.zeroEntries()

        # Assemble in-place with boundary conditions applied natively by PETSc
        dolfinx.fem.petsc.assemble_matrix(
            self._J_petsc, self._jac_res_form, bcs=self._bcs_cached
        )
        self._J_petsc.assemble()

        # Extract CSR data directly from the PETSc matrix
        indptr, indices, data = self._J_petsc.getValuesCSR()
        J_res_sp = csr_matrix((data, indices, indptr), shape=self._J_petsc.getSize())

        # Apply lumped mass matrix inverse for the ODE formulation (J = -M^-1 * J_res)
        J_rhs = -diags(self._M_inv_diag) @ J_res_sp

        return J_rhs.tocsc()

    def compute_jac_banded(self, t: float, state: np.ndarray) -> np.ndarray:
        """Return the tridiagonal Jacobian in banded (3, N) format for LSODA.
        Assumes the Jacobian is strictly tridiagonal (true for 1D linear elements)."""
        J = self.compute_jac_rhs(t, state)  # CSC matrix
        n = J.shape[0]
        banded = np.zeros((3, n))
        # Extract three diagonals from the sparse matrix
        banded[0, 1:] = J.diagonal(1)  # super‑diagonal
        banded[1, :] = J.diagonal(0)  # main diagonal
        banded[2, :-1] = J.diagonal(-1)  # sub‑diagonal
        return banded
