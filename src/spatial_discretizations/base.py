"""Abstract spatial discretization interface."""

from abc import ABC, abstractmethod

import numpy as np
from scipy.sparse import spmatrix


class SpatialDiscretization(ABC):
    """Protocol that every spatial discretization must implement."""

    @property
    @abstractmethod
    def state_size(self) -> int:
        """Number of ODE variables (internal degrees of freedom)."""
        ...

    @abstractmethod
    def get_initial_state(self) -> np.ndarray:
        """Initial condition vector of length state_size."""
        ...

    @abstractmethod
    def compute_rhs(self, t: float, state: np.ndarray) -> np.ndarray:
        """Right‑hand side f(t, y) of the ODE system y' = f(t, y)."""
        ...

    @property
    @abstractmethod
    def sparsity_pattern(self) -> spmatrix:
        """
        Sparse matrix holding the structural pattern of the Jacobian ∂f/∂y.
        Return None if a dense Jacobian is acceptable.
        """
        ...

    @abstractmethod
    def get_full_solution(self, state: np.ndarray) -> np.ndarray:
        """
        Map the state vector to the full spatial solution (including
        Dirichlet boundaries) on the mesh nodes.
        """
        ...

    @abstractmethod
    def get_node_coordinates(self) -> np.ndarray:
        """Spatial coordinates of the mesh nodes (for plotting)."""
        ...

    @abstractmethod
    def compute_l2_error(self, state: np.ndarray, ref_state: np.ndarray) -> float:
        """Compute the discrete L² error between two state vectors on the
        mesh used by this discretization."""
        ...
