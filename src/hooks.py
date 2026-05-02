"""Solver observer interface."""
from typing import Protocol
import numpy as np

class SolverHook(Protocol):
    def __call__(self, t: float, y: np.ndarray) -> None:
        """Called inside the RHS with current integrator time and state."""
        ...
