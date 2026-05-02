"""Abstract solver backend interface."""
from typing import Protocol, Callable, Optional, Dict
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class SolverStats:
    success: bool
    message: str = ""
    nfev: int = 0
    njev: int = 0
    nlu: int = 0

@dataclass(frozen=True)
class SolverResult:
    """Standardised return from any backend."""
    data: Dict[float, np.ndarray]
    stats: SolverStats

class SolverBackend(Protocol):
    """Every backend must implement this call signature."""
    def solve(
        self,
        t_eval: np.ndarray,
        y0: np.ndarray,
        rhs: Callable,
        sparsity: Optional[np.ndarray] = None,
        rtol: float = 1e-5,
        atol: float = 1e-6,
        **kwargs
    ) -> SolverResult:
        ...
