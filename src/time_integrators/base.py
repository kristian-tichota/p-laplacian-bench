"""Abstract solver backend interface."""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Protocol

import numpy as np


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


class SolverIntegrator(Protocol):
    """Every backend must implement this call signature."""

    def solve(
        self,
        t_eval: np.ndarray,
        y0: np.ndarray,
        rhs: Callable,
        sparsity: Optional[np.ndarray] = None,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        **kwargs,
    ) -> SolverResult: ...
