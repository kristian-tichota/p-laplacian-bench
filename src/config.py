from dataclasses import dataclass
from typing import Optional
from .model import PLaplacianModel

@dataclass(frozen=True)
class SimulationConfig:
    # Physical parameters
    p: float = 2.5
    h: float = 1.0
    L: float = 1.0
    Nx: int = 1000
    epsilon: float = 1e-6

    # Time integration
    T: float = 0.05

    # Solver options
    method: str = "LSODA"
    sparse: bool = True
    rtol: float = 1e-6
    atol: float = 1e-6

    # Optional reference solver (for error computation)
    ref_method: str = "CVODE"
    ref_rtol: float = 1e-13
    ref_atol: float = 1e-13

    def to_model(self) -> PLaplacianModel:
        """Create the physics part as a PLaplacianModel."""
        return PLaplacianModel(
            p=self.p, h=self.h, L=self.L,
            Nx=self.Nx, epsilon=self.epsilon
        )
