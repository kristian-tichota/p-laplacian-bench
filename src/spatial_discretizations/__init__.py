"""Spatial discretisation factory."""
from .base import SpatialDiscretization
from .fdm import FDMDiscretization
from .fem_fenicsx import FEniCSxDiscretization


def create_discretization(config) -> SpatialDiscretization:
    """`config` is a SimulationConfig (see config.py)."""
    disc_type = getattr(config, "discretization_type", "fdm").lower()
    if disc_type == "fdm":
        return FDMDiscretization(
            p=config.p, h=config.h, L=config.L,
            Nx=config.Nx, epsilon=config.epsilon
        )
    elif disc_type in ("fem", "fenicsx"):
        return FEniCSxDiscretization(
            p=config.p, h=config.h, L=config.L,
            Nx=config.Nx, epsilon=config.epsilon
        )
    else:
        raise ValueError(f"Unknown discretization type: {disc_type}")
