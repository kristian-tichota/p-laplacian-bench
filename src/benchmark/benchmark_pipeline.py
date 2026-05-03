"""Modular benchmarking pipeline."""
import time
import itertools
import numpy as np
import pandas as pd
from dataclasses import fields
from typing import Dict, Optional
from ..time_integrators.base import SolverStats
from ..model import PLaplacianModel
from ..solver import PLaplacianSolver
from ..config import SimulationConfig

class BenchmarkPipeline:
    def __init__(self):
        self.reference_cache: Dict[tuple, Optional[np.ndarray]] = {}

    def _generate_reference(self, model: PLaplacianModel, T: float) -> np.ndarray:
        """High‑accuracy reference using CVODE."""
        key = (model.p, model.epsilon, model.Nx)
        if key not in self.reference_cache:
            ref_config = SimulationConfig(
                p=model.p, epsilon=model.epsilon, Nx=model.Nx,
                method="CVODE", rtol=1e-13, atol=1e-13, T=T
            )
            ref_solver = PLaplacianSolver(model, ref_config)
            data, stats = ref_solver.solve([T])
            if not stats.success:
                self.reference_cache[key] = None
                raise RuntimeError(f"Reference failed: {stats.message}")
            self.reference_cache[key] = data[T]
        return self.reference_cache[key]

    def run_experiment(self, config: SimulationConfig, compute_error: bool = True,
                       check_propagation: bool = False) -> dict:
        model = config.to_model()
        solver = PLaplacianSolver(model, config)
        T = config.T

        t0 = time.perf_counter()
        data, stats = solver.solve([T], check_propagation=check_propagation)
        wall = time.perf_counter() - t0

        if not stats.success:
            return {
                "method": config.method, "sparse": config.sparse,
                "p": config.p, "epsilon": config.epsilon, "Nx": config.Nx,
                "tol": config.rtol, "duration_s": wall,
                "status": f"Failed: {stats.message}"
            }

        err = np.nan
        if compute_error:
            ref = self._generate_reference(model, T)
            if ref is not None:
                dx = model.dx
                err = np.sqrt(np.sum((data[T] - ref)**2) * dx)

        return {
            "method": config.method, "sparse": config.sparse,
            "p": config.p, "epsilon": config.epsilon, "Nx": config.Nx,
            "tol": config.rtol, "duration_s": wall,
            "status": "Success",
            "nfev": stats.nfev, "njev": stats.njev, "nlu": stats.nlu,
            "error_l2": err,
        }

    def run_grid(self, param_grid: dict, T: float = 0.05, compute_error: bool = True) -> pd.DataFrame:
        keys, values = zip(*param_grid.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Gather default values from SimulationConfig fields
        default_values = {
            f.name: f.default
            for f in fields(SimulationConfig)
            if f.default is not None
        }

        total_runs = len(experiments)
        results = []

        for i, exp in enumerate(experiments, start=1):
            # Start with all defaults, then override with experiment values
            config_kwargs = {**default_values}
            check_prop = exp.pop("check_propagation", False)

            # Map known experiment keys to config fields
            # 'tol' -> rtol & atol, 'method' & 'sparse' directly
            for key, value in exp.items():
                if key == "tol":
                    config_kwargs["rtol"] = value
                    config_kwargs["atol"] = value
                elif key in config_kwargs:   # only take known fields
                    config_kwargs[key] = value
                # 'T' may be in experiment; use it to override, else keep default
            # Use T from experiment if provided, else fallback to the method argument
            config_kwargs["T"] = exp.get("T", T)

            config = SimulationConfig(**config_kwargs)

            res = self.run_experiment(config, compute_error=compute_error,
                                      check_propagation=check_prop)
            results.append(res)

            method = config.method
            params_str = ", ".join(f"{k}={v}" for k, v in exp.items() if k != "method")
            status = res.get("status", "")
            duration = res.get("duration_s", 0.0)
            if "Success" in status:
                stats_str = (
                    f"time={duration:.3f}s, "
                    f"nfev={res.get('nfev', 0)}, "
                    f"njev={res.get('njev', 0)}, "
                    f"nlu={res.get('nlu', 0)}"
                )
                if compute_error and pd.notna(res.get("error_l2")):
                    stats_str += f", err={res['error_l2']:.2e}"
            else:
                stats_str = f"FAILED ({status})"
            print(f"[{i}/{total_runs}] {method:5} | {params_str} | {stats_str}")

        return pd.DataFrame(results)
