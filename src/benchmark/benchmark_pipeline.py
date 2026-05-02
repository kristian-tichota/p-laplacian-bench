"""Modular benchmarking pipeline."""
import time
import itertools
import numpy as np
import pandas as pd
from typing import Callable, Dict, Optional, Any
from ..backends.base import SolverStats
from ..model import PLaplacianModel
from ..solver import PLaplacianSolver

class BenchmarkPipeline:
    def __init__(self, solver_factory: Callable[..., PLaplacianSolver] = None):
        self.solver_factory = solver_factory or (lambda model: PLaplacianSolver(model))
        self.reference_cache: Dict[tuple, Optional[np.ndarray]] = {}

    def _generate_reference(self, model: PLaplacianModel, T: float) -> np.ndarray:
        """High‑accuracy reference using CVODE."""
        key = (model.p, model.epsilon, model.Nx)
        if key not in self.reference_cache:
            ref_solver = PLaplacianSolver(model)
            data, stats = ref_solver.solve([T], method="CVODE", sparse=True, rtol=1e-13, atol=1e-13)
            if not stats.success:
                self.reference_cache[key] = None
                raise RuntimeError(f"Reference failed: {stats.message}")
            self.reference_cache[key] = data[T]
        return self.reference_cache[key]

    def run_experiment(self, config: dict, T: float, compute_error: bool = True) -> dict:
        model = PLaplacianModel(p=config["p"], epsilon=config["epsilon"], Nx=config["Nx"])
        solver = self.solver_factory(model)
        method = config["method"]
        sparse = config["sparse"]
        tol = config.get("tol", 1e-3)
        check_prop = config.get("check_propagation", False)

        t0 = time.perf_counter()
        data, stats = solver.solve([T], method=method, sparse=sparse, rtol=tol, atol=tol, check_propagation=check_prop)
        wall = time.perf_counter() - t0

        if not stats.success:
            return {**config, "duration_s": wall, "status": f"Failed: {stats.message}"}

        err = np.nan
        if compute_error:
            ref = self._generate_reference(model, T)
            if ref is not None:
                dx = model.dx
                err = np.sqrt(np.sum((data[T] - ref)**2) * dx)

        return {
            **config,
            "duration_s": wall,
            "status": "Success",
            "nfev": stats.nfev,
            "njev": stats.njev,
            "nlu": stats.nlu,
            "error_l2": err,
        }

    def run_grid(self, param_grid: dict, T: float = 0.05, compute_error: bool = True) -> pd.DataFrame:
        keys, values = zip(*param_grid.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        total_runs = len(experiments)
        results = []
        
        for i, exp in enumerate(experiments, start=1):
            res = self.run_experiment(exp, T, compute_error)
            results.append(res)
            
            method = exp.get("method", "Unknown")
            
            params_str = ", ".join(f"{k}={v}" for k, v in exp.items() if k != "method")
            
            status = res.get("status", "")
            duration = res.get("duration_s", 0.0)
            
            if "Success" in status:
                stats = (
                    f"time={duration:.3f}s, "
                    f"nfev={res.get('nfev', 0)}, "
                    f"njev={res.get('njev', 0)}, "
                    f"nlu={res.get('nlu', 0)}"
                )
                if compute_error and pd.notna(res.get("error_l2")):
                    stats += f", err={res['error_l2']:.2e}"
            else:
                stats = f"FAILED ({status})"
                
            print(f"[{i}/{total_runs}] {method:5} | {params_str} | {stats}")
            
        return pd.DataFrame(results)
