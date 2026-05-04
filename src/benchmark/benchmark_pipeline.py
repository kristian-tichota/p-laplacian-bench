"""Modular benchmarking pipeline, discretisation‑agnostic."""

import itertools
import time
from dataclasses import fields
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import SimulationConfig
from ..solver import PLaplacianSolver


class BenchmarkPipeline:
    def __init__(self):
        self.reference_cache: Dict[tuple, Optional[np.ndarray]] = {}

    def _generate_reference(self, config: SimulationConfig) -> np.ndarray:
        """High‑accuracy reference using same discretisation type."""
        key = (config.discretization_type, config.p, config.epsilon, config.Nx)
        if key not in self.reference_cache:
            ref_config = SimulationConfig(
                p=config.p,
                h=config.h,
                L=config.L,
                Nx=config.Nx,
                epsilon=config.epsilon,
                discretization_type=config.discretization_type,
                method=config.ref_method,
                rtol=config.ref_rtol,
                atol=config.ref_atol,
                sparse=True,
                T=config.T,
            )
            disc = ref_config.to_discretization()
            solver = PLaplacianSolver(disc, ref_config)
            data, stats = solver.solve([config.T])
            if not stats.success:
                self.reference_cache[key] = None
                raise RuntimeError(f"Reference failed: {stats.message}")
            self.reference_cache[key] = data[config.T]
        return self.reference_cache[key]

    def run_experiment(
        self,
        config: SimulationConfig,
        compute_error: bool = True,
        check_propagation: bool = False,
    ) -> dict:
        # Create the discretisation for this experiment
        disc = config.to_discretization()
        solver = PLaplacianSolver(disc, config)

        t0 = time.perf_counter()
        data, stats = solver.solve([config.T], check_propagation=check_propagation)
        wall = time.perf_counter() - t0

        if not stats.success:
            return {
                "method": config.method,
                "sparse": config.sparse,
                "p": config.p,
                "epsilon": config.epsilon,
                "Nx": config.Nx,
                "tol": config.rtol,
                "duration_s": wall,
                "status": f"Failed: {stats.message}",
            }

        err = np.nan
        if compute_error:
            ref = self._generate_reference(config)
            if ref is not None:
                # Use the discretisation’s own error metric
                err = disc.compute_l2_error(data[config.T], ref)

        return {
            "method": config.method,
            "sparse": config.sparse,
            "p": config.p,
            "epsilon": config.epsilon,
            "Nx": config.Nx,
            "tol": config.rtol,
            "duration_s": wall,
            "status": "Success",
            "nfev": stats.nfev,
            "njev": stats.njev,
            "nlu": stats.nlu,
            "error_l2": err,
        }

    def run_grid(
        self, param_grid: dict, T: float = 0.05, compute_error: bool = True
    ) -> pd.DataFrame:
        keys, values = zip(*param_grid.items())
        experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

        # Gather defaults from SimulationConfig fields
        default_values = {
            f.name: f.default for f in fields(SimulationConfig) if f.default is not None
        }

        total_runs = len(experiments)
        results = []

        for i, exp in enumerate(experiments, start=1):
            config_kwargs = {**default_values}
            check_prop = exp.pop("check_propagation", False)

            # Map experiment keys to config fields
            for key, value in exp.items():
                if key == "tol":
                    config_kwargs["rtol"] = value
                    config_kwargs["atol"] = value
                elif key in config_kwargs:
                    config_kwargs[key] = value
            config_kwargs["T"] = exp.get("T", T)

            config = SimulationConfig(**config_kwargs)

            res = self.run_experiment(
                config, compute_error=compute_error, check_propagation=check_prop
            )
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
