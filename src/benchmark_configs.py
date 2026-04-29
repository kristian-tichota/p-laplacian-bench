"""Benchmark configuration dataclass and registry."""
from dataclasses import dataclass
from typing import Callable, Optional
import src.plotting as plots


@dataclass
class BenchmarkConfig:
    name: str               # human-readable identifier
    flag: str               # CLI flag (e.g. "sparsity")
    grid: dict              # parameter grid for benchmark_suite
    T: float = 0.05
    compute_error: bool = False
    plot_func: Optional[Callable] = None
    run_func: Optional[Callable] = None
    run_kwargs: Optional[dict] = None
    plot_filename: str = "benchmark.pdf"


# ── Registry of all benchmarks ─────────────────────────────────────
benchmarks = {
    "sparsity": BenchmarkConfig(
        name="sparsity",
        flag="sparsity",
        grid={
            "method": ["LSODA", "BDF"],
            "sparse": [True, False],
            "p": [2.5],
            "epsilon": [1e-6],
            "Nx": [50, 100, 200, 500, 1000, 1500, 2000],
            "tol": [1e-6],
        },
        compute_error=False,
        plot_func=plots.plot_sparsity_scaling,
        plot_filename="sparsity_scaling.pdf",
    ),
    "work_effort": BenchmarkConfig(
        name="work_effort",
        flag="work",
        grid={
            "method": ["LSODA", "BDF", "Radau"],
            "sparse": [True],
            "p": [2.5],
            "epsilon": [1e-6],
            "Nx": [1000],
            "tol": [1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
        },
        compute_error=True,
        plot_func=plots.plot_work_effort,
        plot_filename="work_effort.pdf",
    ),
    "epsilon": BenchmarkConfig(
        name="epsilon_sweep",
        flag="epsilon",
        grid={
            "method": ["LSODA", "BDF", "Radau"],
            "sparse": [True],
            "p": [2.5],
            "epsilon": [1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
            "Nx": [1000],
            "tol": [1e-6],
        },
        compute_error=False,
        plot_func=plots.plot_epsilon_sweep,
        plot_filename="epsilon_sweep.pdf",
    ),
    "psweep": BenchmarkConfig(
        name="p_sweep",
        flag="psweep",
        grid={
            "method": ["LSODA", "BDF", "Radau"],
            "sparse": [True],
            "p": [1.5, 1.8, 2.0, 2.5, 3.0, 4.0],
            "epsilon": [1e-4],
            "Nx": [1000],
            "tol": [1e-6],
        },
        compute_error=False,
        plot_func=plots.plot_p_sweep,
        plot_filename="p_sweep.pdf",
    ),
    "cvode_work": BenchmarkConfig(
        name="cvode_work_effort",
        flag="cvode-work",
        grid={
            "method": ["LSODA", "CVODE"],
            "sparse": [True],
            "p": [3],
            "epsilon": [1e-6],
            "Nx": [5000],
            "tol": [1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
        },
        compute_error=True,
        plot_func=plots.plot_cvode_work_effort,
        plot_filename="cvode_work_effort.pdf",
    ),
    "extreme_nx": BenchmarkConfig(
        name="extreme_nx",
        flag="extreme-nx",
        grid={
            "method": ["LSODA", "CVODE"],
            "sparse": [True],
            "p": [2.5],
            "epsilon": [1e-6],
            "Nx": [1000, 5000, 10000, 25000, 50000, 100000],
            "tol": [1e-6],
        },
        compute_error=False,
        plot_func=plots.plot_extreme_nx,
        plot_filename="extreme_nx_scaling.pdf",
    ),
    "extreme_p": BenchmarkConfig(
        name="extreme_p",
        flag="extreme-p",
        grid={
            "method": ["LSODA", "CVODE"],
            "sparse": [True],
            "p": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            "epsilon": [1e-6],
            "Nx": [1000],
            "tol": [1e-6],
        },
        compute_error=False,
        plot_func=plots.plot_extreme_p,
        plot_filename="extreme_p_scaling.pdf",
    ),
    "singular_epsilon": BenchmarkConfig(
        name="singular_epsilon",
        flag="singular-epsilon",
        grid={
            "method": ["LSODA", "CVODE"],
            "sparse": [True],
            "p": [1.25, 1.1, 1.05, 1.01],
            "epsilon": [1e-6, 1e-8, 1e-10, 1e-12, 1e-14, 1e-30],
            "Nx": [200],
            "tol": [1e-13],
        },
        T=0.05,
        compute_error=False,
        run_kwargs={"check_propagation": True},
        plot_func=plots.plot_singular_epsilon,
        plot_filename="stability_matrix.pdf",
    ),
}
