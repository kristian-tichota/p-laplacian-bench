"""Orchestrate benchmarks: run, print, plot."""
import itertools
import numpy as np
import pandas as pd
from src.benchmark import benchmark_suite, run_single_benchmark, generate_reference_solution
from src.benchmark_configs import BenchmarkConfig, benchmarks


def run_benchmark_config(config: BenchmarkConfig) -> pd.DataFrame:
    """Run a single benchmark configuration (grid or special function)."""
    df = benchmark_suite(config.grid, T=config.T, compute_error=config.compute_error)

    # Print summary
    print_cols = [c for c in ["method", "tol", "Nx", "p", "epsilon",
                              "duration_s", "status"] if c in df.columns]
    print(f"\n--- {config.name} Results ---")
    print(df[print_cols].to_string(index=False))

    # Plot if a function is provided
    if config.plot_func:
        config.plot_func(df)

    return df
