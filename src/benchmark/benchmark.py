"""Benchmark convenience wrapper – delegates to pipeline."""
from .benchmark_pipeline import BenchmarkPipeline

def benchmark_suite(param_grid, T=0.05, compute_error=True, run_kwargs=None):
    return BenchmarkPipeline().run_grid(param_grid, T, compute_error)
