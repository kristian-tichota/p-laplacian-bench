import time
import itertools
import pandas as pd
import numpy as np
from .solver import PLaplacianSolver


def compute_l2_error(test_solution, reference_solution, dx):
    """Computes the discrete L2 norm between two solutions."""
    return np.sqrt(np.sum((test_solution - reference_solution) ** 2) * dx)


def generate_reference_solution(p, epsilon, Nx, T):
    """Computes a highly accurate ground-truth solution using Radau."""
    solver = PLaplacianSolver(p=p, h=1.0, Nx=Nx, epsilon=epsilon)

    # Force strict tolerances and use Radau with the sparse Jacobian
    data, stats = solver.solve([T], method="CVODE", sparse=True, rtol=1e-13, atol=1e-13, max_steps=1000000)

    if not stats or not stats.get("success", False):
        raise RuntimeError(f"Reference failed: {stats.get('message', 'Unknown error')}")

    return data[T]


def run_single_benchmark(params, ref_u=None, T=0.05, trials=3):
    """Executes a solver run and compares it against the reference array (if provided)."""
    method, sparse, p, epsilon, Nx, tol = params

    if sparse and method == "RK45":
        return {"duration_s": np.nan, "status": "Skipped (Unsupported)"}

    solver = PLaplacianSolver(p=p, h=1.0, Nx=Nx, epsilon=epsilon)
    data = None
    stats = None

    try:
        durations = []
        for _ in range(trials):
            start = time.perf_counter()
            data, stats = solver.solve([T], method=method, sparse=sparse, rtol=tol, atol=tol)
            durations.append(time.perf_counter() - start)

        best_time = min(durations)

        if not data or not stats or not stats.get("success", False):
            err_msg = stats.get("message", "Unknown Solver Failure") if stats else "No stats returned"
            return {"duration_s": best_time, "status": f"Failed: {err_msg}"}

        test_u = data.get(T)
        if test_u is None:
            return {"duration_s": best_time, "status": "Failed: Target time T not reached"}

        if ref_u is not None:
            error_l2 = compute_l2_error(test_u, ref_u, solver.dx)
        else:
            error_l2 = np.nan

        return {
            "duration_s": best_time,
            "status": "Success",
            "sparse": str(sparse),
            "nfev": stats.get("nfev", 0),
            "njev": stats.get("njev", 0),
            "nlu": stats.get("nlu", 0),
            "error_l2": error_l2,
        }

    except Exception as e:
        return {"duration_s": np.nan, "status": f"Failed: {type(e).__name__}"}

def benchmark_suite(param_grid, T=0.05, compute_error=True):
    """Iterates through the parameter grid and compiles results."""
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

    results = []
    total_runs = len(experiments)

    print(f"Starting benchmark suite with {total_runs} configurations...\n")

    # Warmup run (JIT compilation overhead removal)
    PLaplacianSolver(p=2.5, h=1.0, Nx=50, epsilon=1e-6).solve([0.001], method="BDF", sparse=True)

    reference_cache = {}

    for idx, exp in enumerate(experiments, 1):
        p, eps, Nx = exp["p"], exp["epsilon"], exp["Nx"]
        tol = exp.get("tol", 1e-3) 
        
        ref_key = (p, eps, Nx)
        ref_u = None

        if compute_error:
            if ref_key not in reference_cache:
                try:
                    print(f"  -> Generating ground truth for p={p}, eps={eps}, Nx={Nx}...")
                    reference_cache[ref_key] = generate_reference_solution(p, eps, Nx, T)
                except Exception as e:
                    print(f"  -> Ground truth generation failed: {e}")
                    reference_cache[ref_key] = None
            
            ref_u = reference_cache[ref_key]

            if ref_u is None:
                exp.update({"duration_s": np.nan, "status": "Failed: No Reference", "error_l2": np.nan})
                results.append(exp)
                continue

        params = (exp["method"], exp["sparse"], p, eps, Nx, tol)
        metrics = run_single_benchmark(params, ref_u=ref_u, T=T)

        exp.update(metrics)
        results.append(exp)

        s_status = metrics.get("sparse_status", str(exp["sparse"]))
        duration_val = exp.get("duration_s", np.nan)
        duration_str = f"{duration_val:.4f}s" if not np.isnan(duration_val) else "N/A"
        
        print(
            f"[{idx:2d}/{total_runs}] {exp['method']:<5} | "
            f"Sparse: {s_status:<5} | tol: {tol:<7.1e} | Nx: {Nx:<4} | "
            f"Err: {exp.get('error_l2', np.nan):.2e} | "
            f"Time: {duration_str:>7} | "
            f"fev: {exp.get('nfev', 0):<4} | gev: {exp.get('njev', 0):<3} -> {exp['status']}"
        )
        
    return pd.DataFrame(results)
