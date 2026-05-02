"""Single entry point for the p‑Laplacian suite (argparse)."""

import sys
import argparse
from src.plotter import run_simulation
from src.benchmark import benchmark_suite
from src.benchmark.benchmark_configs import benchmarks
from src.benchmark.benchmark_runner import run_benchmark_config
from src.model import PLaplacianModel
from src.solver import PLaplacianSolver

def warmup_jit():
    """Warm‑up Numba JIT to avoid timing distortion in benchmarks."""
    model = PLaplacianModel(p=2.5, Nx=100, epsilon=1e-6)
    solver = PLaplacianSolver(model)
    solver.solve([0.001], method="LSODA", sparse=True, rtol=1e-3, atol=1e-3)

def profile_model():
    import cProfile, pstats
    model = PLaplacianModel(p=6, h=1.0, Nx=1000, epsilon=1e-6)
    solver = PLaplacianSolver(model)

    def _run():
        solver.solve([0.05], method="CVODE", sparse=True, rtol=1e-6, atol=1e-6)

    profiler = cProfile.Profile()
    profiler.enable()
    _run()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(20)


def build_parser():
    parser = argparse.ArgumentParser(
        description="p‑Laplacian PDE Solver and Benchmarking Suite"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim_parser = subparsers.add_parser("simulate", help="Run a standard simulation")
    sim_parser.add_argument("--live", action="store_true", help="Enable real‑time GPU‑accelerated plotting")
    sim_parser.add_argument("--p", type=float, default=4.5, help="Nonlinearity index")
    sim_parser.add_argument("--Nx", type=int, default=5000, help="Grid resolution")
    sim_parser.add_argument("--epsilon", type=float, default=1e-6, help="Regularization parameter")
    sim_parser.add_argument("--tol", type=float, default=1e-6, help="Solver tolerance (rtol/atol)")
    sim_parser.add_argument("--method", type=str, default="LSODA", help="Solver backend method")

    bench_parser = subparsers.add_parser("benchmark", help="Run a custom benchmark grid")
    bench_parser.add_argument("--Nx", type=int, nargs="+",
                              default=[50, 200, 1000])
    bench_parser.add_argument("--p", type=float, nargs="+",
                              default=[1.5, 2.0, 3.0])
    bench_parser.add_argument("--epsilon", type=float, nargs="+",
                              default=[1e-6, 1e-3, 0.0])
    bench_parser.add_argument("--methods", nargs="+",
                              default=["BDF", "Radau", "LSODA", "RK45"])
    bench_parser.add_argument("--T", type=float, default=0.01,
                              help="Simulation end time")
    bench_parser.add_argument("--skip-error", action="store_true",
                              help="Do not compute reference L2 error")
    bench_parser.add_argument("--tol", type=float, nargs="+", default=[1e-3], help="Solver tolerances to test")
    trials_parser = subparsers.add_parser("trials", help="Run preset benchmark suite")
    trials_parser.add_argument("--all", action="store_true",
                               help="Run all available benchmarks")
    for cfg in benchmarks.values():
        trials_parser.add_argument(f"--{cfg.flag}", action="store_true",
                                   help=f"Run {cfg.name} benchmark")

    subparsers.add_parser("profile", help="Profile a fixed degenerate run")

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    print("Executing JIT warmup...")
    warmup_jit()
    print("JIT ready!")
    
    if args.command == "simulate":
        run_simulation(
            live=args.live,
            p=args.p,
            Nx=args.Nx,
            epsilon=args.epsilon,
            tol=args.tol,
            method=args.method
        )

    elif args.command == "benchmark":
        grid = {
            "method": args.methods,
            "sparse": [True, False],
            "p": args.p,
            "epsilon": args.epsilon,
            "Nx": args.Nx,
            "tol": args.tol
        }
        df = benchmark_suite(grid, T=args.T, compute_error=not args.skip_error)
        print(df.to_string(index=False))

    elif args.command == "trials":
        if args.all:
            selected = set(benchmarks.keys())
        else:
            selected = {name for name, cfg in benchmarks.items()
                        if getattr(args, cfg.flag.replace("-", "_"), False)}
        if not selected:
            print("No trial selected. Use --all or one of the trial flags.")
            sys.exit(1)
        for name in selected:
            cfg = benchmarks[name]
            print(f"\n{'='*60}\n Running {cfg.name} \n{'='*60}")
            run_benchmark_config(cfg)

    elif args.command == "profile":
        profile_model()
