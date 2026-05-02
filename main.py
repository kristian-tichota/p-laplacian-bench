import argparse
from trials import main as run_trials
from src.plotter import run_simulation
from src.benchmark import benchmark_suite

def main():
    parser = argparse.ArgumentParser(
        description="p-Laplacian PDE Solver and Benchmarking Suite"
    )

    subparsers = parser.add_subparsers(dest="task", required=True)

    sim_parser = subparsers.add_parser(
        "simulate", help="Run a standard simulation and generate plots"
    )
    sim_parser.add_argument("--live", action="store_true", help="Enable realtime GPU-accelerated plotting during integration")

    bench_parser = subparsers.add_parser(
        "benchmark", help="Run the intelligent benchmarking suite"
    )

    trials_parser = subparsers.add_parser("trials", help="Run the full experiment suite")
    trials_parser.add_argument("--sparsity", action="store_true", help="Run the sparsity scaling benchmark")
    trials_parser.add_argument("--work", action="store_true", help="Run the work-effort (precision) benchmark")
    trials_parser.add_argument("--epsilon", action="store_true", help="Run the epsilon regularization sensitivity benchmark")
    trials_parser.add_argument("--psweep", action="store_true", help="Run the nonlinearity index (p-value) benchmark")
    trials_parser.add_argument("--cvode-work", action="store_true", help="Run the CVODE vs LSODA work-effort benchmark")
    trials_parser.add_argument("--extreme-nx", action="store_true", help="Run the massive Nx scaling benchmark")
    trials_parser.add_argument("--extreme-p", action="store_true", help="Run the hyper-degenerate p scaling benchmark")
    trials_parser.add_argument("--all", action="store_true", help="Run all available benchmarks")
    trials_parser.add_argument("--singular-epsilon", action="store_true", help="Run the singular regime (p=1.25) epsilon crash test")

    
    bench_parser.add_argument(
        "--skip-error",
        action="store_true",
        help="Skip generating the ground truth solution and calculating L2 error",
    )
    
    bench_parser.add_argument(
        "--Nx",
        type=int,
        nargs="+",
        default=[50, 200, 1000],
        help="List of spatial resolutions to test (e.g., 50 200 1000)",
    )

    bench_parser.add_argument(
        "--p",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 3.0],
        help="List of p-values to test",
    )

    bench_parser.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=[1e-6, 1e-3, 0.0],
        help="List of epsilon regularization values to test",
    )

    bench_parser.add_argument(
        "--methods",
        nargs="+",
        default=["BDF", "Radau", "LSODA", "RK45"],
        help="List of ODE solvers to evaluate",
    )

    bench_parser.add_argument(
        "--T",
        type=float,
        default=0.01,
        help="Simulation end time for all benchmark runs",
    )

    args = parser.parse_args()

    if args.task == "simulate":
        run_simulation(live=args.live)

    elif args.task == "benchmark":
        grid = {
            "method": args.methods,
            "sparse": [True, False],
            "p": args.p,
            "epsilon": args.epsilon,
            "Nx": args.Nx,
        }
        df = benchmark_suite(grid, T=args.T, compute_error=not args.skip_error)

    elif args.task == "trials":
        run_trials(args)

if __name__ == "__main__":
    main()
