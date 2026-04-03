import argparse
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

    bench_parser = subparsers.add_parser(
        "benchmark", help="Run the intelligent benchmarking suite"
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
        run_simulation()

    elif args.task == "benchmark":
        grid = {
            "method": args.methods,
            "sparse": [
                True,
                False,
            ],
            "p": args.p,
            "epsilon": args.epsilon,
            "Nx": args.Nx,
        }

        df = benchmark_suite(grid, T=args.T)


if __name__ == "__main__":
    main()
