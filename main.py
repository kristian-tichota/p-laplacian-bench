"""Single entry point for the p‑Laplacian suite."""

import argparse
import sys

from src.benchmark import benchmark_suite
from src.benchmark.benchmark_configs import benchmarks
from src.benchmark.benchmark_runner import run_benchmark_config
from src.config import SimulationConfig
from src.plotter import run_simulation
from src.solver import PLaplacianSolver


def warmup_jit():
    """Warm‑up Numba JIT to avoid timing distortion in benchmarks."""
    config = SimulationConfig(p=2.5, Nx=100, epsilon=1e-6)
    disc = config.to_discretization()
    solver = PLaplacianSolver(disc, config)
    solver.solve([0.001])


def profile_model(args: argparse.Namespace):
    import cProfile
    import pstats

    config = config_from_args(args)
    disc = config.to_discretization()
    solver = PLaplacianSolver(disc, config)

    def _run():
        solver.solve([0.05])

    profiler = cProfile.Profile()
    profiler.enable()
    _run()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumulative")
    stats.print_stats(20)


def add_physics_arguments(
    parser: argparse.ArgumentParser,
    default_p: float = 4.5,
    default_Nx: int = 5000,
    default_epsilon: float = 1e-6,
    default_tol: float = 1e-6,
):
    parser.add_argument("--p", type=float, default=default_p, help="Nonlinearity index")
    parser.add_argument("--Nx", type=int, default=default_Nx, help="Grid resolution")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=default_epsilon,
        help="Regularization parameter",
    )
    parser.add_argument(
        "--tol", type=float, default=default_tol, help="Solver tolerance (rtol / atol)"
    )


def add_method_argument(
    parser: argparse.ArgumentParser,
    dest: str = "method",
    default: str = "LSODA",
    choices=None,
    plural: bool = False,
):
    if choices is None:
        choices = ["LSODA", "BDF", "Radau", "RK45", "CVODE", "FENICSX_DIRECT"]
    if plural:
        parser.add_argument(
            "--methods",
            nargs="+",
            default=["BDF", "Radau", "LSODA", "RK45"],
            choices=choices,
            help="Solver backends to compare (space separated)",
        )
    else:
        parser.add_argument(
            "--method",
            choices=choices,
            default=default,
            help="Solver backend method",
        )


def add_simulate_args(parser: argparse.ArgumentParser):
    add_physics_arguments(parser)
    add_method_argument(parser)
    parser.add_argument("--discretization", choices=["fdm", "fem"], default="fdm")
    parser.add_argument(
        "--live", action="store_true", help="Enable real‑time GPU‑accelerated plotting"
    )


def add_profile_args(parser: argparse.ArgumentParser):
    add_physics_arguments(
        parser, default_p=6.0, default_Nx=1000, default_epsilon=1e-6, default_tol=1e-6
    )
    add_method_argument(parser, default="LSODA")


def add_benchmark_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--Nx",
        type=int,
        nargs="+",
        default=[50, 200, 1000],
        help="Grid resolutions to test",
    )
    parser.add_argument(
        "--p",
        type=float,
        nargs="+",
        default=[1.5, 2.0, 3.0],
        help="Nonlinearity indices",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        nargs="+",
        default=[1e-6, 1e-3, 0.0],
        help="Regularization parameters",
    )
    parser.add_argument(
        "--tol", type=float, nargs="+", default=[1e-3], help="Solver tolerances"
    )
    add_method_argument(parser, plural=True)
    parser.add_argument(
        "--T", type=float, default=0.01, help="Simulation end time for each run"
    )
    parser.add_argument(
        "--skip-error", action="store_true", help="Do not compute reference L2 error"
    )


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        p=args.p,
        Nx=args.Nx,
        epsilon=args.epsilon,
        method=args.method,
        rtol=args.tol,
        atol=args.tol,
        discretization_type=getattr(args, "discretization", "fdm"),
    )


def run_simulate_command(args: argparse.Namespace):
    config = config_from_args(args)
    run_simulation(config, live=args.live)


def run_profile_command(args: argparse.Namespace):
    profile_model(args)


def run_benchmark_command(args: argparse.Namespace):
    grid = {
        "method": args.methods,
        "sparse": [True, False],
        "p": args.p,
        "epsilon": args.epsilon,
        "Nx": args.Nx,
        "tol": args.tol,
    }
    df = benchmark_suite(grid, T=args.T, compute_error=not args.skip_error)
    print(df.to_string(index=False))


def run_trials_command(args: argparse.Namespace):
    if args.all:
        selected = set(benchmarks.keys())
    else:
        selected = {
            name
            for name, cfg in benchmarks.items()
            if getattr(args, cfg.flag.replace("-", "_"), False)
        }
    if not selected:
        print("No trial selected. Use --all or one of the trial flags.")
        sys.exit(1)
    for name in selected:
        cfg = benchmarks[name]
        print(f"\n{'=' * 60}\n Running {cfg.name} \n{'=' * 60}")
        run_benchmark_config(cfg)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="p‑Laplacian PDE Solver and Benchmarking Suite"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sim_parser = subparsers.add_parser("simulate", help="Run a standard simulation")
    add_simulate_args(sim_parser)
    sim_parser.set_defaults(func=run_simulate_command)

    bench_parser = subparsers.add_parser(
        "benchmark", help="Run a custom benchmark grid"
    )
    add_benchmark_args(bench_parser)
    bench_parser.set_defaults(func=run_benchmark_command)

    trials_parser = subparsers.add_parser("trials", help="Run preset benchmark suite")
    trials_parser.add_argument(
        "--all", action="store_true", help="Run all available benchmarks"
    )
    for cfg in benchmarks.values():
        trials_parser.add_argument(
            f"--{cfg.flag}", action="store_true", help=f"Run {cfg.name} benchmark"
        )
    trials_parser.set_defaults(func=run_trials_command)

    profile_parser = subparsers.add_parser(
        "profile", help="Profile a fixed degenerate run"
    )
    add_profile_args(profile_parser)
    profile_parser.set_defaults(func=run_profile_command)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("Executing JIT warmup...")
    warmup_jit()
    print("JIT ready!")

    args.func(args)


if __name__ == "__main__":
    main()
