"""Command‑line entry point for the p‑Laplacian benchmarking suite."""

import argparse
from src.benchmark_configs import benchmarks
from src.benchmark_runner import run_benchmark_config


def main():
    parser = argparse.ArgumentParser(
        description="p‑Laplacian PDE Solver Benchmarking Suite"
    )
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    # Add a flag for each registered benchmark
    for cfg in benchmarks.values():
        parser.add_argument(f"--{cfg.flag}", action="store_true",
                            help=f"Run {cfg.name} benchmark")
    args = parser.parse_args()

    # Determine which benchmarks to run
    if args.all:
        selected = set(benchmarks.keys())
    else:
        selected = {name for name, cfg in benchmarks.items()
            if getattr(args, cfg.flag.replace('-', '_'), False)}

    if not selected:
        print("No benchmark selected. Use --all or one of the flags. Exiting.")
        return

    for name in selected:
        cfg = benchmarks[name]
        print(f"\n{'='*60}\n Running {cfg.name} \n{'='*60}")
        run_benchmark_config(cfg)


if __name__ == "__main__":
    main()
