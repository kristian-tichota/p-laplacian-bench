import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.benchmark import benchmark_suite

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

def plot_sparsity_scaling(df):
    """Plots the execution time scaling of sparse vs dense Jacobians across Nx."""
    plot_df = df[df["status"] == "Success"].copy()
    
    if plot_df.empty:
        print("No successful runs to plot for Sparsity Scaling.")
        return

    plt.figure(figsize=(9, 6))

    ax = sns.lineplot(
        data=plot_df,
        x="Nx",
        y="duration_s",
        hue="method",
        style="sparse",
        markers=True,
        dashes=True,
        palette="Set1",
        linewidth=2.5,
        markersize=9
    )

    plt.ylabel("Duration (seconds, Log Scale)", fontweight='bold')
    plt.xlabel("Grid Resolution ($N_x$)", fontweight='bold')

    ax.set_yscale("log")
    
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls="--", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(
        handles=handles, 
        labels=labels, 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("sparsity_scaling.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("Saved sparsity_scaling.pdf")


def plot_work_effort(df):
    """Plots the Work-Effort (Precision) diagram with visual hierarchy and annotations."""
    plot_df = df[(df["status"] == "Success") & (df["error_l2"] > 0)].copy()
    
    if plot_df.empty:
        print("No successful runs to plot for Work-Effort.")
        return

    # Critical: Sort by tolerance (descending) so the line connects points 
    # in the order of the parameter sweep, not by execution time.
    plot_df = plot_df.sort_values(by=["method", "tol"], ascending=[True, False])

    plt.figure(figsize=(10, 7))

    custom_palette = {
        "LSODA": "#d7191c",  
        "BDF": "#2b83ba",    
        "Radau": "#4daf4a",  
    }

    # sort=False ensures seaborn respects our explicit DataFrame sorting
    ax = sns.lineplot(
        data=plot_df,
        x="duration_s",
        y="error_l2",
        hue="method",
        style="method",
        markers=["o", "s", "^"],
        dashes=False, 
        palette=custom_palette,
        linewidth=2.5,
        markersize=9,
        sort=False 
    )

    # Annotate points with their tolerance values
    for _, row in plot_df.iterrows():
        # Format tolerance cleanly, e.g., '1e-04'
        tol_str = f"{row['tol']:.0e}"
        
        ax.annotate(
            tol_str,
            (row['duration_s'], row['error_l2']),
            textcoords="offset points",
            xytext=(10, 0), # Offset text 10 points to the right
            ha='left',
            va='center',
            fontsize=8,
            color=custom_palette.get(row['method'], 'black'),
            alpha=0.85
        )

    # Visual hierarchy: push LSODA to the front
    for line in ax.lines:
        color = line.get_color()
        if color == "#d7191c": 
            line.set_linewidth(3.0)
            line.set_zorder(10)

    plt.ylabel("L2 Error (Log Scale)", fontweight='bold')
    plt.xlabel("Execution Time (seconds, Log Scale)", fontweight='bold')

    ax.set_xscale("log")
    ax.set_yscale("log")

    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls="--", alpha=0.4)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend(
        handles=handles, 
        labels=labels, 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        frameon=True,
        title="Method"
    )

    plt.tight_layout()
    plt.savefig("work_effort.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("Saved work_effort.pdf")

def run_sparsity_benchmark():
    """
    Core Benchmark: Sparse Overhead vs. Scaling Payoff
    Demonstrates that sparsity adds overhead at low Nx, 
    but provides massive speedups at high Nx.
    """
    grid_scaling = {
        "method": ["LSODA", "BDF"],
        "sparse": [True, False],
        "p": [2.5],
        "epsilon": [1e-6],
        "Nx": [50, 100, 200, 500, 1000, 1500, 2000] 
    }
    
    print("\n--- Running Sparsity vs. Dense Scaling Benchmark ---")
    df_scaling = benchmark_suite(grid_scaling, T=0.05, compute_error=False)
    
    print("\n--- Sparsity Benchmark Results ---")
    print(df_scaling[["method", "sparse", "Nx", "duration_s", "status"]].to_string(index=False))
    
    plot_sparsity_scaling(df_scaling)
    return df_scaling


def run_work_effort_benchmark():
    """
    Work-Effort Benchmark: Accuracy vs. Runtime
    Demonstrates which methods achieve the lowest error for 
    a given computational time by varying solver tolerances.
    """
    grid_effort = {
        "method": ["LSODA", "BDF", "Radau"],
        "sparse": [True],
        "p": [2.5],
        "epsilon": [1e-6],
        "Nx": [1000],  
        "tol": [1e-2, 1e-4, 1e-6, 1e-8, 1e-10] 
    }
    
    print("\n--- Running Heavy Work-Effort Benchmark ---")
    df_effort = benchmark_suite(grid_effort, T=0.05, compute_error=True)
    
    print("\n--- Work-Effort Benchmark Results ---")
    print(df_effort[["method", "tol", "duration_s", "error_l2", "status"]].to_string(index=False))
    
    plot_work_effort(df_effort)
    return df_effort


def main(args):
    """
    Main entry point for trials. Expects 'args' from the top-level main.py parser.
    """
    if not any([args.sparsity, args.work, args.all]):
        print("No specific benchmark requested. Running all benchmarks by default.\n")
        args.all = True

    if args.sparsity or args.all:
        run_sparsity_benchmark()

    if args.work or args.all:
        run_work_effort_benchmark()


if __name__ == "__main__":
    class DummyArgs:
        sparsity = False
        work = False
        all = True
    main(DummyArgs())
