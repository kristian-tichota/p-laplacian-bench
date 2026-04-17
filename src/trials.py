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

    plt.title("Execution Time Scaling: Sparse vs. Dense Jacobians", pad=15, fontweight='bold')
    plt.ylabel("Duration (seconds, Log Scale)", fontweight='bold')
    plt.xlabel("Grid Resolution ($N_x$)", fontweight='bold')

    ax.set_yscale("log")
    
    # Customize grid for log scale
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls="--", alpha=0.4)

    # Clean up and relocate the legend outside the plot
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(
        handles=handles, 
        labels=labels, 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        frameon=True
    )

    plt.tight_layout()
    plt.savefig("sparsity_scaling.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved sparsity_scaling.png")

def main():
    # ---------------------------------------------------------
    # Core Benchmark: Sparse Overhead vs. Scaling Payoff
    # Demonstrates that sparsity adds overhead at low Nx, 
    # but provides massive speedups at high Nx.
    # ---------------------------------------------------------
    grid_scaling = {
        "method": ["LSODA", "BDF"],
        "sparse": [True, False],
        "p": [2.5],
        "epsilon": [1e-6],
        "Nx": [100, 200, 500, 1000, 1500, 2000] 
    }
    
    print("--- Running Sparsity vs. Dense Scaling Benchmark ---")
    df_scaling = benchmark_suite(grid_scaling, T=0.05, compute_error=False)
    
    print("\n--- Benchmark Results ---")
    print(df_scaling[["method", "sparse", "Nx", "duration_s", "status"]].to_string(index=False))
    
    plot_sparsity_scaling(df_scaling)

if __name__ == "__main__":
    main()
