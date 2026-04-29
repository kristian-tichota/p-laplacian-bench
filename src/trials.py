import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.benchmark import benchmark_suite

sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

METHOD_STYLE = {
    "LSODA": {"color": "#d7191c", "marker": "o"},
    "BDF":   {"color": "#2b83ba", "marker": "s"},
    "Radau": {"color": "#4daf4a", "marker": "^"},
    "CVODE": {"color": "#984ea3", "marker": "D"},
}


def _apply_method_style(ax):
    """After plotting, set colours & markers manually to enforce consistency."""
    for line in ax.lines:
        label = line.get_label()
        if label in METHOD_STYLE:
            style = METHOD_STYLE[label]
            line.set_color(style["color"])
            line.set_marker(style["marker"])
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for h, lbl in zip(handles, labels):
        if lbl in METHOD_STYLE:
            style = METHOD_STYLE[lbl]
            h.set_color(style["color"])
            h.set_marker(style["marker"])
            h.set_markersize(9)
        new_handles.append(h)
    ax.legend(handles=new_handles, labels=labels, bbox_to_anchor=(1.05, 1),
              loc='upper left', frameon=True, title="Method")


def export_detailed_log(df, pdf_filename):
    """Exports the complete DataFrame to a timestamped text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = pdf_filename.replace(".pdf", "")
    txt_filename = f"{base_name}_{timestamp}.txt"
    
    with open(txt_filename, "w") as f:
        f.write(f"Detailed Calculation Logs: {base_name}\n")
        f.write(f"Generated at: {timestamp}\n")
        f.write("=" * 100 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n" + "=" * 100 + "\n")
        
    print(f"Saved {txt_filename}")


def plot_cvode_work_effort(df):
    plot_df = df[(df["status"] == "Success") & (df["error_l2"] > 0)].copy()
    if plot_df.empty:
        print("No successful runs to plot for CVODE Work-Effort.")
        return

    plot_df = plot_df.sort_values(by=["method", "tol"], ascending=[True, False])
    plt.figure(figsize=(10, 7))

    ax = sns.lineplot(
        data=plot_df,
        x="duration_s",
        y="error_l2",
        hue="method",
        style="method",
        markers=[METHOD_STYLE[m]["marker"] for m in plot_df["method"].unique()],
        dashes=False,
        palette={m: s["color"] for m, s in METHOD_STYLE.items()},
        linewidth=2.5,
        markersize=9,
        sort=False,
    )

    for _, row in plot_df.iterrows():
        tol_str = f"{row['tol']:.0e}"
        ax.annotate(
            tol_str,
            (row['duration_s'], row['error_l2']),
            textcoords="offset points",
            xytext=(10, 0),
            ha='left', va='center',
            fontsize=8,
            color=METHOD_STYLE[row['method']]["color"],
            alpha=0.85,
        )

    plt.ylabel("L2 Error (Log Scale)", fontweight='bold')
    plt.xlabel("Execution Time (seconds, Log Scale)", fontweight='bold')
    ax.set_xscale("log"); ax.set_yscale("log")
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls="--", alpha=0.4)
    _apply_method_style(ax)
    plt.tight_layout()
    plt.savefig("cvode_work_effort.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("Saved cvode_work_effort.pdf")
    export_detailed_log(df, "cvode_work_effort.pdf")


def plot_extreme_nx(df):
    plot_df = df[df["status"] == "Success"].copy()
    if plot_df.empty:
        print("No successful runs to plot for Extreme Nx.")
        return

    plt.figure(figsize=(9, 6))
    ax = sns.lineplot(
        data=plot_df,
        x="Nx",
        y="duration_s",
        hue="method",
        style="method",
        markers=[METHOD_STYLE[m]["marker"] for m in plot_df["method"].unique()],
        dashes=False,
        palette={m: s["color"] for m, s in METHOD_STYLE.items()},
        linewidth=2.5,
        markersize=9,
    )
    plt.ylabel("Duration (seconds, Log Scale)", fontweight='bold')
    plt.xlabel("Grid Resolution ($N_x$, Log Scale)", fontweight='bold')
    ax.set_xscale("log"); ax.set_yscale("log")
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls="--", alpha=0.4)
    _apply_method_style(ax)
    plt.tight_layout()
    plt.savefig("extreme_nx_scaling.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("Saved extreme_nx_scaling.pdf")
    export_detailed_log(df, "extreme_nx_scaling.pdf")


def plot_extreme_p(df):
    plot_df = df[df["status"] == "Success"].copy()
    if plot_df.empty:
        print("No successful runs to plot for Extreme P.")
        return

    plt.figure(figsize=(9, 6))
    ax = sns.lineplot(
        data=plot_df,
        x="p",
        y="duration_s",
        hue="method",
        style="method",
        markers=[METHOD_STYLE[m]["marker"] for m in plot_df["method"].unique()],
        dashes=False,
        palette={m: s["color"] for m, s in METHOD_STYLE.items()},
        linewidth=2.5,
        markersize=9,
    )
    plt.ylabel("Duration (seconds, Log Scale)", fontweight='bold')
    plt.xlabel("Nonlinearity Index ($p$)", fontweight='bold')
    ax.set_yscale("log")
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls="--", alpha=0.4)
    _apply_method_style(ax)
    plt.tight_layout()
    plt.savefig("extreme_p_scaling.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("Saved extreme_p_scaling.pdf")
    export_detailed_log(df, "extreme_p_scaling.pdf")


def plot_sparsity_scaling(df):
    plot_df = df[df["status"] == "Success"].copy()
    if plot_df.empty:
        print("No successful runs to plot for Sparsity Scaling.")
        return

    plt.figure(figsize=(9, 6))
    
    sparse_df = plot_df[plot_df["sparse"] == "True"]
    if not sparse_df.empty:
        sns.lineplot(
            data=sparse_df,
            x="Nx",
            y="duration_s",
            hue="method",
            style="method",
            markers=[METHOD_STYLE[m]["marker"] for m in sparse_df["method"].unique()],
            dashes=False,
            palette={m: s["color"] for m, s in METHOD_STYLE.items()},
            linewidth=2.5,
            markersize=9,
        )

    dense_df = plot_df[plot_df["sparse"] == "False"]
    if not dense_df.empty:
        ax = sns.lineplot(
            data=dense_df,
            x="Nx",
            y="duration_s",
            hue="method",
            style="method",
            markers=[METHOD_STYLE[m]["marker"] for m in dense_df["method"].unique()],
            dashes=[(4, 4)] * len(dense_df["method"].unique()), 
            palette={m: s["color"] for m, s in METHOD_STYLE.items()},
            linewidth=2.5,
            markersize=9,
        )
    else:
        ax = plt.gca()

    plt.ylabel("Duration (seconds, Log Scale)", fontweight='bold')
    plt.xlabel("Grid Resolution ($N_x$)", fontweight='bold')
    ax.set_yscale("log")
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls="--", alpha=0.4)
    
    _apply_method_style(ax)

    handles, labels = ax.get_legend_handles_labels()
    unique_legend = {}
    for h, l in zip(handles, labels):
        if l not in unique_legend and l in METHOD_STYLE:
            unique_legend[l] = h
            
    sparse_handle, = ax.plot([], [], color='gray', linestyle='-')
    dense_handle, = ax.plot([], [], color='gray', linestyle='--')
            
    ax.legend(
        list(unique_legend.values()) + [sparse_handle, dense_handle], 
        list(unique_legend.keys()) + ["Sparse", "Dense"], 
        bbox_to_anchor=(1.05, 1), 
        loc='upper left', 
        frameon=True, 
        title="Method & Sparsity"
    )

    plt.tight_layout()
    plt.savefig("sparsity_scaling.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("Saved sparsity_scaling.pdf")
    export_detailed_log(df, "sparsity_scaling.pdf")
def plot_work_effort(df):
    plot_df = df[(df["status"] == "Success") & (df["error_l2"] > 0)].copy()
    if plot_df.empty:
        print("No successful runs to plot for Work-Effort.")
        return

    plot_df = plot_df.sort_values(by=["method", "tol"], ascending=[True, False])
    plt.figure(figsize=(10, 7))

    ax = sns.lineplot(
        data=plot_df,
        x="duration_s",
        y="error_l2",
        hue="method",
        style="method",
        markers=[METHOD_STYLE[m]["marker"] for m in plot_df["method"].unique()],
        dashes=False,
        palette={m: s["color"] for m, s in METHOD_STYLE.items()},
        linewidth=2.5,
        markersize=9,
        sort=False,
    )

    for _, row in plot_df.iterrows():
        tol_str = f"{row['tol']:.0e}"
        ax.annotate(
            tol_str,
            (row['duration_s'], row['error_l2']),
            textcoords="offset points",
            xytext=(10, 0),
            ha='left', va='center',
            fontsize=8,
            color=METHOD_STYLE[row['method']]["color"],
            alpha=0.85,
        )

    for line in ax.lines:
        if line.get_label() == "LSODA":
            line.set_linewidth(3.0)
            line.set_zorder(10)

    plt.ylabel("L2 Error (Log Scale)", fontweight='bold')
    plt.xlabel("Execution Time (seconds, Log Scale)", fontweight='bold')
    ax.set_xscale("log"); ax.set_yscale("log")
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls="--", alpha=0.4)
    _apply_method_style(ax)
    plt.tight_layout()
    plt.savefig("work_effort.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("Saved work_effort.pdf")
    export_detailed_log(df, "work_effort.pdf")


def plot_epsilon_sweep(df):
    plot_df = df[df["status"] == "Success"].copy()
    if plot_df.empty:
        print("No successful runs to plot for Epsilon Sweep.")
        return

    plt.figure(figsize=(9, 6))
    ax = sns.lineplot(
        data=plot_df,
        x="epsilon",
        y="duration_s",
        hue="method",
        style="method",
        markers=[METHOD_STYLE[m]["marker"] for m in plot_df["method"].unique()],
        dashes=False,
        palette={m: s["color"] for m, s in METHOD_STYLE.items()},
        linewidth=2.5,
        markersize=9,
    )
    plt.ylabel("Duration (seconds, Log Scale)", fontweight='bold')
    plt.xlabel(r"Regularization Parameter ($\epsilon$, Log Scale)", fontweight='bold')
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.invert_xaxis()
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls="--", alpha=0.4)
    _apply_method_style(ax)
    plt.tight_layout()
    plt.savefig("epsilon_sweep.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("Saved epsilon_sweep.pdf")
    export_detailed_log(df, "epsilon_sweep.pdf")


def plot_p_sweep(df):
    plot_df = df[df["status"] == "Success"].copy()
    if plot_df.empty:
        print("No successful runs to plot for P Sweep.")
        return

    plt.figure(figsize=(9, 6))
    ax = sns.lineplot(
        data=plot_df,
        x="p",
        y="duration_s",
        hue="method",
        style="method",
        markers=[METHOD_STYLE[m]["marker"] for m in plot_df["method"].unique()],
        dashes=False,
        palette={m: s["color"] for m, s in METHOD_STYLE.items()},
        linewidth=2.5,
        markersize=9,
    )
    plt.ylabel("Duration (seconds, Log Scale)", fontweight='bold')
    plt.xlabel("Nonlinearity Index ($p$)", fontweight='bold')
    ax.set_yscale("log")
    plt.axvline(x=2.0, color='gray', linestyle='--', alpha=0.5, zorder=0)
    plt.text(2.05, 0.95, 'Linear Case ($p=2.0$)',
             transform=ax.get_xaxis_transform(),
             color='gray', fontsize=10, va='top', ha='left')
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls="--", alpha=0.4)
    _apply_method_style(ax)
    plt.tight_layout()
    plt.savefig("p_sweep.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("Saved p_sweep.pdf")
    export_detailed_log(df, "p_sweep.pdf")


def run_sparsity_benchmark():
    grid_scaling = {
        "method": ["LSODA", "BDF"],
        "sparse": [True, False],
        "p": [2.5],
        "epsilon": [1e-6],
        "Nx": [50, 100, 200, 500, 1000, 1500, 2000],
        "tol": [1e-6],
    }
    print("\n--- Running Sparsity vs. Dense Scaling Benchmark (tol=1e-6) ---")
    df_scaling = benchmark_suite(grid_scaling, T=0.05, compute_error=False)
    print("\n--- Sparsity Benchmark Results ---")
    print(df_scaling[["method", "sparse", "Nx", "duration_s", "status"]].to_string(index=False))
    plot_sparsity_scaling(df_scaling)
    return df_scaling


def run_work_effort_benchmark():
    grid_effort = {
        "method": ["LSODA", "BDF", "Radau"],
        "sparse": [True],
        "p": [2.5],
        "epsilon": [1e-6],
        "Nx": [1000],
        "tol": [1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
    }
    print("\n--- Running Heavy Work-Effort Benchmark ---")
    df_effort = benchmark_suite(grid_effort, T=0.05, compute_error=True)
    print("\n--- Work-Effort Benchmark Results ---")
    print(df_effort[["method", "tol", "duration_s", "error_l2", "status"]].to_string(index=False))
    plot_work_effort(df_effort)
    return df_effort

def run_singular_epsilon_benchmark():
    eps_values = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-12, 1e-14, 0.0]

    grid = {
        "method": ["LSODA", "BDF", "Radau"],
        "sparse": [True],
        "p": [1.25],
        "epsilon": eps_values,
        "Nx": [1000],
        "tol": [1e-6],
    }
    print("\n--- Running Singular Epsilon Crash Test (p=1.25) ---")
    df = benchmark_suite(grid, T=0.05, compute_error=False)

    print("\n--- Singular Epsilon Results ---")
    print(df[["method", "epsilon", "duration_s", "status"]].to_string(index=False))
    plot_singular_epsilon(df)
    return df


def plot_singular_epsilon(df):
    plt.figure(figsize=(9, 6))

    success = df[df["status"] == "Success"].copy()
    failed  = df[df["status"] != "Success"].copy()

    for method in success["method"].unique():
        subset = success[success["method"] == method]
        style = METHOD_STYLE.get(method, {})
        plt.plot(subset["epsilon"], subset["duration_s"],
                 color=style.get("color", "black"),
                 marker=style.get("marker", "o"),
                 linewidth=2.5, markersize=9, label=method)

    for method in failed["method"].unique():
        subset = failed[failed["method"] == method]
        style = METHOD_STYLE.get(method, {})
        plt.scatter(subset["epsilon"], [np.nan] * len(subset),
                    marker="x", s=120, color=style.get("color", "red"),
                    linewidths=2, label=f"{method} (failed)")

    plt.xscale("log")
    plt.yscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel(r"Regularization $\varepsilon$ (log scale)", fontweight="bold")
    plt.ylabel("Execution Time (s, log scale)", fontweight="bold")
    plt.title("Solver Robustness in Singular Regime (p=1.25)")
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls="--", alpha=0.4)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1),
               loc='upper left', frameon=True)

    plt.tight_layout()
    plt.savefig("singular_epsilon_crash.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("Saved singular_epsilon_crash.pdf")
    export_detailed_log(df, "singular_epsilon_crash.pdf")

def run_epsilon_sweep_benchmark():
    grid_epsilon = {
        "method": ["LSODA", "BDF", "Radau"],
        "sparse": [True],
        "p": [2.5],
        "epsilon": [1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
        "Nx": [1000],
        "tol": [1e-6],
    }
    print("\n--- Running Regularization Sensitivity (Epsilon Sweep) Benchmark ---")
    df_epsilon = benchmark_suite(grid_epsilon, T=0.05, compute_error=False)
    print("\n--- Epsilon Sweep Benchmark Results ---")
    print(df_epsilon[["method", "epsilon", "duration_s", "status"]].to_string(index=False))
    plot_epsilon_sweep(df_epsilon)
    return df_epsilon


def run_p_sweep_benchmark():
    grid_p = {
        "method": ["LSODA", "BDF", "Radau"],
        "sparse": [True],
        "p": [1.5, 1.8, 2.0, 2.5, 3.0, 4.0],
        "epsilon": [1e-4],
        "Nx": [1000],
        "tol": [1e-6],
    }
    print("\n--- Running Nonlinearity Index (P Sweep) Benchmark ---")
    df_p = benchmark_suite(grid_p, T=0.05, compute_error=False)
    print("\n--- P Sweep Benchmark Results ---")
    print(df_p[["method", "p", "duration_s", "status"]].to_string(index=False))
    plot_p_sweep(df_p)
    return df_p


def run_cvode_work_effort_benchmark():
    grid = {
        "method": ["LSODA", "CVODE"],
        "sparse": [True],
        "p": [3],
        "epsilon": [1e-6],
        "Nx": [5000],
        "tol": [1e-2, 1e-4, 1e-6, 1e-8, 1e-10],
    }
    print("\n--- Running CVODE vs LSODA Work-Effort Benchmark ---")
    df = benchmark_suite(grid, T=0.05, compute_error=True)
    print("\n--- CVODE vs LSODA Work-Effort Results ---")
    print(df[["method", "tol", "duration_s", "error_l2", "status"]].to_string(index=False))
    plot_cvode_work_effort(df)
    return df


def run_extreme_nx_benchmark():
    grid = {
        "method": ["LSODA", "CVODE"],
        "sparse": [True],
        "p": [2.5],
        "epsilon": [1e-6],
        "Nx": [1000, 5000, 10000, 25000, 50000, 100000],
        "tol": [1e-6],
    }
    print("\n--- Running Extreme Nx Benchmark ---")
    df = benchmark_suite(grid, T=0.05, compute_error=False)
    print("\n--- Extreme Nx Results ---")
    print(df[["method", "Nx", "duration_s", "status"]].to_string(index=False))
    plot_extreme_nx(df)
    return df


def run_extreme_p_benchmark():
    grid = {
        "method": ["LSODA", "CVODE"],
        "sparse": [True],
        "p": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "epsilon": [1e-6],
        "Nx": [1000],
        "tol": [1e-6],
    }
    print("\n--- Running Extreme P (Hyper-Degenerate) Benchmark ---")
    df = benchmark_suite(grid, T=0.05, compute_error=False)
    print("\n--- Extreme P Results ---")
    print(df[["method", "p", "duration_s", "status"]].to_string(index=False))
    plot_extreme_p(df)
    return df


def main(args):
    flags = [
        args.sparsity, args.work, args.epsilon, args.psweep,
        args.cvode_work, args.extreme_nx, args.extreme_p, args.singular_epsilon, args.all,
    ]
    if not any(flags):
        print("No specific benchmark requested. Running all benchmarks by default.\n")
        args.all = True

    if args.sparsity or args.all:
        run_sparsity_benchmark()
    if args.work or args.all:
        run_work_effort_benchmark()
    if args.epsilon or args.all:
        run_epsilon_sweep_benchmark()
    if args.psweep or args.all:
        run_p_sweep_benchmark()
    if args.cvode_work or args.all:
        run_cvode_work_effort_benchmark()
    if args.extreme_nx or args.all:
        run_extreme_nx_benchmark()
    if args.extreme_p or args.all:
        run_extreme_p_benchmark()
    if args.singular_epsilon or args.all:
        run_singular_epsilon_benchmark()


if __name__ == "__main__":
    class DummyArgs:
        sparsity = False
        work = False
        epsilon = False
        psweep = False
        cvode_work = False
        extreme_nx = False
        extreme_p = False
        singular_epsilon = False
        all = True
    main(DummyArgs())
