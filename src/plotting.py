"""Consistent styling, plot templates, and export helpers for benchmarks."""

import os
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Callable

# ── Results directory ─────────────────────────────────────────────
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Global method style ────────────────────────────────────────────
METHOD_STYLE = {
    "LSODA": {"color": "#d7191c", "marker": "o"},
    "BDF":   {"color": "#2b83ba", "marker": "s"},
    "Radau": {"color": "#4daf4a", "marker": "^"},
    "CVODE": {"color": "#984ea3", "marker": "D"},
}


def apply_method_style(ax):
    """Force colours & markers to match METHOD_STYLE for all lines."""
    for line in ax.lines:
        label = line.get_label()
        if label in METHOD_STYLE:
            style = METHOD_STYLE[label]
            line.set_color(style["color"])
            line.set_marker(style["marker"])
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for h, lbl in zip(handles, labels):
        if lbl in METHOD_STYLE:
            style = METHOD_STYLE[lbl]
            h.set_color(style["color"])
            h.set_marker(style["marker"])
            h.set_markersize(9)
        new_handles.append(h)
    ax.legend(handles=new_handles, labels=labels,
              bbox_to_anchor=(1.05, 1), loc='upper left',
              frameon=True, title="Method")


def export_detailed_log(df: pd.DataFrame, base_filename: str):
    """Save a timestamped text file inside RESULTS_DIR."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = base_filename.replace(".pdf", "")
    fname = f"{stem}_{timestamp}.txt"
    full_path = os.path.join(RESULTS_DIR, fname)
    with open(full_path, "w") as f:
        f.write(f"Detailed Calculation Logs: {stem}\n")
        f.write(f"Generated at: {timestamp}\n")
        f.write("=" * 100 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n" + "=" * 100 + "\n")
    print(f"Saved {full_path}")


# ═══════════════════════════════════════════════════════════════════
# Reusable line‑plot skeleton
# ═══════════════════════════════════════════════════════════════════
def line_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    hue: str = "method",
    style: str = "method",
    xlabel: str = "",
    ylabel: str = "",
    xlog: bool = False,
    ylog: bool = False,
    invert_x: bool = False,
    filename: str = "output.pdf",
    annotation_col: str | None = None,
    highlight_lsoda: bool = False,
    extra_customisations: Callable | None = None,
):
    """Generic line‑plot for benchmark results, with consistent styling."""
    plot_df = df[(df["status"] == "Success")].copy()
    if plot_df.empty:
        print("No successful runs to plot.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax = sns.lineplot(
        data=plot_df,
        x=x, y=y,
        hue=hue,
        style=style,
        markers=[METHOD_STYLE[m]["marker"] for m in plot_df[hue].unique()],
        dashes=False,
        palette={m: s["color"] for m, s in METHOD_STYLE.items()},
        linewidth=2.5,
        markersize=9,
        sort=False,  # keep original order
    )

    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if invert_x:
        ax.invert_xaxis()

    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls="--", alpha=0.4)

    # Optional annotation of tolerance / parameter
    if annotation_col:
        for _, row in plot_df.iterrows():
            ax.annotate(
                f"{row[annotation_col]:.0e}",
                (row[x], row[y]),
                textcoords="offset points",
                xytext=(10, 0),
                ha='left', va='center',
                fontsize=8,
                color=METHOD_STYLE[row[hue]]["color"],
                alpha=0.85,
            )

    if highlight_lsoda:
        for line in ax.lines:
            if line.get_label() == "LSODA":
                line.set_linewidth(3.0)
                line.set_zorder(10)

    if extra_customisations:
        extra_customisations(ax, plot_df)

    apply_method_style(ax)
    plt.tight_layout()
    # Save inside RESULTS_DIR
    filepath = os.path.join(RESULTS_DIR, filename)
    plt.savefig(filepath, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {filepath}")
    export_detailed_log(df, filename)


# ═══════════════════════════════════════════════════════════════════
# Concrete plot functions (mostly thin wrappers around line_plot)
# ═══════════════════════════════════════════════════════════════════

def plot_work_effort(df: pd.DataFrame):
    line_plot(
        df,
        x="duration_s", y="error_l2",
        xlog=True, ylog=True,
        xlabel="Execution Time (seconds, Log Scale)",
        ylabel="L2 Error (Log Scale)",
        annotation_col="tol",
        highlight_lsoda=True,
        filename="work_effort.pdf",
    )


def plot_cvode_work_effort(df: pd.DataFrame):
    line_plot(
        df,
        x="duration_s", y="error_l2",
        xlog=True, ylog=True,
        xlabel="Execution Time (seconds, Log Scale)",
        ylabel="L2 Error (Log Scale)",
        annotation_col="tol",
        filename="cvode_work_effort.pdf",
    )


def plot_epsilon_sweep(df: pd.DataFrame):
    line_plot(
        df,
        x="epsilon", y="duration_s",
        xlog=True, ylog=True,
        invert_x=True,
        xlabel=r"Regularization Parameter ($\epsilon$, Log Scale)",
        ylabel="Duration (seconds, Log Scale)",
        filename="epsilon_sweep.pdf",
    )


def _p_sweep_custom(ax, df):
    ax.axvline(x=2.0, color='gray', linestyle='--', alpha=0.5, zorder=0)
    ax.text(2.05, 0.95, 'Linear Case ($p=2.0$)',
            transform=ax.get_xaxis_transform(),
            color='gray', fontsize=10, va='top', ha='left')


def plot_p_sweep(df: pd.DataFrame):
    line_plot(
        df,
        x="p", y="duration_s",
        ylog=True,
        xlabel="Nonlinearity Index ($p$)",
        ylabel="Duration (seconds, Log Scale)",
        extra_customisations=_p_sweep_custom,
        filename="p_sweep.pdf",
    )


def plot_extreme_nx(df: pd.DataFrame):
    line_plot(
        df,
        x="Nx", y="duration_s",
        xlog=True, ylog=True,
        xlabel="Grid Resolution ($N_x$, Log Scale)",
        ylabel="Duration (seconds, Log Scale)",
        filename="extreme_nx_scaling.pdf",
    )


def plot_extreme_p(df: pd.DataFrame):
    line_plot(
        df,
        x="p", y="duration_s",
        ylog=True,
        xlabel="Nonlinearity Index ($p$)",
        ylabel="Duration (seconds, Log Scale)",
        filename="extreme_p_scaling.pdf",
    )


def plot_sparsity_scaling(df: pd.DataFrame):
    """Sparsity scaling – separate handling because of dense/sparse legend."""
    plot_df = df[df["status"] == "Success"].copy()
    if plot_df.empty:
        print("No successful runs.")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    sparse_df = plot_df[plot_df["sparse"] == "True"]
    if not sparse_df.empty:
        sns.lineplot(
            data=sparse_df,
            x="Nx", y="duration_s",
            hue="method", style="method",
            markers=[METHOD_STYLE[m]["marker"] for m in sparse_df["method"].unique()],
            dashes=False,
            palette={m: s["color"] for m, s in METHOD_STYLE.items()},
            linewidth=2.5, markersize=9,
            ax=ax,
        )

    dense_df = plot_df[plot_df["sparse"] == "False"]
    if not dense_df.empty:
        sns.lineplot(
            data=dense_df,
            x="Nx", y="duration_s",
            hue="method", style="method",
            markers=[METHOD_STYLE[m]["marker"] for m in dense_df["method"].unique()],
            dashes=[(4, 4)] * len(dense_df["method"].unique()),
            palette={m: s["color"] for m, s in METHOD_STYLE.items()},
            linewidth=2.5, markersize=9,
            ax=ax,
        )

    ax.set_yscale("log")
    ax.set_xlabel("Grid Resolution ($N_x$)", fontweight='bold')
    ax.set_ylabel("Duration (seconds, Log Scale)", fontweight='bold')
    plt.grid(True, which="major", ls="-", alpha=0.8)
    plt.grid(True, which="minor", ls="--", alpha=0.4)

    # ── Apply METHOD_STYLE colours & markers to the lines ──────────
    for line in ax.lines:
        label = line.get_label()
        if label in METHOD_STYLE:
            style = METHOD_STYLE[label]
            line.set_color(style["color"])
            line.set_marker(style["marker"])
            line.set_markersize(9)

    # ── Build clean legend without duplicates ──────────────────────
    handles, labels = ax.get_legend_handles_labels()

    # De-duplicate: keep only the first occurrence of each label
    seen = set()
    unique_handles, unique_labels = [], []
    for h, l in zip(handles, labels):
        if l not in seen:
            seen.add(l)
            unique_handles.append(h)
            unique_labels.append(l)

    # Filter to only method lines (skip other possible artefacts)
    method_handles = [h for h, l in zip(unique_handles, unique_labels) if l in METHOD_STYLE]
    method_labels = [l for l in unique_labels if l in METHOD_STYLE]

    # Add dummy entries for Sparse / Dense
    sparse_handle, = ax.plot([], [], color='gray', linestyle='-', label='Sparse')
    dense_handle, = ax.plot([], [], color='gray', linestyle='--', label='Dense')

    ax.legend(
        handles=method_handles + [sparse_handle, dense_handle],
        labels=method_labels + ["Sparse", "Dense"],
        bbox_to_anchor=(1.05, 1), loc='upper left',
        frameon=True, title="Method & Sparsity",
    )

    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, "sparsity_scaling.pdf")
    plt.savefig(filepath, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {filepath}")
    export_detailed_log(df, "sparsity_scaling.pdf")


def plot_singular_epsilon(df: pd.DataFrame):
    """Heatmap for the singular epsilon crash test."""
    plot_df = df.copy()
    success_mask = (plot_df['status'] == 'Success')
    successes = plot_df[success_mask]
    if successes.empty:
        print("No successful runs to plot.")
        return

    vmin = successes['duration_s'].min()
    vmax = successes['duration_s'].max()

    p_vals = sorted(plot_df['p'].unique(), reverse=True)
    eps_vals = sorted(plot_df['epsilon'].unique(), reverse=True)
    methods = plot_df['method'].unique()

    fig, axes = plt.subplots(1, len(methods), figsize=(5 * len(methods), 5), sharey=True)
    if len(methods) == 1:
        axes = [axes]

    for ax, method in zip(axes, methods):
        ax.set_facecolor("#ff4c4c")
        m_df = plot_df[(plot_df['method'] == method) & (plot_df['status'] == 'Success')]
        pivot = m_df.pivot(index='p', columns='epsilon', values='duration_s')
        pivot = pivot.reindex(index=p_vals, columns=eps_vals)

        sns.heatmap(pivot, ax=ax, cmap="viridis", vmin=vmin, vmax=vmax,
                    cbar=(ax == axes[-1]),
                    cbar_kws={'label': 'Execution Time (s)'} if ax == axes[-1] else None,
                    linewidths=1.5, linecolor='white', square=True)
        ax.set_title(f"{method}", fontweight='bold')
        ax.set_ylabel("Nonlinearity Index ($p$)" if ax == axes[0] else "")
        ax.set_xlabel(r"Regularization ($\epsilon$)")

        x_labels = [f"0.0" if val == 0 else f"$10^{{{int(np.log10(val))}}}$" for val in eps_vals]
        ax.set_xticklabels(x_labels, rotation=45, ha="right")

    fig.text(0.5, -0.05, "Red Cells Indicate Catastrophic Solver Failure",
             ha='center', fontsize=11, fontweight='bold', color='#ff4c4c')
    plt.tight_layout()
    filepath = os.path.join(RESULTS_DIR, "stability_matrix.pdf")
    plt.savefig(filepath, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved {filepath}")
    export_detailed_log(df, "stability_matrix.pdf")
