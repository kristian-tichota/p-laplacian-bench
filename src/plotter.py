import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .solver import PLaplacianSolver
from .config import SimulationConfig
from .live_plot import LivePlotHook


def run_simulation(config: SimulationConfig, live: bool = False):
    times_to_plot = [0.005, 0.015, 0.035, 0.065]
    disc = config.to_discretization()
    solver = PLaplacianSolver(disc, config)

    if live:
        hook = LivePlotHook(disc)
        import threading
        threading.Thread(target=solver.solve, args=(times_to_plot,),
                         kwargs={"hook": hook}, daemon=True).start()
        hook.start_plotter()
        return {}, {}
    else:
        results, stats = solver.solve(times_to_plot)

    x = disc.get_node_coordinates()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    colors = cm.magma(np.linspace(0.2, 0.85, len(times_to_plot)))

    ax.text(0.5, 0.8, f"p = {config.p}", fontsize=12, color="gray",
            alpha=0.7, transform=ax.transAxes)

    for i, t in enumerate(times_to_plot):
        u_vals = results[t]
        c = colors[i]
        ax.plot(x, u_vals, color=c, linewidth=2.5, label=f"$t = {t:.3f}$ s")
        ax.fill_between(x, u_vals, color=c, alpha=0.05)

    ax.set(xlim=(0, 0.8), ylim=(0, config.h * 1.05))
    ax.legend()
    plt.tight_layout()
    plt.show()
