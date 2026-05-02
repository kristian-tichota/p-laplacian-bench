import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from .model import PLaplacianModel
from .solver import PLaplacianSolver
from .live_plot import LivePlotHook


def run_simulation(live=False, p=4.5, Nx=5000, epsilon=1e-6, tol=1e-6, method="LSODA"):
    times_to_plot = [0.001, 0.005, 0.015, 0.035, 0.065]

    model = PLaplacianModel(p=p, h=1.0, Nx=Nx, epsilon=epsilon)
    solver = PLaplacianSolver(model)

    if live:
        hook = LivePlotHook(model.Nx, model.dx, model.p, model.h, model.epsilon)
        import threading
        # Run integration in a background thread, hook records frames
        threading.Thread(target=solver.solve, args=(times_to_plot,),
                         kwargs={"method": method, "hook": hook, "rtol": tol, "atol": tol}, daemon=True).start()
        # Block main thread with the replay GUI
        hook.start_plotter()
        return {}, {}
    else:
        results, stats = solver.solve(times_to_plot)

    x = model.x
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    colors = cm.magma(np.linspace(0.2, 0.85, len(times_to_plot)))

    ax.text(0.5, 0.8, f"p = {model.p}", fontsize=12, color="gray", alpha=0.7,
            transform=ax.transAxes)

    for i, t in enumerate(times_to_plot):
        u_vals = results[t]
        c = colors[i]
        front_indices = np.where(u_vals > 1e-3)[0]
        front_x = x[front_indices[-1]] if len(front_indices) > 0 else 0
        ax.plot(x, u_vals, color=c, linewidth=2.5, label=f"$t = {t:.3f}$ s")
        ax.fill_between(x, u_vals, color=c, alpha=0.05)
        ax.plot(front_x, 0, marker="o", color=c, markersize=6,
                markeredgecolor="white", markeredgewidth=1.5, zorder=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_color("#333333")
    ax.spines["left"].set_color("#333333")
    ax.grid(True, which="major", axis="both", linestyle=":", color="gray", alpha=0.3)
    ax.set(xlim=(0, 0.8), ylim=(0, model.h * 1.05), xlabel="$x$", ylabel="$u(t,x)$")
    ax.legend(frameon=False, fontsize=11, loc="upper right", bbox_to_anchor=(1, 0.95))
    plt.tight_layout()
    plt.show()
