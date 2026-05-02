import time
import numpy as np
import queue
import threading
from src.physics import fast_p_laplacian_rhs

class LivePlotHook:
    def __init__(self, Nx, dx, p, h, epsilon, throttle=0.05):
        self.Nx = Nx
        self.dx = dx
        self.p = p
        self.h = h
        self.epsilon = epsilon
        self.throttle = throttle
        self.frame_queue = queue.Queue(maxsize=5)
        self._last_update = 0.0

    def wrapped_rhs(self, t, u):
        # Call the original with all required arguments
        dudt = fast_p_laplacian_rhs(t, u, self.p, self.dx, self.h, self.epsilon)
        now = time.perf_counter()
        if now - self._last_update >= self.throttle:
            self.frame_queue.put((t, u.copy()))
            self._last_update = now
        return dudt

    def start_plotter(self, title="Real‑time p‑Laplacian"):
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore
        app = pg.mkQApp(title)
        pw = pg.PlotWidget()
        pw.show()
        curve = pw.plot(pen=pg.mkPen('r', width=2))
        # … set axes limits, etc.
        def update():
            try:
                t, u = self.frame_queue.get_nowait()
                x = np.linspace(0, 1, self.Nx + 1)
                u_full = np.empty(self.Nx + 1)
                u_full[0] = self.h
                u_full[1:-1] = u
                u_full[-1] = 0.0
                curve.setData(x, u_full)
                pw.setTitle(f"t = {t:.4f}")
            except queue.Empty:
                pass
        timer = QtCore.QTimer()
        timer.timeout.connect(update)
        timer.start(int(self.throttle * 1000))
        QtCore.QCoreApplication.instance().exec_()
