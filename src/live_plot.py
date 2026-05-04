"""Live plotting hook using the spatial discretisation object."""

import queue
import time

import numpy as np

from .spatial_discretizations.base import SpatialDiscretization


class LivePlotHook:
    """
    Records integration snapshots for later replay.
    Implements the SolverHook protocol: callable(t, full_u).
    """

    def __init__(
        self, discretization: SpatialDiscretization, fps=144, sim_dt_per_frame=0.0002
    ):
        self.disc = discretization
        self.fps = fps
        self.sim_dt_per_frame = sim_dt_per_frame
        self.frame_queue = queue.Queue(maxsize=30)
        self._last_t = 0.0
        self._last_u = None
        self._next_frame_t = 0.0
        self.history = []

    def __call__(self, t, full_u):
        """Called by the solver with the full spatial solution."""
        if self._last_u is None:
            self._last_t = t
            self._last_u = full_u.copy()
            self._next_frame_t = t
            return
        if t > self._last_t + 1e-12:
            while self._next_frame_t <= t:
                weight = (self._next_frame_t - self._last_t) / (t - self._last_t)
                interp_u = self._last_u + weight * (full_u - self._last_u)
                self.frame_queue.put((self._next_frame_t, interp_u.copy()))
                self._next_frame_t += self.sim_dt_per_frame
            self._last_t = t
            self._last_u = full_u.copy()

    def start_plotter(self, title="Real-time p-Laplacian"):
        import pyqtgraph as pg
        from pyqtgraph.Qt import QtCore, QtWidgets

        app = pg.mkQApp(title)
        main_window = QtWidgets.QWidget()
        main_window.setWindowTitle(title)
        main_window.resize(800, 600)
        layout = QtWidgets.QVBoxLayout()
        main_window.setLayout(layout)

        pw = pg.PlotWidget()
        curve = pw.plot(pen=pg.mkPen("r", width=2))
        layout.addWidget(pw)

        controls_layout = QtWidgets.QHBoxLayout()
        play_btn = QtWidgets.QPushButton("Play Replay (10s)")
        play_btn.setEnabled(False)
        controls_layout.addWidget(play_btn)

        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setEnabled(False)
        controls_layout.addWidget(slider)
        layout.addLayout(controls_layout)
        main_window.show()

        self._history_local = []
        self.auto_scroll = True
        self.last_data_time = time.time()
        self.is_finished = False
        self.replay_timer = QtCore.QTimer()
        self.replay_float_idx = 0.0
        self.replay_speed = 1.0

        # Get the node coordinates once
        x = self.disc.get_node_coordinates()

        def render_state(t_val, u_val, mode_text):
            curve.setData(x, u_val)
            pw.setTitle(f"t = {t_val:.4f} ({mode_text})")

        def update_slider_bounds():
            if len(self._history_local) > 0:
                slider.setEnabled(True)
                slider.setMinimum(0)
                slider.setMaximum(len(self._history_local) - 1)

        def consume_live_queue():
            try:
                t_val, u_val = self.frame_queue.get_nowait()
                self._history_local.append((t_val, u_val))
                self.last_data_time = time.time()
                update_slider_bounds()
                if self.auto_scroll:
                    slider.blockSignals(True)
                    slider.setValue(len(self._history_local) - 1)
                    slider.blockSignals(False)
                    render_state(t_val, u_val, "Live")
            except queue.Empty:
                if not self.is_finished and len(self._history_local) > 0:
                    if time.time() - self.last_data_time > 1.0:
                        self.is_finished = True
                        play_btn.setEnabled(True)

        live_timer = QtCore.QTimer()
        live_timer.timeout.connect(consume_live_queue)
        live_timer.start(int(1000 / self.fps))

        def on_slider_pressed():
            self.auto_scroll = False
            self.replay_timer.stop()

        def on_slider_moved():
            idx = slider.value()
            t_val, u_val = self._history_local[idx]
            render_state(t_val, u_val, "History")

        def on_slider_released():
            if slider.value() == slider.maximum() and not self.is_finished:
                self.auto_scroll = True

        slider.sliderPressed.connect(on_slider_pressed)
        slider.valueChanged.connect(
            lambda: on_slider_moved() if not self.auto_scroll else None
        )
        slider.sliderReleased.connect(on_slider_released)

        def start_replay():
            if not self._history_local:
                return
            self.auto_scroll = False
            self.replay_float_idx = 0.0
            slider.setValue(0)
            total_ticks_in_10s = 10.0 * self.fps
            self.replay_speed = len(self._history_local) / total_ticks_in_10s
            self.replay_timer.start(int(1000 / self.fps))

        def step_replay():
            self.replay_float_idx += self.replay_speed
            idx = int(self.replay_float_idx)
            if idx >= len(self._history_local) - 1:
                idx = len(self._history_local) - 1
                self.replay_timer.stop()
            slider.blockSignals(True)
            slider.setValue(idx)
            slider.blockSignals(False)
            t_val, u_val = self._history_local[idx]
            render_state(t_val, u_val, "Replay")

        play_btn.clicked.connect(start_replay)
        self.replay_timer.timeout.connect(step_replay)
        QtCore.QCoreApplication.instance().exec_()
