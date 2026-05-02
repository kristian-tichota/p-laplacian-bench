import time
import numpy as np
import queue
from src.physics import fast_p_laplacian_rhs

class LivePlotHook:
    def __init__(self, Nx, dx, p, h, epsilon, fps=144, sim_dt_per_frame=0.0002):
        self.Nx = Nx
        self.dx = dx
        self.p = p
        self.h = h
        self.epsilon = epsilon
        
        self.fps = fps
        self.sim_dt_per_frame = sim_dt_per_frame
        
        self.frame_queue = queue.Queue(maxsize=30)
        
        self._last_t = 0.0
        self._last_u = None
        self._next_frame_t = 0.0

    def wrapped_rhs(self, t, u):
        dudt = fast_p_laplacian_rhs(t, u, self.p, self.dx, self.h, self.epsilon)
        
        if self._last_u is None:
            self._last_t = t
            self._last_u = u.copy()
            self._next_frame_t = t
            return dudt

        if t > self._last_t + 1e-12:
            while self._next_frame_t <= t:
                weight = (self._next_frame_t - self._last_t) / (t - self._last_t)
                interp_u = self._last_u + weight * (u - self._last_u)
                
                self.frame_queue.put((self._next_frame_t, interp_u.copy()))
                self._next_frame_t += self.sim_dt_per_frame
                
            self._last_t = t
            self._last_u = u.copy()
            
        return dudt

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
        curve = pw.plot(pen=pg.mkPen('r', width=2))
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

        # --- State Tracking ---
        self.history = []
        self.auto_scroll = True
        self.last_data_time = time.time()
        self.is_finished = False

        self.replay_timer = QtCore.QTimer()
        self.replay_float_idx = 0.0
        self.replay_speed = 1.0

        def render_state(t_val, u_val, mode_text):
            x = np.linspace(0, 1, self.Nx + 1)
            u_full = np.empty(self.Nx + 1)
            u_full[0] = self.h
            u_full[1:-1] = u_val
            u_full[-1] = 0.0
            curve.setData(x, u_full)
            pw.setTitle(f"t = {t_val:.4f} ({mode_text})")

        def update_slider_bounds():
            if len(self.history) > 0:
                slider.setEnabled(True)
                slider.setMinimum(0)
                slider.setMaximum(len(self.history) - 1)

        # --- Live Pacing Loop ---
        def consume_live_queue():
            try:
                # Pull exactly one frame per GUI tick to maintain strict framerate
                t_val, u_val = self.frame_queue.get_nowait()
                self.history.append((t_val, u_val))
                self.last_data_time = time.time()
                
                update_slider_bounds()
                
                if self.auto_scroll:
                    slider.blockSignals(True)
                    slider.setValue(len(self.history) - 1)
                    slider.blockSignals(False)
                    render_state(t_val, u_val, "Live")
                    
            except queue.Empty:
                if not self.is_finished and len(self.history) > 0:
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
            t_val, u_val = self.history[idx]
            render_state(t_val, u_val, "History")

        def on_slider_released():
            if slider.value() == slider.maximum() and not self.is_finished:
                self.auto_scroll = True

        slider.sliderPressed.connect(on_slider_pressed)
        slider.valueChanged.connect(lambda: on_slider_moved() if not self.auto_scroll else None)
        slider.sliderReleased.connect(on_slider_released)

        def start_replay():
            if not self.history: 
                return
            
            self.auto_scroll = False
            self.replay_float_idx = 0.0
            slider.setValue(0)
            
            total_ticks_in_10s = 10.0 * self.fps
            self.replay_speed = max(1.0, len(self.history) / total_ticks_in_10s)
            
            self.replay_timer.start(int(1000 / self.fps))

        def step_replay():
            self.replay_float_idx += self.replay_speed
            idx = int(self.replay_float_idx)
            
            if idx >= len(self.history) - 1:
                idx = len(self.history) - 1
                self.replay_timer.stop()
                
            slider.blockSignals(True)
            slider.setValue(idx)
            slider.blockSignals(False)
            
            t_val, u_val = self.history[idx]
            render_state(t_val, u_val, "Replay")

        play_btn.clicked.connect(start_replay)
        self.replay_timer.timeout.connect(step_replay)

        QtCore.QCoreApplication.instance().exec_()
