import numpy as np
import matplotlib
matplotlib.use('Agg') # or TKAgg for interactive, but User asked for MP4
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Visualizer:
    def __init__(self, x_start, x_end, track_data=None):
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        self.x_lim = (x_start, x_end)
        self.track_data = track_data or {}
        
        self._setup_plots()
        
    def _setup_plots(self):
        # Panel 1: Speed
        self.ax1.set_ylabel("Speed (m/s)")
        self.ax1.set_title("Neural Speed Profile Replanning Demo")
        self.ax1.grid(True, alpha=0.3)
        self.ax1.set_xlim(self.x_lim)
        self.ax1.set_ylim(0, 45) # Approx 160 km/h max
        
        # Lines
        self.ln_gt, = self.ax1.plot([], [], 'k-', alpha=0.3, label='Original')
        self.ln_plan, = self.ax1.plot([], [], 'b--', alpha=0.5, label='Initial Plan')
        self.ln_exec, = self.ax1.plot([], [], 'r-', linewidth=2, label='Executed')
        self.ln_replan, = self.ax1.plot([], [], 'g-.', linewidth=2, label='Replanned')
        self.ln_curr, = self.ax1.plot([], [], 'ro', markersize=8)
        self.ax1.legend()
        
        # Text
        self.txt_info = self.ax1.text(0.02, 0.95, "", transform=self.ax1.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
        
        # Panel 2: Features
        self.ax2.set_xlabel("Position (m)")
        self.ax2.set_ylabel("Track Features")
        self.ax2.grid(True)
        
        # Plot static features if available
        if 'spdLimit' in self.track_data:
            dat = self.track_data['spdLimit']
            # Resample for clean plot
            x_eval = np.linspace(self.x_lim[0], self.x_lim[1], 500)
            y_eval = np.interp(x_eval, dat[:, 0], dat[:, 1])
            self.ax2.plot(x_eval, y_eval, 'm-', label='Limit (m/s)')
            
        if 'slope' in self.track_data:
             dat = self.track_data['slope']
             x_eval = np.linspace(self.x_lim[0], self.x_lim[1], 500)
             y_eval = np.interp(x_eval, dat[:, 0], dat[:, 1])
             # Normalize for vis? Slope is usually small (-30 to 30 permil)
             # Let's just plot it
             self.ax2.plot(x_eval, y_eval, 'c:', label='Slope')
             
        self.ax2.legend(loc='upper right', fontsize='small')

    def init_anim(self):
        self.ln_exec.set_data([], [])
        self.ln_replan.set_data([], [])
        self.ln_curr.set_data([], [])
        self.txt_info.set_text("")
        return self.ln_exec, self.ln_replan, self.ln_curr, self.txt_info

    def update(self, frame_data):
        # frame_data: dict(t, x, v, plan_v, replan_x, replan_v, latency)
        t = frame_data['t']
        x = frame_data['x']
        v = frame_data['v']
        
        # Update Executed Trace
        # We need history. Ideally the wrapper passes full history up to now?
        # Or we append here. Animator usually iterates.
        # Let's assume frame_data contains history arrays or we accumulate.
        # Matplotlib FuncAnimation is pull-based.
        
        x_hist = frame_data['x_hist']
        v_hist = frame_data['v_hist']
        
        self.ln_exec.set_data(x_hist, v_hist)
        self.ln_curr.set_data([x], [v])
        
        # Replan
        if frame_data['replan_active']:
            self.ln_replan.set_data(frame_data['replan_x'], frame_data['replan_v'])
        else:
            self.ln_replan.set_data([], [])
            
        # Text
        lat_txt = f"{frame_data['latency']:.1f} ms" if frame_data['latency'] > 0 else "-"
        info = f"Time: {t:.1f} s\nSpeed: {v*3.6:.1f} km/h\nReplan Latency: {lat_txt}"
        self.txt_info.set_text(info)
        
        return self.ln_exec, self.ln_replan, self.ln_curr, self.txt_info
