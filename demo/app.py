import argparse
import contextlib
import os
import shutil
import sys
import time
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

from demo.planning import Planner
from demo.simulation import Simulator
from demo.visualizer import Visualizer
from demo.disturbances import StopDisturbance, SpeedLimitDisturbance, TractionLossDisturbance

def load_random_trajectory(h5_path, seed=None):
    if seed is not None:
        np.random.seed(seed)
        
    with h5py.File(h5_path, 'r') as f:
        # Get list of groups
        if 'railTrack' in f:
             # Just get keys that start with traj_
             keys = [k for k in f.keys() if k.startswith('traj_')]
        else:
             keys = list(f.keys())
             
        # Filter for originals (no _seg_)
        orig_keys = [k for k in keys if "_seg_" not in k]
        
        if not orig_keys:
            raise ValueError("No original trajectories found in H5.")
            
        # Pick one
        key = np.random.choice(orig_keys)
        g = f[key]
        
        # Load data
        data = {}
        data['t'] = g['t'][:]
        data['x'] = g['x'][:]
        data['v'] = g['v'][:]
        data['P'] = g['P'][:]

        # Task B: Extract scientific signals and bounds
        if 'accel' in g:
            data['accel_gt'] = g['accel'][:]
        if 'uOpt' in g:
            data['u_gt'] = g['uOpt'][:]
        elif 'FOpt' in g:
            data['u_gt'] = g['FOpt'][:]
            
        # Extract or Infer Bounds
        # Look for explicit bounds first
        a_min, a_max = None, None
        for k_min in ['vdotMin', 'a_min']:
            if k_min in g: a_min = float(g[k_min][()])
        for k_max in ['vdotMax', 'a_max']:
            if k_max in g: a_max = float(g[k_max][()])
            
        # Infer from accel if missing
        if (a_min is None or a_max is None) and 'accel_gt' in data:
            a_min = float(np.percentile(data['accel_gt'], 0.1)) if a_min is None else a_min
            a_max = float(np.percentile(data['accel_gt'], 99.9)) if a_max is None else a_max
            print(f"[Dataset] Inferred bounds from GT accel: [{a_min:.2f}, {a_max:.2f}]")
        
        data['a_min'] = a_min if a_min is not None else -1.2
        data['a_max'] = a_max if a_max is not None else 0.8
        
        # Jerk Bounds
        j_limit = None
        for kj in ['tracJerkMax', 'brkJerkMax', 'j_max']:
            if kj in g: 
                val = float(g[kj][()])
                j_limit = val if j_limit is None else min(j_limit, val)
        if j_limit is not None:
            data['j_max'] = j_limit
        
        # Try to load baseline results (Energy, total time)
        # Check standard names
        for k in ['energy', 'E', 'total_energy']:
            if k in g:
                data['energy_gt'] = g[k][()]
                break
        
        if 't_total' in g:
            data['t_total_gt'] = g['t_total'][()]
        else:
            data['t_total_gt'] = data['t'][-1] # fallback to raw array end
        
        # Track feats?
        # We need a way to get track feats. The H5Dataset class has a helper.
        # Let's simple load railTrack global and interp.
        track_data = {}
        if 'railTrack' in f:
            rt = f['railTrack']
            for k in ['slope', 'spdLimit', 'curvature']:
                if k in rt:
                    track_data[k] = rt[k][:]
                    
        return data, track_data, key

def interpolate_track_feats(x_query, track_data):
    # Returns [N, 3] (slope, limit, curv)
    N = len(x_query)
    out = np.zeros((N, 3), dtype=np.float32)
    
    if 'slope' in track_data:
        d = track_data['slope']
        out[:, 0] = np.interp(x_query, d[:, 0], d[:, 1])
        
    if 'spdLimit' in track_data:
        d = track_data['spdLimit']
        out[:, 1] = np.interp(x_query, d[:, 0], d[:, 1])
        
    if 'curvature' in track_data:
        d = track_data['curvature']
        out[:, 2] = np.interp(x_query, d[:, 0], d[:, 1])
        
    return out

def project_feasible_profile(x, v, a_min, a_max):
    """
    Enforce v_{k+1}^2 <= v_k^2 + 2*amax*dx (forward)
    and v_k^2 <= v_{k+1}^2 - 2*amin*dx (backward)
    """
    v = np.array(v).copy()
    x = np.array(x)
    dx = np.diff(x)
    
    # Forward pass: Acceleration limit
    for i in range(len(v)-1):
        v_max_sq = v[i]**2 + 2 * a_max * dx[i]
        if v[i+1]**2 > v_max_sq:
            v[i+1] = np.sqrt(max(0, v_max_sq))
            
    # Backward pass: Braking limit
    for i in range(len(v)-2, -1, -1):
        # v_k^2 <= v_{k+1}^2 - 2*a_min*dx_k  (since a_min is e.g. -1.2, -2*a_min is +2.4)
        v_feasible_sq = v[i+1]**2 - 2 * a_min * dx[i]
        if v[i]**2 > v_feasible_sq:
            v[i] = np.sqrt(max(0, v_feasible_sq))
            
    return v

# --- FEASIBILITY DIAGNOSTICS ---
def check_feasibility(x_p, v_p, label="Plan", limits=None):
    if len(v_p) < 2: return None
    dx = np.diff(x_p)
    dv = np.diff(v_p)
    v_mid = (v_p[:-1] + v_p[1:]) / 2
    mask = dx > 0.1
    if not np.any(mask): return None
    a_implied = v_mid[mask] * (dv[mask] / dx[mask])
    a_max_p = np.max(a_implied)
    a_min_p = np.min(a_implied)
    
    a_min_lim = -1.2
    a_max_lim = 0.8
    if limits:
        a_min_lim, a_max_lim = limits
    
    exceeds = a_max_p > a_max_lim + 0.1 or a_min_p < a_min_lim - 0.1
    print(f"[{label} Diagnostics] Implied Accel: [{a_min_p:.2f}, {a_max_p:.2f}] m/s^2")
    if exceeds:
        print(f"  WARNING: {label} profile may be untrackable under simulator limits [{a_min_lim}, {a_max_lim}]")
        
    return {
        'min_a': a_min_p,
        'max_a': a_max_p,
        'exceeds': exceeds
    }

class SimulationRunner:
    def __init__(self, config):
        self.cfg = config
        self.planner = None
        self.sim = None
        self.dist = None
        self.traj_data = None
        self.track_global = None
        self.events = {
            'stop_brake_start': None,
            'stop_hold_start': None,
            'stop_hold_end': None, # Release
            'replan_start': None,
            'terminal_start': None
        }
        self.replan_data = {
            'active': False,
            'x': [], 'v': [], 't': [],
            'latency_ms': 0.0,
            'feasible_v': []
        }
        self.history = {'x': [], 'v': [], 't': [], 'a': [], 'j': [], 'a_cmd': [], 'v_ref': [], 'e_t': []}
        self.logs = []
        self.cumulative_energy = 0.0
        self.max_j_app = 0.0

        self._setup()

    def log(self, msg):
        # Internal log + print
        # print(msg)
        self.logs.append(msg)

    def _setup(self):
        # 1. Load Model/Planner
        if self.cfg.run_dir is None:
             raise ValueError("SimulationRunner: run_dir must be specified (NoneType found). Ensure a Run ID is selected.")
             
        try:
            self.planner = Planner(self.cfg.run_dir, ckpt_path=self.cfg.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load planner: {e}")

        # Load Trajectory
        self.traj_data, self.track_global, self.traj_name = load_random_trajectory(
            self.cfg.h5_path, seed=self.cfg.seed
        )
        
        # Setup Initial State
        x0, xf = self.traj_data['x'][0], self.traj_data['x'][-1]
        v0 = self.traj_data['v'][0]
        self.xf = xf
        self.P_initial = self.traj_data['P']
        if self.P_initial.ndim == 1: self.P_initial = self.P_initial[np.newaxis, :]
        self.M_train = float(self.P_initial[0, 7])

        # Limits
        self.a_min = self.traj_data.get('a_min', -1.2)
        self.a_max = self.traj_data.get('a_max', 0.8)
        self.j_max = self.traj_data.get('j_max', self.cfg.j_max)

        # Simulator
        self.sim = Simulator(x0, v0, dt=0.2, tau=self.cfg.tau, j_max=self.j_max, kp=self.cfg.kp,
                             xf=xf, x_stop_window=self.cfg.x_stop_window, x_eps=self.cfg.x_eps, v_eps=self.cfg.v_eps,
                             gamma=self.cfg.gamma, a_max=self.a_max, a_min=self.a_min)
        
        # Initial Plan
        self.sim.set_plan(self.traj_data['x'], self.traj_data['v'], t_start=0.0)

        # Disturbance
        self._setup_disturbance()
        
    def _setup_disturbance(self):
        dist_t = 20.0
        braking_accel = self.cfg.stop_braking_accel
        if self.cfg.scenario == 'stop':
            if self.cfg.stop_respect_limits:
                braking_accel = self.a_min
            self.dist = StopDisturbance(dist_t, duration=10.0, braking_accel=braking_accel)
        elif self.cfg.scenario == 'speed_limit':
            self.dist = SpeedLimitDisturbance(dist_t, limit=10.0, duration=30.0)
        elif self.cfg.scenario == 'traction_loss':
            self.dist = TractionLossDisturbance(dist_t, factor=0.5, duration=20.0)

    def step(self):
        # 1. Step Sim
        self.sim.step(disturbance=self.dist)
        
        # 2. Capture State
        state = self.sim.state
        t, x, v, a = state['t'], state['x'], state['v'], state['a']
        
        # 3. Energy & Jerk
        dt = self.sim.dt
        
        self.history['t'].append(t)
        self.history['x'].append(x)
        self.history['v'].append(v)
        self.history['a'].append(a)
        
        # Collect controller internal states if available
        if 'v_ref' in state:
            self.history['v_ref'].append(state['v_ref'])
        if 'e_t' in state:
            self.history['e_t'].append(state['e_t'])
        
        # Jerk calc
        j_curr = 0.0
        if len(self.history['a']) > 1:
            j_curr = abs(self.history['a'][-1] - self.history['a'][-2]) / dt
        self.max_j_app = max(self.max_j_app, j_curr)
        self.history['j'].append(j_curr)
        # Store a_cmd? For simplicity here using actual a.
        # Original code used a_cmd for j_app. 
        # Sim.state['a'] is effectively a_cmd (the output of the controller before integration, strictly speaking)
        # because Simulator updates `self.state['a'] = a` where `a` is the clamped command.
        self.history['a_cmd'].append(a)
        
        # Energy
        dE = self.M_train * max(0, a) * v * dt
        self.cumulative_energy += dE
        
        # 4. Event Detection & Replanning
        self._check_events(t, x, v)
        self._check_replan(t, x, v)
        
        return state

    def _check_events(self, t, x, v):
        if self.dist:
             # Trigger detection
             if self.dist.t_start <= t < (self.dist.t_start + self.sim.dt):
                 if self.cfg.scenario == 'stop':
                      if self.events['stop_brake_start'] is None:
                          self.events['stop_brake_start'] = t
                          self.log(f"Disturbance Triggered at t={t:.2f}")

             if isinstance(self.dist, StopDisturbance):
                 if self.dist.state_name == "HOLDING" and self.events['stop_hold_start'] is None:
                     self.events['stop_hold_start'] = t
                     self.log(f"Stop Reached at t={t:.2f}")
                 if self.dist.state_name == "RELEASED" and self.events['stop_hold_end'] is None:
                     self.events['stop_hold_end'] = t
                     self.log(f"Release Triggered at t={t:.2f}")
        
        if self.sim.terminal_mode and self.events['terminal_start'] is None:
            self.events['terminal_start'] = t

    def _check_replan(self, t, x, v):
        trigger = False
        if self.cfg.scenario == 'stop' and isinstance(self.dist, StopDisturbance):
            if self.dist.state_name == "RELEASED" and not self.replan_data['active']:
                trigger = True
                self.log(f"Stop released. Triggering Replanning from x={x:.1f}, v={v:.2f}")
        elif self.dist and self.dist.t_start <= t and self.cfg.scenario != 'stop' and not self.replan_data['active']:
             trigger = True
             self.log(f"Disturbance Triggered at t={t:.1f}!")

        if trigger:
            self._execute_replan(t, x, v)
            
    def _execute_replan(self, t, x, v):
        self.events['replan_start'] = t
        t0 = time.time()
        
        x_rem = np.linspace(x, self.xf, 128)
        feats_rem = interpolate_track_feats(x_rem, self.track_global)
        delay_est = 12.0 if self.cfg.scenario == 'stop' else 0.0
        
        if self.cfg.scenario == 'speed_limit':
             feats_rem[:, 1] = np.minimum(feats_rem[:, 1], 10.0)
             
        new_x, new_v_raw = self.planner.plan(
            x, v, self.xf, 0.0, # vf=0
            track_feats=feats_rem,
            P_ref=self.P_initial,
            dur_policy=self.cfg.dur_policy,
            delay=delay_est
        )
        
        new_v_feasible = project_feasible_profile(new_x, new_v_raw, self.a_min, self.a_max)
        
        # Apply Plan
        self.sim.set_plan(new_x, new_v_feasible, t_start=t)
        
        latency = (time.time() - t0) * 1000
        self.replan_data.update({
            'active': True,
            'x': new_x, 'v': new_v_raw, 't': t, # approximate t? No, plan is v(x)
            'feasible_v': new_v_feasible,
            'latency_ms': latency
        })
        self.log(f"Replanned in {latency:.2f} ms")

    def run_full(self):
        t_max = 200.0
        while self.sim.state['t'] < t_max:
            self.step()
            if self.sim.stopped:
                self.log(f"Goal reached at t={self.sim.state['t']:.2f}")
                break
        return self.get_results()

    def get_results(self):
        # Package everything
        # Calculate final metrics for the Web App
        t_arrival_exec = self.history['t'][-1] if self.history['t'] else 0.0
        t_arrival_gt = self.traj_data.get('t_total_gt', self.traj_data['t'][-1])
        total_delay = t_arrival_exec - t_arrival_gt
        
        limit_data = self.track_global.get('spdLimit')
        track_payload = {}
        if limit_data is not None:
            track_payload = {'x': limit_data[:, 0], 'limit': limit_data[:, 1]}
        
        return {
            'history': self.history,
            'events': self.events,
            'replan': self.replan_data,
            'metrics': {
                'energy_mj': self.cumulative_energy / 1e6,
                'delay': total_delay,
                't_arrival_exec': t_arrival_exec,
                't_arrival_gt': t_arrival_gt
            },
            'track': track_payload,
            'sim_limits': {'a_max': self.a_max, 'a_min': self.a_min, 'j_max': self.j_max},
            'j_app_max': self.max_j_app,
            'logs': self.logs,
            'traj_x': self.traj_data['x'],
            'traj_v': self.traj_data['v']
        }

def run_simulation_service(config_dict):
    # Adapter for API
    class Config:
        def __init__(self, d):
            for k,v in d.items(): setattr(self, k, v)
    
    # Defaults
    defaults = {
        'scenario': 'stop', 'seed': 42, 'run_dir': None, 'model_path': None, 'h5_path': 'data/dataset.h5',
        'dur_policy': 'keep', 'project_feasible': True, 'tau': 2.0, 'j_max': 0.8, 'kp': 1.0,
        'x_stop_window': 100.0, 'x_eps': 1.0, 'v_eps': 0.2, 'gamma': 1.5,
        'stop_braking_accel': -1.2, 'stop_respect_limits': False
    }
    defaults.update(config_dict)
    cfg = Config(defaults)
    
    runner = SimulationRunner(cfg)
    results = runner.run_full()
    return results

def main():
    parser = argparse.ArgumentParser(description="Neural Surrogate Demo")
    parser.add_argument("--scenario", type=str, default="stop", choices=['stop', 'speed_limit', 'traction_loss', 'none'])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_mp4", type=str, default=None, help="MP4 file name inside --output_dir (opt-in, needs FFmpeg on PATH)")
    parser.add_argument("--export_json", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None, help="Path to specific .pt checkpoint file")
    parser.add_argument("--output_dir", type=str, default="demo_outputs", help="Directory to save outputs")
    parser.add_argument("--h5_path", type=str, default="data/dataset.h5")
    
    parser.add_argument("--dur_policy", type=str, default="keep", choices=['keep', 'recover'], 
                        help="Replanning duration policy: 'keep' (maintain pace) or 'recover' (catch up)")
    
    parser.add_argument("--project_feasible", action="store_true", help="Project replanned profiles onto physical constraints")
    
    # Control Parameters
    parser.add_argument("--tau", type=float, default=2.0, help="Reference lag time constant (s)")
    parser.add_argument("--j_max", type=float, default=0.8, help="Max jerk (m/s^3)")
    parser.add_argument("--kp", type=float, default=1.0, help="Proportional tracking gain")

    # Stopping Parameters
    parser.add_argument("--x_stop_window", type=float, default=100.0, help="Distance from xf to start terminal braking")
    parser.add_argument("--x_eps", type=float, default=1.0, help="Distance tolerance for arrival")
    parser.add_argument("--v_eps", type=float, default=0.2, help="Speed tolerance for arrival")
    parser.add_argument("--gamma", type=float, default=1.5, help="Terminal mode safety factor")
    
    # Task: Stop Disturbance Config
    parser.add_argument("--stop_braking_accel", type=float, default=-1.2, help="Braking accel during STOP disturbance")
    parser.add_argument("--stop_respect_limits", action="store_true", help="If set, use dataset a_min for stop disturbance")

    args = parser.parse_args()
    
    # Setup Output Dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Resolve Paths
    mp4_path = os.path.join(args.output_dir, args.save_mp4) if args.save_mp4 else None
    json_path = os.path.join(args.output_dir, args.export_json) if args.export_json else None
    
    run_dir = args.run_dir
    model_path = args.model_path
    
    # auto-infer run_dir from model_path if run_dir missing
    if run_dir is None and model_path is not None:
        # assume model_path is runs/RUN_NAME/ckpt.pt
        run_dir = os.path.dirname(model_path)
        print(f"Inferred run_dir: {run_dir}")
        
    if run_dir is None:
        parser.error("--run_dir is required (a trained run folder such as runs/run_<timestamp>), or pass --model_path")
    
    # 0. Load Components
    print(f"Loading Planner from {run_dir} (ckpt={model_path})...")
    try:
        planner = Planner(run_dir, ckpt_path=model_path)
    except Exception as e:
        print(f"Failed to load planner: {e}")
        return

    print(f"Loading Trajectory from {args.h5_path}...")
    traj_data, track_global, traj_name = load_random_trajectory(args.h5_path, seed=args.seed)
    print(f"Selected Trajectory: {traj_name}")
    
    x0, xf = traj_data['x'][0], traj_data['x'][-1]
    v0, vf = traj_data['v'][0], traj_data['v'][-1]
    
    # P_initial from GT
    P_initial = traj_data['P']
    # Check shape
    if P_initial.ndim == 1:
        P_initial = P_initial[np.newaxis, :] # [1, 8]
    
    # Task B: Alignment
    # Simulator parameters from dataset
    a_min = traj_data.get('a_min', -1.2)
    a_max = traj_data.get('a_max', 0.8)
    j_max = traj_data.get('j_max', args.j_max)
    print(f"Using Simulator Limits: a=[{a_min:.2f}, {a_max:.2f}] m/s^2, jerk={j_max:.2f} m/s^3")

    # 1. Initial Plan (User requested GT as initial plan)
    print("Using Ground Truth as Initial Plan...")
    plan_x, plan_v = traj_data['x'], traj_data['v']
    
    # 2. Setup Sim
    sim = Simulator(x0, v0, dt=0.2, tau=args.tau, j_max=j_max, kp=args.kp,
                    xf=xf, x_stop_window=args.x_stop_window, x_eps=args.x_eps, v_eps=args.v_eps,
                    gamma=args.gamma, a_max=a_max, a_min=a_min)
    sim.set_plan(plan_x, plan_v, t_start=0.0)
    
    # 3. Setup Disturbance
    dist = None
    dist_t = 20.0 # Trigger at t=20s
    braking_accel_val = args.stop_braking_accel
    
    if args.scenario == 'stop':
        if args.stop_respect_limits:
            braking_accel_val = a_min
            print(f"[Config] Stop Disturbance respecting limits: a_brake={braking_accel_val:.2f}")
        else:
            print(f"[Config] Stop Disturbance using CLI override: a_brake={braking_accel_val:.2f}")
            
        dist = StopDisturbance(dist_t, duration=10.0, braking_accel=braking_accel_val)
        # Annotation moved to after plot setup
    elif args.scenario == 'speed_limit':
        dist = SpeedLimitDisturbance(dist_t, limit=10.0, duration=30.0)
    elif args.scenario == 'traction_loss':
        dist = TractionLossDisturbance(dist_t, factor=0.5, duration=20.0)
        
    # 4. Loop
    t_max = 200.0
    history_x = []
    history_v = []
    
    replan_active = False
    replan_x, replan_v = [], []
    latency_ms = 0.0
    
    # For Animation
    # Panel 1: v vs x
    # Panel 2: v vs t
    # Panel 3: a vs t
    # Panel 4: Features
    fig, (ax1, ax3, ax4, ax2) = plt.subplots(4, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [2, 2, 1.5, 1]})
    plt.subplots_adjust(hspace=0.4)
    
    # Mass from P_ref
    M_train = float(P_initial[0, 7])
    
    initial_diag = check_feasibility(traj_data['x'], traj_data['v'], "Initial", limits=(a_min, a_max))

    # Instrumentation for Scientific v3
    t_trigger = None
    t_stop = None
    t_release = None
    t_replan_start = None
    t_terminal_start = None
    
    a_cmd_history = []
    
    events = {
        'stop_brake_start': None,
        'stop_hold_start': None,
        'stop_hold_end': None,
        'replan_start': None,
        'terminal_start': None
    }

    # Setup static v-x
    mode_str = "Energy-Optimal (Fixed Dur)" if args.dur_policy == 'keep' else "Schedule Recovery"
    ax1.set_title(f"Scenario: {args.scenario} | {traj_name} | Mode: {mode_str}")
    ax1.set_xlim(x0, xf)
    ax1.set_ylim(0, np.max(traj_data['v']) * 1.2 + 5)
    ax1.set_ylabel("v (m/s)")
    ax1.plot(traj_data['x'], traj_data['v'], 'k-', alpha=0.3, label='GT')
    ln_exec, = ax1.plot([], [], 'r-', linewidth=2, label='Executed')
    ln_replan, = ax1.plot([], [], 'g-', linewidth=2, label='Replanned')
    ln_curr, = ax1.plot([], [], 'ro')
    ln_trigger = ax1.axvline(x=-100, color='k', linestyle=':', label='Disturbance')
    ln_replan_start = ax1.axvline(x=-100, color='g', linestyle='-', label='Replan Start')
    ax1.legend(loc='upper right', fontsize='small')
    
    # Setup static v-t
    ax3.set_title("Speed vs Time")
    ax3.set_xlim(0, t_max)
    ax3.set_ylim(0, np.max(traj_data['v']) * 1.2 + 5)
    ax3.set_ylabel("v (m/s)")
    ax3.plot(traj_data['t'], traj_data['v'], 'k-', alpha=0.3, label='GT Target')
    ln_v_t, = ax3.plot([], [], 'r-', label='Executed')
    ln_v_ref, = ax3.plot([], [], 'b--', alpha=0.7, label='Reference (Active Plan)')
    t_trigger_line = ax3.axvline(x=-100, color='k', linestyle=':', label='Trigger')
    t_replan_line = ax3.axvline(x=-100, color='g', linestyle='-', label='Replan')
    ax3.legend(loc='upper right', fontsize='small')

    # Setup static a-t
    ax4.set_title("Acceleration vs Time")
    ax4.set_xlim(0, t_max)
    ax4.set_ylim(a_min - 0.5, a_max + 0.5)
    ax4.set_ylabel("a (m/s^2)")
    ax4.axhline(y=a_max, color='k', linestyle=':', alpha=0.5, label='Limits')
    ax4.axhline(y=a_min, color='k', linestyle=':', alpha=0.5)
    
    # Task: Plot Annotation for Stop
    if args.scenario == 'stop':
         # braking_accel_val needs to be re-derived or we just use what we know
         # easier to re-derive logic briefly or check dist type
         val = args.stop_braking_accel
         if args.stop_respect_limits: val = a_min
         
         ax4.text(0.02, 0.1, f"STOP Accel: {val:.2f} m/s² (Exogenous)",
                 transform=ax4.transAxes, color='red', fontsize='small', bbox=dict(facecolor='white', alpha=0.7))
    
    # Task D: GT Accel Overlay
    if 'accel_gt' in traj_data:
        # Map a_gt(t)
        ax4.plot(traj_data['t'], traj_data['accel_gt'], 'k-', alpha=0.2, label='GT Optimal Accel')
        
    ln_a_t, = ax4.plot([], [], 'r-', label='Executed')
    ax4.legend(loc='upper right', fontsize='small')

    txt_info = ax1.text(0.02, 0.95, "", transform=ax1.transAxes, bbox=dict(facecolor='white', alpha=0.8), fontsize='small')

    # Features
    if 'spdLimit' in track_global:
        d = track_global['spdLimit']
        ax2.plot(d[:, 0], d[:, 1], 'm-', label='Limit')
    ax2.set_ylabel("Limit (m/s)")
    ax2.set_xlabel("Position (m)")
    ax2.legend(fontsize='small')
    
    # Task E: Feasible replan line
    ln_replan_feasible, = ax1.plot([], [], 'g--', linewidth=1.5, label='Feasible Replan')
    ax1.legend(loc='upper right', fontsize='small')
    
    # MP4 output is opt-in (--save_mp4) and needs FFmpeg on PATH.
    writer = None
    if mp4_path:
        if shutil.which("ffmpeg"):
            writer = animation.FFMpegWriter(fps=15)
        else:
            print("FFmpeg not found on PATH: skipping MP4 output.")
    
    current_plan_v_interp = plan_v
    current_plan_x_interp = plan_x
    
    print(f"Starting Simulation...")
    
    # Tracking for T-Plot
    history_t = []
    history_a = []
    cumulative_energy = 0.0
    max_j = 0.0
    
    t_replan_val = -100
    t_trigger_val = -100
    
    with (writer.saving(fig, mp4_path, dpi=100) if writer else contextlib.nullcontext()):
        while sim.state['t'] < t_max:
            t = sim.state['t']
            x = sim.state['x']
            v = sim.state['v']
            a_prev_exec = sim.state['a']
            
            # Control: Schedule Loop handled internally
            sim.step(disturbance=dist)
            
            # State after step
            new_a = sim.state['a']
            dt = sim.dt
            jerk = abs(new_a - a_prev_exec) / dt
            max_j = max(max_j, jerk)
            
            # Energy Increment: P = M*a*v (traction work)
            dE = M_train * max(0, new_a) * v * dt
            cumulative_energy += dE
            
            history_x.append(x)
            history_v.append(v)
            history_t.append(t)
            history_a.append(new_a)
            
            # Check Replan Trigger
            # Scenario: STOP
            trigger_replan = False
            
            if args.scenario == 'stop' and dist:
                # Check for transition to RELEASED
                if dist.state_name == "RELEASED" and not replan_active:
                     print(f"[App] Stop released. Triggering Replanning from x={x:.1f}, v={v:.2f}")
                     trigger_replan = True
                     # Mark replan start
                     ln_replan_start.set_xdata([x])
                     t_replan_val = t
                     t_replan_line.set_xdata([t])
                     
            elif dist and dist.check_trigger(t):
                 if args.scenario != 'stop':
                     print(f"Disturbance Triggered at t={t:.1f}!")
                     trigger_replan = True
                     t_trigger_val = t
                     t_trigger_line.set_xdata([t])
                 else:
                     # Just mark the trigger time for stop scenario for viz
                     ln_trigger.set_xdata([x])
                     t_trigger_val = t
                     t_trigger_line.set_xdata([t])

            # Applied acceleration for jerk metrics
            a_cmd_history.append(new_a)
            
            # --- Event Tracking ---
            if dist and dist.t_start <= t < (dist.t_start + sim.dt):
                if events['stop_brake_start'] is None:
                    events['stop_brake_start'] = t
                    print(f"[Metrics] Disturbance Triggered at t={t:.2f}")

            if args.scenario == 'stop' and isinstance(dist, StopDisturbance):
                # Detect Stop (Brake -> Hold)
                if dist.state_name == "HOLDING" and events['stop_hold_start'] is None:
                    events['stop_hold_start'] = t
                    print(f"[Metrics] Stop Reached at t={t:.2f}")
                # Detect Release (Hold -> Released)
                if dist.state_name == "RELEASED" and events['stop_hold_end'] is None:
                    events['stop_hold_end'] = t
                    print(f"[Metrics] Release Triggered at t={t:.2f}")

            if sim.terminal_mode and events['terminal_start'] is None:
                events['terminal_start'] = t

            trigger_replan = False
            if args.scenario == 'stop' and isinstance(dist, StopDisturbance):
                # Replan exactly at release
                if dist.state_name == "RELEASED" and not replan_active:
                    trigger_replan = True
            elif dist and dist.t_start <= t and not replan_active and args.scenario != 'stop':
                 # Simple trigger for other disturbances
                 trigger_replan = True
                 
            if trigger_replan:
                t0 = time.time()
                events['replan_start'] = t
                x_rem = np.linspace(x, xf, 128)
                feats_rem = interpolate_track_feats(x_rem, track_global)
                delay_est = 12.0 if args.scenario == 'stop' else 0.0
                
                if args.scenario == 'speed_limit':
                     feats_rem[:, 1] = np.minimum(feats_rem[:, 1], 10.0) 
                     
                new_x, new_v_raw = planner.plan(x, v, xf, vf, 
                                               track_feats=interpolate_track_feats(x_rem, track_global),
                                               P_ref=P_initial,
                                               dur_policy=args.dur_policy,
                                               delay=delay_est) 
                
                # Task 2: Feasibility Diagnostics Consistency
                raw_diag = check_feasibility(new_x, new_v_raw, "Replan (RAW)", limits=(a_min, a_max))
                
                # Always project for controller feasibility
                # User request: "Always Track Feasible Replanned Profile"
                # args.project_feasible now controls whether we overwrite the plan viz, 
                # but the controller MUST get the feasible one.
                
                new_v_feasible = project_feasible_profile(new_x, new_v_raw, a_min, a_max)
                feas_diag = check_feasibility(new_x, new_v_feasible, "Replan (FEASIBLE)", limits=(a_min, a_max))

                # Use feasible profile for control
                sim.set_plan(new_x, new_v_feasible, t_start=t)
                
                if args.project_feasible:
                    current_plan_v_interp = new_v_feasible
                else:
                    current_plan_v_interp = new_v_raw
                
                latency_ms = (time.time() - t0) * 1000
                replan_active = True
                replan_x, replan_v = new_x, new_v_raw
                # Plan updated above with set_plan
                current_plan_x_interp = replan_x
                print(f"Replanned in {latency_ms:.2f} ms")

            # Update Plot
            ln_exec.set_data(history_x, history_v)
            ln_curr.set_data([x], [v])
            if replan_active:
                ln_replan.set_data(replan_x, replan_v)
                if args.project_feasible:
                    ln_replan_feasible.set_data(replan_x, current_plan_v_interp)
            
            # v-t plot
            ln_v_t.set_data(history_t, history_v)
            # Reference curve in v-t plot: history of v_target passed to Simulator
            v_ref_history = sim.history['v_ref']
            ln_v_ref.set_data(history_t, v_ref_history[1:]) # sim.history starts with v0, len is N+1
            
            # a-t plot
            ln_a_t.set_data(history_t, history_a)
            
            # Status Overlay
            dist_rem = xf - x
            eta_str = "--:--"
            if v > 1.0:
                eta_s = dist_rem / v
                eta_str = f"{int(eta_s // 60):02d}:{int(eta_s % 60):02d}"

            # Real-time Metrics
            e_mj = cumulative_energy / 1e6
            curr_jerk = 0.0
            if len(a_cmd_history) > 1:
                curr_jerk = abs(a_cmd_history[-1] - a_cmd_history[-2]) / sim.dt
                
            # Get latest e_t
            curr_e_t = sim.history['e_t'][-1] if sim.history.get('e_t') else 0.0

            info_text = (
                f"t: {t:.1f}s | x: {x:.0f}m\n"
                f"v: {v*3.6:.1f}km/h | e_t: {curr_e_t:+.1f}s\n"
                f"a: {new_a:.2f} m/s^2 | j_app: {curr_jerk:.2f}\n"
                f"E_proxy: {e_mj:.2f} MJ\n"
                f"ETA: {eta_str} | Replan: {latency_ms:.1f}ms"
            )
            if sim.terminal_mode:
                info_text += "\nTERMINAL MODE ACTIVE"
            txt_info.set_text(info_text)
            
            if writer:
                writer.grab_frame()
            
            # Arrival condition
            if sim.stopped:
                t_arrival_exec = t
                print(f"Goal reached at t={t_arrival_exec:.2f}")
                ax3.axvline(x=t_arrival_exec, color='b', linestyle='--', label='Arrival')
                ax3.legend(loc='upper right', fontsize='small')
                break
            
    # Post-Loop Logging
    history_a_arr = np.array(history_a)
    history_t_arr = np.array(history_t)
    history_cmd_a = np.array(a_cmd_history)
    
    # Task 1: Multi-tier Jerk Metrics
    # Note: j_app has length N-1
    j_app = np.abs(np.diff(history_cmd_a)) / sim.dt
    j_obs = np.abs(np.diff(history_a_arr)) / sim.dt
    
    max_j_app = np.max(j_app) if len(j_app) > 0 else 0.0
    max_j_obs = np.max(j_obs) if len(j_obs) > 0 else 0.0
    
    # Task: Steady Jerk Exclusions (Refined)
    # Exclude Stop Boundary Windows (+/- 1s) and Terminal Activation (+/- 1s)
    mask_steady = np.ones_like(j_app, dtype=bool)
    
    def mask_window(t_center, half_width=1.0):
        if t_center is None: return
        idx_start = int(max(0, (t_center - half_width)) / sim.dt)
        idx_end = int(min(len(mask_steady), (t_center + half_width) / sim.dt))
        # j_app indices are aligned with t[1:] basically... close enough for 1s window
        mask_steady[idx_start:idx_end+1] = False
        
    mask_window(events['stop_brake_start'])
    mask_window(events['stop_hold_end'])
    mask_window(events['terminal_start'])
    
    # Also exclude Replan start (handled by old logic, let's include it here for completeness)
    mask_window(events['replan_start'], half_width=2.0) # 2s for replan transient
    
    # Exclude Arrival Transient (Final stop jerk)
    if sim.stopped:
        mask_window(sim.state['t'], half_width=1.0)
    
    max_j_steady = np.max(j_app[mask_steady]) if np.any(mask_steady) else 0.0
    # Removed old block for max_j_steady calculation as it's superseded by the refined block above.
    # max_j_steady is already computed using mask_steady. The line below was using undefined 'mask' and is redundant.
    # max_j_steady = np.max(j_app[mask]) if np.any(mask) else 0.0
    max_a_exec = np.max(np.abs(history_a_arr))
    
    # Task: Split Metrics (Stop vs Nominal)
    # Define Stop Interval Mask
    mask_stop = np.zeros_like(history_a_arr, dtype=bool)
    if events['stop_brake_start'] is not None and events['stop_hold_end'] is not None:
        idx_start = int(events['stop_brake_start'] / sim.dt)
        idx_end = int(events['stop_hold_end'] / sim.dt)
        # Safe bounds check
        idx_start = max(0, min(idx_start, len(mask_stop)-1))
        idx_end = max(0, min(idx_end, len(mask_stop)-1))
        mask_stop[idx_start:idx_end+1] = True
        
    mask_nominal = ~mask_stop
    
    # Also exclude Arrival from Nominal to avoid Sim-Termination artifacts (e.g. v_eps cut)
    if sim.stopped:
        idx_cut = int(1.0 / sim.dt)
        if len(mask_nominal) > idx_cut:
            mask_nominal[-idx_cut:] = False
    
    # Accel
    max_a_stop = np.max(np.abs(history_a_arr[mask_stop])) if np.any(mask_stop) else 0.0
    max_a_nom = np.max(np.abs(history_a_arr[mask_nominal])) if np.any(mask_nominal) else 0.0
    
    # Jerk (j_app has len = N-1)
    # Align masks to j_app
    mask_stop_j = mask_stop[1:] 
    mask_nominal_j = mask_nominal[1:]
    
    max_j_stop = np.max(j_app[mask_stop_j]) if np.any(mask_stop_j) else 0.0
    max_j_nom = np.max(j_app[mask_nominal_j]) if np.any(mask_nominal_j) else 0.0
    
    # Debug moved to end
    
    # Task 3: Delay Breakdown
    t_arrival_exec = sim.state['t']
    t_arrival_gt = traj_data.get('t_total_gt', traj_data['t'][-1])
    
    stop_brake = 0.0
    stop_hold = 0.0
    post_stop = 0.0
    
    if events['stop_brake_start'] is not None and events['stop_hold_start'] is not None:
        stop_brake = events['stop_hold_start'] - events['stop_brake_start']
    if events['stop_hold_start'] is not None and events['stop_hold_end'] is not None:
        stop_hold = events['stop_hold_end'] - events['stop_hold_start']
    if events['stop_hold_end'] is not None:
        post_stop = t_arrival_exec - events['stop_hold_end']
    else:
        post_stop = t_arrival_exec # No stop occurred
        
    total_delay = t_arrival_exec - t_arrival_gt
    
    total_energy_mj = cumulative_energy / 1e6
    gt_energy_mj = traj_data.get('energy_gt', None)
    
    print("\n" + "="*40)
    print("      SCIENTIFIC METRICS SUMMARY (v3)")
    print("="*40)
    print(f"Dataset Bounds:       a=[{a_min:.2f}, {a_max:.2f}], j={j_max:.2f}")
    print("-" * 40)
    print(f"Durations & Delay Breakdown:")
    print(f"  Braking phase:      {stop_brake:.2f} s")
    print(f"  Hold phase:         {stop_hold:.2f} s")
    print(f"  Post-stop travel:   {post_stop:.2f} s")
    print(f"  TOTAL DURATION:     {t_arrival_exec:.2f} s (GT: {t_arrival_gt:.2f} s)")
    print(f"  TOTAL DELAY:        {total_delay:.2f} s")
    print("-" * 40)
    print(f"Dynamics Metrics:")
    print(f"  Max Acceleration (Total):   {max_a_exec:.3f} m/s^2")
    if args.scenario == 'stop':
        print(f"    - Stop Disturbance:       {max_a_stop:.3f} m/s^2 (Exogenous)")
        print(f"    - Nominal Tracking:       {max_a_nom:.3f} m/s^2")
        
    print(f"  Max Applied Jerk (Total):   {max_j_app:.3f} m/s^3 (Limit: {j_max:.2f})")
    if args.scenario == 'stop':
        print(f"    - Stop Disturbance:       {max_j_stop:.3f} m/s^3")
        print(f"    - Nominal Tracking:       {max_j_nom:.3f} m/s^3")
        
    print(f"  Steady Track Jerk:          {max_j_steady:.3f} m/s^3 (Excl. Transients)")
    print(f"  Observed max Jerk:          {max_j_obs:.3f} m/s^3 (includes switches)")
    print("-" * 40)
    print(f"Energy Proxy (Exec):  {total_energy_mj:.2f} MJ")
    if gt_energy_mj is not None:
        dE_pct = 100 * (total_energy_mj - gt_energy_mj) / max(gt_energy_mj, 1e-6)
        print(f"GT Energy (Dataset):  {gt_energy_mj:.2f} MJ ({dE_pct:+.1f}%)")
    print("="*40 + "\n")

    if json_path:
        out = {
            't': sim.history['t'],
            'x': sim.history['x'],
            'v': sim.history['v'],
            'a': sim.history['a'],
            'j_app_max': max_j_app,
            'j_steady_max': max_j_steady,
            'delay_breakdown': {'brake': stop_brake, 'hold': stop_hold, 'travel': post_stop},
            'energy_proxy': total_energy_mj,
            'delay': total_delay
        }
        with open(json_path, 'w') as f:
            json.dump(out, f)
            print(f"Exported scientific trace to {json_path}")

if __name__ == "__main__":
    main()
