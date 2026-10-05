import os
import threading
import uuid
import time
import numpy as np
import traceback
from typing import Dict, Any, Optional, List

from speed_profile_generator.webapp.models import (
    JobState, JobStatusResponse, SimulationRequest, SimulationResult,
    SeriesData, EventData, SimulatorMeta
)
from speed_profile_generator.webapp import config as webapp_config
import demo.app as demo_app

# --- Job Manager ---
class JobManager:
    def __init__(self):
        self.jobs: Dict[str, JobStatusResponse] = {}
        self.lock = threading.Lock()

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        with self.lock:
            self.jobs[job_id] = JobStatusResponse(
                job_id=job_id,
                state=JobState.PENDING
            )
        return job_id

    def update_job(self, job_id: str, state: JobState, result: Optional[SimulationResult] = None, error: Optional[str] = None):
        with self.lock:
            if job_id in self.jobs:
                job = self.jobs[job_id]
                job.state = state
                if result: job.result = result
                if error: job.error = error

    def get_job(self, job_id: str) -> Optional[JobStatusResponse]:
        with self.lock:
            return self.jobs.get(job_id)

job_manager = JobManager()

# --- Simulation Worker ---
def run_simulation_worker(job_id: str, request: SimulationRequest):
    try:
        # Strict validation
        if not request.run_id or not request.traj_id:
            missing = []
            if not request.run_id: missing.append("run_id")
            if not request.traj_id: missing.append("traj_id")
            raise ValueError(f"Missing required parameters: {', '.join(missing)}")

        job_manager.update_job(job_id, JobState.RUNNING)
        
        # 1. Prepare Config
        config = request.params.copy()
        config['scenario'] = request.scenario
        config['stop_respect_limits'] = request.respect_limits
        config['dur_policy'] = request.duration_policy
        config['h5_path'] = request.h5_path
        config['run_dir'] = os.path.join(webapp_config.RUNS_DIR, request.run_id)
        config['traj_id'] = request.traj_id # Ensure it's passed
            
        # 2. Run Simulation
        raw_results = demo_app.run_simulation_service(config)
        
        # 3. Validate
        validate_results(raw_results)
        
        # 4. Process Results (Strict Schema V9)
        processed_results = package_v9_results(raw_results, request)
        
        # 5. Save/Update Job
        sim_result = SimulationResult(**processed_results)
        job_manager.update_job(job_id, JobState.DONE, result=sim_result)
        
    except Exception as e:
        traceback.print_exc()
        job_manager.update_job(job_id, JobState.ERROR, error=str(e))

def validate_results(raw: Dict[str, Any]):
    history = raw['history']
    t_s = np.array(history['t'])
    if len(t_s) < 2:
        raise ValueError("Simulation history too short.")
    if np.max(t_s) > 3600:
         raise ValueError(f"Simulation time extreme: {np.max(t_s):.1f}s. Check unit conversion.")

def package_v9_results(raw: Dict[str, Any], request: SimulationRequest) -> Dict[str, Any]:
    history = raw['history']
    replan = raw['replan']
    track = raw['track']
    limits = raw['sim_limits']
    
    # 1. Executed path (Time Domain)
    exec_t = np.array(history['t'])
    exec_x = np.array(history['x'])
    exec_v = np.array(history['v'])
    exec_a = np.array(history['a'])
    
    # 2. Global Ground Truth
    gt_x = np.array(raw.get('traj_x', []))
    gt_v = np.array(raw.get('traj_v', []))
    
    # 3. AI Plan (Space Domain)
    plan_x = gt_x.copy()
    plan_v = gt_v.copy()
    feasible_plan_x = None
    feasible_plan_v = None

    if replan.get('active'):
        plan_x = np.array(replan['x'])
        plan_v = np.array(replan['v'])
        feasible_plan_x = np.array(replan['x'])
        feasible_plan_v = np.array(replan.get('feasible_v', []))

    # 4. Reference Calculation (V9 definition: v_plan(x_exec(t)))
    ref_v_time = None
    try:
        # Determine source plan for reference
        # Selection logic: if 'raw' or 'both' requested, reference usually means the raw suggestion
        # But if 'feasible' only, use feasible. Default to feasible for safety if in doubt.
        src_v = plan_v
        src_x = plan_x
        if request.plan_visibility == "feasible" and feasible_plan_v is not None:
            src_v = feasible_plan_v
            src_x = feasible_plan_x
            
        if len(src_x) > 0:
            # Stability: Sort and deduplicate
            sort_idx = np.argsort(src_x)
            ux, uidx = np.unique(src_x[sort_idx], return_index=True)
            uv = src_v[sort_idx][uidx]
            
            # Interpolate
            ref_v_time = np.interp(exec_x, ux, uv)
            
            # Clamp to [0, v_limit_max + 2.0]
            v_limit_max = np.max(track.get('limit', [60.0]))
            ref_v_time = np.clip(ref_v_time, 0, v_limit_max + 2.0)
            
            # NaN check
            if np.any(np.isnan(ref_v_time)):
                ref_v_time = None
    except Exception as e:
        print(f"[V9 Service] Reference calculation failed: {e}")
        ref_v_time = None

    # GT onto Time domain
    gt_v_time = None
    if len(gt_x) > 0:
        gt_v_time = np.interp(exec_x, gt_x, gt_v)

    # 5. Downsample
    N = len(exec_t)
    target = request.downsample_points
    step = max(1, int(np.ceil(N / target)))
    idx = slice(0, N, step)
    
    def ds(arr): 
        if arr is None or len(arr) == 0: return None
        return (np.array(arr)[idx]).tolist()

    series = {
        'exec_t_s': ds(exec_t),
        'exec_x_m': ds(exec_x),
        'exec_v_mps': ds(exec_v),
        'exec_a_mps2': ds(exec_a),
        'ref_v_mps_time': ds(ref_v_time),
        'plan_x_m': plan_x.tolist(),
        'plan_v_mps': plan_v.tolist(),
        'feasible_plan_x_m': feasible_plan_x.tolist() if feasible_plan_x is not None else None,
        'feasible_plan_v_mps': feasible_plan_v.tolist() if feasible_plan_v is not None else None,
        'gt_x_m': gt_x.tolist(),
        'gt_v_mps_pos': gt_v.tolist(),
        'gt_v_mps_time': ds(gt_v_time),
        'x_limit_m': track.get('x', []),
        'v_limit_mps': track.get('limit', [])
    }
    
    meta = {
        'dt_s': 0.2,
        't0_s': float(exec_t[0]),
        'tf_s': float(exec_t[-1]),
        'x0_m': float(exec_x[0]),
        'xf_m': float(exec_x[-1]),
        'a_min': float(limits['a_min']),
        'a_max': float(limits['a_max']),
        'run_id': request.run_id,
        'traj_id': request.traj_id,
        'scenario': request.scenario,
        
        # Narrative data
        'start_station_id': "A",
        'end_station_id': "B",
        't_disturb': raw['events'].get('stop_brake_start'),
        't_stop_reached': raw['events'].get('stop_hold_start'),
        't_release': raw['events'].get('stop_hold_end'),
        't_replan': raw['events'].get('replan_start'),
        't_terminal_on': raw['events'].get('terminal_start'),
        't_terminal_off': None,
        't_end': float(exec_t[-1])
    }
    
    event_list = []
    colors = {'stop_brake_start': 'red', 'stop_hold_start': 'orange', 'stop_hold_end': 'green', 'replan_start': 'blue', 'terminal_start': 'purple'}
    labels = {'stop_brake_start': 'Braking Disturbance', 'stop_hold_start': 'Stopped', 'stop_hold_end': 'Resuming', 'replan_start': 'Replanning Triggered', 'terminal_start': 'Terminal Braking'}
    
    for key, t_val in raw['events'].items():
        if t_val is not None:
            closest_idx = np.argmin(np.abs(exec_t - t_val))
            x_val = float(exec_x[closest_idx])
            event_list.append({'type': key, 't_s': t_val, 'x_m': x_val, 'label': labels.get(key, key), 'color': colors.get(key, 'gray')})
    
    event_list.sort(key=lambda x: x['t_s'])
    
    return {
        'meta': meta,
        'series': series,
        'events': event_list,
        'metrics': raw['metrics'],
        'logs': raw['logs']
    }

def start_simulation_job(request: SimulationRequest) -> str:
    job_id = job_manager.create_job()
    thread = threading.Thread(target=run_simulation_worker, args=(job_id, request))
    thread.daemon = True
    thread.start()
    return job_id
