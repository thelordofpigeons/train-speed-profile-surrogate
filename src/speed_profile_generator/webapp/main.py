from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import glob
import h5py
import numpy as np
from typing import List

from speed_profile_generator.webapp.models import (
    SimulationRequest, SimulationResult, JobStatusResponse, 
    JobState, RunInfo, TrajectoryInfo
)
import speed_profile_generator.webapp.services as services
from speed_profile_generator.webapp import config

app = FastAPI(title="Speed Profile Generator Viz")

# CORS for local dev (the Vite dev server runs on a different port)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _check_h5_path(h5_path: str) -> str:
    # Only files under the folder of the configured dataset may be opened, so a
    # caller cannot use the query parameter to probe arbitrary paths.
    root = os.path.realpath(os.path.dirname(config.DEFAULT_H5_PATH) or ".")
    real = os.path.realpath(h5_path)
    try:
        inside = os.path.commonpath([root, real]) == root
    except ValueError:  # different drives on Windows
        inside = False
    if not inside:
        raise HTTPException(status_code=400, detail="h5_path must be inside the configured data folder")
    if not os.path.exists(real):
        raise HTTPException(status_code=404, detail="Dataset H5 not found")
    return real

# --- Routes ---

@app.get("/runs", response_model=List[RunInfo])
def list_runs():
    # Each subdirectory of the runs folder is one training run
    base = config.RUNS_DIR
    if not os.path.exists(base):
        return []

    runs = []
    for d in os.listdir(base):
        path = os.path.join(base, d)
        if os.path.isdir(path):
            runs.append(RunInfo(id=d, path=path))
    # Sort by name (run folders are timestamped) descending
    runs.sort(key=lambda x: x.id, reverse=True)
    return runs

@app.get("/runs/{run_id}/trajectories", response_model=List[TrajectoryInfo])
def list_trajectories(run_id: str, h5_path: str = config.DEFAULT_H5_PATH):
    # The dataset is not shipped with the repo; the path comes from the query
    # parameter or SPEED_PROFILE_H5_PATH.
    h5_path = _check_h5_path(h5_path)

    try:
        trajs = []
        with h5py.File(h5_path, 'r') as f:
            keys = [k for k in f.keys() if k.startswith('traj_') and "_seg_" not in k]
            for k in keys:
                g = f[k]
                t_arr = g['t']
                dur = t_arr[-1] - t_arr[0]
                length = g['x'][-1] - g['x'][0]
                trajs.append(TrajectoryInfo(id=k, duration=float(dur), length=float(length)))

        return trajs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/runs/{run_id}/artifacts")
def list_artifacts(run_id: str):
    # Simulation artifacts (mp4, pdf, png) are written by demo.app to its output
    # directory. All jobs share that directory, which is fine for a local tool.
    base = config.OUTPUTS_DIR
    if not os.path.exists(base): return []
    files = glob.glob(os.path.join(base, "*"))
    return [os.path.basename(f) for f in files]

@app.post("/simulate", response_model=JobStatusResponse)
def run_simulation(req: SimulationRequest):
    req.h5_path = _check_h5_path(req.h5_path)

    job_id = services.start_simulation_job(req)
    return JobStatusResponse(job_id=job_id, state=JobState.PENDING)

@app.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job(job_id: str):
    job = services.job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

# Static Files (Frontend)
# In production, frontend build goes to `webapp/static`
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
