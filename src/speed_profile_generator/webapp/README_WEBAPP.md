# Speed Profile Generator: Visualizer Web App

A local web application that replays closed-loop simulations of the trained
speed profile generator: a FastAPI backend (`main.py`, `services.py`) that runs
the simulator from `demo/`, and a React 18 + Vite + Plotly frontend (`frontend/`).

## Requirements: dataset and trained run (not included)

**This app does not show anything useful from the repository alone.** It needs
two inputs that are not shipped:

1. **An HDF5 dataset** of optimal trajectories. The dataset used for the
   original work is under NDA and is not distributed (see
   `docs/dataset_and_nda.md` for why and for the expected layout). The backend
   lists the `traj_*` groups of this file in the trajectory selector and the
   simulator replays them.
2. **A trained run directory** under `runs/<run_id>/` (model checkpoint and
   config produced by `training/train_h5.py`). Checkpoints trained on the NDA
   data are not distributed either.

Without them the UI loads, `GET /runs` returns an empty list, and
`GET /runs/{id}/trajectories` and `POST /simulate` answer `404 Dataset H5 not found`.
To get a populated UI you have to provide your own HDF5 file in the expected
layout and train a run on it. The synthetic generator described in the main
README trains the same `SpeedProfileGenerator` architecture, but it produces no
HDF5 file and no track statistics, so its output cannot feed this app.

Where the app looks (all overridable by environment variable):

| Variable                    | Default           | Meaning                                  |
|-----------------------------|-------------------|------------------------------------------|
| `SPEED_PROFILE_H5_PATH`     | `data/dataset.h5` | HDF5 dataset                             |
| `SPEED_PROFILE_RUNS_DIR`    | `runs`            | folder with one subfolder per run        |
| `SPEED_PROFILE_OUTPUTS_DIR` | `demo_outputs`    | where `demo.app` writes mp4/pdf/png      |
| `SPEED_PROFILE_HOST`        | `127.0.0.1`       | bind address                             |
| `SPEED_PROFILE_PORT`        | `8000`            | port                                     |

`GET /runs/{id}/trajectories` also accepts an `h5_path` query parameter.

## What it shows

- Run selector (any folder under `runs/`) and trajectory selector (any `traj_*`
  entry of the HDF5 file).
- Disturbance scenarios in the UI: nominal (none), temporary stop, manual
  replan. The command line demo (`python -m demo.app`) additionally offers
  speed limit change and traction loss; see `demo/README.md`.
- Engineering dashboard with three synchronized Plotly views: speed vs
  position (planned, feasible plan, executed, ground truth, speed limit),
  speed vs time, and acceleration vs time with the acceleration bounds.
- Visual playback with a track view and instruments, driven by one global
  playback cursor (play, pause, seek, speed) shared with the dashboard.
- Event markers (disturbance, replan trigger, terminal braking) and the
  simulator metrics (delay, energy, jerk) returned with each result.

## API routes

| Route                          | Purpose                                              |
|--------------------------------|------------------------------------------------------|
| `GET /runs`                    | list run folders                                     |
| `GET /runs/{id}/trajectories`  | list trajectories of the HDF5 dataset                |
| `GET /runs/{id}/artifacts`     | list files in the outputs folder                     |
| `POST /simulate`               | start a simulation, returns a `job_id`               |
| `GET /jobs/{job_id}`           | poll job state (`pending`, `running`, `done`, `error`) and fetch the result |

The built frontend, if present in `static/`, is served at `/`.

## Prerequisites

- Python 3.12 with the project installed (`pip install -r requirements.txt`,
  `pip install -r requirements-webapp.txt`)
- Node.js 18+ for the frontend (developed with Node 24)

## Development mode

Run backend and frontend separately to get hot reloading.

Terminal 1, from the repository root:

```bash
# bash
PYTHONPATH=src python -m speed_profile_generator.webapp.main
```

```powershell
# PowerShell
$env:PYTHONPATH="src"; python -m speed_profile_generator.webapp.main
```

The API is on `http://localhost:8000`. The backend imports the `demo` package,
so start it from the repository root.

Terminal 2:

```bash
cd src/speed_profile_generator/webapp/frontend
npm install
npm run dev
```

The UI is on `http://localhost:5173`; the Vite dev server proxies `/runs`,
`/simulate` and `/jobs` to port 8000.

## Production build (single server)

The frontend is built from source; the bundle is build output and is not
committed.

```bash
scripts/build_frontend.sh        # or scripts/build_frontend.ps1 on Windows
```

This runs `npm ci && npm run build` and writes the bundle to
`src/speed_profile_generator/webapp/static/`. Then start the backend as above
and open `http://localhost:8000`. On Windows, `run_webapp.ps1` does the install,
the build (if the bundle is missing) and the start in one step.

## Notes

- Simulation jobs run in background threads inside the server process and are
  kept in memory; they are lost on restart.
- All jobs write their mp4/pdf/png artifacts to the same outputs folder, which
  is fine for a local single-user tool.
- The backend binds to localhost by default and enables permissive CORS for
  local development. Do not expose it on a network as is.
