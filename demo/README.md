# Neural Surrogate Model - Visual Demo

This module demonstrates the real-time replanning capabilities of the Neural Surrogate Model.

## Overview
The demo:
1.  Loads a random original trajectory from the H5 dataset.
2.  Generates an initial speed profile using the trained model (`demo.planning`).
3.  Simulates a train tracking this profile in the time domain (`demo.simulation`).
4.  Injects a disturbance (e.g., Speed Limit, Stop).
5.  Triggers a **replan** using the neural model at the disturbance time.
6.  Visualizes the result and replan latency.

## Prerequisites
- Python 3.12 and the project installed (`pip install -r requirements.txt` then `pip install -e .`)
- An HDF5 dataset (not included, see the main README) and a trained run folder
- FFmpeg on PATH, only if you pass `--save_mp4`

## Usage

Run the demo from the project root:

```bash
python -m demo.app \
    --scenario speed_limit \
    --save_mp4 demo.mp4 \
    --run_dir runs/run_20260216_112435
```

### Arguments
- `--scenario`: Type of disturbance.
    - `stop`: Force a stop (v_limit=0) for 10s.
    - `speed_limit`: Impose 10 m/s limit for 30s.
    - `traction_loss`: Reduce max acceleration by 50%.
    - `none`: No disturbance.
- `--seed`: Random seed for trajectory selection.
- `--run_dir`: Trained run folder (required, or give `--model_path`).
- `--save_mp4`: Optional MP4 file name, written inside `--output_dir` (default `demo_outputs/`, git-ignored). Needs FFmpeg on PATH; skipped with a message otherwise.
- `--export_json`: Optional JSON trace file name, also written inside `--output_dir`.

## Output
- **MP4 Video**: showing speed (GT, Plan, Exec, Replan) and track features.
- **JSON Trace**: time-series data of the simulation.
