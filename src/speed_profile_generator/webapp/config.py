import os

# The dataset is not distributed with this repository (see docs/dataset_and_nda.md).
# Point these at your own HDF5 file and training runs directory.
DEFAULT_H5_PATH = os.environ.get("SPEED_PROFILE_H5_PATH", "data/dataset.h5")
RUNS_DIR = os.environ.get("SPEED_PROFILE_RUNS_DIR", "runs")
OUTPUTS_DIR = os.environ.get("SPEED_PROFILE_OUTPUTS_DIR", "demo_outputs")
HOST = os.environ.get("SPEED_PROFILE_HOST", "127.0.0.1")
PORT = int(os.environ.get("SPEED_PROFILE_PORT", "8000"))
