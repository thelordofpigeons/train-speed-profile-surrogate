# Dataset and confidentiality

## The dataset is not included

The model in this repository was trained and evaluated on a dataset of
energy-optimal trajectories computed offline with a high-fidelity optimal
control solver for real railway infrastructure. That dataset was used in a
project carried out with the LAMIH laboratory (Universite Polytechnique
Hauts-de-France). It is covered by a non-disclosure agreement, is
confidential and not redistributable, and the author is not free to
publish it, in whole or in part.

For the same reason the repository deliberately contains none of the
following:

- The dataset itself, in any form (`.h5`, `.mat`, `.csv`, `.npy` or
  otherwise), or any excerpt or sample of it.
- Trained checkpoints (`.pt`, `.pth`). Weights fitted to the data can leak
  properties of it.
- Train and validation split files, which list real trajectory identifiers.
- Normalisation statistics and station index maps computed from the data.
- Plots, animations or videos rendered from real trajectories or real
  track profiles, including the figures of the report.
- Any detail that would identify an industrial partner, a line or a train.

The code that reads such a dataset is included, so the method is fully
inspectable, but nothing here lets you retrain or evaluate on the original
data. Run-time products of the real-data pipeline (`runs/`, `artifacts/`,
the `splits/` directory, `*.h5`) are listed in `.gitignore` so they cannot be
committed by accident. Keep it that way if you work with your own data.

## What you can run without it

- The unit tests (`python -m pytest -q`). They use synthetic tensors or a
  small mock HDF5 file built in a temporary directory.
- The synthetic training path described in the README quick start. It uses
  the kinematic generator in `utils/data_gen.py` and the V4 network with a
  reduced conditioning vector. It shows the training mechanics and the
  exact boundary conditions, and it does not reproduce the reported metrics.
- The model, positional encoding, boundary-condition wrapper and loss code,
  which can be imported and exercised on random inputs.

## What needs your own data

Training with `training/train_h5.py`, validation with
`validation/validate_h5.py`, the CLI demo (`demo/app.py`) and the web
visualiser all read an HDF5 file in the layout below, and the last two also
need a trained run directory (`runs/<run>/`).

### Expected HDF5 layout

This layout was read from `data/h5_dataset.py` and `scripts/mat_to_h5.py`.

```
dataset.h5
  railTrack/
    slope        float array, shape (N, 2): column 0 position, column 1 value
    spdLimit     float array, shape (N, 2): position, speed limit
    curvature    float array, shape (N, 2): position, curvature
  index          string array: names of the trajectory groups below
  traj_<i>/      one group per optimal trajectory (<i> zero-padded to 5 digits)
    t            time stamps, shape (T,)
    x            position along the track, shape (T,), non-decreasing
    v            reference speed, shape (T,)
    xi           normalised position (x - x0) / (xf - x0), shape (T,)
    P            parameter vector, shape (8,)
    accel        optional reference acceleration, shape (T,)
  traj_<i>_seg_<k>/
    same fields, for the k-th sub-segment cut from trajectory i
```

The parameter vector `P` is, in order:

| Index | Content |
| --- | --- |
| 0 | start position `x0` |
| 1 | end position `xf` |
| 2 | start speed `v0` |
| 3 | end speed `vf` |
| 4 | start station id (integer-valued) |
| 5 | end station id (integer-valued) |
| 6 | trip duration |
| 7 | train mass `M` |

From `P` the loader derives the six-value conditioning vector
`[L, v0, vf, duration, M, v_seg_max]`, where `L = xf - x0` and
`v_seg_max` is the largest speed limit along the segment (or a fallback
when no track data is present). Station ids are mapped to dense embedding
indices, with index 0 reserved for unknown stations.

Track features (26 values per sampled position) are interpolated from the
`railTrack` curves at relative lookahead offsets, as described in
`docs/technical_notes.md`.

### Bringing your own data

1. Produce optimal speed profiles with a solver of your choice for your own
   track and trains. Any source works as long as it fills the fields above.
2. If your solver exports MATLAB v7.3 files with a `sol` cell array and a
   `railTrack` structure, `scripts/mat_to_h5.py` converts them:

   ```
   python scripts/mat_to_h5.py --input path/to/export.mat --output data/dataset.h5
   ```

   Otherwise write the HDF5 file directly with `h5py` in the layout above.
3. Point `data.h5_path` in `configs/train_h5_config_v4_full.yaml` at the file.
   The configuration's `device` is `cuda`; change it to `cpu` if you have no
   GPU, and expect training to be slow at the configured model size.
4. Train: `python -m speed_profile_generator.training.train_h5 --config configs/train_h5_config_v4_full.yaml`.
   On first use the loader creates a trajectory-level train and validation
   split under `splits_dir`, and station index maps under `artifacts/`.
   Each run writes to `runs/run_<timestamp>/` (configuration, normalisation
   statistics, track statistics, `best_model.pt`, logs).
5. Validate a run: `python -m speed_profile_generator.validation.validate_h5 --run_dir runs/run_<timestamp>`.

The loader hard-codes a few choices that you may need to adapt to a
different network: `num_stations=100` for the station embedding table,
two extra conditioning scalars (log length and a sub-segment flag), and the
seven lookahead offsets.

## Provenance of the numbers in the README

The aggregate figures in the README (mean RMSE 0.20 m/s, peak velocity error
0.26 m/s, inference under 10 ms on CPU) are quoted from the author's
project report. They describe the confidential dataset and cannot be
checked from this repository. The benchmark script measures latency on
random inputs on your own hardware, which is a separate and reproducible
statement.
