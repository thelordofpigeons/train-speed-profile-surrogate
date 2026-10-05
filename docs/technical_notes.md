# Technical notes

These notes consolidate the method behind the code in this repository. The
problem statement, formulas and aggregate results come from the author's
report (see `CITATION.cff`); the implementation details were read from the
source and are marked as such where they differ from the report. No figure
from the report is reproduced, because those figures are rendered from the
confidential dataset (see `docs/dataset_and_nda.md`).

## 1. Problem

A train running between two stops should reach the second stop on time while
using as little electrical energy as possible. The driving strategy (when to
accelerate, cruise, coast and brake) is the decision. Optimal control
solvers (dynamic programming, pseudospectral methods, Pontryagin-based
numerical optimisation) give accurate answers, but a single trajectory can
take seconds to several minutes. That is too slow for Automatic Train
Operation (ATO), Driver Advisory Systems (DAS), or any setting where the
profile must be recomputed after a disturbance, a timetable change or a
traffic event.

The approach here is to solve the optimal control problem offline for many
operating conditions, then train a neural network to reproduce the mapping

    v(x) = F(x, P, T)

where `x` is the position along the segment, `P` collects boundary
conditions and train parameters, and `T` is the local track description
(slope, speed limit, curvature). Inference is a single forward pass.

## 2. Reference optimal control problem

Objective: minimise electrical energy over the trip,

    J = integral over [0, tf] of P_elec(F(t), v(t)) dt

Spatial dynamics, with `x` in `[x0, xf]`, mass `M`, traction force `F_trac`
and resistance `F_res`:

    M * v(x) * dv/dx = F_trac(x) - F_res(x)

Resistance follows the Davis form `F_res(v) = A + B*v + C*v^2`, with `A`
for rolling resistance, `B` for mechanical friction and `C` for aerodynamic
drag. Slope enters as an additional gravitational term in the time-domain
formulation.

Constraints: `v(x0) = v0`, `v(xf) = vf`, `v(x) <= V_target(x)` (speed limit),
and acceleration and braking limits.

Optimal solutions have a bang-singular-bang structure: acceleration at
maximum traction power, cruise, coasting with traction cut, then braking.
The network does not receive this structure explicitly; it learns it from
the data.

## 3. Network

The model is a coordinate-based network: it predicts the speed at one
normalised position, given the operating conditions, rather than emitting a
whole sequence.

Source: `src/speed_profile_generator/models/` and `utils/bc.py`.

**Input.** The normalised position `xi = (x - x0) / (xf - x0)`, clamped to
`[0, 1]`, so segments of different length share one input domain. This is
`compute_xi` in `utils/bc.py`.

**Positional encoding.** `SinusoidalPositionalEncoding` maps `xi` to
`[sin(w_0 xi), cos(w_0 xi), ..., sin(w_{N-1} xi), cos(w_{N-1} xi)]` with
geometrically spaced frequencies `w_i = base^(i / (N - 1))`. The V4
configuration uses `N = 16` frequencies and `base = 10`, giving 32 encoded
features.

**Track features.** Appended to the encoding. The track is sampled at seven
relative lookahead offsets `{-0.2, -0.1, 0.0, 0.1, 0.2, 0.4, 0.7}` (fractions
of the segment length) for three attributes (slope, speed limit, curvature),
which gives 21 values. Five event features are added: minimum speed limit
ahead, maximum curvature ahead, distance to the next speed-limit drop, a
drop indicator flag and the drop magnitude. Total: 26 (`track_feat_dim: 26`).
The lookahead is relative to the segment, so it is scale-invariant.

**Conditioning vector.** Six continuous values: segment length `L`, start
speed `v0`, end speed `vf`, target duration, train mass `M`, and a
segment speed cap `v_seg_max`. In the implementation this vector is
extended with two learned station embeddings (start and end station index)
and two extra scalars (log length and a flag marking sub-segments), so the
tensor handed to the blocks is `6 + 2*emb_dim + 2` wide. Station
embeddings are an implementation detail that the report does not describe.

**Modulated residual blocks.** The trunk is an input projection, then `K`
blocks, then a linear output layer. The V4 configuration uses `K = 8` blocks
and a hidden width of 256 (`configs/train_h5_config_v4_full.yaml`). Each
block, as implemented in `models/modulated_blocks.py`, computes

    h_out = h_in + A(h_in) * B(P)

where `A` is a linear layer followed by SiLU and `B` is a linear projection
of the conditioning vector, broadcast over the positions of a trajectory.
The report presents this as feature-wise linear modulation (FiLM). To be
precise about the code: the modulation is multiplicative only. There is no
separate additive shift term, so it behaves as a conditioning-dependent gate
on the residual branch rather than a full scale-and-shift FiLM layer.

**Hard boundary conditions.** The raw network output `y(xi)` is not used
directly. The prediction is

    v(xi) = (1 - xi) * v0 + xi * vf + xi * (1 - xi) * y(xi)

so `v(0) = v0` and `v(1) = vf` for any `y`, by construction (`apply_hard_bc`
in `utils/bc.py`; `tests/test_hard_bc_invariant.py` checks the invariant).
Boundary conditions are therefore exact and cost nothing in the loss. Speed
limits are not enforced this way; they are soft and handled by a penalty
(section 4).

## 4. Losses

Two families appear in the report. Both are implemented; the second is the
V4 training objective.

### 4.1 Hybrid supervised loss (V1 to V3 era)

    L = L_MSE + l1 * L_acc + l2 * L_curv + l3 * L_peak

- `L_MSE = mean((v_pred - v_true)^2)`.
- `L_acc` penalises the difference between predicted and reference
  acceleration `a = dv/dt`.
- `L_curv = sum((v_{i+2} - 2 v_{i+1} + v_i)^2)`, a second-difference
  smoothness penalty that damps oscillations a high-frequency encoding can
  otherwise produce.
- `L_peak = (max(v_pred) - max(v_true))^2`, which anchors the global scale of
  the profile. Peak speed strongly influences both energy and travel time.

### 4.2 Physics-informed loss (V4)

Implemented in `training/losses_v3.py` (`SpeedProfileLossV3`, whose V4
methods carry a `_v4` suffix). Derivatives are taken along the spatial grid
on possibly non-uniform spacing, with a minimum and maximum step guard
(`DX_MIN_M`, `DX_MAX_M`).

- Reconstruction: mean squared error on speed, optionally weighted towards
  the cruise regime (`use_cruise_weighting`).
- Curvature smoothness: mean of `(d2v/dx2)^2`, with the second derivative
  clamped.
- Speed limit: `max(0, v_pred - v_limit)^2`. The code adds an optional
  extra weight near the limit.
- Non-negativity: penalises `v < 0`.
- Energy proxy: with `a = v * dv/dx`, the term `a^2 * v`, which punishes
  strong acceleration at high speed.
- Power actuation: traction force `F_trac = M*a + F_res` with the Davis
  resistance, traction power `P = F_trac * v`, and a `log1p` penalty on the
  positive part of the power in megawatts. The `power_*` constants in the
  shipped configuration are example training settings, not measured train
  parameters.
- Optimality residual (code only, not in the report): a simplified penalty on
  `P / v`, motivated by the Hamiltonian condition from Pontryagin's
  principle. Its weight in the V4 configuration is small (`lambda_pmp: 0.05`).

Physics terms are introduced gradually. For the first `warmup_epochs` their
weight multiplier is 0, then it rises linearly to 1 over `ramp_epochs`
(`training/train_h5.py`).

## 5. Training modes

Training on whole trajectories is memory-hungry, and derivative-based
losses need ordered positions. The V4 pipeline therefore alternates:

- **Point mode.** Random positions are sampled from each trajectory. Each
  sample is a tuple of conditioning vector, `xi`, track features and the
  reference speed. This breaks temporal correlation between samples and is
  cheap per step, so it carries most of the optimisation.
- **Trajectory mode.** At intervals the loop processes complete ordered
  sequences `x_1 ... x_T`. This is where the curvature, acceleration, energy,
  power and optimality terms are computed.

In the V4 configuration the trajectory batches are interleaved with the
point batches each epoch (`traj_batches_per_epoch`, `traj_batch_size`).
Splits are made at trajectory level, and all sub-segments of a trajectory
stay on the same side of the split, to avoid leakage.

Optimiser, as read from `training/train_h5.py`: AdamW with the learning
rate and weight decay from the configuration. The scheduler in the training
loop is a `StepLR` with `gamma=1.0`, so the learning rate is effectively
constant. Early stopping is controlled by `early_stopping_patience`.

## 6. Metrics

Defined in the report and computed by `validation/validate_h5.py` and
`validation/metrics.py`:

- **RMSE**: `sqrt(mean((v_pred - v_ref)^2))` over a trajectory.
- **Peak velocity error**: `| max(v_pred) - max(v_ref) |`.
- **Smoothness ratio**: `S = sum((d2 v_pred)^2) / sum((d2 v_ref)^2)`. Values
  near 1 mean the prediction is about as smooth as the optimal solution.
  `validate_h5.py` computes a close variant, the ratio of the mean absolute
  second differences of prediction and reference.

Aggregate results reported by the author on the held-out set (confidential
data, not reproducible from this repository):

| Quantity | Value |
| --- | --- |
| RMSE, mean over validation trajectories | 0.20 m/s |
| Peak velocity error | 0.26 m/s |
| Boundary conditions | exact by construction |
| Inference time | under 10 ms on a standard CPU, under 5 ms on a modern GPU |

The report's baseline (V1) was a plain network regressing the speed
profile without physical constraints or track conditioning. It showed
unstable cruise plateaus, wrong peak speeds, boundary violations and large
errors on long trajectories, with RMSE up to 33 m/s and worst cases above
30 m/s. V4 changed four things at once: hard boundary conditions,
physics-informed loss terms, track-feature conditioning and a deeper
residual trunk. The report does not isolate their individual contributions,
and neither does this repository.

The report also checked predictions against dynamic-programming (Bellman)
solutions on sub-segments of the validation tracks, and evaluated on track
segments absent from training. Those experiments need the dataset.

## 7. Inference

Inference is a forward pass over a grid of positions, so latency does not
depend on how hard the optimal control problem was. The sub-10 ms CPU and
sub-5 ms GPU figures above are from the report. A benchmark script
(`scripts/benchmark_inference_cpu.py`) times the model on random
inputs; its numbers describe your machine, not the report's.

## 8. Limitations

- The network approximates the output of a solver. It carries no optimality
  or feasibility guarantee beyond the boundary conditions.
- Speed limits, acceleration limits and non-negativity are soft: they are
  penalised, not guaranteed. A deployed system would need a check or
  projection step after inference. The demo includes a projection option
  (`--project_feasible`) as a starting point.
- Generalisation is bounded by the diversity of the training trajectories.
  Segments, trains or conditions far outside the training distribution are
  not covered by any result here.
- The model is a decision-support component. It is not a safety-critical
  controller, and final authority stays with certified train control.
- The synthetic path in this repository (README quick start) uses a
  reduced conditioning vector and no track features, so it demonstrates the
  mechanics of training and the hard boundary conditions, not the reported
  accuracy.

## 9. Future validation

Listed in the report as future work, not done here: hardware-in-the-loop
tests, field trials, and transparency and traceability requirements under
the European AI Act. Environmental benefit figures in the report are
qualitative and are not claimed by this repository.
