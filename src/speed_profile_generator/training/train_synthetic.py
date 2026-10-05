"""
Synthetic training path (reduced conditioning).

Trains the current SpeedProfileGenerator on profiles from the kinematic
accel-coast-brake generator (utils/data_gen.py). It exists so that the model
can be trained and smoke-tested without the industrial dataset.

Differences from the real-data pipeline (train_h5.py):
  - No track features (slope / speed limit / curvature): track_feat_dim = 0.
  - No station ids: both ids are 0 (the UNK embedding).
  - Only an MSE loss on v_ref (optional smoothness term), no physics losses.
  - The conditioning vector is the same 6-dim schema as the real pipeline,
    [L, v0, vf, dur, M, v_seg_max], where dur is estimated from the profile
    and v_seg_max is the generator's speed cap.
The hard boundary conditions are enforced by the model exactly as in V4.
"""
import json
import os
import time
from typing import Dict, Optional

import torch
import yaml

from speed_profile_generator.data.dataset import SpeedProfileDataset
from speed_profile_generator.models.speed_profile_generator import SpeedProfileGenerator
from speed_profile_generator.training.losses import SpeedProfileLoss
from speed_profile_generator.utils.preprocessor import PPreprocessor

DEFAULT_CFG: Dict = {
    "seed": 42,
    "data": {"n_train": 2000, "n_val": 200, "num_points": 100, "dist_range": [500.0, 10000.0]},
    "model": {"hidden_dim": 64, "num_blocks": 4, "num_freqs": 16, "freq_base": 10.0, "num_stations": 100, "emb_dim": "auto"},
    "training": {"steps": 500, "batch_size": 64, "lr": 1.0e-3, "weight_decay": 1.0e-4,
                 "grad_clip_norm": 1.0, "lambda_smooth": 0.0, "log_every": 50},
}


def load_config(path: Optional[str] = None) -> Dict:
    """Merge an optional YAML file over DEFAULT_CFG (one level deep)."""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_CFG.items()}
    if path:
        with open(path, "r") as f:
            user = yaml.safe_load(f) or {}
        for k, v in user.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def make_synthetic_tensors(n: int, num_points: int, dist_range, seed: int) -> Dict[str, torch.Tensor]:
    """
    Generate n profiles and map them onto the V4 conditioning layout.

    Returns:
        x:      [N, T, 1] physical positions (m)
        v_ref:  [N, T, 1] reference speed (m/s)
        P_raw:  [N, 8]    [x0, xf, v0, vf, st_start, st_end, dur, M]
        P_cont: [N, 6]    [L, v0, vf, dur, M, v_seg_max]  (raw, not normalised)
    """
    ds = SpeedProfileDataset(num_samples=n, dist_range=tuple(dist_range), seed=seed, num_points=num_points)
    x = torch.stack([d["x"] for d in ds.data])          # [N, T, 1]
    v_ref = torch.stack([d["v_ref"] for d in ds.data])  # [N, T, 1]
    x0 = torch.cat([d["x0"] for d in ds.data]).unsqueeze(-1)  # [N, 1]
    xf = torch.cat([d["xf"] for d in ds.data]).unsqueeze(-1)
    v0 = torch.cat([d["v0"] for d in ds.data]).unsqueeze(-1)
    vf = torch.cat([d["vf"] for d in ds.data]).unsqueeze(-1)
    P = torch.stack([d["P"] for d in ds.data])          # [N, 4] = mass, v_max, a_max, b_max
    mass, v_max = P[:, 0:1], P[:, 1:2]

    # Travel time: constant acceleration inside each interval => dt = 2 dx / (v_i + v_i+1)
    dx = x[:, 1:, 0] - x[:, :-1, 0]
    v_sum = (v_ref[:, 1:, 0] + v_ref[:, :-1, 0]).clamp(min=1e-3)
    dur = (2.0 * dx / v_sum).sum(dim=1, keepdim=True)

    zeros = torch.zeros_like(x0)
    P_raw = torch.cat([x0, xf, v0, vf, zeros, zeros, dur, mass], dim=1)
    P_cont = torch.cat([xf - x0, v0, vf, dur, mass, v_max], dim=1)
    return {"x": x, "v_ref": v_ref, "P_raw": P_raw, "P_cont": P_cont}


def build_model(model_cfg: Dict, extra_features_dim: int = 2) -> SpeedProfileGenerator:
    return SpeedProfileGenerator(
        param_dim=6,
        hidden_dim=model_cfg["hidden_dim"],
        num_blocks=model_cfg["num_blocks"],
        track_feat_dim=0,
        num_stations=model_cfg.get("num_stations", 100),
        emb_dim=model_cfg.get("emb_dim", "auto"),
        num_freqs=model_cfg["num_freqs"],
        freq_base=model_cfg["freq_base"],
        extra_features_dim=extra_features_dim,  # log_length + is_segment, as in train_h5.py
    )


def model_inputs(batch: Dict[str, torch.Tensor], preprocessor: PPreprocessor):
    """Build (x, P_feats, P_raw, station_ids) like the point loop in train_h5.py."""
    P_raw = batch["P_raw"]
    length = torch.clamp(P_raw[:, 1] - P_raw[:, 0], min=1e-6)
    log_length = torch.log(length).unsqueeze(-1)
    is_seg = torch.zeros_like(log_length)
    P_feats = torch.cat([preprocessor.transform_cont(batch["P_cont"]), log_length, is_seg], dim=-1)
    st_ids = torch.zeros((P_raw.size(0), 2), dtype=torch.long)
    return batch["x"], P_feats, P_raw, st_ids


@torch.no_grad()
def evaluate(model, data, preprocessor, batch_size: int = 256) -> Dict[str, float]:
    model.eval()
    se, n, bc_err = 0.0, 0, 0.0
    for i in range(0, data["x"].size(0), batch_size):
        b = {k: v[i:i + batch_size] for k, v in data.items()}
        x, P_feats, P_raw, st = model_inputs(b, preprocessor)
        v = model.forward_x(x, P_feats, P_raw, station_ids=st)
        se += ((v - b["v_ref"]) ** 2).sum().item()
        n += v.numel()
        bc_err = max(bc_err,
                     (v[:, 0, 0] - P_raw[:, 2]).abs().max().item(),
                     (v[:, -1, 0] - P_raw[:, 3]).abs().max().item())
    return {"rmse": (se / n) ** 0.5, "bc_max_abs_err": bc_err}


def train_synthetic(cfg: Dict, out_dir: Optional[str] = None, verbose: bool = True) -> Dict:
    """Run the synthetic training loop on CPU. Returns the loss history and final metrics."""
    seed = cfg["seed"]
    d, m, t = cfg["data"], cfg["model"], cfg["training"]
    torch.manual_seed(seed)

    train = make_synthetic_tensors(d["n_train"], d["num_points"], d["dist_range"], seed)
    val = make_synthetic_tensors(d["n_val"], d["num_points"], d["dist_range"], seed + 1)

    preprocessor = PPreprocessor()
    preprocessor.fit_cont([row.numpy() for row in train["P_cont"]])

    model = build_model(m)
    n_params = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(t["lr"]), weight_decay=float(t["weight_decay"]))
    criterion = SpeedProfileLoss(lambda_smooth=float(t["lambda_smooth"]))

    if verbose:
        print(f"Synthetic training on CPU | params={n_params:,} | steps={t['steps']} | batch={t['batch_size']}")

    history = []
    n_train = train["x"].size(0)
    start = time.time()
    for step in range(1, int(t["steps"]) + 1):
        model.train()
        idx = torch.randint(0, n_train, (int(t["batch_size"]),))
        batch = {k: v[idx] for k, v in train.items()}
        x, P_feats, P_raw, st = model_inputs(batch, preprocessor)

        optimizer.zero_grad()
        v_pred = model.forward_x(x, P_feats, P_raw, station_ids=st)
        loss, l_mse, _ = criterion(v_pred, batch["v_ref"], x)
        loss.backward()
        if t.get("grad_clip_norm"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(t["grad_clip_norm"]))
        optimizer.step()

        history.append(loss.item())
        if verbose and (step == 1 or step % int(t["log_every"]) == 0 or step == int(t["steps"])):
            print(f"step {step:5d} | loss {loss.item():10.4f} | rmse {l_mse.sqrt().item():7.3f} m/s")

    metrics = evaluate(model, val, preprocessor)
    metrics.update({"n_params": n_params, "train_seconds": time.time() - start,
                    "first_loss": history[0], "last_loss": history[-1]})
    if verbose:
        print(f"val RMSE {metrics['rmse']:.3f} m/s | max BC error {metrics['bc_max_abs_err']:.2e}")

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pt"))
        preprocessor.save(os.path.join(out_dir, "p_stats.json"))
        with open(os.path.join(out_dir, "config.yaml"), "w") as f:
            yaml.dump(cfg, f)
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        if verbose:
            print(f"Checkpoint written to {out_dir}")

    return {"history": history, "metrics": metrics}
