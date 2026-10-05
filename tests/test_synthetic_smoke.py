import math

import torch
from speed_profile_generator.training.train_synthetic import load_config, train_synthetic


def test_synthetic_training_smoke():
    """A few CPU steps through the same code path as scripts/train_synthetic.py."""
    cfg = load_config()
    cfg["data"].update({"n_train": 64, "n_val": 16, "num_points": 50})
    cfg["model"].update({"hidden_dim": 32, "num_blocks": 2})
    cfg["training"].update({"steps": 15, "batch_size": 16, "log_every": 100})

    torch.manual_seed(0)
    out = train_synthetic(cfg, out_dir=None, verbose=False)

    history = out["history"]
    assert len(history) == 15
    assert all(math.isfinite(h) for h in history)
    # Training should not diverge from the first step.
    assert history[-1] < history[0] * 1.5

    m = out["metrics"]
    assert math.isfinite(m["rmse"])
    # Hard boundary conditions are exact whatever the weights are.
    assert m["bc_max_abs_err"] < 1e-4


def test_synthetic_checkpoint_written(tmp_path):
    cfg = load_config()
    cfg["data"].update({"n_train": 32, "n_val": 8, "num_points": 30})
    cfg["model"].update({"hidden_dim": 16, "num_blocks": 1})
    cfg["training"].update({"steps": 2, "batch_size": 8, "log_every": 100})

    train_synthetic(cfg, out_dir=str(tmp_path), verbose=False)
    assert (tmp_path / "best_model.pt").exists()
    assert (tmp_path / "p_stats.json").exists()
