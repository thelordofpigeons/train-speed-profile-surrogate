"""
CPU inference latency benchmark for SpeedProfileGenerator.

Builds the model from the model section of a training config (default: the V4
config), feeds it RANDOM inputs (no dataset involved) and reports the latency
of one forward pass that produces a full speed profile of T points.

The weights are random: latency depends on the architecture and input shapes,
not on trained values.

Usage:
    python scripts/benchmark_inference_cpu.py
    python scripts/benchmark_inference_cpu.py --threads 1 --points 128 --runs 500
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from speed_profile_generator.models.speed_profile_generator import SpeedProfileGenerator


def build_model(m_cfg):
    return SpeedProfileGenerator(
        param_dim=m_cfg["param_dim"],
        hidden_dim=m_cfg["hidden_dim"],
        num_blocks=m_cfg["num_blocks"],
        track_feat_dim=m_cfg.get("track_feat_dim", 0),
        num_stations=100,
        emb_dim=m_cfg.get("emb_dim", 4),
        num_freqs=m_cfg.get("num_freqs", 16),
        freq_base=m_cfg.get("freq_base", 10.0),
        extra_features_dim=2,  # log_length + is_segment, as in train_h5.py
    )


def benchmark(args):
    torch.set_num_threads(args.threads)
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    m_cfg = cfg["model"]

    model = build_model(m_cfg).eval()
    n_params = sum(p.numel() for p in model.parameters())

    B, T = args.batch, args.points
    track_dim = m_cfg.get("track_feat_dim", 0)

    # P_raw layout: [x0, xf, v0, vf, st_start, st_end, dur, M]
    P_raw = torch.tensor([[0.0, 10000.0, 10.0, 20.0, 0.0, 0.0, 600.0, 4.0e5]]).repeat(B, 1)
    x = torch.linspace(0.0, 10000.0, T).view(1, T, 1).repeat(B, 1, 1)
    P_feats = torch.randn(B, m_cfg["param_dim"] + 2)  # 6 continuous + 2 extras
    track_feats = torch.randn(B, T, track_dim) if track_dim > 0 else None

    with torch.no_grad():
        for _ in range(args.warmup):
            model.forward_x(x, P_feats, P_raw, track_feats=track_feats)

        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            model.forward_x(x, P_feats, P_raw, track_feats=track_feats)
            times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    print(f"Config: {os.path.relpath(args.config, REPO_ROOT)}")
    print(f"Model: hidden_dim={m_cfg['hidden_dim']}, num_blocks={m_cfg['num_blocks']}, "
          f"track_feat_dim={track_dim}, parameters={n_params:,}")
    print(f"Device: cpu, threads={args.threads}, torch={torch.__version__}")
    print(f"Input: batch={B}, points={T} (random values)")
    print(f"Latency over {args.runs} runs (ms): "
          f"median={np.median(times):.3f}  p95={np.percentile(times, 95):.3f}  "
          f"mean={times.mean():.3f}  min={times.min():.3f}")


def main():
    parser = argparse.ArgumentParser(description="CPU inference latency benchmark (random inputs)")
    parser.add_argument("--config", type=str,
                        default=os.path.join(REPO_ROOT, "configs", "train_h5_config_v4_full.yaml"))
    parser.add_argument("--threads", type=int, default=1, help="torch.set_num_threads")
    parser.add_argument("--points", type=int, default=128, help="Points per profile (T)")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=500)
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
