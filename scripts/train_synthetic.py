"""
Train the V4 SpeedProfileGenerator on the synthetic kinematic generator (CPU).

This is the supported route for running the training code without the
industrial dataset. It uses a reduced conditioning set (no track features,
no station ids); see src/speed_profile_generator/training/train_synthetic.py.

Usage:
    python scripts/train_synthetic.py --steps 200 --batch_size 64 --out_dir runs/synthetic
"""
import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from speed_profile_generator.training.train_synthetic import load_config, train_synthetic


def main():
    parser = argparse.ArgumentParser(description="Synthetic (reduced-conditioning) training on CPU")
    parser.add_argument("--config", type=str, default=os.path.join(REPO_ROOT, "configs", "synthetic_smoke.yaml"))
    parser.add_argument("--steps", type=int, default=None, help="Override training.steps")
    parser.add_argument("--batch_size", type=int, default=None, help="Override training.batch_size")
    parser.add_argument("--n_train", type=int, default=None, help="Override data.n_train")
    parser.add_argument("--out_dir", type=str, default="runs/synthetic")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config if os.path.exists(args.config) else None)
    if args.steps is not None: cfg["training"]["steps"] = args.steps
    if args.batch_size is not None: cfg["training"]["batch_size"] = args.batch_size
    if args.n_train is not None: cfg["data"]["n_train"] = args.n_train
    if args.seed is not None: cfg["seed"] = args.seed

    train_synthetic(cfg, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
