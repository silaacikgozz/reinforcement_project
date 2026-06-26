"""Builds D_logs.npz (the team's pooled offline dataset) using the
simulator's own mixed-behavior generator (60% greedy_nearest + 40% noisy
random) -- this stands in for "pooling all three members' logged
trajectories": it's the instructor-provided, ready-made equivalent, and
avoids the inconsistency of our three roles using three different state
encodings (750-d masked vector, 181-d here, 59-d in multi_agent).
Usage: python build_dataset.py --out ../../logs/D_logs.npz --n 200000"""
import argparse, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from drone_dispatch_env import Config, generate_offline_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="../../logs/D_logs.npz")
ap.add_argument("--n", type=int, default=200000)
ap.add_argument("--config", default="../../simulator/configs/eval_standard.yaml")
args = ap.parse_args()
cfg = Config.from_yaml(args.config)
generate_offline_dataset(args.out, config=cfg, min_transitions=args.n, base_seed=1000)
print(f"saved -> {args.out}")
