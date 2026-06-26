"""Evaluates a trained DDPG actor on DroneControl-v0 over many seeds.

No baseline comparison here -- the simulator ships no random/heuristic
baseline for this sub-env, and the spec's "beat greedy_nearest" bar is
specific to the dispatch env. The deliverable is a working, stable
DDPG with mean ± std reported across the 3 training seeds, evaluated on a
held-out set of episode seeds distinct from training.

Usage:
    python code/role_b_policy/ddpg/evaluate_agent.py --config configs/ddpg.yaml
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml
import gymnasium as gym

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ""))

from networks import Actor                # noqa: E402
from drone_dispatch_env import Config      # noqa: E402


def run_episodes(actor: Actor, env_cfg: Config, seeds):
    env = gym.make("DroneControl-v0", config=env_cfg)
    returns, successes, energies = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done, ep_return = False, 0.0
        while not done:
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                a = actor(x).squeeze(0).numpy()
            obs, r, term, trunc, info = env.step(a)
            ep_return += r
            done = term or trunc
        success = term and not trunc and env.unwrapped.soc > 0.0
        returns.append(ep_return)
        successes.append(float(success))
        energies.append(float(env.unwrapped.cfg.init_soc - env.unwrapped.soc))
    return {"mean_return": float(np.mean(returns)), "std_return": float(np.std(returns)),
            "success_rate": float(np.mean(successes)), "mean_energy_used": float(np.mean(energies))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval_seeds", default="0,1,2,3,4,5,6,7,8,9")
    args = ap.parse_args()

    repo_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    with open(os.path.join(repo_root, args.config)) as f:
        hp = yaml.safe_load(f)
    run_name = hp["variant"]
    env_cfg = Config.from_yaml(os.path.join(repo_root, hp["env_config"]))
    eval_seeds = [int(s) for s in args.eval_seeds.split(",") if s != ""]

    per_seed = []
    for seed in hp["seeds"]:
        best_path = os.path.join(repo_root, "weights", f"{run_name}_seed{seed}_best.pt")
        final_path = os.path.join(repo_root, "weights", f"{run_name}_seed{seed}.pt")
        weights_path = best_path if os.path.exists(best_path) else final_path
        if not os.path.exists(weights_path):
            print(f"  [skip] no weights at {weights_path} yet")
            continue
        which = "best" if weights_path == best_path else "final"
        actor = Actor(7, hp["hidden_sizes"])
        actor.load_state_dict(torch.load(weights_path, map_location="cpu"))
        actor.eval()
        m = run_episodes(actor, env_cfg, eval_seeds)
        per_seed.append(m)
        print(f"  seed {seed} ({which}): success_rate={m['success_rate']:.2f}  mean_return={m['mean_return']:.2f}  "
              f"mean_energy_used={m['mean_energy_used']:.3f}")

    if per_seed:
        agg = {
            "success_rate_mean": float(np.mean([m["success_rate"] for m in per_seed])),
            "success_rate_std": float(np.std([m["success_rate"] for m in per_seed])),
            "mean_return_mean": float(np.mean([m["mean_return"] for m in per_seed])),
            "mean_return_std": float(np.std([m["mean_return"] for m in per_seed])),
        }
        print("\n=== DDPG summary (mean ± std across training seeds) ===")
        print(f"success_rate: {agg['success_rate_mean']:.3f} ± {agg['success_rate_std']:.3f}")
        print(f"mean_return : {agg['mean_return_mean']:.2f} ± {agg['mean_return_std']:.2f}")

        out_csv = os.path.join(repo_root, "logs", f"{run_name}_comparison.csv")
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric", "mean", "std"])
            w.writerow(["success_rate", agg["success_rate_mean"], agg["success_rate_std"]])
            w.writerow(["mean_return", agg["mean_return_mean"], agg["mean_return_std"]])
        print(f"\nwritten -> {out_csv}")


if __name__ == "__main__":
    main()
