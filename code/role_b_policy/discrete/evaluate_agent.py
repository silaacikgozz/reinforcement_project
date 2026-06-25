"""Loads trained weights for a Role B discrete run (3 training seeds) and
produces the required baseline-comparison table, same protocol as Role A's
evaluate_agent.py: 5 held-out eval seeds per trained model, mean ± std
across the 3 training seeds.

Usage:
    python code/role_b_policy/discrete/evaluate_agent.py --config configs/reinforce_gae.yaml
    python code/role_b_policy/discrete/evaluate_agent.py --config configs/a2c.yaml
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ""))

from common.obs_encoding import flatten_obs, flat_dim   # noqa: E402
from networks import PolicyValueNet                       # noqa: E402
from drone_dispatch_env import Config, evaluate, make_baseline   # noqa: E402


class TrainedPolicy:
    """Satisfies agent_interface.Policy: greedy (argmax over masked logits),
    no sampling, for a fair deterministic comparison against the baselines."""

    def __init__(self, net: PolicyValueNet, env_cfg: Config):
        self.net = net
        self.env_cfg = env_cfg

    def act(self, obs):
        vec = flatten_obs(obs, self.env_cfg)
        x = torch.as_tensor(vec, dtype=torch.float32).unsqueeze(0)
        m = torch.as_tensor(obs["action_mask"], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, _ = self.net(x, m)
        return int(torch.argmax(logits, dim=1).item())


def load_policy(weights_path: str, hp: dict, env_cfg: Config) -> TrainedPolicy:
    net = PolicyValueNet(flat_dim(env_cfg), env_cfg.n_actions, hp["hidden_sizes"])
    net.load_state_dict(torch.load(weights_path, map_location="cpu"))
    net.eval()
    return TrainedPolicy(net, env_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval_config", default="simulator/configs/eval_standard.yaml")
    ap.add_argument("--eval_seeds", default="0,1,2,3,4")
    args = ap.parse_args()

    repo_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    with open(os.path.join(repo_root, args.config)) as f:
        hp = yaml.safe_load(f)
    run_name = hp["variant"]

    eval_cfg = Config.from_yaml(os.path.join(repo_root, args.eval_config))
    eval_seeds = [int(s) for s in args.eval_seeds.split(",") if s != ""]

    results = {}
    for name in ["random", "greedy_nearest", "milp_rolling"]:
        m = evaluate(make_baseline(name, eval_cfg), eval_cfg, eval_seeds)["mean"]
        results[name] = {"cost_per_order_mean": m["cost_per_order"], "cost_per_order_std": float("nan")}

    per_seed_costs = []
    for seed in hp["seeds"]:
        weights_path = os.path.join(repo_root, "weights", f"{run_name}_seed{seed}.pt")
        if not os.path.exists(weights_path):
            print(f"  [skip] no weights at {weights_path} yet")
            continue
        policy = load_policy(weights_path, hp, eval_cfg)
        m = evaluate(policy, eval_cfg, eval_seeds)["mean"]
        per_seed_costs.append(m["cost_per_order"])
        print(f"  seed {seed}: cost_per_order={m['cost_per_order']:.3f}  success={m['success_rate']:.3f}")

    if per_seed_costs:
        results[run_name] = {
            "cost_per_order_mean": float(np.mean(per_seed_costs)),
            "cost_per_order_std": float(np.std(per_seed_costs)),
        }

    print("\n=== Baseline comparison (cost_per_order, lower is better) ===")
    for name, r in results.items():
        std_str = f" ± {r['cost_per_order_std']:.3f}" if not np.isnan(r["cost_per_order_std"]) else ""
        print(f"{name:20s} {r['cost_per_order_mean']:8.3f}{std_str}")

    out_csv = os.path.join(repo_root, "logs", f"{run_name}_comparison.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "cost_per_order_mean", "cost_per_order_std"])
        for name, r in results.items():
            w.writerow([name, r["cost_per_order_mean"], r["cost_per_order_std"]])
    print(f"\nwritten -> {out_csv}")


if __name__ == "__main__":
    main()
