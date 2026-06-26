"""Evaluate Dyna-Q vs baselines. Usage:
python evaluate_agent.py --config ../../configs/dyna_q.yaml"""
import argparse, csv, os, sys
import numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "role_a_dqn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ""))
from common.obs_encoding import flatten_obs, flat_dim
from networks import QNetwork
from drone_dispatch_env import Config, evaluate, make_baseline


class Policy:
    def __init__(self, net, cfg): self.net, self.cfg = net, cfg
    def act(self, obs):
        vec = flatten_obs(obs, self.cfg)
        with torch.no_grad():
            q = self.net(torch.as_tensor(vec, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        return int(np.argmax(np.where(obs["action_mask"].astype(bool), q, -np.inf)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--eval_config", default="simulator/configs/eval_standard.yaml")
    ap.add_argument("--eval_seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    with open(os.path.join(repo_root, args.config)) as f: hp = yaml.safe_load(f)
    run_name = hp["variant"]
    cfg = Config.from_yaml(os.path.join(repo_root, args.eval_config))
    seeds = [int(s) for s in args.eval_seeds.split(",") if s]

    results = {}
    for name in ["random", "greedy_nearest", "milp_rolling"]:
        m = evaluate(make_baseline(name, cfg), cfg, seeds)["mean"]
        results[name] = (m["cost_per_order"], None)

    costs = []
    for seed in hp["seeds"]:
        wp = os.path.join(repo_root, "weights", f"{run_name}_seed{seed}.pt")
        if not os.path.exists(wp): continue
        net = QNetwork(flat_dim(cfg), cfg.n_actions, hp["hidden_sizes"])
        net.load_state_dict(torch.load(wp, map_location="cpu")); net.eval()
        m = evaluate(Policy(net, cfg), cfg, seeds)["mean"]
        costs.append(m["cost_per_order"])
        print(f"  seed {seed}: cost_per_order={m['cost_per_order']:.3f}")
    if costs:
        results[run_name] = (float(np.mean(costs)), float(np.std(costs)))

    print("\n=== Baseline comparison ===")
    for n, (m, s) in results.items():
        print(f"{n:20s} {m:8.3f}" + (f" ± {s:.3f}" if s is not None else ""))
    with open(os.path.join(repo_root, "logs", f"{run_name}_comparison.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["method", "cost_per_order_mean", "cost_per_order_std"])
        for n, (m, s) in results.items(): w.writerow([n, m, s])


if __name__ == "__main__":
    main()
