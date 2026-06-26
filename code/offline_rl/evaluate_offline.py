"""Runs each offline-trained policy in the LIVE env (with proper action
masking applied here, at decision time -- the training itself never saw
the mask, only this evaluation step does) and reports cost_per_order vs
baselines. This is where naive-DQN-offline's failure and CQL's fix
actually become visible as a number, not just a Q-value curve.
Usage: python evaluate_offline.py --config ../../configs/offline_cql.yaml"""
import argparse, os, sys
import numpy as np, torch, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "role_a_dqn"))
from networks import QNetwork
from drone_dispatch_env import Config, evaluate, make_baseline


def flatten_obs181(obs):  # matches drone_dispatch_env.offline._flatten_obs exactly
    return np.concatenate([obs["drones"].flatten(), obs["orders"].flatten(),
                            obs["time"].astype(np.float32)]).astype(np.float32)


class OfflinePolicy:
    def __init__(self, net): self.net = net
    def act(self, obs):
        vec = flatten_obs181(obs)
        with torch.no_grad():
            q = self.net(torch.as_tensor(vec, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        return int(np.argmax(np.where(obs["action_mask"].astype(bool), q, -np.inf)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, help="one or more offline_*.yaml")
    ap.add_argument("--eval_config", default="../../simulator/configs/eval_standard.yaml")
    ap.add_argument("--eval_seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    cfg = Config.from_yaml(args.eval_config)
    seeds = [int(s) for s in args.eval_seeds.split(",") if s]

    print("=== Baseline comparison (cost_per_order, lower is better) ===")
    for name in ["random", "greedy_nearest", "milp_rolling"]:
        m = evaluate(make_baseline(name, cfg), cfg, seeds)["mean"]
        print(f"{name:20s} {m['cost_per_order']:.3f}")

    for cpath in args.configs:
        with open(cpath) as f: hp = yaml.safe_load(f)
        net = QNetwork(181, hp["n_actions"], hp["hidden_sizes"])
        net.load_state_dict(torch.load(hp["weights_path"], map_location="cpu")); net.eval()
        m = evaluate(OfflinePolicy(net), cfg, seeds)["mean"]
        print(f"{hp['run_name']:20s} {m['cost_per_order']:.3f}  success={m['success_rate']:.3f}")


if __name__ == "__main__":
    main()
