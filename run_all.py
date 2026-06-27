"""Single entry point graders run: loads every saved model across all
roles + joint components and prints the full results table vs baselines.
Self-contained on purpose (network classes defined inline, not imported
from each role's folder) to avoid module-name collisions across role_a_dqn/
role_b_policy/role_c_planning, all of which have their own agent.py/
train.py/networks.py.

Usage: python run_all.py --config simulator/configs/eval_standard.yaml --seeds 0,1,2,3,4
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn, yaml
from drone_dispatch_env import Config, evaluate, make_baseline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))
from common.obs_encoding import flatten_obs, flat_dim  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


# ---------- inline network definitions (mirror each role's, kept independent) ----------
def _mlp(in_dim, hidden, out_dim):
    layers, prev = [], in_dim
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU()]; prev = h
    layers.append(nn.Linear(prev, out_dim))
    return nn.Sequential(*layers)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        layers, prev = [], obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]; prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_actions)
    def forward(self, x): return self.head(self.trunk(x))

class DuelingQNetwork(nn.Module):
    def __init__(self, obs_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        layers, prev = [], obs_dim
        for h in hidden: layers += [nn.Linear(prev, h), nn.ReLU()]; prev = h
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(prev, 1); self.advantage_head = nn.Linear(prev, n_actions)
    def forward(self, x):
        h = self.trunk(x); v = self.value_head(h); a = self.advantage_head(h)
        return v + (a - a.mean(dim=1, keepdim=True))

class PolicyValueNet(nn.Module):  # Role B discrete (REINFORCE/A2C) -- policy head only needed here
    def __init__(self, obs_dim, n_actions, hidden=(256, 256)):
        super().__init__()
        layers, prev = [], obs_dim
        for h in hidden: layers += [nn.Linear(prev, h), nn.ReLU()]; prev = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(prev, n_actions)
        self.value_head = nn.Linear(prev, 1)
    def forward(self, x): return self.policy_head(self.trunk(x))


# ---------- policy wrappers ----------
class MaskedQPolicy:
    def __init__(self, net, env_cfg): self.net, self.env_cfg = net, env_cfg
    def act(self, obs):
        vec = flatten_obs(obs, self.env_cfg)
        with torch.no_grad():
            q = self.net(torch.as_tensor(vec, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        return int(np.argmax(np.where(obs["action_mask"].astype(bool), q, -np.inf)))

class OfflinePolicy:
    def __init__(self, net): self.net = net
    def act(self, obs):
        vec = np.concatenate([obs["drones"].flatten(), obs["orders"].flatten(),
                               obs["time"].astype(np.float32)]).astype(np.float32)
        with torch.no_grad():
            q = self.net(torch.as_tensor(vec, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
        return int(np.argmax(np.where(obs["action_mask"].astype(bool), q, -np.inf)))


# ---------- per-method evaluation ----------
def eval_seeded(config_path, eval_cfg, eval_seeds, net_class, obs_dim_fn, policy_class=MaskedQPolicy):
    full_path = os.path.join(REPO_ROOT, config_path)
    if not os.path.exists(full_path):
        return None
    with open(full_path) as f:
        hp = yaml.safe_load(f)
    obs_dim = obs_dim_fn(eval_cfg)
    costs = []
    for seed in hp.get("seeds", [0]):
        wp = os.path.join(REPO_ROOT, "weights", f"{hp['variant']}_seed{seed}.pt")
        if not os.path.exists(wp):
            continue
        net = net_class(obs_dim, eval_cfg.n_actions, hp["hidden_sizes"])
        net.load_state_dict(torch.load(wp, map_location="cpu")); net.eval()
        policy = policy_class(net) if policy_class is OfflinePolicy else policy_class(net, eval_cfg)
        m = evaluate(policy, eval_cfg, eval_seeds)["mean"]
        costs.append(m["cost_per_order"])
    if not costs:
        return None
    return float(np.mean(costs)), float(np.std(costs))


def eval_offline(weights_name, eval_cfg, eval_seeds):
    wp = os.path.join(REPO_ROOT, "weights", f"{weights_name}.pt")
    if not os.path.exists(wp):
        return None
    net = QNetwork(181, eval_cfg.n_actions, [256, 256])
    net.load_state_dict(torch.load(wp, map_location="cpu")); net.eval()
    m = evaluate(OfflinePolicy(net), eval_cfg, eval_seeds)["mean"]
    return m["cost_per_order"], None


MAIN_TABLE_METHODS = [
    # (display name, config path, net class, dueling-style obs_dim fn)
    ("dqn",            "configs/dqn.yaml",                     QNetwork,       flat_dim, MaskedQPolicy),
    ("double_dqn",     "configs/double_dqn.yaml",               QNetwork,       flat_dim, MaskedQPolicy),
    ("dueling_dqn",    "configs/dueling_dqn.yaml",               DuelingQNetwork, flat_dim, MaskedQPolicy),
    ("reinforce_gae",  "configs/reinforce_gae.yaml",             PolicyValueNet, flat_dim, MaskedQPolicy),
    ("a2c",            "configs/a2c.yaml",                       PolicyValueNet, flat_dim, MaskedQPolicy),
    ("a2c_noadvnorm",  "configs/a2c_ablation_noadvnorm.yaml",    PolicyValueNet, flat_dim, MaskedQPolicy),
    ("dyna_q",         "configs/dyna_q.yaml",                    QNetwork,       flat_dim, MaskedQPolicy),
    ("dyna_q_n0",      "configs/dyna_q_ablation_n0.yaml",        QNetwork,       flat_dim, MaskedQPolicy),
]
OFFLINE_METHODS = ["offline_naive", "offline_cql", "offline_bc"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="simulator/configs/eval_standard.yaml")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(REPO_ROOT, args.config)
    cfg = Config.from_yaml(cfg_path)
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]

    results = {}
    for name in ["random", "greedy_nearest", "milp_rolling"]:
        m = evaluate(make_baseline(name, cfg), cfg, seeds)["mean"]
        results[name] = {"cost_per_order_mean": m["cost_per_order"], "cost_per_order_std": None}

    for name, config_path, net_class, dim_fn, policy_class in MAIN_TABLE_METHODS:
        r = eval_seeded(config_path, cfg, seeds, net_class, dim_fn, policy_class)
        results[name] = ({"cost_per_order_mean": r[0], "cost_per_order_std": r[1]} if r is not None
                          else {"cost_per_order_mean": None, "cost_per_order_std": None, "note": "weights not found"})

    for name in OFFLINE_METHODS:
        r = eval_offline(name, cfg, seeds)
        results[name] = ({"cost_per_order_mean": r[0], "cost_per_order_std": r[1]} if r is not None
                          else {"cost_per_order_mean": None, "cost_per_order_std": None, "note": "weights not found"})

    print("=== Main table (DroneDispatch-v0, cost_per_order, lower is better) ===")
    print(json.dumps(results, indent=2))

    print("\n=== DDPG (DroneControl-v0) -- different env/metric, reported separately ===")
    print("see weights/ddpg_seed*_best.pt; run code/role_b_policy/ddpg/evaluate_agent.py for success_rate")

    print("\n=== Multi-agent IDQN -- different env/metric, reported separately ===")
    print("see weights/idqn_seed0.pt; run code/multi_agent/evaluate_agent.py for team_return")


if __name__ == "__main__":
    main()
