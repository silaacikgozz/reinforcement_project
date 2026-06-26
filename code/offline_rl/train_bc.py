"""Behavioral cloning baseline: supervised classifier obs->action on the
dataset's (observation, action) pairs. No reward/Q-learning at all --
this is the bar CQL must also beat (spec requirement), since BC can look
deceptively good (just imitates a 60%-greedy behavior policy) without any
genuine value-based reasoning.
Usage: python train_bc.py --data ../../logs/D_logs.npz --config ../../configs/offline_bc.yaml"""
import argparse, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "role_a_dqn"))
from networks import QNetwork
from drone_dispatch_env import load_offline_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="../../logs/D_logs.npz")
ap.add_argument("--config", required=True)
args = ap.parse_args()
with open(args.config) as f: hp = yaml.safe_load(f)
d = load_offline_dataset(args.data)
obs, act = d["observations"], d["actions"]
obs_dim, n_actions = obs.shape[1], hp["n_actions"]

torch.manual_seed(hp.get("seed", 0)); rng = np.random.default_rng(hp.get("seed", 0))
net = QNetwork(obs_dim, n_actions, hp["hidden_sizes"])  # logits head reused as a classifier
opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
N = len(obs)
for step in range(1, hp["train_steps"] + 1):
    idx = rng.integers(0, N, hp["batch_size"])
    o = torch.as_tensor(obs[idx]); a = torch.as_tensor(act[idx]).long()
    loss = F.cross_entropy(net(o), a)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 2000 == 0: print(f"step {step}/{hp['train_steps']} ce_loss={loss.item():.3f}", flush=True)
torch.save(net.state_dict(), hp["weights_path"])
print(f"saved -> {hp['weights_path']}")
