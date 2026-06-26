"""CQL (Conservative Q-Learning, Kumar et al. 2020) fix for the naive
offline DQN's overestimation. Adds one term to the same TD loss:
  alpha * (logsumexp_a Q(s,a) - Q(s,a_data)).mean()
This directly penalizes Q being high for actions OTHER than the one the
behavior policy actually took at s, which is exactly the failure mode
train_naive.py demonstrates (Q rising for actions never validated by data).
Usage: python train_cql.py --data ../../logs/D_logs.npz --config ../../configs/offline_cql.yaml"""
import argparse, csv, os, sys
import numpy as np, torch, torch.nn.functional as F, yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "role_a_dqn"))
from networks import QNetwork
from drone_dispatch_env import load_offline_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../logs/D_logs.npz")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config) as f: hp = yaml.safe_load(f)
    d = load_offline_dataset(args.data)
    obs, act, rew, nobs = d["observations"], d["actions"], d["rewards"], d["next_observations"]
    done = (d["terminals"] | d["timeouts"]).astype(np.float32)
    obs_dim, n_actions = obs.shape[1], hp["n_actions"]

    torch.manual_seed(hp.get("seed", 0)); rng = np.random.default_rng(hp.get("seed", 0))
    q = QNetwork(obs_dim, n_actions, hp["hidden_sizes"])
    qt = QNetwork(obs_dim, n_actions, hp["hidden_sizes"]); qt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=hp["lr"])
    alpha = hp["cql_alpha"]

    os.makedirs(os.path.dirname(hp["log_path"]), exist_ok=True)
    f = open(hp["log_path"], "w", newline=""); w = csv.writer(f)
    w.writerow(["step", "loss", "mean_q", "max_q", "cql_term"])

    N = len(obs)
    for step in range(1, hp["train_steps"] + 1):
        idx = rng.integers(0, N, hp["batch_size"])
        o = torch.as_tensor(obs[idx]); a = torch.as_tensor(act[idx])
        r = torch.as_tensor(rew[idx]); no = torch.as_tensor(nobs[idx])
        dn = torch.as_tensor(done[idx])
        qvals = q(o)
        qsa = qvals.gather(1, a.unsqueeze(1).long()).squeeze(1)
        with torch.no_grad():
            target = r + hp["gamma"] * (1 - dn) * qt(no).max(dim=1).values
        td_loss = F.smooth_l1_loss(qsa, target)
        cql_term = (torch.logsumexp(qvals, dim=1) - qsa).mean()
        loss = td_loss + alpha * cql_term
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
        opt.step()
        if step % hp["target_update_every"] == 0: qt.load_state_dict(q.state_dict())
        if step % hp["eval_every"] == 0:
            with torch.no_grad():
                allq = q(torch.as_tensor(obs[rng.integers(0, N, 2000)]))
            w.writerow([step, float(td_loss.item()), float(allq.mean()), float(allq.max()), float(cql_term.item())]); f.flush()
            print(f"step {step}/{hp['train_steps']} td_loss={td_loss.item():.2f} "
                  f"mean_q={allq.mean():.2f} cql_term={cql_term.item():.2f}", flush=True)
    f.close()
    torch.save(q.state_dict(), hp["weights_path"])
    print(f"saved -> {hp['weights_path']}")


if __name__ == "__main__":
    main()
