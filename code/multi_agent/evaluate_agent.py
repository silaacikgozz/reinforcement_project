"""Head-to-head: IDQN (decentralized, parameter-shared) team_return vs
centralized Dueling DQN's episode_return (same underlying reward weights,
so the two totals are comparable even though the envs are wrapped
differently). Usage: python evaluate_agent.py --config ../../configs/idqn.yaml"""
import argparse, os, sys
import numpy as np, torch, yaml, gymnasium as gym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "role_a_dqn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from networks import QNetwork, DuelingQNetwork
from common.obs_encoding import flatten_obs, flat_dim
from drone_dispatch_env import Config, evaluate

OBS_DIM, N_ACTIONS = 59, 4
ONES_MASK = np.ones(N_ACTIONS, dtype=np.float32)


def idqn_team_return(net, env_cfg, seeds):
    rets = []
    for s in seeds:
        env = gym.make("DroneDispatchMA-v0", config=env_cfg)
        obs, _ = env.reset(seed=s)
        total, done_all = 0.0, False
        while not done_all:
            actions = {}
            for a in obs:
                with torch.no_grad():
                    q = net(torch.as_tensor(obs[a], dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                actions[a] = int(np.argmax(q))
            obs, rewards, terms, truncs, _ = env.step(actions)
            total += sum(rewards.values())
            done_all = all(terms.values()) or all(truncs.values())
        rets.append(total)
    return float(np.mean(rets)), float(np.std(rets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--centralized_weights", default="weights/dueling_dqn_seed0.pt")
    ap.add_argument("--eval_seeds", default="0,1,2,3,4")
    args = ap.parse_args()
    with open(args.config) as f: hp = yaml.safe_load(f)
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    env_cfg = Config.from_yaml(os.path.join(repo_root, hp["env_config"]))
    seeds = [int(s) for s in args.eval_seeds.split(",") if s]

    net = QNetwork(OBS_DIM, N_ACTIONS, hp["hidden_sizes"])
    wp = os.path.join(repo_root, "weights", "idqn_seed0.pt")
    net.load_state_dict(torch.load(wp, map_location="cpu")); net.eval()
    idqn_mean, idqn_std = idqn_team_return(net, env_cfg, seeds)
    print(f"IDQN (decentralized, parameter-shared)  team_return = {idqn_mean:.2f} ± {idqn_std:.2f}")

    cw_path = os.path.join(repo_root, args.centralized_weights)
    if os.path.exists(cw_path):
        cnet = DuelingQNetwork(flat_dim(env_cfg), env_cfg.n_actions, [256, 256])
        cnet.load_state_dict(torch.load(cw_path, map_location="cpu")); cnet.eval()

        class P:
            def act(self, obs):
                vec = flatten_obs(obs, env_cfg)
                with torch.no_grad():
                    q = cnet(torch.as_tensor(vec, dtype=torch.float32).unsqueeze(0)).squeeze(0).numpy()
                return int(np.argmax(np.where(obs["action_mask"].astype(bool), q, -np.inf)))
        m = evaluate(P(), env_cfg, seeds)["mean"]
        print(f"Centralized Dueling DQN                 episode_return = {m['episode_return']:.2f}")
    else:
        print(f"  [skip] centralized weights not found at {cw_path}")


if __name__ == "__main__":
    main()
