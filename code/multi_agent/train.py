"""IDQN with parameter sharing on DroneDispatchMA-v0 (Ch. 21). ONE shared
QNetwork (reused from Role A, unmodified) is used by every drone -- every
agent's (obs,action,reward,next_obs,done) tuple goes into the SAME replay
buffer, since parameter sharing treats all drones as homogeneous samples
for one network. No action mask here (env's 4 actions are always
nominally callable; invalid attempts are silently no-op'd internally), so
we pass an all-ones mask to Role A's DQNAgent/ReplayBuffer unchanged.
Usage: python train.py --config ../../configs/idqn.yaml"""
import argparse, csv, os, sys, time
import numpy as np, torch, yaml, gymnasium as gym
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "role_a_dqn"))
from agent import DQNAgent
from replay_buffer import ReplayBuffer
from drone_dispatch_env import Config

OBS_DIM, N_ACTIONS = 59, 4
ONES_MASK = np.ones(N_ACTIONS, dtype=np.float32)


def quick_eval(agent, env_cfg, seed):
    env = gym.make("DroneDispatchMA-v0", config=env_cfg)
    obs, _ = env.reset(seed=seed)
    total_r, delivered_proxy, t = 0.0, 0.0, 0
    done_all = False
    while not done_all:
        actions = {a: agent.greedy_action(obs[a], ONES_MASK) for a in obs}
        obs, rewards, terms, truncs, _ = env.step(actions)
        total_r += sum(rewards.values())
        t += 1
        done_all = all(terms.values()) or all(truncs.values())
    return total_r


def batched_act(agent, obs_dict, eps, rng):
    """One forward pass for all agents at once instead of 8 separate calls
    (was the main speed bottleneck -- ~4 steps/s on slower CPUs)."""
    keys = list(obs_dict.keys())
    x = torch.as_tensor(np.stack([obs_dict[k] for k in keys]), dtype=torch.float32)
    with torch.no_grad():
        q = agent.q_net(x).numpy()
    actions = {}
    for i, k in enumerate(keys):
        if rng.random() < eps:
            actions[k] = int(rng.integers(0, N_ACTIONS))
        else:
            actions[k] = int(np.argmax(q[i]))
    return actions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_steps", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as f: hp = yaml.safe_load(f)
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    env_cfg = Config.from_yaml(os.path.join(repo_root, hp["env_config"]))
    train_steps = args.train_steps or hp["train_steps"]
    seed = hp.get("seed", 0)

    agent = DQNAgent(OBS_DIM, N_ACTIONS, hidden_sizes=hp["hidden_sizes"], dueling=False, double=True,
                      gamma=hp["gamma"], lr=hp["lr"], target_update_every=hp["target_update_every"],
                      device="cpu", seed=seed)
    buffer = ReplayBuffer(hp["buffer_size"], OBS_DIM, N_ACTIONS, seed=seed)

    log_path = os.path.join(repo_root, "logs", "idqn_seed0.csv")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    f = open(log_path, "w", newline=""); w = csv.writer(f)
    w.writerow(["step", "episode", "return", "loss", "eval_return"])

    env = gym.make("DroneDispatchMA-v0", config=env_cfg)
    obs, _ = env.reset(seed=seed)
    ep_return, loss_val, episode = 0.0, float("nan"), 0
    eval_every = hp.get("eval_every", 5000)
    heartbeat = max(1, min(500, eval_every // 2))

    agent_rng = np.random.default_rng(seed + 5000)
    t0 = time.time()
    for step in range(1, train_steps + 1):
        eps = max(hp["epsilon_end"], hp["epsilon_start"] - step / hp["epsilon_decay_steps"] * (hp["epsilon_start"] - hp["epsilon_end"]))
        actions = batched_act(agent, obs, eps, agent_rng)
        nobs, rewards, terms, truncs, _ = env.step(actions)
        done_all = all(terms.values()) or all(truncs.values())
        for a in obs:  # every agent's transition -> the SAME shared buffer
            buffer.add(obs[a], actions[a], rewards[a], nobs[a], ONES_MASK, terms[a])
        ep_return += sum(rewards.values())
        obs = nobs

        if len(buffer) >= hp["batch_size"]:
            loss_val = agent.learn(buffer.sample(hp["batch_size"]))
        if step % heartbeat == 0:
            rate = step / max(time.time() - t0, 1e-6)
            print(f"  step {step}/{train_steps} ep={episode} loss={loss_val:.3f} eps={eps:.3f} "
                  f"{rate:.0f} steps/s ETA {(train_steps-step)/max(rate,1e-6)/60:.1f} min", flush=True)
        eval_ret = float("nan")
        if step % eval_every == 0:
            eval_ret = quick_eval(agent, env_cfg, seed + 999000)
            print(f"  [eval @ step {step}] team_return={eval_ret:.2f}", flush=True)
        if done_all or step % eval_every == 0:
            w.writerow([step, episode, ep_return, loss_val, eval_ret]); f.flush()
        if done_all:
            episode += 1; ep_return = 0.0
            obs, _ = env.reset(seed=seed + episode)
    f.close()
    wp = os.path.join(repo_root, "weights", "idqn_seed0.pt")
    torch.save(agent.q_net.state_dict(), wp)
    print(f"saved -> {wp}")


if __name__ == "__main__":
    main()
