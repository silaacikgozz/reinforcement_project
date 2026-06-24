"""Training entry point for Role A. Single command:

    python code/role_a_dqn/train.py --config configs/dqn.yaml

Reads every hyperparameter from the yaml (nothing hardcoded here per
Section 6 of the spec). Trains one agent per seed in `seeds`, writes:
  logs/<run_name>_seed<k>.csv      (step, episode, return, loss, epsilon,
                                     eval_cost_per_order [NaN unless this is
                                     an eval checkpoint])
  weights/<run_name>_seed<k>.pt    (q_net state_dict, for evaluate_agent.py)

`variant` in the config selects which required stage this run is:
  dqn         -> double=False, dueling=False
  double_dqn  -> double=True,  dueling=False
  dueling_dqn -> double=True,  dueling=True   (stacks on top of Double, matching
                                                the course's cumulative DQN -> Double -> Dueling chain)
`target_network: true/false` is the Section-4 ablation knob (kept independent
of `variant` so we can run the ablation on any of the three).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import yaml
import gymnasium as gym

sys.path.insert(0, os.path.dirname(__file__))                       # role_a_dqn/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "")) # code/

from common.obs_encoding import flatten_obs, flat_dim   # noqa: E402
from agent import DQNAgent                              # noqa: E402
from replay_buffer import ReplayBuffer                  # noqa: E402
from drone_dispatch_env import Config                    # noqa: E402


VARIANT_FLAGS = {
    "dqn":         dict(double=False, dueling=False),
    "double_dqn":  dict(double=True,  dueling=False),
    "dueling_dqn": dict(double=True,  dueling=True),
}


def linear_epsilon(step: int, start: float, end: float, decay_steps: int) -> float:
    frac = min(1.0, step / max(1, decay_steps))
    return start + frac * (end - start)


def quick_eval(agent: DQNAgent, env_cfg: Config, seed: int) -> float:
    """One greedy (epsilon=0) episode, for a cheap in-training progress signal.
    Official, multi-seed baseline comparison happens in evaluate_agent.py --
    this is NOT the number we report."""
    env = gym.make("DroneDispatch-v0", config=env_cfg)
    obs, _ = env.reset(seed=seed)
    delivered, cost_terms = 0, 0.0
    done = False
    while not done:
        vec = flatten_obs(obs, env_cfg)
        a = agent.greedy_action(vec, obs["action_mask"])
        obs, r, term, trunc, info = env.step(a)
        done = term or trunc
    s = env.unwrapped.stats
    delivered = max(s["delivered"], 1)
    cost = s["energy"] + s["late_cost"] + s["drop_cost"] + s["depletion_cost"]
    return cost / delivered


def train_one_seed(env_cfg: Config, hp: dict, seed: int, run_name: str,
                    log_dir: str, weights_dir: str, train_steps: int | None = None,
                    checkpoint_every: int = 10000):
    train_steps = train_steps or hp["train_steps"]
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_dim = flat_dim(env_cfg)
    n_actions = env_cfg.n_actions
    flags = VARIANT_FLAGS[hp["variant"]]

    agent = DQNAgent(
        obs_dim=obs_dim, n_actions=n_actions,
        hidden_sizes=hp["hidden_sizes"],
        dueling=flags["dueling"], double=flags["double"],
        use_target_network=hp.get("target_network", True),
        gamma=hp["gamma"], lr=hp["lr"],
        target_update_every=hp["target_update_every"],
        device="cpu", seed=seed,
    )
    buffer = ReplayBuffer(hp["buffer_size"], obs_dim, n_actions, seed=seed)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}_seed{seed}.csv")
    ckpt_path = os.path.join(weights_dir, f"{run_name}_seed{seed}_ckpt.pt")

    start_step = 0
    episode = 0
    env_rng = np.random.default_rng(seed)

    # ---- resume from checkpoint if one exists ----
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        agent.q_net.load_state_dict(ckpt["q_net"])
        if agent.use_target_network:
            agent.target_net.load_state_dict(ckpt["target_net"])
        agent.opt.load_state_dict(ckpt["opt"])
        agent._learn_steps = ckpt["learn_steps"]
        start_step = ckpt["step"]
        episode = ckpt["episode"]
        env_rng.bit_generator.state = ckpt["env_rng_state"]
        f = open(log_path, "a", newline="")
        print(f"[{run_name} seed={seed}] resumed from step {start_step}/{train_steps}", flush=True)
    else:
        f = open(log_path, "w", newline="")
        csv.writer(f).writerow(["step", "episode", "return", "loss", "epsilon", "eval_cost_per_order"])
        print(f"[{run_name} seed={seed}] starting fresh, target={train_steps} steps "
              f"(obs_dim={obs_dim}, n_actions={n_actions}, device=cpu)", flush=True)

    writer = csv.writer(f)

    env = gym.make("DroneDispatch-v0", config=env_cfg)
    obs, _ = env.reset(seed=int(env_rng.integers(0, 2**31 - 1)))
    vec = flatten_obs(obs, env_cfg)
    mask = obs["action_mask"]

    ep_return = 0.0
    loss_val = float("nan")
    eval_every = hp.get("eval_every", 5000)
    heartbeat_every = max(1, min(500, eval_every // 2))  # print to console more often than we eval/checkpoint

    t0 = time.time()
    for step in range(start_step + 1, train_steps + 1):
        epsilon = linear_epsilon(step, hp["epsilon_start"], hp["epsilon_end"], hp["epsilon_decay_steps"])
        action = agent.act(vec, mask, epsilon)
        next_obs, reward, term, trunc, info = env.step(action)
        next_vec = flatten_obs(next_obs, env_cfg)
        next_mask = next_obs["action_mask"]
        done = term or trunc

        buffer.add(vec, action, reward, next_vec, next_mask, done)
        vec, mask = next_vec, next_mask
        ep_return += reward

        if len(buffer) >= hp["batch_size"]:
            batch = buffer.sample(hp["batch_size"])
            loss_val = agent.learn(batch)

        if step % heartbeat_every == 0:
            elapsed = time.time() - t0
            rate = (step - start_step) / max(elapsed, 1e-6)
            remaining = (train_steps - step) / max(rate, 1e-6)
            print(f"  step {step}/{train_steps}  ep={episode}  loss={loss_val:.3f}  "
                  f"eps={epsilon:.3f}  {rate:.0f} steps/s  ETA {remaining/60:.1f} min", flush=True)

        eval_cost = float("nan")
        if step % eval_every == 0:
            eval_cost = quick_eval(agent, env_cfg, seed=seed + 999_000)
            print(f"  [eval @ step {step}] cost_per_order={eval_cost:.2f}", flush=True)

        if done or step % eval_every == 0:
            writer.writerow([step, episode, ep_return, loss_val, epsilon, eval_cost])
            f.flush()

        if done:
            episode += 1
            ep_return = 0.0
            obs, _ = env.reset(seed=int(env_rng.integers(0, 2**31 - 1)))
            vec = flatten_obs(obs, env_cfg)
            mask = obs["action_mask"]

        if step % checkpoint_every == 0:
            torch.save({
                "q_net": agent.q_net.state_dict(),
                "target_net": agent.target_net.state_dict() if agent.use_target_network else agent.q_net.state_dict(),
                "opt": agent.opt.state_dict(),
                "learn_steps": agent._learn_steps,
                "step": step,
                "episode": episode,
                "env_rng_state": env_rng.bit_generator.state,
            }, ckpt_path)

    f.close()
    torch.save(agent.q_net.state_dict(), os.path.join(weights_dir, f"{run_name}_seed{seed}.pt"))
    print(f"[{run_name} seed={seed}] reached step {train_steps} (+{time.time()-t0:.1f}s this call) -> {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_steps", type=int, default=None,
                     help="override train_steps from the config (for quick smoke tests)")
    args = ap.parse_args()

    with open(args.config) as fh:
        hp = yaml.safe_load(fh)

    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    env_cfg = Config.from_yaml(os.path.join(repo_root, hp["env_config"]))
    run_name = hp["variant"] + ("_notarget" if not hp.get("target_network", True) else "")
    log_dir = os.path.join(repo_root, "logs")
    weights_dir = os.path.join(repo_root, "weights")

    for seed in hp["seeds"]:
        train_one_seed(env_cfg, hp, seed, run_name, log_dir, weights_dir,
                        train_steps=args.train_steps)


if __name__ == "__main__":
    main()
