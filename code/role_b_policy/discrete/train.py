"""Training entry point for Role B's discrete methods. Single command:

    python code/role_b_policy/discrete/train.py --config configs/reinforce_gae.yaml
    python code/role_b_policy/discrete/train.py --config configs/a2c.yaml

`variant` selects the rollout strategy (same network/update code otherwise):
  reinforce_gae -> rollout_steps: null   (update once per finished episode)
  a2c           -> rollout_steps: <int>  (update every N steps, bootstrapped)
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

sys.path.insert(0, os.path.dirname(__file__))                              # discrete/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "")) # code/

from common.obs_encoding import flatten_obs, flat_dim   # noqa: E402
from agent import PGAgent                                # noqa: E402
from drone_dispatch_env import Config                    # noqa: E402


def quick_eval(agent: PGAgent, env_cfg: Config, seed: int) -> float:
    env = gym.make("DroneDispatch-v0", config=env_cfg)
    obs, _ = env.reset(seed=seed)
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
    rollout_steps = hp.get("rollout_steps")  # None -> REINFORCE (full episode)

    obs_dim = flat_dim(env_cfg)
    n_actions = env_cfg.n_actions
    agent = PGAgent(
        obs_dim=obs_dim, n_actions=n_actions, hidden_sizes=hp["hidden_sizes"],
        gamma=hp["gamma"], lam=hp["gae_lambda"], lr=hp["lr"],
        value_coef=hp.get("value_coef", 0.5), entropy_coef=hp.get("entropy_coef", 0.01),
        normalize_advantages=hp.get("normalize_advantages", True),
        device="cpu", seed=seed,
    )

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}_seed{seed}.csv")
    ckpt_path = os.path.join(weights_dir, f"{run_name}_seed{seed}_ckpt.pt")

    start_step, episode = 0, 0
    env_rng = np.random.default_rng(seed)

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        agent.net.load_state_dict(ckpt["net"])
        agent.opt.load_state_dict(ckpt["opt"])
        start_step = ckpt["step"]
        episode = ckpt["episode"]
        env_rng.bit_generator.state = ckpt["env_rng_state"]
        f = open(log_path, "a", newline="")
        print(f"[{run_name} seed={seed}] resumed from step {start_step}/{train_steps}", flush=True)
    else:
        f = open(log_path, "w", newline="")
        csv.writer(f).writerow(["step", "episode", "return", "policy_loss", "value_loss", "entropy", "eval_cost_per_order"])
        print(f"[{run_name} seed={seed}] starting fresh, target={train_steps} steps "
              f"(obs_dim={obs_dim}, n_actions={n_actions}, rollout_steps={rollout_steps})", flush=True)

    writer = csv.writer(f)
    env = gym.make("DroneDispatch-v0", config=env_cfg)
    obs, _ = env.reset(seed=int(env_rng.integers(0, 2**31 - 1)))
    vec = flatten_obs(obs, env_cfg)
    mask = obs["action_mask"]

    ep_return = 0.0
    last_stats = {}
    eval_every = hp.get("eval_every", 10000)
    heartbeat_every = max(1, min(500, eval_every // 2))
    steps_since_rollout = 0

    t0 = time.time()
    for step in range(start_step + 1, train_steps + 1):
        action, logprob, value = agent.act(vec, mask)
        next_obs, reward, term, trunc, info = env.step(action)
        next_vec = flatten_obs(next_obs, env_cfg)
        next_mask = next_obs["action_mask"]
        done = term or trunc

        agent.store(vec, mask, action, logprob, value, reward, done)
        ep_return += reward
        vec, mask = next_vec, next_mask
        steps_since_rollout += 1

        do_update = done or (rollout_steps is not None and steps_since_rollout >= rollout_steps)
        if do_update:
            last_stats = agent.finish_rollout(vec, mask, done) or last_stats
            steps_since_rollout = 0

        if step % heartbeat_every == 0:
            elapsed = time.time() - t0
            rate = (step - start_step) / max(elapsed, 1e-6)
            remaining = (train_steps - step) / max(rate, 1e-6)
            pl = last_stats.get("policy_loss", float("nan"))
            vl = last_stats.get("value_loss", float("nan"))
            ent = last_stats.get("entropy", float("nan"))
            print(f"  step {step}/{train_steps}  ep={episode}  ploss={pl:.3f}  vloss={vl:.3f}  "
                  f"ent={ent:.3f}  {rate:.0f} steps/s  ETA {remaining/60:.1f} min", flush=True)

        eval_cost = float("nan")
        if step % eval_every == 0:
            eval_cost = quick_eval(agent, env_cfg, seed=seed + 999_000)
            print(f"  [eval @ step {step}] cost_per_order={eval_cost:.2f}", flush=True)

        if done or step % eval_every == 0:
            writer.writerow([step, episode, ep_return, last_stats.get("policy_loss", ""),
                              last_stats.get("value_loss", ""), last_stats.get("entropy", ""), eval_cost])
            f.flush()

        if done:
            episode += 1
            ep_return = 0.0
            obs, _ = env.reset(seed=int(env_rng.integers(0, 2**31 - 1)))
            vec = flatten_obs(obs, env_cfg)
            mask = obs["action_mask"]

        if step % checkpoint_every == 0:
            torch.save({
                "net": agent.net.state_dict(), "opt": agent.opt.state_dict(),
                "step": step, "episode": episode, "env_rng_state": env_rng.bit_generator.state,
            }, ckpt_path)

    f.close()
    torch.save(agent.net.state_dict(), os.path.join(weights_dir, f"{run_name}_seed{seed}.pt"))
    print(f"[{run_name} seed={seed}] reached step {train_steps} (+{time.time()-t0:.1f}s this call) -> {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_steps", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as fh:
        hp = yaml.safe_load(fh)

    repo_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    env_cfg = Config.from_yaml(os.path.join(repo_root, hp["env_config"]))
    run_name = hp["variant"]
    log_dir = os.path.join(repo_root, "logs")
    weights_dir = os.path.join(repo_root, "weights")

    for seed in hp["seeds"]:
        train_one_seed(env_cfg, hp, seed, run_name, log_dir, weights_dir, train_steps=args.train_steps)


if __name__ == "__main__":
    main()
