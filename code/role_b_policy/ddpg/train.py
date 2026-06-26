"""Training entry point for Role B's DDPG (TD3-stabilized), on DroneControl-v0.

DroneControl-v0 has no shipped baseline/evaluate() utility -- `quick_eval`
below is our own metric: success rate and mean return over a fixed set of
eval seeds. There is no "beat greedy_nearest" bar for this sub-env; the
deliverable is a working, stable policy.

Usage:
    python code/role_b_policy/ddpg/train.py --config configs/ddpg.yaml
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

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ""))

from agent import DDPGAgent              # noqa: E402
from replay_buffer import ReplayBuffer   # noqa: E402
from drone_dispatch_env import Config    # noqa: E402


def quick_eval(agent: DDPGAgent, env_cfg: Config, seeds) -> dict:
    env = gym.make("DroneControl-v0", config=env_cfg)
    returns, successes = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done, ep_return = False, 0.0
        while not done:
            a = agent.act(obs, noise_std=0.0)  # deterministic for eval
            obs, r, term, trunc, info = env.step(a)
            ep_return += r
            done = term or trunc
        success = term and not trunc and env.unwrapped.soc > 0.0
        returns.append(ep_return)
        successes.append(float(success))
    return {"mean_return": float(np.mean(returns)), "success_rate": float(np.mean(successes))}


def linear_decay(step, start, end, decay_steps):
    frac = min(1.0, step / max(1, decay_steps))
    return start + frac * (end - start)


def train_one_seed(env_cfg: Config, hp: dict, seed: int, run_name: str,
                    log_dir: str, weights_dir: str, train_steps: int | None = None,
                    checkpoint_every: int = 10000):
    train_steps = train_steps or hp["train_steps"]
    obs_dim = 7
    action_low, action_high = [0.0, -1.0], [1.0, 1.0]

    agent = DDPGAgent(obs_dim, action_low, action_high, hidden_sizes=hp["hidden_sizes"],
                       gamma=hp["gamma"], tau=hp["tau"], actor_lr=hp["actor_lr"],
                       critic_lr=hp["critic_lr"], policy_delay=hp.get("policy_delay", 2),
                       target_noise_std=hp.get("target_noise_std", 0.2),
                       target_noise_clip=hp.get("target_noise_clip", 0.5),
                       device="cpu", seed=seed)
    buffer = ReplayBuffer(hp["buffer_size"], obs_dim, 2, seed=seed)

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}_seed{seed}.csv")
    ckpt_path = os.path.join(weights_dir, f"{run_name}_seed{seed}_ckpt.pt")
    best_path = os.path.join(weights_dir, f"{run_name}_seed{seed}_best.pt")

    start_step, episode = 0, 0
    env_rng = np.random.default_rng(seed)
    best_success_rate, best_mean_return = -1.0, float("-inf")

    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path)
            agent.actor.load_state_dict(ckpt["actor"])
            agent.actor_target.load_state_dict(ckpt["actor_target"])
            agent.critic1.load_state_dict(ckpt["critic1"])
            agent.critic1_target.load_state_dict(ckpt["critic1_target"])
            agent.critic2.load_state_dict(ckpt["critic2"])
            agent.critic2_target.load_state_dict(ckpt["critic2_target"])
            agent.actor_opt.load_state_dict(ckpt["actor_opt"])
            agent.critic_opt.load_state_dict(ckpt["critic_opt"])
            agent._update_count = ckpt.get("update_count", 0)
            start_step, episode = ckpt["step"], ckpt["episode"]
            env_rng.bit_generator.state = ckpt["env_rng_state"]
            best_success_rate = ckpt.get("best_success_rate", -1.0)
            best_mean_return = ckpt.get("best_mean_return", float("-inf"))
            f = open(log_path, "a", newline="")
            print(f"[{run_name} seed={seed}] resumed from step {start_step}/{train_steps}", flush=True)
        except (KeyError, RuntimeError) as e:
            # Stale checkpoint from a different agent version (e.g. a
            # pre-TD3 single-critic checkpoint with "critic" instead of
            # "critic1"/"critic2"). Rather than crash the whole training
            # run over a leftover file, warn loudly and start this seed
            # fresh -- the agent was just constructed above and is already
            # a valid, untouched fresh agent.
            print(f"[{run_name} seed={seed}] WARNING: checkpoint at {ckpt_path} is incompatible "
                  f"(missing key {e}) -- likely from an older agent version. "
                  f"Deleting it and starting this seed fresh.", flush=True)
            os.remove(ckpt_path)
            f = open(log_path, "w", newline="")
            csv.writer(f).writerow(["step", "episode", "return", "actor_loss", "critic_loss",
                                     "noise_std", "eval_mean_return", "eval_success_rate"])
    else:
        f = open(log_path, "w", newline="")
        csv.writer(f).writerow(["step", "episode", "return", "actor_loss", "critic_loss",
                                 "noise_std", "eval_mean_return", "eval_success_rate"])
        print(f"[{run_name} seed={seed}] starting fresh, target={train_steps} steps "
              f"(obs_dim=7, action_dim=2, TD3-stabilized)", flush=True)

    writer = csv.writer(f)
    env = gym.make("DroneControl-v0", config=env_cfg)
    obs, _ = env.reset(seed=int(env_rng.integers(0, 2**31 - 1)))

    ep_return, last_stats = 0.0, {}
    eval_every = hp.get("eval_every", 5000)
    heartbeat_every = max(1, min(500, eval_every // 2))
    warmup_steps = hp.get("warmup_steps", 1000)

    t0 = time.time()
    for step in range(start_step + 1, train_steps + 1):
        noise_std = linear_decay(step, hp["noise_start"], hp["noise_end"], hp["noise_decay_steps"])
        if step <= warmup_steps:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, noise_std=noise_std)
        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc

        buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_return += reward

        if len(buffer) >= hp["batch_size"] and step > warmup_steps:
            stats = agent.learn(buffer.sample(hp["batch_size"]))
            if not np.isnan(stats.get("actor_loss", float("nan"))):
                last_stats = stats
            else:
                last_stats["critic_loss"] = stats["critic_loss"]

        if step % heartbeat_every == 0:
            elapsed = time.time() - t0
            rate = (step - start_step) / max(elapsed, 1e-6)
            remaining = (train_steps - step) / max(rate, 1e-6)
            al = last_stats.get("actor_loss", float("nan"))
            cl = last_stats.get("critic_loss", float("nan"))
            print(f"  step {step}/{train_steps}  ep={episode}  aloss={al:.3f}  closs={cl:.3f}  "
                  f"noise={noise_std:.3f}  {rate:.0f} steps/s  ETA {remaining/60:.1f} min", flush=True)

        eval_metrics = {}
        if step % eval_every == 0:
            eval_metrics = quick_eval(agent, env_cfg, seeds=[90001, 90002, 90003, 90004, 90005])
            print(f"  [eval @ step {step}] mean_return={eval_metrics['mean_return']:.2f}  "
                  f"success_rate={eval_metrics['success_rate']:.2f}", flush=True)
            is_better = (eval_metrics["success_rate"] > best_success_rate or
                         (eval_metrics["success_rate"] == best_success_rate and
                          eval_metrics["mean_return"] > best_mean_return))
            if is_better:
                best_success_rate = eval_metrics["success_rate"]
                best_mean_return = eval_metrics["mean_return"]
                torch.save(agent.actor.state_dict(), best_path)
                print(f"  [new best @ step {step}] success_rate={best_success_rate:.2f} -> saved {best_path}", flush=True)

        if done or step % eval_every == 0:
            writer.writerow([step, episode, ep_return, last_stats.get("actor_loss", ""),
                              last_stats.get("critic_loss", ""), noise_std,
                              eval_metrics.get("mean_return", ""), eval_metrics.get("success_rate", "")])
            f.flush()

        if done:
            episode += 1
            ep_return = 0.0
            obs, _ = env.reset(seed=int(env_rng.integers(0, 2**31 - 1)))

        if step % checkpoint_every == 0:
            torch.save({
                "actor": agent.actor.state_dict(), "actor_target": agent.actor_target.state_dict(),
                "critic1": agent.critic1.state_dict(), "critic1_target": agent.critic1_target.state_dict(),
                "critic2": agent.critic2.state_dict(), "critic2_target": agent.critic2_target.state_dict(),
                "actor_opt": agent.actor_opt.state_dict(), "critic_opt": agent.critic_opt.state_dict(),
                "update_count": agent._update_count,
                "step": step, "episode": episode, "env_rng_state": env_rng.bit_generator.state,
                "best_success_rate": best_success_rate, "best_mean_return": best_mean_return,
            }, ckpt_path)

    f.close()
    torch.save(agent.actor.state_dict(), os.path.join(weights_dir, f"{run_name}_seed{seed}.pt"))
    print(f"[{run_name} seed={seed}] reached step {train_steps} (+{time.time()-t0:.1f}s this call) "
          f"best_success_rate={best_success_rate:.2f} -> {log_path}")


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
