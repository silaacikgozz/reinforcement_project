"""Dyna-Q training loop. Reuses Role A's obs encoding + replay buffer.
Usage: python train.py --config ../../../configs/dyna_q.yaml
`planning_steps` in config is the Section-4 ablation knob (n sweep)."""
import argparse, csv, os, sys, time
import numpy as np, torch, yaml, gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "role_a_dqn"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ""))
sys.path.insert(0, os.path.dirname(__file__))
from common.obs_encoding import flatten_obs, flat_dim   # noqa: E402
from replay_buffer import ReplayBuffer                    # noqa: E402  (Role A's)
from dyna_agent import DynaQAgent                              # noqa: E402
from drone_dispatch_env import Config                     # noqa: E402


def quick_eval(agent, env_cfg, seed):
    env = gym.make("DroneDispatch-v0", config=env_cfg)
    obs, _ = env.reset(seed=seed); done = False
    while not done:
        a = agent.greedy_action(flatten_obs(obs, env_cfg), obs["action_mask"])
        obs, r, term, trunc, info = env.step(a); done = term or trunc
    s = env.unwrapped.stats
    return (s["energy"] + s["late_cost"] + s["drop_cost"] + s["depletion_cost"]) / max(s["delivered"], 1)


def train_one_seed(env_cfg, hp, seed, run_name, log_dir, weights_dir, train_steps=None, checkpoint_every=10000):
    train_steps = train_steps or hp["train_steps"]
    obs_dim, n_actions = flat_dim(env_cfg), env_cfg.n_actions
    agent = DynaQAgent(obs_dim, n_actions, planning_steps=hp["planning_steps"],
                        hidden_sizes=hp["hidden_sizes"], gamma=hp["gamma"], lr=hp["lr"],
                        target_update_every=hp["target_update_every"], device="cpu", seed=seed)
    buffer = ReplayBuffer(hp["buffer_size"], obs_dim, n_actions, seed=seed)

    os.makedirs(log_dir, exist_ok=True); os.makedirs(weights_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{run_name}_seed{seed}.csv")
    ckpt_path = os.path.join(weights_dir, f"{run_name}_seed{seed}_ckpt.pt")
    start_step, episode = 0, 0
    env_rng = np.random.default_rng(seed)

    if os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path)
            agent.q_net.load_state_dict(ckpt["q_net"]); agent.target_net.load_state_dict(ckpt["target_net"])
            agent.opt.load_state_dict(ckpt["opt"]); agent.model.load_state_dict(ckpt["model"])
            agent.model_opt.load_state_dict(ckpt["model_opt"])
            start_step, episode = ckpt["step"], ckpt["episode"]
            env_rng.bit_generator.state = ckpt["env_rng_state"]
            f = open(log_path, "a", newline="")
            print(f"[{run_name} seed={seed}] resumed from step {start_step}/{train_steps}", flush=True)
        except (KeyError, RuntimeError) as e:
            print(f"[{run_name} seed={seed}] WARNING: incompatible checkpoint ({e}), starting fresh", flush=True)
            os.remove(ckpt_path)
            f = open(log_path, "w", newline="")
            csv.writer(f).writerow(["step", "episode", "return", "loss", "model_loss", "eval_cost_per_order"])
    else:
        f = open(log_path, "w", newline="")
        csv.writer(f).writerow(["step", "episode", "return", "loss", "model_loss", "eval_cost_per_order"])
        print(f"[{run_name} seed={seed}] starting fresh, target={train_steps} (planning_steps={hp['planning_steps']})", flush=True)

    writer = csv.writer(f)
    env = gym.make("DroneDispatch-v0", config=env_cfg)
    obs, _ = env.reset(seed=int(env_rng.integers(0, 2**31 - 1)))
    vec, mask = flatten_obs(obs, env_cfg), obs["action_mask"]
    ep_return, loss_val, model_loss_val = 0.0, float("nan"), float("nan")
    eval_every = hp.get("eval_every", 10000)
    heartbeat_every = max(1, min(500, eval_every // 2))

    t0 = time.time()
    for step in range(start_step + 1, train_steps + 1):
        eps = max(hp["epsilon_end"], hp["epsilon_start"] - step / hp["epsilon_decay_steps"] * (hp["epsilon_start"] - hp["epsilon_end"]))
        action = agent.act(vec, mask, eps)
        next_obs, reward, term, trunc, info = env.step(action)
        next_vec, next_mask = flatten_obs(next_obs, env_cfg), next_obs["action_mask"]
        done = term or trunc
        buffer.add(vec, action, reward, next_vec, next_mask, done)
        vec, mask = next_vec, next_mask
        ep_return += reward

        if len(buffer) >= hp["batch_size"]:
            loss_val = agent.learn(buffer.sample(hp["batch_size"]))          # real-experience update
            model_loss_val = agent.train_model(buffer.sample(hp["batch_size"]))
            if agent.planning_steps > 0:
                agent.plan(buffer, next_mask)                                  # simulated updates

        if step % heartbeat_every == 0:
            rate = step / max(time.time() - t0, 1e-6)
            print(f"  step {step}/{train_steps}  ep={episode}  loss={loss_val:.3f}  mloss={model_loss_val:.3f}  "
                  f"eps={eps:.3f}  {rate:.0f} steps/s  ETA {(train_steps-step)/max(rate,1e-6)/60:.1f} min", flush=True)

        eval_cost = float("nan")
        if step % eval_every == 0:
            eval_cost = quick_eval(agent, env_cfg, seed + 999_000)
            print(f"  [eval @ step {step}] cost_per_order={eval_cost:.2f}", flush=True)
        if done or step % eval_every == 0:
            writer.writerow([step, episode, ep_return, loss_val, model_loss_val, eval_cost]); f.flush()
        if done:
            episode += 1; ep_return = 0.0
            obs, _ = env.reset(seed=int(env_rng.integers(0, 2**31 - 1)))
            vec, mask = flatten_obs(obs, env_cfg), obs["action_mask"]
        if step % checkpoint_every == 0:
            torch.save({"q_net": agent.q_net.state_dict(), "target_net": agent.target_net.state_dict(),
                        "opt": agent.opt.state_dict(), "model": agent.model.state_dict(),
                        "model_opt": agent.model_opt.state_dict(), "step": step, "episode": episode,
                        "env_rng_state": env_rng.bit_generator.state}, ckpt_path)

    f.close()
    torch.save(agent.q_net.state_dict(), os.path.join(weights_dir, f"{run_name}_seed{seed}.pt"))
    print(f"[{run_name} seed={seed}] reached step {train_steps} (+{time.time()-t0:.1f}s) -> {log_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--train_steps", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as fh: hp = yaml.safe_load(fh)
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    env_cfg = Config.from_yaml(os.path.join(repo_root, hp["env_config"]))
    run_name = hp["variant"]
    for seed in hp["seeds"]:
        train_one_seed(env_cfg, hp, seed, run_name, os.path.join(repo_root, "logs"),
                        os.path.join(repo_root, "weights"), train_steps=args.train_steps)


if __name__ == "__main__":
    main()
