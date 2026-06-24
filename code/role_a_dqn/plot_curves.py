"""Generates the learning-curve figures required by Section 4 (item 3):
mean ± std over the 3 seeds, plus a moving-average trend, with the
greedy_nearest / milp_rolling baselines overlaid as reference lines --
same spirit as the report screenshots, but built from our own real CSV logs
in logs/, not redrawn by hand.

Usage:
    python code/role_a_dqn/plot_curves.py --run_names dqn,double_dqn,dueling_dqn

Reads logs/<run_name>_seed<k>.csv (written by train.py) and
logs/<run_name>_comparison.csv (written by evaluate_agent.py, for the
baseline reference lines if present).

Saves PNGs to logs/<run_name>_curve.png and one overlay logs/all_methods_overlay.png.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _moving_avg(x: np.ndarray, window: int) -> np.ndarray:
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="valid")


def load_run(log_dir: str, run_name: str):
    """Returns dict seed -> (steps, eval_costs) using only the eval checkpoint
    rows (the cheap quick_eval column), since that's the directly comparable
    signal across seeds (training return is noisy/uncomparable raw)."""
    per_seed = {}
    for path in sorted(glob.glob(os.path.join(log_dir, f"{run_name}_seed*.csv"))):
        seed = int(path.split("seed")[-1].split(".")[0].split("_")[0])
        steps, costs = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["eval_cost_per_order"] not in ("nan", ""):
                    steps.append(int(row["step"]))
                    costs.append(float(row["eval_cost_per_order"]))
        if steps:
            per_seed[seed] = (np.array(steps), np.array(costs))
    return per_seed


def load_baseline_refs(log_dir: str, run_name: str):
    path = os.path.join(log_dir, f"{run_name}_comparison.csv")
    refs = {}
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["method"] in ("greedy_nearest", "random", "milp_rolling"):
                    refs[row["method"]] = float(row["cost_per_order_mean"])
    return refs


def plot_one(run_name: str, log_dir: str, out_dir: str):
    per_seed = load_run(log_dir, run_name)
    if not per_seed:
        print(f"[skip] no logs found for {run_name}")
        return
    refs = load_baseline_refs(log_dir, run_name)

    # align onto the common step grid (all seeds use the same eval_every)
    common_steps = sorted(set.intersection(*[set(s.tolist()) for s, _ in per_seed.values()]))
    if not common_steps:
        print(f"[skip] seeds for {run_name} have no overlapping eval steps yet")
        return
    matrix = np.array([[dict(zip(s, c))[step] for step in common_steps]
                        for s, c in per_seed.values()])  # (n_seeds, n_points)
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)

    plt.figure(figsize=(8, 5))
    for s, (steps, costs) in per_seed.items():
        plt.plot(steps, costs, alpha=0.25, linewidth=1, color="tab:red")
    plt.plot(common_steps, mean, color="tab:red", linewidth=2, label=f"{run_name} (mean of {len(per_seed)} seeds)")
    plt.fill_between(common_steps, mean - std, mean + std, color="tab:red", alpha=0.15, label="±1 std")

    colors = {"greedy_nearest": "tab:green", "milp_rolling": "tab:blue", "random": "gray"}
    for name, val in refs.items():
        plt.axhline(val, color=colors.get(name, "black"), linestyle="--", linewidth=1.5, label=name)

    plt.xlabel("training step")
    plt.ylabel("cost per order (eval, lower is better)")
    plt.title(f"{run_name}: cost/order over training vs baselines")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{run_name}_curve.png")
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"saved -> {out_path}")


def plot_overlay(run_names: list[str], log_dir: str, out_dir: str):
    plt.figure(figsize=(8, 5))
    color_cycle = ["tab:red", "tab:orange", "tab:purple", "tab:brown"]
    refs = {}
    for i, run_name in enumerate(run_names):
        per_seed = load_run(log_dir, run_name)
        if not per_seed:
            continue
        common_steps = sorted(set.intersection(*[set(s.tolist()) for s, _ in per_seed.values()]))
        if not common_steps:
            continue
        matrix = np.array([[dict(zip(s, c))[step] for step in common_steps]
                            for s, c in per_seed.values()])
        mean = matrix.mean(axis=0)
        plt.plot(common_steps, mean, color=color_cycle[i % len(color_cycle)], linewidth=2, label=run_name)
        refs.update(load_baseline_refs(log_dir, run_name))

    colors = {"greedy_nearest": "tab:green", "milp_rolling": "tab:blue", "random": "gray"}
    for name, val in refs.items():
        plt.axhline(val, color=colors.get(name, "black"), linestyle="--", linewidth=1.5, label=name)

    plt.xlabel("training step")
    plt.ylabel("cost per order (eval, lower is better)")
    plt.title("DQN family vs baselines")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, "all_methods_overlay.png")
    plt.savefig(out_path, dpi=130)
    plt.close()
    print(f"saved -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_names", default="dqn,double_dqn,dueling_dqn")
    ap.add_argument("--log_dir", default=None)
    args = ap.parse_args()

    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    log_dir = args.log_dir or os.path.join(repo_root, "logs")
    run_names = args.run_names.split(",")

    for run_name in run_names:
        plot_one(run_name, log_dir, log_dir)
    plot_overlay(run_names, log_dir, log_dir)


if __name__ == "__main__":
    main()
