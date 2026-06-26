"""Moving-average reward vs episode, 3 seeds, mean+-std (matches the
"reward per episode" style some teams use, alongside plot_curves.py's
cost-per-order-vs-step view). Uses already-logged data, no retraining.
Usage: python plot_reward_curves.py --run_name dyna_q"""
import argparse, csv, glob, os
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def moving_avg(x, w=20):
    if len(x) < w: return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--log_dir", default=None)
    ap.add_argument("--window", type=int, default=20)
    args = ap.parse_args()
    log_dir = args.log_dir or os.path.join(os.path.dirname(__file__), "..", "..", "logs")

    per_seed = {}
    for path in sorted(glob.glob(os.path.join(log_dir, f"{args.run_name}_seed*.csv"))):
        eps, rets = [], []
        with open(path) as f:
            for row in csv.DictReader(f):
                if row["return"] not in ("", "nan"):
                    eps.append(int(row["episode"])); rets.append(float(row["return"]))
        if rets:
            per_seed[path] = moving_avg(np.array(rets), args.window)

    if not per_seed:
        print("no logs found"); return
    n = min(len(v) for v in per_seed.values())
    mat = np.array([v[:n] for v in per_seed.values()])
    mean, std = mat.mean(axis=0), mat.std(axis=0)
    x = np.arange(n)

    plt.figure(figsize=(8, 5))
    for v in per_seed.values():
        plt.plot(v[:n], alpha=0.25, color="tab:purple", linewidth=1)
    plt.plot(x, mean, color="tab:purple", linewidth=2, label=f"{args.run_name} (mean of {len(per_seed)} seeds)")
    plt.fill_between(x, mean - std, mean + std, color="tab:purple", alpha=0.15, label="±1 std")
    plt.xlabel("episode"); plt.ylabel(f"reward (moving avg, window={args.window})")
    plt.title(f"{args.run_name}: reward per episode, multi-seed")
    plt.legend(); plt.tight_layout()
    out = os.path.join(log_dir, f"{args.run_name}_reward_curve.png")
    plt.savefig(out, dpi=130)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
