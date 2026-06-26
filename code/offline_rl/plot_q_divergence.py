"""Plots mean_q over training for naive vs CQL, side by side -- the single
clearest visual of the failure-and-fix story.
Usage: python plot_q_divergence.py"""
import csv, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

log_dir = os.path.join(os.path.dirname(__file__), "..", "..", "logs")

def read(name):
    steps, mq = [], []
    with open(os.path.join(log_dir, f"{name}.csv")) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"])); mq.append(float(row["mean_q"]))
    return steps, mq

s1, q1 = read("offline_naive")
s2, q2 = read("offline_cql")
plt.figure(figsize=(8, 5))
plt.plot(s1, q1, color="tab:red", linewidth=2, label="naive offline DQN")
plt.plot(s2, q2, color="tab:green", linewidth=2, label="CQL")
plt.xlabel("training step"); plt.ylabel("mean Q-value (eval batch)")
plt.title("Offline RL: Q-value divergence, naive vs CQL")
plt.legend(); plt.tight_layout()
out = os.path.join(log_dir, "offline_q_divergence.png")
plt.savefig(out, dpi=130)
print(f"saved -> {out}")
