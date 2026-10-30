"""Regenerate docs/RL-incremental-implementation.png.

Shows the incremental sample-average update Q_{k+1} = Q_k + (1/k)(R_k - Q_k)
converging on the true action value q*(a) from noisy rewards.

    python scripts/make_rl_incremental_figure.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "docs" / "RL-incremental-implementation.png"

rng = np.random.default_rng(8)
q_star, steps = 1.0, 300
rewards = rng.normal(q_star, 1.0, steps)

q, estimates = 0.0, []
for k, r in enumerate(rewards, start=1):
    q += (r - q) / k  # incremental sample average
    estimates.append(q)

k = np.arange(1, steps + 1)
fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
ax.scatter(k, rewards, s=8, color="#9aa5b1", alpha=0.6, label="Reward $R_k$")
ax.plot(k, estimates, color="#1f6feb", lw=2.2, label="Estimate $Q_{k+1}$")
ax.axhline(q_star, color="#d1242f", ls="--", lw=1.4, label="True value $q^*(a)$")
ax.set_xlabel("Times action $a$ was selected ($k$)")
ax.set_ylabel("Value")
ax.set_ylim(-2.2, 4.6)
ax.set_xlim(0, steps)
ax.set_title(r"Incremental update: $Q_{k+1} = Q_k + \frac{1}{k}\,(R_k - Q_k)$", fontsize=12)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper right", framealpha=0.95, ncols=3, fontsize=9)
fig.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}")
