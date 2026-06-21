"""Generate the README figures from assets/figdata/figdata.json.

Fig1  results.{png,svg}   — EX / finished / valid-SQL, base vs RL-R1 vs RL-R2
Fig2  training.{png,svg}  — (a) reward learning curve  (b) gradient-signal density
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- mandatory editable-text rcParams (nature-figure skill) ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams.update({
    "font.size": 12,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 1.0,
    "axes.edgecolor": "#4D4D4D",
    "legend.frameon": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "figdata" / "figdata.json").read_text())

# unified palette: base = neutral, R1 = blue, R2 = green (the winner / gain)
C_BASE, C_R1, C_R2 = "#9A9A9A", "#3775BA", "#2E9E44"
GRID = "#E6E6E6"


def rolling(y, w=12):
    y = np.asarray(y, float)
    k = np.convolve(y, np.ones(w) / w, mode="valid")
    return np.arange(w - 1, len(y)) + 1, k


# ============================ Fig 1 — results ============================
ev = data["evals"]
cats = ["Execution\naccuracy", "Finished\nrate", "Valid-SQL\nrate"]
keys = ["ex", "finished", "valid_sql"]
base = [ev["base"][k] * 100 for k in keys]
r1 = [ev["r1"][k] * 100 for k in keys]
r2 = [ev["r2"][k] * 100 for k in keys]

fig, ax = plt.subplots(figsize=(7.4, 4.4))
x = np.arange(len(cats))
w = 0.26
b0 = ax.bar(x - w, base, w, label="Base · Qwen3.5-9B", color=C_BASE, edgecolor="white", linewidth=1.2)
b1 = ax.bar(x, r1, w, label="RL · R1 (execution-only)", color=C_R1, edgecolor="white", linewidth=1.2)
b2 = ax.bar(x + w, r2, w, label="RL · R2 (partial credit)", color=C_R2, edgecolor="white", linewidth=1.2)

for bars in (b0, b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.1,
                f"{bar.get_height():.1f}", ha="center", va="bottom",
                fontsize=10, color="#333333")

# hero delta on the Execution-accuracy group (base -> R2)
ax.text(x[0] + w, r2[0] + 6.0, "+7.2 pt", ha="center", va="bottom",
        fontsize=12, fontweight="bold", color=C_R2)

ax.set_ylim(0, 100)
ax.set_ylabel("Percent (%)")
ax.set_xticks(x)
ax.set_xticklabels(cats)
ax.yaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9.5)
ax.set_title("Reinforcement learning improves agentic Text-to-SQL\nfull BIRD-dev · n = 1534 · greedy decoding",
             fontsize=12.5, fontweight="bold", pad=10)
fig.tight_layout()
fig.savefig(HERE / "results.png", dpi=200, bbox_inches="tight")
fig.savefig(HERE / "results.svg", bbox_inches="tight")
plt.close(fig)

# ====================== Fig 2 — training dynamics ======================
r1s = np.array(data["r1_v2"], float)   # [reward, frac_zero]
r2s = np.array(data["r2_v2"], float)
steps1 = np.arange(1, len(r1s) + 1)
steps2 = np.arange(1, len(r2s) + 1)

fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 4.2))

# (a) reward learning curve — rolling mean ± std, normalized to each arm's max
n1 = r1s[:, 0] / 1.0
n2 = r2s[:, 0] / 7.0
for n, c, lab in [(n1, C_R1, "R1 (execution-only)"), (n2, C_R2, "R2 (partial credit)")]:
    xs, m = rolling(n)
    axa.plot(xs, m, color=c, lw=2.6, label=lab)
axa.set_xlabel("GRPO step")
axa.set_ylabel("Mean reward (norm. to arm max)")
axa.set_xlim(0, 150)
axa.set_ylim(bottom=0)
axa.yaxis.grid(True, color=GRID, linewidth=0.8)
axa.set_axisbelow(True)
axa.legend(loc="upper left", fontsize=9.5)
axa.set_title("a   Reward rises during training", loc="left", fontsize=12.5, fontweight="bold")

# (b) gradient-signal density = 1 - frac_reward_zero_std (rolling mean)
g1 = 1.0 - r1s[:, 1]
g2 = 1.0 - r2s[:, 1]
xs, m = rolling(g1); axb.plot(xs, m, color=C_R1, lw=2.6, label="R1 (execution-only)")
xs, m = rolling(g2); axb.plot(xs, m, color=C_R2, lw=2.6, label="R2 (partial credit)")
axb.set_xlabel("GRPO step")
axb.set_ylabel("Frac. groups with gradient\n(1 − zero-reward-std)")
axb.set_xlim(0, 150)
axb.set_ylim(bottom=0)
axb.yaxis.grid(True, color=GRID, linewidth=0.8)
axb.set_axisbelow(True)
axb.legend(loc="upper left", fontsize=9.5)
axb.set_title("b   Partial credit gives a denser gradient", loc="left", fontsize=12.5, fontweight="bold")

fig.tight_layout(pad=1.5)
fig.savefig(HERE / "training.png", dpi=200, bbox_inches="tight")
fig.savefig(HERE / "training.svg", bbox_inches="tight")
plt.close(fig)

# quick numeric receipt
print("Fig1 EX %:", [round(v, 1) for v in (base[0], r1[0], r2[0])])
print("Fig2 grad-density mean (last 50 steps) R1 vs R2:",
      round(float(g1[-50:].mean()), 3), round(float(g2[-50:].mean()), 3))
print("saved: results.png/svg, training.png/svg")
