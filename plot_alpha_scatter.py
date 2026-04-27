"""Generate alpha_d scatter plot: measured vs MLWD-predicted."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "figure.dpi": 300,
})

data = json.load(open("results/combined-ols-calibration.json"))
combined = data["combined"]
y_true = np.array(combined["y_true"])
y_pred = np.array(combined["y_pred"])

# Per-model data for coloring
model_labels = []
for m, info in [("qwen25_7b", data.get("qwen25_7b", {})),
                ("llama_31_8b", data.get("llama_31_8b", {})),
                ("qwen3_14b", data.get("qwen3_14b", {}))]:
    n = data["n_samples"].get(m, 0)
    model_labels.extend([m] * n)

colors = {"qwen25_7b": "#2196F3", "llama_31_8b": "#FF9800", "qwen3_14b": "#4CAF50"}
labels = {"qwen25_7b": "Qwen2.5-7B", "llama_31_8b": "LLaMA-3.1-8B", "qwen3_14b": "Qwen3-14B"}

fig, ax = plt.subplots(figsize=(4.5, 4.0))

# Diagonal reference
lo = min(y_true.min(), y_pred.min()) - 0.005
hi = max(y_true.max(), y_pred.max()) + 0.005
ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5, zorder=1)

# Scatter by model
for m in ["qwen25_7b", "llama_31_8b", "qwen3_14b"]:
    mask = [l == m for l in model_labels]
    ax.scatter(np.array(y_true)[mask], np.array(y_pred)[mask],
               c=colors[m], label=labels[m], s=36, alpha=0.8,
               edgecolors="white", linewidths=0.4, zorder=2)

# Highlight high-interference region
ax.axvspan(0.06, hi, alpha=0.08, color="red", zorder=0)
ax.text(0.075, hi * 0.85, r"$\alpha_d > 0.06$", fontsize=8, color="red", alpha=0.7)

ax.set_xlabel(r"$\alpha_d$ measured")
ax.set_ylabel(r"$\alpha_d$ predicted (MLWD Ridge)")
ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")

# Annotate metrics
mae = combined["mae"]
r2 = combined["r2"]
ax.text(0.97, 0.03, f"MAE = {mae:.3f}\n$R^2$ = {r2:.2f}",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

fig.tight_layout()
out = Path("figures/alpha_scatter.png")
fig.savefig(out, bbox_inches="tight")
print(f"Saved to {out}")
