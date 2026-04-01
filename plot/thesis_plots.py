"""Generate all Chapter 6 thesis figures.

Each ``plot_*`` function produces one figure and saves it to *output_path*.
Call :func:`generate_all` to produce every figure in one shot.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from plot.style import setup_thesis_style, COLORS, MARKERS, LABELS, SYSTEM_ORDER
from analysis.metrics import compute_metrics


def _ordered_systems(available: dict) -> list:
    """Return system keys in canonical order, filtering to those present."""
    return [s for s in SYSTEM_ORDER if s in available]


# ------------------------------------------------------------------
# Fig 6-x: Goodput vs request arrival rate
# ------------------------------------------------------------------

def plot_goodput_vs_rate(
    results_dir: str,
    model: str = "qwen2.5-7b",
    output_path: str = "fig_goodput_vs_rate.pdf",
    rates: Optional[List[int]] = None,
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> None:
    """Goodput vs request arrival rate for all systems."""
    if rates is None:
        rates = [2, 4, 6, 8, 10]

    rdir = Path(results_dir)
    # Discover systems
    systems = [
        d.name for d in rdir.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ]
    systems = [s for s in SYSTEM_ORDER if s in systems]

    fig, ax = plt.subplots()
    for sys_name in systems:
        goodputs = []
        valid_rates = []
        for rate in rates:
            fpath = rdir / sys_name / f"rate_{rate}.json"
            if not fpath.exists():
                continue
            m = compute_metrics(
                str(fpath), slo_ttft=slo_ttft,
                slo_tpot=slo_tpot, warmup_s=warmup_s,
            )
            goodputs.append(m.goodput)
            valid_rates.append(rate)
        if valid_rates:
            ax.plot(
                valid_rates, goodputs,
                marker=MARKERS.get(sys_name, "o"),
                color=COLORS.get(sys_name, "gray"),
                label=LABELS.get(sys_name, sys_name),
                linewidth=1.5, markersize=6,
            )

    ax.set_xlabel("Request arrival rate (req/s)")
    ax.set_ylabel("Goodput (req/s)")
    ax.set_title(f"Goodput vs Arrival Rate ({model})")
    ax.legend(loc="best")
    fig.savefig(output_path)
    plt.close(fig)


# ------------------------------------------------------------------
# Fig 6-x: SLO attainment bar chart
# ------------------------------------------------------------------

def plot_slo_attainment(
    results_dir: str,
    model: str = "qwen2.5-7b",
    output_path: str = "fig_slo_attainment.pdf",
    rate: int = 4,
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> None:
    """SLO attainment bar chart comparing all systems at a fixed rate."""
    rdir = Path(results_dir)
    names, attainments = [], []
    for sys_name in SYSTEM_ORDER:
        fpath = rdir / sys_name / f"rate_{rate}.json"
        if not fpath.exists():
            continue
        m = compute_metrics(
            str(fpath), slo_ttft=slo_ttft,
            slo_tpot=slo_tpot, warmup_s=warmup_s,
        )
        names.append(sys_name)
        attainments.append(m.slo_attainment)

    if not names:
        return

    fig, ax = plt.subplots()
    x = np.arange(len(names))
    colors = [COLORS.get(n, "gray") for n in names]
    bars = ax.bar(x, attainments, color=colors, width=0.6, edgecolor="white")

    # Value labels on bars
    for bar, val in zip(bars, attainments):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS.get(n, n) for n in names], rotation=15, ha="right")
    ax.set_ylabel("SLO Attainment (%)")
    ax.set_title(f"SLO Attainment at {rate} req/s ({model})")
    ax.set_ylim(0, 105)
    fig.savefig(output_path)
    plt.close(fig)


# ------------------------------------------------------------------
# Fig 6-x: Tail latency comparison (P99 TTFT & P99 TPOT)
# ------------------------------------------------------------------

def plot_tail_latency(
    results_dir: str,
    model: str = "qwen2.5-7b",
    output_path: str = "fig_tail_latency.pdf",
    rate: int = 4,
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> None:
    """Grouped bar chart of P99 TTFT and P99 TPOT for all systems."""
    rdir = Path(results_dir)
    names, p99_ttfts, p99_tpots = [], [], []
    for sys_name in SYSTEM_ORDER:
        fpath = rdir / sys_name / f"rate_{rate}.json"
        if not fpath.exists():
            continue
        m = compute_metrics(
            str(fpath), slo_ttft=slo_ttft,
            slo_tpot=slo_tpot, warmup_s=warmup_s,
        )
        names.append(sys_name)
        p99_ttfts.append(m.p99_ttft_ms)
        p99_tpots.append(m.p99_tpot_ms)

    if not names:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    x = np.arange(len(names))
    w = 0.5

    # P99 TTFT
    colors = [COLORS.get(n, "gray") for n in names]
    ax1.bar(x, p99_ttfts, width=w, color=colors, edgecolor="white")
    ax1.axhline(y=slo_ttft, color="red", linestyle="--", linewidth=0.8, label=f"SLO={slo_ttft:.0f}ms")
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS.get(n, n) for n in names], rotation=15, ha="right")
    ax1.set_ylabel("P99 TTFT (ms)")
    ax1.set_title("P99 TTFT")
    ax1.legend(fontsize=8)

    # P99 TPOT
    ax2.bar(x, p99_tpots, width=w, color=colors, edgecolor="white")
    ax2.axhline(y=slo_tpot, color="red", linestyle="--", linewidth=0.8, label=f"SLO={slo_tpot:.0f}ms")
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS.get(n, n) for n in names], rotation=15, ha="right")
    ax2.set_ylabel("P99 TPOT (ms)")
    ax2.set_title("P99 TPOT")
    ax2.legend(fontsize=8)

    fig.suptitle(f"Tail Latency at {rate} req/s ({model})", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ------------------------------------------------------------------
# Fig 6-x: Interference prediction accuracy scatter
# ------------------------------------------------------------------

def plot_interference_accuracy(
    mlwd_results_path: str,
    output_path: str = "fig_interference_accuracy.pdf",
) -> None:
    """Scatter plot: MLWD vs SM-Only vs Profile-MLP predicted slowdown.

    Expected JSON format::

        {
          "ground_truth": [1.0, 1.12, ...],
          "mlwd": [1.01, 1.10, ...],
          "sm_only": [0.98, 1.05, ...],
          "profile_mlp": [1.02, 1.15, ...]
        }
    """
    with open(mlwd_results_path) as f:
        data = json.load(f)

    gt = np.array(data["ground_truth"])
    methods = {
        "MLWD (ours)": ("mlwd", "#9467bd", "o"),
        "SM-Only": ("sm_only", "#1f77b4", "s"),
        "Profile-MLP": ("profile_mlp", "#ff7f0e", "^"),
    }

    fig, ax = plt.subplots(figsize=(5, 5))
    # Perfect prediction line
    lo, hi = gt.min() * 0.95, gt.max() * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="Perfect")

    for label, (key, color, marker) in methods.items():
        pred = np.array(data.get(key, []))
        if len(pred) == 0:
            continue
        ax.scatter(
            gt, pred, c=color, marker=marker, s=20,
            alpha=0.6, label=label, edgecolors="none",
        )

    ax.set_xlabel("Ground-truth slowdown")
    ax.set_ylabel("Predicted slowdown")
    ax.set_title("Interference Prediction Accuracy")
    ax.legend(loc="upper left")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(output_path)
    plt.close(fig)


# ------------------------------------------------------------------
# Fig 6-x: Ablation study bar chart
# ------------------------------------------------------------------

def plot_ablation(
    results_dir: str,
    output_path: str = "fig_ablation.pdf",
    rate: int = 4,
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> None:
    """Grouped bar chart showing goodput and SLO attainment for ablation variants."""
    rdir = Path(results_dir)
    ablation_systems = ["cbs_full", "cbs_nomig", "cbs_norole"]
    names, goodputs, attainments = [], [], []

    for sys_name in ablation_systems:
        fpath = rdir / sys_name / f"rate_{rate}.json"
        if not fpath.exists():
            continue
        m = compute_metrics(
            str(fpath), slo_ttft=slo_ttft,
            slo_tpot=slo_tpot, warmup_s=warmup_s,
        )
        names.append(sys_name)
        goodputs.append(m.goodput)
        attainments.append(m.slo_attainment)

    if not names:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    x = np.arange(len(names))
    colors = [COLORS.get(n, "gray") for n in names]
    w = 0.5

    # Goodput
    bars1 = ax1.bar(x, goodputs, width=w, color=colors, edgecolor="white")
    for bar, val in zip(bars1, goodputs):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
            f"{val:.2f}", ha="center", va="bottom", fontsize=8,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels([LABELS.get(n, n) for n in names])
    ax1.set_ylabel("Goodput (req/s)")
    ax1.set_title("Goodput")

    # SLO Attainment
    bars2 = ax2.bar(x, attainments, width=w, color=colors, edgecolor="white")
    for bar, val in zip(bars2, attainments):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=8,
        )
    ax2.set_xticks(x)
    ax2.set_xticklabels([LABELS.get(n, n) for n in names])
    ax2.set_ylabel("SLO Attainment (%)")
    ax2.set_title("SLO Attainment")
    ax2.set_ylim(0, 105)

    fig.suptitle("Ablation Study", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ------------------------------------------------------------------
# Fig 6-x: Lambda sensitivity curve
# ------------------------------------------------------------------

def plot_lambda_sensitivity(
    results_dir: str,
    output_path: str = "fig_lambda_sensitivity.pdf",
    lambda_values: Optional[List[float]] = None,
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> None:
    """Dual-axis plot: goodput and SLO attainment vs lambda_ext."""
    from analysis.sensitivity import lambda_sensitivity

    sweep = lambda_sensitivity(
        results_dir,
        lambda_values=lambda_values,
        slo_ttft=slo_ttft,
        slo_tpot=slo_tpot,
        warmup_s=warmup_s,
    )
    if not sweep["values"]:
        return

    vals = sweep["values"]
    fig, ax1 = plt.subplots()
    color_gp = COLORS["cbs_full"]
    color_slo = COLORS["disagg_static"]

    ax1.plot(
        vals, sweep["goodput"],
        marker="o", color=color_gp, linewidth=1.5,
        markersize=6, label="Goodput",
    )
    ax1.set_xlabel(r"$\lambda_{\mathrm{ext}}$")
    ax1.set_ylabel("Goodput (req/s)", color=color_gp)
    ax1.tick_params(axis="y", labelcolor=color_gp)

    ax2 = ax1.twinx()
    ax2.plot(
        vals, sweep["slo_attainment"],
        marker="s", color=color_slo, linewidth=1.5,
        markersize=6, linestyle="--", label="SLO Attainment",
    )
    ax2.set_ylabel("SLO Attainment (%)", color=color_slo)
    ax2.tick_params(axis="y", labelcolor=color_slo)
    ax2.set_ylim(0, 105)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.set_title(r"Sensitivity to $\lambda_{\mathrm{ext}}$")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ------------------------------------------------------------------
# Fig 6-x: Real vs simulation consistency validation
# ------------------------------------------------------------------

def plot_real_vs_sim(
    real_dir: str,
    sim_dir: str,
    output_path: str = "fig_real_vs_sim.pdf",
    rates: Optional[List[int]] = None,
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> None:
    """Scatter + line plot comparing real cluster vs simulation results.

    Plots goodput and P99 TTFT side by side for each arrival rate.
    """
    if rates is None:
        rates = [2, 4, 6, 8]

    real_gp, sim_gp = [], []
    real_ttft, sim_ttft = [], []
    valid_rates = []

    for rate in rates:
        rpath = Path(real_dir) / f"rate_{rate}.json"
        spath = Path(sim_dir) / f"rate_{rate}.json"
        if not rpath.exists() or not spath.exists():
            continue
        mr = compute_metrics(str(rpath), slo_ttft=slo_ttft, slo_tpot=slo_tpot, warmup_s=warmup_s)
        ms = compute_metrics(str(spath), slo_ttft=slo_ttft, slo_tpot=slo_tpot, warmup_s=warmup_s)
        valid_rates.append(rate)
        real_gp.append(mr.goodput)
        sim_gp.append(ms.goodput)
        real_ttft.append(mr.p99_ttft_ms)
        sim_ttft.append(ms.p99_ttft_ms)

    if not valid_rates:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Goodput comparison
    ax1.plot(valid_rates, real_gp, "o-", color=COLORS["cbs_full"], label="Real", linewidth=1.5)
    ax1.plot(valid_rates, sim_gp, "s--", color=COLORS["disagg_static"], label="Simulation", linewidth=1.5)
    ax1.set_xlabel("Request arrival rate (req/s)")
    ax1.set_ylabel("Goodput (req/s)")
    ax1.set_title("Goodput")
    ax1.legend()

    # P99 TTFT comparison
    ax2.plot(valid_rates, real_ttft, "o-", color=COLORS["cbs_full"], label="Real", linewidth=1.5)
    ax2.plot(valid_rates, sim_ttft, "s--", color=COLORS["disagg_static"], label="Simulation", linewidth=1.5)
    ax2.set_xlabel("Request arrival rate (req/s)")
    ax2.set_ylabel("P99 TTFT (ms)")
    ax2.set_title("P99 TTFT")
    ax2.legend()

    fig.suptitle("Real Cluster vs Simulation", y=1.02)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ------------------------------------------------------------------
# Generate all figures
# ------------------------------------------------------------------

def generate_all(
    results_dir: str,
    output_dir: str,
    model: str = "qwen2.5-7b",
    mlwd_results_path: Optional[str] = None,
    real_dir: Optional[str] = None,
    sim_dir: Optional[str] = None,
) -> None:
    """Produce every Chapter 6 figure and save to *output_dir*."""
    setup_thesis_style()
    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir)

    plot_goodput_vs_rate(results_dir, model, str(out / "fig_goodput_vs_rate.pdf"))
    plot_slo_attainment(results_dir, model, str(out / "fig_slo_attainment.pdf"))
    plot_tail_latency(results_dir, model, str(out / "fig_tail_latency.pdf"))
    plot_ablation(results_dir, str(out / "fig_ablation.pdf"))
    plot_lambda_sensitivity(results_dir, str(out / "fig_lambda_sensitivity.pdf"))

    if mlwd_results_path and Path(mlwd_results_path).exists():
        plot_interference_accuracy(mlwd_results_path, str(out / "fig_interference_accuracy.pdf"))

    if real_dir and sim_dir:
        plot_real_vs_sim(real_dir, sim_dir, str(out / "fig_real_vs_sim.pdf"))
