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
# Fig 6-x: Threshold sensitivity (theta_ceil, theta_floor)
# ------------------------------------------------------------------

def plot_threshold_sensitivity(
    results_dir: str,
    output_path: str = "fig_threshold_sensitivity.pdf",
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> None:
    """Bar chart: goodput and migration count vs theta_ceil / theta_floor."""
    from analysis.sensitivity import threshold_sensitivity

    sweep = threshold_sensitivity(
        results_dir,
        slo_ttft=slo_ttft,
        slo_tpot=slo_tpot,
        warmup_s=warmup_s,
    )
    if not sweep.get("ceil_values"):
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # theta_ceil
    vals = sweep["ceil_values"]
    gp = sweep["ceil_goodput"]
    slo = sweep["ceil_slo"]
    x = np.arange(len(vals))
    ax1.bar(x - 0.15, gp, 0.3, color=COLORS["cbs_full"], label="Goodput")
    ax1_t = ax1.twinx()
    ax1_t.plot(x, slo, "s--", color=COLORS["disagg_static"], label="SLO%")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(v) for v in vals])
    ax1.set_xlabel(r"$\theta_{\mathrm{ceil}}$")
    ax1.set_ylabel("Goodput (req/s)")
    ax1_t.set_ylabel("SLO Attainment (%)")
    ax1_t.set_ylim(80, 100)
    ax1.set_title(r"$\theta_{\mathrm{ceil}}$ 敏感性")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_t.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    # theta_floor
    vals2 = sweep["floor_values"]
    gp2 = sweep["floor_goodput"]
    slo2 = sweep["floor_slo"]
    x2 = np.arange(len(vals2))
    ax2.bar(x2 - 0.15, gp2, 0.3, color=COLORS["cbs_full"], label="Goodput")
    ax2_t = ax2.twinx()
    ax2_t.plot(x2, slo2, "s--", color=COLORS["disagg_static"], label="SLO%")
    ax2.set_xticks(x2)
    ax2.set_xticklabels([str(v) for v in vals2])
    ax2.set_xlabel(r"$\theta_{\mathrm{floor}}$")
    ax2.set_ylabel("Goodput (req/s)")
    ax2_t.set_ylabel("SLO Attainment (%)")
    ax2_t.set_ylim(80, 100)
    ax2.set_title(r"$\theta_{\mathrm{floor}}$ 敏感性")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ------------------------------------------------------------------
# Fig 6-x: Mu sensitivity curve
# ------------------------------------------------------------------

def plot_mu_sensitivity(
    results_dir: str,
    output_path: str = "fig_mu_sensitivity.pdf",
    mu_values: Optional[List[float]] = None,
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> None:
    """Dual-axis plot: goodput and SLO attainment vs mu."""
    from analysis.sensitivity import mu_sensitivity

    sweep = mu_sensitivity(
        results_dir,
        mu_values=mu_values,
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

    ax1.plot(vals, sweep["goodput"], marker="o", color=color_gp,
             linewidth=1.5, markersize=6, label="Goodput")
    ax1.set_xlabel(r"$\mu$")
    ax1.set_ylabel("Goodput (req/s)", color=color_gp)
    ax1.tick_params(axis="y", labelcolor=color_gp)

    ax2 = ax1.twinx()
    ax2.plot(vals, sweep["slo_attainment"], marker="s", color=color_slo,
             linewidth=1.5, markersize=6, linestyle="--", label="SLO%")
    ax2.set_ylabel("SLO Attainment (%)", color=color_slo)
    ax2.tick_params(axis="y", labelcolor=color_slo)
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")
    ax1.set_title(r"$\mu$ 敏感性分析")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ------------------------------------------------------------------
# Fig 6-x: Bursty & long-context workload comparison
# ------------------------------------------------------------------

def plot_workload_comparison(
    results_dir: str,
    output_path: str = "fig_workload_comparison.pdf",
    slo_ttft: float = 2000.0,
    slo_tpot: float = 100.0,
    warmup_s: float = 120.0,
) -> None:
    """Grouped bar chart: goodput under uniform / bursty / long-context."""
    rdir = Path(results_dir)
    workloads = ["uniform", "bursty", "long_context"]
    wl_labels = ["均匀负载", "突发负载", "长上下文"]
    systems = ["disagg_static", "coloc_sarathi", "cbs_full"]

    data = {wl: {} for wl in workloads}
    for wl in workloads:
        for sys_name in systems:
            # Try to find any rate file
            wl_dir = rdir / wl
            if not wl_dir.exists():
                continue
            for sub in sorted(wl_dir.iterdir()):
                fpath = sub / f"{sys_name}.json" if sub.is_dir() else None
                if fpath and fpath.exists():
                    m = compute_metrics(str(fpath), slo_ttft=slo_ttft,
                                        slo_tpot=slo_tpot, warmup_s=warmup_s)
                    data[wl][sys_name] = m.goodput
                    break

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(workloads))
    n_sys = len(systems)
    w = 0.22

    for i, sys_name in enumerate(systems):
        vals = [data[wl].get(sys_name, 0) for wl in workloads]
        offset = (i - n_sys / 2 + 0.5) * w
        ax.bar(x + offset, vals, w,
               color=COLORS.get(sys_name, "gray"),
               label=LABELS.get(sys_name, sys_name),
               edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(wl_labels)
    ax.set_ylabel("Goodput (req/s)")
    ax.set_title("不同负载模式下的 Goodput 对比")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

def generate_all(
    results_dir: str,
    output_dir: str,
    model: str = "qwen2.5-7b",
    mlwd_results_path: Optional[str] = None,
    real_dir: Optional[str] = None,
    sim_dir: Optional[str] = None,
) -> None:
    """Produce every Chapter 6 figure and save to *output_dir*.

    Complete figure list (10 figures):
      1. fig_goodput_vs_rate.pdf      — §6.3 Goodput vs 到达率曲线
      2. fig_slo_attainment.pdf       — §6.3.4 SLO 达标率柱状图
      3. fig_tail_latency.pdf         — §6.3.5 P99 TTFT / TPOT 对比
      4. fig_interference_accuracy.pdf— §6.2 干扰估算散点图
      5. fig_ablation.pdf             — §6.4 消融实验柱状图
      6. fig_lambda_sensitivity.pdf   — §6.5.1 λ 敏感性曲线
      7. fig_threshold_sensitivity.pdf— §6.5.2 迁移阈值敏感性
      8. fig_mu_sensitivity.pdf       — §6.5.3 μ 敏感性曲线
      9. fig_real_vs_sim.pdf          — §6.3.3 真实 vs 模拟一致性
     10. fig_workload_comparison.pdf  — §6.3.6 不同负载模式对比
    """
    setup_thesis_style()
    os.makedirs(output_dir, exist_ok=True)
    out = Path(output_dir)

    # Core scheduling evaluation
    plot_goodput_vs_rate(results_dir, model, str(out / "fig_goodput_vs_rate.pdf"))
    plot_slo_attainment(results_dir, model, str(out / "fig_slo_attainment.pdf"))
    plot_tail_latency(results_dir, model, str(out / "fig_tail_latency.pdf"))

    # Ablation
    plot_ablation(results_dir, str(out / "fig_ablation.pdf"))

    # Sensitivity analysis
    plot_lambda_sensitivity(results_dir, str(out / "fig_lambda_sensitivity.pdf"))
    plot_threshold_sensitivity(results_dir, str(out / "fig_threshold_sensitivity.pdf"))
    plot_mu_sensitivity(results_dir, str(out / "fig_mu_sensitivity.pdf"))

    # Workload comparison (bursty / long-context)
    plot_workload_comparison(results_dir, str(out / "fig_workload_comparison.pdf"))

    # MLWD accuracy (requires separate data file)
    if mlwd_results_path and Path(mlwd_results_path).exists():
        plot_interference_accuracy(mlwd_results_path, str(out / "fig_interference_accuracy.pdf"))

    # Real vs simulation consistency
    if real_dir and sim_dir:
        plot_real_vs_sim(real_dir, sim_dir, str(out / "fig_real_vs_sim.pdf"))
