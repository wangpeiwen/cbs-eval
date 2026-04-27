"""第三章配套图表：基于实验数据生成。

图表清单：
  1. 干扰敏感度热力图（两模型对比，按 phase 分）
  2. Prefill vs Decode 阶段异构性（CI 对比柱状图）
  3. 共置干扰系数 α_d 随 prefill_seq 变化趋势
  4. 四维敏感度随 seq_len 变化趋势（折线图）

Usage:
    python3 -m plot.chap3_plots
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from pathlib import Path

# 字体注册
from plot.style import setup_thesis_style, COLORS


def _load(path):
    with open(path) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────
# 图 1: 干扰敏感度热力图（两模型 × 两阶段 = 4 个子图）
# ─────────────────────────────────────────────────────────

def plot_sensitivity_heatmap(
    *model_args,
    output_path: str = "results/figures/fig_sensitivity_heatmap.jpg",
):
    """NxM heatmap: rows=model, cols=phase. Accepts (name, data) pairs.
    Only common complete rows across ALL models."""
    setup_thesis_style()

    models = list(model_args)
    n_models = len(models)

    dims = ["sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw"]
    dim_labels = ["$\\sigma_{\\mathrm{bs}}$", "$\\sigma_{\\mathrm{cu}}$",
                  "$\\sigma_{\\mathrm{l2}}$", "$\\sigma_{\\mathrm{bw}}$"]
    seq_lengths = [32, 64, 128, 512, 2048]
    batches = [1, 4]
    phases = ["prefill", "decode"]

    # Find (b, s) combos where ALL models have complete data for BOTH phases
    common_bs = []
    for b in batches:
        for s in seq_lengths:
            all_ok = True
            for phase in phases:
                key = f"b{b}_s{s}_{phase}"
                for _, sens in models:
                    entry = sens.get(key, {})
                    if not all(entry.get(d) is not None and entry.get(d) != 0 for d in dims):
                        all_ok = False
            if all_ok:
                common_bs.append((b, s))

    if not common_bs:
        print("No common complete entries, skipping heatmap.")
        return

    row_labels = [f"b={b}, s={s}" for b, s in common_bs]

    fig, axes = plt.subplots(n_models, 2, figsize=(8, 4 * n_models))
    if n_models == 1:
        axes = [axes]

    vmin, vmax = 0, 2.0
    im = None

    for row, (model_name, sens) in enumerate(models):
        for col, phase in enumerate(phases):
            ax = axes[row][col]

            matrix = []
            for b, s in common_bs:
                key = f"b{b}_s{s}_{phase}"
                entry = sens[key]
                matrix.append([entry.get(d, 0) for d in dims])
            matrix = np.array(matrix)

            im = ax.imshow(matrix, cmap="YlOrRd", aspect="auto",
                           vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(dim_labels)))
            ax.set_xticklabels(dim_labels, fontsize=10)
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=8)
            ax.set_title(f"{model_name} — {phase.capitalize()}", fontsize=11)

            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    val = matrix[i, j]
                    color = "white" if val > 1.0 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=8, color=color)

    # Colorbar on the right, outside all subplots
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="$\\sigma$")
    fig.tight_layout(rect=[0, 0, 0.89, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────
# 图 2: Prefill vs Decode 阶段异构性（CI 对比）
# ─────────────────────────────────────────────────────────

def plot_phase_heterogeneity(
    qwen_ci: dict, llama_ci: dict,
    qwen_sens: dict, llama_sens: dict,
    output_path: str = "results/figures/fig_phase_heterogeneity.jpg",
):
    """Grouped bar chart: CI_attn and CI_ffn for prefill vs decode proxy."""
    setup_thesis_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    seq_lengths = [32, 64, 128, 512, 2048]
    x = np.arange(len(seq_lengths))
    w = 0.35

    for ax, (model_name, ci_data, sens_data) in zip(
        [ax1, ax2],
        [("Qwen2.5-7B", qwen_ci, qwen_sens),
         ("LLaMA-3.1-8B", llama_ci, llama_sens)],
    ):
        ci_attn = [ci_data.get(f"b1_s{s}", {}).get("ci_attn", 0) for s in seq_lengths]
        ci_ffn = [ci_data.get(f"b1_s{s}", {}).get("ci_ffn", 0) for s in seq_lengths]

        ax.bar(x - w/2, ci_attn, w, label="CI$_{\\mathrm{attn}}$", color="#4c72b0", edgecolor="white")
        ax.bar(x + w/2, ci_ffn, w, label="CI$_{\\mathrm{ffn}}$", color="#dd8452", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seq_lengths])
        ax.set_xlabel("Sequence Length $s$")
        ax.set_ylabel("Compute Intensity CI (FLOP/Byte)")
        ax.set_title(model_name)
        ax.legend()
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.suptitle("Attention vs FFN Compute Intensity ($b$=1)", fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────
# 图 3: α_d 随 prefill_seq 变化趋势
# ─────────────────────────────────────────────────────────

def plot_alpha_d_trend(
    coloc: dict,
    output_path: str = "results/figures/fig_alpha_d_trend.jpg",
):
    """Line plot: α_d vs prefill_seq for different (decode_batch, prefill_batch) combos."""
    setup_thesis_style()

    pairs = coloc["pairs"]
    fig, ax = plt.subplots(figsize=(7, 4.5))

    combos = {}
    for p in pairs:
        key = (p["decode_batch"], p["prefill_batch"])
        combos.setdefault(key, []).append(p)

    colors_list = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    markers_list = ["o", "s", "^", "D"]

    for i, ((db, pb), ps) in enumerate(sorted(combos.items())):
        ps.sort(key=lambda x: x["prefill_seq"])
        seqs = [p["prefill_seq"] for p in ps]
        alphas = [p["alpha_d"] for p in ps]
        ax.plot(seqs, alphas,
                marker=markers_list[i % len(markers_list)],
                color=colors_list[i % len(colors_list)],
                label=f"d_batch={db}, p_batch={pb}",
                linewidth=1.5, markersize=6)

    ax.set_xlabel("Prefill Sequence Length")
    ax.set_ylabel("Interference Coefficient $\\alpha_d$")
    ax.set_title("$\\alpha_d$ vs Prefill Sequence Length (Qwen2.5-7B)")
    ax.legend(fontsize=9)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────
# 图 4: 四维敏感度随 seq_len 变化（Decode 阶段）
# ─────────────────────────────────────────────────────────

def plot_sensitivity_trend(
    qwen_sens: dict, llama_sens: dict,
    output_path: str = "results/figures/fig_sensitivity_trend.jpg",
):
    """Line plot: σ_bs/cu/l2/bw vs seq_len for decode phase, b=1."""
    setup_thesis_style()

    dims = ["sigma_bs", "sigma_cu", "sigma_l2", "sigma_bw"]
    dim_labels = ["σ$_{\\mathrm{bs}}$", "σ$_{\\mathrm{cu}}$",
                  "σ$_{\\mathrm{l2}}$", "σ$_{\\mathrm{bw}}$"]
    colors_dim = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    seq_lengths = [32, 64, 128, 512, 2048]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (model_name, sens) in zip(
        [ax1, ax2],
        [("Qwen2.5-7B", qwen_sens), ("LLaMA-3.1-8B", llama_sens)],
    ):
        for d, dim in enumerate(dims):
            vals = []
            valid_seqs = []
            for s in seq_lengths:
                key = f"b1_s{s}_decode"
                entry = sens.get(key, {})
                v = entry.get(dim)
                if v is not None:
                    vals.append(v)
                    valid_seqs.append(s)
            if vals:
                ax.plot(valid_seqs, vals, marker="o", color=colors_dim[d],
                        label=dim_labels[d], linewidth=1.5, markersize=5)

        ax.set_xlabel("Sequence Length $s$")
        ax.set_ylabel("Interference Sensitivity $\\sigma$")
        ax.set_title(f"{model_name} (Decode, b=1)")
        ax.legend(fontsize=9)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())

    fig.suptitle("Decode-phase Interference Sensitivity vs Sequence Length ($b$=1)", fontsize=12, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────

def main():
    qwen_sens = _load("results/qwen-2.5-7b-sensitivity.json")
    llama_sens = _load("results/llama-3.1-8b-sensitivity.json")
    qwen3_sens = _load("results/qwen3-14b-sensitivity.json")
    qwen_ci = _load("results/qwen2.5-7B-ci.json")
    llama_ci = _load("results/llama-3.1-8B-ci.json")
    qwen_coloc = _load("results/qwen2.5-7b-colocation.json")

    plot_sensitivity_heatmap(
        ("Qwen2.5-7B", qwen_sens),
        ("LLaMA-3.1-8B", llama_sens),
        ("Qwen3-14B", qwen3_sens),
    )
    plot_phase_heterogeneity(qwen_ci, llama_ci, qwen_sens, llama_sens)
    plot_alpha_d_trend(qwen_coloc)
    plot_sensitivity_trend(qwen_sens, llama_sens)
    plot_alpha_prediction()
    plot_weight_bar()

    print("\nAll chap3 figures generated.")


# ─────────────────────────────────────────────────────────
# 图 5: α_d 估算值 vs 真值散点图
# ─────────────────────────────────────────────────────────

def plot_alpha_prediction(
    calibration_path: str = "results/combined-ols-calibration.json",
    output_path: str = "results/figures/fig_alpha_prediction.jpg",
):
    """Scatter: predicted α_d vs ground truth, two models in different colors."""
    setup_thesis_style()

    data = _load(calibration_path)
    mae = data["combined"]["mae"]
    r2 = data["combined"]["r2"]

    # Detect per-model keys dynamically
    model_keys = [k for k in data.keys()
                  if k not in ("method", "combined", "n_samples", "weights", "intercept",
                               "scaler_mean", "scaler_scale", "cv_mae", "cv_r2")]

    model_colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8c564b"]
    model_markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(5.5, 5))

    all_true, all_pred = [], []
    for mk in model_keys:
        all_true.extend(data[mk]["y_true"])
        all_pred.extend(data[mk]["y_pred"])

    lo = min(min(all_true), min(all_pred)) - 0.01
    hi = max(max(all_true), max(all_pred)) + 0.01
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="Perfect prediction")

    n_total = data["n_samples"]["total"]
    for i, mk in enumerate(model_keys):
        n_i = len(data[mk]["y_true"])
        label = mk.replace("_", " ").replace("25 ", "2.5-").replace("31 ", "3.1-").replace("3 ", "3-")
        label = f"{label} (n={n_i})"
        ax.scatter(data[mk]["y_true"], data[mk]["y_pred"],
                   c=model_colors[i % len(model_colors)],
                   marker=model_markers[i % len(model_markers)],
                   s=40, alpha=0.7, edgecolors="white", linewidth=0.5, label=label)

    ax.set_xlabel("Ground-truth $\\alpha_d$")
    ax.set_ylabel("Predicted $\\hat{\\alpha}_d$")
    ax.set_title(f"Interference Coefficient Prediction\n(n={n_total}, MAE={mae:.4f}, $R^2$={r2:.3f})")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ─────────────────────────────────────────────────────────
# 图 6: 加权系数柱状图
# ─────────────────────────────────────────────────────────

def plot_weight_bar(
    calibration_path: str = "results/combined-ols-calibration.json",
    output_path: str = "results/figures/fig_ols_weights.jpg",
):
    """Bar chart of Ridge regression coefficients, top-10 by absolute value."""
    setup_thesis_style()

    data = _load(calibration_path)
    weights = data["weights"]

    # Sort by absolute value, take top 10
    sorted_items = sorted(weights.items(), key=lambda x: -abs(x[1]))[:10]
    names = [k for k, _ in sorted_items]
    vals = [v for _, v in sorted_items]

    # Pretty labels
    label_map = {
        "v_sigma_bs": "$\\sigma^v_{\\mathrm{bs}}$", "v_sigma_cu": "$\\sigma^v_{\\mathrm{cu}}$",
        "v_sigma_l2": "$\\sigma^v_{\\mathrm{l2}}$", "v_sigma_bw": "$\\sigma^v_{\\mathrm{bw}}$",
        "v_ci_attn": "$\\mathrm{CI}^v_{\\mathrm{attn}}$", "v_ci_ffn": "$\\mathrm{CI}^v_{\\mathrm{ffn}}$",
        "v_r_attn": "$r^v_{\\mathrm{attn}}$", "v_r_ffn": "$r^v_{\\mathrm{ffn}}$",
        "v_baseline": "$T^v_{\\mathrm{base}}$",
        "a_sigma_bs": "$\\sigma^a_{\\mathrm{bs}}$", "a_sigma_cu": "$\\sigma^a_{\\mathrm{cu}}$",
        "a_sigma_l2": "$\\sigma^a_{\\mathrm{l2}}$", "a_sigma_bw": "$\\sigma^a_{\\mathrm{bw}}$",
        "a_ci_attn": "$\\mathrm{CI}^a_{\\mathrm{attn}}$", "a_ci_ffn": "$\\mathrm{CI}^a_{\\mathrm{ffn}}$",
        "a_r_attn": "$r^a_{\\mathrm{attn}}$", "a_r_ffn": "$r^a_{\\mathrm{ffn}}$",
        "a_baseline": "$T^a_{\\mathrm{base}}$",
    }
    pretty_names = [label_map.get(n, n) for n in names]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(pretty_names))
    colors = ["#4c72b0" if v >= 0 else "#c44e52" for v in vals]
    bars = ax.barh(x, vals, color=colors, height=0.6, edgecolor="white")

    for bar, val in zip(bars, vals):
        w = bar.get_width()
        ha = "left" if w >= 0 else "right"
        ax.text(w, bar.get_y() + bar.get_height() / 2, f" {val:+.4f}",
                ha=ha, va="center", fontsize=9)

    ax.set_yticks(x)
    ax.set_yticklabels(pretty_names, fontsize=11)
    ax.set_xlabel("Coefficient (standardized)")
    n = data["n_samples"]["total"]
    r2 = data["combined"]["r2"]
    ax.set_title(f"Ridge Regression Coefficients (n={n}, $R^2$={r2:.3f})")
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.invert_yaxis()

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
