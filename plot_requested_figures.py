"""Generate requested MLWD academic figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib import font_manager as fm
from matplotlib.font_manager import FontProperties

from mlwd.extrapolate import extrapolate_full


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "results" / "mlwd"
OUT_DIR = ROOT / "results" / "figures"
MODELS = ["qwen2.5-7b", "llama-3.1-8b", "qwen3-14b"]
MODEL_LABELS = {
    "qwen2.5-7b": "Qwen2.5-7B",
    "llama-3.1-8b": "LLaMA-3.1-8B",
    "qwen3-14b": "Qwen3-14B",
}
CN_FONT = FontProperties(fname=str(ROOT / "fonts" / "Songti.ttc"))
EN_FONT = FontProperties(fname=str(ROOT / "fonts" / "TimesNewRoman.ttf"))
BOLD_EN_FONT = FontProperties(fname=str(ROOT / "fonts" / "TimesNewRoman-Bold.ttf"))


def setup_style() -> None:
    for font_path in [
        ROOT / "fonts" / "TimesNewRoman.ttf",
        ROOT / "fonts" / "TimesNewRoman-Bold.ttf",
        ROOT / "fonts" / "TimesNewRoman-Italic.ttf",
        ROOT / "fonts" / "Songti.ttc",
    ]:
        if font_path.exists():
            fm.fontManager.addfont(str(font_path))

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "font.sans-serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "axes.unicode_minus": False,
        }
    )


def load_model_data(model: str) -> dict:
    with (DATA_DIR / model / "mlwd_complete.json").open() as f:
        return json.load(f)


def plot_model_decomposition() -> None:
    fields = [
        ("sigma_bs", "块调度器", "#4C78A8"),
        ("sigma_cu", "计算单元", "#72B7B2"),
        ("sigma_l2", "L2 缓存", "#F58518"),
        ("sigma_bw", "内存带宽", "#E45756"),
    ]
    rows: list[str] = []
    values: list[list[float]] = []

    for model in MODELS:
        data = load_model_data(model)
        for phase, phase_label in [("prefill", "预填充"), ("decode", "解码")]:
            entries = [v for v in data.values() if v.get("phase") == phase]
            means = np.array(
                [
                    np.mean([max(float(e.get(field, 0.0)), 0.0) for e in entries])
                    for field, _, _ in fields
                ]
            )
            values.append((means / means.sum() * 100.0).tolist())
            rows.append(f"{MODEL_LABELS[model]} / {phase_label}")

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    y = np.arange(len(rows))
    left = np.zeros(len(rows))
    for idx, (_, label, color) in enumerate(fields):
        widths = np.array([row[idx] for row in values])
        ax.barh(
            y,
            widths,
            left=left,
            height=0.68,
            color=color,
            edgecolor="white",
            linewidth=1.0,
            label=label,
        )
        for yi, x0, w in zip(y, left, widths):
            if w >= 7:
                ax.text(
                    x0 + w / 2,
                    yi,
                    f"{w:.0f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=10,
                    fontproperties=BOLD_EN_FONT,
                )
        left += widths

    ax.set_yticks(y)
    ax.set_yticklabels(rows)
    for label in ax.get_yticklabels():
        label.set_fontproperties(CN_FONT)
    for label in ax.get_xticklabels():
        label.set_fontproperties(EN_FONT)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("平均 MLWD 敏感性的相对贡献（%）", fontproperties=CN_FONT)
    ax.set_title("逐模型干扰敏感性分解", pad=12, fontproperties=CN_FONT)
    ax.legend(
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.12),
        frameon=False,
        columnspacing=1.1,
        handlelength=1.8,
        prop=CN_FONT,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.grid(axis="y", visible=False)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"fig_model_decomposition.{suffix}")
    plt.close(fig)


def plot_extrapolation_cost_3d() -> None:
    seqs = [32, 64, 128, 512, 2048]
    batches = [1, 4, 16]
    x = np.log2(seqs)
    y = np.array(batches)
    xg, yg = np.meshgrid(x, y)

    cost_cmap = LinearSegmentedColormap.from_list(
        "academic_cost",
        ["#FAF8F1", "#D9E8E2", "#94C3B8", "#4C88A8", "#263B63"],
    )
    surfaces = []
    for model in MODELS:
        data = load_model_data(model)
        full = extrapolate_full(data, model, batch_sizes=batches, seq_lengths=seqs)
        z = np.zeros_like(xg, dtype=float)
        measured_x, measured_y, measured_z = [], [], []

        for i, batch in enumerate(batches):
            for j, seq in enumerate(seqs):
                pre = full[f"b{batch}_s{seq}_prefill"]
                dec = full[f"b{batch}_s{seq}_decode"]
                cost_ms = float(pre["baseline_ms"]) + float(dec["baseline_ms"])
                z[i, j] = np.log10(cost_ms)
                if pre.get("source") == "measured" and dec.get("source") == "measured":
                    measured_x.append(np.log2(seq))
                    measured_y.append(batch)
                    measured_z.append(np.log10(cost_ms))
        surfaces.append((model, z, measured_x, measured_y, measured_z))

    norm = Normalize(
        vmin=min(float(z.min()) for _, z, *_ in surfaces),
        vmax=max(float(z.max()) for _, z, *_ in surfaces),
    )

    fig = plt.figure(figsize=(13.8, 4.9))
    for idx, (model, z, measured_x, measured_y, measured_z) in enumerate(surfaces, 1):
        ax = fig.add_subplot(1, 3, idx, projection="3d")
        ax.plot_surface(
            xg,
            yg,
            z,
            cmap=cost_cmap,
            norm=norm,
            edgecolor="white",
            linewidth=0.65,
            alpha=0.94,
            shade=True,
            antialiased=True,
        )
        ax.scatter(
            measured_x,
            measured_y,
            measured_z,
            color="#7A2535",
            edgecolor="white",
            linewidth=0.7,
            s=44,
            depthshade=False,
            label="实测点" if idx == 1 else None,
        )
        ax.set_title(MODEL_LABELS[model], pad=4, fontsize=13, fontproperties=EN_FONT)
        ax.set_xlabel("序列长度", labelpad=6, fontsize=10, fontproperties=CN_FONT)
        ax.set_ylabel("批大小", labelpad=6, fontsize=10, fontproperties=CN_FONT)
        ax.set_zlabel("log10 成本（ms）", labelpad=6, fontsize=10, fontproperties=CN_FONT)
        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in seqs])
        ax.set_yticks(batches)
        for label in ax.get_xticklabels() + ax.get_yticklabels() + ax.get_zticklabels():
            label.set_fontproperties(EN_FONT)
        ax.tick_params(axis="both", which="major", labelsize=9, pad=0)
        ax.tick_params(axis="z", which="major", labelsize=9, pad=1)
        ax.view_init(elev=26, azim=-58)
        ax.xaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.yaxis.pane.set_facecolor((1, 1, 1, 0.0))
        ax.zaxis.pane.set_facecolor((1, 1, 1, 0.0))
        if idx == 1:
            ax.legend(
                loc="upper left",
                frameon=False,
                fontsize=9,
                markerscale=0.75,
                prop=CN_FONT,
            )

    fig.suptitle(
        "参数化外推成本曲面（预填充 + 解码基线延迟）",
        y=0.98,
        fontsize=15,
        fontproperties=CN_FONT,
    )
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=cost_cmap)
    mappable.set_array([])
    fig.subplots_adjust(left=0.02, right=0.92, bottom=0.06, top=0.82, wspace=0.02)
    cax = fig.add_axes([0.945, 0.28, 0.012, 0.42])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("log10 成本（ms）", fontsize=10, fontproperties=CN_FONT)
    cbar.ax.tick_params(labelsize=9)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(EN_FONT)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"fig_extrapolation_cost_3d.{suffix}")
    plt.close(fig)


def plot_extrapolation_cost_heatmap() -> None:
    seqs = [32, 64, 128, 512, 2048]
    batches = [1, 4, 16]
    cost_cmap = LinearSegmentedColormap.from_list(
        "academic_cost_2d",
        ["#FAF8F1", "#D9E8E2", "#94C3B8", "#4C88A8", "#263B63"],
    )

    matrices = []
    measured_masks = []
    for model in MODELS:
        data = load_model_data(model)
        full = extrapolate_full(data, model, batch_sizes=batches, seq_lengths=seqs)
        mat = np.zeros((len(batches), len(seqs)), dtype=float)
        measured = np.zeros_like(mat, dtype=bool)

        for i, batch in enumerate(batches):
            for j, seq in enumerate(seqs):
                pre = full[f"b{batch}_s{seq}_prefill"]
                dec = full[f"b{batch}_s{seq}_decode"]
                cost_ms = float(pre["baseline_ms"]) + float(dec["baseline_ms"])
                mat[i, j] = np.log10(cost_ms)
                measured[i, j] = (
                    pre.get("source") == "measured"
                    and dec.get("source") == "measured"
                )
        matrices.append(mat)
        measured_masks.append(measured)

    norm = Normalize(
        vmin=min(float(mat.min()) for mat in matrices),
        vmax=max(float(mat.max()) for mat in matrices),
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), sharey=True)
    for ax, model, mat, measured in zip(axes, MODELS, matrices, measured_masks):
        im = ax.imshow(
            mat,
            cmap=cost_cmap,
            norm=norm,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
        )

        for i in range(len(batches)):
            for j in range(len(seqs)):
                text_color = "white" if norm(mat[i, j]) > 0.58 else "#1f2933"
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=text_color,
                    fontproperties=EN_FONT,
                )
                if measured[i, j]:
                    ax.scatter(
                        j,
                        i,
                        s=72,
                        facecolors="none",
                        edgecolors="#7A2535",
                        linewidths=1.6,
                    )

        ax.set_title(MODEL_LABELS[model], fontproperties=EN_FONT, fontsize=13, pad=8)
        ax.set_xticks(np.arange(len(seqs)))
        ax.set_xticklabels([str(s) for s in seqs], fontproperties=EN_FONT)
        ax.set_yticks(np.arange(len(batches)))
        ax.set_yticklabels([str(b) for b in batches], fontproperties=EN_FONT)
        ax.set_xlabel("序列长度", fontproperties=CN_FONT)
        ax.set_xticks(np.arange(-0.5, len(seqs), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(batches), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylabel("批大小", fontproperties=CN_FONT)
    fig.suptitle("参数化外推成本热力图（预填充 + 解码基线延迟）", fontproperties=CN_FONT, fontsize=15, y=1.03)
    fig.subplots_adjust(left=0.07, right=0.9, bottom=0.17, top=0.78, wspace=0.08)

    cax = fig.add_axes([0.92, 0.22, 0.014, 0.5])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("log10 成本（ms）", fontproperties=CN_FONT, fontsize=10)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(EN_FONT)

    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markerfacecolor="none",
            markeredgecolor="#7A2535",
            markeredgewidth=1.6,
            markersize=8,
            label="实测点",
        )
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.07, 0.96),
        frameon=False,
        prop=CN_FONT,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        fig.savefig(OUT_DIR / f"fig_extrapolation_cost_heatmap.{suffix}")
    plt.close(fig)


def main() -> None:
    setup_style()
    plot_model_decomposition()
    plot_extrapolation_cost_heatmap()
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
