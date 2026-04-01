"""Matplotlib style configuration for thesis figures."""
import matplotlib
import matplotlib.pyplot as plt


def setup_thesis_style():
    """Apply consistent thesis-quality style to all matplotlib figures."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.figsize": (6, 4),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",
    })
    # Try to set Chinese font for mixed CJK/Latin text
    try:
        matplotlib.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


COLORS = {
    "disagg_static": "#1f77b4",
    "coloc_sarathi": "#ff7f0e",
    "cbs_nomig": "#2ca02c",
    "cbs_norole": "#d62728",
    "cbs_full": "#9467bd",
}

MARKERS = {
    "disagg_static": "o",
    "coloc_sarathi": "s",
    "cbs_nomig": "^",
    "cbs_norole": "D",
    "cbs_full": "*",
}

LABELS = {
    "disagg_static": "Disagg-Static",
    "coloc_sarathi": "Coloc-Sarathi",
    "cbs_nomig": "CBS-NoMig",
    "cbs_norole": "CBS-NoRole",
    "cbs_full": "CBS-Full",
}

# Canonical ordering for consistent legend/bar placement
SYSTEM_ORDER = [
    "disagg_static",
    "coloc_sarathi",
    "cbs_nomig",
    "cbs_norole",
    "cbs_full",
]
