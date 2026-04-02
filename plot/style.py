"""Matplotlib style configuration for thesis figures.

Fonts are bundled in ``fonts/`` so the project works on servers without
system-wide CJK fonts.  Chinese uses SimSun/Songti, English uses
Times New Roman.  Base font size is 12pt (小四).
"""
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

# ── Font registration ────────────────────────────────────────────────
_FONT_DIR = Path(__file__).resolve().parent.parent / "fonts"

_REGISTERED = False


def _register_fonts():
    """Register bundled TTF/TTC fonts with matplotlib (once)."""
    global _REGISTERED
    if _REGISTERED:
        return
    for fp in _FONT_DIR.glob("*"):
        if fp.suffix.lower() in (".ttf", ".ttc", ".otf"):
            fm.fontManager.addfont(str(fp))
    _REGISTERED = True


# ── Style setup ──────────────────────────────────────────────────────

def setup_thesis_style():
    """Apply consistent thesis-quality style to all matplotlib figures.

    - Chinese text: Songti SC (宋体)
    - English / math: Times New Roman
    - Base font size: 12 pt (小四)
    """
    _register_fonts()

    # Clear matplotlib font cache so newly registered fonts are found
    fm._load_fontmanager(try_read_cache=False)

    plt.rcParams.update({
        # Font family fallback chain: Times New Roman for Latin, Songti for CJK
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Songti SC", "SimSun", "DejaVu Serif"],
        "font.sans-serif": ["Songti SC", "SimSun", "DejaVu Sans"],
        "mathtext.fontset": "stix",  # STIX matches Times New Roman well
        "axes.unicode_minus": False,

        # 小四 = 12pt
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        # Figure defaults
        "figure.figsize": (6, 4),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",

        # Aesthetics
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.framealpha": 0.8,
        "legend.edgecolor": "0.8",
    })


# ── Colour / marker / label palettes ────────────────────────────────

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
