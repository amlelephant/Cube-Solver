"""Shared matplotlib style for every paper figure.

Palette is the validated categorical set from the dataviz reference
instance (light mode), used in fixed slot order and never cycled.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIGS = Path(__file__).resolve().parents[1] / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

# categorical slots, fixed order
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100",
    "#e87ba4", "#008300", "#4a3aa7", "#e34948")
CAT = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e3e2dd"
SURFACE = "#ffffff"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK2,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.size": 0,
    "ytick.major.size": 0,
    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "lines.linewidth": 2.0,
    "patch.linewidth": 0,
    "figure.dpi": 160,
})


def grid(ax, axis="y"):
    ax.grid(axis=axis, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    p = FIGS / name
    fig.savefig(p, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(p.with_suffix(".png"), bbox_inches="tight", pad_inches=0.02,
                dpi=220)
    plt.close(fig)
    print(f"  wrote figures/{name}")
    return p
