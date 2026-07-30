#!/usr/bin/env python3
# fig_style.py -- shared style for all CARTS paper figures
# usage: import fig_style; fig_style.apply()
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CARTS_BLUE  = "#2A7FB8"
NAIVE_AMBER = "#F4A300"
C4_GRAY     = "#7A7A7A"
NEG_RED     = "#D9534F"

def apply():
    plt.rcParams.update({
        "font.family":      "sans-serif",
        "font.size":        9,
        "axes.labelsize":   9,
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.fontsize":  7.5,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "axes.grid":        True,
        "axes.grid.axis":   "y",
        "grid.linewidth":   0.4,
        "grid.alpha":       0.35,
        "legend.frameon":   False,
        "savefig.bbox":     "tight",
        "pdf.fonttype":     42,
    })
