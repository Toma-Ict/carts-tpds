#!/usr/bin/env python3
# fig_headline_seeds.py -- Fig 7 redesign: raincloud-lite.
# Per-seed dots + mean diamond + min-max whisker, gpd=8 and gpd=16
# series. No bars, no box plots (n=6). Assert-gated against Table IV
# and S30 anchors (verify_headline_seeds.py, all passed 2026-07-15).
import csv, statistics as st
from collections import defaultdict
import fig_style
fig_style.apply()
import matplotlib.pyplot as plt

G8CSV, G16CSV = "scale_multiseed_gpd8_results.csv", "scale_multiseed_gpd16_results.csv"
OUT = "fig_headline_seeds_v4.pdf"
DARK, LIGHT = fig_style.CARTS_BLUE, "#1D9E75"
ANCH_G8  = {4: (10.3, 6), 8: (5.9, 6), 16: (1.0, 4)}
ANCH_G16 = {4: (9.45, 6), 8: (5.34, 6)}
TOL = 0.05
XPOS = {4: 0, 8: 1, 16: 2}

def load(path):
    d = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        d[int(r["num_domains"])][int(r["seed"])] = float(r["new_vs_c0"])
    return d

g8, g16 = load(G8CSV), load(G16CSV)
for anch, data in ((ANCH_G8, g8), (ANCH_G16, g16)):
    for dom, (em, eg) in anch.items():
        v = list(data[dom].values())
        assert len(v) == 6 and abs(st.mean(v) - em) <= TOL \
            and sum(1 for x in v if x >= 0) == eg, "anchor fail dom=%d" % dom
print("all anchors verified -- plotting")

fig, ax = plt.subplots(figsize=(3.5, 2.7))
JIT = [-0.05, -0.03, -0.01, 0.01, 0.03, 0.05]

def series(data, offset, color, label):
    first = True
    for dom in sorted(data):
        x0 = XPOS[dom] + offset
        v = [data[dom][s] for s in sorted(data[dom])]
        ax.plot([x0, x0], [min(v), max(v)], color=color,
                linewidth=0.8, alpha=0.6, zorder=2)
        ax.scatter([x0 + j for j in JIT], v, s=13, color=color,
                   alpha=0.9, linewidths=0, zorder=3)
        ax.scatter([x0], [st.mean(v)], marker="D", s=32, color=color,
                   edgecolors="white", linewidths=0.6, zorder=4,
                   label=label if first else None)
        first = False

series(g8,  -0.13, DARK,  "gpd = 8")
series(g16, +0.13, LIGHT, "gpd = 16")
ax.axhline(0, color="black", linewidth=0.6, zorder=1)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["4", "8", "16"])
ax.set_xlim(-0.5, 2.5); ax.set_ylim(-4, 27)
ax.set_xlabel("number of domains")
ax.set_ylabel("completion-time gain vs C0 (%)")
ax.legend(loc="upper right", handletextpad=0.4)
ax.annotate("two neutral seeds",
            xy=(-0.13, 0.0), xytext=(-0.40, 3.8), fontsize=6.5,
            color=DARK, arrowprops=dict(arrowstyle="->", color=DARK, lw=0.7))
ax.annotate("washing out", xy=(1.87, 1.0), xytext=(1.62, 6.5),
            fontsize=6.5, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#555555", lw=0.7))
fig.savefig(OUT)
print("wrote", OUT)
