#!/usr/bin/env python3
# fig_bandwidth_bridge_v2.py -- Fig 5 redesign: markers, sequential
# blues (ordered DPU rate), direct peak labels (no legend), starred
# crossovers. No theory curve (M3: no closed form predicts cycles;
# kappa regime-dependent) and no global regime shading (crossover is
# per-line). Anchors: Table V cells, verified 2026-07-15.
import csv, statistics as st
from collections import defaultdict
import fig_style
fig_style.apply()
import matplotlib.pyplot as plt

CSV = "sensitivity_gpd8_results.csv"
OUT = "fig_bandwidth_bridge_v4.pdf"
BWS  = [5, 10, 20, 50]
DPUS = [(20500, fig_style.CARTS_BLUE, "o", "DPU = 20.5 GB/s"),
        (50000, "#E8A33D", "s", "DPU = 50 GB/s"),
        (83900, "#8E6C9E", "^", "DPU = 83.9 GB/s")]
LOCK = {(5,20500):10.3,(5,50000):6.9,(5,83900):6.9,
 (10,20500):6.1,(10,50000):11.8,(10,83900):9.6,
 (20,20500):3.4,(20,50000):6.3,(20,83900):10.7,
 (50,20500):3.4,(50,50000):3.4,(50,83900):3.4}
TOL = 0.05
PEAK = {20500: 5, 50000: 10, 83900: 20}

d = defaultdict(list)
for r in csv.DictReader(open(CSV)):
    d[(int(r["bw_gbs"]), int(r["dpu_bytes_per_us"]))].append(float(r["new_vs_c0"]))
mean = {k: st.mean(v) for k, v in d.items()}
for k, e in LOCK.items():
    assert len(d[k]) == 6 and abs(mean[k] - e) <= TOL, "anchor fail %s" % (k,)
print("all anchors verified -- plotting")

fig, ax = plt.subplots(figsize=(3.5, 2.8))
for dpu, col, mk, lbl in DPUS:
    y = [mean[(b, dpu)] for b in BWS]
    ax.plot(BWS, y, marker=mk, markersize=4, linewidth=1.3,
            color=col, zorder=3)
    pk = PEAK[dpu]
    ax.plot([pk], [mean[(pk, dpu)]], marker="*", markersize=10,
            color=col, markeredgecolor="white", markeredgewidth=0.5,
            zorder=4)
    ax.annotate(lbl, xy=(pk, mean[(pk, dpu)]),
                xytext=(10, 8), textcoords="offset points",
                ha="left", fontsize=7, color=col)
ax.set_xscale("log")
ax.set_xticks(BWS); ax.set_xticklabels(["5", "10", "20", "50"])
ax.minorticks_off()
ax.set_xlabel("inter-domain link bandwidth (GB/s, log scale)")
ax.set_ylabel("mean gain vs C0 (%)")
ax.set_ylim(0, 14.5)
ax.annotate("$\\star$ peak $\\approx$ transfer$\\leftrightarrow$DPU crossover,\n"
            "moves right as the DPU speeds up (M2)",
            xy=(0.97, 0.05), xycoords="axes fraction",
            ha="right", fontsize=6.5, color="#555555")
fig.savefig(OUT)
print("wrote", OUT)
