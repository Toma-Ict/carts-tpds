#!/usr/bin/env python3
# fig_necessity_composite.py -- Fig 8 redesign: (a) grouped bars +
# (b,c) robustness heatmaps; replaces Tables V/VI in the paper.
# Data computed from CSVs, assert-gated against verified anchors
# (verify_necessity_numbers.py, 2026-07-15; Table VI (10,83.9)
# corrected +6.9 -> +6.8, full-precision 6.8471).
import csv, statistics as st
from collections import defaultdict
import fig_style
fig_style.apply()
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

CSV, C4CSV = "sensitivity_gpd8_results.csv", "c4_always_dedup_results.csv"
OUT  = "fig_necessity_composite_v2.pdf"
BWS  = [5, 10, 20, 50]
DPUS = [20500, 50000, 83900]
DPU_LBL = ["20.5", "50", "83.9"]
TOL  = 0.05

LOCK_CA = {(5,20500):(10.3,6),(5,50000):(6.9,4),(5,83900):(6.9,4),
 (10,20500):(6.1,6),(10,50000):(11.8,6),(10,83900):(9.6,5),
 (20,20500):(3.4,6),(20,50000):(6.3,6),(20,83900):(10.7,6),
 (50,20500):(3.4,6),(50,50000):(3.4,6),(50,83900):(3.4,6)}
LOCK_C4 = {(5,20500):(6.3,4),(5,50000):(6.9,4),(5,83900):(7.2,4),
 (10,20500):(4.7,4),(10,50000):(6.4,4),(10,83900):(6.8,4),
 (20,20500):(-2.6,4),(20,50000):(5.3,4),(20,83900):(6.2,4),
 (50,20500):(-32.7,0),(50,50000):(-3.4,3),(50,83900):(3.0,4)}
LOCK_NAIVE = {5: 6.94, 10: 5.83, 20: -1.49, 50: -32.70}

def load(path, col):
    d = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(int(r["bw_gbs"]), int(r["dpu_bytes_per_us"]))].append(float(r[col]))
    return d

ca, old = load(CSV, "new_vs_c0"), load(CSV, "old_vs_c0")
c4 = load(C4CSV, "c4_vs_c0")

def cell(d, bw, dpu):
    v = d[(bw, dpu)]
    assert len(v) == 6, "expected 6 seeds at (%d,%d)" % (bw, dpu)
    return st.mean(v), sum(1 for x in v if x >= 0)

for (bw, dpu), (em, eg) in LOCK_CA.items():
    m, g = cell(ca, bw, dpu)
    assert abs(m - em) <= TOL and g == eg, "CA anchor fail (%d,%d)" % (bw, dpu)
for (bw, dpu), (em, eg) in LOCK_C4.items():
    m, g = cell(c4, bw, dpu)
    assert abs(m - em) <= TOL and g == eg, "C4 anchor fail (%d,%d)" % (bw, dpu)
for bw, e in LOCK_NAIVE.items():
    m, _ = cell(old, bw, 20500)
    assert abs(m - e) <= TOL, "NAIVE anchor fail bw=%d" % bw
print("all anchors verified -- plotting")

fig = plt.figure(figsize=(7.1, 5.0), constrained_layout=True)
gs = fig.add_gridspec(2, 2, height_ratios=[0.95, 1.0])
axa = fig.add_subplot(gs[0, :])
axb = fig.add_subplot(gs[1, 0])
axc = fig.add_subplot(gs[1, 1])

x = range(len(BWS)); w = 0.26
ca_m  = [cell(ca,  b, 20500)[0] for b in BWS]
nv_m  = [cell(old, b, 20500)[0] for b in BWS]
c4_m  = [cell(c4,  b, 20500)[0] for b in BWS]
axa.bar([i - w for i in x], ca_m, w, color=fig_style.CARTS_BLUE,
        label="CARTS (cost-aware)", zorder=3)
axa.bar(list(x), nv_m, w, color=fig_style.NAIVE_AMBER,
        label="Naive placement", zorder=3)
axa.bar([i + w for i in x], c4_m, w, color=fig_style.C4_GRAY,
        label="Always-dedup (C4)", zorder=3)
axa.axhline(0, color="black", linewidth=0.6, zorder=2)
axa.set_xticks(list(x)); axa.set_xticklabels(["%d" % b for b in BWS])
axa.set_xlabel("inter-domain link bandwidth (GB/s), DPU = 20.5 GB/s")
axa.set_ylabel("gain vs C0 (%)")
axa.set_ylim(-40, 15)
axa.legend(ncol=3, loc="lower left")
axa.annotate("collapse\n$-$32.7%", xy=(3.05, -32.7), xytext=(2.45, -22),
             fontsize=7, color=fig_style.NEG_RED, ha="center",
             arrowprops=dict(arrowstyle="->", color=fig_style.NEG_RED, lw=0.8))
axa.set_title("(a) mean completion-time gain vs bandwidth", fontsize=9, loc="left")


# v2 changes: pcolormesh with white cell gaps (HierMoE-style);
# title capitalization "(c) Always-dedup"; colorbar label
# "Mean completion-time gain (%)".
import numpy as np
norm = TwoSlopeNorm(vmin=-33, vcenter=0, vmax=12)
def heat(ax, d, title):
    M = np.array([[cell(d, bw, dpu)[0] for dpu in DPUS] for bw in BWS])
    G = [[cell(d, bw, dpu)[1] for dpu in DPUS] for bw in BWS]
    im = ax.pcolormesh(M, cmap="RdBu", norm=norm,
                       edgecolors="white", linewidth=1.5)
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(DPU_LBL)
    ax.set_yticks([0.5, 1.5, 2.5, 3.5])
    ax.set_yticklabels(["%d" % b for b in BWS])
    ax.tick_params(length=0)
    ax.set_xlabel("DPU rate (GB/s)")
    ax.set_title(title, fontsize=9, loc="left")
    for i in range(4):
        for j in range(3):
            v, g = M[i][j], G[i][j]
            col = "white" if (v >= 9 or v <= -15) else "black"
            ax.text(j + 0.5, i + 0.40, "%+.1f" % v, ha="center",
                    va="center", fontsize=8, color=col)
            ax.text(j + 0.5, i + 0.72, "%d/6" % g, ha="center",
                    va="center", fontsize=6, color=col)
    return im

im = heat(axb, ca, "(b) CARTS (cost-aware)")
heat(axc, c4, "(c) Always-dedup (C4)")
axb.set_ylabel("bandwidth (GB/s)")
cb = fig.colorbar(im, ax=(axb, axc), fraction=0.035, pad=0.02)
cb.set_label("Mean completion-time gain (%)", fontsize=8)
fig.savefig(OUT)
print("wrote", OUT)
