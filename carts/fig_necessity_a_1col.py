#!/usr/bin/env python3
# fig_necessity_a_1col.py -- APPEND-ONLY. Single-column version of panel (a)
# from fig_necessity_composite_v2.py. Same data, same anchor gates; only the
# layout changes (legend above plot, compact figsize, short labels).
# Does not modify fig_necessity_composite_v2.py.
import csv, statistics as st
from collections import defaultdict
import fig_style
fig_style.apply()
import matplotlib.pyplot as plt

CSV, C4CSV = "sensitivity_gpd8_results.csv", "c4_always_dedup_results.csv"
OUT  = "fig_necessity_a_1col.pdf"
BWS  = [5, 10, 20, 50]
TOL  = 0.05
LOCK_CA_20K = {5: 10.3, 10: 6.1, 20: 3.4, 50: 3.4}
LOCK_C4_20K = {5: 6.3, 10: 4.7, 20: -2.6, 50: -32.7}
LOCK_NAIVE  = {5: 6.94, 10: 5.83, 20: -1.49, 50: -32.70}


def load(path, col):
    d = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(int(r["bw_gbs"]), int(r["dpu_bytes_per_us"]))].append(
                float(r[col]))
    return d


ca, old = load(CSV, "new_vs_c0"), load(CSV, "old_vs_c0")
c4 = load(C4CSV, "c4_vs_c0")


def cell(d, bw, dpu=20500):
    v = d[(bw, dpu)]
    assert len(v) == 6, "expected 6 seeds at (%d,%d)" % (bw, dpu)
    return st.mean(v)


for bw, e in LOCK_CA_20K.items():
    assert abs(cell(ca, bw) - e) <= TOL, "CA anchor fail bw=%d" % bw
for bw, e in LOCK_C4_20K.items():
    assert abs(cell(c4, bw) - e) <= TOL, "C4 anchor fail bw=%d" % bw
for bw, e in LOCK_NAIVE.items():
    assert abs(cell(old, bw) - e) <= TOL, "NAIVE anchor fail bw=%d" % bw
print("anchors verified -- plotting single-column panel (a)")


ca_m = [cell(ca, b) for b in BWS]
nv_m = [cell(old, b) for b in BWS]
c4_m = [cell(c4, b) for b in BWS]

fig, ax = plt.subplots(figsize=(3.4, 2.05), constrained_layout=True)
x = range(len(BWS))
w = 0.14
gap = 0.17

ax.bar([i - gap for i in x], ca_m, w, color=fig_style.CARTS_BLUE,
       label="CARTS", zorder=3)
ax.bar(list(x), nv_m, w, color=fig_style.NAIVE_AMBER,
       label="Naive", zorder=3)
ax.bar([i + gap for i in x], c4_m, w, color=fig_style.C4_GRAY,
       label="Always-dedup", zorder=3)

ax.axhline(0, color="black", linewidth=0.6, zorder=2)
ax.set_xticks(list(x))
ax.set_xticklabels(["%d" % b for b in BWS], fontsize=8)
ax.set_xlabel("inter-domain bandwidth (GB/s)", fontsize=8)
ax.set_ylabel("gain vs C0 (%)", fontsize=8)
ax.set_ylim(-38, 14)
ax.tick_params(axis="y", labelsize=8)

ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0),
          frameon=False, fontsize=7.5, handlelength=1.1,
          columnspacing=0.9, handletextpad=0.4, borderpad=0.2)

ax.annotate("$-$32.7%", xy=(3.05, -32.7), xytext=(2.30, -24),
            fontsize=7, color=fig_style.NEG_RED, ha="center",
            arrowprops=dict(arrowstyle="->", color=fig_style.NEG_RED, lw=0.7))

ax.set_xlim(-0.55, len(BWS) - 0.45)
fig.savefig(OUT)
print("wrote", OUT)
